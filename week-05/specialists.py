"""The four specialists, and the shared state of one job (TeamRun).

Both coordinators (pipeline.py and orchestrator.py) use the same TeamRun, so the
specialists behave the same way whichever coordinator calls them:

- analyst(): the supply analyst, a Claude agent with the supply MCP server's tools.
  Its brief is checked in code: every order it reports must match a check_coverage
  result, so no number in the brief can come from anywhere but the tools.
- supplier_reply() and customer_notice(order_id): a writer drafts, then the reviewer
  checks the draft. One rewrite is allowed after REVISE. Every draft is saved to a
  run folder in week-05/drafts/ for a person to check. Nothing is ever sent.

Rules enforced here in code, not left to the prompts:
- The supplier writer never sees customer names or order numbers: it gets totals only.
- The customer writer gets one order's facts, built from orders.csv and the checked
  brief, with no supplier and no penalty. The recipient address is added by code.
- No customer notices when the supplier offers options: which orders are late depends
  on a choice a person has to make first.
- Before review, code scans each draft for names and order numbers its reader must
  not see (customers in supplier replies; suppliers, other customers, and penalties
  in customer notices).
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from common import (DATA_DIR, DRAFTS_DIR, HERE, REPO, RULE, Tee, Usage, email_block,
                    load_api_key, parse_email, preview, read_prompt, reply_subject,
                    run_tool_loop, single_turn)
from supply_mcp_server import read_csv

SERVER_SCRIPT = HERE / "supply_mcp_server.py"
SERVER_LOG = HERE / "mcp_server.log"  # *.log is git-ignored
ANALYST_MAX_TOOL_CALLS = 10
ANALYST_ATTEMPTS = 2   # one retry if the brief fails the checks
WRITE_ROUNDS = 2       # first draft + one rewrite after REVISE
NO_REPLY = "NO REPLY NEEDED"
BRIEF_FIELDS = ["part_id", "what_changed", "orders_at_risk", "options",
                "decisions_for_a_person", "supplier_questions"]


class BriefError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Checking the analyst's brief
# ---------------------------------------------------------------------------

def parse_json_object(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise BriefError("The brief isn't JSON: no {...} object found.")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise BriefError(f"The brief isn't valid JSON: {exc}.") from None


def check_brief(text, coverage_results):
    """Parse the brief and check it against the check_coverage results. Returns the brief,
    with each order at risk replaced by the tool's own record of that order."""
    brief = parse_json_object(text)
    if not isinstance(brief, dict) or sorted(brief) != sorted(BRIEF_FIELDS):
        raise BriefError(f"The brief must have exactly these fields: {', '.join(BRIEF_FIELDS)}.")
    for key in ("part_id", "what_changed"):
        if not isinstance(brief[key], str) or not brief[key].strip():
            raise BriefError(f"{key} must be a non-empty string.")
    for key in BRIEF_FIELDS[2:]:
        if not isinstance(brief[key], list):
            raise BriefError(f"{key} must be a list (an empty list if none).")

    at_risk = {}  # order_id -> list of tool records where that order was at risk
    for result in coverage_results:
        for order in result["orders"]:
            if not order["covered"]:
                at_risk.setdefault(order["order_id"], []).append(order)

    checked = []
    for entry in brief["orders_at_risk"]:
        if not isinstance(entry, dict) or "order_id" not in entry:
            raise BriefError("Each orders_at_risk entry must be an object with an order_id.")
        oid = str(entry["order_id"]).strip()
        match = next((o for o in at_risk.get(oid, [])
                      if o["units_short"] == entry.get("units_short")
                      and o["ship_day"] == entry.get("ship_day")
                      and o["ready_day"] == entry.get("ready_day")), None)
        if match is None:
            raise BriefError(f"orders_at_risk entry {oid} doesn't match any check_coverage result "
                             "(order_id, units_short, ship_day, and ready_day must come from the tool).")
        checked.append({k: match[k] for k in ("order_id", "customer", "model", "bikes",
                                              "units_short", "ship_day", "ready_day",
                                              "days_late", "late_penalty")})
    brief["orders_at_risk"] = checked

    for option in brief["options"]:
        if not isinstance(option, dict) or not isinstance(option.get("order_ids"), list):
            raise BriefError("Each option must be an object with a name and an order_ids list.")
        for oid in option["order_ids"]:
            if str(oid).strip() not in at_risk:
                raise BriefError(f"Option {option.get('name')!r} lists {oid}, but no check_coverage "
                                 "result shows that order at risk.")

    if not brief["options"] and coverage_results:
        expected = set(coverage_results[-1]["orders_at_risk"])
        reported = {o["order_id"] for o in checked}
        if reported != expected:
            raise BriefError("orders_at_risk must list exactly the orders at risk in your last "
                             f"check_coverage result: {', '.join(sorted(expected)) or 'none'}.")
    return brief


# ---------------------------------------------------------------------------
# Draft checks done in code
# ---------------------------------------------------------------------------

def find_leaks(text, forbidden):
    lowered = text.lower()
    return sorted({term for term in forbidden if term and term.lower() in lowered})


def first_word(review):
    for line in review.splitlines():
        line = line.strip().strip("*").strip().upper()
        if line:
            return "APPROVE" if line.startswith("APPROVE") else (
                "REVISE" if line.startswith("REVISE") else "UNCLEAR")
    return "UNCLEAR"


# ---------------------------------------------------------------------------
# One job
# ---------------------------------------------------------------------------

class TeamRun:
    def __init__(self, client, email, coordinator):
        self.client = client
        self.email = email
        self.coordinator = coordinator
        self.usage = Usage()
        self.brief = None
        self.analyst_runs = 0
        self.supplier_done = False
        self.drafts = []      # (file name or None, status, recipient)
        self.summary = None
        self.needs_decision = None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = DRAFTS_DIR / f"{stamp}-{coordinator}-{email['path'].stem}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.orders = {o["order_id"]: o for o in read_csv("orders.csv")}
        self.suppliers = read_csv("suppliers.csv")

    # ----- supply analyst -------------------------------------------------

    async def analyst(self):
        """Run the supply analyst. Returns the checked brief (also saved as brief.json)."""
        if self.brief is not None:
            return self.brief
        self.analyst_runs += 1
        system = read_prompt("analyst")
        coverage_results = []
        server = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)],
                                       cwd=str(REPO))
        print(f"{RULE}\nSupply analyst (MCP server: {SERVER_SCRIPT.name})\n{RULE}")
        with open(SERVER_LOG, "a", encoding="utf-8") as errlog:
            async with stdio_client(server, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_tools = (await session.list_tools()).tools
                    tools = [{"name": t.name, "description": t.description or "",
                              "input_schema": t.input_schema} for t in mcp_tools]
                    names = {t["name"] for t in tools}

                    async def handle(name, args):
                        if name not in names:
                            return f"Unknown tool: {name}", True
                        text, is_error = await call_mcp_tool(session, name, args)
                        if name == "check_coverage" and not is_error:
                            coverage_results.append(json.loads(text))
                        return text, is_error

                    messages = [{"role": "user", "content": (
                        "Days are counted from today (day 0). Prepare your brief for this "
                        f"supplier email.\n\n{email_block(self.email)}")}]
                    for attempt in range(1, ANALYST_ATTEMPTS + 1):
                        text, stop = await run_tool_loop(
                            self.client, self.usage, "supply analyst", system, tools, messages,
                            handle, ANALYST_MAX_TOOL_CALLS)
                        print(f"[supply analyst] {stop}.")
                        try:
                            brief = check_brief(text, coverage_results)
                            break
                        except BriefError as exc:
                            print(f"[supply analyst] brief rejected by code check: {exc}\n")
                            if attempt == ANALYST_ATTEMPTS:
                                raise RuntimeError(f"The analyst's brief failed the checks: {exc}")
                            messages.append({"role": "user", "content": (
                                f"Your brief failed a check: {exc} Fix it, using tools if you "
                                "need to, and finish with only the corrected JSON brief.")})
        self.brief = brief
        (self.run_dir / "brief.json").write_text(json.dumps(brief, indent=2), encoding="utf-8")
        print(f"Brief:\n{json.dumps(brief, indent=2)}\n")
        return brief

    # ----- facts each writer may see --------------------------------------

    def supplier_facts(self):
        if self.brief is None:
            return {"note": "No supply analysis was run for this email."}
        b = self.brief
        late = [o["days_late"] for o in b["orders_at_risk"] if o["days_late"] is not None]
        return {
            "part_id": b["part_id"],
            "what_changed": b["what_changed"],
            "orders_delayed": len(b["orders_at_risk"]),
            "total_units_short": sum(o["units_short"] for o in b["orders_at_risk"]),
            "days_late_range": [min(late), max(late)] if late else None,
            "options": [{"name": o.get("name"), "orders_delayed": len(o["order_ids"])}
                        for o in b["options"]],
            "decisions_for_a_person": b["decisions_for_a_person"],
            "supplier_questions": b["supplier_questions"],
        }

    def customer_facts(self, entry):
        row = self.orders[entry["order_id"]]
        return {
            "customer": row["customer"],
            "order_id": entry["order_id"],
            "model": row["model"],
            "quantity": int(row["qty"]),
            "planned_ship_in_days": entry["ship_day"],
            "days_later_than_planned": (entry["days_late"] if entry["days_late"] is not None
                                        else "not yet known"),
        }

    # ----- writer + reviewer ----------------------------------------------

    def write_and_review(self, writer, writer_input, facts, audience, forbidden, label):
        """Draft, check in code, review; one rewrite after REVISE. Returns (text, status, notes).
        facts is the text the reviewer checks the draft against: what the writer was given."""
        writer_system, reviewer_system = read_prompt(writer), read_prompt("reviewer")
        messages = [{"role": "user", "content": (
            f"{writer_input}\n\nWrite only the body of the email, ready for a person to check "
            "and send, with no notes before or after it.")}]
        notes = []
        for round_no in range(1, WRITE_ROUNDS + 1):
            draft = single_turn(self.client, self.usage, writer, writer_system, messages)
            print(f"[{writer}] draft {round_no} for {label}:\n{preview(draft, 600)}\n")
            if draft.strip().upper() == NO_REPLY:
                return NO_REPLY, "no reply needed", notes
            leaks = find_leaks(draft, forbidden)
            if leaks:
                verdict = "REVISE"
                feedback = ("Code check: the draft must not mention " + ", ".join(leaks) +
                            ". Remove every mention.")
            else:
                review = single_turn(self.client, self.usage, "reviewer", reviewer_system, [{
                    "role": "user", "content": (
                        f"Recipient: {audience}\n\n<facts>\n{facts}\n</facts>"
                        f"\n\n<draft>\n{draft}\n</draft>")}])
                verdict, feedback = first_word(review), review
            print(f"[reviewer] round {round_no}: {verdict}\n{preview(feedback, 600)}\n")
            notes.append(f"Round {round_no}: {verdict}\n{feedback}")
            if verdict == "APPROVE":
                return draft, "approved" if round_no == 1 else "approved after one revision", notes
            if round_no < WRITE_ROUNDS:
                messages.append({"role": "user", "content": (
                    f"A reviewer asked for changes:\n\n{feedback}\n\nWrite the full revised "
                    "draft. Output only the email itself, with no notes to the reviewer.")})
        return draft, "NOT APPROVED - a person must fix this draft before using it", notes

    def save_draft(self, filename, header, text, status, notes, recipient):
        body = "\n".join(header + [f"Review: {status}", "Status: draft only - not sent",
                                   "", "DRAFT", text, "", "REVIEW NOTES", *notes]) + "\n"
        (self.run_dir / filename).write_text(body, encoding="utf-8")
        self.drafts.append((filename, status, recipient))
        return self.run_dir / filename

    # ----- supplier reply ---------------------------------------------------

    def supplier_reply(self):
        if self.supplier_done:
            raise ValueError("A supplier reply was already drafted for this email.")
        self.supplier_done = True
        facts = self.supplier_facts()
        forbidden = set()
        for row in self.orders.values():
            forbidden |= {row["customer"], row["order_id"], row["contact_email"]}
        print(f"{RULE}\nSupplier reply\n{RULE}")
        # The reviewer sees exactly what the writer saw: the supplier's email and the brief.
        writer_input = f"{email_block(self.email)}\n\n<brief>\n{json.dumps(facts, indent=2)}\n</brief>"
        text, status, notes = self.write_and_review(
            "supplier-writer", writer_input, writer_input, "supplier", forbidden, "the supplier")
        if text == NO_REPLY:
            self.drafts.append((None, "no reply needed", "supplier"))
            return {"status": "no reply needed"}
        path = self.save_draft("supplier-reply.txt", [
            f"To: {self.email['from_name']} <{self.email['sender']}>",
            f"Subject: {reply_subject(self.email['subject'])}",
            "From: Purchasing, Brookfield Bicycle Company",
            f"In reply to: {self.email['path'].name}"], text, status, notes, "supplier")
        return {"status": status, "saved_to": str(path.relative_to(REPO))}

    # ----- customer notice --------------------------------------------------

    def customer_notice(self, order_id):
        order_id = str(order_id).strip().upper()
        if self.brief is None:
            raise ValueError("Run the supply analyst first: notices are written from its brief.")
        if self.brief["options"]:
            raise ValueError("The supplier offered options, so which orders are late depends on a "
                             "choice a person must make first. No customer notices for this email.")
        entry = next((o for o in self.brief["orders_at_risk"] if o["order_id"] == order_id), None)
        if entry is None:
            listed = ", ".join(o["order_id"] for o in self.brief["orders_at_risk"]) or "none"
            raise ValueError(f"{order_id} is not in the brief's orders_at_risk ({listed}).")
        if any(f == f"notice-{order_id}.txt" for f, _, _ in self.drafts):
            raise ValueError(f"A notice for {order_id} was already drafted.")
        facts = self.customer_facts(entry)
        row = self.orders[order_id]
        forbidden = {"penalt"}
        for s in self.suppliers:
            forbidden |= {s["supplier"], s["contact_name"], s["contact_email"]}
        for other in self.orders.values():
            if other["order_id"] != order_id and other["customer"] != row["customer"]:
                forbidden |= {other["customer"], other["order_id"]}
        print(f"{RULE}\nCustomer notice for {order_id}\n{RULE}")
        facts_json = json.dumps(facts, indent=2)
        text, status, notes = self.write_and_review(
            "customer-writer", f"<order_facts>\n{facts_json}\n</order_facts>",
            facts_json, "customer", forbidden, order_id)
        path = self.save_draft(f"notice-{order_id}.txt", [
            f"To: {row['customer']} <{row['contact_email']}>",
            f"Subject: Update on your Brookfield order {order_id}",
            "From: Customer Service, Brookfield Bicycle Company"], text, status, notes, "customer")
        return {"order_id": order_id, "status": status, "saved_to": str(path.relative_to(REPO))}

    # ----- end of the job ---------------------------------------------------

    def finish(self, summary, needs_decision):
        self.summary, self.needs_decision = summary, needs_decision
        lines = [f"Coordinator: {self.coordinator}", f"Email: {self.email['path'].name}",
                 f"Subject: {self.email['subject']}",
                 f"A person needs to make a decision: {'yes' if needs_decision else 'no'}", "",
                 "SUMMARY FOR THE PURCHASING MANAGER", summary, "", "DRAFTS"]
        lines += [f"- {f or '(none)'} [{r}]: {s}" for f, s, r in self.drafts] or ["- none"]
        lines += ["", "USAGE", self.usage.report()]
        (self.run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def call_mcp_tool(session, name, args):
    """Run one MCP tool. Returns (text, is_error). (From week 4.)"""
    result = await session.call_tool(name, args)
    parts = []
    for block in getattr(result, "content", None) or []:
        parts.append(block.text if getattr(block, "type", "") == "text" else str(block))
    text = "\n".join(parts)
    if not text and getattr(result, "structured_content", None) is not None:
        text = json.dumps(result.structured_content)
    return text or "(empty result)", bool(getattr(result, "is_error", False))


# ---------------------------------------------------------------------------
# Command line shared by both coordinators
# ---------------------------------------------------------------------------

def run_coordinator(name, description, coordinate):
    """Parse --email/--save, run coordinate(team) for one email, print the results."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--email", metavar="PATH", required=True,
                        help="one supplier email (.eml), e.g. week-05/brookfield-bikes/"
                             "supplier-emails/01-northgate-controller-delay.eml")
    parser.add_argument("--save", metavar="FILE",
                        help="also write everything printed to FILE (UTF-8), e.g. week-05/runs/...")
    args = parser.parse_args()
    if not Path(args.email).is_file():
        sys.exit(f"Email file not found: {args.email}")

    sys.stdout.reconfigure(encoding="utf-8")
    log = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        log = open(args.save, "w", encoding="utf-8")
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)

    code = 0
    try:
        email = parse_email(args.email)
        team = TeamRun(anthropic.Anthropic(api_key=load_api_key()), email, name)
        print(f"Coordinator: {name}\nEmail: {email['path'].name}\nFrom: {email['sender']}\n"
              f"Subject: {email['subject']}\nRun folder: {team.run_dir.relative_to(REPO)}\n")
        asyncio.run(coordinate(team))
        print(f"{RULE}\nRESULT ({name})\n{RULE}")
        print(f"Needs a decision by a person: {'yes' if team.needs_decision else 'no'}")
        print(f"Summary: {team.summary}\n")
        print("Drafts (nothing was sent):")
        for f, s, r in team.drafts:
            print(f"  {f or '(none)'} [{r}]: {s}")
        if not team.drafts:
            print("  none")
        print(f"\nSaved in {team.run_dir.relative_to(REPO)}\n\n{team.usage.report()}")
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.", file=sys.stderr)
        code = 130
    except anthropic.AuthenticationError:
        print("The API key was rejected (authentication error).", file=sys.stderr)
        code = 1
    except anthropic.RateLimitError:
        print("Rate limited by the API; try again shortly.", file=sys.stderr)
        code = 1
    except anthropic.APIStatusError as exc:
        print(f"API error {exc.status_code}: {exc.message}", file=sys.stderr)
        code = 1
    except anthropic.APIConnectionError:
        print("Couldn't reach the Anthropic API (connection error).", file=sys.stderr)
        code = 1
    except RuntimeError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        code = 1
    finally:
        if log:
            sys.stdout.flush()
            sys.stdout, sys.stderr = sys.stdout.stream, sys.stderr.stream
            log.close()
            print(f"Output saved to {args.save}")
    sys.exit(code)
