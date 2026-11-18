"""The four specialists both coordinators share, and the review step around the writers.

Specialists (all on claude-sonnet-5; each reads its system prompt from
<company folder>/prompts/ every time it runs, so prompt edits take effect right away).
The company folder is set with --org (default: week-05/brookfield-bikes); its CSV files are
read through supply_mcp_server.load_table, which works out the columns from each header:

- run_analyst(email): the supply analyst, an agent with the tools of
  supply_mcp_server.py. It may make at most 10 tool calls in total; after the 10th it
  must finish without tools. Returns its brief (a JSON object), which code checks
  against the check_coverage results, so every number in it came from the tools.
- write_supplier_reply(email, brief): one model call. All customer names (and their
  order numbers and contact addresses) are removed from what it sees first.
- write_customer_notice(order): one model call, given only that one order's facts.
- review(draft, facts, recipient): one model call. Returns APPROVE or REVISE with reasons.

reviewed_draft() is the review step: a writer drafts, the reviewer checks, and the writer
revises at most 2 times. Then code checks the final draft: no customer name in a
supplier draft, and no supplier name in a customer draft. A draft is marked for a person
if the reviewer didn't approve it, the code check failed, or its order has a late
penalty. A customer notice is also marked when the supplier offered options, since
whether that order is late depends on the option a person chooses. Drafts are saved in week-05/drafts/<run>/. Nothing is ever sent.

Every handoff between the coordinator and a specialist, or between two specialists, is
printed, along with the tokens each specialist call used. The API key is read at run
time (common.load_api_key) and never written anywhere.

TeamRun holds one job's state for the coordinators (pipeline.py and orchestrator.py).
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from common import (DRAFTS_DIR, HERE, REPO, RULE, STOP_REASONS, Tee, Usage, ask_claude,
                    company_name, org_dir, org_name, resolve_org, set_org,
                    email_block, load_api_key, parse_email, read_prompt, reply_subject,
                    response_text, single_turn, preview)
from supply_mcp_server import coverage, load_table

SPECIALIST_MODEL = "claude-sonnet-5"
SERVER_SCRIPT = HERE / "supply_mcp_server.py"
SERVER_LOG = HERE / "mcp_server.log"  # *.log is git-ignored
ANALYST_MAX_TOOL_CALLS = 10   # in total, for the whole analyst run
ANALYST_ATTEMPTS = 2          # one retry (without more tool calls than the limit) if the brief fails the checks
MAX_REVISIONS = 2             # a draft is revised at most twice after REVISE
NO_REPLY = "NO REPLY NEEDED"
BRIEF_FIELDS = ["part_id", "what_changed", "orders_at_risk", "options",
                "decisions_for_a_person", "supplier_questions"]
HANDOFF_CHARS = 2500          # how much of each handoff's contents to print
SLACK_LOOKAHEAD = 14          # how many more days of delay the slack check tries
SLACK_TIGHT_DAYS = 3          # at or below this, the supplier is asked to hold the new date

ANALYST, SUPPLIER_WRITER, CUSTOMER_WRITER, REVIEWER = (
    "supply analyst", "supplier writer", "customer writer", "reviewer")


class BriefError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Shared client, token counts, and handoff printing
# ---------------------------------------------------------------------------

_default = {"client": None, "usage": Usage()}


def _ctx(client, usage):
    """The client and Usage to use: the caller's, or this module's own."""
    if client is None:
        if _default["client"] is None:
            _default["client"] = anthropic.Anthropic(api_key=load_api_key())
        client = _default["client"]
    return client, (usage if usage is not None else _default["usage"])


def handoff(sender, receiver, what, payload=None):
    print(f"\n>>> HANDOFF: {sender} -> {receiver}: {what}")
    if payload is not None:
        text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
        print("    " + preview(text, HANDOFF_CHARS).replace("\n", "\n    "))


def _print_tokens(agent, response):
    u = response.usage
    print(f"    [{agent}] tokens this call: {u.input_tokens:,} in, {u.output_tokens:,} out")


def _one_call(client, usage, agent, prompt_name, user_text):
    """One model call for a writer or the reviewer. Returns the text."""
    messages = [{"role": "user", "content": user_text}]
    response = ask_claude(client, usage, agent, read_prompt(prompt_name), messages,
                          model=SPECIALIST_MODEL)
    _print_tokens(agent, response)
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"{agent} {STOP_REASONS.get(response.stop_reason, response.stop_reason)}.")
    text = response_text(response)
    if agent in (SUPPLIER_WRITER, CUSTOMER_WRITER):
        text = strip_subject(text)
    return text


def strip_subject(draft):
    """Drop a "Subject: ..." line a writer put at the top of the body (the saved draft
    already has its own Subject header)."""
    lines = draft.strip().splitlines()
    if lines and re.match(r"\**\s*subject\s*:", lines[0], re.IGNORECASE):
        lines = lines[1:]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Specialist 1: the supply analyst
# ---------------------------------------------------------------------------

def parse_json_object(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise BriefError("The brief isn't JSON: no {...} object found.")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise BriefError(f"The brief isn't valid JSON: {exc}.") from None


def flatten_orders(value):
    """orders_at_risk as a list, even when the analyst grouped it (by option, or keyed by
    order ID). Duplicates of the same order are dropped. Other values pass through."""
    if not isinstance(value, dict):
        return value
    found = []

    def collect(v):
        if isinstance(v, dict) and "order_id" in v:
            found.append(v)
        elif isinstance(v, dict):
            for inner in v.values():
                collect(inner)
        elif isinstance(v, list):
            for inner in v:
                collect(inner)

    collect(value)
    seen, orders = set(), []
    for o in found:
        if o["order_id"] not in seen:
            seen.add(o["order_id"])
            orders.append(o)
    return orders


def normalize_option(option):
    """Accept common variations in how the analyst writes an option: the order list under
    another key (orders_at_risk, order_ids_at_risk, ...), orders as objects instead of IDs,
    or the name under "option" or "label". Changes option in place."""
    if not isinstance(option, dict):
        return
    if "name" not in option:
        for key in ("option", "label", "option_name", "id"):
            if isinstance(option.get(key), str):
                option["name"] = option.pop(key)
                break
    if not isinstance(option.get("order_ids"), list):
        key = next((k for k, v in option.items() if "order" in k.lower() and isinstance(v, list)),
                   None)
        if key is not None:
            option["order_ids"] = option.pop(key)
    if isinstance(option.get("order_ids"), list):
        option["order_ids"] = [o.get("order_id") if isinstance(o, dict) else o
                               for o in option["order_ids"]]


def check_brief(text, coverage_results):
    """Parse the brief and check it against the check_coverage results. Returns the brief,
    with each order at risk replaced by the tool's own record of that order."""
    brief = parse_json_object(text)
    if isinstance(brief, dict):
        brief["orders_at_risk"] = flatten_orders(brief.get("orders_at_risk"))
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
        checked.append({k: match[k] for k in ("order_id", "customer", "product", "quantity",
                                              "units_short", "ship_day", "ready_day",
                                              "days_late", "late_penalty")})
    brief["orders_at_risk"] = checked

    for option in brief["options"]:
        normalize_option(option)
        if not isinstance(option, dict) or not isinstance(option.get("order_ids"), list):
            raise BriefError('Each option must look like {"name": "Option A", "order_ids": '
                             '["O-1234"]}, with an empty order_ids list if it puts no order at risk.')
        for oid in option["order_ids"]:
            if str(oid).strip() not in at_risk:
                raise BriefError(f"Option {option.get('name')!r} lists {oid}, but no check_coverage "
                                 "result shows that order at risk.")
        # Each option's impact comes from the check_coverage result for that option: the
        # latest one whose orders at risk are exactly the option's order_ids.
        ids = {str(i).strip() for i in option["order_ids"]}
        result = next((r for r in reversed(coverage_results)
                       if set(r["orders_at_risk"]) == ids), None)
        if result is None:
            listed = ", ".join(sorted(ids)) or "none"
            raise BriefError(f"Option {option.get('name')!r}: its order_ids ({listed}) don't match "
                             "the orders at risk in any check_coverage result. Check each option "
                             "separately and copy its orders at risk.")
        option["impact"] = [{k: o[k] for k in ("order_id", "units_short", "days_late")}
                            for o in result["orders"] if not o["covered"]]
        # Every order at risk under some option belongs in orders_at_risk too (customer
        # notices and the late-penalty check use that list). Add any the analyst left out,
        # straight from the tool's record for this option.
        listed = {o["order_id"] for o in checked}
        for o in result["orders"]:
            if not o["covered"] and o["order_id"] not in listed:
                checked.append({k: o[k] for k in ("order_id", "customer", "product", "quantity",
                                                  "units_short", "ship_day", "ready_day",
                                                  "days_late", "late_penalty")})
                print(f"[code check] added {o['order_id']} to orders_at_risk (at risk under "
                      f"{option.get('name')})")

    if not brief["options"] and coverage_results:
        expected = set(coverage_results[-1]["orders_at_risk"])
        reported = {o["order_id"] for o in checked}
        if reported != expected:
            raise BriefError("orders_at_risk must list exactly the orders at risk in your last "
                             f"check_coverage result: {', '.join(sorted(expected)) or 'none'}.")

    # The free-text fields may mention only orders (and their customers) that are at risk,
    # in orders_at_risk or under one of the options.
    at_risk_ids = {o["order_id"] for o in checked}
    for option in brief["options"]:
        at_risk_ids |= {str(i).strip() for i in option["order_ids"]}
    stray = set()
    for row in load_table("orders.csv"):
        if row["order_id"] in at_risk_ids:
            continue
        for key in ("what_changed", "decisions_for_a_person", "supplier_questions"):
            texts = brief[key] if isinstance(brief[key], list) else [brief[key]]
            for text in texts:
                found = find_names(str(text), [row["customer"]])
                if found or re.search(rf"\b{re.escape(row['order_id'])}\b", str(text)):
                    stray.add(f"{row['customer']} ({row['order_id']}) in {key}")
    if stray:
        raise BriefError("These customers' orders are not at risk, so don't mention them: "
                         + "; ".join(sorted(stray)) + ".")
    return brief


def find_slack(part_id, arrivals):
    """How many more days the new schedule can slip before an order is late. Uses the
    same coverage() code as the check_coverage tool, delaying every arrival by 1, 2, ...
    up to SLACK_LOOKAHEAD days."""
    for extra in range(1, SLACK_LOOKAHEAD + 1):
        shifted = [{"qty": a["qty"], "day": a["day"] + extra} for a in arrivals]
        late = [o for o in coverage(part_id, shifted)["orders"] if not o["covered"]]
        if late:
            o = late[0]
            days = extra - 1
            penalty = ", late penalty" if o["late_penalty"] else ""
            return {
                "part_id": part_id,
                "extra_days": days,
                "first_order_affected": {k: o[k] for k in ("order_id", "customer",
                                                          "late_penalty", "units_short")},
                "note": (f"The new {part_id} schedule can slip {days} more "
                         f"day{'' if days == 1 else 's'} with no order late. At {extra} more "
                         f"day{'' if extra == 1 else 's'}, {o['order_id']} ({o['customer']}"
                         f"{penalty}) would be {o['units_short']} units short."),
            }
    return {"part_id": part_id, "extra_days": None, "first_order_affected": None,
            "note": (f"No order would be late even if the new {part_id} schedule slipped "
                     f"{SLACK_LOOKAHEAD} more days.")}


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


async def run_analyst(email, *, client=None, usage=None):
    """The supply analyst agent. Returns its checked brief as a JSON object (a dict).

    It stops after ANALYST_MAX_TOOL_CALLS tool calls: from then on it is called with
    tool_choice "none", so it has to finish its brief with what it has."""
    client, usage = _ctx(client, usage)
    system = read_prompt("analyst")
    coverage_results = []
    calls = 0
    server = StdioServerParameters(command=sys.executable,
                                   args=[str(SERVER_SCRIPT), "--org", str(org_dir())],
                                   cwd=str(REPO))
    print(f"{RULE}\nSupply analyst ({SPECIALIST_MODEL}; MCP server: {SERVER_SCRIPT.name})\n{RULE}")
    with open(SERVER_LOG, "a", encoding="utf-8") as errlog:
        async with stdio_client(server, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = [{"name": t.name, "description": t.description or "",
                          "input_schema": t.input_schema}
                         for t in (await session.list_tools()).tools]
                names = {t["name"] for t in tools}
                messages = [{"role": "user", "content": (
                    "Days are counted from today (day 0). Prepare your brief for this "
                    f"supplier email.\n\n{email_block(email)}")}]

                for attempt in range(1, ANALYST_ATTEMPTS + 1):
                    # The agent loop: runs until the analyst answers without a tool call.
                    while True:
                        out_of_calls = calls >= ANALYST_MAX_TOOL_CALLS
                        response = ask_claude(
                            client, usage, ANALYST, system, messages, tools,
                            model=SPECIALIST_MODEL,
                            tool_choice={"type": "none"} if out_of_calls else None)
                        _print_tokens(ANALYST, response)
                        messages.append({"role": "assistant", "content": response.content})
                        if response.stop_reason != "tool_use":
                            if response.stop_reason != "end_turn":
                                why = STOP_REASONS.get(response.stop_reason, response.stop_reason)
                                raise RuntimeError(f"The supply analyst {why}.")
                            break
                        results = []
                        for block in response.content:
                            if block.type != "tool_use":
                                continue
                            if calls >= ANALYST_MAX_TOOL_CALLS:
                                text, is_error = "Tool-call limit reached; not run.", True
                            elif block.name not in names:
                                calls += 1
                                text, is_error = f"Unknown tool: {block.name}", True
                            else:
                                calls += 1
                                print(f"[{ANALYST}] tool call {calls}/{ANALYST_MAX_TOOL_CALLS}: "
                                      f"{block.name} {json.dumps(block.input, ensure_ascii=False)}")
                                try:
                                    text, is_error = await call_mcp_tool(session, block.name, block.input)
                                except Exception as exc:
                                    text, is_error = f"Error: {exc}", True
                                if block.name == "check_coverage" and not is_error:
                                    coverage_results.append(json.loads(text))
                                print(f"  result{' (error)' if is_error else ''}: {preview(text)}\n")
                            results.append({"type": "tool_result", "tool_use_id": block.id,
                                            "content": text, "is_error": is_error})
                        if calls >= ANALYST_MAX_TOOL_CALLS:
                            print(f"[{ANALYST}] reached {ANALYST_MAX_TOOL_CALLS} tool calls; "
                                  "it must finish now.")
                            results.append({"type": "text", "text": (
                                f"You have used all {ANALYST_MAX_TOOL_CALLS} tool calls. Finish "
                                "now with your JSON brief, using only the results you have.")})
                        messages.append({"role": "user", "content": results})

                    try:
                        brief = check_brief(response_text(response), coverage_results)
                        break
                    except BriefError as exc:
                        print(f"[{ANALYST}] brief rejected by code check: {exc}")
                        if attempt == ANALYST_ATTEMPTS:
                            raise RuntimeError(f"The analyst's brief failed the checks: {exc}")
                        messages.append({"role": "user", "content": (
                            f"Your brief failed a check: {exc} Fix it (you have "
                            f"{max(0, ANALYST_MAX_TOOL_CALLS - calls)} tool calls left) and "
                            "finish with only the corrected JSON brief.")})
    print(f"[{ANALYST}] finished after {calls} tool call(s).")
    # No order at risk under a new schedule: check, in code, how much more delay it can take.
    # Skipped when the email didn't change the schedule (the analyst checked the same one).
    if not brief["orders_at_risk"] and not brief["options"] and coverage_results:
        last = coverage_results[-1]
        key = lambda arrivals: sorted((a["qty"], a["day"]) for a in arrivals)
        if key(last["arrivals_used"]) != key(last["replaced_schedule"] or []):
            brief["slack"] = find_slack(brief["part_id"], last["arrivals_used"])
            print(f"[slack check] {brief['slack']['note']}")
        else:
            print("[slack check] skipped: the email didn't change the schedule.")
    return brief


# ---------------------------------------------------------------------------
# Specialist 2: the supplier writer
# ---------------------------------------------------------------------------

def _customer_terms():
    """Customer names, order numbers, and contact addresses, longest first."""
    terms = {}
    for row in load_table("orders.csv"):
        terms[row["customer"]] = "a customer"
        terms[row["order_id"]] = "an order"
        terms[row["contact_email"]] = "a customer contact"
    return sorted(terms.items(), key=lambda kv: -len(kv[0]))


def remove_customers(value):
    """Replace every customer name, order number, and contact address in value
    (a string, or any JSON-like structure) with a neutral word."""
    if isinstance(value, str):
        for term, neutral in _customer_terms():
            value = re.sub(re.escape(term), neutral, value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [remove_customers(v) for v in value]
    if isinstance(value, dict):
        return {k: remove_customers(v) for k, v in value.items()}
    return value


def day_dates(texts):
    """{"day N": calendar date} for every "day N" in texts, counting today as day 0, so the
    supplier writer can give dates instead of our internal day numbers."""
    today = date.today()
    days = sorted({int(n) for t in texts for n in re.findall(r"\bday (\d+)\b", str(t))})
    table = {"today (day 0)": f"{today:%A, %B} {today.day}, {today.year}"}
    for n in days:
        d = today + timedelta(days=n)
        table[f"day {n}"] = f"{d:%A, %B} {d.day}, {d.year}"
    return table


def impact_totals(orders):
    """Totals for a group of orders at risk: no customers, no order numbers."""
    late = [o["days_late"] for o in orders if o["days_late"] is not None]
    return {"orders_delayed": len(orders),
            "units_short": sum(o["units_short"] for o in orders),
            "days_late_range": [min(late), max(late)] if late else None}


def supplier_brief(brief):
    """What the supplier writer may see: totals, not orders, with customers removed, and
    a table turning each "day N" into a date. When the supplier offers options, only one
    of them will happen, so the impact is given per option and never added up."""
    if brief is None:
        return {"note": "No supply analysis was run for this email."}
    texts = [brief["what_changed"], *brief["decisions_for_a_person"],
             *brief["supplier_questions"], *[o.get("name") for o in brief["options"]]]
    facts = {"part_id": brief["part_id"], "what_changed": brief["what_changed"]}
    if brief["options"]:
        facts["impact_by_option"] = [{"option": o.get("name"), **impact_totals(o["impact"])}
                                     for o in brief["options"]]
        facts["impact_note"] = ("Only one option will happen. Give the impact of each option "
                                "separately; never add the options together.")
    else:
        facts["impact"] = impact_totals(brief["orders_at_risk"])
    slack = brief.get("slack")
    if slack and slack["extra_days"] is not None and slack["extra_days"] <= SLACK_TIGHT_DAYS:
        # Only the fact that the date is tight: never our margin, never the order.
        facts["new_date_is_tight"] = True
    facts.update({
        "decisions_for_a_person": brief["decisions_for_a_person"],
        "supplier_questions": brief["supplier_questions"],
        "dates": day_dates(texts),
    })
    return remove_customers(facts)


def supplier_writer_input(email, brief):
    """The supplier writer's input, with all customer names removed. The reviewer checks
    the draft against exactly this."""
    return (f"{remove_customers(email_block(email))}\n\n<brief>\n"
            f"{json.dumps(supplier_brief(brief), indent=2)}\n</brief>")


def write_supplier_reply(email, brief, *, previous=None, feedback=None, client=None, usage=None):
    """One model call: a reply to the supplier. With previous and feedback, a revision."""
    client, usage = _ctx(client, usage)
    text = supplier_writer_input(email, brief)
    text += revision_request(previous, feedback)
    return _one_call(client, usage, SUPPLIER_WRITER, "supplier-writer", text)


# ---------------------------------------------------------------------------
# Specialist 3: the customer writer
# ---------------------------------------------------------------------------

def customer_facts(order):
    """Only this order's facts, from the analyst's checked brief. No supplier, no penalty."""
    return {
        "customer": order["customer"],
        "order_id": order["order_id"],
        "product": order["product"],
        "quantity": order["quantity"],
        "planned_ship_in_days": order["ship_day"],
        "days_later_than_planned": (order["days_late"] if order["days_late"] is not None
                                    else "not yet known"),
    }


def write_customer_notice(order, *, previous=None, feedback=None, client=None, usage=None):
    """One model call: a notice to the customer of this one order."""
    client, usage = _ctx(client, usage)
    text = f"<order_facts>\n{json.dumps(customer_facts(order), indent=2)}\n</order_facts>"
    text += revision_request(previous, feedback)
    return _one_call(client, usage, CUSTOMER_WRITER, "customer-writer", text)


def revision_request(previous, feedback):
    if previous is None:
        return ("\n\nWrite only the body of the email, ready for a person to check and send, "
                "with no notes before or after it.")
    return (f"\n\n<previous_draft>\n{previous}\n</previous_draft>\n\n<reviewer_feedback>\n"
            f"{feedback}\n</reviewer_feedback>\n\nWrite the full revised email, fixing every "
            "point the reviewer raised. Output only the email itself, with no notes.")


# ---------------------------------------------------------------------------
# Specialist 4: the reviewer
# ---------------------------------------------------------------------------

def review(draft, facts, recipient, *, client=None, usage=None):
    """One model call. Returns {"verdict": "APPROVE" or "REVISE", "reasons": text}."""
    client, usage = _ctx(client, usage)
    text = _one_call(client, usage, REVIEWER, "reviewer", (
        f"Recipient: {recipient}\n\n<facts>\n{facts}\n</facts>\n\n<draft>\n{draft}\n</draft>"))
    return {"verdict": parse_verdict(text), "reasons": text}


def parse_verdict(text):
    """APPROVE only if the first line is exactly APPROVE and no line says REVISE. Anything
    else (including "APPROVE is not warranted. REVISE ...") counts as REVISE."""
    lines = [line.strip().strip("*#:.!").strip().upper()
             for line in text.splitlines() if line.strip()]
    if any(line.startswith("REVISE") for line in lines):
        return "REVISE"
    return "APPROVE" if lines and lines[0] == "APPROVE" else "REVISE"


# ---------------------------------------------------------------------------
# The review step and the code checks
# ---------------------------------------------------------------------------

def find_names(text, names):
    lowered = text.lower()
    return sorted({n for n in names if n and n.lower() in lowered})


def customer_names():
    return {row["customer"] for row in load_table("orders.csv")}


def supplier_names():
    """Supplier names, their short names (Northgate, Keystone, ...), and contact names."""
    names = set()
    for s in load_table("suppliers.csv"):
        names |= {s["supplier"], s["supplier"].split()[0], s["contact_name"]}
    return names


def reviewed_draft(writer, write, facts, recipient, label, late_penalty=False, *,
                   coordinator="coordinator", client=None, usage=None, extra_reasons=()):
    """Draft with write(previous, feedback), review, revise at most MAX_REVISIONS times,
    then check the final draft in code. Returns a dict describing the outcome."""
    handoff(coordinator, writer, f"write the {recipient} draft for {label}", facts)
    draft = write(None, None)
    if draft.strip().upper() == NO_REPLY:
        handoff(writer, coordinator, "no reply needed")
        return {"draft": NO_REPLY, "verdict": "no reply needed", "revisions": 0, "notes": [],
                "needs_person": False, "reasons": []}

    notes, revisions = [], 0
    while True:
        handoff(writer, REVIEWER, f"draft {revisions + 1} for {label}", draft)
        result = review(draft, facts, recipient, client=client, usage=usage)
        notes.append(f"Draft {revisions + 1}: {result['verdict']}\n{result['reasons']}")
        if result["verdict"] == "APPROVE" or revisions == MAX_REVISIONS:
            break
        handoff(REVIEWER, writer, f"REVISE draft {revisions + 1}", result["reasons"])
        draft = write(draft, result["reasons"])
        revisions += 1
    handoff(REVIEWER, coordinator, f"{result['verdict']} after {revisions} revision(s)",
            result["reasons"])

    # Code checks on the final draft. These mark the draft; they don't rewrite it.
    if recipient == "supplier":
        leaks = find_names(draft, customer_names())
        leak_reason = "names a customer: "
    else:
        leaks = find_names(draft, supplier_names())
        leak_reason = "names a supplier: "
    reasons = []
    if result["verdict"] != "APPROVE":
        reasons.append(f"the reviewer did not approve it after {MAX_REVISIONS} revisions")
    if leaks:
        reasons.append(leak_reason + ", ".join(leaks))
    if late_penalty:
        reasons.append("an order with a late penalty is affected")
    reasons.extend(extra_reasons)
    print(f"[code check] {label}: " + ("names check failed - " + ", ".join(leaks)
                                       if leaks else "names check passed"))
    return {"draft": draft, "verdict": result["verdict"], "revisions": revisions,
            "notes": notes, "needs_person": bool(reasons), "reasons": reasons, "leaks": leaks}


def draft_summary(recipient, facts, outcome, to):
    """A short summary of a draft, built in code from what its writer was given and from
    the checks, so whoever reads it (the orchestrator, a person) can't be misled."""
    review_part = (f"reviewer: {outcome['verdict']} after {outcome['revisions']} revision(s)")
    if recipient == "supplier":
        def describe(t):
            days = t["days_late_range"]
            span = "" if not days else (f", {days[0]} day{'' if days[0] == 1 else 's'} late"
                                        if days[0] == days[1] else f", {days[0]}-{days[1]} days late")
            return f"{t['orders_delayed']} order(s) delayed, {t['units_short']} units short{span}"
        if "impact_by_option" in facts:
            impact = "per option: " + "; ".join(f"{t['option']}: {describe(t)}"
                                               for t in facts["impact_by_option"])
        elif "impact" in facts:
            impact = describe(facts["impact"])
        else:
            impact = "no supply analysis"
        check = ("names a customer: " + ", ".join(outcome["leaks"]) if outcome["leaks"]
                 else "names no customer")
        return (f"Reply to {to}. The writer was given totals only ({impact}), with no customer "
                f"names or order numbers. Code check: {check}; {review_part}.")
    days = facts["days_later_than_planned"]
    days = f"{days} day{'' if days == 1 else 's'}" if isinstance(days, int) else f"{days} days"
    check = ("names a supplier: " + ", ".join(outcome["leaks"]) if outcome["leaks"]
             else "names no supplier")
    return (f"Notice to {facts['customer']} about order {facts['order_id']} "
            f"({facts['quantity']} {facts['product']}), {days} later than planned. The "
            f"writer was given only this order's facts. Code check: {check}; {review_part}.")


def print_final(title, outcome):
    """After a review step: the final draft, once, with whether it's marked for a person."""
    print(f"\n{title}\n{outcome['draft']}\n"
          f"Marked for a person: {'yes' if outcome['needs_person'] else 'no'}\n")


# ---------------------------------------------------------------------------
# One job, for the coordinators
# ---------------------------------------------------------------------------

class TeamRun:
    def __init__(self, client, email, coordinator):
        self.client = client
        self.email = email
        self.coordinator = coordinator
        self.usage = Usage()
        self.brief = None
        self.supplier_done = False
        self.drafts = []      # dicts: file, recipient, status, needs_person, reasons
        self.summary = None
        self.needs_decision = None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Each company's drafts go in their own folder: week-05/drafts/<company folder>/<run>.
        self.run_dir = DRAFTS_DIR / org_name() / f"{stamp}-{coordinator}-{email['path'].stem}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.orders = {o["order_id"]: o for o in load_table("orders.csv")}
        self.company = company_name()

    async def analyst(self):
        if self.brief is None:
            handoff(self.coordinator, ANALYST, f"analyze {self.email['path'].name}",
                    email_block(self.email))
            self.brief = await run_analyst(self.email, client=self.client, usage=self.usage)
            (self.run_dir / "brief.json").write_text(json.dumps(self.brief, indent=2),
                                                     encoding="utf-8")
            handoff(ANALYST, self.coordinator, "brief (in full below)")
            print(f"ANALYST'S BRIEF\n{json.dumps(self.brief, indent=2)}\nEND OF BRIEF\n")
        return self.brief

    def save_draft(self, filename, header, outcome, recipient, summary):
        flag = ("YES - " + "; ".join(outcome["reasons"])) if outcome["needs_person"] else "no"
        status = (f"{outcome['verdict']} after {outcome['revisions']} revision(s)")
        body = "\n".join(header + [
            f"Needs a person: {flag}", f"Review: {status}", f"Summary: {summary}",
            "Status: draft only - not sent",
            "", "DRAFT", outcome["draft"], "", "REVIEW NOTES", *outcome["notes"]]) + "\n"
        (self.run_dir / filename).write_text(body, encoding="utf-8")
        record = {"file": filename, "recipient": recipient, "status": status,
                  "summary": summary, "draft_text": outcome["draft"],
                  "needs_person": outcome["needs_person"], "reasons": outcome["reasons"],
                  "saved_to": str((self.run_dir / filename).relative_to(REPO))}
        self.drafts.append(record)
        return record

    def supplier_reply(self):
        if self.supplier_done:
            raise ValueError("A supplier reply was already drafted for this email.")
        self.supplier_done = True
        print(f"{RULE}\nSupplier reply\n{RULE}")
        penalty = any(o["late_penalty"] for o in (self.brief or {}).get("orders_at_risk", []))
        outcome = reviewed_draft(
            SUPPLIER_WRITER,
            lambda prev, fb: write_supplier_reply(self.email, self.brief, previous=prev,
                                                  feedback=fb, client=self.client, usage=self.usage),
            supplier_writer_input(self.email, self.brief), "supplier", "the supplier", penalty,
            coordinator=self.coordinator, client=self.client, usage=self.usage)
        print_final("FINAL DRAFT TO SUPPLIER", outcome)
        if outcome["draft"] == NO_REPLY:
            self.drafts.append({"file": None, "recipient": "supplier", "status": "no reply needed",
                                "needs_person": False, "reasons": []})
            return {"status": "no reply needed"}
        return self.save_draft("supplier-reply.txt", [
            f"To: {self.email['from_name']} <{self.email['sender']}>",
            f"Subject: {reply_subject(self.email['subject'])}",
            f"From: Purchasing, {self.company}",
            f"In reply to: {self.email['path'].name}"], outcome, "supplier",
            draft_summary("supplier", supplier_brief(self.brief), outcome,
                          self.email["from_name"] or self.email["sender"]))

    def customer_notice(self, order_id):
        order_id = str(order_id).strip().upper()
        if self.brief is None:
            raise ValueError("Run the supply analyst first: notices are written from its brief.")
        order = next((o for o in self.brief["orders_at_risk"] if o["order_id"] == order_id), None)
        if order is None:
            listed = ", ".join(o["order_id"] for o in self.brief["orders_at_risk"]) or "none"
            raise ValueError(f"{order_id} is not in the brief's orders_at_risk ({listed}).")
        if any(d["file"] == f"notice-{order_id}.txt" for d in self.drafts):
            raise ValueError(f"A notice for {order_id} was already drafted.")
        # When the supplier offers options, whether this order is late depends on the choice,
        # so the notice is always marked for a person, naming the options that delay it.
        extra = []
        if self.brief["options"]:
            delaying = [str(o.get("name")) for o in self.brief["options"]
                        if order_id in [str(i).strip() for i in o["order_ids"]]]
            extra.append("depends on which option is chosen (late only under "
                         f"{' and '.join(delaying) or 'none of the options'}; the notice uses "
                         "the analyst's numbers for that case)")
        print(f"{RULE}\nCustomer notice for {order_id}\n{RULE}")
        outcome = reviewed_draft(
            CUSTOMER_WRITER,
            lambda prev, fb: write_customer_notice(order, previous=prev, feedback=fb,
                                                   client=self.client, usage=self.usage),
            json.dumps(customer_facts(order), indent=2), "customer", order_id,
            order["late_penalty"], coordinator=self.coordinator, client=self.client,
            usage=self.usage, extra_reasons=extra)
        print_final(f"FINAL DRAFT TO CUSTOMER {order_id}", outcome)
        row = self.orders[order_id]
        return self.save_draft(f"notice-{order_id}.txt", [
            f"To: {row['customer']} <{row['contact_email']}>",
            f"Subject: Update on your {self.company} order {order_id}",
            f"From: Customer Service, {self.company}"], outcome, "customer",
            draft_summary("customer", customer_facts(order), outcome, row["customer"]))

    def result_lines(self):
        """The end-of-run summary: drafts saved, drafts marked for a person, decisions for
        a person, and tokens. Printed at the end and saved as summary.txt."""
        saved = [d for d in self.drafts if d["file"]]
        marked = [d for d in saved if d["needs_person"]]
        decisions = (self.brief or {}).get("decisions_for_a_person", [])
        tin, tout = self.usage.totals()
        lines = [f"Drafts saved ({len(saved)}) - nothing was sent:"]
        lines += [f"  - {d['file']} [{d['recipient']}]: {d['summary']}" for d in saved] or ["  - none"]
        lines += ["", f"Drafts marked for a person ({len(marked)}):"]
        lines += [f"  - {d['file']}: {'; '.join(d['reasons'])}" for d in marked] or ["  - none"]
        lines += ["", f"Decisions for a person ({len(decisions)}):"]
        lines += [f"  - {d}" for d in decisions] or ["  - none"]
        slack = (self.brief or {}).get("slack")
        if slack:
            lines += ["", f"Watch: {slack['note']}"]
        lines += ["", f"Total tokens: {tin + tout:,} ({tin:,} input + {tout:,} output)",
                  "", self.usage.report()]
        return lines

    def finish(self, summary, needs_decision):
        self.summary, self.needs_decision = summary, needs_decision
        lines = [f"Coordinator: {self.coordinator}", f"Email: {self.email['path'].name}",
                 f"Subject: {self.email['subject']}",
                 f"A person needs to make a decision: {'yes' if needs_decision else 'no'}", "",
                 "SUMMARY FOR THE PURCHASING MANAGER", summary, "", *self.result_lines()]
        (self.run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Command line shared by both coordinators
# ---------------------------------------------------------------------------

def run_coordinator(name, description, coordinate):
    """Parse --org/--email/--save, run coordinate(team) for one email, print the results."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--org", metavar="FOLDER",
                        help="company folder: a folder name in week-05 (e.g. copperline-coffee) "
                             "or a path (default: week-05/brookfield-bikes)")
    parser.add_argument("--email", metavar="PATH", required=True,
                        help="one supplier email (.eml), e.g. week-05/brookfield-bikes/"
                             "supplier-emails/01-northgate-controller-delay.eml")
    parser.add_argument("--save", metavar="FILE",
                        help="also write everything printed to FILE (UTF-8), e.g. week-05/runs/...")
    args = parser.parse_args()
    set_org(resolve_org(args.org))
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
        print(f"Company: {team.company} ({org_dir().relative_to(REPO)})\n"
              f"Coordinator: {name}\nEmail: {email['path'].name}\nFrom: {email['sender']}\n"
              f"Subject: {email['subject']}\nSpecialist model: {SPECIALIST_MODEL}\n"
              f"Run folder: {team.run_dir.relative_to(REPO)}\n")
        asyncio.run(coordinate(team))
        print(f"{RULE}\nSUMMARY ({name})\n{RULE}")
        print(f"{team.summary}\n")
        print("\n".join(team.result_lines()))
        print(f"\nDrafts folder: {team.run_dir.relative_to(REPO)}")
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
