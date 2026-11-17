"""Coordinator 2: an orchestrator agent for one supplier email.

Claude, with the prompt in brookfield-bikes/prompts/orchestrator.txt, decides which
specialists the email needs and calls them through four tools:
  run_supply_analyst, draft_supplier_reply, draft_customer_notice(order_id), finish.
Each tool runs the same specialist code as pipeline.py (specialists.TeamRun), so the
same checks apply: every draft is reviewed and saved, customer notices are only for
orders in the analyst's checked brief, and nothing is sent.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\orchestrator.py --email week-05\\brookfield-bikes\\supplier-emails\\02-kendall-tire-options.eml
    ... --save week-05\\runs\\orchestrator-02.txt     (also save the output)
"""

import json

from common import RULE, email_block, read_prompt, run_tool_loop
from specialists import run_coordinator

MAX_TOOL_CALLS = 12
NO_INPUT = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

TOOLS = [
    {
        "name": "run_supply_analyst",
        "description": (
            "Have the supply analyst work out how this email affects customer orders, using "
            "Brookfield's stock, order, and shipment data. Returns the analyst's brief as JSON: "
            "part_id, what_changed, orders_at_risk, options, decisions_for_a_person, "
            "supplier_questions. Takes no inputs; the analyst already has the email."
        ),
        "input_schema": NO_INPUT,
        "strict": True,
    },
    {
        "name": "draft_supplier_reply",
        "description": (
            "Have the supplier writer draft a reply to the supplier, using the analyst's brief if "
            "it has been run. The draft is reviewed and saved for a person to check; it is never "
            "sent. Returns the review status. Takes no inputs. Can be called once."
        ),
        "input_schema": NO_INPUT,
        "strict": True,
    },
    {
        "name": "draft_customer_notice",
        "description": (
            "Have the customer writer draft a notice to the customer of one order in the "
            "analyst's orders_at_risk. The draft is reviewed and saved for a person to check; it "
            "is never sent. Returns the review status. Call once per order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string",
                                        "description": "An order_id from orders_at_risk, e.g. O-1043."}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "finish",
        "description": "End the job. Call exactly once, last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string",
                            "description": "A short summary for Brookfield's purchasing manager."},
                "needs_decision": {"type": "boolean",
                                   "description": "True if a person at Brookfield must make a decision."},
            },
            "required": ["summary", "needs_decision"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


async def coordinate(team):
    done = {"value": False}

    async def handle(name, args):
        try:
            if name == "run_supply_analyst":
                if team.brief is not None:
                    return json.dumps(team.brief, indent=2), False
                brief = await team.analyst()
                print(f"{RULE}\nBack to the orchestrator\n{RULE}")
                return json.dumps(brief, indent=2), False
            if name == "draft_supplier_reply":
                result = team.supplier_reply()
            elif name == "draft_customer_notice":
                result = team.customer_notice(args.get("order_id", ""))
            elif name == "finish":
                if not isinstance(args.get("summary"), str) or not args["summary"].strip():
                    raise ValueError("summary must be a non-empty string.")
                if not isinstance(args.get("needs_decision"), bool):
                    raise ValueError("needs_decision must be true or false.")
                team.finish(args["summary"].strip(), args["needs_decision"])
                done["value"] = True
                return "Job finished.", False
            else:
                return f"Unknown tool: {name}", True
            print(f"{RULE}\nBack to the orchestrator\n{RULE}")
            return json.dumps(result), False
        except ValueError as exc:
            return f"Error: {exc}", True

    messages = [{"role": "user", "content": (
        f"Coordinate Brookfield's response to this supplier email.\n\n{email_block(team.email)}")}]
    print(f"{RULE}\nOrchestrator\n{RULE}")
    text, stop = await run_tool_loop(team.client, team.usage, "orchestrator",
                                     read_prompt("orchestrator"), TOOLS, messages, handle,
                                     MAX_TOOL_CALLS, is_done=lambda: done["value"])
    print(f"[orchestrator] {stop}.")
    if not done["value"]:
        team.finish(f"The orchestrator stopped without calling finish ({stop}). "
                    f"Its last words: {text or '(none)'}", needs_decision=True)


if __name__ == "__main__":
    run_coordinator("orchestrator", __doc__.split("\n")[0], coordinate)
