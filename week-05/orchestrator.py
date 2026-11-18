"""Coordinator 2: an orchestrator agent for one supplier email.

Claude (claude-sonnet-5, with the prompt in brookfield-bikes/prompts/orchestrator.txt)
decides which specialists the email needs, and calls them through four tools:
  run_supply_analyst, draft_supplier_reply, draft_customer_notice(order_id), finish.
Each tool runs the same specialist code as pipeline.py (specialists.TeamRun), so every
draft goes through the same review step and code checks, and nothing is sent.

The orchestrator may make at most 10 tool calls. If it reaches the limit without
calling finish, it gets one last turn without tools to write its summary, and the job
is marked as needing a decision by a person. Each choice it makes is printed, and at
the end it prints the same summary as the pipeline.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\orchestrator.py --email week-05\\brookfield-bikes\\supplier-emails\\02-kendall-tire-options.eml
    ... --save week-05\\runs\\orchestrator-02.txt     (also write everything printed to that file)
"""

import json

from common import (RULE, STOP_REASONS, ask_claude, email_block, preview, read_prompt,
                    response_text)
from specialists import run_coordinator

ORCHESTRATOR_MODEL = "claude-sonnet-5"
AGENT = "orchestrator"
MAX_TOOL_CALLS = 10
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
            "it has been run. The draft is reviewed (up to 2 revisions), checked, and saved for a "
            "person; it is never sent. Returns a short summary of the draft, the review status, "
            "and whether a person must check it. Takes no inputs. Can be called once."
        ),
        "input_schema": NO_INPUT,
        "strict": True,
    },
    {
        "name": "draft_customer_notice",
        "description": (
            "Have the customer writer draft a notice to the customer of one order in the "
            "analyst's orders_at_risk. The draft is reviewed (up to 2 revisions), checked, and "
            "saved for a person; it is never sent. If the supplier offered options, the notice "
            "is marked for a person, since the delay depends on the option chosen. Returns a "
            "short summary of the draft and the review status. Call once per order."
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
        "description": (
            "End the job. Call exactly once, last. You never see the drafts themselves: "
            "describe each draft only from the summary its drafting tool returned."),
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


async def run_tool(team, name, args):
    """Run one of the orchestrator's tools. Returns (result text, is_error)."""
    try:
        if name == "run_supply_analyst":
            return json.dumps(await team.analyst(), indent=2), False
        if name == "draft_supplier_reply":
            return json.dumps(team.supplier_reply()), False
        if name == "draft_customer_notice":
            return json.dumps(team.customer_notice(args.get("order_id", ""))), False
        if name == "finish":
            if not isinstance(args.get("summary"), str) or not args["summary"].strip():
                raise ValueError("summary must be a non-empty string.")
            if not isinstance(args.get("needs_decision"), bool):
                raise ValueError("needs_decision must be true or false.")
            team.finish(args["summary"].strip(), args["needs_decision"])
            return "Job finished.", False
        return f"Unknown tool: {name}", True
    except ValueError as exc:
        return f"Error: {exc}", True


def call(team, system, messages, tool_choice=None):
    response = ask_claude(team.client, team.usage, AGENT, system, messages, TOOLS,
                          model=ORCHESTRATOR_MODEL, tool_choice=tool_choice)
    u = response.usage
    print(f"    [{AGENT}] tokens this call: {u.input_tokens:,} in, {u.output_tokens:,} out")
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason not in ("tool_use", "end_turn"):
        raise RuntimeError(f"The orchestrator {STOP_REASONS.get(response.stop_reason, response.stop_reason)}.")
    note = response_text(response)
    if note:
        print(f"[{AGENT}] says: {note}")
    return response


async def coordinate(team):
    system = read_prompt("orchestrator")
    messages = [{"role": "user", "content": (
        f"Coordinate Brookfield's response to this supplier email. You can make at most "
        f"{MAX_TOOL_CALLS} tool calls, including finish.\n\n{email_block(team.email)}")}]
    print(f"{RULE}\nOrchestrator ({ORCHESTRATOR_MODEL}; at most {MAX_TOOL_CALLS} tool calls)\n{RULE}")
    calls, finished, nudged = 0, False, False

    while not finished:
        out_of_calls = calls >= MAX_TOOL_CALLS
        response = call(team, system, messages, {"type": "none"} if out_of_calls else None)

        if response.stop_reason != "tool_use":
            text = response_text(response)
            if out_of_calls or nudged:
                # No tool calls left (or it won't call finish): close the job in code.
                why = (f"reached the limit of {MAX_TOOL_CALLS} tool calls" if out_of_calls
                       else "stopped without calling finish")
                print(f"[{AGENT}] {why}; the job is closed and marked for a person.")
                team.finish(f"The orchestrator {why}. Its summary: {text or '(none)'}",
                            needs_decision=True)
                break
            nudged = True
            messages.append({"role": "user", "content": "Call finish to end the job."})
            continue

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if finished or calls >= MAX_TOOL_CALLS:
                text, is_error = ("Not run: the job is already finished." if finished else
                                  f"Not run: the limit of {MAX_TOOL_CALLS} tool calls was reached."), True
            else:
                calls += 1
                args = json.dumps(block.input) if block.input else ""
                print(f"\n[{AGENT}] CHOICE {calls}/{MAX_TOOL_CALLS}: {block.name} {args}".rstrip())
                # The specialists print their own handoffs (in specialists.TeamRun).
                text, is_error = await run_tool(team, block.name, block.input)
                finished = block.name == "finish" and not is_error
                print(f"[{AGENT}] result of {block.name}{' (error)' if is_error else ''}: "
                      f"{preview(text, 400)}")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": text, "is_error": is_error})
        if finished:
            break
        if calls >= MAX_TOOL_CALLS:
            print(f"[{AGENT}] used all {MAX_TOOL_CALLS} tool calls; one last turn to summarize.")
            results.append({"type": "text", "text": (
                f"You have used all {MAX_TOOL_CALLS} tool calls and can't call finish. Write "
                "your summary for the purchasing manager now, in plain text.")})
        messages.append({"role": "user", "content": results})

    print(f"[{AGENT}] done after {calls} tool call(s).")


if __name__ == "__main__":
    run_coordinator("orchestrator", __doc__.split("\n")[0], coordinate)
