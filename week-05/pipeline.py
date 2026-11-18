"""Coordinator 1: a fixed pipeline for one supplier email.

The specialists from specialists.py run in the same order for every email:
  1. the supply analyst writes its brief (with the supply MCP server's tools);
  2. a customer notice for each order at risk (skipped if there are none). Each notice
     goes through the review step;
  3. a reply to the supplier, which also goes through the review step.
At the end it prints a summary: drafts saved, drafts marked for a person, decisions for
a person, and total tokens. Nothing is sent.

If the supplier offers options, the notices are still drafted, but each one is marked
for a person: whether that order is late depends on the option a person chooses.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\pipeline.py --email week-05\\brookfield-bikes\\supplier-emails\\01-northgate-controller-delay.eml
    .venv\\Scripts\\python.exe week-05\\pipeline.py --org copperline-coffee --email week-05\\copperline-coffee\\supplier-emails\\01-danforth-pump-delay.eml
    ... --save week-05\\runs\\pipeline-01.txt     (also write everything printed to that file)

--org names the company folder (default: week-05/brookfield-bikes). Drafts are saved in
week-05/drafts/<company folder>/.
"""

from common import RULE
from specialists import run_coordinator


def summarize(brief):
    """A short summary for the purchasing manager, built in code from the brief."""
    parts = [brief["what_changed"]]
    if brief["options"]:
        for o in brief["options"]:
            ids = ", ".join(o["order_ids"]) or "no orders"
            parts.append(f"{o.get('name')}: puts {ids} at risk.")
    elif brief["orders_at_risk"]:
        parts.append("Orders at risk: " + "; ".join(
            f"{o['order_id']} ({o['customer']}), {o['units_short']} units short, "
            f"{o['days_late'] if o['days_late'] is not None else 'unknown'} "
            f"day{'' if o['days_late'] == 1 else 's'} late"
            f"{', LATE PENALTY' if o['late_penalty'] else ''}"
            for o in brief["orders_at_risk"]) + ".")
    else:
        parts.append("No customer orders are at risk.")
        if brief.get("slack"):
            parts.append(brief["slack"]["note"])
    return " ".join(parts)


async def coordinate(team):
    # Step 1: the supply analyst.
    print(f"{RULE}\nSTEP 1 of 3: supply analyst\n{RULE}")
    brief = await team.analyst()

    # Step 2: a customer notice for each order at risk.
    print(f"{RULE}\nSTEP 2 of 3: customer notices\n{RULE}")
    if not brief["orders_at_risk"]:
        print("Skipped: no orders are at risk.")
    else:
        if brief["options"]:
            print("The supplier offered options: each notice will be marked for a person.")
        for order in brief["orders_at_risk"]:
            team.customer_notice(order["order_id"])

    # Step 3: a reply to the supplier.
    print(f"{RULE}\nSTEP 3 of 3: supplier reply\n{RULE}")
    team.supplier_reply()

    team.finish(summarize(brief),
                needs_decision=bool(brief["decisions_for_a_person"] or brief["options"]))


if __name__ == "__main__":
    run_coordinator("pipeline", __doc__.split("\n")[0], coordinate)
