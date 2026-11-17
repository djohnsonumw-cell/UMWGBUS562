"""Coordinator 1: a fixed pipeline for one supplier email.

The steps are written in code and run the same way for every email:
  1. the supply analyst writes its brief (with the supply MCP server's tools);
  2. the supplier writer drafts a reply, and the reviewer checks it;
  3. for each order at risk, the customer writer drafts a notice, and the reviewer
     checks it (skipped when the supplier offers options: a person decides first);
  4. code writes the summary for the purchasing manager.
Drafts are saved in week-05/drafts/<run>/. Nothing is sent.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\pipeline.py --email week-05\\brookfield-bikes\\supplier-emails\\01-northgate-controller-delay.eml
    ... --save week-05\\runs\\pipeline-01.txt     (also save the output)
"""

from specialists import run_coordinator


def summarize(brief):
    risk = brief["orders_at_risk"]
    parts = [brief["what_changed"]]
    if brief["options"]:
        for o in brief["options"]:
            ids = ", ".join(o["order_ids"]) or "no orders"
            parts.append(f"{o.get('name')}: puts {ids} at risk.")
        parts.append("No customer notices were drafted, because which orders are late depends "
                     "on the option chosen.")
    elif risk:
        parts.append("Orders at risk: " + "; ".join(
            f"{o['order_id']} ({o['customer']}), {o['units_short']} units short, "
            f"{o['days_late'] if o['days_late'] is not None else 'unknown'} days late"
            f"{', LATE PENALTY' if o['late_penalty'] else ''}" for o in risk) + ".")
    else:
        parts.append("No customer orders are at risk.")
    for d in brief["decisions_for_a_person"]:
        parts.append(f"Decision needed: {d}")
    return " ".join(parts)


async def coordinate(team):
    brief = await team.analyst()
    team.supplier_reply()
    if not brief["options"]:
        for order in brief["orders_at_risk"]:
            team.customer_notice(order["order_id"])
    team.finish(summarize(brief),
                needs_decision=bool(brief["decisions_for_a_person"] or brief["options"]))


if __name__ == "__main__":
    run_coordinator("pipeline", __doc__.split("\n")[0], coordinate)
