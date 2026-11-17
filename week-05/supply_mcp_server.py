"""MCP server for Brookfield Bicycle Company's supply data: parts, orders, coverage.

Built with the official Python MCP SDK (the `mcp` package). It reads the CSV files in
week-05/brookfield-bikes (or the folder given with --data) each time a tool runs, so
edits to the data take effect right away. Days are counted from today (day 0).

Tools (all read-only)
- get_part(part_id): stock on hand, scheduled shipments, and the bike models that
  use the part (with parts per bike).
- get_orders_for_part(part_id): the open orders that need the part, earliest ship
  day first, with units needed (bikes x parts per bike), ship day, and late penalty.
- check_coverage(part_id, arrivals): a new list of arrivals ({qty, day}) that
  replaces ALL of the part's scheduled shipments. Orders are filled complete, in
  ship-day order: an order ships only when stock covers all of it, and no order
  ships ahead of an earlier one, so a short order holds up the ones after it. Each
  order comes back covered, or with its units short on its ship day and the first
  day it can ship (ready_day; null if this schedule never covers it).

Because orders are filled complete and in order, an order can ship on the first day
that stock on hand plus everything arrived by then (an arrival on that day counts)
covers it and every order before it. All arithmetic is done here, in code, so an
agent never has to work out coverage itself.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py                 # serve over stdio
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py --show MC-500    # part + coverage as scheduled
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py --show MC-500 ^
        --arrivals "[{\\"qty\\": 120, \\"day\\": 5}, {\\"qty\\": 280, \\"day\\": 26}]"

Add it to Claude Code (from the top of the repository):
    claude mcp add brookfield-supply -- .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py

The server never calls Claude, so it needs no API key. In stdio mode, stdout
carries the MCP protocol, so all logging goes to stderr.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "brookfield-bikes"
DATA_DIR = DEFAULT_DATA  # set from --data in main()
DATA_FILES = ("parts.csv", "orders.csv", "inbound.csv", "bill-of-materials.csv")

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                            idempotentHint=True, openWorldHint=False)


class Arrival(TypedDict):
    qty: Annotated[int, Field(ge=0, description="Units arriving.")]
    day: Annotated[int, Field(ge=0, description="Day they arrive, counted from today (day 0).")]


def log(message):
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_csv(name):
    with open(DATA_DIR / name, newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def to_int(value, what):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolError(f"Bad number in the data for {what}: {value!r}") from None


def find_part(part_id):
    key = (part_id or "").strip().upper()
    parts = {p["part_id"].upper(): p for p in read_csv("parts.csv")}
    if key not in parts:
        raise ToolError(f"Unknown part_id {part_id!r}. Known parts: {', '.join(sorted(parts))}.")
    return key, parts[key]


def part_info(part_id):
    part_id, part = find_part(part_id)
    models = [{"model": b["model"], "parts_per_bike": to_int(b["qty_per_bike"], "qty_per_bike")}
              for b in read_csv("bill-of-materials.csv") if b["part_id"].upper() == part_id]
    shipments = [{"po_number": r["po_number"], "supplier": r["supplier"],
                  "qty": to_int(r["qty"], r["po_number"]),
                  "day": to_int(r["arrives_in_days"], r["po_number"])}
                 for r in read_csv("inbound.csv") if r["part_id"].upper() == part_id]
    return {
        "part_id": part_id,
        "description": part["description"],
        "supplier": part["supplier"],
        "on_hand": to_int(part["on_hand"], f"{part_id} on_hand"),
        "scheduled_shipments": sorted(shipments, key=lambda s: s["day"]),
        "used_in": models,
    }


def orders_for_part(part_id):
    """(part info, orders that need the part, earliest ship day first; ties by order_id)."""
    info = part_info(part_id)
    per_bike = {m["model"]: m["parts_per_bike"] for m in info["used_in"]}
    orders = []
    for o in read_csv("orders.csv"):
        if o["model"] not in per_bike:
            continue
        bikes = to_int(o["qty"], f"{o['order_id']} qty")
        orders.append({
            "order_id": o["order_id"],
            "customer": o["customer"],
            "model": o["model"],
            "bikes": bikes,
            "units_needed": bikes * per_bike[o["model"]],
            "ship_day": to_int(o["ship_in_days"], f"{o['order_id']} ship_in_days"),
            "late_penalty": o["late_penalty"].strip().lower() == "yes",
        })
    orders.sort(key=lambda o: (o["ship_day"], o["order_id"]))
    return info, orders


def check_arrivals(arrivals):
    if not isinstance(arrivals, list):
        raise ToolError("arrivals must be a list of {qty, day}.")
    clean = []
    for n, a in enumerate(arrivals, 1):
        qty = a.get("qty") if isinstance(a, dict) else None
        day = a.get("day") if isinstance(a, dict) else None
        for name, value in (("qty", qty), ("day", day)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ToolError(f"arrivals item {n}: {name} must be a whole number, 0 or more.")
        clean.append({"qty": qty, "day": day})
    return sorted(clean, key=lambda a: a["day"])


def coverage(part_id, arrivals=None):
    """Coverage of each order. arrivals (if given) replaces the part's scheduled shipments."""
    info, orders = orders_for_part(part_id)
    scheduled = [{"qty": s["qty"], "day": s["day"]} for s in info["scheduled_shipments"]]
    used = scheduled if arrivals is None else check_arrivals(arrivals)

    def stock_by(day):
        return info["on_hand"] + sum(a["qty"] for a in used if a["day"] <= day)

    candidate_days = sorted({0} | {a["day"] for a in used})
    needed_through = 0  # units for this order and every order before it
    results = []
    for o in orders:
        needed_through += o["units_needed"]
        available_for_it = stock_by(o["ship_day"]) - (needed_through - o["units_needed"])
        if available_for_it >= o["units_needed"]:
            results.append({**o, "covered": True})
            continue
        ready_day = next((d for d in candidate_days if stock_by(d) >= needed_through), None)
        if ready_day is not None:
            ready_day = max(ready_day, o["ship_day"])
        results.append({
            **o,
            "covered": False,
            "units_short": min(o["units_needed"], o["units_needed"] - available_for_it),
            "ready_day": ready_day,
            "days_late": None if ready_day is None else ready_day - o["ship_day"],
        })

    return {
        "part_id": info["part_id"],
        "on_hand": info["on_hand"],
        "arrivals_used": used,
        "replaced_schedule": scheduled if arrivals is not None else None,
        "total_units_needed": needed_through,
        "total_units_available": info["on_hand"] + sum(a["qty"] for a in used),
        "orders": results,
        "orders_at_risk": [r["order_id"] for r in results if not r["covered"]],
        "note": ("Orders are filled complete, in ship-day order. For an order that isn't "
                 "covered: units_short is how many more units it needs on its ship day, and "
                 "ready_day is the first day it can ship (null: never, with these arrivals). "
                 "days_late = ready_day - ship_day."),
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def create_server():
    server = MCPServer(
        name="brookfield-supply",
        instructions=(
            "Brookfield Bicycle Company supply data. Days are counted from today (day 0). "
            "Use get_part and get_orders_for_part to look things up, and check_coverage for "
            "any question about whether orders will be covered. Never work out coverage "
            "yourself, and report only numbers that came from these tools."
        ),
    )

    def get_part(part_id: str) -> dict[str, Any]:
        return part_info(part_id)

    server.add_tool(get_part, name="get_part", annotations=READ_ONLY, description=(
        "Look up one part by part_id (for example MC-500). Returns its description, supplier, "
        "units on hand, scheduled shipments (po_number, qty, day: days from today), and the "
        "bike models that use it, with parts per bike."
    ))

    def get_orders_for_part(part_id: str) -> dict[str, Any]:
        info, orders = orders_for_part(part_id)
        return {"part_id": info["part_id"], "orders": orders}

    server.add_tool(get_orders_for_part, name="get_orders_for_part", annotations=READ_ONLY,
                    description=(
        "List the open customer orders that need a part, earliest ship day first. Each order "
        "has order_id, customer, model, bikes, units_needed (bikes x parts per bike), ship_day "
        "(days from today), and late_penalty (true or false)."
    ))

    def check_coverage(part_id: str, arrivals: list[Arrival]) -> dict[str, Any]:
        return coverage(part_id, arrivals)

    server.add_tool(check_coverage, name="check_coverage", annotations=READ_ONLY, description=(
        "Check whether a part's open orders are covered by stock on hand plus a new list of "
        "arrivals. arrivals REPLACES all of the part's scheduled shipments: list every arrival "
        "you expect, each as {\"qty\": units, \"day\": days from today}. To check the current "
        "schedule, pass the scheduled shipments from get_part. Orders are filled complete, in "
        "ship-day order, so a short order holds up the ones after it. Each order comes back "
        "with covered true, or covered false with units_short (on its ship day), ready_day (the "
        "first day it can ship; null if never), days_late, and late_penalty. Call it once per "
        "schedule or option you need to check."
    ))
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", metavar="FOLDER", help="data folder (default: week-05/brookfield-bikes)")
    parser.add_argument("--show", metavar="PART_ID",
                        help="print the part and its coverage, then exit (for testing)")
    parser.add_argument("--arrivals", metavar="JSON",
                        help='with --show: new arrivals, e.g. [{"qty": 120, "day": 5}]')
    args = parser.parse_args()

    global DATA_DIR
    if args.data:
        DATA_DIR = Path(args.data).resolve()
    missing = [n for n in DATA_FILES if not (DATA_DIR / n).is_file()]
    if missing:
        sys.exit(f"{DATA_DIR} is missing: {', '.join(missing)}")

    if args.show:
        sys.stdout.reconfigure(encoding="utf-8")
        try:
            arrivals = json.loads(args.arrivals) if args.arrivals else None
            print(json.dumps(part_info(args.show), indent=2))
            print(json.dumps(coverage(args.show, arrivals), indent=2))
        except (ToolError, json.JSONDecodeError) as exc:
            sys.exit(f"Error: {exc}")
        return

    log(f"brookfield-supply MCP server ready (data: {DATA_DIR.name}).")
    create_server().run()  # stdio


if __name__ == "__main__":
    main()
