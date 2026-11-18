"""MCP server for a company's supply data: parts, orders, coverage.

Built with the official Python MCP SDK (the `mcp` package). It serves the company folder
named with --org (default: week-05/brookfield-bikes), which holds parts.csv, orders.csv,
inbound.csv, bill-of-materials.csv, and suppliers.csv. It reads the CSV files each time a
tool runs, so edits to the data take effect right away. Days are counted from today (day 0).

Column names differ between companies, so none are assumed. load_table() works out from
each file's header row which column holds what (for example, the product, the quantity,
the parts per unit, and the ship day), and returns rows under standard names. Run with
--columns to see what it found.

Tools (all read-only)
- get_part(part_id): stock on hand, scheduled shipments, and the products that use the
  part (with parts per unit).
- get_orders_for_part(part_id): the open orders that need the part, earliest ship day
  first, with units needed (quantity x parts per unit), ship day, and late penalty.
- check_coverage(part_id, arrivals): a new list of arrivals ({qty, day}) that replaces
  ALL of the part's scheduled shipments. Orders are filled complete, in ship-day order:
  an order ships only when stock covers all of it, and no order ships ahead of an
  earlier one, so a short order holds up the ones after it. Each order comes back
  covered, or with its units short on its ship day and the first day it can ship
  (ready_day; null if this schedule never covers it).

Because orders are filled complete and in order, an order can ship on the first day that
stock on hand plus everything arrived by then (an arrival on that day counts) covers it
and every order before it. All arithmetic is done here, in code, so an agent never has to
work out coverage itself.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py                          # serve Brookfield over stdio
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py --org copperline-coffee  # serve Copperline
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py --org copperline-coffee --columns
    .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py --show MC-500 ^
        --arrivals "[{\\"qty\\": 120, \\"day\\": 5}, {\\"qty\\": 280, \\"day\\": 26}]"

Add it to Claude Code (from the top of the repository):
    claude mcp add brookfield-supply -- .venv\\Scripts\\python.exe week-05\\supply_mcp_server.py

The server never calls Claude, so it needs no API key. In stdio mode, stdout carries the
MCP protocol, so all logging goes to stderr.
"""

import argparse
import csv
import json
import re
import sys
from typing import Annotated, Any, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from common import ORG_CSV_FILES, org_dir, org_name, resolve_org, set_org

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                            idempotentHint=True, openWorldHint=False)


class Arrival(TypedDict):
    qty: Annotated[int, Field(ge=0, description="Units arriving.")]
    day: Annotated[int, Field(ge=0, description="Day they arrive, counted from today (day 0).")]


def log(message):
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Working out the columns from each header row
# ---------------------------------------------------------------------------

# For each file: each standard field, with patterns tried in order against the header names
# (lower case, with spaces and punctuation turned into "_"). Fields are matched in the order
# listed, and a column used by one field can't be used by another. Fields marked optional
# may be missing.
SCHEMAS = {
    "parts.csv": [
        ("part_id", [r"^part(_?(id|no|number|code))?$", r"^(sku|item_?id)$", r"part"]),
        ("on_hand", [r"on_?hand", r"in_?stock", r"stock|inventory"]),
        ("supplier", [r"^(supplier|vendor)(_?name)?$", r"supplier|vendor"]),
        ("description", [r"desc", r"^name$"], "optional"),
    ],
    "bill-of-materials.csv": [
        ("part_id", [r"^part(_?(id|no|number|code))?$", r"part"]),
        ("parts_per_unit", [r"(^|_)per(_|$)", r"qty_?per|quantity_?per", r"usage|needed"]),
        ("product", [r"^(model|product|item|machine|bike)(_?(name|id))?$",
                     r"model|product|machine"]),
    ],
    "inbound.csv": [
        ("po_number", [r"^po(_?(number|no|id))?$", r"purchase_?order", r"^po"]),
        ("part_id", [r"^part(_?(id|no|number|code))?$", r"part"]),
        ("supplier", [r"^(supplier|vendor)(_?name)?$", r"supplier|vendor"], "optional"),
        ("day", [r"arriv.*day|day.*arriv", r"eta|due", r"day"]),
        ("qty", [r"^(qty|quantity|units|amount|count)$", r"qty|quantity|units"]),
    ],
    "orders.csv": [
        ("order_id", [r"^order(_?(id|no|number))?$", r"order"]),
        ("contact_email", [r"e_?mail"]),
        ("customer", [r"^(customer|client|account)(_?name)?$", r"customer|client"]),
        ("late_penalty", [r"penalt", r"late"]),
        ("ship_day", [r"ship.*day|day.*ship", r"due|deadline", r"day"]),
        ("product", [r"^(model|product|item|machine|bike)(_?(name|id))?$",
                     r"model|product|machine"]),
        ("quantity", [r"^(qty|quantity|units|count|amount|bikes|machines)$",
                      r"qty|quantity|units"]),
    ],
    "suppliers.csv": [
        ("contact_email", [r"e_?mail"]),
        ("supplier", [r"^(supplier|vendor|company)(_?name)?$", r"supplier|vendor"]),
        ("contact_name", [r"contact_?name|^contact$", r"name"]),
        ("part_id", [r"^part(_?(id|no|number|code))?$", r"part"], "optional"),
    ],
}


class DataError(ToolError):
    pass


def normalize(header):
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def map_columns(name, headers):
    """{standard field: column name in this file}, worked out from the header row."""
    free = list(headers)
    mapping = {}
    for field, patterns, *flags in SCHEMAS[name]:
        for pattern in patterns:
            hits = [h for h in free if re.search(pattern, normalize(h))]
            if len(hits) > 1:
                raise DataError(f"{name}: can't tell which column holds {field}: "
                                f"{', '.join(hits)} all look like it.")
            if hits:
                mapping[field] = hits[0]
                free.remove(hits[0])
                break
        else:
            if not flags:
                raise DataError(f"{name}: no column looks like {field}. "
                                f"Columns: {', '.join(headers)}.")
    return mapping


def load_table(name):
    """Rows of one CSV file in the company folder, with standard field names."""
    with open(org_dir() / name, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    mapping = map_columns(name, headers)
    return [{field: row.get(col, "") for field, col in mapping.items()} for row in rows]


def column_report():
    """Which column was found for each standard field, file by file."""
    lines = [f"Company folder: {org_dir()}"]
    for name in ORG_CSV_FILES:
        with open(org_dir() / name, newline="", encoding="utf-8-sig") as f:
            headers = [h.strip() for h in next(csv.reader(f), [])]
        mapping = map_columns(name, headers)
        lines.append(f"{name}: " + ", ".join(f"{k} = {v}" for k, v in mapping.items()))
    return "\n".join(lines)


def to_int(value, what):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolError(f"Bad number in the data for {what}: {value!r}") from None


def is_yes(value):
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def find_part(part_id):
    key = (part_id or "").strip().upper()
    parts = {p["part_id"].upper(): p for p in load_table("parts.csv")}
    if key not in parts:
        raise ToolError(f"Unknown part_id {part_id!r}. Known parts: {', '.join(sorted(parts))}.")
    return key, parts[key]


def part_info(part_id):
    part_id, part = find_part(part_id)
    products = [{"product": b["product"],
                 "parts_per_unit": to_int(b["parts_per_unit"], f"{b['product']} parts per unit")}
                for b in load_table("bill-of-materials.csv") if b["part_id"].upper() == part_id]
    shipments = [{"po_number": r["po_number"], "supplier": r.get("supplier", part["supplier"]),
                  "qty": to_int(r["qty"], r["po_number"]),
                  "day": to_int(r["day"], r["po_number"])}
                 for r in load_table("inbound.csv") if r["part_id"].upper() == part_id]
    return {
        "part_id": part_id,
        "description": part.get("description", ""),
        "supplier": part["supplier"],
        "on_hand": to_int(part["on_hand"], f"{part_id} on hand"),
        "scheduled_shipments": sorted(shipments, key=lambda s: s["day"]),
        "used_in": products,
    }


def orders_for_part(part_id):
    """(part info, orders that need the part, earliest ship day first; ties by order_id)."""
    info = part_info(part_id)
    per_unit = {p["product"]: p["parts_per_unit"] for p in info["used_in"]}
    orders = []
    for o in load_table("orders.csv"):
        if o["product"] not in per_unit:
            continue
        quantity = to_int(o["quantity"], f"{o['order_id']} quantity")
        orders.append({
            "order_id": o["order_id"],
            "customer": o["customer"],
            "product": o["product"],
            "quantity": quantity,
            "units_needed": quantity * per_unit[o["product"]],
            "ship_day": to_int(o["ship_day"], f"{o['order_id']} ship day"),
            "late_penalty": is_yes(o["late_penalty"]),
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
        name=f"{org_name()}-supply",
        instructions=(
            f"Supply data for {org_name()}. Days are counted from today (day 0). "
            "Use get_part and get_orders_for_part to look things up, and check_coverage for "
            "any question about whether orders will be covered. Never work out coverage "
            "yourself, and report only numbers that came from these tools."
        ),
    )

    def get_part(part_id: str) -> dict[str, Any]:
        return part_info(part_id)

    server.add_tool(get_part, name="get_part", annotations=READ_ONLY, description=(
        "Look up one part by part_id. Returns its description, supplier, units on hand, "
        "scheduled shipments (po_number, qty, day: days from today), and the products that "
        "use it, with parts per unit."
    ))

    def get_orders_for_part(part_id: str) -> dict[str, Any]:
        info, orders = orders_for_part(part_id)
        return {"part_id": info["part_id"], "orders": orders}

    server.add_tool(get_orders_for_part, name="get_orders_for_part", annotations=READ_ONLY,
                    description=(
        "List the open customer orders that need a part, earliest ship day first. Each order "
        "has order_id, customer, product, quantity, units_needed (quantity x parts per unit), "
        "ship_day (days from today), and late_penalty (true or false)."
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
    parser.add_argument("--org", "--data", metavar="FOLDER",
                        help="company folder: a folder name in week-05 (e.g. copperline-coffee) "
                             "or a path (default: week-05/brookfield-bikes)")
    parser.add_argument("--columns", action="store_true",
                        help="print which column was found for each field, then exit")
    parser.add_argument("--show", metavar="PART_ID",
                        help="print the part and its coverage, then exit (for testing)")
    parser.add_argument("--arrivals", metavar="JSON",
                        help='with --show: new arrivals, e.g. [{"qty": 120, "day": 5}]')
    args = parser.parse_args()
    set_org(resolve_org(args.org))

    if args.columns or args.show:
        sys.stdout.reconfigure(encoding="utf-8")
        try:
            if args.columns:
                print(column_report())
            if args.show:
                arrivals = json.loads(args.arrivals) if args.arrivals else None
                print(json.dumps(part_info(args.show), indent=2))
                print(json.dumps(coverage(args.show, arrivals), indent=2))
        except (ToolError, json.JSONDecodeError) as exc:
            sys.exit(f"Error: {exc}")
        return

    try:
        column_report()  # fail at start-up, not mid-run, if a column can't be found
    except ToolError as exc:
        sys.exit(f"Error: {exc}")
    log(f"{org_name()}-supply MCP server ready (data: {org_dir()}).")
    create_server().run()  # stdio


if __name__ == "__main__":
    main()
