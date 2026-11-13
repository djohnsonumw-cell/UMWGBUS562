"""MCP server for an organization's HR: policy search and leave balances.

The organization is a folder holding hr-docs/ (HTML policies), employees.csv,
sample-emails/, and system_prompt.txt. Choose it with --org (a path, or a folder
name inside week-04); without --org it uses week-04/hartwell-hr.

Tools
- search_policy(query): the 4 HR-document sections closest to the query, each
  with its source file, so the caller can cite it.
- get_leave_balance: the leave balances from employees.csv. The column names
  aren't fixed: the email, name, and balance columns are worked out from the
  header row (for example "email" or "Email Address"; "vacation_days_left" or
  "PTO Days Remaining"). Columns that aren't recognized are never returned.
  * HR mode (default): takes an employee's work email and returns that record.
  * Employee mode (--as-employee ADDRESS): takes no inputs and returns only that
    one employee's balance, so a client can't look up anyone else.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py                       # HR mode (stdio)
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --org ridgeline-hotels --search "PTO carryover"
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --rebuild             # rebuild the index, then exit
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --search "jury duty"  # test a search, then exit
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --org ridgeline-hotels --columns  # show column mapping

The search reuses the approach from week-03/ask.py: each HTML document is split
into one chunk per section, each chunk starts with the document title and the
section heading, and the chunks are stored in a Chroma database using Chroma's
built-in embedding model. Each organization has its own database in
week-04/chroma_db/<org folder name>/ (git-ignored). The index is rebuilt
automatically when the documents change, or on demand with --rebuild.

The server never calls Claude, so it needs no API key. In stdio mode, stdout
carries the MCP protocol, so all logging goes to stderr.
"""

import argparse
import contextlib
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
from bs4 import BeautifulSoup
from chromadb.config import Settings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

HERE = Path(__file__).resolve().parent
DEFAULT_ORG = HERE / "hartwell-hr"
DB_ROOT = HERE / "chroma_db"  # git-ignored; one subfolder per organization
COLLECTION = "hr_docs"

# Set by configure(); every function reads these at call time.
DATA_DIR = DOCS_DIR = EMPLOYEES_CSV = DB_DIR = MANIFEST = None


def resolve_org(value):
    """--org value -> organization folder: a path, or a folder name inside week-04."""
    if not value:
        return DEFAULT_ORG
    for candidate in (Path(value), HERE / value):
        if candidate.is_dir():
            return candidate.resolve()
    sys.exit(f"Organization folder not found: {value}")


def configure(org_dir):
    """Point the server at one organization folder and its own Chroma database."""
    global DATA_DIR, DOCS_DIR, EMPLOYEES_CSV, DB_DIR, MANIFEST
    DATA_DIR = Path(org_dir)
    DOCS_DIR = DATA_DIR / "hr-docs"
    EMPLOYEES_CSV = DATA_DIR / "employees.csv"
    DB_DIR = DB_ROOT / DATA_DIR.name
    MANIFEST = DB_DIR / "manifest.json"
    missing = [p.name for p in (DOCS_DIR, EMPLOYEES_CSV) if not p.exists()]
    if missing:
        sys.exit(f"{DATA_DIR} is missing: {', '.join(missing)}")


def org_name():
    """The organization's name from a policy's <title> ('Policy | Org'), else the folder name."""
    for path in doc_files():
        with contextlib.suppress(Exception):
            soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
            if soup.title:
                _, org = split_title(soup.title.get_text())
                if org:
                    return org
    return DATA_DIR.name.replace("-", " ").title()

TOP_K = 4
MAX_CHUNK_CHARS = 1500
SUPPORTED = {".html", ".htm"}

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                            idempotentHint=True, openWorldHint=False)


def log(message):
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Parsing and chunking (adapted from week-03/ask.py, HTML only)
# ---------------------------------------------------------------------------

def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def split_title(raw):
    """'Remote Work Policy | Hartwell Foods' -> ('Remote Work Policy', 'Hartwell Foods')."""
    parts = [p.strip() for p in raw.split("|")]
    return parts[0], (parts[1] if len(parts) > 1 else "")


def html_block_text(el):
    """Readable text for one block element, keeping list and table structure."""
    if el.name in ("ul", "ol"):
        return "\n".join("- " + clean(li.get_text(" ")) for li in el.find_all("li"))
    if el.name == "table":
        rows = []
        for tr in el.find_all("tr"):
            rows.append(" | ".join(clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])))
        return "\n".join(rows)
    return clean(el.get_text(" "))


def parse_html(path):
    """Return (title, org, note, sections), where sections is [(heading, text)]."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title_tag, org = split_title(soup.title.get_text()) if soup.title else ("", "")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    root = soup.find("main") or soup.body or soup
    h1 = root.find("h1")
    title = clean(h1.get_text()) if h1 else (title_tag or path.stem)

    sections, intro = [], []
    heading, h2_heading, body = None, None, []
    for el in root.find_all(["h2", "h3", "h4", "p", "ul", "ol", "table", "pre", "blockquote"]):
        # Skip elements nested inside a block we already take whole (e.g. <p> in <li>).
        if el.find_parent(["ul", "ol", "table", "pre", "blockquote"]):
            continue
        if el.name in ("h2", "h3", "h4"):
            if heading is not None:
                sections.append((heading, "\n".join(body)))
            text = clean(el.get_text())
            if el.name == "h2":
                h2_heading = heading = text
            else:
                heading = f"{h2_heading} > {text}" if h2_heading else text
            body = []
            continue
        text = html_block_text(el)
        if not text:
            continue
        (body if heading is not None else intro).append(text)
    if heading is not None:
        sections.append((heading, "\n".join(body)))

    note = " ".join(intro)
    if len(note) > 250:  # a real introduction, not a version line
        sections.insert(0, ("Introduction", "\n".join(intro)))
        note = ""
    if not sections and intro:
        sections, note = [("Full document", "\n".join(intro))], ""
    return title, org, note, sections


def split_long(text):
    """Split an over-long section at paragraph (or sentence) boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    pieces = re.split(r"\n\s*\n|\n", text)
    if len(pieces) == 1:
        pieces = re.split(r"(?<=[.!?])\s+", text)
    parts, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += piece + "\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_document(path):
    title, org, note, sections = parse_html(path)
    doc_line = f"{org} - {title}" if org else title
    chunks = []
    for heading, text in sections:
        if not text.strip():
            continue
        parts = split_long(text)
        for n, part in enumerate(parts, 1):
            label = heading if len(parts) == 1 else f"{heading} (part {n} of {len(parts)})"
            header = [doc_line, f"Section: {label}"]
            if note:
                header.append(f"About this document: {note}")
            chunks.append({"text": "\n".join(header) + "\n\n" + part,
                           "source": path.name, "title": title, "section": label})
    return chunks


# ---------------------------------------------------------------------------
# Chroma index
# ---------------------------------------------------------------------------

def doc_files():
    return sorted(p for p in DOCS_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("."))


def fingerprint():
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in doc_files()}


def open_db():
    return chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))


def build_index():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Indexing {DOCS_DIR} ...")
    all_chunks = []
    for path in doc_files():
        try:
            chunks = chunk_document(path)
        except Exception as exc:  # one bad file shouldn't stop the rebuild
            log(f"  skipped {path.name}: {exc}")
            continue
        log(f"  {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    client = open_db()
    with contextlib.suppress(Exception):  # nothing to delete on the first build
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    if all_chunks:
        collection.add(
            ids=[f"{c['source']}::{i}" for i, c in enumerate(all_chunks)],
            documents=[c["text"] for c in all_chunks],
            metadatas=[{k: c[k] for k in ("source", "title", "section")} for c in all_chunks],
        )
    MANIFEST.write_text(json.dumps(fingerprint(), indent=2), encoding="utf-8")
    log(f"Stored {len(all_chunks)} chunks in {DB_DIR.relative_to(HERE)}/.")
    return collection


def get_collection():
    """Open the index, rebuilding it if it's missing or the documents changed."""
    if not MANIFEST.exists():
        return build_index()
    if json.loads(MANIFEST.read_text(encoding="utf-8")) != fingerprint():
        log("The HR documents changed since the index was built; rebuilding.")
        return build_index()
    try:
        return open_db().get_collection(COLLECTION)
    except Exception:
        return build_index()


def search(collection, query):
    # Chroma can print while loading its embedding model; keep stdout clean for MCP.
    with contextlib.redirect_stdout(sys.stderr):
        result = collection.query(query_texts=[query], n_results=TOP_K)
    hits = []
    for rank, (text, meta, dist) in enumerate(
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), 1):
        hits.append({"rank": rank, "source": meta["source"], "section": meta["section"],
                     "similarity": round(1 - dist, 3), "text": text})
    return hits


# ---------------------------------------------------------------------------
# Employee records
# ---------------------------------------------------------------------------

def normalize_email(value):
    """Accept 'name@x', ' NAME@X ', or 'Dana Whitfield <name@x>'."""
    value = value.strip()
    m = re.search(r"<([^<>]+)>", value)
    return (m.group(1) if m else value).strip().lower()


# Column names differ between organizations, so they're recognized by the words in
# the header (after lowercasing and turning punctuation into spaces).
LEAVE_WORDS = ("vacation", "pto", "paid time off", "sick", "personal", "leave", "holiday",
               "annual", "time off", "float", "floating", "bereavement", "comp")
AMOUNT_WORDS = ("day", "days", "hour", "hours", "hrs", "balance", "remaining", "left",
                "available", "accrued")
INFO_WORDS = ("department", "dept", "team", "division", "location", "hotel", "property",
              "site", "store", "branch", "office", "role", "title", "position", "job",
              "manager", "supervisor", "reports to")
NAME_EXCLUDE = {"manager", "supervisor", "user", "file", "hotel", "property", "team"}


def _norm(header):
    return " " + re.sub(r"[^a-z0-9]+", " ", header.lower()).strip() + " "


def _has_word(header, words):
    text = _norm(header)
    return any(f" {w} " in text for w in words)


def _is_number(value):
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", value.strip()))


def read_employees():
    """(header, rows) from the employee file. Re-read on every call."""
    with EMPLOYEES_CSV.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]
        return list(reader.fieldnames or []), rows


def column_map(header, rows):
    """Work out which columns hold the email, name, other job info, and leave balances."""
    def values(col):
        return [(r.get(col) or "").strip() for r in rows if (r.get(col) or "").strip()]

    email = next((h for h in header if "email" in _norm(h).replace(" ", "")
                  or _norm(h).strip() in ("mail", "e mail", "address")), None)
    if email is None:  # fall back to the column whose values all look like addresses
        email = next((h for h in header
                      if values(h) and all(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v)
                                           for v in values(h))), None)

    name = next((h for h in header if h != email and _has_word(h, ("name",))
                 and not set(_norm(h).split()) & NAME_EXCLUDE), None)

    balances = [h for h in header if h not in (email, name)
                and values(h) and all(_is_number(v) for v in values(h))
                and (_has_word(h, LEAVE_WORDS) or _has_word(h, AMOUNT_WORDS))]
    info = [h for h in header if h not in (email, name, *balances) and _has_word(h, INFO_WORDS)]
    ignored = [h for h in header if h not in (email, name, *balances, *info)]
    return {"email": email, "name": name, "info": info, "balances": balances, "ignored": ignored}


def checked_column_map():
    header, rows = read_employees()
    cols = column_map(header, rows)
    if not cols["email"] or not cols["balances"]:
        sys.exit(f"Couldn't work out the columns of {EMPLOYEES_CSV}: found email column "
                 f"{cols['email']!r} and balance columns {cols['balances']}. Header: {header}")
    return cols


def find_employee(email):
    """The employee row for this email, or None."""
    target = normalize_email(email)
    header, rows = read_employees()
    col = column_map(header, rows)["email"]
    for row in rows:
        if col and (row.get(col) or "").strip().lower() == target:
            return row
    return None


def balance_or_error(email):
    row = find_employee(email) if email.strip() else None
    if row is None:
        raise ToolError(
            f"No employee record matches '{email.strip()}'. Do not state, estimate, or guess "
            "any leave balance for this person. Say that their balance couldn't be found and "
            "that someone in HR will check their record."
        )
    header, rows = read_employees()
    cols = column_map(header, rows)

    def number(value):
        value = (value or "").strip()
        if not _is_number(value):
            return value or None
        return int(value) if re.fullmatch(r"-?\d+", value) else float(value)

    record = {"email": row[cols["email"]].strip()}
    if cols["name"]:
        record["name"] = row[cols["name"]].strip()
    for col in cols["info"]:  # e.g. department, hotel, role, manager/supervisor
        record[col] = (row.get(col) or "").strip() or None
    # Keep each balance under its own column name, so units (days, hours) stay clear.
    record["leave_balances"] = {col: number(row.get(col)) for col in cols["balances"]}
    record["source"] = EMPLOYEES_CSV.name
    return record


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def create_server(collection, cols, as_employee=None):
    org = org_name()
    mode = f"employee mode for {as_employee}" if as_employee else "HR mode"
    topics = ", ".join(p.stem.replace("-", " ") for p in doc_files())
    balances = ", ".join(cols["balances"])
    server = MCPServer(
        name=DATA_DIR.name,
        instructions=(
            f"{org} HR tools. Use search_policy for policy questions and cite the "
            "source file of each policy fact. Use get_leave_balance for leave balances. "
            "Never state a policy or a balance that didn't come from these tools. "
            f"This server is running in {mode}."
        ),
    )

    def search_policy(query: str) -> dict[str, Any]:
        if not query.strip():
            raise ToolError("The query is empty. Describe the policy topic to search for.")
        return {"query": query, "results": search(collection, query)}

    server.add_tool(search_policy, name="search_policy", annotations=READ_ONLY, description=(
        f"Search {org}'s HR policy documents ({topics}).\n\n"
        "Returns the 4 document sections most similar to the query, best first. Each "
        "result has the source file name, the section heading, a similarity score "
        "(0 to 1), and the section text. Cite the source file for each fact you use. "
        "If none of the sections answer the question, say the policies don't cover it."
    ))

    if as_employee:
        def get_leave_balance() -> dict[str, Any]:
            return balance_or_error(as_employee)

        description = (
            f"Get the leave balances ({balances}) of the employee this server is set up "
            "for, from the employee file. Takes no inputs. It can't look up anyone else; "
            "if asked about another employee's balance, say you can't share it."
        )
    else:
        def get_leave_balance(email: str) -> dict[str, Any]:
            return balance_or_error(email)

        description = (
            f"For HR staff: get an employee's leave balances ({balances}), plus their "
            "name and job details, by their work email. If no record matches, the tool "
            "returns an error; then don't state any balance."
        )
    server.add_tool(get_leave_balance, name="get_leave_balance",
                    description=description, annotations=READ_ONLY)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--org", metavar="FOLDER",
                        help="organization folder (a path, or a folder name inside week-04); "
                             "default: hartwell-hr")
    parser.add_argument("--columns", action="store_true",
                        help="show which employee-file columns were recognized, then exit")
    parser.add_argument("--as-employee", metavar="EMAIL",
                        help="employee mode: get_leave_balance takes no inputs and returns "
                             "only this employee's balance")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the Chroma index from the HR documents, then exit")
    parser.add_argument("--search", metavar="QUERY",
                        help="print the search results for QUERY, then exit (for testing)")
    args = parser.parse_args()
    configure(resolve_org(args.org))
    cols = checked_column_map()

    if args.columns:
        sys.stdout.reconfigure(encoding="utf-8")
        print(f"Organization: {org_name()} ({DATA_DIR})")
        print(f"Database:     {DB_DIR}")
        for key in ("email", "name", "info", "balances", "ignored"):
            print(f"{key:13} {cols[key]}")
        return

    if args.rebuild:
        with contextlib.redirect_stdout(sys.stderr):
            build_index()
        return

    # Load the index and warm up the embedding model before serving, so the
    # first tool call doesn't time out.
    with contextlib.redirect_stdout(sys.stderr):
        collection = get_collection()
        collection.query(query_texts=["warm-up"], n_results=1)

    if args.search:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(search(collection, args.search), indent=2, ensure_ascii=False))
        return

    as_employee = normalize_email(args.as_employee) if args.as_employee else None
    if as_employee and find_employee(as_employee) is None:
        log(f"Warning: no record in {EMPLOYEES_CSV.name} matches {as_employee}; "
            "get_leave_balance will return an error.")
    if cols["ignored"]:
        log(f"Employee-file columns not used: {cols['ignored']}")
    log(f"{DATA_DIR.name} MCP server ready "
        f"({'employee mode: ' + as_employee if as_employee else 'HR mode'}).")
    create_server(collection, cols, as_employee).run()  # stdio


if __name__ == "__main__":
    main()
