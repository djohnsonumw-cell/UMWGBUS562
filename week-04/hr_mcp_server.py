"""MCP server for Hartwell Foods HR: policy search and leave balances.

Tools
- search_policy(query): the 4 HR-document sections closest to the query, each
  with its source file, so the caller can cite it.
- get_leave_balance: vacation, sick, and personal days left, from employees.csv.
  * HR mode (default): takes an employee's work email and returns that record.
  * Employee mode (--as-employee ADDRESS): takes no inputs and returns only that
    one employee's balance, so a client can't look up anyone else.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py                       # HR mode (stdio)
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --rebuild             # rebuild the index, then exit
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --search "jury duty"  # test a search, then exit

The search reuses the approach from week-03/ask.py: each HTML document is split
into one chunk per section, each chunk starts with the document title and the
section heading, and the chunks are stored in a Chroma database
(week-04/chroma_db/, git-ignored) using Chroma's built-in embedding model. The
index is rebuilt automatically when the documents change, or on demand with
--rebuild.

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
DATA_DIR = HERE / "hartwell-hr"
DOCS_DIR = DATA_DIR / "hr-docs"
EMPLOYEES_CSV = DATA_DIR / "employees.csv"
DB_DIR = HERE / "chroma_db"  # git-ignored; rebuild with --rebuild
MANIFEST = DB_DIR / "manifest.json"
COLLECTION = "hr_docs"

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
    log(f"Indexing {DOCS_DIR.relative_to(HERE)} ...")
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


def find_employee(email):
    """The employees.csv row for this email, or None. Re-read on every call."""
    target = normalize_email(email)
    with EMPLOYEES_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["email"].strip().lower() == target:
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

    def days(field):
        return int(row[field]) if row[field].strip().isdigit() else row[field].strip()

    return {
        "email": row["email"].strip(),
        "name": row["name"].strip(),
        "department": row["department"].strip(),
        "manager": row["manager"].strip() or None,
        "vacation_days_left": days("vacation_days_left"),
        "sick_days_left": days("sick_days_left"),
        "personal_days_left": days("personal_days_left"),
        "source": EMPLOYEES_CSV.name,
    }


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def create_server(collection, as_employee=None):
    mode = f"employee mode for {as_employee}" if as_employee else "HR mode"
    server = MCPServer(
        name="hartwell-hr",
        instructions=(
            "Hartwell Foods HR tools. Use search_policy for policy questions and cite the "
            "source file of each policy fact. Use get_leave_balance for leave balances. "
            "Never state a policy or a balance that didn't come from these tools. "
            f"This server is running in {mode}."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def search_policy(query: str) -> dict[str, Any]:
        """Search Hartwell Foods' HR policy documents (leave, sick days, jury duty,
        bereavement, parental and family leave, benefits enrollment, remote work,
        reporting workplace concerns).

        Returns the 4 document sections most similar to the query, best first. Each
        result has the source file name, the section heading, a similarity score
        (0 to 1), and the section text. Cite the source file for each fact you use.
        If none of the sections answer the question, say the policies don't cover it.
        """
        if not query.strip():
            raise ToolError("The query is empty. Describe the policy topic to search for.")
        return {"query": query, "results": search(collection, query)}

    if as_employee:
        def get_leave_balance() -> dict[str, Any]:
            return balance_or_error(as_employee)

        description = (
            "Get the leave balance (vacation, sick, and personal days left) of the employee "
            "this server is set up for. Takes no inputs. It can't look up anyone else; if "
            "asked about another employee's balance, say you can't share it."
        )
    else:
        def get_leave_balance(email: str) -> dict[str, Any]:
            return balance_or_error(email)

        description = (
            "For HR staff: get an employee's leave balance (vacation, sick, and personal "
            "days left), plus their name, department, and manager, by their work email "
            "(e.g. dana.whitfield@hartwellfoods.example). If no record matches, the tool "
            "returns an error; then don't state any balance."
        )
    server.add_tool(get_leave_balance, name="get_leave_balance",
                    description=description, annotations=READ_ONLY)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--as-employee", metavar="EMAIL",
                        help="employee mode: get_leave_balance takes no inputs and returns "
                             "only this employee's balance")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the Chroma index from the HR documents, then exit")
    parser.add_argument("--search", metavar="QUERY",
                        help="print the search results for QUERY, then exit (for testing)")
    args = parser.parse_args()

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
    log(f"hartwell-hr MCP server ready ({'employee mode: ' + as_employee if as_employee else 'HR mode'}).")
    create_server(collection, as_employee).run()  # stdio


if __name__ == "__main__":
    main()
