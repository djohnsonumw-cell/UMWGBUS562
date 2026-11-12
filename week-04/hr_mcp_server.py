"""MCP server for Hartwell Foods HR: policy search and leave balance lookup.

Tools:
    search_policy(query)   the 4 HR-document chunks closest to the query, with source files
    get_leave_balance      vacation, sick, and personal days left from employees.csv

Modes (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py
        HR staff mode: get_leave_balance(email) looks up any employee by work email.
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
        Employee mode: get_leave_balance() takes no inputs and returns only that
        employee's balance, so the model can't ask for anyone else's.
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --rebuild
        Rebuild the Chroma database from hartwell-hr/hr-docs, then exit.

The server talks MCP over stdio, so stdout belongs to the protocol: every log
line goes to stderr. It never calls Claude and needs no API key.

The search reuses the approach from week-03/ask.py: one chunk per HTML section,
each chunk prefixed with the document title and section heading, stored in a
Chroma database with Chroma's built-in embedding model (cosine similarity).
"""

import argparse
import contextlib
import csv
import json
import re
import sys
from email.utils import parseaddr
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from chromadb.config import Settings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

HERE = Path(__file__).resolve().parent
DATA = HERE / "hartwell-hr"
DOCS_DIR = DATA / "hr-docs"
EMPLOYEES_CSV = DATA / "employees.csv"
DB_DIR = HERE / "chroma_db"  # git-ignored; rebuild with --rebuild
MANIFEST = DB_DIR / "manifest.json"
COLLECTION = "hr_docs"

TOP_K = 4
MAX_CHUNK_CHARS = 1500  # longer sections are split at paragraph boundaries


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Parsing and chunking (adapted from week-03/ask.py; the HR docs are all HTML)
# ---------------------------------------------------------------------------

def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def split_title(raw):
    """'Vacation and Time Off Policy | Hartwell Foods' -> (title, org)."""
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
    """Return (title, org, note, sections) where sections is [(heading, text)]."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title_tag, org = split_title(soup.title.get_text()) if soup.title else ("", "")
    # The footer is the same disclaimer on every page, so it isn't indexed.
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    root = soup.find("main") or soup.body or soup
    h1 = root.find("h1")
    title = clean(h1.get_text()) if h1 else (title_tag or path.stem)

    sections, intro = [], []
    heading, h2_heading, body = None, None, []
    for el in root.find_all(["h2", "h3", "h4", "p", "ul", "ol", "table", "pre", "blockquote"]):
        if el.find_parent(["ul", "ol", "table", "pre", "blockquote"]):
            continue  # already included with its parent block
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
        (intro if heading is None else body).append(text)
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
            lines = [doc_line, f"Section: {label}"]
            if note:
                lines.append(f"About this document: {note}")
            chunks.append({"text": "\n".join(lines) + "\n\n" + part,
                           "source": path.name, "title": title, "section": label})
    return chunks


# ---------------------------------------------------------------------------
# Chroma database in week-04/chroma_db
# ---------------------------------------------------------------------------

def doc_files():
    return sorted(p for p in DOCS_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in (".html", ".htm"))


def fingerprint():
    return {p.name: [p.stat().st_size, int(p.stat().st_mtime)] for p in doc_files()}


def open_db():
    return chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))


def build_index():
    log(f"Indexing {DOCS_DIR.relative_to(HERE)} ...")
    all_chunks = []
    for path in doc_files():
        chunks = chunk_document(path)
        log(f"  {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    client = open_db()
    with contextlib.suppress(Exception):  # nothing to delete on the first build
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    # Keep anything Chroma prints (e.g. the first embedding-model download) off stdout.
    with contextlib.redirect_stdout(sys.stderr):
        collection.add(
            ids=[f"{c['source']}::{i}" for i, c in enumerate(all_chunks)],
            documents=[c["text"] for c in all_chunks],
            metadatas=[{k: c[k] for k in ("source", "title", "section")} for c in all_chunks],
        )
    MANIFEST.write_text(json.dumps(fingerprint(), indent=2), encoding="utf-8")
    log(f"Stored {len(all_chunks)} chunks in {DB_DIR.relative_to(HERE)}/.")
    return collection


def get_collection():
    """Open the index, building it if it's missing or the documents changed."""
    if MANIFEST.exists():
        try:
            if json.loads(MANIFEST.read_text(encoding="utf-8")) == fingerprint():
                return open_db().get_collection(COLLECTION)
            log("The HR documents changed since the last build; rebuilding the index.")
        except Exception as exc:
            log(f"Could not open the index ({exc}); rebuilding it.")
    return build_index()


# ---------------------------------------------------------------------------
# Leave balances from employees.csv
# ---------------------------------------------------------------------------

def normalize_email(value):
    """Accept 'name@x' or 'Name <name@x>'; compare without case."""
    return parseaddr(value or "")[1].strip().lower()


def find_employee(email):
    """The employee's row from employees.csv (read fresh each call), or None."""
    wanted = normalize_email(email)
    if not wanted:
        return None
    with EMPLOYEES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if normalize_email(row["email"]) == wanted:
                return row
    return None


def balance_result(email):
    row = find_employee(email)
    if row is None:
        raise ToolError(
            f"No employee record matches {email!r}. No leave balance is available for "
            "this person: do not state, estimate, or guess any balance."
        )
    return {
        "email": row["email"],
        "name": row["name"],
        "vacation_days_left": int(row["vacation_days_left"]),
        "sick_days_left": int(row["sick_days_left"]),
        "personal_days_left": int(row["personal_days_left"]),
    }


# ---------------------------------------------------------------------------
# The MCP server
# ---------------------------------------------------------------------------

def make_server(as_employee=None):
    collection = get_collection()
    server = MCPServer(
        name="hartwell-hr",
        log_level="WARNING",  # keep routine INFO logs out of the client's terminal
        instructions="Hartwell Foods HR tools: search the HR policy documents and look up "
                     "employee leave balances. Cite the source file for each policy fact.",
    )

    @server.tool()
    def search_policy(query: str) -> dict:
        """Search Hartwell Foods' HR policy documents (leave and vacation, parental and
        family leave, benefits enrollment, remote work, reporting workplace concerns).

        Returns the 4 document sections closest in meaning to the query, best first, each
        with its source file, section heading, similarity score (0-1), and full text.
        Write the query as a plain question or topic, for example "carry over unused
        vacation". Base policy statements only on the returned text, cite the source
        file for each fact, and treat low-similarity results as possibly irrelevant.
        """
        if not query.strip():
            raise ToolError("The search query is empty. Pass a question or topic to search for.")
        result = collection.query(query_texts=[query], n_results=TOP_K)
        results = [
            {"rank": rank, "source": meta["source"], "section": meta["section"],
             "similarity": round(1 - dist, 3), "text": text}
            for rank, (text, meta, dist) in enumerate(
                zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), 1)
        ]
        return {"query": query, "results": results}

    if as_employee:
        @server.tool()
        def get_leave_balance() -> dict:
            """Get the leave balance of the employee this session belongs to: vacation,
            sick, and personal days left. Takes no inputs. It can't look up any other
            employee; if someone asks about a coworker's balance, say it can't be shared.
            """
            return balance_result(as_employee)
    else:
        @server.tool()
        def get_leave_balance(email: str) -> dict:
            """Look up an employee's leave balance by work email address (for example
            dana.whitfield@hartwellfoods.example): vacation, sick, and personal days left.

            For HR staff use. When replying to an employee, look up only the sender's own
            email address, never a coworker's. If no record matches, the tool returns an
            error: then don't state any balance.
            """
            return balance_result(email)

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--as-employee", metavar="EMAIL",
                        help="employee mode: get_leave_balance returns only this employee's balance")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the Chroma database from hartwell-hr/hr-docs, then exit")
    args = parser.parse_args()

    if args.rebuild:
        build_index()
        return
    if args.as_employee:
        if find_employee(args.as_employee) is None:
            log(f"Warning: no employee record matches {args.as_employee!r}; "
                "get_leave_balance will return an error.")
        log(f"Employee mode: leave balances limited to {args.as_employee}.")
    else:
        log("HR staff mode: get_leave_balance takes an employee's work email.")
    make_server(args.as_employee).run()  # stdio transport


if __name__ == "__main__":
    main()
