"""Answer questions about a folder of documents with RAG.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-03\\ask.py "How much can I spend on dinner in Boston?"
    .venv\\Scripts\\python.exe week-03\\ask.py            # interactive: ask several questions
    .venv\\Scripts\\python.exe week-03\\ask.py --rebuild  # re-index after adding/removing documents
    .venv\\Scripts\\python.exe week-03\\ask.py --docs week-03\\larkfield-docs "What is the wire fee?"
    .venv\\Scripts\\python.exe week-03\\ask.py "..." --save week-03\\runs\\q1.txt

How it works:
1. Every document in the folder (HTML, PDF, Markdown, plain text) is split into
   one chunk per section. Each chunk starts with the document title and the
   section heading, so it makes sense on its own. The folder defaults to
   week-03/copperline-docs; choose another with --docs.
2. The chunks are stored in a Chroma database using Chroma's built-in embedding
   model. Each document folder gets its own database, in chroma_db/<folder name>/.
3. For a question, the 6 most similar chunks are retrieved and sent to Claude,
   which answers only from those sources and cites the source file for each fact.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import anthropic
import chromadb
import pymupdf
from bs4 import BeautifulSoup
from chromadb.config import Settings

HERE = Path(__file__).resolve().parent
DEFAULT_DOCS = HERE / "copperline-docs"
DB_ROOT = HERE / "chroma_db"  # one subfolder per document folder; git-ignored
COLLECTION = "documents"

# The API key lives in the folder that contains the repository, never in the repo.
KEY_FILES = [HERE.parents[1] / "API-Key.txt", HERE.parents[1] / "API Key.txt"]

MODEL = "claude-sonnet-5"
TOP_K = 6
PREVIEW_CHARS = 150

# Sections longer than this are split at paragraph boundaries into parts.
MAX_CHUNK_CHARS = 1500

SUPPORTED = {".html", ".htm", ".pdf", ".md", ".markdown", ".txt"}

# Some sections use a term that another section of the same document defines.
# For each (file, term), the sentences or list items that define the term are
# copied, labeled as context, into every other chunk that uses the term.
# Add entries here if new documents need the same treatment.
CONTEXT_TERMS = {
    "month-end-close-calendar.html": [
        {"term": r"business day", "defined_in": "How the close calendar works"},
    ],
    "travel-expense-policy.html": [
        {"term": r"Tier [12]", "defined_in": "Meal limits by city tier"},
    ],
}

SYSTEM_PROMPT = """You answer questions about an organization's policies and \
procedures. You are given numbered sources retrieved from its documents.

Rules:
- Answer only from the sources provided. Do not use outside knowledge or guess.
- After every fact, cite the source file it came from in square brackets, for \
example [travel-expense-policy.html].
- If the sources do not contain the answer, say clearly that the documents \
provided don't answer the question. If they answer only part of it, answer that \
part and say what is missing.
- Keep the answer short and direct, concise yet comprehensive"""


# ---------------------------------------------------------------------------
# Parsing: each parser returns (title, note, sections, extra_chunks), where
# sections is a list of (heading, text) and note is a short line about the
# document (version, owner) or "".
# ---------------------------------------------------------------------------

def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def split_title(raw):
    """'Vendor Payment Terms | Copperline Manufacturing' -> (title, org)."""
    parts = [p.strip() for p in raw.split("|")]
    return parts[0], (parts[1] if len(parts) > 1 else "")


def html_block_text(el):
    """Readable text for one block element, keeping list and table structure."""
    if el.name in ("ul", "ol"):
        return "\n".join("- " + clean(li.get_text(" ")) for li in el.find_all("li"))
    if el.name == "table":
        rows = []
        for tr in el.find_all("tr"):
            cells = [clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
            rows.append(" | ".join(cells))
        return "\n".join(rows)
    return clean(el.get_text(" "))


def parse_html(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title_tag, org = split_title(soup.title.get_text()) if soup.title else ("", "")

    footer = soup.find("footer")
    footer_text = clean(footer.get_text(" ")) if footer else ""
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    root = soup.find("main") or soup.body or soup
    h1 = root.find("h1")
    title = clean(h1.get_text()) if h1 else (title_tag or path.stem)

    sections, intro = [], []
    heading, h2_heading, body = None, None, []
    blocks = root.find_all(["h2", "h3", "h4", "p", "ul", "ol", "table", "pre", "blockquote"])
    for el in blocks:
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
        if heading is None:
            intro.append(text)
        else:
            body.append(text)
    if heading is not None:
        sections.append((heading, "\n".join(body)))

    note = " ".join(intro)
    if len(note) > 250:  # a real introduction, not a version line
        sections.insert(0, ("Introduction", "\n".join(intro)))
        note = ""
    if not sections and intro:
        sections = [("Full document", "\n".join(intro))]

    extra = [("Contact", footer_text)] if footer_text else []
    return title, org, note, sections, extra


def parse_pdf(path):
    """Find headings by font: bold or larger than the body text."""
    doc = pymupdf.open(path)
    lines = []  # (size, bold, text)
    for page in doc:
        height = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                y = line["bbox"][1]
                if y < height * 0.03 or y > height * 0.95:  # page header/footer band
                    continue
                text = "".join(s["text"] for s in spans).strip()
                size = max(s["size"] for s in spans)
                bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
                lines.append((round(size, 1), bold, text))
    meta_title = (doc.metadata or {}).get("title", "")
    doc.close()

    if not lines:
        raise ValueError("no extractable text (it may be a scanned image)")

    weights = Counter()
    for size, bold, text in lines:
        weights[size] += len(text)
    body_size = weights.most_common(1)[0][0]
    lines = [ln for ln in lines if ln[0] >= body_size * 0.8]  # drop fine print

    def is_heading(size, bold, text):
        return (bold or size > body_size * 1.15) and len(text) < 120 and not text.endswith(".")

    heading_sizes = Counter(s for s, b, t in lines if is_heading(s, b, t))
    section_size = min(
        (s for s, n in heading_sizes.items() if n == max(heading_sizes.values())),
        default=None,
    )

    title, org = split_title(meta_title) if meta_title else ("", "")
    big = [t for s, b, t in lines if section_size and is_heading(s, b, t) and s > section_size]

    sections, intro, heading, para = [], [], None, []

    def add_line(text):
        # Re-join lines the page wrapped mid-sentence.
        if para and para[-1].endswith("-") and text[:1].islower():
            para[-1] = para[-1][:-1] + text
        else:
            para.append(text)

    for size, bold, text in lines:
        if is_heading(size, bold, text) and section_size and size <= section_size:
            if heading is not None:
                sections.append((heading, " ".join(para)))
            heading, para = text, []
        elif is_heading(size, bold, text) and size > (section_size or 0):
            continue  # title / letterhead lines
        elif heading is None:
            intro.append(text)
        else:
            add_line(text)
    if heading is not None:
        sections.append((heading, " ".join(para)))

    if not title:
        title = big[-1] if big else path.stem
    if not org and len(big) > 1:
        org = big[0]
    # Drop letterhead lines from the intro (e.g. "Finance Department, Accounts Payable").
    note = " ".join(t for t in intro if t.endswith("."))
    if not sections:
        sections = [("Full document", " ".join(intro))]
        note = ""
    return title, org, note, sections, []


def parse_text(path):
    """Markdown or plain text: '#' headings if present, otherwise one section."""
    text = path.read_text(encoding="utf-8")
    title, sections, heading, body, intro = path.stem.replace("-", " ").title(), [], None, [], []
    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m and len(m.group(1)) == 1 and heading is None and not sections:
            title = m.group(2).strip()
        elif m:
            if heading is not None:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = m.group(2).strip(), []
        elif heading is None:
            intro.append(line)
        else:
            body.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(body).strip()))
    intro_text = "\n".join(intro).strip()
    if intro_text:
        sections.insert(0, ("Introduction" if sections else "Full document", intro_text))
    return title, "", "", sections, []


PARSERS = {".html": parse_html, ".htm": parse_html, ".pdf": parse_pdf,
           ".md": parse_text, ".markdown": parse_text, ".txt": parse_text}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_long(text):
    """Split an over-long section at paragraph (or line) boundaries."""
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


def context_notes(filename, sections):
    """For CONTEXT_TERMS: {term_regex: (defining_heading, definition_text)}."""
    notes = {}
    for rule in CONTEXT_TERMS.get(filename, []):
        source = dict(sections).get(rule["defined_in"])
        if not source:
            continue
        pieces = re.split(r"\n|(?<=[.!?])\s+", source)
        hits = [p.strip() for p in pieces if re.search(rule["term"], p, re.I)]
        if hits:
            notes[rule["term"]] = (rule["defined_in"], " ".join(hits))
    return notes


def chunk_document(path):
    title, org, note, sections, extra = PARSERS[path.suffix.lower()](path)
    sections = [(h, t) for h, t in sections if t.strip()]
    notes = context_notes(path.name, sections)
    doc_line = f"{org} - {title}" if org else title

    chunks = []
    for heading, text in sections:
        added = [
            f"[Context from the section \"{src}\" of this document: {definition}]"
            for term, (src, definition) in notes.items()
            if src != heading and re.search(term, text, re.I)
        ]
        parts = split_long(text)
        for n, part in enumerate(parts, 1):
            label = heading if len(parts) == 1 else f"{heading} (part {n} of {len(parts)})"
            lines = [doc_line, f"Section: {label}"]
            if note:
                lines.append(f"About this document: {note}")
            chunks.append({
                "text": "\n".join(lines) + "\n\n" + part + ("\n\n" + "\n".join(added) if added else ""),
                "source": path.name,
                "title": title,
                "section": label,
            })
    for heading, text in extra:
        chunks.append({
            "text": f"{doc_line}\nSection: {heading}\n\n{text}",
            "source": path.name, "title": title, "section": heading,
        })
    return chunks


def doc_files(docs_dir):
    return sorted(p for p in docs_dir.iterdir() if p.is_file() and not p.name.startswith("."))


def fingerprint(docs_dir):
    return {
        "docs_dir": str(docs_dir),
        "files": {p.name: [p.stat().st_size, int(p.stat().st_mtime)] for p in doc_files(docs_dir)},
    }


# ---------------------------------------------------------------------------
# Chroma database: one per document folder, in chroma_db/<folder name>/
# ---------------------------------------------------------------------------

def db_dir(docs_dir):
    return DB_ROOT / docs_dir.name


def open_db(docs_dir):
    return chromadb.PersistentClient(path=str(db_dir(docs_dir)),
                                     settings=Settings(anonymized_telemetry=False))


def build_index(docs_dir):
    print(f"Indexing documents in {docs_dir.name}/ ...")
    all_chunks, seen = [], set()
    for path in doc_files(docs_dir):
        if path.suffix.lower() not in SUPPORTED:
            print(f"  skipped {path.name}: unsupported file type {path.suffix or '(none)'}")
            continue
        try:
            chunks = chunk_document(path)
        except Exception as exc:  # one bad file shouldn't stop the rebuild
            print(f"  skipped {path.name}: {exc}")
            continue
        kept = []
        for c in chunks:
            # The same footer appears on every page; keep only its first copy.
            body = c["text"].split("\n\n", 1)[-1]
            if c["section"] == "Contact" and body in seen:
                continue
            seen.add(body)
            kept.append(c)
        print(f"  {path.name}: {len(kept)} chunks")
        all_chunks.extend(kept)

    client = open_db(docs_dir)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # nothing to delete on the first build
    collection = client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    if all_chunks:
        collection.add(
            ids=[f"{c['source']}::{i}" for i, c in enumerate(all_chunks)],
            documents=[c["text"] for c in all_chunks],
            metadatas=[{k: c[k] for k in ("source", "title", "section")} for c in all_chunks],
        )
    manifest = db_dir(docs_dir) / "manifest.json"
    manifest.write_text(json.dumps(fingerprint(docs_dir), indent=2), encoding="utf-8")
    print(f"Stored {len(all_chunks)} chunks in {db_dir(docs_dir).relative_to(HERE)}.\n")
    return collection


def get_collection(docs_dir):
    manifest = db_dir(docs_dir) / "manifest.json"
    if not manifest.exists():
        return build_index(docs_dir)
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved.get("docs_dir") != str(docs_dir):
        # Two different folders with the same name would share a database.
        print(f"Note: chroma_db/{db_dir(docs_dir).name} was built from a different "
              f"folder named {docs_dir.name}; rebuilding it.\n")
        return build_index(docs_dir)
    if saved != fingerprint(docs_dir):
        print("Note: the documents have changed since the index was built. "
              "Run with --rebuild to update it.\n")
    return open_db(docs_dir).get_collection(COLLECTION)


# ---------------------------------------------------------------------------
# Retrieval and answering
# ---------------------------------------------------------------------------

def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    for path in KEY_FILES:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    sys.exit("No API key found. Set ANTHROPIC_API_KEY or put the key in "
             f"{KEY_FILES[0].name} in {KEY_FILES[0].parent}.")


def retrieve(collection, question):
    result = collection.query(query_texts=[question], n_results=TOP_K)
    hits = []
    for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": text, "source": meta["source"], "section": meta["section"],
                     "similarity": 1 - dist})
    return hits


def format_hits(hits):
    lines = [f"Top {len(hits)} chunks:"]
    for rank, h in enumerate(hits, 1):
        lines.append(f"  {rank}. similarity {h['similarity']:.3f} | {h['source']}")
        lines.append(f"     {clean(h['text'])[:PREVIEW_CHARS]}...")
    return "\n".join(lines)


def format_saved(hits, answer_text):
    """The --save file: each retrieved chunk in full, then ANSWER, then the answer."""
    parts = [
        f"Rank {rank} | similarity {h['similarity']:.3f} | source {h['source']}\n{h['text']}"
        for rank, h in enumerate(hits, 1)
    ]
    return "\n\n".join(parts) + "\n\nANSWER\n" + answer_text + "\n"


def answer(client, question, hits):
    sources = "\n\n".join(
        f'<source id="{i}" file="{h["source"]}">\n{h["text"]}\n</source>'
        for i, h in enumerate(hits, 1)
    )
    message = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"}],
    )
    if message.stop_reason == "refusal":
        return "Claude declined to answer this question."
    return "".join(b.text for b in message.content if b.type == "text").strip()


def ask(collection, client, question, save_path=None):
    hits = retrieve(collection, question)
    print(format_hits(hits) + "\n")
    try:
        answer_text = answer(client, question, hits)
    except anthropic.AuthenticationError:
        answer_text = "Error: Claude rejected the API key. Check the key file."
    except anthropic.RateLimitError:
        answer_text = "Error: rate limited by the Claude API. Wait a moment and try again."
    except anthropic.APIStatusError as exc:
        answer_text = f"Error: Claude API error {exc.status_code}: {exc.message}"
    except anthropic.APIConnectionError:
        answer_text = "Error: could not reach the Claude API. Check your internet connection."
    print("Answer:\n" + answer_text + "\n")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(format_saved(hits, answer_text), encoding="utf-8")
        print(f"Saved this run to {save_path}")


def main():
    # Windows uses a legacy code page when output is piped; Claude's answers
    # can contain dashes and curly quotes, so always write UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question", nargs="*", help="your question (omit to ask interactively)")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS,
                        help="folder of documents to answer from (default: week-03/copperline-docs)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the database after adding or removing documents")
    parser.add_argument("--save", type=Path, metavar="FILE",
                        help="also write the retrieved chunks and the answer to FILE (UTF-8)")
    args = parser.parse_args()

    docs_dir = args.docs.resolve()
    if not docs_dir.is_dir():
        sys.exit(f"Document folder not found: {args.docs}")
    if args.save and not args.question:
        sys.exit("--save needs a question on the command line.")

    collection = build_index(docs_dir) if args.rebuild else get_collection(docs_dir)
    if args.rebuild and not args.question:
        return

    client = anthropic.Anthropic(api_key=load_api_key())
    if args.question:
        ask(collection, client, " ".join(args.question), args.save)
        return
    print(f"Ask a question about the documents in {docs_dir.name}/ (blank line to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print()
        ask(collection, client, question)


if __name__ == "__main__":
    main()
