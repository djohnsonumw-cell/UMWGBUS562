"""Hartwell Foods HR agent: drafts replies to employee emails using Claude tool use.

Three ways to run it:

  --email PATH   Handle one .eml file. The draft is written to week-04/drafts/.
  --inbox        Sign in to the Gmail sandbox with IMAP (same login as load_inbox.py)
                 and handle each unread inbox email, one at a time, as its own agent
                 run. Each email is marked read before work starts, so none is handled
                 twice. The reply is created as a real Gmail draft in the same thread.
                 Drafts that need review get the Gmail label "Needs HR Review" (on the
                 draft and the original email); the subject is left as "Re: ..." so the
                 thread isn't split. Nothing is sent.
  --watch        Check the inbox every 30 seconds and handle new unread email as in
                 --inbox, with these rules enforced here in code (not by Claude):
                   * Only senders in Agent-Allowlist.txt (beside Agent-Email.txt) are
                     handled. Each line is "your.address@gmail.com = employee@...".
                     Mail from anyone else is left unread and never shown to Claude.
                   * An allowed sender is treated as the employee on their line.
                   * The sender must pass Gmail's DMARC check (or SPF and DKIM), so a
                     forged From: line can't trigger a reply.
                   * Automatic mail (Auto-Submitted other than "no", or Precedence
                     bulk/auto_reply) is left unread.
                   * After save_reply, a reply with needs_human_review false is sent
                     with Gmail SMTP; otherwise it's kept as a labeled draft. Claude has
                     no send tool.
                   * At most 5 replies are sent, then it stops. Ctrl+C also stops it.
                   * Your real addresses and the password are replaced with <hidden>
                     in everything printed or saved with --save.

Add --save FILE to also write everything printed to FILE (UTF-8).

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\agent.py --email week-04\\hartwell-hr\\sample-emails\\03-megan-time-off.eml
    .venv\\Scripts\\python.exe week-04\\agent.py --inbox --save week-04\\runs\\inbox-run.txt
    .venv\\Scripts\\python.exe week-04\\agent.py --watch --save week-04\\runs\\watch-run.txt

The API key comes from the ANTHROPIC_API_KEY environment variable, or else from
"API Key.txt" in the folder above the repository. The Gmail login comes from
Agent-Email.txt in that same folder. Neither the key, the password, nor the
sandbox address is ever printed or saved.
"""

import argparse
import asyncio
import imaplib
import json
import os
import re
import smtplib
import sys
import time
from datetime import datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import format_datetime, make_msgid, parseaddr
from pathlib import Path

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from load_inbox import IMAP_HOST, LOGIN_FILE, read_login

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SERVER_SCRIPT = HERE / "hr_mcp_server.py"
SYSTEM_PROMPT_FILE = HERE / "hartwell-hr" / "system_prompt.txt"
DRAFTS_DIR = HERE / "drafts"
SERVER_LOG = HERE / "mcp_server.log"  # *.log is git-ignored
KEY_FILES = [REPO.parent / "API Key.txt", REPO.parent / "API-Key.txt"]
ALLOWLIST_FILE = LOGIN_FILE.parent / "Agent-Allowlist.txt"

MODEL = "claude-sonnet-5"
MAX_TOOL_CALLS = 10
MAX_TOKENS = 16000
PREVIEW_CHARS = 300
REVIEW_LABEL = "Needs HR Review"  # Gmail label for drafts that need a person in HR
SMTP_HOST = "smtp.gmail.com"
POLL_SECONDS = 30
MAX_SENT = 5
AUTO_PRECEDENCE = {"bulk", "auto_reply"}
RULE = "=" * 70

SAVE_REPLY_TOOL = {
    "name": "save_reply",
    "description": (
        "Save the finished reply to the employee as a draft for the HR team to review. "
        "This only saves the draft; it never sends anything. Call it exactly once, "
        "when the reply is complete."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply_text": {
                "type": "string",
                "description": "The full text of the reply email to the employee.",
            },
            "needs_human_review": {
                "type": "boolean",
                "description": (
                    "True if a person in HR must handle or check this before anything "
                    "goes out (harassment, discrimination, safety, medical, or legal "
                    "matters, or questions the policies don't answer)."
                ),
            },
        },
        "required": ["reply_text", "needs_human_review"],
        "additionalProperties": False,
    },
    "strict": True,
}

# In --watch mode, tell Claude the truth: an unflagged reply goes out without review.
WATCH_SAVE_REPLY_TOOL = {
    **SAVE_REPLY_TOOL,
    "description": (
        "Save the finished reply to the employee. Call it exactly once, when the reply "
        "is complete. If needs_human_review is false, the reply may be emailed to the "
        "employee automatically, without a person reading it first. If it is true, it "
        "is kept as a draft for HR. If you're unsure, set needs_human_review to true."
    ),
}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

class Tee:
    """Write to the terminal and to a log file at the same time (for --save)."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):  # encoding, isatty, ...
        return getattr(self.stream, name)


class Redactor:
    """Replace secrets (real addresses, password) with <hidden> in everything written."""

    def __init__(self, stream, secrets):
        self.stream = stream
        values = sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True)
        self.patterns = [re.compile(re.escape(v), re.IGNORECASE) for v in values]

    def write(self, text):
        for pattern in self.patterns:
            text = pattern.sub("<hidden>", text)
        self.stream.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def preview(text, limit=PREVIEW_CHARS):
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "..."


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_api_key():
    """The key from ANTHROPIC_API_KEY, or from the key file outside the repo."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    for path in KEY_FILES:
        if path.is_file():
            key = path.read_text(encoding="utf-8-sig").strip()
            if key:
                return key
    sys.exit("No API key found. Set ANTHROPIC_API_KEY or put the key in "
             f"'{KEY_FILES[0].name}' in {REPO.parent}.")


def read_allowlist(path):
    """{real address: employee address} from 'real = employee' lines. Never echoes values."""
    if not path.is_file():
        sys.exit(f"Allowlist not found: {path.name} should be beside {LOGIN_FILE.name}.")
    allow = {}
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("=") != 1:
            sys.exit(f"{path.name} line {n}: expected 'your.address@gmail.com = employee address'.")
        real, employee = (parseaddr(part.strip())[1].strip().lower() for part in line.split("="))
        if "@" not in real or "@" not in employee:
            sys.exit(f"{path.name} line {n}: both sides need an email address.")
        allow[real] = employee
    if not allow:
        sys.exit(f"{path.name} has no entries.")
    return allow


def parse_email(raw, label):
    """Turn raw message bytes into the fields the agent needs."""
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    _, address = parseaddr(str(msg.get("From", "")))
    part = msg.get_body(preferencelist=("plain",))
    body = part.get_content() if part is not None else ""
    return {
        "label": label,
        "sender": address.strip().lower(),
        "subject": str(msg.get("Subject", "")).strip(),
        "body": body.strip(),
        "message_id": str(msg.get("Message-ID", "")).strip(),
        "references": str(msg.get("References", "")).strip(),
    }


def is_automatic(headers):
    """Auto-Submitted (anything but 'no', per RFC 3834) or Precedence bulk/auto_reply."""
    auto = str(headers.get("Auto-Submitted", "")).strip().lower()
    if auto and auto != "no":
        return True
    return str(headers.get("Precedence", "")).strip().lower() in AUTO_PRECEDENCE


def passes_sender_check(headers):
    """Gmail's own (topmost) Authentication-Results must show DMARC pass, or SPF+DKIM pass."""
    results = headers.get_all("Authentication-Results") or []
    if not results:
        return False
    top = str(results[0]).lower()
    if "mx.google.com" not in top:
        return False
    return "dmarc=pass" in top or ("spf=pass" in top and "dkim=pass" in top)


def reply_subject(subject):
    """'Re: <original subject>'. Kept unchanged so Gmail keeps the reply in the thread."""
    subject = subject or "(no subject)"
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def check_reply_args(args):
    reply_text = args.get("reply_text")
    needs_review = args.get("needs_human_review")
    if not isinstance(reply_text, str) or not reply_text.strip():
        raise ValueError("reply_text must be a non-empty string.")
    if not isinstance(needs_review, bool):
        raise ValueError("needs_human_review must be true or false.")
    return reply_text.strip(), needs_review


# ---------------------------------------------------------------------------
# save_reply back ends. Each returns (message for Claude, draft text to print).
# ---------------------------------------------------------------------------

def file_saver(email):
    """--email mode: write the draft to week-04/drafts/."""
    def save(args):
        reply_text, needs_review = check_reply_args(args)
        DRAFTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        local = re.sub(r"[^a-z0-9.-]", "_", email["sender"].split("@")[0])
        path = DRAFTS_DIR / f"{stamp}-{local}.txt"
        draft = (
            f"To: {email['sender']}\n"
            f"Subject: {reply_subject(email['subject'])}\n"
            f"Needs human review: {'YES' if needs_review else 'no'}\n"
            f"In reply to: {email['label']}\n"
            f"Status: DRAFT - not sent\n"
            f"\n{reply_text}\n"
        )
        path.write_text(draft, encoding="utf-8")
        return f"Draft saved (not sent) to {path.relative_to(REPO)}.", draft
    return save


def pending_saver(email, pending):
    """--watch mode: hold the reply; the code decides afterwards to send it or draft it."""
    def save(args):
        reply_text, needs_review = check_reply_args(args)
        pending.update(text=reply_text, needs_human_review=needs_review)
        outcome = "keep it as a Gmail draft for HR" if needs_review else "send it"
        draft = (
            f"To: the allowed sender who is treated as {email['sender']}\n"
            f"Subject: {reply_subject(email['subject'])}\n"
            f"Needs human review: {'YES' if needs_review else 'no'}\n"
            f"In reply to: {email['label']}\n"
            f"Next: the program will {outcome}\n"
            f"\n{reply_text}\n"
        )
        return "Reply saved.", draft
    return save


class Gmail:
    """IMAP for reading the inbox and creating drafts; SMTP only for --watch sends."""

    def __init__(self, keep_password=False):
        address, password = read_login(LOGIN_FILE)
        self.address = address  # used only in message headers; never printed
        self._password = password if keep_password else None
        self.imap = None
        self._connect(password)
        password = None

    # -- connection --------------------------------------------------------

    def _connect(self, password):
        try:
            self.imap = imaplib.IMAP4_SSL(IMAP_HOST, 993)
        except OSError as exc:
            sys.exit(f"Couldn't connect to {IMAP_HOST}: {exc.strerror or type(exc).__name__}")
        try:
            self.imap.login(self.address, password)
        except imaplib.IMAP4.error:
            sys.exit(f"Gmail sign-in failed. Check {LOGIN_FILE.name} and that IMAP is on.")
        self.drafts = self._find_drafts_mailbox()
        self._select_inbox()

    def reconnect(self):
        if self._password is None:
            raise RuntimeError("can't reconnect without the login")
        self.close()
        self._connect(self._password)

    def secrets(self):
        """Values the Redactor must hide."""
        values = [self.address]
        if self._password:
            values += [self._password, self._password.replace(" ", "")]
        return values

    def _find_drafts_mailbox(self):
        """The mailbox flagged \\Drafts (its name depends on the Gmail language)."""
        status, lines = self.imap.list()
        if status == "OK":
            for line in lines or []:
                text = line.decode("utf-8", "replace") if isinstance(line, bytes) else str(line)
                m = re.match(r'\((?P<flags>[^)]*)\)\s+(?:"[^"]*"|NIL)\s+(?P<name>.+)$', text)
                if m and "\\Drafts" in m.group("flags"):
                    return m.group("name").strip()
        return '"[Gmail]/Drafts"'

    def _select_inbox(self):
        status, _ = self.imap.select("INBOX")
        if status != "OK":
            sys.exit("Couldn't open INBOX.")

    # -- reading -----------------------------------------------------------

    def unread_uids(self):
        self.imap.noop()  # pick up mail that arrived since the last check
        status, data = self.imap.uid("search", None, "UNSEEN")
        if status != "OK":
            raise RuntimeError("couldn't search the inbox for unread messages")
        return data[0].split() if data and data[0] else []

    def peek_headers(self, uid):
        """Headers only, without marking the message read."""
        status, data = self.imap.uid("fetch", uid, "(BODY.PEEK[HEADER])")
        if status != "OK":
            raise RuntimeError(f"couldn't read the headers of message UID {uid.decode()}")
        raw = next((part[1] for part in data or [] if isinstance(part, tuple)), b"")
        return BytesParser(policy=policy.default).parsebytes(raw, headersonly=True)

    def fetch_and_mark_read(self, uid):
        """Fetch one message without marking it, then mark it read before any work."""
        status, data = self.imap.uid("fetch", uid, "(X-GM-THRID BODY.PEEK[])")
        if status != "OK":
            raise RuntimeError(f"couldn't fetch message UID {uid.decode()}")
        raw, thread = None, None
        for part in data or []:
            if isinstance(part, tuple):
                m = re.search(rb"X-GM-THRID (\d+)", part[0])
                thread = m.group(1) if m else thread
                raw = part[1]
        if raw is None:
            raise RuntimeError(f"message UID {uid.decode()} came back empty")
        status, _ = self.imap.uid("store", uid, "+FLAGS", "(\\Seen)")
        if status != "OK":
            raise RuntimeError(f"couldn't mark message UID {uid.decode()} as read")
        return raw, thread

    # -- writing -----------------------------------------------------------

    def make_reply(self, email, reply_text, to_address):
        msg = EmailMessage()
        msg["From"] = self.address
        msg["To"] = to_address
        msg["Subject"] = reply_subject(email["subject"])
        msg["Date"] = format_datetime(datetime.now().astimezone())
        msg["Message-ID"] = make_msgid(domain=self.address.split("@")[-1])
        if email["message_id"]:
            msg["In-Reply-To"] = email["message_id"]
            msg["References"] = f"{email['references']} {email['message_id']}".strip()
        msg.set_content(reply_text)
        return msg

    def save_draft(self, msg, original_thread, original_uid, needs_review):
        """Put msg in Drafts. Returns (same thread: True/False/None, label note)."""
        data = msg.as_bytes(policy=policy.SMTP)  # CRLF line endings
        status, resp = self.imap.append(self.drafts, "(\\Draft)",
                                        imaplib.Time2Internaldate(datetime.now().astimezone()),
                                        data)
        if status != "OK":
            raise RuntimeError("Gmail didn't accept the draft")
        m = re.search(rb"APPENDUID \d+ (\d+)", b" ".join(
            r if isinstance(r, bytes) else str(r).encode() for r in resp or []))
        draft_uid = m.group(1) if m else None
        return self._check_and_label(draft_uid, original_thread, original_uid, needs_review)

    def send(self, msg):
        """--watch only: send msg with Gmail SMTP. Raises on failure."""
        if self._password is None:
            raise RuntimeError("sending isn't enabled in this mode")
        with smtplib.SMTP_SSL(SMTP_HOST, 465, timeout=60) as smtp:
            smtp.login(self.address, self._password)
            smtp.send_message(msg)

    def _add_label(self, uid):
        status, _ = self.imap.uid("store", uid, "+X-GM-LABELS", f'("{REVIEW_LABEL}")')
        return status == "OK"

    def _check_and_label(self, draft_uid, original_thread, original_uid, needs_review):
        threaded, draft_labeled = None, False
        if draft_uid:
            try:
                self.imap.select(self.drafts)
                status, data = self.imap.uid("fetch", draft_uid, "(X-GM-THRID)")
                found = re.search(rb"X-GM-THRID (\d+)", b" ".join(
                    p if isinstance(p, bytes) else b"" for p in (data or [])))
                if status == "OK" and found and original_thread:
                    threaded = found.group(1) == original_thread
                if needs_review:
                    draft_labeled = self._add_label(draft_uid)
            finally:
                self._select_inbox()
        if not needs_review:
            return threaded, "none"
        original_labeled = self._add_label(original_uid)
        done = [name for name, ok in (("draft", draft_labeled), ("original email", original_labeled)) if ok]
        if not done:
            return threaded, f"FAILED to add '{REVIEW_LABEL}' - check it by hand"
        missing = "" if len(done) == 2 else " (couldn't label the other one)"
        return threaded, f"'{REVIEW_LABEL}' added to the {' and '.join(done)}{missing}"

    def draft_saver(self, email, thread, original_uid):
        """--inbox mode: save_reply creates a Gmail draft in the same thread."""
        def save(args):
            reply_text, needs_review = check_reply_args(args)
            msg = self.make_reply(email, reply_text, email["sender"])  # From address, never Reply-To
            threaded, label_note = self.save_draft(msg, thread, original_uid, needs_review)
            draft = (
                f"To: {email['sender']}\n"
                f"Subject: {msg['Subject']}\n"
                f"Needs human review: {'YES' if needs_review else 'no'}\n"
                f"Gmail label: {label_note}\n"
                f"In reply to: {email['label']}\n"
                f"Same thread as the original: {thread_note(threaded)}\n"
                f"Status: Gmail draft - not sent\n"
                f"\n{reply_text}\n"
            )
            return "Draft created in Gmail (not sent).", draft
        return save

    def close(self):
        try:
            self.imap.logout()
        except Exception:
            pass


def thread_note(threaded):
    return {True: "yes", False: "NO (Gmail started a new thread)", None: "couldn't check"}[threaded]


# ---------------------------------------------------------------------------
# One agent run
# ---------------------------------------------------------------------------

async def call_mcp_tool(session, name, args):
    """Run one MCP tool. Returns (text, is_error)."""
    result = await session.call_tool(name, args)
    parts = []
    for block in getattr(result, "content", None) or []:
        parts.append(block.text if getattr(block, "type", "") == "text" else str(block))
    text = "\n".join(parts)
    if not text and getattr(result, "structured_content", None) is not None:
        text = json.dumps(result.structured_content)
    return text or "(empty result)", bool(getattr(result, "is_error", False))


async def run_agent(client, email, save, save_tool=SAVE_REPLY_TOOL):
    """Handle one email. Returns (tool_calls, tokens_in, tokens_out)."""
    sender = email["sender"]
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()  # read every run

    print(f"Email:  {email['label']}")
    print(f"Sender: {sender}")
    print(f"Subject: {email['subject']}")
    if "@" not in sender:
        print("Stopped: couldn't find a sender address in the From: line.")
        return 0, 0, 0
    print(f"Starting {SERVER_SCRIPT.name} --as-employee {sender} ...")

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT), "--as-employee", sender],
        cwd=str(REPO),
    )
    tool_calls = tokens_in = tokens_out = 0
    draft = None
    stop = None

    with open(SERVER_LOG, "a", encoding="utf-8") as errlog:
        async with stdio_client(server, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = (await session.list_tools()).tools
                mcp_names = {t.name for t in mcp_tools}
                tools = [{"name": t.name, "description": t.description or "",
                          "input_schema": t.input_schema} for t in mcp_tools]
                tools.append(save_tool)
                print(f"Tools:  {', '.join(t['name'] for t in tools)}\n")

                messages = [{
                    "role": "user",
                    "content": (
                        "Prepare a reply to this employee email.\n\n"
                        "<email>\n"
                        f"From: {sender}\n"
                        f"Subject: {email['subject']}\n\n"
                        f"{email['body']}\n"
                        "</email>"
                    ),
                }]

                while stop is None:
                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        system=system_prompt,
                        tools=tools,
                        messages=messages,
                    )
                    tokens_in += response.usage.input_tokens
                    tokens_out += response.usage.output_tokens
                    messages.append({"role": "assistant", "content": response.content})

                    if response.stop_reason != "tool_use":
                        reasons = {
                            "end_turn": "Claude finished and asked for no more tools.",
                            "max_tokens": f"Claude's reply hit the {MAX_TOKENS}-token limit.",
                            "refusal": "Claude declined the request (refusal).",
                            "stop_sequence": "Claude's reply hit a stop sequence.",
                        }
                        stop = reasons.get(
                            response.stop_reason,
                            f"Claude's reply wasn't a tool request (stop reason: {response.stop_reason}).")
                        final_text = "".join(b.text for b in response.content if b.type == "text").strip()
                        if final_text:
                            print(f"Claude: {preview(final_text)}\n")
                        break

                    results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        if tool_calls >= MAX_TOOL_CALLS:
                            stop = (f"Reached the limit of {MAX_TOOL_CALLS} tool calls "
                                    "before Claude finished.")
                            break
                        tool_calls += 1
                        print(f"Step {tool_calls}: {block.name}")
                        print(f"  inputs: {json.dumps(block.input, ensure_ascii=False)}")

                        try:
                            if block.name == "save_reply":
                                if draft is not None:
                                    raise ValueError("A reply was already saved; save_reply runs only once.")
                                text, draft = save(block.input)
                                is_error = False
                            elif block.name in mcp_names:
                                text, is_error = await call_mcp_tool(session, block.name, block.input)
                            else:
                                text, is_error = f"Unknown tool: {block.name}", True
                        except Exception as exc:
                            text, is_error = f"Error: {exc}", True

                        print(f"  result{' (error)' if is_error else ''}: {preview(text)}\n")
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": text, "is_error": is_error})

                    if stop is None:
                        messages.append({"role": "user", "content": results})

    print(RULE)
    print(f"Stopped: {stop}")
    print(f"Tool calls: {tool_calls}")
    print(RULE)
    print(draft if draft else "No reply was saved.")
    print(RULE)
    print(f"Tokens for this email: {tokens_in:,} input, {tokens_out:,} output")
    return tool_calls, tokens_in, tokens_out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def email_mode(client, path):
    raw = Path(path).read_bytes()
    email = parse_email(raw, str(path))
    asyncio.run(run_agent(client, email, file_saver(email)))


def inbox_mode(client):
    print(f"Connecting to {IMAP_HOST} ...")
    gmail = Gmail()
    try:
        print("Signed in.")
        uids = gmail.unread_uids()
        print(f"Unread emails in the inbox: {len(uids)}\n")
        total_in = total_out = handled = 0
        for n, uid in enumerate(uids, 1):
            print(f"{'#' * 70}\nEmail {n} of {len(uids)}\n{'#' * 70}")
            try:
                raw, thread = gmail.fetch_and_mark_read(uid)
            except Exception as exc:
                print(f"Skipped: {exc}\n")
                continue
            print("(Marked as read in Gmail before starting.)")
            email = parse_email(raw, f"inbox message UID {uid.decode()}")
            try:
                _, t_in, t_out = asyncio.run(run_agent(client, email, gmail.draft_saver(email, thread, uid)))
            except anthropic.AuthenticationError:
                sys.exit("The API key was rejected (authentication error).")
            except (anthropic.APIError, OSError, RuntimeError) as exc:
                print(f"Stopped: this email failed ({type(exc).__name__}: {exc}). It's already "
                      "marked read; mark it unread in Gmail to try it again.\n")
                continue
            handled += 1
            total_in += t_in
            total_out += t_out
            print()
        print(RULE)
        print(f"Handled {handled} of {len(uids)} unread emails. Nothing was sent.")
        print(f"Total tokens: {total_in:,} input, {total_out:,} output")
    finally:
        gmail.close()


def watch_mode(client):
    allow = read_allowlist(ALLOWLIST_FILE)
    gmail = Gmail(keep_password=True)
    # From here on, every printed or saved line has the real addresses and password hidden.
    hidden = list(allow) + gmail.secrets()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Redactor(real_out, hidden), Redactor(real_err, hidden)

    sent = drafted = 0
    total_in = total_out = 0
    passed_over = set()  # UIDs left unread on purpose; reported once
    try:
        print("Signed in to Gmail.")
        print(f"Allowlist: {len(allow)} sender(s), treated as: "
              f"{', '.join(sorted(set(allow.values())))}")
        print(f"Watching the inbox every {POLL_SECONDS} seconds. Stops after {MAX_SENT} "
              "replies are sent, or press Ctrl+C.\n")
        while sent < MAX_SENT:
            try:
                uids = gmail.unread_uids()
            except (imaplib.IMAP4.abort, imaplib.IMAP4.error, OSError, RuntimeError):
                print(f"[{datetime.now():%H:%M:%S}] Lost the Gmail connection; reconnecting.")
                gmail.reconnect()
                continue

            for uid in uids:
                if uid in passed_over:
                    continue
                stamp = f"[{datetime.now():%H:%M:%S}] UID {uid.decode()}"
                try:
                    headers = gmail.peek_headers(uid)
                except (imaplib.IMAP4.abort, OSError):
                    print(f"{stamp}: lost the Gmail connection; reconnecting.")
                    gmail.reconnect()
                    break
                except (imaplib.IMAP4.error, RuntimeError):
                    print(f"{stamp}: left unread - couldn't read its headers.")
                    passed_over.add(uid)
                    continue
                real_sender = parseaddr(str(headers.get("From", "")))[1].strip().lower()
                if real_sender not in allow:
                    print(f"{stamp}: left unread - sender isn't on the allowlist.")
                    passed_over.add(uid)
                    continue
                if is_automatic(headers):
                    print(f"{stamp}: left unread - automatic mail.")
                    passed_over.add(uid)
                    continue
                if not passes_sender_check(headers):
                    print(f"{stamp}: left unread - failed Gmail's sender check (possible forgery).")
                    passed_over.add(uid)
                    continue

                employee = allow[real_sender]
                print(f"{'#' * 70}\n{stamp}: allowed sender, treated as {employee}\n{'#' * 70}")
                try:
                    raw, thread = gmail.fetch_and_mark_read(uid)
                except (imaplib.IMAP4.abort, OSError):
                    print("Lost the Gmail connection; reconnecting.")
                    gmail.reconnect()
                    break
                except (imaplib.IMAP4.error, RuntimeError):
                    print("Skipped: couldn't fetch it.")
                    passed_over.add(uid)
                    continue
                print("(Marked as read in Gmail before starting.)")
                email = parse_email(raw, f"inbox message UID {uid.decode()}")
                email["sender"] = employee  # the agent only ever sees the employee identity

                pending = {}
                try:
                    _, t_in, t_out = asyncio.run(run_agent(
                        client, email, pending_saver(email, pending), WATCH_SAVE_REPLY_TOOL))
                except anthropic.AuthenticationError:
                    sys.exit("The API key was rejected (authentication error).")
                except (anthropic.APIError, OSError, RuntimeError) as exc:
                    print(f"Stopped: this email failed ({type(exc).__name__}). It's marked read; "
                          "mark it unread in Gmail to try again.\n")
                    continue
                total_in += t_in
                total_out += t_out

                if not pending:
                    print("Result: no reply was saved, so nothing was sent or drafted.\n")
                    continue
                msg = gmail.make_reply(email, pending["text"], real_sender)
                if pending["needs_human_review"]:
                    threaded, label_note = gmail.save_draft(msg, thread, uid, True)
                    drafted += 1
                    print(f"Result: kept as a Gmail draft (needs review). Label: {label_note}. "
                          f"Same thread: {thread_note(threaded)}.\n")
                    continue
                try:
                    gmail.send(msg)
                except (smtplib.SMTPException, OSError) as exc:
                    code = getattr(exc, "smtp_code", "")
                    threaded, _ = gmail.save_draft(msg, thread, uid, False)
                    drafted += 1
                    print(f"Result: SENDING FAILED ({type(exc).__name__} {code}). Saved as a "
                          f"Gmail draft instead. Same thread: {thread_note(threaded)}.\n")
                    continue
                sent += 1
                print(f"Result: SENT with Gmail SMTP ({sent} of {MAX_SENT}).\n")
                if sent >= MAX_SENT:
                    break

            if sent >= MAX_SENT:
                print(f"Reached the limit of {MAX_SENT} sent replies; stopping.")
                break
            for _ in range(POLL_SECONDS):  # short sleeps so Ctrl+C stops it quickly
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        gmail.close()
        print(RULE)
        print(f"Replies sent: {sent}. Kept as drafts: {drafted}.")
        print(f"Total tokens: {total_in:,} input, {total_out:,} output")
        sys.stdout.flush()
        sys.stdout, sys.stderr = real_out, real_err


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--email", metavar="PATH", help="path of one employee email file (.eml)")
    source.add_argument("--inbox", action="store_true",
                        help="handle each unread email in the Gmail sandbox inbox (drafts only)")
    source.add_argument("--watch", action="store_true",
                        help=f"check every {POLL_SECONDS}s; handle allowlisted senders; send "
                             f"unflagged replies (at most {MAX_SENT})")
    parser.add_argument("--save", metavar="FILE",
                        help="also write everything printed to FILE (UTF-8)")
    args = parser.parse_args()
    if args.email and not Path(args.email).is_file():
        sys.exit(f"Email file not found: {args.email}")

    sys.stdout.reconfigure(encoding="utf-8")
    log = None
    if args.save:
        try:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            log = open(args.save, "w", encoding="utf-8")
        except OSError as exc:
            sys.exit(f"Can't write the --save file {args.save}: {exc.strerror}")
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)

    code = 0
    try:
        client = anthropic.Anthropic(api_key=load_api_key())
        if args.watch:
            watch_mode(client)
        elif args.inbox:
            inbox_mode(client)
        else:
            email_mode(client, args.email)
    except SystemExit as exc:  # print the reason now, so --save captures it
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            code = 1
        else:
            code = exc.code or 0
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.", file=sys.stderr)
        code = 130
    except anthropic.AuthenticationError:
        print("The API key was rejected (authentication error).", file=sys.stderr)
        code = 1
    except anthropic.RateLimitError:
        print("Rate limited by the API; try again shortly.", file=sys.stderr)
        code = 1
    except anthropic.APIStatusError as exc:
        print(f"API error {exc.status_code}: {exc.message}", file=sys.stderr)
        code = 1
    except anthropic.APIConnectionError:
        print("Couldn't reach the Anthropic API (connection error).", file=sys.stderr)
        code = 1
    finally:
        if log:
            sys.stdout.flush()
            sys.stdout, sys.stderr = sys.stdout.stream, sys.stderr.stream
            log.close()
            print(f"Output saved to {args.save}")
    sys.exit(code)


if __name__ == "__main__":
    main()
