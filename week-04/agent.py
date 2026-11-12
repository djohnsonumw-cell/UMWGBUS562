"""Hartwell Foods HR agent: drafts a reply to one employee email using Claude tool use.

For the email given with --email, the agent:
  1. reads the sender's address from the From: header,
  2. starts hr_mcp_server.py in employee mode (--as-employee <sender>), so Claude can
     look up only that sender's leave balance,
  3. offers Claude the server's tools plus save_reply, which writes a draft to
     week-04/drafts/ (it saves only; nothing is ever sent),
  4. loops until Claude stops asking for tools or 10 tool calls have run.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\agent.py --email week-04\\hartwell-hr\\sample-emails\\03-megan-time-off.eml

The API key comes from the ANTHROPIC_API_KEY environment variable, or else from
"API Key.txt" in the folder above the repository. It is never printed or saved.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SERVER_SCRIPT = HERE / "hr_mcp_server.py"
SYSTEM_PROMPT_FILE = HERE / "hartwell-hr" / "system_prompt.txt"
DRAFTS_DIR = HERE / "drafts"
SERVER_LOG = HERE / "mcp_server.log"  # *.log is git-ignored
KEY_FILES = [REPO.parent / "API Key.txt", REPO.parent / "API-Key.txt"]

MODEL = "claude-sonnet-5"
MAX_TOOL_CALLS = 10
MAX_TOKENS = 16000
PREVIEW_CHARS = 300

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


def read_email(path):
    """Return (sender_address, subject, body) from an .eml file."""
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    _, address = parseaddr(str(msg.get("From", "")))
    address = address.strip().lower()
    if "@" not in address:
        sys.exit(f"Couldn't find a sender address in the From: line of {path}.")
    part = msg.get_body(preferencelist=("plain",))
    body = part.get_content() if part is not None else ""
    return address, str(msg.get("Subject", "")).strip(), body.strip()


def preview(text, limit=PREVIEW_CHARS):
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "..."


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def save_reply(args, sender, subject, email_path):
    """Write the draft to week-04/drafts/. Saves only; never sends."""
    reply_text = args.get("reply_text")
    needs_review = args.get("needs_human_review")
    if not isinstance(reply_text, str) or not reply_text.strip():
        raise ValueError("reply_text must be a non-empty string.")
    if not isinstance(needs_review, bool):
        raise ValueError("needs_human_review must be true or false.")

    DRAFTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    local = re.sub(r"[^a-z0-9.-]", "_", sender.split("@")[0])
    path = DRAFTS_DIR / f"{stamp}-{local}.txt"
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    draft = (
        f"To: {sender}\n"
        f"Subject: {reply_subject}\n"
        f"Needs human review: {'YES' if needs_review else 'no'}\n"
        f"In reply to: {Path(email_path).name}\n"
        f"Status: DRAFT - not sent\n"
        f"\n{reply_text.strip()}\n"
    )
    path.write_text(draft, encoding="utf-8")
    return path, draft


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


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run(email_path):
    sender, subject, body = read_email(email_path)
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()  # read every run
    client = anthropic.Anthropic(api_key=load_api_key())

    print(f"Email:  {email_path}")
    print(f"Sender: {sender}")
    print(f"Starting {SERVER_SCRIPT.name} --as-employee {sender} ...")

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT), "--as-employee", sender],
        cwd=str(REPO),
    )
    with open(SERVER_LOG, "w", encoding="utf-8") as errlog:
        async with stdio_client(server, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = (await session.list_tools()).tools
                mcp_names = {t.name for t in mcp_tools}
                tools = [{"name": t.name, "description": t.description or "",
                          "input_schema": t.input_schema} for t in mcp_tools]
                tools.append(SAVE_REPLY_TOOL)
                print(f"Tools:  {', '.join(t['name'] for t in tools)}\n")

                messages = [{
                    "role": "user",
                    "content": (
                        "Prepare a reply to this employee email.\n\n"
                        "<email>\n"
                        f"From: {sender}\n"
                        f"Subject: {subject}\n\n"
                        f"{body}\n"
                        "</email>"
                    ),
                }]

                tool_calls = 0
                tokens_in = tokens_out = 0
                draft = None
                stop_reason_text = None

                while stop_reason_text is None:
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
                        stop_reason_text = reasons.get(
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
                            stop_reason_text = (f"Reached the limit of {MAX_TOOL_CALLS} "
                                                "tool calls before Claude finished.")
                            break
                        tool_calls += 1
                        print(f"Step {tool_calls}: {block.name}")
                        print(f"  inputs: {json.dumps(block.input, ensure_ascii=False)}")

                        try:
                            if block.name == "save_reply":
                                if draft is not None:
                                    raise ValueError("A draft was already saved; save_reply runs only once.")
                                path, draft = save_reply(block.input, sender, subject, email_path)
                                text, is_error = f"Draft saved (not sent) to {path.relative_to(REPO)}.", False
                            elif block.name in mcp_names:
                                text, is_error = await call_mcp_tool(session, block.name, block.input)
                            else:
                                text, is_error = f"Unknown tool: {block.name}", True
                        except Exception as exc:
                            text, is_error = f"Error: {exc}", True

                        print(f"  result{' (error)' if is_error else ''}: {preview(text)}\n")
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": text, "is_error": is_error})

                    if stop_reason_text is None:
                        messages.append({"role": "user", "content": results})

    print("=" * 70)
    print(f"Stopped: {stop_reason_text}")
    print(f"Tool calls: {tool_calls}")
    print("=" * 70)
    if draft:
        print(draft)
    else:
        print("No draft was saved.")
    print("=" * 70)
    print(f"Total tokens: {tokens_in:,} input, {tokens_out:,} output")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--email", required=True, metavar="PATH",
                        help="path of one employee email file (.eml)")
    args = parser.parse_args()
    if not Path(args.email).is_file():
        sys.exit(f"Email file not found: {args.email}")
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        asyncio.run(run(args.email))
    except anthropic.AuthenticationError:
        sys.exit("The API key was rejected (authentication error).")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the API; try again shortly.")
    except anthropic.APIStatusError as exc:
        sys.exit(f"API error {exc.status_code}: {exc.message}")
    except anthropic.APIConnectionError:
        sys.exit("Couldn't reach the Anthropic API (connection error).")


if __name__ == "__main__":
    main()
