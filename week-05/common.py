"""Shared helpers for the week-05 agent team (adapted from week-04/agent.py).

- load_api_key, parse_email, preview, Tee: copied from week 4.
- ask_claude: one Messages API call with this project's model settings.
- run_tool_loop: the week-4 tool-use loop, made general so any agent can use it
  (the supply analyst with MCP tools, the orchestrator with its own tools).
- Usage: token and call counts per agent, printed at the end of a run.
"""

import json
import os
import sys
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = HERE / "brookfield-bikes"
PROMPTS_DIR = DATA_DIR / "prompts"
DRAFTS_DIR = HERE / "drafts"
KEY_FILES = [REPO.parent / "API Key.txt", REPO.parent / "API-Key.txt"]

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
# If a safety classifier declines a request, the API retries it on a fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Server-side fallbacks are only offered for these models (not, e.g., claude-sonnet-5).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
                   "claude-fable-5-2", "claude-fable-5-1"}
PREVIEW_CHARS = 300
RULE = "=" * 70

STOP_REASONS = {
    "end_turn": "finished",
    "max_tokens": f"hit the {MAX_TOKENS}-token limit",
    "refusal": "declined the request (refusal)",
    "stop_sequence": "hit a stop sequence",
}


# ---------------------------------------------------------------------------
# Setup and email helpers (from week 4)
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


def read_prompt(name):
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def parse_email(path):
    """An .eml file -> the fields the agents need."""
    msg = BytesParser(policy=policy.default).parsebytes(Path(path).read_bytes())
    name, address = parseaddr(str(msg.get("From", "")))
    part = msg.get_body(preferencelist=("plain",))
    body = part.get_content() if part is not None else ""
    return {
        "path": Path(path),
        "from_name": name.strip(),
        "sender": address.strip().lower(),
        "subject": str(msg.get("Subject", "")).strip(),
        "body": body.strip(),
    }


def email_block(email):
    return (f"<email>\nFrom: {email['from_name']} <{email['sender']}>\n"
            f"Subject: {email['subject']}\n\n{email['body']}\n</email>")


def reply_subject(subject):
    subject = subject or "(no subject)"
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def preview(text, limit=PREVIEW_CHARS):
    text = str(text).replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "..."


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

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ---------------------------------------------------------------------------
# Claude calls
# ---------------------------------------------------------------------------

class Usage:
    """Calls and tokens per agent, so the two coordinators can be compared."""

    def __init__(self):
        self.rows = {}

    def add(self, agent, response):
        row = self.rows.setdefault(agent, [0, 0, 0])
        row[0] += 1
        row[1] += response.usage.input_tokens
        row[2] += response.usage.output_tokens

    def totals(self):
        """(input tokens, output tokens) across all agents."""
        return (sum(r[1] for r in self.rows.values()), sum(r[2] for r in self.rows.values()))

    def report(self):
        lines = [f"{'Agent':18} {'API calls':>9} {'Input tokens':>13} {'Output tokens':>14}"]
        totals = [0, 0, 0]
        for agent, (calls, tin, tout) in self.rows.items():
            lines.append(f"{agent:18} {calls:>9} {tin:>13,} {tout:>14,}")
            totals = [totals[0] + calls, totals[1] + tin, totals[2] + tout]
        lines.append(f"{'Total':18} {totals[0]:>9} {totals[1]:>13,} {totals[2]:>14,}")
        return "\n".join(lines)


def ask_claude(client, usage, agent, system, messages, tools=None, model=MODEL,
               tool_choice=None):
    """One Messages API call. Returns the response (content is appended by the caller)."""
    kwargs = {"tools": tools} if tools else {}
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    if model in FALLBACK_MODELS:
        kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
    response = client.beta.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=system,
        messages=messages,
        **kwargs,
    )
    usage.add(agent, response)
    return response


def response_text(response):
    return "".join(b.text for b in response.content if b.type == "text").strip()


def single_turn(client, usage, agent, system, messages, model=MODEL):
    """A call with no tools. Returns the text; exits the step with an error if Claude didn't finish."""
    response = ask_claude(client, usage, agent, system, messages, model=model)
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"{agent} {STOP_REASONS.get(response.stop_reason, response.stop_reason)}.")
    return response_text(response)


async def run_tool_loop(client, usage, agent, system, tools, messages, handle_tool,
                        max_calls, is_done=lambda: False):
    """The week-4 agent loop, for any agent.

    handle_tool(name, input) is async and returns (text, is_error). The loop stops when
    Claude ends its turn, when is_done() is true after a round of tool calls (for a
    'finish' tool), or after max_calls tool calls. Returns (final_text, stop_message).
    """
    calls = 0
    while True:
        response = ask_claude(client, usage, agent, system, messages, tools)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response_text(response), STOP_REASONS.get(
                response.stop_reason, f"stopped ({response.stop_reason})")

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if calls >= max_calls:
                text, is_error = (f"Tool-call limit ({max_calls}) reached; finish now without "
                                  "more tools."), True
            else:
                calls += 1
                print(f"[{agent}] step {calls}: {block.name}")
                print(f"  inputs: {json.dumps(block.input, ensure_ascii=False)}")
                try:
                    text, is_error = await handle_tool(block.name, block.input)
                except Exception as exc:
                    text, is_error = f"Error: {exc}", True
                print(f"  result{' (error)' if is_error else ''}: {preview(text)}\n")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": text, "is_error": is_error})
        if is_done():
            return "", "finished (finish tool)"
        if calls >= max_calls and all(r["is_error"] for r in results):
            return "", f"stopped at the limit of {max_calls} tool calls"
        messages.append({"role": "user", "content": results})
