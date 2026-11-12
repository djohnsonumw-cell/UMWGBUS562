"""Load the Hartwell sample emails into a Gmail sandbox inbox as unread messages.

Signs in to Gmail with IMAP and APPENDs each .eml file from
week-04/hartwell-hr/sample-emails directly into INBOX, unread and dated today.
Nothing is sent: APPEND places a message in the mailbox without any SMTP delivery.

The login comes from Agent-Email.txt in the folder above the repository, read at
run time. The password is never printed or saved, and neither is the address.
Gmail needs an app password here (Google Account > Security > App passwords),
not your normal password, and IMAP must be turned on in Gmail settings.

Agent-Email.txt can use labeled lines or plain lines:
    EMAIL_ADDRESS=you@gmail.com        you@gmail.com
    APP_PASSWORD=xxxxxxxxxxxxxxxx      xxxx xxxx xxxx xxxx
(Labels are matched loosely: anything with "email"/"address" or "pass"; ":" works too.)

Running it again skips messages already in the inbox (each sample gets a fixed
Message-ID), so it won't create duplicates.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\load_inbox.py
"""

import argparse
import imaplib
import re
import sys
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import format_datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SAMPLES_DIR = HERE / "hartwell-hr" / "sample-emails"
LOGIN_FILE = REPO.parent / "Agent-Email.txt"
IMAP_HOST = "imap.gmail.com"
MAILBOX = "INBOX"

EMAIL_KEYS = {"user", "username", "login"}  # plus any label containing "email" or "address"


def read_login(path):
    """Return (address, password) from the login file. Never echoes either value."""
    if not path.is_file():
        sys.exit(f"Login file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]

    address = password = None
    for line in lines:  # "key: value" or "key = value"
        m = re.match(r"^([A-Za-z _]+?)\s*[:=]\s*(.+)$", line)
        if not m:
            continue
        key = re.sub(r"[ _]+", "_", m.group(1).strip().lower())  # "App Password" -> "app_password"
        value = m.group(2).strip().strip("'\"")
        if "pass" in key:
            if password is None:
                password = value
        elif key in EMAIL_KEYS or "email" in key or "address" in key:
            if address is None:
                address = value
    if address is None or password is None:  # plain lines: address, then password
        plain = [ln for ln in lines if not re.match(r"^[A-Za-z _]+?\s*[:=]", ln)]
        if address is None:
            address = next((ln for ln in plain if "@" in ln and " " not in ln), None)
        if password is None:
            password = next((ln for ln in plain if ln != address), None)

    if not address or "@" not in address or not password:
        sys.exit(f"Couldn't find both an email address and a password in {path.name}. "
                 "Put the address on the first line and the app password on the second.")
    # Google shows app passwords as four groups of four letters; the spaces aren't part of it.
    if re.fullmatch(r"[A-Za-z]{4}( [A-Za-z]{4}){3}", password):
        password = password.replace(" ", "")
    return address, password


def prepare(path, now):
    """The message bytes with today's date and a fixed Message-ID, CRLF line endings."""
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    del msg["Date"]
    msg["Date"] = format_datetime(now)
    message_id = msg.get("Message-ID")
    if not message_id:
        message_id = f"<hartwell-sample-{path.stem}@load-inbox.local>"
        msg["Message-ID"] = message_id
    data = msg.as_bytes(policy=policy.default.clone(linesep="\r\n"))
    return data, str(message_id).strip(), str(msg.get("Subject", "")).strip()


def already_in_inbox(imap, message_id):
    status, data = imap.search(None, "HEADER", "Message-ID", f'"{message_id}"')
    return status == "OK" and bool(data and data[0].split())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--login-file", type=Path, default=LOGIN_FILE, help=argparse.SUPPRESS)
    args = parser.parse_args()

    files = sorted(SAMPLES_DIR.glob("*.eml"))
    if not files:
        sys.exit(f"No .eml files found in {SAMPLES_DIR}")
    address, password = read_login(args.login_file)

    print(f"Connecting to {IMAP_HOST} ...")
    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST, 993)
    except OSError as exc:
        sys.exit(f"Couldn't connect to {IMAP_HOST}: {exc}")

    try:
        try:
            imap.login(address, password)
        except imaplib.IMAP4.error:
            sys.exit("Sign-in failed. Check the address and app password in "
                     f"{args.login_file.name}, and that IMAP is on in Gmail settings.")
        finally:
            password = None  # drop it as soon as it's been used
        print("Signed in.")

        status, _ = imap.select(MAILBOX)
        if status != "OK":
            sys.exit(f"Couldn't open {MAILBOX}.")

        now = datetime.now().astimezone()
        added = skipped = 0
        for path in files:
            data, message_id, subject = prepare(path, now)
            if already_in_inbox(imap, message_id):
                print(f"  skipped {path.name} (already in the inbox)")
                skipped += 1
                continue
            # Empty flags = no \Seen, so the message arrives unread.
            status, resp = imap.append(MAILBOX, "", imaplib.Time2Internaldate(now), data)
            if status != "OK":
                print(f"  FAILED  {path.name}: {resp}")
                continue
            print(f"  added   {path.name}  ({subject})")
            added += 1

        print(f"Done: {added} added, {skipped} skipped. Nothing was sent.")
    finally:
        try:
            imap.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
