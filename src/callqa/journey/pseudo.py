"""Account numbers never travel: a keyed digest and a story number do.

`account_key` is an HMAC with the installation's secret key (the one that
already names recordings, ingestion._install_key), so it cannot be reversed
by trying every account number. The only way back to the account is the
private map written beside the dataset, in a folder restricted to its owner,
and `callqa journey reveal`, which logs every use.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path

from callqa.ingestion import _digest

_DIGITS = re.compile(r"\d+")


def normalise_account(branch: object, account: object) -> str:
    """'017' / '0335145' and '17' / '335145' are the same account."""
    def clean(value: object) -> str:
        text = str(value if value is not None else "").strip()
        if text.endswith(".0"):          # a number that went through Excel
            text = text[:-2]
        digits = "".join(_DIGITS.findall(text))
        return digits.lstrip("0") or ("0" if digits else text.strip())
    b, a = clean(branch), clean(account)
    return f"{b}/{a}" if b else a


def account_key(branch: object, account: object) -> str:
    return "s" + _digest("acct:" + normalise_account(branch, account))[:15]


def number_stories(first_contact: dict[str, datetime]) -> dict[str, int]:
    """Story numbers 1..N by first contact (ties by key), so a report reads in
    the order the cases began."""
    ordered = sorted(first_contact, key=lambda k: (first_contact[k], k))
    return {key: n for n, key in enumerate(ordered, start=1)}


def story_label(story_no: int) -> str:
    return f"סיפור {story_no:03d}"


def write_private_map(folder: Path, rows: list[tuple[int, str, str, str]]) -> Path:
    """(story_no, story_key, branch, account) -> private/accounts.csv."""
    from callqa.portable import make_private_dir
    from callqa.state import atomic_write_text

    make_private_dir(folder)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["story_no", "story_key", "branch", "account"])
    for row in sorted(rows):
        writer.writerow(row)
    path = folder / "accounts.csv"
    atomic_write_text(path, buffer.getvalue())
    return path


def read_private_map(folder: Path) -> dict[int, tuple[str, str, str]]:
    path = folder / "accounts.csv"
    out: dict[int, tuple[str, str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["story_no"])] = (row["story_key"], row["branch"], row["account"])
    return out
