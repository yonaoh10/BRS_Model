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


_RUNNING_CODE = re.compile(r"^B\d{4,6}$")
_CODE_LIKE = re.compile(r"^[A-Za-z]{0,3}\d{1,10}$")
_HEBREW = re.compile(r"[֐-׿]")


class BankerCodes:
    """Bankers by a running code, never by the bank's user name (the rule of
    every Atlas deliverable: B0001... per export). A value that already is a
    running code (as ATL_R01 writes them) is kept, so the bank's own tables
    still read across; any other value - a user name, an employee number -
    is replaced by the next free code, and the mapping stays in the private
    folder only. A value that looks like a person's name is counted as such
    so the import report can say the export carried names."""

    # Codes assigned here start at B10001: the bank's own running codes have
    # four digits (B0001..), so a code given to a user name here can never
    # collide with one the export already carried for someone else.
    FIRST_ASSIGNED = 10001

    def __init__(self) -> None:
        self.raw_to_code: dict[str, str] = {}
        self.name_like = 0
        self._next = self.FIRST_ASSIGNED

    def code_for(self, raw: object) -> str | None:
        text = str(raw if raw is not None else "").strip()
        if not text:
            return None
        if text in self.raw_to_code:
            return self.raw_to_code[text]
        if _RUNNING_CODE.match(text):
            self.raw_to_code[text] = text
            return text
        if not _CODE_LIKE.match(text) and (" " in text or _HEBREW.search(text)
                                           or any(ch.isalpha() for ch in text)):
            self.name_like += 1
        code = f"B{self._next:05d}"
        self._next += 1
        self.raw_to_code[text] = code
        return code

    @property
    def rewritten(self) -> int:
        return sum(1 for raw, code in self.raw_to_code.items() if raw != code)

    def rows(self) -> list[tuple[str, str]]:
        return sorted((code, raw) for raw, code in self.raw_to_code.items() if raw != code)


def write_private_bankers(folder: Path, codes: BankerCodes) -> Path | None:
    """(code, raw value) -> private/bankers.csv; nothing when no value was
    rewritten (the export already used running codes)."""
    rows = codes.rows()
    if not rows:
        return None
    from callqa.portable import make_private_dir
    from callqa.state import atomic_write_text

    make_private_dir(folder)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["banker_code", "source_value"])
    for row in rows:
        writer.writerow(row)
    path = folder / "bankers.csv"
    atomic_write_text(path, buffer.getvalue())
    return path


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
