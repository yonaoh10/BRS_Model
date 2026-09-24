"""Banker sessions in the Atlas log, built the way the bank's Atlas project
builds them (ATL_R01_repeat_contacts, block 6), and what each one was (ATL_R02).

The Atlas log holds one row for every screen a banker opens on an account and
every action taken there: when, which banker (a running code, never a name),
from which unit, and the operation code. A session is a run of rows on one
account; the rows are taken in time order, and a new session starts when

    the banker changes,
    more than `gap_min` minutes passed since the previous row (30), or
    the row is the code that opens the customer screen (201).

A session lasts from its first row to its last - a lower bound on the time
worked. Its banker and unit are those of its first row. A peek is a session of
one row, the opening code.

Rows of the same second keep the order of the export (ATL_R02 numbers the rows
of ATLR_ROWS in file order and relies on it), so the file order is the tie
break - never a guess.

A session's unit class (ATL_R02), in this order: the banking centre (109), the
back office (136), the account's own branch, or another branch or unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from callqa.journey.models import AtlasRules
from callqa.journey.vocab import Units, normalise_op_code

UnitClass = Literal["center", "back", "own", "other"]
UNIT_CLASSES: tuple[UnitClass, ...] = ("center", "back", "own", "other")
UNIT_CLASS_HE = {"center": "מרכז הבנקאות", "back": "תפעול עורפי", "own": "סניף החשבון",
                 "other": "סניף או יחידה אחרים"}
SESSION_KIND_HE = {"execute": "ביצוע או שינוי", "info": "מידע ושאילתות",
                   "unclassified": "פעולה שלא סווגה", "open": "פתיחת מסך בלבד",
                   "not_customer": "לא על הלקוח"}
OP_CATEGORY_HE = SESSION_KIND_HE


@dataclass(frozen=True)
class AtlasRow:
    account: str            # the Atlas account key (ACC_KEY), never an account number
    at: datetime
    banker: str
    unit: str
    op: str                 # normalised operation code
    description: str = ""
    order: int = 0          # the row's place in the export


def build_sessions(rows: list[AtlasRow], *, gap_min: float = 30.0,
                   start_op: str = "201") -> list[list[AtlasRow]]:
    """The rows cut into sessions (ATL_R01 B6), in account and time order."""
    ordered = sorted(rows, key=lambda r: (r.account, r.at, r.order))
    sessions: list[list[AtlasRow]] = []
    prev: AtlasRow | None = None
    for row in ordered:
        new = (prev is None or row.account != prev.account or row.banker != prev.banker
               or (row.at - prev.at).total_seconds() / 60.0 > gap_min
               or row.op == start_op)
        if new:
            sessions.append([row])
        else:
            sessions[-1].append(row)
        prev = row
    return sessions


def unit_class(units: Units, code: str | None, own_branch: str | None) -> UnitClass:
    """ATL_R02 U_CLASS. The editable unit list decides the centre and the back
    office; every other unit - a branch or a headquarters unit - is the
    account's own branch when its number is the account's branch, else other."""
    kind = units.kind(code, own_branch)
    if kind == "center":
        return "center"
    if kind == "back_office":
        return "back"
    c = (code or "").lstrip("0")
    if own_branch and c and c == str(own_branch).lstrip("0"):
        return "own"
    return "other"


def rules_from_config(atlas) -> AtlasRules:
    """`journey.atlas` of the config as the rules a dataset keeps."""
    w = atlas.windows
    cover = (datetime(atlas.coverage_from.year, atlas.coverage_from.month,
                      atlas.coverage_from.day) if atlas.coverage_from else None)
    return AtlasRules(gap_min=atlas.gap_min, start_op=normalise_op_code(atlas.start_op),
                      call_out_before=w.call_out_before, call_in_after=w.call_in_after,
                      msg_out_before=w.msg_out_before, msg_in_after=w.msg_in_after,
                      unk_before=w.unk_before, unk_after=w.unk_after, coverage_from=cover)
