"""The bank's repeat-contact handoff workbook (the shape prepared for an
external vendor in 2026): three sheets.

    interactions   A first contact of the account, B branch, C account,
                   D contact time, E id (a hex call id, or UM-... for a
                   correspondence), F service segment (ignored), G declared
                   repeat count, H audio file name (recorded calls only)
    call files     A call id, B file name - one row per audio file; a call
                   recorded in several parts has several rows
    messages       A correspondence (UM-...), B message (COR-...), C sent at,
                   D Inbound/Outbound, E subject, F body

Sheets are recognised by what is in them, not by name or order: a SAS/EG
import renamed and reordered them once already. Columns are read by
position, so Hebrew or missing headers make no difference; a header row is
simply a row whose id column is not an id.
"""

from __future__ import annotations

from pathlib import Path

from callqa.journey.importers.build import (
    Built,
    RawInteraction,
    RawMessage,
    RawSegment,
    build_dataset,
)
from callqa.journey.importers.common import (
    COR_ID,
    HEX_CALL,
    SEGMENT_FILE,
    UM_ID,
    AudioSource,
    text,
)
from callqa.journey.models import ImportReport
from callqa.journey.timeparse import TimeParseError, parse_dt
from callqa.journey.xlsx import Sheet, Workbook, read_xlsx

SOURCE = "handoff-workbook"
SAMPLE_ROWS = 60


def _column(sheet: Sheet, col: int) -> list[str]:
    values = [text(sheet.cell(r, col)) for r in range(min(len(sheet.rows), SAMPLE_ROWS))]
    return [v for v in values if v]


def _share(sheet: Sheet, col: int, match) -> float:
    """The share of a column's sample that looks like an id. A first row that
    does not (a header) is left out, so a small sheet is judged fairly."""
    values = _column(sheet, col)
    if values and not match(values[0]):
        values = values[1:]
    if not values:
        return 0.0
    return sum(1 for v in values if match(v)) / len(values)


def _is_contact_id(value: str) -> bool:
    return bool(HEX_CALL.match(value) or UM_ID.match(value))


def classify(workbook: Workbook) -> dict[str, Sheet]:
    """{'interactions'|'files'|'messages': sheet}, by content."""
    found: dict[str, Sheet] = {}
    for sheet in workbook.sheets:
        if "messages" not in found and _share(sheet, 0, UM_ID.match) > 0.8 and _share(sheet, 1, COR_ID.match) > 0.8:
            found["messages"] = sheet
        elif "files" not in found and _share(sheet, 0, HEX_CALL.match) > 0.8 and \
                _share(sheet, 1, SEGMENT_FILE.match) > 0.8:
            found["files"] = sheet
        elif "interactions" not in found and _share(sheet, 4, _is_contact_id) > 0.8:
            found["interactions"] = sheet
    return found


def _int(value: object) -> int | None:
    try:
        return int(float(text(value)))
    except ValueError:
        return None


def read_workbook_rows(workbook: Workbook, report: ImportReport
                       ) -> tuple[list[RawInteraction], list[RawSegment], list[RawMessage]]:
    sheets = classify(workbook)
    if "interactions" not in sheets:
        report.add("no_interactions_sheet", "error",
                   "no sheet has contact ids (hex call ids or UM-...) in column E")
        return [], [], []
    for kind, what in (("files", "call-to-files"), ("messages", "messages")):
        if kind not in sheets:
            report.add(f"no_{kind}_sheet", "warning", f"no {what} sheet was found")

    interactions: list[RawInteraction] = []
    skipped = bad_time = 0
    for r, row in enumerate(sheets["interactions"].rows, start=1):
        rid = text(row[4] if len(row) > 4 else None)
        if not (HEX_CALL.match(rid) or UM_ID.match(rid)):
            if any(text(v) for v in row):
                skipped += 1
            continue
        try:
            at = parse_dt(row[3], date1904=workbook.date1904)
        except (TimeParseError, IndexError):
            bad_time += 1
            continue
        first = None
        try:
            first = parse_dt(row[0], date1904=workbook.date1904)
        except (TimeParseError, IndexError):
            pass
        is_um = bool(UM_ID.match(rid))
        interactions.append(RawInteraction(
            row=r, branch=text(row[1] if len(row) > 1 else ""),
            account=text(row[2] if len(row) > 2 else ""), at=at,
            channel="message" if is_um else "call",
            source_id=rid.upper() if is_um else rid.lower(),
            file_name=text(row[7] if len(row) > 7 else ""),
            declared_first_at=first,
            declared_repeat_count=_int(row[6]) if len(row) > 6 else None))
    if skipped:
        report.add("rows_skipped", "info",
                   "rows in the interactions sheet without a contact id (headers, notes)",
                   count=skipped)
    if bad_time:
        report.add("bad_time", "error", "contacts whose time could not be read", count=bad_time)

    segments: list[RawSegment] = []
    if "files" in sheets:
        for row in sheets["files"].rows:
            key, name = text(row[0] if row else ""), text(row[1] if len(row) > 1 else "")
            if HEX_CALL.match(key) and name:
                segments.append(RawSegment(call_key=key.lower(), file_name=name))

    messages: list[RawMessage] = []
    bad_msg = 0
    if "messages" in sheets:
        for row in sheets["messages"].rows:
            um = text(row[0] if row else "")
            if not UM_ID.match(um):
                continue
            try:
                at = parse_dt(row[2], date1904=workbook.date1904)
            except (TimeParseError, IndexError):
                bad_msg += 1
                continue
            messages.append(RawMessage(
                correspondence_id=um.upper(), message_id=text(row[1]).upper(), at=at,
                direction=text(row[3] if len(row) > 3 else ""),
                subject=text(row[4] if len(row) > 4 else ""),
                body=text(row[5] if len(row) > 5 else "")))
    if bad_msg:
        report.add("bad_message_time", "warning", "messages whose time could not be read",
                   count=bad_msg)
    known_um = {i.source_id for i in interactions if i.channel == "message"}
    stray = {m.correspondence_id for m in messages} - known_um
    if stray:
        report.add("messages_without_contact", "warning",
                   "correspondences in the messages sheet with no row in the interactions "
                   "sheet; their messages are not part of any story", count=len(stray))
    return interactions, segments, messages


def import_workbook(path: str | Path, *, audio: str | Path | None = None,
                    redact_messages: bool = True) -> Built:
    report = ImportReport(source=SOURCE)
    workbook = read_xlsx(path)
    interactions, segments, messages = read_workbook_rows(workbook, report)
    source = AudioSource.open(audio) if audio else None
    return build_dataset(SOURCE, interactions, segments, messages, report, audio=source,
                         redact_messages=redact_messages)
