"""The journey dataset contract: a folder of CSV files anyone can export.

    manifest.yaml       optional: what the batch declares about itself (v2:
                        contract_version, angles, date_order, coverage_from,
                        windows, data_end, ... - see models.Manifest)
    holidays.yaml       optional: bank holidays in the batch's window
    export_manifest.csv optional: file, rows - the exporter's own counts
    interactions.csv    interaction_id, account_ref, started_at, channel        (required)
                        branch, recorded, direction, status, call_key,
                        correspondence_id, talk_seconds, banker_code, unit_code (optional)
    call_segments.csv   call_key, seq, file_name [, recorded_at]
    messages.csv        correspondence_id, message_id, sent_at, direction [, body, subject]

`account_ref` is whatever identifies the account in the export (e.g.
"17/335145"); it is turned into a keyed digest at import and printed nowhere.
Only the columns above are read - any other column in a file is dropped and
listed in the import report. A column whose name says it holds a person's
identity (an ID number, a phone, a name, a user name) is not merely dropped:
it fails the import, because such a file must not be on this machine at all.

channel: call | message | branch | other
direction: inbound | outbound | (empty)
status: answered | abandoned | (empty)
recorded: 1/0, yes/no (a call with segments is recorded regardless)
"""

from __future__ import annotations

from pathlib import Path

from callqa.ingestion import _csv_reader, _read_text_any_encoding, _visible
from callqa.journey.importers.build import (
    Built,
    RawInteraction,
    RawMessage,
    RawSegment,
    build_dataset,
)
from callqa.journey.importers.common import AudioSource
from callqa.journey.importers.manifest import (
    check_declared_layers,
    check_row_counts,
    day_first_of,
    declared_row_counts,
    load_holidays,
    load_manifest,
)
from callqa.journey.importers.units import read_units
from callqa.journey.models import ImportReport
from callqa.journey.timeparse import (
    TimeParseError,
    detect_day_first,
    has_slashed_dates,
    parse_dt,
    try_parse_dt,
    tz_suffix,
)

SOURCE = "journey-contract-v1"
SOURCE_V2 = "journey-contract-v2"

# The call-centre table's own columns (T4418 / T4425 / T4444), contract v2:
# carried raw on the contact, decoded nowhere here.
FACT_COLUMNS = ("direction_code", "churn_call_code", "call_status_code", "call_source_code",
                "employee_target_type_code", "cti_at", "match_gap_sec", "caller_is_owner",
                "service_mode_code", "phone_meeting_ind", "manui_moked_ind", "chat_id",
                "whatsapp_id", "story_id")
MESSAGE_V2_COLUMNS = ("send_method", "channel_msg_code", "message_status_code", "read_at",
                      "template_code", "main_category_code", "sub_category_code",
                      "customer_mood", "ai_correspondence_ind", "banker_code", "unit_code",
                      "call_key")
ALLOWED = {
    "interactions.csv": {"interaction_id", "account_ref", "started_at", "channel", "branch",
                         "recorded", "direction", "status", "call_key", "correspondence_id",
                         "talk_seconds", "banker_code", "unit_code", *FACT_COLUMNS},
    "call_segments.csv": {"call_key", "seq", "file_name", "recorded_at", "duration_seconds"},
    "messages.csv": {"correspondence_id", "message_id", "sent_at", "direction", "body",
                     "subject", *MESSAGE_V2_COLUMNS},
}
REQUIRED = {
    "interactions.csv": {"interaction_id", "account_ref", "started_at", "channel"},
    "call_segments.csv": {"call_key", "file_name"},
    "messages.csv": {"correspondence_id", "message_id", "sent_at", "direction"},
}
_CHANNELS = {"call", "message", "chat", "whatsapp", "branch", "other"}
_TRUE = {"1", "1.0", "yes", "y", "true", "כן"}
_FALSE = {"0", "0.0", "no", "n", "false", "לא"}


def _flag(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None
# Column names that say the file carries a person's identity. The match is on
# the whole normalised name (lower case, separators dropped), so "national_id"
# and "National ID" fail while "unit_code" or "banker_code" do not.
FORBIDDEN_COLUMNS = {
    "nationalid", "idnumber", "identitynumber", "identity", "teudatzehut", "tz", "ssn",
    "passport", "phone", "phonenumber", "mobile", "cellphone", "telephone", "msisdn",
    "name", "fullname", "firstname", "lastname", "customername", "username", "user",
    "email", "emailaddress", "address",
    "תז", "תעודתזהות", "מספרזהות", "טלפון", "נייד", "מספרטלפון", "שם", "שםלקוח", "שםפרטי",
    "שםמשפחה", "שםמלא", "שםמשתמש", "דואל", "אימייל", "כתובת",
}
# Files a batch folder may hold besides the tables; anything else is reported.
KNOWN_FILES = {"manifest.yaml", "holidays.yaml", "export_manifest.csv", "checks.csv",
               "interactions.csv", "call_segments.csv", "messages.csv", "units.csv",
               "readme.md", "readme.txt"}


def _norm_name(column: str) -> str:
    return "".join(ch for ch in column.lower() if ch.isalnum() or "֐" <= ch <= "׿")


def _rows(folder: Path, name: str, report: ImportReport,
          rows_read: dict[str, int] | None = None) -> list[dict[str, str]]:
    path = folder / name
    if not path.exists():
        return []
    reader = _csv_reader(_read_text_any_encoding(path))
    header = [_visible(c or "").strip().lower() for c in (reader.fieldnames or [])]
    forbidden = [c for c in header if _norm_name(c) in FORBIDDEN_COLUMNS]
    if forbidden:
        # Reported by column name only - never a value from it.
        report.add("forbidden_columns", "error",
                   f"{name} carries a column that names a person's identity, which must "
                   f"not be exported to this tool: {', '.join(forbidden)}")
        return []
    missing = REQUIRED[name] - set(header)
    if missing:
        report.add("missing_columns", "error",
                   f"{name} lacks required column(s): {', '.join(sorted(missing))}")
        return []
    dropped = [c for c in header if c and c not in ALLOWED[name]]
    if dropped:
        report.dropped_columns[name] = dropped
    out = []
    for raw in reader:
        raw.pop(None, None)
        row = {_visible(k or "").strip().lower(): _visible(v or "").strip()
               for k, v in raw.items() if k}
        row = {k: v for k, v in row.items() if k in ALLOWED[name]}
        if any(row.values()):
            out.append(row)
    if rows_read is not None:
        rows_read[name] = len(out)
    return out


def _unknown_files(folder: Path, report: ImportReport) -> None:
    """A file the contract does not know is said, not skipped: it may be the
    one table the exporter misnamed."""
    names = sorted(p.name for p in folder.iterdir()
                   if p.is_file() and p.name.lower() not in KNOWN_FILES
                   and not p.name.startswith("."))
    if names:
        report.add("unknown_files", "info",
                   "files in the batch folder that are no table of the contract; not read",
                   count=len(names), examples=names[:5])


def _split_ref(ref: str) -> tuple[str, str]:
    """'17/335145' or '17-335145' -> branch, account; otherwise no branch."""
    for sep in ("/", "-", " "):
        if sep in ref:
            a, b = ref.split(sep, 1)
            return a.strip(), b.strip()
    return "", ref.strip()


def _date_order(rows: list[dict[str, str]], column: str, name: str, report: ImportReport,
                declared: bool | None) -> bool:
    """Day-first or month-first for one column of one file. The batch may
    declare it (manifest date_order); otherwise the column decides when any
    value has a day above 12, and a column of slashed dates that nothing
    decides is refused - a US file read day-first swaps months and days
    silently, which is worse than stopping."""
    values = [r.get(column, "") for r in rows]
    if declared is not None:
        return declared
    detected = detect_day_first(values)
    if detected is None and has_slashed_dates(values):
        report.add("date_order_ambiguous", "error",
                   f"{name}.{column}: dd/mm or mm/dd cannot be told from the values; "
                   "set date_order in manifest.yaml (day_first | month_first)")
    return True if detected is None else detected


def read_contract_rows(folder: Path, report: ImportReport, *, day_first: bool | None = None,
                       rows_read: dict[str, int] | None = None
                       ) -> tuple[list[RawInteraction], list[RawSegment], list[RawMessage]]:
    interactions: list[RawInteraction] = []
    bad = 0
    ambiguous: list[str] = []
    tz = 0
    rows = _rows(folder, "interactions.csv", report, rows_read)
    order = _date_order(rows, "started_at", "interactions.csv", report, day_first)
    for n, row in enumerate(rows, start=2):
        tz += tz_suffix(row.get("started_at"))
        try:
            at = parse_dt(row["started_at"], day_first=order)
        except TimeParseError as exc:
            if str(exc).startswith("ambiguous_numeric_date"):
                ambiguous.append(row["started_at"][:20])
            bad += 1
            continue
        channel = row.get("channel", "").lower()
        if channel not in _CHANNELS:
            channel = "other"
        branch = row.get("branch", "")
        ref_branch, account = _split_ref(row.get("account_ref", ""))
        source_id = row.get("call_key") or row.get("correspondence_id") or row.get("interaction_id", "")
        talk = None
        try:
            talk = float(row["talk_seconds"]) if row.get("talk_seconds") else None
        except ValueError:
            pass
        status = row.get("status", "").lower()
        facts = {k: row[k] for k in FACT_COLUMNS if row.get(k)}
        if "cti_at" in facts:
            cti = try_parse_dt(facts["cti_at"], day_first=order)
            facts["cti_at"] = cti.isoformat() if cti else ""
        interactions.append(RawInteraction(
            row=n, branch=branch or ref_branch, account=account, at=at, channel=channel,
            source_id=source_id, direction=row.get("direction", "").lower() or "unknown",
            answer=status if status in ("answered", "abandoned") else "unknown",
            talk_seconds=talk, banker_code=row.get("banker_code") or None,
            unit_code=row.get("unit_code") or None,
            recorded=_flag(row.get("recorded", "")) if row.get("recorded") else None,
            facts={k: v for k, v in facts.items() if v}))
    if ambiguous:
        report.add("ambiguous_numeric_date", "error",
                   "interactions.csv: started_at holds a number that could be an Excel day "
                   "count or SAS days since 1960; export a date, or SAS seconds",
                   count=len(ambiguous), examples=ambiguous[:5])
    if bad:
        report.add("bad_time", "error", "interactions whose started_at could not be read",
                   count=bad)
    if tz:
        report.add("tz_stripped", "info",
                   "timestamps carried a zone suffix; the clock reading was kept as the "
                   "bank's local time and the zone dropped", count=tz)
    for i in interactions:
        if i.direction not in ("inbound", "outbound"):
            i.direction = {"i": "inbound", "o": "outbound"}.get(i.direction, "unknown")

    segments: list[RawSegment] = []
    rows = _rows(folder, "call_segments.csv", report, rows_read)
    order = _date_order(rows, "recorded_at", "call_segments.csv", report, day_first)
    for row in rows:
        seq = None
        try:
            seq = int(float(row["seq"])) if row.get("seq") else None
        except ValueError:
            pass
        segments.append(RawSegment(call_key=row["call_key"].lower(), file_name=row["file_name"],
                                   seq=seq,
                                   recorded_at=try_parse_dt(row.get("recorded_at"),
                                                            day_first=order)))

    messages: list[RawMessage] = []
    rows = _rows(folder, "messages.csv", report, rows_read)
    order = _date_order(rows, "sent_at", "messages.csv", report, day_first)
    bad_msg = 0
    for row in rows:
        at = try_parse_dt(row.get("sent_at"), day_first=order)
        if at is None:
            bad_msg += 1
            continue
        messages.append(RawMessage(correspondence_id=row["correspondence_id"].upper(),
                                   message_id=row["message_id"], at=at,
                                   direction=row.get("direction", ""),
                                   subject=row.get("subject", ""), body=row.get("body", ""),
                                   send_method=row.get("send_method", "").upper()[:4],
                                   channel_msg_code=row.get("channel_msg_code") or None,
                                   template_code=row.get("template_code") or None,
                                   call_key=(row.get("call_key") or "").lower() or None,
                                   banker_code=row.get("banker_code") or None,
                                   unit_code=row.get("unit_code") or None))
    if bad_msg:
        report.add("bad_message_time", "warning",
                   "messages whose sent_at could not be read (left out of the threads)",
                   count=bad_msg)
    return interactions, segments, messages


def import_contract(folder: str | Path, *, audio: str | Path | None = None,
                    redact_messages: bool = True, atlas: bool = False) -> Built:
    """`atlas` says whether an Atlas source accompanies the batch (the caller
    knows; it is attached afterwards), so a manifest that declares the Atlas
    layer can be checked against it."""
    folder = Path(folder)
    report = ImportReport(source=SOURCE)
    manifest = load_manifest(folder, report)
    if not folder.is_dir():
        report.add("no_interactions", "error", f"batch folder not found: {folder}")
        return build_dataset(SOURCE, [], [], [], report)
    _unknown_files(folder, report)
    if not (folder / "interactions.csv").exists():
        report.add("no_interactions", "error", "interactions.csv is missing")
    rows_read: dict[str, int] = {}
    interactions, segments, messages = read_contract_rows(
        folder, report, day_first=day_first_of(manifest), rows_read=rows_read)
    check_row_counts(declared_row_counts(folder, report), rows_read, report)
    present = {"vendor"}
    if atlas:
        present.add("atlas")
    if audio:
        present.add("vendor")
    check_declared_layers(manifest, present, report)
    source = AudioSource.open(audio) if audio else None
    built = build_dataset(SOURCE, interactions, segments, messages, report, audio=source,
                         redact_messages=redact_messages)
    ds = built.dataset
    ds.holidays = load_holidays(folder, report)
    ds.units = read_units(folder, report)
    if manifest is not None:
        ds.manifest = manifest
        ds.contract_version = manifest.contract_version
        ds.data_end = manifest.data_end
        if manifest.contract_version == 2:
            ds.source = SOURCE_V2
    return built

