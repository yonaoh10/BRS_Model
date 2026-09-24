"""The journey dataset contract (v1): a folder of CSV files anyone can export.

    manifest.yaml       optional: {contract_version: 1, source: ..., exported_at: ...}
    interactions.csv    interaction_id, account_ref, started_at, channel        (required)
                        branch, recorded, direction, status, call_key,
                        correspondence_id, talk_seconds, banker_code, unit_code (optional)
    call_segments.csv   call_key, seq, file_name [, recorded_at]
    messages.csv        correspondence_id, message_id, sent_at, direction, body [, subject]

`account_ref` is whatever identifies the account in the export (e.g.
"17/335145"); it is turned into a keyed digest at import and printed nowhere.
Only the columns above are read - any other column in a file is dropped and
listed in the import report, so a name or ID column that rode along in an
export never reaches the dataset.

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
from callqa.journey.models import ImportReport
from callqa.journey.timeparse import TimeParseError, parse_dt, try_parse_dt
from callqa.resources import load_yaml

SOURCE = "journey-contract-v1"

ALLOWED = {
    "interactions.csv": {"interaction_id", "account_ref", "started_at", "channel", "branch",
                         "recorded", "direction", "status", "call_key", "correspondence_id",
                         "talk_seconds", "banker_code", "unit_code"},
    "call_segments.csv": {"call_key", "seq", "file_name", "recorded_at"},
    "messages.csv": {"correspondence_id", "message_id", "sent_at", "direction", "body",
                     "subject"},
}
REQUIRED = {
    "interactions.csv": {"interaction_id", "account_ref", "started_at", "channel"},
    "call_segments.csv": {"call_key", "file_name"},
    "messages.csv": {"correspondence_id", "message_id", "sent_at", "direction"},
}
_CHANNELS = {"call", "message", "branch", "other"}
_TRUE = {"1", "yes", "y", "true", "כן"}


def _rows(folder: Path, name: str, report: ImportReport) -> list[dict[str, str]]:
    path = folder / name
    if not path.exists():
        return []
    reader = _csv_reader(_read_text_any_encoding(path))
    header = [_visible(c or "").strip().lower() for c in (reader.fieldnames or [])]
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
    return out


def _split_ref(ref: str) -> tuple[str, str]:
    """'17/335145' or '17-335145' -> branch, account; otherwise no branch."""
    for sep in ("/", "-", " "):
        if sep in ref:
            a, b = ref.split(sep, 1)
            return a.strip(), b.strip()
    return "", ref.strip()


def read_contract_rows(folder: Path, report: ImportReport
                       ) -> tuple[list[RawInteraction], list[RawSegment], list[RawMessage]]:
    interactions: list[RawInteraction] = []
    bad = 0
    for n, row in enumerate(_rows(folder, "interactions.csv", report), start=2):
        try:
            at = parse_dt(row["started_at"])
        except TimeParseError:
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
        interactions.append(RawInteraction(
            row=n, branch=branch or ref_branch, account=account, at=at, channel=channel,
            source_id=source_id, direction=row.get("direction", "").lower() or "unknown",
            answer=status if status in ("answered", "abandoned") else "unknown",
            talk_seconds=talk, banker_code=row.get("banker_code") or None,
            unit_code=row.get("unit_code") or None,
            recorded=row.get("recorded", "").lower() in _TRUE if row.get("recorded") else None))
    if bad:
        report.add("bad_time", "error", "interactions whose started_at could not be read",
                   count=bad)
    for i in interactions:
        if i.direction not in ("inbound", "outbound"):
            i.direction = {"i": "inbound", "o": "outbound"}.get(i.direction, "unknown")

    segments: list[RawSegment] = []
    for row in _rows(folder, "call_segments.csv", report):
        seq = None
        try:
            seq = int(float(row["seq"])) if row.get("seq") else None
        except ValueError:
            pass
        segments.append(RawSegment(call_key=row["call_key"].lower(), file_name=row["file_name"],
                                   seq=seq, recorded_at=try_parse_dt(row.get("recorded_at"))))

    messages: list[RawMessage] = []
    for row in _rows(folder, "messages.csv", report):
        at = try_parse_dt(row.get("sent_at"))
        if at is None:
            continue
        messages.append(RawMessage(correspondence_id=row["correspondence_id"].upper(),
                                   message_id=row["message_id"], at=at,
                                   direction=row.get("direction", ""),
                                   subject=row.get("subject", ""), body=row.get("body", "")))
    return interactions, segments, messages


def import_contract(folder: str | Path, *, audio: str | Path | None = None,
                    redact_messages: bool = True) -> Built:
    folder = Path(folder)
    report = ImportReport(source=SOURCE)
    manifest = folder / "manifest.yaml"
    if manifest.exists():
        data = load_yaml(manifest) or {}
        version = int(data.get("contract_version", 1)) if isinstance(data, dict) else 1
        if version != 1:
            report.add("contract_version", "error",
                       f"contract version {version} is not supported (this reads v1)")
    if not (folder / "interactions.csv").exists():
        report.add("no_interactions", "error", "interactions.csv is missing")
    interactions, segments, messages = read_contract_rows(folder, report)
    source = AudioSource.open(audio) if audio else None
    return build_dataset(SOURCE, interactions, segments, messages, report, audio=source,
                         redact_messages=redact_messages)

