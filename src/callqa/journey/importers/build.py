"""From source rows to a JourneyDataset: the part every importer shares.

Importers only read their shape into RawInteraction / RawSegment / RawMessage
rows. Everything that decides what the data means - which contacts belong to
one story, which files belong to one call and in what order, what is a
return, which ids are safe to show - happens here, once.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from callqa.journey.importers.common import (
    AudioSource,
    call_id_for,
    dataset_id,
    safe_ref,
    segment_parts,
    shown_file,
)
from callqa.journey.models import (
    CallAudio,
    CentreFacts,
    ImportReport,
    Interaction,
    JourneyDataset,
    Message,
    Segment,
    Story,
)
from callqa.journey.pseudo import BankerCodes, account_key, number_stories
from callqa.journey.timeparse import try_parse_dt

WRITTEN = ("message", "chat", "whatsapp")     # channels whose content is a thread of messages
_TRUE = {"1", "1.0", "yes", "y", "true", "כן"}
_FALSE = {"0", "0.0", "no", "n", "false", "לא"}


def _centre_facts(raw: dict[str, str]) -> CentreFacts | None:
    """The call-centre columns of a v2 row as the model carries them; codes
    stay text, flags become booleans, the matched time a datetime."""
    if not raw:
        return None

    def flag(key: str) -> bool | None:
        v = raw.get(key, "").strip().lower()
        return True if v in _TRUE else False if v in _FALSE else None

    def num(key: str) -> float | None:
        try:
            return float(raw[key]) if raw.get(key) else None
        except ValueError:
            return None

    return CentreFacts(
        direction_code=raw.get("direction_code") or None,
        churn_call_code=raw.get("churn_call_code") or None,
        call_status_code=raw.get("call_status_code") or None,
        call_source_code=raw.get("call_source_code") or None,
        employee_target_type_code=raw.get("employee_target_type_code") or None,
        cti_at=try_parse_dt(raw.get("cti_at")) if raw.get("cti_at") else None,
        match_gap_sec=num("match_gap_sec"), caller_is_owner=flag("caller_is_owner"),
        service_mode_code=raw.get("service_mode_code") or None,
        phone_meeting_ind=flag("phone_meeting_ind"), manui_moked_ind=flag("manui_moked_ind"),
        chat_id=raw.get("chat_id") or None, whatsapp_id=raw.get("whatsapp_id") or None,
        story_id=raw.get("story_id") or None)


@dataclass
class RawInteraction:
    row: int
    branch: str
    account: str
    at: datetime
    channel: str                 # call / message / branch / other
    source_id: str = ""          # call key or correspondence id
    file_name: str = ""
    declared_first_at: datetime | None = None
    declared_repeat_count: int | None = None
    direction: str = "unknown"
    answer: str = "unknown"
    talk_seconds: float | None = None
    banker_code: str | None = None
    unit_code: str | None = None
    recorded: bool | None = None
    facts: dict[str, str] = field(default_factory=dict)   # contract v2 FACT_COLUMNS, raw


@dataclass
class RawSegment:
    call_key: str
    file_name: str
    seq: int | None = None
    recorded_at: datetime | None = None


@dataclass
class RawMessage:
    correspondence_id: str
    message_id: str
    at: datetime
    direction: str
    subject: str = ""
    body: str = ""
    send_method: str = ""
    channel_msg_code: str | None = None
    template_code: str | None = None
    call_key: str | None = None
    banker_code: str | None = None
    unit_code: str | None = None


@dataclass
class Built:
    dataset: JourneyDataset
    private_rows: list[tuple[int, str, str, str]] = field(default_factory=list)
    bankers: BankerCodes = field(default_factory=BankerCodes)   # raw banker value -> B code


def _direction(value: str) -> str:
    v = (value or "").strip().lower()
    if v in ("inbound", "in", "i", "נכנס", "נכנסת"):
        return "inbound"
    if v in ("outbound", "out", "o", "יוצא", "יוצאת"):
        return "outbound"
    return "unknown"


DUPLICATE_WINDOW_SEC = 60.0


def _merge_duplicates(rows: list[RawInteraction]) -> tuple[list[RawInteraction], int]:
    """One row per contact. A row with the same account and the same call or
    correspondence id as an earlier row, within a minute of it, is that
    contact listed again (a SAS join that fanned out, a call listed once per
    recorded part): it is folded into the first row - which keeps what it
    had and takes what it lacked - and counted. Rows without an id are never
    merged: nothing says they are the same contact."""
    kept: list[RawInteraction] = []
    seen: dict[tuple[str, str, str], list[RawInteraction]] = {}
    merged = 0
    for raw in rows:
        if not raw.source_id:
            kept.append(raw)
            continue
        key = (raw.branch, raw.account, raw.source_id.lower())
        twin = next((k for k in seen.get(key, [])
                     if abs((k.at - raw.at).total_seconds()) <= DUPLICATE_WINDOW_SEC), None)
        if twin is None:
            seen.setdefault(key, []).append(raw)
            kept.append(raw)
            continue
        merged += 1
        for name in ("file_name", "declared_first_at", "declared_repeat_count", "talk_seconds",
                     "banker_code", "unit_code"):
            if getattr(twin, name) in (None, "") and getattr(raw, name) not in (None, ""):
                setattr(twin, name, getattr(raw, name))
        if twin.direction == "unknown" and raw.direction != "unknown":
            twin.direction = raw.direction
        if twin.answer == "unknown" and raw.answer != "unknown":
            twin.answer = raw.answer
        if raw.recorded:
            twin.recorded = True
    return kept, merged


def build_dataset(source: str, interactions: list[RawInteraction], segments: list[RawSegment],
                  messages: list[RawMessage], report: ImportReport, *,
                  audio: AudioSource | None = None, redact_messages: bool = True,
                  created: datetime | None = None, bankers: BankerCodes | None = None) -> Built:
    """`bankers` is the registry of running banker codes for this batch; the
    same one is handed to the Atlas attachment so one banker is one code
    across the layers. Omitted = a fresh one."""
    created = created or datetime.now(UTC).replace(tzinfo=None)
    bankers = bankers if bankers is not None else BankerCodes()

    # --- calls and their files -------------------------------------------
    by_call: dict[str, list[RawSegment]] = defaultdict(list)
    for seg in segments:
        by_call[seg.call_key.lower()].append(seg)
    calls: dict[str, CallAudio] = {}
    mixed_keys = 0
    for key, segs in by_call.items():
        # order: an explicit seq, else the NICE segment key, else the row order
        def order(s: RawSegment, idx: int) -> tuple:
            parts = segment_parts(s.file_name)
            return (s.seq if s.seq is not None else (parts[1] if parts else idx), idx)
        ordered = [s for _i, s in sorted(((order(s, i), s) for i, s in enumerate(segs)),
                                         key=lambda t: t[0])]
        nice_keys = {p[0] for s in ordered if (p := segment_parts(s.file_name))}
        if len(nice_keys) > 1:
            mixed_keys += 1
        seen_files: set[str] = set()
        out: list[Segment] = []
        for s in ordered:
            if s.file_name.lower() in seen_files:
                continue
            seen_files.add(s.file_name.lower())
            out.append(Segment(seq=len(out) + 1, file_name=s.file_name, recorded_at=s.recorded_at))
        calls[key] = CallAudio(call_key=key, call_id=call_id_for(key), segments=out)
    if mixed_keys:
        report.add("segment_key_mismatch", "warning",
                   "the files mapped to one call carry different recorder call keys "
                   "in their names; the mapping table was followed", count=mixed_keys)

    # --- interactions and stories ------------------------------------------
    interactions, duplicates = _merge_duplicates(interactions)
    if duplicates:
        report.add("duplicate_contacts", "warning",
                   "rows that repeat a contact already listed (same account, same call or "
                   "correspondence id, within a minute) - a join that fanned out, or a call "
                   "listed once per part; counted once", count=duplicates)
    story_of: dict[tuple[str, str], str] = {}
    branch_of: dict[str, str] = {}
    private: dict[str, tuple[str, str]] = {}
    out_interactions: list[Interaction] = []
    used_ids: set[str] = set()
    declared: dict[str, int | None] = {}
    declared_first: dict[str, datetime] = {}
    file_only = 0
    for raw in interactions:
        skey = story_of.setdefault((raw.branch, raw.account), account_key(raw.branch, raw.account))
        branch_of.setdefault(skey, raw.branch)
        private.setdefault(skey, (raw.branch, raw.account))
        if raw.declared_repeat_count is not None:
            declared.setdefault(skey, raw.declared_repeat_count)
        if raw.declared_first_at is not None:
            declared_first.setdefault(skey, raw.declared_first_at)
        call_key = corr = None
        call_id = None
        recorded = bool(raw.recorded)
        if raw.channel == "call" and raw.source_id:
            call_key = raw.source_id.lower()
            call_id = call_id_for(call_key)
            if call_key in calls:
                recorded = True
            elif raw.file_name:
                # a file named on the row but absent from the mapping table
                calls[call_key] = CallAudio(call_key=call_key, call_id=call_id, segments=[
                    Segment(seq=1, file_name=raw.file_name)])
                recorded = True
                file_only += 1
            base = call_id
        elif raw.channel in WRITTEN:
            corr = raw.source_id
            base = safe_ref("um" if raw.channel == "message" else raw.channel[:4],
                            raw.source_id or f"row{raw.row}")
        else:
            base = safe_ref(raw.channel[:3] or "int", raw.source_id or f"row{raw.row}")
        iid = base
        n = 2
        while iid in used_ids:
            iid = f"{base}-{n}"
            n += 1
        used_ids.add(iid)
        out_interactions.append(Interaction(
            interaction_id=iid, story_key=skey, at=raw.at, channel=raw.channel,
            recorded=recorded if raw.channel == "call" else raw.channel in WRITTEN,
            direction=raw.direction, answer=raw.answer, call_key=call_key,
            call_id=call_id if recorded else None, correspondence_id=corr,
            talk_seconds=raw.talk_seconds, banker_code=bankers.code_for(raw.banker_code),
            unit_code=raw.unit_code, source_row=raw.row, facts=_centre_facts(raw.facts)))
    if file_only:
        report.add("file_not_in_mapping", "warning",
                   "calls whose file is named on the interaction row but missing from "
                   "the call-to-files table; the single named file was used",
                   count=file_only)

    # --- messages ----------------------------------------------------------
    out_messages: list[Message] = []
    for m in messages:
        body, subject = m.body, m.subject
        if redact_messages:
            from callqa.redaction import redact_text
            body, _ = redact_text(body)
            subject, _ = redact_text(subject)
        out_messages.append(Message(
            message_id=m.message_id, correspondence_id=m.correspondence_id, at=m.at,
            direction=_direction(m.direction), subject=subject, body=body,
            send_method=m.send_method if m.send_method in ("MAN", "AUTO") else "",
            channel_msg_code=m.channel_msg_code, template_code=m.template_code,
            call_key=m.call_key, banker_code=bankers.code_for(m.banker_code),
            unit_code=m.unit_code))
    corr_first_dir: dict[str, str] = {}
    for m in sorted(out_messages, key=lambda m: m.at):
        corr_first_dir.setdefault(m.correspondence_id.upper(), m.direction)
    for i in out_interactions:
        if i.channel in WRITTEN and i.direction == "unknown" and i.correspondence_id:
            i.direction = corr_first_dir.get(i.correspondence_id.upper(), "unknown")

    # --- audio presence ----------------------------------------------------
    if audio is not None:
        referenced: set[str] = set()
        missing_calls = 0
        for call in calls.values():
            for seg in call.segments:
                member = audio.find(seg.file_name)
                if member is None:
                    call.missing_segments.append(seg.seq)
                else:
                    referenced.add(member)
            if call.missing_segments:
                missing_calls += 1
        if missing_calls:
            report.add("audio_missing", "warning",
                       "calls with at least one mapped file absent from the audio source; "
                       "they are listed and not transcribed", count=missing_calls)
        orphans = [f for f in audio.files() if f not in referenced]
        if orphans:
            hints = []
            known = {segment_parts(s.file_name)[0]: c.call_id for c in calls.values()
                     for s in c.segments if segment_parts(s.file_name)}
            for f in orphans:
                parts = segment_parts(f)
                hint = f" (possibly part of call {known[parts[0]]})" if parts and parts[0] in known else ""
                hints.append(shown_file(f) + hint)
            report.add("audio_orphans", "warning",
                       "audio files in the source that no call refers to; not used",
                       count=len(orphans), examples=hints)

    # --- stories -----------------------------------------------------------
    by_story: dict[str, list[Interaction]] = defaultdict(list)
    for i in out_interactions:
        by_story[i.story_key].append(i)
    first = {k: min(i.at for i in v) for k, v in by_story.items()}
    numbers = number_stories(first)
    stories: list[Story] = []
    mismatch = []
    first_mismatch = 0
    for key, items in by_story.items():
        dec = declared.get(key)
        if dec is not None and dec != len(items) - 1:
            mismatch.append(numbers[key])
        dfirst = declared_first.get(key)
        if dfirst is not None and abs((dfirst - first[key]).total_seconds()) > 60:
            first_mismatch += 1
        stories.append(Story(story_key=key, story_no=numbers[key], branch=branch_of.get(key),
                             first_at=first[key], last_at=max(i.at for i in items),
                             declared_repeat_count=dec))
    stories.sort(key=lambda s: s.story_no)
    if mismatch:
        report.add("declared_repeats_differ", "info",
                   "stories whose declared repeat count is not their contacts minus one "
                   "(the contacts themselves were counted)", count=len(mismatch),
                   examples=[f"סיפור {n:03d}" for n in sorted(mismatch)])
    if first_mismatch:
        report.add("declared_first_differs", "info",
                   "stories whose declared first-contact time is not their earliest contact",
                   count=first_mismatch)

    c = report.counts
    c.stories = len(stories)
    c.interactions = len(out_interactions)
    c.returns = len(out_interactions) - len(stories)
    c.calls = sum(1 for i in out_interactions if i.channel == "call")
    c.recorded_calls = sum(1 for i in out_interactions if i.channel == "call" and i.recorded)
    c.unrecorded_calls = c.calls - c.recorded_calls
    c.correspondences = sum(1 for i in out_interactions if i.channel == "message")
    c.chats = sum(1 for i in out_interactions if i.channel in ("chat", "whatsapp"))
    c.messages = len(out_messages)
    c.audio_files_mapped = sum(len(call.segments) for call in calls.values())
    c.multi_file_calls = sum(1 for call in calls.values() if len(call.segments) > 1)
    c.bankers_rewritten = bankers.rewritten
    if bankers.name_like:
        report.add("banker_names_rejected", "warning",
                   "banker values that look like a person's name or user name; each was "
                   "replaced by a running code (the mapping stays in the private folder) - "
                   "the export should carry codes, not names", count=bankers.name_like)

    ds_id = dataset_id(created.strftime("%Y%m%d"),
                       sorted(f"{i.story_key}|{i.at.isoformat()}|{i.interaction_id}"
                              for i in out_interactions))
    dataset = JourneyDataset(
        dataset_id=ds_id, source=source, created_at=created,
        audio_source=str(audio.path) if audio else None,
        stories=stories, interactions=sorted(out_interactions, key=lambda i: (i.at, i.interaction_id)),
        calls=calls, messages=sorted(out_messages, key=lambda m: (m.at, m.message_id)),
        report=report)
    rows = [(numbers[k], k, b, a) for k, (b, a) in private.items()]
    return Built(dataset=dataset, private_rows=rows, bankers=bankers)
