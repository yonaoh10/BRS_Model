"""The journey dataset: what every importer produces and every later stage reads.

A dataset is one batch of customer stories. An account never appears by its
number: at import it becomes `story_key` (a keyed digest, see pseudo.py) and
a running story number, and the number itself stays only in the private map.

Channels:
    call     a phone contact. `recorded` says whether audio exists; `answer`
             whether it was answered or abandoned (unknown without a calls
             table such as the bank's T4418 or the Atlas export).
    message  a written correspondence (a thread of messages).
    branch / other   a contact known only from a log line.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

CONTRACT_VERSION = 1

Channel = Literal["call", "message", "branch", "other"]
Direction = Literal["inbound", "outbound", "unknown"]
Answer = Literal["answered", "abandoned", "unknown"]
OpCategory = Literal["open", "info", "execute", "not_customer", "unclassified"]
Coverage = Literal["full", "partial", "none"]


class Segment(BaseModel):
    """One audio file of a call. `file_name` is the recorder's name, needed to
    find the file; it is never printed (recorder names are digit runs)."""

    seq: int
    file_name: str
    recorded_at: datetime | None = None


class CallAudio(BaseModel):
    call_key: str            # the source's call id, normalised (lower-case)
    call_id: str             # the pipeline's id for the call (safe, stable)
    segments: list[Segment] = Field(default_factory=list)
    missing_segments: list[int] = Field(default_factory=list)  # seqs with no file in the source

    @property
    def complete(self) -> bool:
        return bool(self.segments) and not self.missing_segments


class Interaction(BaseModel):
    interaction_id: str      # unique in the dataset
    story_key: str
    at: datetime
    channel: Channel
    recorded: bool = False
    direction: Direction = "unknown"
    answer: Answer = "unknown"
    call_key: str | None = None
    call_id: str | None = None
    correspondence_id: str | None = None
    talk_seconds: float | None = None
    banker_code: str | None = None
    unit_code: str | None = None
    source_row: int | None = None   # row in the source, for the audit trail only


class Message(BaseModel):
    message_id: str
    correspondence_id: str
    at: datetime
    direction: Direction
    subject: str = ""
    body: str = ""


class AtlasOp(BaseModel):
    at: datetime
    op_code: str
    op_category: OpCategory = "unclassified"
    description: str = ""


class AtlasSession(BaseModel):
    session_id: str
    story_key: str
    banker_code: str
    unit_code: str
    start: datetime
    end: datetime
    ops: list[AtlasOp] = Field(default_factory=list)
    n_ops: int = 0
    matched_interaction_id: str | None = None

    @property
    def minutes(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds() / 60.0)

    @property
    def has_execute(self) -> bool:
        return any(op.op_category == "execute" for op in self.ops)

    @property
    def view_only(self) -> bool:
        cats = {op.op_category for op in self.ops}
        return not ({"execute", "unclassified"} & cats)


class Story(BaseModel):
    story_key: str
    story_no: int
    branch: str | None = None
    first_at: datetime
    last_at: datetime
    declared_repeat_count: int | None = None
    atlas_coverage: Coverage = "none"
    source_labels: dict[str, str] = Field(default_factory=dict)  # e.g. an external status/topic


class ImportCounts(BaseModel):
    stories: int = 0
    interactions: int = 0
    returns: int = 0
    calls: int = 0
    recorded_calls: int = 0
    unrecorded_calls: int = 0
    correspondences: int = 0
    messages: int = 0
    audio_files_mapped: int = 0
    multi_file_calls: int = 0
    atlas_sessions: int = 0
    atlas_ops: int = 0


class ImportIssue(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    count: int = 1
    examples: list[str] = Field(default_factory=list)   # already safe to print


class ImportReport(BaseModel):
    source: str
    counts: ImportCounts = Field(default_factory=ImportCounts)
    issues: list[ImportIssue] = Field(default_factory=list)
    dropped_columns: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def add(self, code: str, severity: str, message: str, *, count: int = 1,
            examples: list[str] | None = None) -> None:
        self.issues.append(ImportIssue(code=code, severity=severity, message=message,
                                       count=count, examples=(examples or [])[:5]))


class JourneyDataset(BaseModel):
    contract_version: int = CONTRACT_VERSION
    dataset_id: str
    source: str
    created_at: datetime
    audio_source: str | None = None     # a folder or a ZIP holding the recordings
    stories: list[Story] = Field(default_factory=list)
    interactions: list[Interaction] = Field(default_factory=list)
    calls: dict[str, CallAudio] = Field(default_factory=dict)   # by call_key
    messages: list[Message] = Field(default_factory=list)
    atlas_sessions: list[AtlasSession] = Field(default_factory=list)
    report: ImportReport

    def story(self, story_key: str) -> Story:
        for s in self.stories:
            if s.story_key == story_key:
                return s
        raise KeyError(story_key)

    def interactions_of(self, story_key: str) -> list[Interaction]:
        return sorted((i for i in self.interactions if i.story_key == story_key),
                      key=lambda i: (i.at, i.interaction_id))
