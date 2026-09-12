"""Pydantic data models for every pipeline artifact.

Each pipeline stage reads the previous stage's JSON artifact and writes its
own; all artifacts are validated by the models below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Speaker = Literal["banker", "customer"]
CallStatus = Literal["success", "needs_human_review", "failed"]

# Exit codes for the `process` command (per spec).
EXIT_SUCCESS = 0
EXIT_NEEDS_HUMAN_REVIEW = 1
EXIT_FAILED = 2

STATUS_EXIT_CODES: dict[str, int] = {
    "success": EXIT_SUCCESS,
    "needs_human_review": EXIT_NEEDS_HUMAN_REVIEW,
    "failed": EXIT_FAILED,
}


class CallInput(BaseModel):
    """The input to process_call: one recording plus its metadata."""

    call_id: str
    audio_path: Path
    banker_id: str = "unknown"
    call_date: str | None = None
    call_type: str | None = None
    banker_channel: Literal["L", "R"] | None = None
    banker_name: str | None = None  # used only for name redaction; never reported


class CallMeta(BaseModel):
    """Stage 1 (ingestion) output: probed audio metadata."""

    call_id: str
    banker_id: str
    file_name: str
    duration_sec: float
    channels: int
    sample_rate: int
    codec: str | None = None
    call_date: str | None = None
    call_type: str | None = None
    banker_channel: Literal["L", "R"] | None = None


class VADSegment(BaseModel):
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class AudioArtifact(BaseModel):
    """Stage 2 (audio) output: converted WAV paths + VAD speech segments."""

    call_id: str
    is_stereo: bool
    banker_wav: str | None = None      # stereo path only
    customer_wav: str | None = None    # stereo path only
    mono_wav: str | None = None        # mono path only
    sample_rate: int
    vad_engine: str
    # How the file's channels were actually used. A file can declare two
    # channels and still carry one recording (see audio.probe_channels).
    channel_layout: Literal["stereo", "dual_mono", "single_channel", "mono"] = "mono"
    banker_segments: list[VADSegment] = Field(default_factory=list)
    customer_segments: list[VADSegment] = Field(default_factory=list)
    mono_segments: list[VADSegment] = Field(default_factory=list)


class Word(BaseModel):
    word: str
    start: float
    end: float
    probability: float | None = None


class TranscriptSegment(BaseModel):
    speaker: Speaker | None = None
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)
    avg_logprob: float | None = None


class TranscriptQuality(BaseModel):
    low_confidence_segments: list[int] = Field(default_factory=list)
    mean_logprob: float | None = None
    low_confidence_ratio: float = 0.0


class Transcript(BaseModel):
    """Stage 3 (asr) output. RAW text - lives only under output/transcripts/."""

    call_id: str
    language: str
    engine: str
    segments: list[TranscriptSegment]
    quality: TranscriptQuality = TranscriptQuality()


class TranscriptBundle(BaseModel):
    """ASR stage output: per-channel transcripts (stereo) or the mono mix."""

    call_id: str
    banker: Transcript | None = None
    customer: Transcript | None = None
    mono: Transcript | None = None


class DialogTurn(BaseModel):
    speaker: Speaker
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class RoleSignalRecord(BaseModel):
    """One piece of evidence behind the banker/customer decision."""

    name: str
    weight: float
    votes_for: int
    detail: str


class DiarizationQualityRecord(BaseModel):
    """What the diarizer produced and what had to be cleaned up."""

    speakers_found: int = 2
    reassigned_sec: float = 0.0
    words_attributed: int = 0
    words_by_nearest: int = 0
    smoothed_islands: int = 0


class DialogTranscript(BaseModel):
    """Stage 4 (speakers) output: merged, time-ordered dialog. RAW text."""

    call_id: str
    attribution_mode: Literal["stereo", "mono_diarized", "mock"]
    # 1.0 on a stereo recording, where the roles are known rather than
    # inferred. On the mono path, 0.0 means the signals were split evenly.
    role_confidence: float = 1.0
    turns: list[DialogTurn]
    role_signals: list[RoleSignalRecord] = Field(default_factory=list)
    diarization: DiarizationQualityRecord | None = None


class RedactedTurn(BaseModel):
    speaker: Speaker
    start: float
    end: float
    text: str


class RedactedTranscript(BaseModel):
    """Stage 5 (redaction) output: the ONLY text allowed to reach the judge,
    reports, or logs."""

    call_id: str
    engine: str
    turns: list[RedactedTurn]
    redaction_counts: dict[str, int] = Field(default_factory=dict)


class SpeechRateWPM(BaseModel):
    banker: float = 0.0
    customer: float = 0.0


class Features(BaseModel):
    """Stage 6 output: objective conversational features (field names per spec)."""

    call_id: str
    talk_ratio: float
    longest_banker_monologue_sec: float
    interruptions_by_banker: int
    interruptions_by_customer: int
    patience_median_sec: float | None
    banker_question_count: int
    banker_questions_per_minute: float
    speech_rate_wpm: SpeechRateWPM
    dead_air_total_sec: float
    call_duration_sec: float
    banker_speech_sec: float
    customer_speech_sec: float


class Evidence(BaseModel):
    quote: str
    timestamp: str  # "mm:ss"
    speaker: Speaker


class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning_he: str
    evidence: list[Evidence] = Field(default_factory=list)


class JudgeResponse(BaseModel):
    """The strict JSON schema the LLM judge must produce."""

    scores: dict[str, DimensionScore]
    strengths_he: list[str] = Field(default_factory=list)
    development_area_he: str = ""
    summary_he: str = ""


class ScoreCard(BaseModel):
    """Stage 7 output: per-call scores + full audit trail."""

    call_id: str
    banker_id: str
    scores: dict[str, DimensionScore]
    weighted_total: float = Field(ge=0, le=100)
    gate_failed: bool = False
    failed_gates: list[str] = Field(default_factory=list)
    strengths_he: list[str] = Field(default_factory=list)
    development_area_he: str = ""
    summary_he: str = ""
    # Audit / determinism fields
    judge_engine: str
    model: str
    prompt_sha256: str
    prompt_version: str
    retries: int = 0
    latency_sec: float = 0.0
    n_samples: int = 1
    truncated: bool = False
    timestamp: str  # ISO-8601 UTC


class CallResult(BaseModel):
    """The status envelope every call ends in (returned by process_call)."""

    call_id: str
    status: CallStatus
    error: str | None = None
    stages_completed: list[str] = Field(default_factory=list)
    report_path: str | None = None
    scorecard_path: str | None = None

    @property
    def exit_code(self) -> int:
        return STATUS_EXIT_CODES[self.status]
