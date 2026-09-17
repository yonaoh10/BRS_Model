"""Schemas for the NEW sidecar artifacts the ops layer writes. These are
additional records beside the pipeline's own artifacts — no existing stage
contract in callqa.models changes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CallRunSummary(BaseModel):
    call_id: str
    status: str
    error: str | None = None
    seconds: float = 0.0                                   # end-to-end wall time
    stage_seconds: dict[str, float] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """One machine-readable record per batch: what produced it and how it went."""

    run_id: str
    command: str
    started_at: str
    finished_at: str | None = None
    fingerprint: dict = Field(default_factory=dict)        # ops.provenance.fingerprint
    counts: dict[str, int] = Field(default_factory=dict)   # status -> n
    cost: dict = Field(default_factory=dict)               # gpu_hours / rate / estimated
    calls: list[CallRunSummary] = Field(default_factory=list)


class CallRunLink(BaseModel):
    """Links a call to the run that produced it, carrying that run's fingerprint
    so `callqa verify` can compare it to the current environment."""

    call_id: str
    run_id: str
    fingerprint: dict = Field(default_factory=dict)
