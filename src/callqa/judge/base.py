"""Judge protocol and request container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from callqa.models import Features, RedactedTranscript
from callqa.rubric import RubricDimension


@dataclass
class JudgeRequest:
    """One judging call: prompts plus the structured context they were built from."""

    call_id: str
    system_prompt: str
    user_prompt: str
    dimensions: list[RubricDimension]
    redacted: RedactedTranscript
    features: Features
    attempt: int = 0  # retry attempt index (validation errors appended to prompt)


@runtime_checkable
class Judge(Protocol):
    name: str

    def complete(self, request: JudgeRequest) -> str:
        """Return the raw model output (expected to be strict JSON)."""
        ...
