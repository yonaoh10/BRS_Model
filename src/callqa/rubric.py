"""Rubric loading (config/rubric.yaml) and scorecard math (weights + gate rule)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from callqa.models import DimensionScore

GATE_CAP = 59.0


class RubricDimension(BaseModel):
    id: str
    name_he: str
    weight: float = Field(gt=0, le=1)
    scale: str = "1-5"
    gate: bool = False
    signal: str = "llm"  # llm | feature | hybrid
    features: list[str] = Field(default_factory=list)
    anchors: dict[str, str]

    @field_validator("anchors")
    @classmethod
    def _anchors_complete(cls, v: dict[str, str]) -> dict[str, str]:
        missing = {"1", "3", "5"} - set(v)
        if missing:
            raise ValueError(f"rubric anchors missing levels: {sorted(missing)}")
        return v


class Rubric(BaseModel):
    version: str
    dimensions: list[RubricDimension]

    @field_validator("dimensions")
    @classmethod
    def _weights_sum_to_one(cls, v: list[RubricDimension]) -> list[RubricDimension]:
        total = sum(d.weight for d in v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rubric weights must sum to 1.0, got {total}")
        return v

    @property
    def by_id(self) -> dict[str, RubricDimension]:
        return {d.id: d for d in self.dimensions}

    @property
    def gate_ids(self) -> list[str]:
        return [d.id for d in self.dimensions if d.gate]


def load_rubric(path: str | Path = "config/rubric.yaml") -> Rubric:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Rubric.model_validate(data)


def weighted_total(rubric: Rubric, scores: dict[str, DimensionScore]) -> tuple[float, bool, list[str]]:
    """Compute the weighted 0-100 total and apply the gate rule.

    Scores 1..5 map linearly to 0..100 per dimension ((score-1)/4*100).
    Any gate dimension scored <= 2 caps the total at 59 and sets gate_failed.
    Returns (total, gate_failed, failed_gate_ids).
    """
    missing = [d.id for d in rubric.dimensions if d.id not in scores]
    if missing:
        raise ValueError(f"scores missing rubric dimensions: {missing}")
    total = sum(
        d.weight * (scores[d.id].score - 1) / 4.0 * 100.0 for d in rubric.dimensions
    )
    failed_gates = [d.id for d in rubric.dimensions if d.gate and scores[d.id].score <= 2]
    gate_failed = bool(failed_gates)
    if gate_failed:
        total = min(total, GATE_CAP)
    return round(total, 1), gate_failed, failed_gates
