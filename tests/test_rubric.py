"""Unit tests: rubric loading, weighted total, and the gate-cap rule."""

from __future__ import annotations

import pytest

from callqa.models import DimensionScore
from callqa.rubric import load_rubric, weighted_total


@pytest.fixture(scope="module")
def rubric():
    return load_rubric("config/rubric.yaml")


def make_scores(rubric, default: int = 3, **overrides: int) -> dict[str, DimensionScore]:
    return {
        d.id: DimensionScore(score=overrides.get(d.id, default), reasoning_he="בדיקה")
        for d in rubric.dimensions
    }


def test_rubric_loads_with_eight_dimensions(rubric) -> None:
    assert len(rubric.dimensions) == 8
    assert abs(sum(d.weight for d in rubric.dimensions) - 1.0) < 1e-9
    assert rubric.gate_ids == ["identification", "compliance"]


def test_weighted_total_all_threes(rubric) -> None:
    total, gate_failed, failed = weighted_total(rubric, make_scores(rubric, default=3))
    # score 3 maps to (3-1)/4*100 = 50 in every dimension -> total 50.
    assert total == 50.0
    assert not gate_failed and failed == []


def test_weighted_total_all_fives(rubric) -> None:
    total, gate_failed, _ = weighted_total(rubric, make_scores(rubric, default=5))
    assert total == 100.0
    assert not gate_failed


def test_gate_cap(rubric) -> None:
    scores = make_scores(rubric, default=5, identification=2)
    total, gate_failed, failed = weighted_total(rubric, scores)
    # Without the gate the total would be 100 - 0.15*75 = 88.75 -> capped at 59.
    assert total == 59.0
    assert gate_failed and failed == ["identification"]


def test_gate_not_triggered_at_three(rubric) -> None:
    scores = make_scores(rubric, default=5, compliance=3)
    total, gate_failed, _ = weighted_total(rubric, scores)
    assert not gate_failed
    assert total == 92.5  # 100 - 0.15 * 50


def test_gate_cap_only_lowers_never_raises(rubric) -> None:
    # All dimensions terrible AND a gate failed: total stays below the cap.
    scores = make_scores(rubric, default=1, identification=1)
    total, gate_failed, _ = weighted_total(rubric, scores)
    assert gate_failed
    assert total == 0.0


def test_missing_dimension_raises(rubric) -> None:
    scores = make_scores(rubric)
    scores.pop("closure")
    with pytest.raises(ValueError, match="missing"):
        weighted_total(rubric, scores)
