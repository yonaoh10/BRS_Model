"""Unit tests: QWK against hand-computed values + calibration plumbing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from callqa.calibration import (
    CalibrationError,
    _mean_rounded,
    _safe_qwk,
    calibrate,
)
from callqa.models import DimensionScore, ScoreCard
from callqa.rubric import load_rubric


@pytest.fixture(scope="module")
def rubric():
    return load_rubric("config/rubric.yaml")


def test_qwk_perfect_agreement() -> None:
    assert _safe_qwk([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_qwk_hand_computed_reversal() -> None:
    # For ratings [1,2,3] vs [3,2,1]:
    # observed weighted disagreement 0.5, expected 0.25 -> QWK = 1 - 0.5/0.25 = -1.
    assert _safe_qwk([1, 2, 3], [3, 2, 1]) == -1.0


def test_qwk_hand_computed_partial() -> None:
    # QWK is symmetric and penalizes distant disagreements quadratically:
    # a one-step disagreement on one item out of many stays close to 1.
    a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5, 1, 2, 3, 4, 4]
    value = _safe_qwk(a, b)
    assert value is not None and 0.9 < value < 1.0


def test_qwk_degenerate_constant() -> None:
    assert _safe_qwk([3, 3, 3], [3, 3, 3]) == 1.0


def test_mean_rounded() -> None:
    assert _mean_rounded([3, 4]) == 4  # 3.5 rounds to 4
    assert _mean_rounded([3, 3, 4]) == 3


def make_card(rubric, call_id: str, scores: dict[str, int], banker: str = "B1") -> ScoreCard:
    return ScoreCard(
        call_id=call_id,
        banker_id=banker,
        scores={
            d.id: DimensionScore(score=scores.get(d.id, 3), reasoning_he="x")
            for d in rubric.dimensions
        },
        weighted_total=50.0,
        judge_engine="mock",
        model="mock",
        prompt_sha256="0" * 64,
        prompt_version="1.0.0",
        timestamp=datetime.now(UTC).isoformat(),
    )


def test_calibrate_end_to_end(rubric) -> None:
    dim_ids = [d.id for d in rubric.dimensions]
    cards = [
        make_card(rubric, "C1", dict.fromkeys(dim_ids, 4)),
        make_card(rubric, "C2", dict.fromkeys(dim_ids, 2)),
        make_card(rubric, "C3", dict.fromkeys(dim_ids, 5)),
    ]
    ratings = {
        "C1": [dict.fromkeys(dim_ids, 4), dict.fromkeys(dim_ids, 4)],
        "C2": [dict.fromkeys(dim_ids, 2)],
        "C3": [dict.fromkeys(dim_ids, 5)],
    }
    result = calibrate(cards, ratings, rubric)
    assert result.n_calls == 3
    assert result.overall_qwk == 1.0
    assert result.overall_pass
    assert result.n_doubly_rated == 1
    assert result.human_vs_human_qwk == 1.0
    # Confusion matrices place everything on the diagonal.
    for d in result.dimensions:
        assert sum(d.confusion[i][i] for i in range(5)) == d.n
        assert d.mae == 0.0


def test_calibrate_no_overlap_raises(rubric) -> None:
    cards = [make_card(rubric, "C1", {})]
    with pytest.raises(CalibrationError, match="no calls present in both"):
        calibrate(cards, {"OTHER": [{}]}, rubric)
