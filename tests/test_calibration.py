"""Unit tests: QWK against hand-computed values + calibration plumbing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from callqa.calibration import (
    MIN_CALLS_FOR_PASS,
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


def test_qwk_is_undefined_on_constant_ratings() -> None:
    """Kappa measures agreement above chance. With no variance there is no
    chance to beat, so "perfect agreement" is not an answer that can be
    given - and reporting 1.0 let a rater who always says 3 certify the judge."""
    assert _safe_qwk([3, 3, 3], [3, 3, 3]) is None
    assert _safe_qwk([3, 3, 3], [4, 4, 4]) is None
    assert _safe_qwk([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


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
    assert not result.overall_pass, "three calls is not a calibration"
    assert result.n_doubly_rated == 1
    # Both raters gave the same constant score, so their agreement is not
    # measurable either - reported as unknown rather than as perfect.
    assert result.human_vs_human_qwk is None
    # Confusion matrices place everything on the diagonal.
    for d in result.dimensions:
        assert sum(d.confusion[i][i] for i in range(5)) == d.n
        assert d.mae == 0.0


def test_a_calibration_needs_enough_calls_to_mean_anything(rubric) -> None:
    """Perfect agreement over a handful of calls is not evidence. The PASS
    verdict is what tells a bank the judge may be trusted, so it needs a
    sample, agreement on the whole set AND on every dimension."""
    dim_ids = [d.id for d in rubric.dimensions]
    scores = [1, 2, 3, 4, 5] * 5
    cards = [make_card(rubric, f"C{i}", dict.fromkeys(dim_ids, s))
             for i, s in enumerate(scores)]
    ratings = {f"C{i}": [dict.fromkeys(dim_ids, s)] for i, s in enumerate(scores)}

    result = calibrate(cards, ratings, rubric)

    assert result.n_calls == len(scores) >= MIN_CALLS_FOR_PASS
    assert result.overall_qwk == 1.0
    assert result.mean_dimension_qwk == 1.0
    assert result.overall_pass


def test_one_broken_dimension_fails_the_whole_calibration(rubric) -> None:
    """Pooling across dimensions hides the one that disagrees: eight kappas
    below 0.25 still pooled to 0.90 because the overall LEVEL was right."""
    dim_ids = [d.id for d in rubric.dimensions]
    broken = dim_ids[0]
    scores = [1, 2, 3, 4, 5] * 5
    cards = [make_card(rubric, f"C{i}", dict.fromkeys(dim_ids, s))
             for i, s in enumerate(scores)]
    ratings = {
        f"C{i}": [{d: (6 - s if d == broken else s) for d in dim_ids}]
        for i, s in enumerate(scores)
    }

    result = calibrate(cards, ratings, rubric)

    assert broken in result.flagged_dimensions
    assert not result.overall_pass


def test_calibrate_no_overlap_raises(rubric) -> None:
    cards = [make_card(rubric, "C1", {})]
    with pytest.raises(CalibrationError, match="no calls present in both"):
        calibrate(cards, {"OTHER": [{}]}, rubric)


def test_scorecards_from_different_rubrics_are_not_pooled(rubric) -> None:
    """A score under an old rubric is not comparable to one under a new rubric.
    aggregation and calibration must keep only the newest cohort, never average
    or QWK across a rubric/prompt change."""
    from callqa.aggregation import aggregate_bankers, single_rubric_cohort

    dim_ids = [d.id for d in rubric.dimensions]
    old = make_card(rubric, "OLD", dict.fromkeys(dim_ids, 5)).model_copy(
        update={"rubric_sha256": "OLDSHA", "timestamp": "2020-01-01T00:00:00+00:00"})
    new = make_card(rubric, "NEW", dict.fromkeys(dim_ids, 1)).model_copy(
        update={"rubric_sha256": "NEWSHA", "timestamp": "2026-01-01T00:00:00+00:00"})

    assert [c.call_id for c in single_rubric_cohort([old, new])] == ["NEW"]

    aggs, _, _ = aggregate_bankers([old, new], rubric)
    assert aggs["B1"].n_calls == 1, "the old-rubric card must not be pooled in"

    # calibrate drops the old cohort, so a rating that only covers the old call
    # leaves nothing in common → it must refuse rather than certify on stale data
    with pytest.raises(CalibrationError):
        calibrate([old, new], {"OLD": [dict.fromkeys(dim_ids, 5)]}, rubric)


def test_a_constant_gate_dimension_blocks_pass(rubric) -> None:
    """A judge that always outputs the same score for a gate dimension has zero
    discriminative power: its per-dimension kappa is undefined. It must be
    flagged (and block PASS), not silently treated as agreement."""
    dim_ids = [d.id for d in rubric.dimensions]
    cards, ratings = [], {}
    # 25 calls; humans span 1..5 on every dim. The judge is perfect on all dims
    # EXCEPT the gate dim 'identification', where it always answers 3.
    for i in range(25):
        h = (i % 5) + 1
        cards.append(make_card(rubric, f"C{i}",
                               {d: (3 if d == "identification" else h) for d in dim_ids}))
        ratings[f"C{i}"] = [dict.fromkeys(dim_ids, h)]

    result = calibrate(cards, ratings, rubric)
    ident = next(d for d in result.dimensions if d.dimension_id == "identification")
    assert ident.qwk is None and (ident.mae or 0) > 0
    assert ident.flagged, "a constant, discriminative-power-zero gate dim must be flagged"
    assert "identification" in result.flagged_dimensions
    assert result.overall_pass is False
