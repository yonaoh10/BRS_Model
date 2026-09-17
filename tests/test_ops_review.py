"""The review feedback loop: held calls -> reviewer verdict -> calibration set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from callqa.calibration import load_human_ratings
from callqa.ops.review import ReviewError, pending_reviews, record_review

DIMS = ["identification", "compliance", "empathy", "listening",
        "clarity", "resolution", "suitability", "closure"]
_FULL = dict.fromkeys(DIMS, 3)


def _result(output_dir: Path, call_id: str, status: str) -> None:
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "results" / f"{call_id}.json").write_text(
        json.dumps({"call_id": call_id, "status": status}), encoding="utf-8")


def test_pending_lists_held_calls_without_a_verdict(tmp_path: Path) -> None:
    out, ratings = tmp_path / "output", tmp_path / "human_ratings.csv"
    _result(out, "H1", "needs_human_review")
    _result(out, "H2", "needs_human_review")
    _result(out, "OK1", "success")
    assert pending_reviews(out, ratings) == ["H1", "H2"]

    record_review(ratings, "H1", "reviewer-a", _FULL, DIMS)
    assert pending_reviews(out, ratings) == ["H2"]           # H1 now has a verdict


def test_a_recorded_verdict_is_consumed_by_calibration(tmp_path: Path) -> None:
    ratings = tmp_path / "human_ratings.csv"
    scores = {"identification": 4, "compliance": 2, "empathy": 5, "listening": 3,
              "clarity": 4, "resolution": 5, "suitability": 2, "closure": 3}
    record_review(ratings, "CALLX", "reviewer-a", scores, DIMS)

    parsed = load_human_ratings(ratings, DIMS)
    assert "CALLX" in parsed
    assert parsed["CALLX"][0] == scores                       # exactly what calibrate will use


def test_a_second_rater_is_appended_not_rejected(tmp_path: Path) -> None:
    ratings = tmp_path / "human_ratings.csv"
    record_review(ratings, "CALLX", "reviewer-a", _FULL, DIMS)
    record_review(ratings, "CALLX", "reviewer-b", dict.fromkeys(DIMS, 4), DIMS)
    parsed = load_human_ratings(ratings, DIMS)
    assert len(parsed["CALLX"]) == 2                          # two independent raters


def test_bad_verdicts_are_rejected(tmp_path: Path) -> None:
    ratings = tmp_path / "human_ratings.csv"
    with pytest.raises(ReviewError, match="missing"):
        record_review(ratings, "C", "r", {"identification": 3}, DIMS)       # incomplete
    with pytest.raises(ReviewError, match="1..5"):
        record_review(ratings, "C", "r", {**_FULL, "clarity": 9}, DIMS)     # out of range
    with pytest.raises(ReviewError, match="unknown"):
        record_review(ratings, "C", "r", {**_FULL, "bogus": 3}, DIMS)       # bad dimension

    record_review(ratings, "C", "r", _FULL, DIMS)
    with pytest.raises(ReviewError, match="already reviewed"):
        record_review(ratings, "C", "r", _FULL, DIMS)                       # duplicate
