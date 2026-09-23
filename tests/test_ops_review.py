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


def test_a_verdict_into_an_empty_ratings_file_is_not_swallowed(tmp_path: Path) -> None:
    """An existing-but-empty human_ratings.csv took the append path without
    writing a header, so csv.DictReader read the first verdict AS the header:
    the reviewer's judgement disappeared and every later row was misaligned.
    A reviewer's verdict on a held call is the most valuable thing this system
    produces, and it was being lost in the most silent way possible."""
    import csv

    from callqa.ops.review import record_review

    ratings = tmp_path / "calibration" / "human_ratings.csv"
    ratings.parent.mkdir(parents=True)
    ratings.write_text("", encoding="utf-8")        # touched, never written to

    dims = _dimension_ids()
    record_review(ratings, "C1", "rater-1", {d: 3 for d in dims}, dims)

    rows = list(csv.DictReader(ratings.open(encoding="utf-8-sig")))
    assert len(rows) == 1, f"the verdict was lost: {ratings.read_text()!r}"
    assert rows[0]["call_id"] == "C1"
    assert rows[0]["rater_id"] == "rater-1"


def _dimension_ids() -> list[str]:
    from callqa.rubric import load_rubric

    return [d.id for d in load_rubric().dimensions]


def test_a_score_for_a_dimension_the_file_lacks_is_refused_not_dropped(tmp_path: Path) -> None:
    """After a rubric gains a dimension, the existing file's header has no
    column for it; the verdict was written without that score while the CLI
    printed "recorded". The reviewer's judgement is the scarce input - refuse
    loudly rather than keep part of it."""
    ratings = tmp_path / "human_ratings.csv"
    ratings.write_text("call_id,rater_id,a,b\nC0,r,3,3\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="new_dim"):
        record_review(ratings, "C1", "r", {"a": 4, "b": 4, "new_dim": 5}, ["a", "b", "new_dim"])
    assert "C1" not in ratings.read_text(encoding="utf-8")    # nothing half-written
