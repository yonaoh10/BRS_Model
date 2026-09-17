"""Drift monitoring: baseline capture + statistically-grounded flags."""

from __future__ import annotations

import json
from pathlib import Path

from callqa.ops.drift import (
    build_baseline,
    check_drift,
    psi,
    wilson_interval,
)


def test_psi_math() -> None:
    assert psi([10, 10, 10, 10, 10], [10, 10, 10, 10, 10]) == 0.0
    assert psi([40, 10, 0, 0, 0], [0, 0, 0, 10, 40]) > 0.25   # a large shift


def test_wilson_interval_widens_with_small_n() -> None:
    lo_big, hi_big = wilson_interval(0.2, 1000)
    lo_small, hi_small = wilson_interval(0.2, 20)
    assert (hi_small - lo_small) > (hi_big - lo_big)          # less data -> wider band


def _write(out: Path, scores: list[float], statuses: list[str]) -> None:
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    for i, (sc, st) in enumerate(zip(scores, statuses, strict=True)):
        (out / "scores" / f"C{i}.json").write_text(
            json.dumps({"call_id": f"C{i}", "weighted_total": sc}), encoding="utf-8")
        (out / "results" / f"C{i}.json").write_text(
            json.dumps({"call_id": f"C{i}", "status": st}), encoding="utf-8")


def test_no_drift_on_the_same_window(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _write(out, [78, 80, 82, 79, 81, 77, 83, 80] * 3, ["success"] * 24)
    baseline = build_baseline(out)
    results = {r.name: r.status for r in check_drift(out, baseline)}
    assert results["score"] == "ok"
    assert results["review_rate"] == "ok"


def test_a_score_collapse_is_flagged(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _write(out, [78, 80, 82, 79, 81, 77, 83, 80] * 3, ["success"] * 24)
    baseline = build_baseline(out)
    # a new telephony/codec makes every call score much lower
    for p in (out / "scores").glob("*.json"):
        p.write_text(json.dumps({"call_id": p.stem, "weighted_total": 30}), encoding="utf-8")
    results = {r.name: r.status for r in check_drift(out, baseline)}
    assert results["score"] == "flag"


def test_a_review_rate_spike_is_flagged(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _write(out, [80] * 30, ["success"] * 30)                 # baseline: nobody held
    baseline = build_baseline(out)
    _write(out, [80] * 30, ["needs_human_review"] * 15 + ["success"] * 15)   # now half held
    results = {r.name: r.status for r in check_drift(out, baseline)}
    assert results["review_rate"] == "flag"


def test_a_signal_with_too_little_data_is_skipped_not_judged(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _write(out, [80, 81], ["success", "success"])            # below the min-samples floor
    baseline = build_baseline(out)
    results = {r.name: r.status for r in check_drift(out, baseline)}
    assert results.get("score") == "skip"
