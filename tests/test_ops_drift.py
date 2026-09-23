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
    # force: this test is about the CURRENT window being too small to judge,
    # not about the baseline, which is refused below 20 calls on its own.
    baseline = build_baseline(out, force=True)
    results = {r.name: r.status for r in check_drift(out, baseline)}
    assert results.get("score") == "skip"


def test_a_baseline_from_a_handful_of_calls_is_refused(tmp_path: Path) -> None:
    """A one-call baseline was accepted as success; a healthy 30-call window
    checked against it then read PSI 1.8 and "drift DETECTED" - a page to
    whoever is on call, about nothing."""
    import pytest

    from callqa.ops.drift import DriftBaselineError

    out = tmp_path / "output"
    _write(out, [80.0], ["success"])
    with pytest.raises(DriftBaselineError, match="at least 20"):
        build_baseline(out)
    assert build_baseline(out, force=True)["n_calls"] == 1     # deliberate override


def test_a_mostly_constant_signal_does_not_collapse_the_control_band() -> None:
    """When more than half the samples equal the median, the MAD is zero, and a
    zero-width band flagged ANY deviation - the normal shape of a count such as
    identifiers-masked-per-call. Only a truly constant baseline keeps it."""
    from callqa.ops.drift import _control_limits

    mostly_two = [2.0] * 12 + [1.0, 3.0, 4.0, 1.0, 3.0]
    med, lo, hi = _control_limits(mostly_two)
    assert lo < med < hi, (lo, med, hi)
    assert lo <= 1.0 and hi >= 3.0                  # ordinary variation is in-band

    constant = [0.0] * 10
    _, lo, hi = _control_limits(constant)
    assert lo == hi == 0.0                          # here any change IS a change


def test_transcription_quality_is_read_from_the_nested_transcripts(tmp_path: Path) -> None:
    """`quality` lives on each channel's transcript, not on the bundle. Read
    off the bundle, both transcription signals collected zero samples on every
    call ever processed, and disappeared from every report silently."""
    from callqa.ops.drift import collect_signals

    out = tmp_path / "output"
    (out / "transcripts").mkdir(parents=True)
    for i in range(3):
        bundle = {"call_id": f"C{i}",
                  "banker": {"quality": {"mean_logprob": -0.3, "low_confidence_ratio": 0.05}},
                  "customer": {"quality": {"mean_logprob": -0.5, "low_confidence_ratio": 0.10}}}
        (out / "transcripts" / f"C{i}.json").write_text(json.dumps(bundle), encoding="utf-8")
    (out / "transcripts" / "M.json").write_text(json.dumps(
        {"call_id": "M", "mono": {"quality": {"mean_logprob": -0.4,
                                              "low_confidence_ratio": 0.2}}}), encoding="utf-8")
    s = collect_signals(out)
    assert len(s.logprob) == 7 and len(s.low_conf) == 7
