"""Drift monitoring: catch slow decay before anyone complains.

The signals are already computed and persisted per call by the core. This reads
them from the on-disk artifacts, compares a current window to a known-good
baseline, and flags movement with standard statistical limits (documented in
docs/MLOPS.md), not numbers picked by feel:

  - score distribution  -> Population Stability Index (PSI)
  - review rate         -> Wilson 99% interval on the baseline proportion
  - the rest            -> Shewhart 3-sigma control limits on baseline median+MAD

Some signals live in the RAW transcripts/ dir; this reader inherits that
PII-restricted access, exactly as the pipeline has it.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DRIFT_DIR = "drift"
SCORE_BINS = [0, 20, 40, 60, 80, 100]
PSI_WARN, PSI_FLAG = 0.10, 0.25       # standard PSI bands (bank model-risk)
WILSON_Z = 2.576                       # 99%
CONTROL_SIGMA = 3.0                    # Shewhart
MIN_BASELINE_CALLS = 20                # as calibration: fewer is noise
_MAD_TO_SIGMA = 1.4826
_MIN_SAMPLES = 5                       # below this a signal is reported, not judged
_EPS = 1e-9                            # absorb float noise at interval boundaries


# ----------------------------------------------------------- collect signals

@dataclass
class Signals:
    scores: list[float] = field(default_factory=list)
    n_total: int = 0
    n_review: int = 0
    redaction_per_call: list[float] = field(default_factory=list)
    role_confidence: list[float] = field(default_factory=list)
    alignment: list[float] = field(default_factory=list)
    logprob: list[float] = field(default_factory=list)
    low_conf: list[float] = field(default_factory=list)


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect_signals(output_dir: Path) -> Signals:
    s = Signals()
    for p in sorted((output_dir / "scores").glob("*.json")):
        card = _load(p)
        if isinstance(card, dict) and card.get("weighted_total") is not None:
            s.scores.append(float(card["weighted_total"]))
    for p in sorted((output_dir / "results").glob("*.json")):
        res = _load(p)
        if isinstance(res, dict) and res.get("status"):
            s.n_total += 1
            if res["status"] == "needs_human_review":
                s.n_review += 1
    for p in sorted((output_dir / "redacted").glob("*.json")):
        red = _load(p)
        if isinstance(red, dict) and isinstance(red.get("redaction_counts"), dict):
            s.redaction_per_call.append(float(sum(red["redaction_counts"].values())))
    for p in sorted((output_dir / "transcripts").glob("*.dialog.json")):
        d = _load(p)
        if not isinstance(d, dict):
            continue
        if d.get("role_confidence") is not None:
            s.role_confidence.append(float(d["role_confidence"]))
        diar = d.get("diarization")
        if isinstance(diar, dict) and diar.get("words_attributed"):
            s.alignment.append(float(diar.get("words_by_nearest", 0))
                                / float(diar["words_attributed"]))
    for p in sorted((output_dir / "transcripts").glob("*.json")):
        if p.name.endswith(".dialog.json"):
            continue
        t = _load(p)
        if not isinstance(t, dict):
            continue
        # The file is a TranscriptBundle: {banker, customer} on a stereo call,
        # {mono} on a single-channel one. `quality` lives on each of those, not
        # on the bundle. Reading it off the bundle meant these two documented
        # signals collected zero samples on every call ever processed, and
        # vanished from every baseline and every report without a word.
        for part in ("banker", "customer", "mono"):
            q = (t.get(part) or {}).get("quality") if isinstance(t.get(part), dict) else None
            if not isinstance(q, dict):
                continue
            if q.get("mean_logprob") is not None:
                s.logprob.append(float(q["mean_logprob"]))
            if q.get("low_confidence_ratio") is not None:
                s.low_conf.append(float(q["low_confidence_ratio"]))
    return s


# ------------------------------------------------------------------- stats

def _hist(values: list[float], bins: list[int]) -> list[int]:
    """Bin counts over half-open bins [b0,b1), [b1,b2), ..., last bin closed.
    Values outside the range clamp into the nearest edge bin."""
    counts = [0] * (len(bins) - 1)
    last = len(bins) - 2
    for v in values:
        if v <= bins[0]:
            counts[0] += 1
            continue
        if v >= bins[-1]:
            counts[last] += 1
            continue
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
    return counts


def psi(baseline_counts: list[int], current_counts: list[int]) -> float:
    tb, tc = sum(baseline_counts) or 1, sum(current_counts) or 1
    total = 0.0
    for b, c in zip(baseline_counts, current_counts, strict=True):
        pb = max(b / tb, 1e-4)
        pc = max(c / tc, 1e-4)
        total += (pc - pb) * math.log(pc / pb)
    return round(total, 4)


def wilson_interval(p: float, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _control_limits(values: list[float]) -> tuple[float, float, float]:
    """Median +/- 3 robust sigmas, where sigma comes from the MAD.

    The MAD is zero whenever more than half the samples equal the median - the
    NORMAL case for a count like redaction_per_call, or a ratio that is usually
    exactly 0.0 - and a zero sigma collapsed the band to a single point, so any
    deviation at all read as drift. That is not a robust estimate; it is a
    degenerate one. When the MAD is zero, the ordinary standard deviation is
    used instead. Only a genuinely CONSTANT baseline - every sample identical -
    keeps a zero-width band, because there any change really is a change.
    """
    med = statistics.median(values)
    sigma = statistics.median([abs(v - med) for v in values]) * _MAD_TO_SIGMA
    if sigma == 0 and len(set(values)) > 1:
        sigma = statistics.pstdev(values)
    return med, med - CONTROL_SIGMA * sigma, med + CONTROL_SIGMA * sigma


# --------------------------------------------------------------- baseline

class DriftBaselineError(ValueError):
    """The window offered as a baseline is too small to be a reference."""


def build_baseline(output_dir: Path, force: bool = False) -> dict:
    s = collect_signals(output_dir)
    n = s.n_total or len(s.scores)
    # A reference built from one call was accepted and reported as success;
    # a perfectly healthy 30-call window checked against it then read PSI 1.8,
    # "drift DETECTED", and a page to whoever is on call. The floor is the same
    # one calibration uses, for the same reason.
    if n < MIN_BASELINE_CALLS and not force:
        raise DriftBaselineError(
            f"a drift baseline needs at least {MIN_BASELINE_CALLS} processed calls "
            f"from a known-good period; this output directory has {n}. A smaller "
            "reference flags normal variation as drift. Process more calls first, "
            "or pass --force if you understand the numbers will be noise."
        )
    signals: dict = {}
    if s.scores:
        signals["score"] = {"type": "psi", "bins": SCORE_BINS,
                            "counts": _hist(s.scores, SCORE_BINS)}
    if s.n_total:
        signals["review_rate"] = {"type": "proportion",
                                  "p": round(s.n_review / s.n_total, 4), "n": s.n_total}
    for name, vals in (("redaction_per_call", s.redaction_per_call),
                       ("role_confidence", s.role_confidence),
                       ("alignment", s.alignment),
                       ("transcription_logprob", s.logprob),
                       ("transcription_low_conf", s.low_conf)):
        if len(vals) >= _MIN_SAMPLES:
            med, lo, hi = _control_limits(vals)
            signals[name] = {"type": "control", "median": round(med, 4),
                             "lo": round(lo, 4), "hi": round(hi, 4), "n": len(vals)}
    return {"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_calls": s.n_total or len(s.scores), "signals": signals}


# ---------------------------------------------------------------- monitor

@dataclass
class SignalResult:
    name: str
    status: str           # ok | warn | flag | skip
    detail: str


def _check_score(base: dict, s: Signals) -> SignalResult:
    if len(s.scores) < _MIN_SAMPLES:
        return SignalResult("score", "skip", f"only {len(s.scores)} scored calls")
    value = psi(base["counts"], _hist(s.scores, base["bins"]))
    status = "flag" if value > PSI_FLAG else ("warn" if value > PSI_WARN else "ok")
    return SignalResult("score", status, f"PSI={value} (warn>{PSI_WARN}, flag>{PSI_FLAG})")


def _check_review(base: dict, s: Signals) -> SignalResult:
    if s.n_total < _MIN_SAMPLES:
        return SignalResult("review_rate", "skip", f"only {s.n_total} calls")
    rate = s.n_review / s.n_total
    lo, hi = wilson_interval(base["p"], s.n_total)
    status = "flag" if (rate < lo - _EPS or rate > hi + _EPS) else "ok"
    return SignalResult("review_rate", status,
                        f"{rate:.3f} vs baseline {base['p']} (99% band [{lo:.3f},{hi:.3f}])")


def _check_control(name: str, base: dict, vals: list[float]) -> SignalResult:
    if len(vals) < _MIN_SAMPLES:
        return SignalResult(name, "skip", f"only {len(vals)} samples")
    # Compare like-for-like: the current window's MEDIAN against control limits
    # built from the baseline median. (Comparing a mean to median-based limits
    # flagged skewed-but-unchanged data.)
    cur = statistics.median(vals)
    lo, hi = base["lo"], base["hi"]
    if lo == hi:                                    # constant baseline (e.g. all-stereo)
        status = "ok" if abs(cur - base["median"]) < 1e-6 else "flag"
    else:
        status = "flag" if (cur < lo - _EPS or cur > hi + _EPS) else "ok"
    return SignalResult(name, status,
                        f"median {cur:.4f} vs baseline [{lo:.4f},{hi:.4f}]")


def check_drift(output_dir: Path, baseline: dict) -> list[SignalResult]:
    s = collect_signals(output_dir)
    sig = baseline.get("signals", {})
    results: list[SignalResult] = []
    if "score" in sig:
        results.append(_check_score(sig["score"], s))
    if "review_rate" in sig:
        results.append(_check_review(sig["review_rate"], s))
    for name, vals in (("redaction_per_call", s.redaction_per_call),
                       ("role_confidence", s.role_confidence),
                       ("alignment", s.alignment),
                       ("transcription_logprob", s.logprob),
                       ("transcription_low_conf", s.low_conf)):
        if name in sig:
            results.append(_check_control(name, sig[name], vals))
    return results


def append_history(output_dir: Path, results: list[SignalResult]) -> None:
    path = output_dir / DRIFT_DIR / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": datetime.now(UTC).isoformat(timespec="seconds"),
           "signals": {r.name: r.status for r in results}}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
