"""Calibration of the LLM judge against human QA ratings (stage 10).

Computes per-dimension and overall quadratic-weighted kappa (QWK)
system-vs-human (human = mean of raters, rounded), human-vs-human agreement
on doubly-rated calls, MAE, and 5x5 confusion matrices.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from callqa.models import ScoreCard
from callqa.rubric import Rubric

logger = logging.getLogger(__name__)

QWK_PASS_THRESHOLD = 0.70
# A kappa over a handful of calls is noise. Two rated calls used to be enough
# to declare the judge calibrated against a human.
MIN_CALLS_FOR_PASS = 20
QWK_DIMENSION_FLAG_THRESHOLD = 0.60
LABELS = [1, 2, 3, 4, 5]


class CalibrationError(ValueError):
    pass


@dataclass
class DimensionCalibration:
    dimension_id: str
    n: int
    qwk: float | None
    mae: float | None
    confusion: list[list[int]]  # 5x5, rows=human, cols=system
    flagged: bool


@dataclass
class CalibrationResult:
    n_calls: int
    overall_qwk: float | None
    overall_pass: bool
    dimensions: list[DimensionCalibration]
    human_vs_human_qwk: float | None
    human_vs_human_kappa: float | None
    n_doubly_rated: int
    flagged_dimensions: list[str] = field(default_factory=list)
    # Mean of the per-dimension kappas. Reported beside the pooled value
    # because the two disagree exactly when one dimension is broken.
    mean_dimension_qwk: float | None = None

    def to_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "overall_qwk": self.overall_qwk,
            "mean_dimension_qwk": self.mean_dimension_qwk,
            "overall_pass": self.overall_pass,
            "qwk_pass_threshold": QWK_PASS_THRESHOLD,
            "min_calls_for_pass": MIN_CALLS_FOR_PASS,
            "dimension_flag_threshold": QWK_DIMENSION_FLAG_THRESHOLD,
            "human_vs_human": {
                "qwk": self.human_vs_human_qwk,
                "cohen_kappa": self.human_vs_human_kappa,
                "n_doubly_rated_calls": self.n_doubly_rated,
            },
            "dimensions": [
                {
                    "dimension_id": d.dimension_id,
                    "n": d.n,
                    "qwk": d.qwk,
                    "mae": d.mae,
                    "flagged": d.flagged,
                    "confusion": d.confusion,
                }
                for d in self.dimensions
            ],
            "flagged_dimensions": self.flagged_dimensions,
        }


def load_human_ratings(
    path: Path, dimension_ids: list[str]
) -> dict[str, list[dict[str, int]]]:
    """call_id -> list of per-rater {dimension_id: score} dicts."""
    if not path.exists():
        raise CalibrationError(f"human ratings file not found: {path}")
    ratings: dict[str, list[dict[str, int]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [d for d in dimension_ids if d not in (reader.fieldnames or [])]
        if missing:
            raise CalibrationError(
                f"human_ratings.csv is missing rubric dimension columns: {missing}"
            )
        for i, row in enumerate(reader, start=1):
            call_id = (row.get("call_id") or "").strip()
            if not call_id:
                raise CalibrationError(f"human_ratings.csv row {i}: empty call_id")
            per_dim: dict[str, int] = {}
            for dim in dimension_ids:
                raw = (row.get(dim) or "").strip()
                if not raw:
                    continue
                score = int(raw)
                if not 1 <= score <= 5:
                    raise CalibrationError(
                        f"human_ratings.csv row {i}: {dim}={score} outside 1-5"
                    )
                per_dim[dim] = score
            ratings.setdefault(call_id, []).append(per_dim)
    return ratings


def _safe_qwk(a: list[int], b: list[int], weights: str | None = "quadratic") -> float | None:
    """Quadratic-weighted kappa, or None where it is not defined.

    Kappa measures agreement ABOVE chance. When every rating is the same value
    there is no variance to work with and chance agreement is total, so kappa
    is undefined. Returning 1.0 there reported a rater who has only ever said
    "3" as being in perfect agreement with the model, and one agreeing call
    was enough to pass the whole calibration.
    """
    if len(a) < 2:
        return None
    if len(set(a)) == 1 or len(set(b)) == 1:
        return None
    try:
        value = cohen_kappa_score(a, b, labels=LABELS, weights=weights)
    except ValueError:
        return None
    if value != value:  # NaN
        return None
    return round(float(value), 4)


def _mean_rounded(values: list[int]) -> int:
    return int(round(sum(values) / len(values)))


def calibrate(
    scorecards: list[ScoreCard],
    human_ratings: dict[str, list[dict[str, int]]],
    rubric: Rubric,
) -> CalibrationResult:
    # Calibration must judge one rubric/prompt version at a time: pooling scores
    # from different rubrics into one QWK verdict is meaningless (and would let a
    # bad rubric hide behind a good one).
    from callqa.aggregation import single_rubric_cohort

    scorecards = single_rubric_cohort(scorecards)
    dim_ids = [d.id for d in rubric.dimensions]
    system_by_call = {c.call_id: c for c in scorecards}
    common_calls = sorted(set(system_by_call) & set(human_ratings))
    if not common_calls:
        raise CalibrationError(
            "no calls present in both system scores and human ratings "
            f"(system: {len(system_by_call)}, human: {len(human_ratings)})"
        )

    dims: list[DimensionCalibration] = []
    pooled_human: list[int] = []
    pooled_system: list[int] = []
    for dim_id in dim_ids:
        human: list[int] = []
        system: list[int] = []
        for call_id in common_calls:
            rater_scores = [r[dim_id] for r in human_ratings[call_id] if dim_id in r]
            card = system_by_call[call_id]
            if not rater_scores or dim_id not in card.scores:
                continue
            human.append(_mean_rounded(rater_scores))
            system.append(card.scores[dim_id].score)
        confusion = [[0] * 5 for _ in range(5)]
        for h, s in zip(human, system, strict=True):
            confusion[h - 1][s - 1] += 1
        qwk = _safe_qwk(human, system)
        mae = round(sum(abs(h - s) for h, s in zip(human, system, strict=True)) / len(human), 3) if human else None
        dims.append(
            DimensionCalibration(
                dimension_id=dim_id,
                n=len(human),
                qwk=qwk,
                mae=mae,
                confusion=confusion,
                flagged=qwk is not None and qwk < QWK_DIMENSION_FLAG_THRESHOLD,
            )
        )
        pooled_human.extend(human)
        pooled_system.extend(system)

    overall_qwk = _safe_qwk(pooled_human, pooled_system)
    # Pooling across dimensions hides a dimension that disagrees: a set whose
    # eight per-dimension kappas were all below 0.25 still pooled to 0.90,
    # because pooling rewards getting the overall LEVEL right. So passing also
    # requires enough rated calls and no flagged dimension.
    dimension_qwks = [d.qwk for d in dims if d.qwk is not None]
    mean_dimension_qwk = (
        round(sum(dimension_qwks) / len(dimension_qwks), 4) if dimension_qwks else None
    )

    # Human-vs-human agreement on doubly-rated calls (first two raters).
    hh_a: list[int] = []
    hh_b: list[int] = []
    doubly_rated = 0
    for _call_id, raters in human_ratings.items():
        if len(raters) < 2:
            continue
        doubly_rated += 1
        for dim_id in dim_ids:
            if dim_id in raters[0] and dim_id in raters[1]:
                hh_a.append(raters[0][dim_id])
                hh_b.append(raters[1][dim_id])
    hh_qwk = _safe_qwk(hh_a, hh_b) if hh_a else None
    hh_kappa = _safe_qwk(hh_a, hh_b, weights=None) if hh_a else None

    result = CalibrationResult(
        n_calls=len(common_calls),
        overall_qwk=overall_qwk,
        overall_pass=(
            overall_qwk is not None
            and overall_qwk >= QWK_PASS_THRESHOLD
            and mean_dimension_qwk is not None
            and mean_dimension_qwk >= QWK_PASS_THRESHOLD
            and len(common_calls) >= MIN_CALLS_FOR_PASS
            and not any(d.flagged for d in dims)
        ),
        mean_dimension_qwk=mean_dimension_qwk,
        dimensions=dims,
        human_vs_human_qwk=hh_qwk,
        human_vs_human_kappa=hh_kappa,
        n_doubly_rated=doubly_rated,
        flagged_dimensions=[d.dimension_id for d in dims if d.flagged],
    )
    logger.info(
        "calibration done: calls=%d overall_qwk=%s pass=%s",
        result.n_calls, result.overall_qwk, result.overall_pass,
    )
    return result
