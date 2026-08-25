"""Per-banker aggregation over accumulated per-call ScoreCards (stage 9)."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from callqa.models import Evidence, ScoreCard
from callqa.rubric import Rubric

logger = logging.getLogger(__name__)


def load_scorecards(output_dir: Path) -> list[ScoreCard]:
    scores_dir = output_dir / "scores"
    cards = []
    if scores_dir.exists():
        for path in sorted(scores_dir.glob("*.json")):
            cards.append(ScoreCard.model_validate_json(path.read_text(encoding="utf-8")))
    return cards


@dataclass
class EvidencePick:
    dimension_id: str
    call_id: str
    score: int
    evidence: Evidence


@dataclass
class BankerAggregate:
    banker_id: str
    n_calls: int
    mean_total: float
    dim_means: dict[str, float]
    strongest_dim: str
    weakest_dim: str
    strength_evidence: EvidencePick | None
    development_evidence: EvidencePick | None
    cards: list[ScoreCard] = field(default_factory=list)


def _pick_evidence(
    cards: list[ScoreCard], dimension_id: str, best: bool
) -> EvidencePick | None:
    """Pick a verbatim quote for a dimension from the banker's calls:
    the highest-scoring call for a strength, lowest for a development area."""
    candidates = [
        (card.scores[dimension_id].score, card)
        for card in cards
        if dimension_id in card.scores and card.scores[dimension_id].evidence
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].call_id), reverse=best)
    score, card = candidates[0]
    return EvidencePick(
        dimension_id=dimension_id,
        call_id=card.call_id,
        score=score,
        evidence=card.scores[dimension_id].evidence[0],
    )


def aggregate_bankers(
    cards: list[ScoreCard], rubric: Rubric, group_comparison: str = "median"
) -> tuple[dict[str, BankerAggregate], dict[str, float], float]:
    """Returns (per-banker aggregates, per-dimension group stat, group total stat)."""
    stat_fn = statistics.median if group_comparison == "median" else statistics.mean

    group_dim: dict[str, float] = {}
    for dim in rubric.dimensions:
        values = [c.scores[dim.id].score for c in cards if dim.id in c.scores]
        group_dim[dim.id] = round(stat_fn(values), 2) if values else 0.0
    group_total = round(stat_fn([c.weighted_total for c in cards]), 1) if cards else 0.0

    by_banker: dict[str, list[ScoreCard]] = {}
    for card in cards:
        by_banker.setdefault(card.banker_id, []).append(card)

    aggregates: dict[str, BankerAggregate] = {}
    for banker_id, banker_cards in sorted(by_banker.items()):
        dim_means = {}
        for dim in rubric.dimensions:
            values = [c.scores[dim.id].score for c in banker_cards if dim.id in c.scores]
            dim_means[dim.id] = round(statistics.mean(values), 2) if values else 0.0
        # Strongest/weakest by mean score; rubric order breaks ties deterministically.
        ordered = [d.id for d in rubric.dimensions]
        strongest = max(ordered, key=lambda d: (dim_means[d], -ordered.index(d)))
        weakest = min(ordered, key=lambda d: (dim_means[d], ordered.index(d)))
        aggregates[banker_id] = BankerAggregate(
            banker_id=banker_id,
            n_calls=len(banker_cards),
            mean_total=round(statistics.mean(c.weighted_total for c in banker_cards), 1),
            dim_means=dim_means,
            strongest_dim=strongest,
            weakest_dim=weakest,
            strength_evidence=_pick_evidence(banker_cards, strongest, best=True),
            development_evidence=_pick_evidence(banker_cards, weakest, best=False),
            cards=sorted(banker_cards, key=lambda c: c.call_id),
        )
    logger.info("aggregation done: bankers=%d calls=%d", len(aggregates), len(cards))
    return aggregates, group_dim, group_total
