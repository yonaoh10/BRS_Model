"""Call quality inside customer journeys: the link no outside vendor has.

Every recorded call can carry a scorecard (the rubric the management report
uses). Here each scored call is placed in its story, and the question is
asked the other way round: did the calls after which the customer came back
because of a handling failure score lower than the others - and on which
dimension?

    led_to_failure   the next contact of the story is a return classified
                     as a handling failure (unclosed loop / runaround)
    other            every other scored call (followed by a legitimate
                     return, a new topic, or nothing)

A Welch interval on the difference of the quality index; per dimension, the
difference of the 1-5 means. Descriptive, not causal, and shown only with
enough calls on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from callqa.journey.analysis import shown_category
from callqa.journey.rules import StoryFacts
from callqa.journey.vocab import Taxonomy
from callqa.models import ScoreCard
from callqa.reporting.executive.stats import mean_ci, welch_diff

MIN_GROUP = 8


@dataclass
class DimGap:
    dim_id: str
    name: str
    mean_fail: float
    mean_other: float

    @property
    def diff(self) -> float:
        return self.mean_fail - self.mean_other


@dataclass
class JourneyQuality:
    n_scored: int
    n_fail: int
    n_other: int
    mean_fail: float | None
    mean_other: float | None
    diff: float | None
    low: float | None
    high: float | None
    significant: bool
    dims: list[DimGap] = field(default_factory=list)
    enough: bool = False

    @property
    def worst_dim(self) -> DimGap | None:
        return min(self.dims, key=lambda d: d.diff, default=None)


def _load_cards(scores_dir: Path, call_ids: set[str]) -> dict[str, ScoreCard]:
    out = {}
    for cid in call_ids:
        path = scores_dir / f"{cid}.json"
        if not path.exists():
            continue
        try:
            out[cid] = ScoreCard.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
    return out


def journey_quality(facts: list[StoryFacts], output_dir: Path, taxonomy: Taxonomy,
                    dim_names: dict[str, str]) -> JourneyQuality | None:
    failure = taxonomy.failure_categories
    placed: list[tuple[str, bool]] = []          # (call_id, led to a failed return)
    for f in facts:
        contacts = f.timeline.contacts
        for n, c in enumerate(contacts):
            cid = c.interaction.call_id
            if c.kind != "recorded_call" or not cid:
                continue
            nxt = contacts[n + 1] if n + 1 < len(contacts) else None
            j = f.judgements.get(nxt.interaction.interaction_id) if nxt is not None else None
            placed.append((cid, j is not None and shown_category(j) in failure))
    if not placed:
        return None
    cards = _load_cards(output_dir / "scores", {cid for cid, _ in placed})
    fail = [cards[c] for c, led in placed if led and c in cards]
    other = [cards[c] for c, led in placed if not led and c in cards]
    if not cards:
        return None
    d = welch_diff([c.weighted_total for c in fail], [c.weighted_total for c in other])
    dims = []
    for dim_id, name in dim_names.items():
        a = [c.scores[dim_id].score for c in fail if dim_id in c.scores]
        b = [c.scores[dim_id].score for c in other if dim_id in c.scores]
        if a and b:
            dims.append(DimGap(dim_id, name, mean_ci(a).mean, mean_ci(b).mean))
    return JourneyQuality(
        n_scored=len(fail) + len(other), n_fail=len(fail), n_other=len(other),
        mean_fail=mean_ci([c.weighted_total for c in fail]).mean if fail else None,
        mean_other=mean_ci([c.weighted_total for c in other]).mean if other else None,
        diff=d.diff, low=d.low, high=d.high, significant=d.significant,
        dims=sorted(dims, key=lambda g: g.diff),
        enough=len(fail) >= MIN_GROUP and len(other) >= MIN_GROUP)
