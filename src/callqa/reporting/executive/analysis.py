"""Every number in the management report, computed once, from the gated batch.

The report's credibility with management rests on three habits kept here:

* Every estimate carries its uncertainty (a confidence interval) and its n,
  and a small group is labelled "few calls" rather than ranked.
* A difference is called a difference only when it is both statistically
  clear and large enough to matter. Comparisons among many groups (bankers,
  call types) use 99% intervals, because with forty bankers a 95% rule flags
  two of them by chance alone.
* The index is decomposed exactly: 100 minus the index is the sum of the
  points lost on each dimension plus the gate penalty, so "where the points
  go" adds up to what the headline says.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date

from callqa.models import DimensionScore, ScoreCard
from callqa.reporting.executive import stats
from callqa.reporting.executive.dataset import Batch, CallRecord
from callqa.rubric import GATE_CAP, Rubric, weighted_total

# Score bands shared with every other report: (lower bound, band index, label).
BANDS = ((80.0, 4, "גבוה"), (60.0, 3, "בינוני"), (40.0, 2, "גבולי"), (0.0, 1, "חלש"))
BAND_LABEL = {band: label for _, band, label in BANDS}
LEVEL_LABELS = ["1", "2", "3", "4", "5"]

MIN_GROUP_N = 8          # below this a group is shown, not compared
MIN_PRACTICAL_DIFF = 3.0  # index points: smaller gaps are not worth a manager's time
GROUP_LEVEL = 0.99       # many-group comparisons
LEVEL = 0.95
MIN_TREND_N = 30
MIN_TREND_SPAN_DAYS = 14
MIN_DRIVER_N = 30
MIN_DRIVER_RHO = 0.15
RARE_SHARE = 0.01        # a call type below 1% of the batch (and < 5 calls) is "אחר"
OTHER_TYPE = "אחר"

DURATION_BUCKETS = (
    ("short", "עד 3 דק'", 0.0, 180.0),
    ("mid", "3–6 דק'", 180.0, 360.0),
    ("long", "6–10 דק'", 360.0, 600.0),
    ("xlong", "מעל 10 דק'", 600.0, float("inf")),
)


def band_of(total: float) -> int:
    for lower, band, _ in BANDS:
        if total >= lower:
            return band
    return 1


# -- result types ---------------------------------------------------------------

@dataclass
class DimStat:
    id: str
    name: str
    weight: float
    gate: bool
    mean: stats.MeanCI
    counts: list[int]              # calls at score 1..5
    low_rate: float                # share scored <= 2
    high_rate: float               # share scored >= 4
    lost_mean: float               # index points lost per call on this dimension
    lost_share: float              # share of ALL lost points (incl. the gate penalty)
    gain_if_floor3: float          # index gain if every score below 3 were a 3
    gates_removed: int             # gate failures that floor would remove
    # The loss including the gate penalty this dimension CAUSED (a call's
    # penalty is split among the gates it failed): what "the biggest source of
    # lost points" must rank by, or a gate dimension hides behind its own cap.
    lost_with_gate: float = 0.0
    lost_share_with_gate: float = 0.0
    focus_bankers: list[tuple[str, int, int]] = field(default_factory=list)  # (id, low, n)
    focus_share: float = 0.0       # share of this dimension's low scores in that group
    focus_call_share: float = 0.0  # the same group's share of all calls


@dataclass
class PeriodStat:
    start: date
    label: str
    n: int
    mean: stats.MeanCI
    gate_rate: float | None
    high_rate: float | None


@dataclass
class GroupStat:
    key: str
    label: str
    n: int
    mean: stats.MeanCI
    gate: stats.PropCI
    diff: stats.DiffCI | None      # vs the rest of the batch (99%)
    flag: str                      # "bad" | "ok" | "none" | "few"
    weakest: str | None = None     # dimension id
    strongest: str | None = None
    dim_means: dict[str, float | None] = field(default_factory=dict)
    low_calls: int = 0             # calls below 60
    dim_low: dict[str, int] = field(default_factory=dict)   # calls scored <= 2, per dimension
    link: str | None = None        # banker report
    filter: dict = field(default_factory=dict)


@dataclass
class Driver:
    key: str
    label: str
    unit: str
    n: int
    corr: stats.Corr
    q1: float | None               # feature value at the 25th / 75th percentile
    q3: float | None
    low_mean: float | None         # mean index in the bottom / top quartile of the feature
    high_mean: float | None
    contrast: stats.DiffCI | None  # top-quartile mean minus bottom-quartile mean
    reportable: bool
    filter_high: dict = field(default_factory=dict)


@dataclass
class GateStat:
    id: str
    name: str
    fails: int
    rate: stats.PropCI
    top_bankers: list[tuple[str, int, int]]     # (banker, fails, calls)
    concentration: float | None                 # share of fails in the top group of bankers
    by_type: list[tuple[str, int, int]]         # (type label, fails, calls)
    top_n: int = 0                              # size of that group
    n_bankers: int = 0
    call_share: float | None = None             # that group's share of all calls


@dataclass
class Example:
    call_id: str
    banker_id: str
    dim: str
    score: int
    total: float
    quote: str
    timestamp: str
    speaker: str
    reasoning: str
    call_report: str | None


@dataclass
class Analysis:
    batch: Batch
    rubric: Rubric
    n_scope: int
    n_scored: int
    n_held: int
    n_failed: int
    total: stats.MeanCI
    median: float | None
    q1: float | None
    q3: float | None
    band_counts: dict[int, int]
    high_rate: stats.PropCI
    low_rate: stats.PropCI          # below 60
    gate_rate: stats.PropCI
    review_rate: stats.PropCI       # held out of (scored + held)
    gate_penalty_mean: float
    first_date: date | None
    last_date: date | None
    granularity: str | None
    dims: list[DimStat]
    histogram: list[tuple[float, float, int]]
    periods: list[PeriodStat]
    trend: stats.Trend | None
    last_vs_prev: stats.DiffCI | None
    segments: dict[str, list[GroupStat]]
    bankers: list[GroupStat]
    drivers: list[Driver]
    gates: list[GateStat]
    best_examples: dict[str, list[Example]]
    worst_examples: dict[str, list[Example]]
    attention: list[CallRecord]
    coverage: dict
    date_basis: str = "none"        # "call" | "processed" | "none": what the time axis is
    undated: int = 0                # scored calls left out of the time series
    banker_reference: float | None = None   # median of banker means (n >= MIN_GROUP_N)

    @property
    def dim_names(self) -> dict[str, str]:
        return {d.id: d.name for d in self.dims}


# -- helpers --------------------------------------------------------------------

def _score(card: ScoreCard, dim_id: str) -> int | None:
    entry = card.scores.get(dim_id)
    return entry.score if entry is not None else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def _flag(diff: stats.DiffCI | None, n: int) -> str:
    if n < MIN_GROUP_N:
        return "few"
    if diff is None or diff.diff is None or not diff.significant:
        return "none"
    if abs(diff.diff) < MIN_PRACTICAL_DIFF:
        return "none"
    return "ok" if diff.diff > 0 else "bad"


def _group(key: str, label: str, members: list[CallRecord], rest: list[CallRecord],
           rubric: Rubric, filt: dict, link: str | None = None) -> GroupStat:
    totals = [r.card.weighted_total for r in members]
    rest_totals = [r.card.weighted_total for r in rest]
    diff = (stats.welch_diff(totals, rest_totals, level=GROUP_LEVEL)
            if len(totals) >= 2 and len(rest_totals) >= 2 else None)
    gates = sum(1 for r in members if r.card.gate_failed)
    dim_means: dict[str, float | None] = {}
    dim_low: dict[str, int] = {}
    for d in rubric.dimensions:
        vals = [s for r in members if (s := _score(r.card, d.id)) is not None]
        dim_means[d.id] = _mean(vals)
        dim_low[d.id] = sum(1 for v in vals if v <= 2)
    scored_dims = [(d.id, m) for d, m in ((d, dim_means[d.id]) for d in rubric.dimensions)
                   if m is not None]
    order = [d.id for d in rubric.dimensions]
    weakest = min(scored_dims, key=lambda x: (x[1], order.index(x[0])))[0] if scored_dims else None
    strongest = (max(scored_dims, key=lambda x: (x[1], -order.index(x[0])))[0]
                 if scored_dims else None)
    return GroupStat(
        key=key, label=label, n=len(members), mean=stats.mean_ci(totals, LEVEL),
        gate=stats.proportion_ci(gates, len(members), LEVEL), diff=diff,
        flag=_flag(diff, len(members)), weakest=weakest, strongest=strongest,
        dim_means=dim_means, low_calls=sum(1 for t in totals if t < 60), link=link,
        dim_low=dim_low,
        filter=filt,
    )


def _floor_gain(cards: list[ScoreCard], rubric: Rubric, dim_id: str) -> tuple[float, int]:
    """Index gain, per call on average, if every score below 3 on one dimension
    were a 3 - the 'meets the basic standard' floor a coaching plan aims for."""
    if not cards:
        return 0.0, 0
    gain = 0.0
    removed = 0
    for card in cards:
        s = _score(card, dim_id)
        if s is None or s >= 3:
            continue
        lifted = dict(card.scores)
        lifted[dim_id] = DimensionScore(score=3, reasoning_he="", evidence=[])
        try:
            new_total, new_gate, _ = weighted_total(rubric, lifted)
        except ValueError:          # a card from a rubric with other dimensions
            continue
        gain += new_total - card.weighted_total
        if card.gate_failed and not new_gate:
            removed += 1
    return gain / len(cards), removed


def _lost_points(card: ScoreCard, rubric: Rubric) -> tuple[dict[str, float], float]:
    """Points lost per dimension, and the gate penalty, for one call.

    100 - raw = sum over dimensions of w*(5-s)/4*100 exactly; the gate cap then
    takes raw down to at most 59, and that difference is the gate penalty.
    """
    lost: dict[str, float] = {}
    raw = 0.0
    for d in rubric.dimensions:
        s = _score(card, d.id)
        if s is None:
            continue
        lost[d.id] = d.weight * (5 - s) / 4.0 * 100.0
        raw += d.weight * (s - 1) / 4.0 * 100.0
    penalty = max(0.0, raw - GATE_CAP) if card.gate_failed else 0.0
    return lost, penalty


# -- the analysis ---------------------------------------------------------------

def analyse(batch: Batch) -> Analysis:
    rubric = batch.rubric
    scored = batch.scored
    cards = [r.card for r in scored]
    totals = [c.weighted_total for c in cards]
    n = len(cards)

    # Overview
    band_counts = Counter(band_of(t) for t in totals)
    gate_fails = sum(1 for c in cards if c.gate_failed)
    n_held = len(batch.held)
    lost_sum: dict[str, float] = defaultdict(float)
    caused: dict[str, float] = defaultdict(float)
    penalty_sum = 0.0
    for card in cards:
        lost, penalty = _lost_points(card, rubric)
        for k, v in lost.items():
            lost_sum[k] += v
        penalty_sum += penalty
        gates = [g for g in card.failed_gates if g in lost]
        for g in gates:
            caused[g] += penalty / len(gates)
    all_lost = sum(lost_sum.values()) + penalty_sum
    n_bankers = len({r.banker_id for r in scored})

    dims: list[DimStat] = []
    for d in rubric.dimensions:
        scores = [s for c in cards if (s := _score(c, d.id)) is not None]
        counts = [sum(1 for s in scores if s == level) for level in range(1, 6)]
        gain, removed = _floor_gain(cards, rubric, d.id)
        focus, focus_share, focus_calls = _focus_bankers(scored, d.id, n_bankers)
        with_gate = lost_sum[d.id] + caused.get(d.id, 0.0)
        dims.append(DimStat(
            id=d.id, name=d.name_he, weight=d.weight, gate=d.gate,
            mean=stats.mean_ci(scores, LEVEL), counts=counts,
            low_rate=_rate(sum(counts[:2]), len(scores)) or 0.0,
            high_rate=_rate(sum(counts[3:]), len(scores)) or 0.0,
            lost_mean=lost_sum[d.id] / n if n else 0.0,
            lost_share=lost_sum[d.id] / all_lost if all_lost else 0.0,
            gain_if_floor3=gain, gates_removed=removed,
            lost_with_gate=with_gate / n if n else 0.0,
            lost_share_with_gate=with_gate / all_lost if all_lost else 0.0,
            focus_bankers=focus, focus_share=focus_share, focus_call_share=focus_calls,
        ))

    edges = [float(x) for x in range(0, 101, 5)]
    counts = stats.histogram(totals, edges)
    histogram = [(edges[i], edges[i + 1], counts[i]) for i in range(len(counts))]

    # The time axis is the date of the CALL. A call with no date in the
    # metadata falls back to the day it was scored, and mixing the two puts
    # every undated call into the last period - a trend made of the batch run.
    # Processing dates are used only when no call has a date at all.
    by_call = [r for r in scored if r.call_date is not None and r.date_source == "call"]
    dated = by_call or [r for r in scored if r.call_date is not None]
    date_basis = "call" if by_call else ("processed" if dated else "none")
    in_scope = [r.call_date for r in batch.records if r.call_date is not None
                and (r.date_source == "call" or date_basis == "processed")]
    first = min(in_scope, default=None)
    last = max(in_scope, default=None)
    granularity = stats.choose_granularity(first, last) if first and last else None
    periods, trend, last_vs_prev = _time(dated, granularity)

    segments = {
        "call_type": _call_type_segments(scored, rubric),
        "duration": _duration_segments(scored, rubric),
        "layout": _layout_segments(scored, rubric),
    }
    bankers, banker_reference = _bankers(scored, rubric)
    drivers = _drivers(scored)
    gates = _gates(scored, rubric)
    best, worst = _examples(scored, rubric)
    attention = sorted(
        (r for r in scored if r.card.gate_failed or r.card.weighted_total < 40),
        key=lambda r: (not r.card.gate_failed, r.card.weighted_total, r.call_id),
    )[:25]

    return Analysis(
        batch=batch, rubric=rubric, n_scope=len(batch.records), n_scored=n,
        n_held=n_held, n_failed=len(batch.failed),
        total=stats.mean_ci(totals, LEVEL),
        median=stats.quantile(totals, 0.5), q1=stats.quantile(totals, 0.25),
        q3=stats.quantile(totals, 0.75), band_counts=dict(band_counts),
        high_rate=stats.proportion_ci(band_counts.get(4, 0), n, LEVEL),
        low_rate=stats.proportion_ci(band_counts.get(1, 0) + band_counts.get(2, 0), n, LEVEL),
        gate_rate=stats.proportion_ci(gate_fails, n, LEVEL),
        review_rate=stats.proportion_ci(n_held, len(batch.records), LEVEL),
        gate_penalty_mean=penalty_sum / n if n else 0.0,
        first_date=first, last_date=last, granularity=granularity,
        dims=dims, histogram=histogram, periods=periods, trend=trend,
        last_vs_prev=last_vs_prev, segments=segments, bankers=bankers, drivers=drivers,
        gates=gates, best_examples=best, worst_examples=worst, attention=attention,
        coverage=_coverage(batch), date_basis=date_basis, undated=n - len(dated),
        banker_reference=banker_reference,
    )


def _focus_bankers(scored: list[CallRecord], dim_id: str, n_bankers: int
                   ) -> tuple[list[tuple[str, int, int]], float, float]:
    """The fifth of the bankers with the most calls scored <= 2 on a dimension,
    with their share of those low scores and their share of all calls. The
    group is worth naming only when the first clearly exceeds the second."""
    low: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for r in scored:
        s = _score(r.card, dim_id)
        if s is None:
            continue
        total[r.banker_id] += 1
        if s <= 2:
            low[r.banker_id] += 1
    if not low:
        return [], 0.0, 0.0
    k = max(1, round(0.2 * n_bankers))
    ranked = sorted(low.items(), key=lambda kv: (-kv[1], -kv[1] / total[kv[0]], kv[0]))[:k]
    share = sum(v for _, v in ranked) / sum(low.values())
    call_share = sum(total[b] for b, _ in ranked) / sum(total.values())
    return [(b, v, total[b]) for b, v in ranked], share, call_share


def _time(dated: list[CallRecord], granularity: str | None
          ) -> tuple[list[PeriodStat], stats.Trend | None, stats.DiffCI | None]:
    if not dated or granularity is None:
        return [], None, None
    buckets: dict[date, list[CallRecord]] = defaultdict(list)
    for r in dated:
        buckets[stats.period_start(r.call_date, granularity)].append(r)
    periods: list[PeriodStat] = []
    for start in sorted(buckets):
        members = buckets[start]
        totals = [r.card.weighted_total for r in members]
        periods.append(PeriodStat(
            start=start, label=stats.period_label(start, granularity), n=len(members),
            mean=stats.mean_ci(totals, LEVEL),
            gate_rate=_rate(sum(1 for r in members if r.card.gate_failed), len(members)),
            high_rate=_rate(sum(1 for t in totals if t >= 80), len(members)),
        ))
    origin = min(r.call_date for r in dated)
    days = [(r.call_date - origin).days for r in dated]
    trend = None
    if len(dated) >= MIN_TREND_N and (max(days) - min(days)) >= MIN_TREND_SPAN_DAYS:
        trend = stats.linear_trend(days, [r.card.weighted_total for r in dated], LEVEL)
    last_vs_prev = None
    if len(periods) >= 2 and periods[-1].n >= MIN_GROUP_N and periods[-2].n >= MIN_GROUP_N:
        last_start, prev_start = periods[-1].start, periods[-2].start
        last_vs_prev = stats.welch_diff(
            [r.card.weighted_total for r in buckets[last_start]],
            [r.card.weighted_total for r in buckets[prev_start]], LEVEL)
    return periods, trend, last_vs_prev


def type_bucket(scored: list[CallRecord]) -> dict[str, str]:
    """call id -> call type label, with rare types folded into 'אחר'."""
    counts = Counter(r.call_type for r in scored)
    floor = max(5, RARE_SHARE * len(scored))
    return {r.call_id: (r.call_type if counts[r.call_type] >= floor else OTHER_TYPE)
            for r in scored}


def _split(scored: list[CallRecord], key_of) -> dict[str, list[CallRecord]]:  # noqa: ANN001
    groups: dict[str, list[CallRecord]] = defaultdict(list)
    for r in scored:
        key = key_of(r)
        if key is not None:
            groups[key].append(r)
    return groups


def _call_type_segments(scored: list[CallRecord], rubric: Rubric) -> list[GroupStat]:
    bucket = type_bucket(scored)
    groups = _split(scored, lambda r: bucket[r.call_id])
    if len(groups) < 2:
        return []
    out = [_group(label, label, members, [r for r in scored if bucket[r.call_id] != label],
                  rubric, {"type": label})
           for label, members in groups.items()]
    return sorted(out, key=lambda g: (-(g.mean.mean or 0), g.label))


def duration_bucket(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    for key, _, lo, hi in DURATION_BUCKETS:
        if lo <= seconds < hi:
            return key
    return None


def _duration_segments(scored: list[CallRecord], rubric: Rubric) -> list[GroupStat]:
    groups = _split(scored, lambda r: duration_bucket(r.duration_sec))
    if len(groups) < 2:
        return []
    labels = {key: label for key, label, _, _ in DURATION_BUCKETS}
    out = []
    for key, _, _, _ in DURATION_BUCKETS:
        if key in groups:
            rest = [r for r in scored if duration_bucket(r.duration_sec) != key]
            out.append(_group(key, labels[key], groups[key], rest, rubric, {"dur": key}))
    return out


def _layout_segments(scored: list[CallRecord], rubric: Rubric) -> list[GroupStat]:
    groups = _split(scored, lambda r: None if r.mono is None else ("mono" if r.mono else "stereo"))
    if len(groups) < 2:
        return []
    labels = {"stereo": "סטריאו — בנקאי ולקוח בערוצים נפרדים",
              "mono": "חד-ערוצי — זיהוי דוברים אוטומטי"}
    return [_group(key, labels[key], groups[key],
                   [r for r in scored if r.mono is not None and (r.mono != (key == "mono"))],
                   rubric, {"layout": key})
            for key in ("stereo", "mono") if key in groups]


def _bankers(scored: list[CallRecord], rubric: Rubric
             ) -> tuple[list[GroupStat], float | None]:
    """Each banker against the TYPICAL banker - the median of the other
    bankers' means - rather than against "everyone else".

    Against everyone else, one very weak banker drags the rest's mean down and
    ordinary bankers come out "significantly better"; with a gate-capped
    minority, most of the team did. The median of banker means is not moved
    by an outlier, and it is the reference line the chart draws.
    """
    groups = _split(scored, lambda r: r.banker_id)
    stats_by: dict[str, GroupStat] = {}
    for banker_id, members in groups.items():
        rest = [r for r in scored if r.banker_id != banker_id]
        stats_by[banker_id] = _group(banker_id, banker_id, members, rest, rubric,
                                     {"banker": banker_id}, link=members[0].banker_report)
    eligible = {b: g.mean.mean for b, g in stats_by.items()
                if g.n >= MIN_GROUP_N and g.mean.mean is not None}
    reference = stats.quantile(list(eligible.values()), 0.5) if eligible else None
    for banker_id, g in stats_by.items():
        others = [m for b, m in eligible.items() if b != banker_id]
        ref = stats.quantile(others, 0.5) if len(others) >= 2 else None
        totals = [r.card.weighted_total for r in groups[banker_id]]
        ci = stats.mean_ci(totals, GROUP_LEVEL)
        if ref is None or ci.mean is None or ci.low is None or ci.high is None:
            g.diff = None
        else:
            g.diff = stats.DiffCI(n1=len(totals), n2=len(others), diff=ci.mean - ref,
                                  low=ci.low - ref, high=ci.high - ref,
                                  significant=not (ci.low <= ref <= ci.high))
        g.flag = _flag(g.diff, g.n)
    out = sorted(stats_by.values(), key=lambda g: (-(g.mean.mean or 0), g.key))
    return out, reference


# (key, Hebrew label, unit, value getter). Interruptions only exist on
# two-channel recordings; a single-channel call's zero is structural.
DRIVERS = (
    ("talk_ratio", "חלקו של הבנקאי בזמן הדיבור", "%",
     lambda f: f.talk_ratio * 100),
    ("interruptions", "קטיעות של הלקוח בידי הבנקאי (לדקה)", "לדקה",
     lambda f: (f.interruptions_by_banker / (f.call_duration_sec / 60.0))
     if f.overlap_metrics_available and f.call_duration_sec > 0 else None),
    ("patience", "זמן המתנה חציוני לפני מענה", "שנ'",
     lambda f: f.patience_median_sec),
    ("questions", "שאלות של הבנקאי לדקה", "לדקה",
     lambda f: f.banker_questions_per_minute),
    ("monologue", "המונולוג הארוך ביותר של הבנקאי", "שנ'",
     lambda f: f.longest_banker_monologue_sec),
    ("dead_air", "שקט בשיחה (אחוז מהמשך)", "%",
     lambda f: (f.dead_air_total_sec / f.call_duration_sec * 100)
     if f.call_duration_sec > 0 else None),
    ("duration", "משך השיחה", "דק'",
     lambda f: f.call_duration_sec / 60.0),
)


def _drivers(scored: list[CallRecord]) -> list[Driver]:
    out: list[Driver] = []
    for key, label, unit, getter in DRIVERS:
        pairs = []
        for r in scored:
            if r.features is None:
                continue
            try:
                value = getter(r.features)
            except (TypeError, ZeroDivisionError):
                value = None
            if value is not None:
                pairs.append((float(value), r.card.weighted_total))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        corr = stats.spearman(xs, ys, LEVEL)
        q1 = stats.quantile(xs, 0.25)
        q3 = stats.quantile(xs, 0.75)
        low = [y for x, y in pairs if q1 is not None and x <= q1]
        high = [y for x, y in pairs if q3 is not None and x >= q3]
        contrast = (stats.welch_diff(high, low, LEVEL)
                    if len(low) >= 2 and len(high) >= 2 and q1 != q3 else None)
        reportable = bool(
            len(pairs) >= MIN_DRIVER_N and corr.rho is not None and corr.significant
            and abs(corr.rho) >= MIN_DRIVER_RHO and contrast is not None
            and contrast.diff is not None and contrast.significant
            and abs(contrast.diff) >= MIN_PRACTICAL_DIFF)
        out.append(Driver(
            key=key, label=label, unit=unit, n=len(pairs), corr=corr, q1=q1, q3=q3,
            low_mean=_mean(low), high_mean=_mean(high), contrast=contrast,
            reportable=reportable,
            filter_high={"feature": key, "min": q3} if q3 is not None else {},
        ))
    return sorted(out, key=lambda d: (not d.reportable, -abs(d.corr.rho or 0), d.key))


def _gates(scored: list[CallRecord], rubric: Rubric) -> list[GateStat]:
    out: list[GateStat] = []
    n = len(scored)
    bucket = type_bucket(scored)
    for d in rubric.dimensions:
        if not d.gate:
            continue
        failing = [r for r in scored if d.id in r.card.failed_gates]
        per_banker = Counter(r.banker_id for r in failing)
        calls_per_banker = Counter(r.banker_id for r in scored)
        ranked = sorted(per_banker.items(), key=lambda kv: (-kv[1], kv[0]))
        concentration = call_share = None
        top = max(1, round(len(calls_per_banker) * 0.2))
        if len(failing) >= 10 and len(calls_per_banker) >= 5:
            conc = sum(k for _, k in ranked[:top]) / len(failing)
            share = sum(calls_per_banker[b] for b, _ in ranked[:top]) / n
            # Concentrated means more of the failures than of the calls: a
            # group that takes 60% of the calls "holding" 55% of the failures
            # is not a concentration.
            if conc - share >= 0.15:
                concentration, call_share = conc, share
        per_type = Counter(bucket[r.call_id] for r in failing)
        calls_per_type = Counter(bucket.values())
        by_type = sorted(((t, per_type.get(t, 0), c) for t, c in calls_per_type.items()),
                         key=lambda x: (-(x[1] / x[2] if x[2] else 0), x[0]))
        out.append(GateStat(
            id=d.id, name=d.name_he, fails=len(failing),
            rate=stats.proportion_ci(len(failing), n, LEVEL),
            top_bankers=[(b, k, calls_per_banker[b]) for b, k in ranked[:5]],
            concentration=concentration, by_type=by_type, top_n=top,
            n_bankers=len(calls_per_banker), call_share=call_share,
        ))
    return out


def _examples(scored: list[CallRecord], rubric: Rubric
              ) -> tuple[dict[str, list[Example]], dict[str, list[Example]]]:
    """Per dimension, the clearest exemplary and problematic moments - only from
    calls whose text may be shown, and only where the judge quoted evidence."""
    best: dict[str, list[Example]] = {}
    worst: dict[str, list[Example]] = {}
    for d in rubric.dimensions:
        pool = []
        for r in scored:
            if not r.text_allowed:
                continue
            entry = r.card.scores.get(d.id)
            if entry is None or not entry.evidence:
                continue
            ev = entry.evidence[0]
            pool.append(Example(
                call_id=r.call_id, banker_id=r.banker_id, dim=d.id, score=entry.score,
                total=r.card.weighted_total, quote=ev.quote, timestamp=ev.timestamp,
                speaker=ev.speaker, reasoning=entry.reasoning_he, call_report=r.call_report,
            ))
        tops = sorted((e for e in pool if e.score == 5), key=lambda e: (-e.total, e.call_id))
        lows = sorted((e for e in pool if e.score <= 2), key=lambda e: (e.score, e.total,
                                                                        e.call_id))
        best[d.id] = _distinct_bankers(tops, 3)
        worst[d.id] = _distinct_bankers(lows, 3)
    return best, worst


def _distinct_bankers(examples: list[Example], k: int) -> list[Example]:
    """Prefer examples from different bankers: a case library of one person's
    calls teaches less and singles that person out."""
    picked: list[Example] = []
    seen: set[str] = set()
    for e in examples:
        if e.banker_id not in seen:
            picked.append(e)
            seen.add(e.banker_id)
        if len(picked) == k:
            return picked
    for e in examples:
        if e not in picked:
            picked.append(e)
        if len(picked) == k:
            break
    return picked


def _coverage(batch: Batch) -> dict:
    held = Counter(reason for r in batch.held for reason in r.held_reasons)
    failed = Counter(r.failed_reason or "כשל" for r in batch.failed)
    entities: Counter[str] = Counter()
    for r in batch.records:
        entities.update(r.redaction_counts)
    return {
        "status": {"success": len(batch.scored), "needs_human_review": len(batch.held),
                   "failed": len(batch.failed)},
        "held_reasons": held.most_common(),
        "failed_reasons": failed.most_common(),
        "entities": entities.most_common(),
        "entities_total": sum(entities.values()),
        "redaction_disabled": sum(1 for r in batch.records if r.redaction_disabled),
        "text_withheld": sum(1 for r in batch.scored if not r.text_allowed),
        "date_sources": Counter(r.date_source for r in batch.scored),
        "mono": sum(1 for r in batch.scored if r.mono),
        "excluded_other_rubric": batch.excluded_other_rubric,
        "calibration": batch.calibration,
    }
