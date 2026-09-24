"""Small-sample statistics for the executive batch report.

Management will act on these numbers ("banker X is below the team", "scores
are improving"), so each one has to come with an honest measure of
uncertainty and has to degrade to "not enough data" instead of a confident
number built on three calls. Hence the conventions every function here keeps:

- Never raise on small n: a function that cannot produce an estimate returns
  its result object with ``None`` fields (and ``significant=False``), and the
  report says "לא די נתונים". Invalid arguments (a level outside (0, 1),
  ``k > n``, non-increasing histogram edges) still raise - those are bugs in
  the caller, not thin data.
- ``None`` and NaN mean "missing" everywhere: they are dropped (pairwise for
  the two-variable functions) and the returned ``n`` counts only what was
  actually used, so a missing feature can never turn into a NaN on the page.
- Pure and deterministic: stdlib only, no randomness, no global state.

The t distribution is the one piece that needs care without scipy. The
two-sided critical values for df 1..30 at 95%/99% are hard-coded from the
standard tables (9 decimals, rounding to the familiar 3-decimal values);
above df 30 the Cornish-Fisher expansion is accurate to better than 1e-4.
Below 30 that expansion is not good enough (5% too small at df=1, which would
make a CI look tighter than it is), and Welch's df is rarely an integer, so
every other df <= 30 case is solved exactly from the t CDF (regularised
incomplete beta, inverted by bisection).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from statistics import NormalDist

__all__ = [
    "Corr",
    "DiffCI",
    "MeanCI",
    "PropCI",
    "Trend",
    "choose_granularity",
    "histogram",
    "linear_trend",
    "mean_ci",
    "period_label",
    "period_start",
    "proportion_ci",
    "quantile",
    "spearman",
    "t_crit",
    "welch_diff",
    "z_crit",
]

Values = Sequence[float | None]

# Two-sided critical values t_{1-alpha/2, df} for df = 1..30 (index df-1).
_T95 = (
    12.706204736, 4.302652730, 3.182446305, 2.776445105, 2.570581836,
    2.446911851, 2.364624252, 2.306004135, 2.262157163, 2.228138852,
    2.200985160, 2.178812830, 2.160368656, 2.144786688, 2.131449546,
    2.119905299, 2.109815578, 2.100922040, 2.093024054, 2.085963447,
    2.079613845, 2.073873068, 2.068657610, 2.063898562, 2.059538553,
    2.055529439, 2.051830516, 2.048407142, 2.045229642, 2.042272456,
)
_T99 = (
    63.656741163, 9.924843201, 5.840909310, 4.604094871, 4.032142984,
    3.707428021, 3.499483297, 3.355387331, 3.249835542, 3.169272673,
    3.105806516, 3.054539589, 3.012275839, 2.976842734, 2.946712883,
    2.920781622, 2.898230520, 2.878440473, 2.860934606, 2.845339710,
    2.831359558, 2.818756061, 2.807335684, 2.796939505, 2.787435814,
    2.778714533, 2.770682957, 2.763262455, 2.756385904, 2.749995654,
)
_T_TABLES = ((0.95, _T95), (0.99, _T99))
_TABLE_MAX_DF = 30


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class MeanCI:
    n: int
    mean: float | None
    sd: float | None
    low: float | None
    high: float | None


@dataclass(frozen=True)
class PropCI:
    k: int
    n: int
    p: float | None
    low: float | None
    high: float | None


@dataclass(frozen=True)
class DiffCI:
    n1: int
    n2: int
    diff: float | None
    low: float | None
    high: float | None
    significant: bool


@dataclass(frozen=True)
class Corr:
    n: int
    rho: float | None
    low: float | None
    high: float | None
    significant: bool


@dataclass(frozen=True)
class Trend:
    n: int
    span_days: float
    slope_per_30d: float | None
    low: float | None
    high: float | None
    significant: bool


# --------------------------------------------------------------------------- helpers


def _clean(values: Values) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        f = float(v)
        if not math.isnan(f):
            out.append(f)
    return out


def _clean_pairs(x: Values, y: Values) -> tuple[list[float], list[float]]:
    if len(x) != len(y):
        raise ValueError(f"paired sequences differ in length: {len(x)} != {len(y)}")
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(x, y, strict=True):
        if a is None or b is None:
            continue
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            continue
        xs.append(fa)
        ys.append(fb)
    return xs, ys


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:  # also rejects NaN
        raise ValueError(f"confidence level must be in (0, 1), got {level!r}")


def _mean_var(xs: list[float]) -> tuple[float, float]:
    """Mean and sample variance (n-1), two-pass with fsum.

    A constant sample is special-cased: fsum(xs)/n is not always exactly the
    common value (3 x 0.1 sums to 0.30000000000000004), and a variance of
    1e-34 instead of 0 would make "no spread" look like a tiny but real one.
    """
    n = len(xs)
    if min(xs) == max(xs):
        return xs[0], 0.0
    mean = math.fsum(xs) / n
    return mean, math.fsum((v - mean) ** 2 for v in xs) / (n - 1)


def _excludes_zero(low: float, high: float) -> bool:
    return low > 0.0 or high < 0.0


# --------------------------------------------------------------------------- critical values


def z_crit(level: float) -> float:
    """Two-sided standard-normal critical value (1.959964 for 0.95)."""
    _check_level(level)
    return NormalDist().inv_cdf((1.0 + level) / 2.0)


def t_crit(df: float, level: float = 0.95) -> float:
    """Two-sided Student-t critical value: P(|T_df| <= t) = level.

    df <= 0 (or NaN) returns inf - no interval can be claimed - and df = inf
    is the normal value. See the module docstring for which method serves
    which df.
    """
    _check_level(level)
    if math.isnan(df) or df <= 0:
        return math.inf
    if math.isinf(df):
        return z_crit(level)
    if df <= _TABLE_MAX_DF:
        if df == int(df):
            for table_level, table in _T_TABLES:
                if abs(level - table_level) < 1e-12:
                    return table[int(df) - 1]
        return _t_crit_exact(float(df), float(level))
    return _t_crit_cornish_fisher(df, level)


def _t_crit_cornish_fisher(df: float, level: float) -> float:
    # Abramowitz & Stegun 26.7.5: t_p = x + g1/v + g2/v^2 + g3/v^3 + g4/v^4.
    x = z_crit(level)
    x2 = x * x
    g1 = x * (x2 + 1) / 4
    g2 = x * ((5 * x2 + 16) * x2 + 3) / 96
    g3 = x * (((3 * x2 + 19) * x2 + 17) * x2 - 15) / 384
    g4 = x * ((((79 * x2 + 776) * x2 + 1482) * x2 - 1920) * x2 - 945) / 92160
    return x + g1 / df + g2 / df**2 + g3 / df**3 + g4 / df**4


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_tail(t: float, df: float) -> float:
    """P(|T_df| > t) = I_{df/(df+t^2)}(df/2, 1/2)."""
    return _betainc(df / 2.0, 0.5, 1.0 / (1.0 + t * t / df))


@lru_cache(maxsize=512)
def _t_crit_exact(df: float, level: float) -> float:
    alpha = 1.0 - level
    lo, hi = 1e-9, 1.0
    while _t_two_sided_tail(hi, df) > alpha:
        lo, hi = hi, hi * 2.0
        if hi > 1e150:  # df far below 1: the interval is unbounded in practice
            return math.inf
    # Geometric bisection: t spans orders of magnitude for small df.
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if _t_two_sided_tail(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-15:
            break
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- estimates


def mean_ci(values: Values, level: float = 0.95) -> MeanCI:
    """Mean with a t-based CI; sd is the sample sd (n-1)."""
    _check_level(level)
    xs = _clean(values)
    n = len(xs)
    if n == 0:
        return MeanCI(0, None, None, None, None)
    mean, var = _mean_var(xs)
    if n == 1:
        return MeanCI(1, mean, None, None, None)
    sd = math.sqrt(var)
    half = t_crit(n - 1, level) * sd / math.sqrt(n)
    return MeanCI(n, mean, sd, mean - half, mean + half)


def proportion_ci(k: int, n: int, level: float = 0.95) -> PropCI:
    """Share k/n with a Wilson score interval.

    Wilson rather than p +- z*se: the naive interval collapses to [0, 0] at
    k=0 and leaves [0, 1] at small n, exactly the gate-failure rates a
    report shows for a single banker.
    """
    _check_level(level)
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid proportion: k={k}, n={n}")
    if n == 0:
        return PropCI(k, 0, None, None, None)
    p = k / n
    z = z_crit(level)
    z2n = z * z / n
    denom = 1.0 + z2n
    centre = (p + z2n / 2.0) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2n / (4.0 * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return PropCI(k, n, p, low, high)


def welch_diff(a: Values, b: Values, level: float = 0.95) -> DiffCI:
    """mean(a) - mean(b) with a Welch (unequal-variance) t interval."""
    _check_level(level)
    xa, xb = _clean(a), _clean(b)
    n1, n2 = len(xa), len(xb)
    if n1 == 0 or n2 == 0:
        return DiffCI(n1, n2, None, None, None, False)
    m1, v1 = _mean_var(xa)
    m2, v2 = _mean_var(xb)
    diff = m1 - m2
    if n1 < 2 or n2 < 2:
        return DiffCI(n1, n2, diff, None, None, False)
    se1, se2 = v1 / n1, v2 / n2
    if se1 == 0.0 and se2 == 0.0:
        return DiffCI(n1, n2, diff, diff, diff, diff != 0.0)
    # Welch-Satterthwaite; lies between min(n1, n2) - 1 and n1 + n2 - 2.
    df = (se1 + se2) ** 2 / (se1**2 / (n1 - 1) + se2**2 / (n2 - 1))
    half = t_crit(df, level) * math.sqrt(se1 + se2)
    low, high = diff - half, diff + half
    return DiffCI(n1, n2, diff, low, high, _excludes_zero(low, high))


def quantile(values: Values, q: float) -> float | None:
    """Quantile by linear interpolation (Hyndman-Fan type 7, numpy's default)."""
    if not 0.0 <= q <= 1.0:  # also rejects NaN
        raise ValueError(f"quantile q must be in [0, 1], got {q!r}")
    xs = sorted(_clean(values))
    if not xs:
        return None
    h = (len(xs) - 1) * q
    i = math.floor(h)
    if i >= len(xs) - 1:
        return xs[-1]
    frac = h - i
    lo, hi = xs[i], xs[i + 1]
    # numpy's lerp: interpolate from the nearer end so the result is monotone
    # in q and exact at the data points - identical to np.quantile.
    if frac >= 0.5:
        return hi - (hi - lo) * (1.0 - frac)
    return lo + (hi - lo) * frac


def _average_ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and xs[order[end + 1]] == xs[order[start]]:
            end += 1
        avg = (start + end) / 2.0 + 1.0  # ranks are 1-based
        for j in range(start, end + 1):
            ranks[order[j]] = avg
        start = end + 1
    return ranks


def spearman(x: Values, y: Values, level: float = 0.95) -> Corr:
    """Spearman rank correlation with a Fisher-z CI (se = 1/sqrt(n-3)).

    Pairs with a missing side are dropped; ties get average ranks. None when
    fewer than 4 pairs or either side is constant (rho is undefined).
    """
    _check_level(level)
    xs, ys = _clean_pairs(x, y)
    n = len(xs)
    if n < 4 or min(xs) == max(xs) or min(ys) == max(ys):
        return Corr(n, None, None, None, False)
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mean_rank = (n + 1) / 2.0  # exact for average ranks, ties or not
    sxy = math.fsum((a - mean_rank) * (b - mean_rank) for a, b in zip(rx, ry, strict=True))
    sxx = math.fsum((a - mean_rank) ** 2 for a in rx)
    syy = math.fsum((b - mean_rank) ** 2 for b in ry)
    rho = max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
    if abs(rho) == 1.0:
        # atanh(+-1) is infinite: the Fisher interval degenerates to the point.
        return Corr(n, rho, rho, rho, True)
    z = math.atanh(rho)
    half = z_crit(level) / math.sqrt(n - 3)
    low, high = math.tanh(z - half), math.tanh(z + half)
    return Corr(n, rho, low, high, _excludes_zero(low, high))


def linear_trend(days: Values, values: Values, level: float = 0.95) -> Trend:
    """OLS slope of values on days, reported per 30 days with a t CI (df = n-2).

    ``days`` is any numeric day axis (e.g. date.toordinal()); only
    differences matter. Pairs with a missing side are dropped.
    """
    _check_level(level)
    xs, ys = _clean_pairs(days, values)
    n = len(xs)
    span = (max(xs) - min(xs)) if xs else 0.0
    if n < 3 or span == 0.0:
        return Trend(n, span, None, None, None, False)
    xm = math.fsum(xs) / n
    ym = math.fsum(ys) / n
    sxx = math.fsum((v - xm) ** 2 for v in xs)
    sxy = math.fsum((u - xm) * (v - ym) for u, v in zip(xs, ys, strict=True))
    slope = sxy / sxx
    ssr = math.fsum((v - ym - slope * (u - xm)) ** 2 for u, v in zip(xs, ys, strict=True))
    se = math.sqrt(ssr / (n - 2) / sxx)
    half = t_crit(n - 2, level) * se
    slope30, low, high = slope * 30.0, (slope - half) * 30.0, (slope + half) * 30.0
    significant = _excludes_zero(low, high) if half > 0.0 else slope != 0.0
    return Trend(n, span, slope30, low, high, significant)


def histogram(values: Values, edges: Sequence[float]) -> list[int]:
    """Counts per bin [edges[i], edges[i+1]); the last bin also includes its top edge.

    Values outside [edges[0], edges[-1]] (and missing ones) are ignored.
    """
    bins = len(edges) - 1
    if bins < 1:
        return []
    if any(not edges[i] < edges[i + 1] for i in range(bins)):
        raise ValueError(f"histogram edges must be strictly increasing: {list(edges)}")
    counts = [0] * bins
    top = edges[-1]
    for v in _clean(values):
        if v == top:
            counts[-1] += 1
            continue
        i = bisect_right(edges, v) - 1
        if 0 <= i < bins:
            counts[i] += 1
    return counts


# --------------------------------------------------------------------------- time periods


def _as_date(d: date) -> date:
    return d.date() if isinstance(d, datetime) else d


def choose_granularity(first: date, last: date) -> str:
    """'day' for spans up to 3 weeks, 'week' up to 26 weeks, else 'month'."""
    span = abs((_as_date(last) - _as_date(first)).days)
    if span <= 21:
        return "day"
    if span <= 26 * 7:
        return "week"
    return "month"


def period_start(d: date, granularity: str) -> date:
    """First day of the period containing d; weeks start on Sunday (Israeli business week)."""
    d = _as_date(d)
    if granularity == "day":
        return d
    if granularity == "week":
        # date.weekday(): Monday=0 .. Sunday=6, so Sunday is 0 days back.
        return d - timedelta(days=(d.weekday() + 1) % 7)
    if granularity == "month":
        return d.replace(day=1)
    raise ValueError(f"unknown granularity: {granularity!r}")


def period_label(start: date, granularity: str) -> str:
    """'24.09' for a day, '21.09' (the Sunday) for a week, '09/2026' for a month."""
    s = period_start(start, granularity)
    if granularity == "month":
        return f"{s.month:02d}/{s.year:04d}"
    return f"{s.day:02d}.{s.month:02d}"
