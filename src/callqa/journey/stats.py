"""Statistics the journey report needs beyond reporting/executive/stats.py.

Returns within one story are not independent - a customer stuck in a loop
contributes several failed returns at once - so a share of returns gets a
cluster bootstrap interval (resampling whole stories), not a Wilson one that
would pretend every return is a separate draw. Time to resolution is a
Kaplan-Meier curve: a story still open at the end of the data is censored,
not counted as resolved and not dropped.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ClusteredShare:
    k: int
    n: int
    p: float | None
    low: float | None
    high: float | None
    clusters: int


def cluster_bootstrap_share(clusters: list[tuple[int, int]], *, level: float = 0.95,
                            reps: int = 2000, seed: int = 7) -> ClusteredShare:
    """A share k/n pooled over clusters of (k_i, n_i), with a percentile
    interval from resampling clusters. Deterministic for a given seed."""
    clusters = [(k, n) for k, n in clusters if n > 0]
    k = sum(c[0] for c in clusters)
    n = sum(c[1] for c in clusters)
    if n == 0:
        return ClusteredShare(0, 0, None, None, None, 0)
    p = k / n
    if len(clusters) < 5:
        return ClusteredShare(k, n, p, None, None, len(clusters))
    rng = random.Random(seed)
    draws = []
    m = len(clusters)
    for _ in range(reps):
        kk = nn = 0
        for _j in range(m):
            ck, cn = clusters[rng.randrange(m)]
            kk += ck
            nn += cn
        draws.append(kk / nn if nn else p)
    draws.sort()
    alpha = (1 - level) / 2
    lo = draws[max(0, int(math.floor(alpha * reps)))]
    hi = draws[min(reps - 1, int(math.ceil((1 - alpha) * reps)) - 1)]
    return ClusteredShare(k, n, p, lo, hi, m)


@dataclass(frozen=True)
class KMStep:
    t: float
    survival: float        # share still unresolved after t
    at_risk: int
    events: int
    censored: int


def kaplan_meier(durations: list[float], resolved: list[bool]) -> list[KMStep]:
    """Kaplan-Meier estimate of 'still unresolved' over time. A step at every
    distinct time with an event or a censoring."""
    if len(durations) != len(resolved):
        raise ValueError("durations and resolved differ in length")
    data = sorted(zip(durations, resolved, strict=True))
    at_risk = len(data)
    s = 1.0
    steps = [KMStep(0.0, 1.0, at_risk, 0, 0)]
    i = 0
    while i < len(data):
        t = data[i][0]
        events = censored = 0
        while i < len(data) and data[i][0] == t:
            if data[i][1]:
                events += 1
            else:
                censored += 1
            i += 1
        if events and at_risk:
            s *= 1 - events / at_risk
        steps.append(KMStep(t, s, at_risk, events, censored))
        at_risk -= events + censored
    return steps


def km_median(steps: list[KMStep]) -> float | None:
    """The first time the curve reaches one half, or None if it never does."""
    for step in steps:
        if step.survival <= 0.5:
            return step.t
    return None


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
