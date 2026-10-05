"""One object for every share the report prints: k of n, its interval, and
whether it may be shown at all.

Every rate in the journey product goes through `rate_of`, so the gate is in
one place: a share on fewer than `min_n` cases is not shown as a rate (the
count is), below `firm_n` it is preliminary, and the interval is Wilson when
the units are independent or a cluster bootstrap by story when they nest in
stories (returns of one customer depend on each other). A template never
divides k by n itself - it prints what the Rate says.

The n >= 10 / < 30 thresholds are this tool's rule, not the bank's: the bank
prints k of n beside every share, and so does this (text_he).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from callqa.journey.stats import cluster_bootstrap_share
from callqa.reporting.executive.stats import proportion_ci

Method = Literal["wilson", "cluster_bootstrap", "none"]


class Rate(BaseModel):
    model_config = {"extra": "forbid"}

    k: int
    n: int
    value: float | None = None       # k / n, None when n is 0
    low: float | None = None         # 95% interval; None when it cannot be given
    high: float | None = None
    clusters: int | None = None      # stories the units nest in, when they do
    method: Method = "none"
    shown: bool = False              # n >= min_n (and n > 0)
    preliminary: bool = True         # n < firm_n
    min_n: int = 10
    firm_n: int = 30

    @property
    def text_he(self) -> str:
        """How the bank reads a share: the percentage when the base allows
        it, the count otherwise - never an empty cell."""
        if not self.n:
            return "אין מקרים"
        if not self.shown or self.value is None:
            return f"{self.k:,} מתוך {self.n:,}"
        return f"{100.0 * self.value:.1f}%"

    @property
    def interval_he(self) -> str:
        if self.low is None or self.high is None or not self.shown:
            return ""
        return f"{100.0 * max(0.0, self.low):.1f}%–{100.0 * min(1.0, self.high):.1f}%"

    @property
    def why_no_interval_he(self) -> str:
        if self.low is not None or not self.n:
            return ""
        if self.method == "cluster_bootstrap":
            return "אין רווח סמך: פחות מחמישה סיפורים בבסיס"
        return "אין רווח סמך"


def rate_of(k: int, n: int, *, min_n: int = 10, firm_n: int = 30,
            clusters: list[tuple[int, int]] | None = None) -> Rate:
    """The rate k of n with its gate and interval. `clusters` = (k_i, n_i)
    per story when the units nest in stories; then the interval is a cluster
    bootstrap and Wilson is not used."""
    r = Rate(k=k, n=n, min_n=min_n, firm_n=firm_n)
    if not n:
        return r
    r.value = k / n
    r.shown = n >= min_n
    r.preliminary = n < firm_n
    if clusters is not None:
        cs = cluster_bootstrap_share(clusters)
        r.low, r.high, r.clusters = cs.low, cs.high, cs.clusters
        r.method = "cluster_bootstrap"
    else:
        ci = proportion_ci(k, n)
        r.low, r.high = ci.low, ci.high
        r.method = "wilson"
    return r
