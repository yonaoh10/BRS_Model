"""The executive report's statistics must match independent references.

Management decisions rest on these intervals, so each estimator is pinned to
values from a source other than this module: printed t tables, closed forms,
textbook examples, and numbers computed once with scipy/numpy (neither of
which the module uses) and hard-coded here.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime

import numpy as np
import pytest

from callqa.reporting.executive.stats import (
    Corr,
    DiffCI,
    MeanCI,
    PropCI,
    Trend,
    _t_crit_exact,
    choose_granularity,
    histogram,
    linear_trend,
    mean_ci,
    period_label,
    period_start,
    proportion_ci,
    quantile,
    spearman,
    t_crit,
    welch_diff,
    z_crit,
)

NAN = float("nan")

# Standard printed tables (two-sided), typed independently of the module's constants.
TABLE_95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
            2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
            2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
TABLE_99 = [63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169,
            3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845,
            2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771, 2.763, 2.756, 2.750]


# --------------------------------------------------------------------------- t / z


@pytest.mark.parametrize("df", range(1, 31))
def test_t_crit_matches_the_printed_tables(df: int) -> None:
    assert round(t_crit(df, 0.95), 3) == TABLE_95[df - 1]
    assert round(t_crit(df, 0.99), 3) == TABLE_99[df - 1]
    assert t_crit(float(df), 0.95) == t_crit(df, 0.95)   # 10.0 is still a table df


def test_t_crit_spec_values() -> None:
    assert t_crit(10, 0.95) == pytest.approx(2.228, abs=5e-4)
    assert t_crit(1000, 0.95) == pytest.approx(1.962339081, abs=1e-6)
    assert t_crit(5, 0.99) == pytest.approx(4.032, abs=5e-4)
    assert t_crit(10) == t_crit(10, 0.95)                  # default level is 95%


def test_exact_solver_reproduces_the_table() -> None:
    """The incomplete-beta solver used for non-table df is an independent
    computation; agreeing with every table entry cross-checks both."""
    for df in range(1, 31):
        assert _t_crit_exact(float(df), 0.95) == pytest.approx(t_crit(df, 0.95), abs=1e-8)
        assert _t_crit_exact(float(df), 0.99) == pytest.approx(t_crit(df, 0.99), abs=1e-8)


def test_t_crit_closed_forms_for_df_1_and_2() -> None:
    # df=1 is Cauchy: t = tan(pi*level/2); df=2: t = level*sqrt(2/(1-level^2)).
    # The small levels are a regression: 1 - P(|T| > t) cancelled, so at df=1
    # a 1e-12 level came out as 1.05e-8 instead of 1.57e-12.
    for level in (1e-12, 1e-6, 0.01, 0.3, 0.5, 0.8, 0.9, 0.975, 0.999):
        cauchy = math.tan(math.pi * level / 2)
        assert t_crit(1, level) == pytest.approx(cauchy, rel=1e-9, abs=0)
        assert t_crit(2, level) == pytest.approx(level * math.sqrt(2 / (1 - level**2)),
                                                 rel=1e-9, abs=0)


@pytest.mark.parametrize(("df", "level", "expected"), [
    (1.5, 0.95, 6.016663104427929),
    (3.5, 0.95, 2.9400886379827282),
    (3.5, 0.99, 5.085702223091256),
    (24.988529290231416, 0.95, 2.059586491365625),   # a Welch df
])
def test_t_crit_non_integer_small_df_is_exact(df: float, level: float, expected: float) -> None:
    """Welch's df is rarely an integer; Cornish-Fisher would be ~1% off at 3.5."""
    assert t_crit(df, level) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(("df", "level", "expected"), [
    (31, 0.95, 2.039513446396408),
    (31, 0.99, 2.744041919294269),
    (40, 0.95, 2.021075390306273),
    (100, 0.99, 2.6258905214380173),
    (1000, 0.95, 1.9623390808264083),
])
def test_t_crit_cornish_fisher_above_30(df: float, level: float, expected: float) -> None:
    assert t_crit(df, level) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("level", [0.9, 0.95, 0.99, 0.999])
def test_t_crit_is_continuous_and_monotone_across_the_df_30_seam(level: float) -> None:
    table_30 = t_crit(30, level)
    assert abs(t_crit(30 + 1e-9, level) - table_30) < 1e-5    # table -> expansion
    assert abs(t_crit(30 - 1e-9, level) - table_30) < 1e-5    # exact solver -> table
    assert t_crit(29.5, level) > table_30 > t_crit(30.5, level) > t_crit(31, level)
    grid = [1 + i * 0.05 for i in range(0, 1200)]             # df 1 .. 61
    values = [t_crit(df, level) for df in grid]
    assert all(b <= a for a, b in zip(values, values[1:], strict=False))
    assert values[-1] > z_crit(level)


def test_t_crit_edges() -> None:
    assert t_crit(0) == math.inf
    assert t_crit(-3) == math.inf
    assert t_crit(NAN) == math.inf
    assert t_crit(math.inf, 0.95) == z_crit(0.95)
    assert t_crit(1e9, 0.99) == pytest.approx(z_crit(0.99), abs=1e-8)
    for bad in (0.0, 1.0, 1.5, -0.1, NAN):
        with pytest.raises(ValueError):
            t_crit(10, bad)
        with pytest.raises(ValueError):
            z_crit(bad)


def test_z_crit() -> None:
    assert z_crit(0.95) == pytest.approx(1.959963984540054, abs=1e-12)
    assert z_crit(0.99) == pytest.approx(2.5758293035489, abs=1e-12)


# --------------------------------------------------------------------------- mean_ci


def test_mean_ci_one_to_ten() -> None:
    # sd = sqrt(sum((i-5.5)^2)/9) = sqrt(82.5/9); half = t(9) * sd / sqrt(10)
    ci = mean_ci(list(range(1, 11)))
    assert ci.n == 10 and ci.mean == 5.5
    assert ci.sd == pytest.approx(math.sqrt(82.5 / 9), abs=1e-12)
    assert ci.low == pytest.approx(3.334149410331831, abs=1e-8)
    assert ci.high == pytest.approx(7.665850589668169, abs=1e-8)
    ci99 = mean_ci(range(1, 11), level=0.99)
    assert ci99.low == pytest.approx(2.3885193567296983, abs=1e-8)
    assert ci99.high == pytest.approx(8.611480643270301, abs=1e-8)


def test_mean_ci_small_n_and_degenerate() -> None:
    assert mean_ci([]) == MeanCI(0, None, None, None, None)
    assert mean_ci([72.5]) == MeanCI(1, 72.5, None, None, None)
    # three 0.1s sum to 0.30000000000000004: the sd must still be exactly 0
    const = mean_ci([0.1, 0.1, 0.1])
    assert const == MeanCI(3, 0.1, 0.0, 0.1, 0.1)


def test_mean_ci_drops_none_and_nan() -> None:
    assert mean_ci([1, None, 2, NAN, 3, np.float64("nan")]) == mean_ci([1, 2, 3])
    assert mean_ci([None, NAN]) == MeanCI(0, None, None, None, None)


def test_mean_ci_matches_numpy_on_a_large_sample() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(70, 15, 1000)
    ci = mean_ci(data.tolist())
    assert ci.mean == pytest.approx(float(data.mean()), abs=1e-10)
    assert ci.sd == pytest.approx(float(data.std(ddof=1)), abs=1e-10)
    half = t_crit(999) * float(data.std(ddof=1)) / math.sqrt(1000)
    assert ci.high - ci.mean == pytest.approx(half, abs=1e-10)


# --------------------------------------------------------------------------- proportion_ci


def test_wilson_reference_values() -> None:
    ci = proportion_ci(5, 20)
    assert ci.p == 0.25
    assert ci.low == pytest.approx(0.11186170140766571, abs=1e-10)
    assert ci.high == pytest.approx(0.468700877618744, abs=1e-10)
    ci99 = proportion_ci(3, 50, 0.99)
    assert ci99.low == pytest.approx(0.015294840009215757, abs=1e-10)
    assert ci99.high == pytest.approx(0.2077990007022969, abs=1e-10)


def test_wilson_boundaries_are_exact() -> None:
    zero = proportion_ci(0, 10)
    assert zero.low == 0.0 and zero.p == 0.0
    assert zero.high == pytest.approx(0.27753279986288926, abs=1e-10)
    full = proportion_ci(10, 10, 0.99)
    assert full.high == 1.0
    assert full.low == pytest.approx(0.6011459066950919, abs=1e-10)


def test_wilson_contains_p_and_stays_in_unit_interval() -> None:
    for n in (1, 2, 5, 17, 100):
        for k in range(n + 1):
            ci = proportion_ci(k, n)
            assert 0.0 <= ci.low <= ci.p <= ci.high <= 1.0


def test_proportion_ci_small_and_invalid() -> None:
    assert proportion_ci(0, 0) == PropCI(0, 0, None, None, None)
    for k, n in ((-1, 5), (6, 5), (1, 0), (0, -1)):
        with pytest.raises(ValueError):
            proportion_ci(k, n)


# --------------------------------------------------------------------------- welch_diff

# Wikipedia, "Welch's t-test", example 1: t = -2.46, df = 24.99, p = 0.021.
A1 = [27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6, 23.1, 19.6, 19.0, 21.7, 21.4]
A2 = [27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2, 21.9, 22.1, 22.9, 20.5, 24.4]


def test_welch_textbook_example() -> None:
    d = welch_diff(A1, A2)
    assert (d.n1, d.n2) == (15, 15)
    assert d.diff == pytest.approx(-2.1666666666666667, abs=1e-12)
    assert d.low == pytest.approx(-3.984096267140929, abs=1e-8)
    assert d.high == pytest.approx(-0.3492370661924069, abs=1e-8)
    assert d.significant                        # p = 0.021 < 0.05
    # the published t statistic follows from the interval: diff / se
    se = (d.high - d.low) / 2 / t_crit(24.988529290231416)
    assert d.diff / se == pytest.approx(-2.455356398286006, abs=1e-6)
    d99 = welch_diff(A1, A2, 0.99)
    assert not d99.significant                  # ... but not at 1%
    assert d99.low == pytest.approx(-4.626460461214432, abs=1e-8)
    assert d99.high == pytest.approx(0.2931271278810961, abs=1e-8)


def test_welch_unequal_sizes_and_symmetry() -> None:
    b1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b2 = [3, 3.5, 9, 12.5, 4, 15, 2, 20]
    d = welch_diff(b1, b2)
    assert d.diff == -3.125
    assert d.low == pytest.approx(-8.828108114091991, abs=1e-8)
    assert d.high == pytest.approx(2.5781081140919913, abs=1e-8)
    assert not d.significant
    flipped = welch_diff(b2, b1)
    assert flipped.diff == -d.diff
    assert flipped.low == pytest.approx(-d.high) and flipped.high == pytest.approx(-d.low)


def test_welch_one_constant_group_uses_the_other_groups_df() -> None:
    # se^2 = var([1..4])/4 and df = 4 - 1 = 3
    d = welch_diff([5, 5, 5], [1, 2, 3, 4])
    assert d.diff == 2.5
    assert d.low == pytest.approx(0.4457397432394794, abs=1e-9)
    assert d.high == pytest.approx(4.554260256760521, abs=1e-9)
    assert d.significant


def test_welch_small_n_and_zero_variance() -> None:
    assert welch_diff([], [1, 2]) == DiffCI(0, 2, None, None, None, False)
    assert welch_diff([3], [1, 2]) == DiffCI(1, 2, 1.5, None, None, False)
    assert welch_diff([4, 4], [1, 1, 1]) == DiffCI(2, 3, 3.0, 3.0, 3.0, True)
    assert welch_diff([2, 2], [2, 2, 2]) == DiffCI(2, 3, 0.0, 0.0, 0.0, False)
    assert welch_diff([1, None, 2, NAN], [5, 6]) == welch_diff([1, 2], [5, 6])


# --------------------------------------------------------------------------- quantile


def test_quantile_matches_numpy_type_7() -> None:
    assert quantile([4, 1, 3, 2], 0.25) == 1.75
    assert quantile([4, 1, 3, 2], 0.5) == 2.5
    assert quantile([4, 1, 3, 2], 0.9) == pytest.approx(3.7, abs=1e-12)
    rng = np.random.default_rng(3)
    for size in (1, 2, 5, 11, 200):
        data = rng.uniform(-50, 150, size).tolist()
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1.0):
            assert quantile(data, q) == pytest.approx(float(np.quantile(data, q)), abs=1e-12)


def test_quantile_edges() -> None:
    assert quantile([], 0.5) is None
    assert quantile([None, NAN], 0.5) is None
    assert quantile([7], 0.3) == 7
    assert quantile([5, None, 1, NAN, 3], 0.5) == 3
    assert quantile([5, 1, 3], 0.0) == 1 and quantile([5, 1, 3], 1.0) == 5
    for bad in (-0.01, 1.01, NAN):
        with pytest.raises(ValueError):
            quantile([1, 2], bad)


# --------------------------------------------------------------------------- spearman


def test_spearman_monotone_is_one() -> None:
    x = list(range(1, 21))
    up = spearman(x, [v**3 for v in x])            # non-linear but monotone
    assert up.rho == 1.0 and up.significant and up.low == up.high == 1.0
    down = spearman(x, [-math.exp(v / 5) for v in x])
    assert down.rho == -1.0 and down.significant


def test_spearman_ties_use_average_ranks() -> None:
    # reference: scipy.stats.spearmanr for rho; CI = tanh(atanh(rho) -+ z*se) with the
    # Bonett-Wright se = sqrt((1 + rho^2/2) / (n - 3))
    c = spearman([10, 20, 20, 30, 40, 50, 50, 60], [3, 1, 4, 1, 5, 9, 2, 6])
    assert c.n == 8
    assert c.rho == pytest.approx(0.527282411152491, abs=1e-12)
    assert c.low == pytest.approx(-0.33556947246642543, abs=1e-10)
    assert c.high == pytest.approx(0.9090175078079343, abs=1e-10)
    assert not c.significant
    assert spearman([1, 2, 2, 3, 4], [1, 3, 2, 4, 4]).rho == pytest.approx(0.9473684210526317)


def test_spearman_significance_follows_the_interval() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=300)
    y = 0.4 * x + rng.normal(size=300)
    c = spearman(x.tolist(), y.tolist())
    assert c.significant and c.low > 0
    z = math.atanh(c.rho)
    se = math.sqrt((1 + c.rho ** 2 / 2) / 297)          # Bonett & Wright (2000)
    assert c.low == pytest.approx(math.tanh(z - z_crit(0.95) * se), abs=1e-12)
    noise = spearman(x.tolist(), rng.normal(size=300).tolist())
    assert noise.significant == (noise.low > 0 or noise.high < 0)


def test_spearman_missing_small_and_constant() -> None:
    c = spearman([1, None, 3, 4, NAN, 6, 7], [2, 5, None, 8, 9, 1, 3])
    assert c.n == 4                                 # only complete pairs count
    assert c == spearman([1, 4, 6, 7], [2, 8, 1, 3])
    assert spearman([1, 2, 3], [3, 1, 2]) == Corr(3, None, None, None, False)
    assert spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) == Corr(5, None, None, None, False)
    assert spearman([1, 2, 3, 4, 5], [2, 2, 2, 2, 2]).rho is None
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


# --------------------------------------------------------------------------- linear_trend


def test_linear_trend_perfect_line() -> None:
    days = list(range(0, 70, 7))
    tr = linear_trend(days, [50 + 0.5 * d for d in days])
    assert tr.n == 10 and tr.span_days == 63
    assert tr.slope_per_30d == pytest.approx(15.0, abs=1e-9)
    assert tr.low == pytest.approx(15.0, abs=1e-6) and tr.high == pytest.approx(15.0, abs=1e-6)
    assert tr.significant
    flat = linear_trend(days, [70.0] * 10)
    assert flat.slope_per_30d == 0.0 and not flat.significant


def test_linear_trend_reference_regression() -> None:
    # reference: scipy.stats.linregress slope/stderr with t(8), times 30
    days = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]
    ys = [60, 62, 61, 65, 64, 66, 70, 69, 71, 73]
    tr = linear_trend(days, ys)
    assert tr.slope_per_30d == pytest.approx(6.155844155844156, abs=1e-9)
    assert tr.low == pytest.approx(4.898023718460065, abs=1e-8)
    assert tr.high == pytest.approx(7.413664593228247, abs=1e-8)
    assert tr.significant
    # absolute day numbers (ordinals) give the same slope as offsets
    shifted = linear_trend([d + date(2026, 6, 1).toordinal() for d in days], ys)
    assert shifted.slope_per_30d == pytest.approx(tr.slope_per_30d, abs=1e-9)
    assert shifted.low == pytest.approx(tr.low, abs=1e-8)


def test_linear_trend_constant_values_is_exactly_flat() -> None:
    # regression: the mean of 3 x 0.1 is not exactly 0.1, which used to leave
    # a slope of about -1.4e-31 instead of 0
    assert linear_trend([0, 0.7, 1.4], [0.1] * 3) == Trend(3, 1.4, 0.0, 0.0, 0.0, False)
    assert linear_trend([0.0, 1.3, 2.9, 7.0], [1 / 3] * 4).slope_per_30d == 0.0


def test_linear_trend_small_and_degenerate() -> None:
    assert linear_trend([], []) == Trend(0, 0.0, None, None, None, False)
    assert linear_trend([1, 5], [60, 70]) == Trend(2, 4.0, None, None, None, False)
    assert linear_trend([3, 3, 3, 3], [60, 70, 80, 90]) == Trend(4, 0.0, None, None, None, False)
    tr = linear_trend([0, 1, None, 2, 3], [1, 2, 99, NAN, 4])
    assert tr.n == 3 and tr.span_days == 3
    with pytest.raises(ValueError):
        linear_trend([1, 2, 3], [1, 2])


# --------------------------------------------------------------------------- histogram


def test_histogram_bin_edges() -> None:
    edges = [0, 40, 60, 80, 100]
    values = [0, 39.99, 40, 59.9, 60, 79.9, 80, 100, 100.1, -1, None, NAN]
    assert histogram(values, edges) == [2, 2, 2, 2]  # 100 lands in the last bin
    assert histogram([], edges) == [0, 0, 0, 0]
    assert histogram([5, 5, 5], [0, 5, 10]) == [0, 3]
    assert histogram([10], [0, 5, 10]) == [0, 1]
    assert histogram([1, 2], [0]) == [] and histogram([1, 2], []) == []
    for bad in ([0, 0, 1], [0, 2, 1]):
        with pytest.raises(ValueError):
            histogram([1], bad)


def test_histogram_counts_every_in_range_value_once() -> None:
    rng = np.random.default_rng(5)
    data = rng.uniform(-10, 110, 1000).tolist()
    edges = [float(e) for e in range(0, 101, 10)]
    counts = histogram(data, edges)
    assert sum(counts) == sum(1 for v in data if 0 <= v <= 100)
    ref, _ = np.histogram([v for v in data if 0 <= v <= 100], bins=edges)
    assert counts == ref.tolist()


# --------------------------------------------------------------------------- periods


def test_choose_granularity() -> None:
    d0 = date(2026, 6, 1)
    assert choose_granularity(d0, d0) == "day"
    assert choose_granularity(d0, date(2026, 6, 22)) == "day"      # 21 days
    assert choose_granularity(d0, date(2026, 6, 23)) == "week"     # 22 days
    assert choose_granularity(d0, date(2026, 11, 30)) == "week"    # 182 days = 26 weeks
    assert choose_granularity(d0, date(2026, 12, 1)) == "month"    # 183 days
    assert choose_granularity(date(2026, 6, 22), d0) == "day"      # order does not matter


def test_period_start_week_begins_on_sunday() -> None:
    sunday, saturday = date(2026, 9, 20), date(2026, 9, 26)
    assert sunday.weekday() == 6 and saturday.weekday() == 5
    assert period_start(sunday, "week") == sunday
    assert period_start(date(2026, 9, 24), "week") == sunday          # Thursday
    assert period_start(saturday, "week") == sunday                    # last day of that week
    assert period_start(date(2026, 9, 27), "week") == date(2026, 9, 27)  # next Sunday
    assert period_start(date(2026, 1, 1), "week") == date(2025, 12, 28)  # across a year
    assert period_start(date(2026, 9, 24), "day") == date(2026, 9, 24)
    assert period_start(date(2026, 9, 24), "month") == date(2026, 9, 1)
    start = period_start(datetime(2026, 9, 24, 15, 30), "week")
    assert type(start) is date and start == sunday
    with pytest.raises(ValueError):
        period_start(sunday, "year")


def test_period_label() -> None:
    assert period_label(date(2026, 9, 24), "day") == "24.09"
    assert period_label(date(2026, 9, 20), "week") == "20.09"
    assert period_label(date(2026, 9, 24), "week") == "20.09"          # normalised to Sunday
    assert period_label(date(2026, 9, 1), "month") == "09/2026"
    assert period_label(date(2026, 3, 5), "day") == "05.03"


# --------------------------------------------------------------------------- general


def test_results_are_deterministic_and_frozen() -> None:
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=50).tolist(), rng.normal(size=50).tolist()
    assert mean_ci(x) == mean_ci(list(x))
    assert spearman(x, y) == spearman(list(x), list(y))
    assert welch_diff(x, y) == welch_diff(x, y)
    assert linear_trend(list(range(50)), y) == linear_trend(list(range(50)), y)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mean_ci(x).mean = 0.0  # type: ignore[misc]


def test_accepts_ints_tuples_and_numpy_scalars() -> None:
    assert mean_ci((1, 2, 3)) == mean_ci([1.0, 2.0, 3.0])
    assert mean_ci([np.int64(1), np.float64(2), 3]) == mean_ci([1, 2, 3])
    assert quantile(np.array([3.0, 1.0, 2.0]), 0.5) == 2.0


INF = math.inf


def test_infinite_values_count_as_missing() -> None:
    # regression: inf used to give sd/CI = nan, raise in fsum (inf + -inf),
    # make quantile return nan, and mark a NaN trend as significant
    assert mean_ci([1, INF, 2, -INF, 3]) == mean_ci([1, 2, 3])
    assert mean_ci([INF, -INF]) == MeanCI(0, None, None, None, None)
    assert welch_diff([1, 2, INF], [5, -INF, 6]) == welch_diff([1, 2], [5, 6])
    assert quantile([INF, 1, 2], 0.5) == 1.5
    assert histogram([INF, 1, -INF], [0, 5, INF]) == [1, 0]
    x, y = [1, 2, 3, 4, 5, INF], [2, 1, 4, 3, 5, 6]
    assert spearman(x, y) == spearman(x[:5], y[:5])
    tr = linear_trend([0, 1, 2, INF], [1, 2, 2, 3])
    assert tr == linear_trend([0, 1, 2], [1, 2, 2])
    assert tr.n == 3 and math.isfinite(tr.slope_per_30d) and not tr.significant


def test_huge_values_scale_instead_of_overflowing() -> None:
    # regression: (v - mean) ** 2 raised OverflowError from about 1e154 on;
    # the intervals must simply scale with the data
    k = 1e200
    base, big = mean_ci(A1), mean_ci([v * k for v in A1])
    assert big.mean == pytest.approx(base.mean * k, rel=1e-12)
    assert big.low == pytest.approx(base.low * k, rel=1e-12)
    assert big.high == pytest.approx(base.high * k, rel=1e-12)
    d, dk = welch_diff(A1, A2), welch_diff([v * k for v in A1], [v * k for v in A2])
    assert dk.low == pytest.approx(d.low * k, rel=1e-12)
    assert dk.high == pytest.approx(d.high * k, rel=1e-12)
    assert dk.significant == d.significant
    days = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]
    ys = [60, 62, 61, 65, 64, 66, 70, 69, 71, 73]
    tr, trk = linear_trend(days, ys), linear_trend(days, [v * k for v in ys])
    assert trk.slope_per_30d == pytest.approx(tr.slope_per_30d * k, rel=1e-12)
    assert trk.low == pytest.approx(tr.low * k, rel=1e-12)
    assert trk.significant


def test_tiny_variances_do_not_divide_by_zero() -> None:
    # regression: se1**2 underflowed to 0 in the Welch df -> ZeroDivisionError.
    # One constant group: df = n1 - 1 = 1, half = t(1) * sd1 / sqrt(2) = t(1) * 5e-161
    d = welch_diff([0, 1e-160], [0, 0])
    # abs=0: approx's default abs=1e-12 would accept anything this small
    assert d.diff == pytest.approx(5e-161, rel=1e-12, abs=0)
    assert d.low == pytest.approx(5e-161 - 12.706204736 * 5e-161, rel=1e-9, abs=0)
    assert d.high == pytest.approx(5e-161 + 12.706204736 * 5e-161, rel=1e-9, abs=0)
    assert not d.significant


def test_values_beyond_float_range_give_none_not_inf_or_nan() -> None:
    assert quantile([-1e308, 1e308], 0.0) == -1e308           # was nan (inf * 0)
    assert quantile([-1e308, 1e308], 0.5) == 0.0              # was -inf
    assert quantile([-1e308, 1e308], 1.0) == 1e308
    wide = mean_ci([1e308, -1e308, 1e308])                   # sd itself overflows
    assert wide.n == 3 and wide.mean == pytest.approx(1e308 / 3)
    assert (wide.sd, wide.low, wide.high) == (None, None, None)
    assert welch_diff([1e308, 1.7e308], [-1e308, -1.7e308]) == DiffCI(
        2, 2, None, None, None, False)                          # the diff overflows
    tr = linear_trend([0, 1, 2, 3], [1e308, -1e308, 1e308, -1e308])
    assert tr.n == 4 and tr.slope_per_30d is None and not tr.significant


@pytest.mark.parametrize("sample", [
    [1e308, -1e308, 5.0, 1e308],
    [1.7e308, 1.7e308, -1.7e308, 0.0, 3.0],
    [1e200, 0.0, 5.0, -3e199],
    [5e-324, 1e-323, 0.0, 2e-323],
    [1e-300, -1e-300, 1e300, 7.0],
])
def test_extreme_magnitudes_never_raise_or_leak_non_finite_numbers(sample: list[float]) -> None:
    other = [v / 3 - 1.0 for v in reversed(sample)]
    days = [float(i * 3) for i in range(len(sample))]
    results = [
        mean_ci(sample), mean_ci(sample, 0.99), welch_diff(sample, other),
        welch_diff(other, sample, 0.99), spearman(sample, other), spearman(days, sample),
        linear_trend(days, sample), linear_trend(sample, other),
    ]
    for res in results:
        for field in dataclasses.fields(res):
            value = getattr(res, field.name)
            if isinstance(value, float) and field.name != "span_days":
                assert math.isfinite(value), (res, field.name)
    for q in (0.0, 0.1, 0.5, 0.77, 1.0):
        assert math.isfinite(quantile(sample, q))
    assert sum(histogram(sample, [-1e308, 0.0, 1e308])) <= len(sample)


def test_invalid_level_raises_even_on_empty_input() -> None:
    for fn in (lambda: mean_ci([], 1.0), lambda: welch_diff([], [], 0),
               lambda: spearman([], [], 2), lambda: linear_trend([], [], -1),
               lambda: proportion_ci(0, 0, 1.0)):
        with pytest.raises(ValueError):
            fn()
