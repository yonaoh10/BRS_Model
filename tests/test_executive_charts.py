"""Executive-report chart primitives: safe to feed untrusted text, readable
without colour or hover, and stable enough to diff.

The charts are inlined into a report opened from file:// on a bank desktop,
so any label that reaches the markup unescaped is script on a manager's
machine, any hex colour is a chart that ignores dark mode and print, and any
nondeterminism is a report that changes between two identical runs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from html.parser import HTMLParser

import pytest
from markupsafe import Markup

from callqa.reporting.executive import charts as C
from callqa.reporting.executive.charts import (
    band_of,
    diverging_bars,
    dot_plot,
    fmt,
    histogram_chart,
    level_class,
    pareto_chart,
    sparkline,
    stacked_levels,
    trend_chart,
)

EVIL = '<script>alert(1)</script> "q" \'s\' & <img src=x onerror=alert(2)>'
LRI, PDI = "⁦", "⁩"


# -- a tiny DOM, enough to check structure the way a browser would -----------------

class _Doc(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
            "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self.stack[-3:]}")
            return
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def all(self, tag: str) -> list[dict[str, str | None]]:
        return [a for t, a in self.elements if t == tag]

    def with_class(self, cls: str) -> list[tuple[str, dict[str, str | None]]]:
        return [(t, a) for t, a in self.elements if cls in (a.get("class") or "").split()]


def parse(markup: str) -> _Doc:
    doc = _Doc()
    doc.feed(str(markup))
    doc.close()
    assert not doc.errors, doc.errors
    assert not doc.stack, f"unclosed: {doc.stack}"
    return doc


# -- realistic inputs -------------------------------------------------------------

def _bins() -> list[tuple[float, float, int]]:
    counts = [0, 0, 1, 10, 49, 124, 249, 321, 176, 70]
    return [(i * 10, i * 10 + 10, c) for i, c in enumerate(counts)]


BANDS = [(0, 40, 1), (40, 60, 2), (60, 80, 3), (80, 100, 4)]


def _levels(label: str = "הקשבה פעילה", filt: dict | None = None) -> list[dict]:
    return [{"label": label, "counts": [40, 110, 260, 380, 210], "note": "ממוצע 3.6",
             "filter": filt if filt is not None else {"dim": "listening"}},
            {"label": "ציות וגילוי נאות", "counts": [5, 90, 300, 400, 205], "note": None,
             "filter": None}]


def _points(n: int = 12, rate: bool = True) -> list[dict]:
    out = []
    for i in range(n):
        mean = 68 + i * 0.5
        out.append({"label": f"{(i % 28) + 1:02d}.09", "mean": mean, "low": mean - 2.5,
                    "high": mean + 2.5, "n": 80 + i, "rate": 0.05 + i / 400 if rate else None})
    return out


def _dot_rows(label: str = "B101", filt: dict | None = None) -> list[dict]:
    return [{"label": label, "mean": 82.4, "low": 78.0, "high": 86.8, "n": 34, "flag": "ok",
             "filter": filt if filt is not None else {"banker": "B101"}, "sub": "34 שיחות"},
            {"label": "B102", "mean": 61.0, "low": 55.1, "high": 66.9, "n": 22, "flag": "bad",
             "filter": None, "sub": None},
            {"label": "B103", "mean": 70.0, "low": None, "high": None, "n": 3, "flag": "muted",
             "filter": None, "sub": "מעט נתונים"}]


def _div_rows(label: str = "יחס דיבור של הבנקאי", filt: dict | None = None) -> list[dict]:
    return [{"label": label, "value": -6.4, "detail": "רבעון עליון מול תחתון",
             "filter": filt if filt is not None else {"feature": "talk_ratio"}},
            {"label": "שאלות לדקה", "value": 3.8, "detail": None, "filter": None}]


# Every chart, called with realistic data, keyed by name.
CHARTS: dict[str, Callable[[], Markup]] = {
    "histogram": lambda: histogram_chart(_bins(), label="התפלגות הציון", x_title="ציון (0–100)",
                                         y_title="מספר שיחות", bands=BANDS,
                                         markers=[(72.4, "חציון"), (80, "יעד")]),
    "stacked": lambda: stacked_levels(_levels(), label="רמות לפי ממד",
                                      level_labels=["1", "2", "3", "4", "5"]),
    "pareto": lambda: pareto_chart([("הקשבה", 3.2), ("אמפתיה", 6.9), ("סגירה", 2.1)],
                                   label="היכן אובדות הנקודות", total_label="סה״כ",
                                   filters=[{"dim": "listening"}, {"dim": "empathy"}, None]),
    "trend": lambda: trend_chart(_points(), label="מגמה", y_title="ציון כולל",
                                 y_range=(40, 100), rate_title="שיעור כשל בשער"),
    "dots": lambda: dot_plot(_dot_rows(), label="לפי בנקאי", reference=71.2,
                             reference_label="ממוצע הבנק", x_range=(40, 100)),
    "diverging": lambda: diverging_bars(_div_rows(), label="גורמים"),
}


# -- fmt / classes ----------------------------------------------------------------

@pytest.mark.parametrize(("value", "digits", "expected"), [
    (None, 1, "—"), (74.0, 1, "74"), (74.25, 1, "74.3"), (74, 1, "74"), (0.05, 1, "0.1"),
    (-0.04, 1, "0"), (-3.25, 1, "-3.3"), (float("nan"), 1, "—"), (float("inf"), 1, "—"),
    (1234.56, 1, "1,234.6"), (0.5, 0, "1"), (2.5, 0, "3"), (99.96, 1, "100"), ("x", 1, "—"),
])
def test_fmt(value: object, digits: int, expected: str) -> None:
    assert fmt(value, digits) == expected  # type: ignore[arg-type]


def test_level_class_rounds_half_up_and_clamps() -> None:
    assert [level_class(v) for v in (1, 1.49, 1.5, 2.5, 4.5, 5, 7, 0)] == [
        "ch-lvl-1", "ch-lvl-1", "ch-lvl-2", "ch-lvl-3", "ch-lvl-5", "ch-lvl-5", "ch-lvl-5",
        "ch-lvl-1"]
    assert level_class(None) == "" and level_class(float("nan")) == ""


def test_band_of_boundaries() -> None:
    assert [band_of(v) for v in (100, 80, 79.99, 60, 59.9, 40, 39.9, 0)] == [4, 4, 3, 3, 2, 2, 1, 1]
    assert band_of(None) == 0 and band_of(float("nan")) == 0


# -- structure every chart shares ---------------------------------------------------

@pytest.mark.parametrize("name", sorted(CHARTS))
def test_chart_is_an_accessible_figure_with_a_table_twin(name: str) -> None:
    out = CHARTS[name]()
    assert isinstance(out, Markup)
    doc = parse(out)
    figure = doc.all("figure")[0]
    assert "ch" in (figure["class"] or "").split()
    svg = doc.all("svg")[0]
    assert svg["class"] == "ch-svg"
    assert svg["role"] in ("img", "group") and svg["aria-label"]
    assert svg["width"] == "100%" and re.fullmatch(r"0 0 \d+(\.\d)? \d+(\.\d)?", svg["viewbox"] or "")
    assert svg["direction"] == "ltr", "axes run left to right; Hebrew text opts into rtl"
    assert doc.all("title"), "native tooltips"
    details = doc.with_class("ch-table")
    assert details and details[0][0] == "details"
    assert "הצג כטבלה" in "".join(doc.text)
    assert doc.all("table") and doc.all("caption") and doc.all("th")


@pytest.mark.parametrize("name", sorted(CHARTS))
def test_no_colour_literals_in_marks(name: str) -> None:
    """Colour comes from page CSS, or dark mode and print cannot restyle it."""
    out = str(CHARTS[name]())
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", out)
    assert not re.search(r"\b(rgb|hsl)a?\(", out)
    for attr in ("fill", "stroke", "stop-color", "color"):
        for value in re.findall(rf'\s{attr}="([^"]*)"', out):
            assert value in ("none", "currentColor"), f'{attr}="{value}"'


@pytest.mark.parametrize("name", sorted(CHARTS))
def test_output_is_deterministic(name: str) -> None:
    assert str(CHARTS[name]()) == str(CHARTS[name]())


@pytest.mark.parametrize("name", sorted(CHARTS))
def test_hebrew_text_is_rtl(name: str) -> None:
    doc = parse(CHARTS[name]())
    texts = [a for t, a in doc.elements if t == "text"]
    assert texts
    # every SVG text that holds Hebrew letters is set right-to-left
    raw = str(CHARTS[name]())
    for m in re.finditer(r"<text([^>]*)>(.*?)</text>", raw):
        if re.search("[֐-׿]", m.group(2)):
            assert 'direction="rtl"' in m.group(1), m.group(0)


def test_markup_width_caps_the_scale_at_the_design_width() -> None:
    doc = parse(pareto_chart([("א", 1.0)], label="x", width=540))
    svg = doc.all("svg")[0]
    assert svg["viewbox"].split()[2] == "540"
    assert "max-width:540px" in (svg["style"] or "")


# -- escaping -----------------------------------------------------------------------

def _evil_calls() -> dict[str, Callable[[], Markup]]:
    f = {"x": EVIL}
    return {
        "sparkline": lambda: sparkline([1, 2], label=EVIL),
        "histogram": lambda: histogram_chart(_bins(), label=EVIL, x_title=EVIL, y_title=EVIL,
                                             bands=BANDS, markers=[(50, EVIL)]),
        "stacked": lambda: stacked_levels([{"label": EVIL, "counts": [1, 2, 3, 4, 5],
                                            "note": EVIL, "filter": f}],
                                          label=EVIL, level_labels=[EVIL] * 5),
        "pareto": lambda: pareto_chart([(EVIL, 2.0)], label=EVIL, unit=EVIL, total_label=EVIL,
                                       filters=[f]),
        "trend": lambda: trend_chart([{"label": EVIL, "mean": 70, "low": 68, "high": 72,
                                       "n": 5, "rate": 0.1}] * 3, label=EVIL, y_title=EVIL,
                                     rate_title=EVIL),
        "dots": lambda: dot_plot([{"label": EVIL, "mean": 70, "low": 60, "high": 80, "n": 9,
                                   "flag": "bad", "filter": f, "sub": EVIL}],
                                 label=EVIL, reference=70, reference_label=EVIL),
        "diverging": lambda: diverging_bars([{"label": EVIL, "value": -2, "detail": EVIL,
                                              "filter": f}], label=EVIL, unit=EVIL),
    }


@pytest.mark.parametrize("name", sorted(_evil_calls()))
def test_every_text_input_is_escaped(name: str) -> None:
    out = str(_evil_calls()[name]())
    assert "<script" not in out and "<img" not in out
    assert "&lt;script&gt;" in out
    doc = parse(out)
    assert {t for t, _ in doc.elements} <= {
        "figure", "svg", "g", "title", "rect", "path", "line", "polyline", "polygon", "circle",
        "text", "tspan", "div", "span", "details", "summary", "table", "caption", "thead",
        "tbody", "tfoot", "tr", "th", "td", "bdi"}
    for _tag, attrs in doc.elements:      # quotes never broke out of an attribute
        assert "onerror" not in attrs and "src" not in attrs
    labels = [a["aria-label"] for _t, a in doc.elements if a.get("aria-label")]
    assert any(EVIL in (lb or "") for lb in labels), "the label reaches assistive tech intact"


@pytest.mark.parametrize("name", ["stacked", "pareto", "dots", "diverging"])
def test_data_filter_is_attribute_escaped_json_that_round_trips(name: str) -> None:
    out = str(_evil_calls()[name]())
    doc = parse(out)
    links = doc.with_class("ch-link")
    assert links, "a row with a filter is clickable"
    for _tag, attrs in links:
        assert json.loads(attrs["data-filter"] or "") == {"x": EVIL}
        assert attrs["tabindex"] == "0" and attrs["role"] == "button" and attrs["aria-label"]
    # raw form: ensure_ascii=False keeps Hebrew readable, quotes are entities
    heb = str(pareto_chart([("א", 1.0)], label="x", filters=[{"סוג": "משכנתאות"}]))
    assert 'data-filter="{&#34;סוג&#34;: &#34;משכנתאות&#34;}"' in heb


def test_markup_arguments_are_reduced_to_text_not_trusted() -> None:
    out = str(pareto_chart([(Markup("<b>הקשבה</b> &amp; אמפתיה"), 2.0)], label=Markup("<i>x</i>")))
    assert "<b>" not in out and "<i>" not in out
    assert "הקשבה &amp; אמפתיה" in out


def test_bidi_controls_in_labels_cannot_reorder_the_chart() -> None:
    out = str(dot_plot([{"label": "B‮101", "mean": 70, "n": 9}], label="x",
                       reference=None, reference_label=""))
    assert "‮" not in out and "B101" in out


# -- empty and missing data ---------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: histogram_chart([], label="x", x_title="a", y_title="b"),
    lambda: histogram_chart([(0, 10, 0), (10, 20, 0)], label="x", x_title="a", y_title="b"),
    lambda: stacked_levels([], label="x", level_labels=[]),
    lambda: pareto_chart([], label="x"),
    lambda: trend_chart([], label="x", y_title="y"),
    lambda: trend_chart([{"label": "01.09", "mean": None, "n": 0}], label="x", y_title="y"),
    lambda: dot_plot([], label="x", reference=None, reference_label=""),
    lambda: diverging_bars([], label="x"),
])
def test_empty_input_renders_a_placeholder(call: Callable[[], Markup]) -> None:
    out = call()
    doc = parse(out)
    assert isinstance(out, Markup)
    assert "אין נתונים" in "".join(doc.text)
    assert doc.all("figure") and doc.with_class("ch-empty")


def test_sparkline_gaps_break_the_line_and_empty_is_graceful() -> None:
    doc = parse(sparkline([70, 71, None, 72, 73, 74], label="ציון ממוצע"))
    assert len(doc.all("polyline")) == 2
    assert doc.all("title") and doc.all("svg")[0]["role"] == "img"
    assert "74" in doc.all("svg")[0]["aria-label"]
    empty = parse(sparkline([None, None], label="x"))
    assert "אין נתונים" in "".join(empty.text)
    single = parse(sparkline([70], label="x", y_range=(0, 100)))
    assert len(single.all("circle")) == 2 and not single.all("polyline")


def test_missing_values_do_not_raise() -> None:
    pts = _points(6)
    pts[2].update(mean=None, low=None, high=None, n=0, rate=None)
    pts[4].update(low=None, high=None)
    doc = parse(trend_chart(pts, label="x", y_title="y", rate_title="שיעור"))
    assert len(doc.with_class("ch-line")) >= 2, "a missing period breaks the line"
    parse(dot_plot([{"label": "B1", "mean": None, "n": 0}], label="x", reference=50,
                   reference_label="r"))
    parse(diverging_bars([{"label": "a", "value": None}, {"label": "b", "value": 0}], label="x"))
    parse(pareto_chart([("a", None), ("b", 0.0)], label="x"))
    parse(stacked_levels([{"label": "a", "counts": [0, 0, 0, 0, 0]},
                          {"label": "b", "counts": [1, None, "x"]}],
                         label="x", level_labels=["1", "2"]))


# -- chart-specific behaviour -----------------------------------------------------------

def test_histogram_labels_counts_and_drops_the_axis_when_few_bins() -> None:
    doc = parse(CHARTS["histogram"]())
    lbls = ["".join(doc.text)]
    assert "321" in lbls[0]
    assert len(doc.with_class("ch-bar")) == 8, "empty bins draw no bar"
    assert {f"ch-band-{i}" for i in range(1, 5)} <= {
        a["class"] for _t, a in doc.with_class("ch-band-1") + doc.with_class("ch-band-2")
        + doc.with_class("ch-band-3") + doc.with_class("ch-band-4")}
    assert len(doc.with_class("ch-ref")) == 2
    assert "חציון 72.4" in "".join(doc.text)
    many = parse(histogram_chart([(i, i + 2, 5 + i) for i in range(0, 100, 2)], label="x",
                                 x_title="a", y_title="b"))
    assert many.with_class("ch-grid"), "too many bins to label: counts go on an axis"


def test_stacked_levels_read_right_to_left_from_level_1() -> None:
    raw = str(CHARTS["stacked"]())
    first_row = raw.split('class="ch-lvl-5"')[0]
    x1 = float(re.search(r'<g class="ch-lvl-1"><rect x="([\d.]+)"', first_row).group(1))
    x5 = float(re.search(r'<g class="ch-lvl-5"><rect x="([\d.]+)"', raw).group(1))
    assert x1 > x5
    doc = parse(raw)
    inside = re.findall(r'<text class="ch-lbl"[^>]*>([^<]*)</text>', raw)
    assert "38%" in inside, "a wide segment carries its share"
    assert "4%" not in inside and "1%" not in inside, "segments under 8% leave it to the tooltip"
    legend = doc.with_class("ch-legend-item")
    assert len(legend) == 5 and len(doc.with_class("ch-sw")) == 5
    assert doc.all("svg")[0]["role"] == "group", "rows are buttons, so the svg is not an img"


def test_pareto_sorts_desc_keeps_filters_aligned_and_accumulates_to_100() -> None:
    doc = parse(CHARTS["pareto"]())
    links = doc.with_class("ch-link")
    assert [json.loads(a["data-filter"] or "") for _t, a in links] == [
        {"dim": "empathy"}, {"dim": "listening"}]
    text = "".join(doc.text)
    assert text.index("אמפתיה") < text.index("הקשבה") < text.index("סגירה")
    assert "מצטבר 100%" in text
    assert "סה״כ: 12.2 נק'" in text


def test_trend_has_rate_panel_thinned_labels_and_latest_label() -> None:
    raw = str(trend_chart(_points(21, rate=False), label="x", y_title="y", width=360))
    svg = raw.split("</svg>")[0]
    shown = re.findall(r">(\d\d\.09)</text>", svg)
    assert shown[-1] == "21.09", "the newest period always keeps its label"
    assert 2 <= len(shown) < 21, "labels are thinned, not dropped"
    assert not parse(raw).with_class("ch-line-2")
    with_rate = parse(CHARTS["trend"]())
    assert with_rate.with_class("ch-line-2") and with_rate.with_class("ch-area")
    assert "שיעור כשל בשער" in "".join(with_rate.text)
    assert len(with_rate.with_class("ch-legend-item")) == 4


def test_dot_plot_flags_reference_and_value_column() -> None:
    doc = parse(CHARTS["dots"]())
    assert doc.with_class("ch-dot-ok") and doc.with_class("ch-dot-bad")
    assert doc.with_class("ch-dot-muted")
    assert doc.with_class("ch-ref") and "ממוצע הבנק 71.2" in "".join(doc.text)
    text = "".join(doc.text)
    assert "▲" in text and "▼" in text, "the flag does not rely on colour alone"
    assert "70.0" in text, "value column keeps one decimal so it aligns"
    assert len(doc.with_class("ch-ci")) == 2
    rows = parse(dot_plot([{"label": f"B{i}", "mean": 60 + i / 2, "n": 9} for i in range(20)],
                          label="x", reference=None, reference_label=""))
    assert not rows.with_class("ch-ref")


def test_diverging_signs_colours_and_isolated_numbers() -> None:
    raw = str(CHARTS["diverging"]())
    doc = parse(raw)
    assert doc.with_class("ch-bar-bad") and doc.with_class("ch-bar-ok")
    assert f"{LRI}\u22126.4{PDI} נק&#39;" in raw, "a signed number is bidi-isolated"
    assert f"{LRI}+3.8{PDI} נק&#39;" in raw
    one_sided = str(diverging_bars([{"label": "a", "value": -3}, {"label": "b", "value": -1}],
                                   label="x"))
    assert "ch-bar-ok" not in one_sided


def test_bidi_isolation_only_where_the_algorithm_would_reorder() -> None:
    assert C._bidi("ציון (0–100)") == f"ציון ({LRI}0–100{PDI})"
    assert C._bidi("3–6 דק'") == f"{LRI}3–6{PDI} דק'"
    assert C._bidi("2026-09-17") == f"{LRI}2026-09-17{PDI}"
    for safe in ("74.3", "62%", "21.09", "09/2026", "B-101", "מצטבר 62%"):
        assert C._bidi(safe) == safe


def test_text_width_estimate_is_monotone_and_errs_wide() -> None:
    assert C._tw("") == 0
    assert C._tw("אבג") < C._tw("אבגד")
    assert C._tw("100", bold=True) > C._tw("100")
    assert C._tw("ויןם") < C._tw("שששש"), "narrow Hebrew letters are narrow"
    # DejaVu Sans bold digits (the widest fallback) measure 0.696em in Chromium
    assert C._tw("88.8", 11, bold=True) >= 0.95 * (3 * 0.696 + 0.38) * 11
    assert C._fit("זיהוי ואימות לקוח", 40).endswith("…")
    assert C._tw(C._fit("זיהוי ואימות לקוח", 40)) <= 40
