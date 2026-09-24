"""Inline SVG chart primitives for the executive ("management") batch report.

Why hand-built SVG instead of a charting library: the report is one
self-contained HTML file that has to open from file:// on an offline Windows
VDI in Edge, print cleanly, and stay readable with scripting disabled.
Server-side SVG does all three and adds no dependency; a JS charting bundle
does none of them.

Every function here keeps a few rules, and the page relies on them:

* Colour lives only in the page stylesheet. Marks carry ``ch-*`` classes and
  never a colour literal, so light/dark mode and print restyle every chart
  together.
* Every text argument is untrusted plain text and is escaped here, where it
  is written out. Callers never pre-escape: a Markup argument is reduced to
  its text. ``data-filter`` payloads are JSON (``ensure_ascii=False``) escaped
  for a double-quoted attribute, so they come back unchanged from the DOM.
* Geometry is laid out for a design ``width`` in CSS pixels. The SVG is
  ``width="100%"`` over a viewBox and capped at that width, so it can shrink
  with its column but its 11px labels never grow. Pass the width of the
  column the chart is placed in.
* Numbers run left to right on every axis, as they do in Hebrew print. Hebrew
  text uses ``direction="rtl"``. A signed number or a numeric range inside
  visible text is wrapped in Unicode isolates. Without them the bidi
  algorithm would print "-3.2" as "3.2-" and turn "60–70" around.
* No value can be read only by hovering or only by colour. Every chart is a
  ``<figure class="ch">`` with an accessible name, native ``<title>``
  tooltips, and a ``<details class="ch-table">`` table of the same numbers.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from markupsafe import Markup, escape

__all__ = [
    "band_of", "diverging_bars", "dot_plot", "fmt", "histogram_chart", "level_class",
    "pareto_chart", "sparkline", "stacked_levels", "trend_chart",
]

DEFAULT_WIDTH = 720      # ~A4 print width and a comfortable single-column width
EMPTY_HE = "אין נתונים"
TABLE_SUMMARY_HE = "הצג כטבלה"
SHOW_CALLS_HE = "הצג את השיחות"
ITEM_HE = "פריט"
CI_HE = "רווח סמך 95%"
BAND_NAMES_HE = {1: "חלש", 2: "גבולי", 3: "בינוני", 4: "גבוה"}

# The page stylesheet sets these sizes and wins over the attributes written
# here. Layout measures text at the same sizes so labels fit where drawn.
_FS_TICK = 10.5          # ch-tick, ch-ref-label
_FS_TEXT = 11.0          # plain text, ch-lbl (600 weight), ch-title
_BOLD = 1.1

_LRI, _PDI = "⁦", "⁩"
_MINUS = "−"
_ELLIPSIS = "…"
_BREAKS = re.compile(r"[\t\n\r\f\v]+")
# Other C0 controls, plus bidi marks/overrides/isolates: a label must not
# reorder the text around it or break the isolates added here.
_CONTROLS = re.compile("[\x00-\x1f\x7f‎‏‪-‮⁦-⁩]")
# Numeric runs that the bidi algorithm reorders in RTL text: a leading sign
# or a range dash. "74.3", "62%" and "21.09" display correctly as they are.
_NUM = r"\d+(?:[.,:/]\d+)*(?:\s?%)?"
_DASHED = rf"\s?[–\-{_MINUS}]\s?[+\-{_MINUS}]?{_NUM}"
_BIDI_RISK = re.compile(
    rf"(?<![\w.])(?:[+\-{_MINUS}]{_NUM}(?:{_DASHED})*|{_NUM}(?:{_DASHED})+)")
_ZERO_WIDTH = frozenset(f"{_LRI}{_PDI}​")


# -- public helpers ---------------------------------------------------------------

def fmt(value: float | None, digits: int = 1) -> str:
    """At most ``digits`` decimals, half-up, thousands grouped: 74.0 -> '74'.

    Half-up on the shortest decimal repr, because that is what a reader
    expects: Python's round(74.25, 1) gives 74.2 (binary float plus banker's
    rounding), and a report that shows 74.2 next to a table saying 74.25 looks
    wrong. Missing, NaN and infinite values are an em dash.
    """
    v = _num(value)
    if v is None:
        return "—"
    places = max(0, int(digits))
    try:
        q = Decimal(repr(v)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    except InvalidOperation:     # beyond Decimal's default precision
        q = Decimal(f"{v:.{places}f}")
    if q.is_zero():
        q = abs(q)               # never '-0'
    text = f"{q:,.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def level_class(score: float | None) -> str:
    """'ch-lvl-1'..'ch-lvl-5' for a 1..5 score or mean (half-up); '' when missing."""
    v = _num(score)
    if v is None:
        return ""
    return f"ch-lvl-{min(5, max(1, math.floor(v + 0.5)))}"


def band_of(total: float | None) -> int:
    """Score band of a 0-100 total: 4 >= 80, 3 >= 60, 2 >= 40, 1 below; 0 when missing."""
    v = _num(total)
    if v is None:
        return 0
    if v >= 80:
        return 4
    if v >= 60:
        return 3
    if v >= 40:
        return 2
    return 1


# -- charts -----------------------------------------------------------------------

def sparkline(values: Sequence[float | None], *, label: str, width: int = 120,
              height: int = 32, y_range: tuple[float, float] | None = None) -> Markup:
    """A word-sized trend for a KPI tile. It returns a bare ``<svg>``, not a
    figure: the tile's own number is the accessible value, and a data table
    inside a tile would crowd it. None values break the line."""
    name = _plain(label)
    w, h = max(24, int(width)), max(12, int(height))
    vals = [_num(v) for v in values]
    present = [v for v in vals if v is not None]
    head = (f'<svg class="ch-svg ch-spark" viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{w}px" direction="ltr" fill="currentColor" role="img" ')
    if not present:
        return Markup(
            f'{head}aria-label="{_attr(f"{name}: {EMPTY_HE}")}"><title>{_vis(name)}: '
            f'{EMPTY_HE}</title><text x="{_c(w / 2)}" y="{_c(h / 2)}" dy="0.35em" '
            f'font-size="10" fill-opacity="0.6" text-anchor="middle" direction="rtl">'
            f'{EMPTY_HE}</text></svg>')
    lo, hi = _num_range(y_range) or (min(present), max(present))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 3.5                    # end-dot radius plus its ring
    n = len(vals)

    def px(i: int) -> float:
        return w / 2 if n == 1 else pad + (w - 2 * pad) * i / (n - 1)

    def py(v: float) -> float:
        return pad + (hi - _clamp(v, lo, hi)) / (hi - lo) * (h - 2 * pad)

    parts: list[str] = []
    for run in _runs([(i, v) for i, v in enumerate(vals)]):
        if len(run) == 1:
            i, v = run[0]
            parts.append(f'<circle class="ch-dot" cx="{_c(px(i))}" cy="{_c(py(v))}" r="1.5"/>')
        else:
            pts = " ".join(f"{_c(px(i))},{_c(py(v))}" for i, v in run)
            parts.append(f'<polyline class="ch-line" points="{pts}" fill="none" '
                         f'stroke="currentColor" vector-effect="non-scaling-stroke"/>')
    last_i, last = [(i, v) for i, v in enumerate(vals) if v is not None][-1]
    parts.append(f'<circle class="ch-dot" cx="{_c(px(last_i))}" cy="{_c(py(last))}" r="2.5"/>')
    summary = (f"{name}: {fmt(last)} (טווח {fmt(min(present))}–{fmt(max(present))}, "
               f"{len(present)} נקודות)")
    return Markup(f'{head}aria-label="{_attr(summary)}"><title>{_vis(summary)}</title>'
                  f'{"".join(parts)}</svg>')


def histogram_chart(bins: Sequence[tuple[float, float, int]], *, label: str, x_title: str,
                    y_title: str, bands: Sequence[tuple[float, float, int]] = (),
                    markers: Sequence[tuple[float, str]] = (),
                    width: int = DEFAULT_WIDTH) -> Markup:
    """Distribution of calls over a numeric range, usually the 0-100 total.

    bands shade the background behind the bars (``ch-band-1..4``) and are
    named along the top. markers are vertical reference lines, labelled
    "<label> <value>" (the value is appended, so pass "חציון", not "חציון 74").
    Bars carry their counts when every count fits on its bar, and then the
    y axis is dropped as redundant ink. Otherwise the counts go on a right-hand
    axis, which is where a Hebrew reader starts.
    """
    name, xt, yt = _plain(label), _plain(x_title), _plain(y_title)
    band_list = list(bands)
    clean: list[tuple[float, float, int]] = []
    for b in bins:
        lo, hi, count = _num(b[0]), _num(b[1]), _num(b[2])
        if lo is not None and hi is not None and hi > lo:
            clean.append((lo, hi, max(0, int(count or 0))))
    total = sum(c for _, _, c in clean)
    if not clean or total == 0:
        return _empty(name)
    clean.sort(key=lambda b: b[0])
    W = _width(width)
    dom_lo, dom_hi = clean[0][0], max(hi for _, hi, _ in clean)
    top_count = max(c for _, _, c in clean)

    pad_l = 8.0
    slot_min = min(hi - lo for lo, hi, _ in clean) / (dom_hi - dom_lo) * (W - 60)
    labelled = len(clean) <= 16 and slot_min - 2 >= _tw(str(top_count), bold=True) + 2
    if labelled:
        y_top, y_ticks = float(top_count), []
    else:
        y_top, y_ticks = _nice_axis(top_count, 4, integer=True)
    tick_w = max((_tw(fmt(t, 0), _FS_TICK) for t in y_ticks), default=0.0)
    plot_r = W - (tick_w + 12 if y_ticks else 8)

    def x(v: float) -> float:
        return pad_l + (_clamp(v, dom_lo, dom_hi) - dom_lo) / (dom_hi - dom_lo) * (plot_r - pad_l)

    # vertical rhythm: y title, marker labels (up to two staggered rows), plot
    rows_y = [28.0, 44.0]
    placed = _place_labels(
        [(x(v), f"{_plain(t)} {fmt(v)}", v) for v, t in _clean_markers(markers, dom_lo, dom_hi)],
        pad_l, plot_r, rows=len(rows_y))
    plot_top = (rows_y[max(r for r, *_ in placed)] + 10) if placed else 20.0
    strip = 17.0 if band_list else 0.0
    bars_top = plot_top + strip + (15.0 if labelled else 4.0)
    base = plot_top + strip + 168.0
    H = base + 42.0

    def y(v: float) -> float:
        return base - (v / y_top) * (base - bars_top) if y_top else base

    deco: list[str] = []
    deco.append(_text(W - 2, 8, yt, cls="ch-title", align="right", rtl=True))
    marker_xs = [mx for _, _, _, mx, _ in placed]
    for b in band_list:
        blo, bhi, idx = _num(b[0]), _num(b[1]), int(_num(b[2]) or 0)
        if blo is None or bhi is None or idx not in BAND_NAMES_HE:
            continue
        bx0, bx1 = x(max(blo, dom_lo)), x(min(bhi, dom_hi))
        if bx1 - bx0 < 1:
            continue
        deco.append(f'<rect class="ch-band-{idx}" x="{_c(bx0)}" y="{_c(plot_top)}" '
                    f'width="{_c(bx1 - bx0)}" height="{_c(base - plot_top)}"/>')
        bname = BAND_NAMES_HE[idx]
        bx = _band_label_x(bx0, bx1, _tw(bname, _FS_TICK), marker_xs)
        if bx is not None:
            deco.append(_text(bx, plot_top + 9, bname, cls="ch-tick", align="middle", rtl=True))
    for t in y_ticks:
        ty = y(t)
        deco.append(_hline(pad_l, plot_r, ty, "ch-grid" if t else "ch-axis"))
        deco.append(_text(plot_r + 6, ty, fmt(t, 0), cls="ch-tick", size=_FS_TICK))
    if not y_ticks:
        deco.append(_hline(pad_l, plot_r, base, "ch-axis"))
    x_ticks = _ticks(dom_lo, dom_hi, max(2, int((plot_r - pad_l) / 56)))
    for t in x_ticks:
        deco.append(_text(x(t), base + 12, fmt(t), cls="ch-tick", size=_FS_TICK, align="middle"))
    deco.append(_text((pad_l + plot_r) / 2, base + 31, xt, cls="ch-title", align="middle",
                      rtl=True))

    marks: list[str] = []
    hits: list[str] = []
    for lo, hi, count in clean:
        x0, x1 = x(lo) + 1, x(hi) - 1          # the 2px surface gap between bars
        if count and x1 > x0:
            marks.append(f'<path class="ch-bar" d="{_vbar(x0, x1, y(count), base, 2.5)}"/>')
            if labelled:
                marks.append(_text((x0 + x1) / 2, y(count) - 8, fmt(count, 0), cls="ch-lbl",
                                   align="middle"))
        tip = (f"{fmt(lo)}–{fmt(hi)}: {_calls(count)} ({_pct(count / total)})")
        hits.append(f'<g><title>{_vis(tip)}</title><rect x="{_c(x(lo))}" y="{_c(plot_top)}" '
                    f'width="{_c(max(1.0, x(hi) - x(lo)))}" height="{_c(base - plot_top)}" '
                    f'fill="none" pointer-events="all"/></g>')
    for row, x0, x1, mx, text in placed:
        ly = rows_y[row]
        beside = x1 <= mx or x0 >= mx
        marks.append(_vline(mx, ly - 6 if beside else ly + 7, base, "ch-ref"))
        marks.append(_text((x0 + x1) / 2, ly, text, cls="ch-ref-label", size=_FS_TICK,
                           align="middle", rtl=True))

    svg = _svg(W, H, name, _group(deco) + "".join(marks) + "".join(hits), interactive=False)
    table = _table(name, [xt, yt, "% מהשיחות"],
                   [[_cell(f"{fmt(lo)}–{fmt(hi)}"), _num_cell(fmt(c, 0)),
                     _num_cell(_pct(c / total))] for lo, hi, c in clean],
                   foot=["סה״כ", fmt(total, 0), "100%"])
    return _figure("hist", svg, table=table)


def stacked_levels(rows: Sequence[Mapping[str, Any]], *, label: str,
                   level_labels: Sequence[str], width: int = DEFAULT_WIDTH) -> Markup:
    """One 100% bar per row showing how its scores split across levels 1..5.

    RTL: the row label sits on the right and each bar reads from level 1 (low)
    at the right to level 5 at the left, matching the legend order. A segment
    shows its percentage only when it is at least 8% of the bar and the text
    fits inside with padding. Smaller segments rely on the tooltip and table.
    """
    name = _plain(label)
    levels = [_plain(t) for t in list(level_labels)[:5]]
    levels += [f"רמה {k}" for k in range(len(levels) + 1, 6)]
    prepared = []
    for r in rows:
        counts = [max(0, int(_num(c) or 0)) for c in list(r.get("counts") or [])[:5]]
        counts += [0] * (5 - len(counts))
        prepared.append((_plain(r.get("label")), counts, _plain(r.get("note")), r.get("filter")))
    if not prepared:
        return _empty(name)
    W = _width(width)
    lab_w = _clamp(max(_tw(lb) for lb, *_ in prepared) + 16, 64, W * 0.32)
    notes = [nt for _, _, nt, _ in prepared if nt]
    note_w = min(max((_tw(nt, _FS_TICK) for nt in notes), default=0) + 14, W * 0.22) if notes else 0
    bar_l, bar_r = note_w + 4, W - lab_w
    pitch, bar_h, top = 30.0, 18.0, 4.0
    H = top + pitch * len(prepared) + 2
    interactive = any(f is not None for *_, f in prepared)

    out: list[str] = []
    table_rows: list[list[str]] = []
    for i, (lb, counts, note, filt) in enumerate(prepared):
        cy = top + pitch * i + pitch / 2
        total = sum(counts)
        shares = [c / total if total else 0.0 for c in counts]
        summary = (" · ".join(f"{levels[k]}: {_pct(shares[k])}" for k in range(5))
                   if total else EMPTY_HE)
        row: list[str] = [_text(W - 4, cy, _fit(lb, lab_w - 12), cls="ch-cat", align="right",
                                rtl=True)]
        if note:
            row.append(_text(bar_l - 8, cy, _fit(note, note_w - 12, _FS_TICK), cls="ch-tick",
                             size=_FS_TICK, align="right", rtl=True))
        if not total:
            row.append(f'<rect class="ch-bar-muted" x="{_c(bar_l)}" y="{_c(cy - bar_h / 2)}" '
                       f'width="{_c(bar_r - bar_l)}" height="{_c(bar_h)}"/>')
            row.append(_text((bar_l + bar_r) / 2, cy, EMPTY_HE, cls="ch-tick", size=_FS_TICK,
                             align="middle", rtl=True))
        drawn = [k for k in range(5) if counts[k]]
        cursor = bar_r
        for k in range(5):
            if not counts[k]:
                continue
            seg = shares[k] * (bar_r - bar_l)
            x1 = cursor - (1 if k != drawn[0] else 0)
            x0 = cursor - seg + (1 if k != drawn[-1] else 0)
            cursor -= seg
            x0 = min(x0, x1 - 0.8)                     # a sliver stays visible
            tip = f"{lb} · {levels[k]}: {_pct(shares[k])} ({_calls(counts[k])})"
            seg_svg = (f'<rect x="{_c(x0)}" y="{_c(cy - bar_h / 2)}" width="{_c(x1 - x0)}" '
                       f'height="{_c(bar_h)}"><title>{_vis(tip)}</title></rect>')
            text = _pct(shares[k])
            if shares[k] >= 0.08 and _tw(text, bold=True) + 8 <= x1 - x0:
                seg_svg += _text((x0 + x1) / 2, cy, text, cls="ch-lbl", align="middle",
                                 extra=' style="fill:currentColor"')
            row.append(f'<g class="ch-lvl-{k + 1}">{seg_svg}</g>')
        out.append(_row_group(row, filt, f"{lb}: {summary}", title=f"{lb} — {summary}",
                              hit=(0, cy - pitch / 2, W, pitch)))
        table_rows.append([_cell(lb)]
                          + [_num_cell(f"{fmt(counts[k], 0)} ({_pct(shares[k])})") for k in range(5)]
                          + [_num_cell(fmt(total, 0)), _cell(note)])
    legend = _legend([("rect", f"ch-lvl-{k + 1}", levels[k]) for k in range(5)])
    head = [ITEM_HE, *levels, "סה״כ", "הערה"]
    svg = _svg(W, H, name, "".join(out), interactive=interactive)
    return _figure("levels", svg, legend=legend, table=_table(name, head, table_rows))


def pareto_chart(items: Sequence[tuple[str, float]], *, label: str, unit: str = "נק'",
                 total_label: str | None = None,
                 filters: Sequence[Mapping[str, Any] | None] | None = None,
                 width: int = DEFAULT_WIDTH) -> Markup:
    """Where the points go: horizontal bars sorted largest first, each with its
    value and the cumulative share of the total up to and including it.

    ``filters`` is aligned with ``items`` as passed (before sorting).
    ``total_label`` captions a footer line, "<total_label>: <sum> <unit>".
    Negative values are drawn as zero-length bars and left out of the shares.
    """
    name, u = _plain(label), _plain(unit)
    filt_list = list(filters or [])
    prepared = []
    for idx, item in enumerate(items):
        v = _num(item[1])
        prepared.append((_plain(item[0]), v, filt_list[idx] if idx < len(filt_list) else None))
    if not prepared:
        return _empty(name)
    # sorted() is stable, so ties keep the caller's order and output is deterministic
    prepared = sorted(prepared, key=lambda p: -(p[1] if p[1] is not None else -math.inf))
    positive = [max(0.0, v) for _, v, _ in prepared if v is not None]
    total = sum(positive)
    vmax = max(positive, default=0.0) or 1.0
    W = _width(width)
    val_texts = [f"{fmt(v)} {u}".strip() for _, v, _ in prepared]
    lab_w = _clamp(max(_tw(lb) for lb, *_ in prepared) + 16, 64, W * 0.32)
    cum_w = _tw("מצטבר 100%", _FS_TICK) + 14
    val_w = max(_tw(t, bold=True) for t in val_texts) + 12
    bar_l, bar_r = cum_w + val_w, W - lab_w
    pitch, bar_h, top = 28.0, 16.0, 4.0
    rows_bottom = top + pitch * len(prepared)
    H = rows_bottom + (26.0 if total_label else 4.0)
    interactive = any(f is not None for *_, f in prepared)

    out = [_group([_vline(bar_r, top + 2, rows_bottom - 2, "ch-axis")])]
    table_rows: list[list[str]] = []
    cum = 0.0
    for i, (lb, v, filt) in enumerate(prepared):
        cy = top + pitch * i + pitch / 2
        share = (max(0.0, v) / total) if (v is not None and total) else None
        cum += share or 0.0
        cum_txt = _pct(cum) if share is not None else "—"
        tip_x = bar_r - (max(0.0, v) / vmax) * (bar_r - bar_l) if v is not None else bar_r
        row = [_text(W - 4, cy, _fit(lb, lab_w - 14), cls="ch-cat", align="right", rtl=True)]
        if v is not None and bar_r - tip_x >= 0.5:
            row.append(f'<path class="ch-bar" d="{_hbar(bar_r, tip_x, cy - bar_h / 2, bar_h, 3)}"/>')
        row.append(_text(tip_x - 6, cy, val_texts[i], cls="ch-lbl", align="right", rtl=True))
        row.append(_text(cum_w - 6, cy, f"מצטבר {cum_txt}", cls="ch-tick", size=_FS_TICK,
                         align="right", rtl=True))
        tip = f"{lb}: {val_texts[i]}"
        if share is not None:
            tip += f" · {_pct(share)} מהסך · מצטבר {cum_txt}"
        out.append(_row_group(row, filt, tip, title=tip, hit=(0, cy - pitch / 2, W, pitch)))
        table_rows.append([_cell(lb), _num_cell(fmt(v)),
                           _num_cell(_pct(share) if share is not None else "—"),
                           _num_cell(cum_txt)])
    foot = None
    if total_label:
        caption = f"{_plain(total_label)}: {fmt(total)} {u}".strip()
        out.append(_group([_text(W - 4, rows_bottom + 12, caption, cls="ch-title", align="right",
                                 rtl=True)]))
        foot = [_plain(total_label), fmt(total), "100%" if total else "—", ""]
    head = [ITEM_HE, f"ערך ({u})" if u else "ערך", "% מהסך", "מצטבר"]
    svg = _svg(W, H, name, "".join(out), interactive=interactive)
    return _figure("pareto", svg, table=_table(name, head, table_rows, foot=foot))


def trend_chart(points: Sequence[Mapping[str, Any]], *, label: str, y_title: str,
                y_range: tuple[float, float] = (0, 100), rate_title: str | None = None,
                width: int = DEFAULT_WIDTH) -> Markup:
    """Mean per period with its confidence band and call volume. Time runs
    left to right, oldest first.

    A secondary rate (0..1, e.g. the share of calls that failed a gate) is
    drawn in its own panel under the mean, sharing the time axis and
    starting its own scale at zero. It is not put on a second y axis over the
    mean: on a shared 0-100 frame a 6% rate is a flat line lying on the
    volume bars, and two scales on one plot suggest a correlation that
    nobody measured. Volume bars sit in a strip of their own at the bottom,
    scaled to the busiest period. X labels are thinned from the newest
    backwards, so the latest period always carries a label.
    """
    name, yt = _plain(label), _plain(y_title)
    pts = []
    for p in points:
        rate = _num(p.get("rate"))
        pts.append({"label": _plain(p.get("label")), "mean": _num(p.get("mean")),
                    "low": _num(p.get("low")), "high": _num(p.get("high")),
                    "n": max(0, int(_num(p.get("n")) or 0)),
                    "rate": None if rate is None else _clamp(rate, 0.0, 1.0)})
    if not pts or all(p["mean"] is None for p in pts):
        return _empty(name)
    rt = _plain(rate_title) if rate_title else ""
    has_rate = bool(rt) and any(p["rate"] is not None for p in pts)
    W = _width(width)
    lo, hi = _num_range(y_range) or (0.0, 100.0)
    y_ticks = _ticks(lo, hi, 5)
    rate_top, rate_ticks = 1.0, []
    if has_rate:
        rate_top, rate_ticks = _nice_axis(
            max(max((p["rate"] or 0.0) for p in pts) * 1.1, 0.02), 2)
    tick_texts = [fmt(t) for t in y_ticks] + [_pct(t) for t in rate_ticks]
    ml, plot_r = 8.0, W - (max(_tw(t, _FS_TICK) for t in tick_texts) + 14)
    n = len(pts)
    slot = (plot_r - ml) / n
    xs = [ml + (i + 0.5) * slot for i in range(n)]

    # Stacked panels sharing one time axis, each titled at its top right like
    # the main one, with a gap wide enough that no panel reads as the
    # continuation of the one above (volume bars under a "0" gridline would
    # look like negative scores).
    top, main_h = 26.0, 168.0
    main_b = top + main_h
    rate_t = main_b + 46.0 if has_rate else main_b
    rate_b = rate_t + 56.0 if has_rate else main_b
    vol_t = rate_b + 44.0
    vol_b = vol_t + 28.0
    H = vol_b + 26.0

    def y(v: float) -> float:
        return main_b - (_clamp(v, lo, hi) - lo) / (hi - lo) * main_h

    def yr(v: float) -> float:
        return rate_b - _clamp(v, 0.0, rate_top) / rate_top * (rate_b - rate_t)

    n_max = max(p["n"] for p in pts) or 1
    deco = [_text(W - 2, 9, yt, cls="ch-title", align="right", rtl=True)]
    for t in y_ticks:
        deco.append(_hline(ml, plot_r, y(t), "ch-axis" if t == y_ticks[0] and t <= lo
                           else "ch-grid"))
        deco.append(_text(plot_r + 7, y(t), fmt(t), cls="ch-tick", size=_FS_TICK))
    if has_rate:
        deco.append(_text(W - 2, rate_t - 17, rt, cls="ch-title", align="right", rtl=True))
        for t in rate_ticks:
            deco.append(_hline(ml, plot_r, yr(t), "ch-axis" if t == 0 else "ch-grid"))
            deco.append(_text(plot_r + 7, yr(t), _pct(t), cls="ch-tick", size=_FS_TICK))
    deco.append(_text(W - 2, vol_t - 17, "מספר שיחות", cls="ch-title", align="right", rtl=True))
    deco.append(_hline(ml, plot_r, vol_t, "ch-grid"))
    deco.append(_text(plot_r + 7, vol_t, fmt(n_max, 0), cls="ch-tick", size=_FS_TICK))
    deco.append(_hline(ml, plot_r, vol_b, "ch-axis"))
    deco.append(_text(plot_r + 7, vol_b, "0", cls="ch-tick", size=_FS_TICK))
    label_w = max(_tw(p["label"], _FS_TICK) for p in pts) + 10
    every = max(1, math.ceil(label_w / slot))
    for i, p in enumerate(pts):
        if (n - 1 - i) % every == 0:
            deco.append(_text(xs[i], vol_b + 12, p["label"], cls="ch-tick", size=_FS_TICK,
                              align="middle", rtl=True))

    marks: list[str] = []
    bar_w = min(slot * 0.6, 16.0)
    for i, p in enumerate(pts):
        if p["n"]:
            top_y = vol_b - p["n"] / n_max * (vol_b - vol_t)
            marks.append(f'<path class="ch-bar-muted" '
                         f'd="{_vbar(xs[i] - bar_w / 2, xs[i] + bar_w / 2, top_y, vol_b, 2)}"/>')
    for run in _runs([(i, (p["low"], p["high"]) if p["mean"] is not None and p["low"] is not None
                       and p["high"] is not None else None) for i, p in enumerate(pts)]):
        if len(run) == 1:
            i, (plo, phi) = run[0]
            marks.append(_vline(xs[i], y(phi), y(plo), "ch-ci"))
        else:
            upper = [f"{_c(xs[i])},{_c(y(b[1]))}" for i, b in run]
            lower = [f"{_c(xs[i])},{_c(y(b[0]))}" for i, b in reversed(run)]
            marks.append(f'<polygon class="ch-area" points="{" ".join(upper + lower)}"/>')
    for run in _runs([(i, p["mean"]) for i, p in enumerate(pts)]):
        if len(run) > 1:
            marks.append(_polyline([(xs[i], y(v)) for i, v in run], "ch-line"))
    show_dots = n <= 26
    last_i = max(i for i, p in enumerate(pts) if p["mean"] is not None)
    for i, p in enumerate(pts):
        if p["mean"] is not None and (show_dots or i == last_i):
            marks.append(f'<circle class="ch-dot" cx="{_c(xs[i])}" cy="{_c(y(p["mean"]))}" '
                         f'r="{4 if i == last_i else 3}"/>')
    last = pts[last_i]
    end_txt = fmt(last["mean"])
    end_w = _tw(end_txt, bold=True)
    end_y = y(last["high"] if last["high"] is not None else last["mean"]) - 9
    if end_y >= top + 2:
        marks.append(_text(min(xs[last_i], plot_r - end_w / 2), end_y, end_txt, cls="ch-lbl",
                           align="middle"))
    else:   # the band touches the top: label beside the dot, clear of the y title
        marks.append(_text(xs[last_i] - 8, y(last["mean"]), end_txt, cls="ch-lbl",
                           align="right"))
    if has_rate:
        for run in _runs([(i, p["rate"]) for i, p in enumerate(pts)]):
            if len(run) > 1:
                marks.append(_polyline([(xs[i], yr(v)) for i, v in run], "ch-line-2"))
        r_last = max(i for i, p in enumerate(pts) if p["rate"] is not None)
        r_txt = _pct(pts[r_last]["rate"])
        marks.append(_text(min(xs[r_last], plot_r - _tw(r_txt, bold=True) / 2),
                           yr(pts[r_last]["rate"]) - 9, r_txt, cls="ch-lbl", align="middle"))

    hits: list[str] = []
    table_rows: list[list[str]] = []
    for i, p in enumerate(pts):
        ci = f"{fmt(p['low'])}–{fmt(p['high'])}" if p["low"] is not None else "—"
        tip = f"{p['label']}: ממוצע {fmt(p['mean'])} ({CI_HE}: {ci}) · {_calls(p['n'])}"
        if has_rate:
            tip += f" · {rt}: {_pct(p['rate'])}"
        hits.append(f'<g><title>{_vis(tip)}</title><rect x="{_c(xs[i] - slot / 2)}" '
                    f'y="{_c(top)}" width="{_c(slot)}" height="{_c(vol_b - top)}" fill="none" '
                    f'pointer-events="all"/></g>')
        row = [_cell(p["label"]), _num_cell(fmt(p["mean"])), _num_cell(ci),
               _num_cell(fmt(p["n"], 0))]
        if has_rate:
            row.append(_num_cell(_pct(p["rate"])))
        table_rows.append(row)
    legend_items = [("line", "ch-line", "ממוצע לתקופה"), ("rect", "ch-area", CI_HE),
                    ("rect", "ch-bar-muted", "מספר שיחות")]
    head = ["תקופה", yt or "ממוצע", CI_HE, "שיחות"]
    if has_rate:
        legend_items.append(("line", "ch-line-2", rt))
        head.append(rt)
    svg = _svg(W, H, name, _group(deco) + "".join(marks) + "".join(hits), interactive=False)
    return _figure("trend", svg, legend=_legend(legend_items),
                   table=_table(name, head, table_rows))


def dot_plot(rows: Sequence[Mapping[str, Any]], *, label: str, reference: float | None,
             reference_label: str, x_range: tuple[float, float] = (0, 100),
             row_height: int = 22, width: int = DEFAULT_WIDTH) -> Markup:
    """One row per group (the caller sorts): the mean as a dot with its
    confidence interval as a whisker, against a reference line.

    Label column on the right (RTL), then ``sub`` (muted), then the plot. The
    mean is also printed in a value column at the left edge, marked ▼ / ▲
    for flag "bad" / "ok", so the flag does not rely on the dot colour
    alone. Long lists repeat the axis at the top.
    """
    name, ref_name = _plain(label), _plain(reference_label)
    prepared = []
    for r in rows:
        flag = r.get("flag")
        prepared.append({"label": _plain(r.get("label")), "mean": _num(r.get("mean")),
                         "low": _num(r.get("low")), "high": _num(r.get("high")),
                         "n": max(0, int(_num(r.get("n")) or 0)),
                         "flag": flag if flag in ("bad", "ok", "muted") else None,
                         "sub": _plain(r.get("sub")), "filter": r.get("filter")})
    if not prepared:
        return _empty(name)
    W = _width(width)
    rh = float(max(16, int(row_height)))
    x_lo, x_hi = _num_range(x_range) or (0.0, 100.0)
    ref = _num(reference)
    lab_w = _clamp(max(_tw(p["label"]) for p in prepared) + 14, 48, W * 0.28)
    subs = [p["sub"] for p in prepared if p["sub"]]
    sub_w = min(max(_tw(s, _FS_TICK) for s in subs) + 14, W * 0.2) if subs else 0.0
    val_w = max(_tw(f"▼ {_fixed(p['mean'])}", bold=True) for p in prepared) + 8
    plot_l, plot_r = val_w + 12, W - lab_w - sub_w - 12

    def x(v: float) -> float:
        return plot_l + (_clamp(v, x_lo, x_hi) - x_lo) / (x_hi - x_lo) * (plot_r - plot_l)

    x_ticks = _ticks(x_lo, x_hi, max(2, int((plot_r - plot_l) / 60)))
    y_cursor = 4.0
    ref_y = None
    if ref is not None:
        ref_y, y_cursor = y_cursor + 6, y_cursor + 16
    top_ticks_y = None
    if len(prepared) > 14:
        top_ticks_y, y_cursor = y_cursor + 6, y_cursor + 16
    rows_top = y_cursor + 2
    rows_bottom = rows_top + rh * len(prepared)
    H = rows_bottom + 24

    deco: list[str] = []
    for t in x_ticks:
        deco.append(_vline(x(t), rows_top, rows_bottom, "ch-grid"))
        deco.append(_text(x(t), rows_bottom + 12, fmt(t), cls="ch-tick", size=_FS_TICK,
                          align="middle"))
        if top_ticks_y is not None:
            deco.append(_text(x(t), top_ticks_y, fmt(t), cls="ch-tick", size=_FS_TICK,
                              align="middle"))
    if ref is not None and ref_y is not None:
        ref_txt = f"{ref_name} {fmt(ref)}".strip()
        rw = _tw(ref_txt, _FS_TICK)
        deco.append(_vline(x(ref), ref_y + 7, rows_bottom, "ch-ref"))
        deco.append(_text(_clamp(x(ref), plot_l + rw / 2, plot_r - rw / 2), ref_y, ref_txt,
                          cls="ch-ref-label", size=_FS_TICK, align="middle", rtl=True))

    out: list[str] = [_group(deco)]
    table_rows: list[list[str]] = []
    for i, p in enumerate(prepared):
        cy = rows_top + rh * i + rh / 2
        mean, plo, phi = p["mean"], p["low"], p["high"]
        ci = f"{fmt(plo)}–{fmt(phi)}" if plo is not None and phi is not None else "—"
        dot_cls = {"bad": "ch-dot ch-dot-bad", "ok": "ch-dot ch-dot-ok",
                   "muted": "ch-dot ch-dot-muted"}.get(p["flag"] or "", "ch-dot")
        row = [_hline(plot_l, plot_r, cy, "ch-grid ch-guide")]
        if plo is not None and phi is not None:
            row.append(_hline(x(plo), x(phi), cy, "ch-ci"))
        if mean is not None:
            row.append(f'<circle class="{dot_cls}" cx="{_c(x(mean))}" cy="{_c(cy)}" r="4"/>')
        glyph = {"bad": "▼", "ok": "▲"}.get(p["flag"] or "")
        value = _esc(_fixed(mean))
        if glyph:
            value = f'<tspan class="ch-flag ch-flag-{p["flag"]}">{glyph}</tspan> {value}'
        row.append(_text(val_w, cy, "", cls="ch-tick" if p["flag"] == "muted" else "ch-lbl",
                         align="right", raw=value))
        cat_cls = "ch-tick" if p["flag"] == "muted" else "ch-cat"
        row.append(_text(W - 4, cy, _fit(p["label"], lab_w - 10), cls=cat_cls, align="right",
                         rtl=True))
        if p["sub"]:
            row.append(_text(W - lab_w - 6, cy, _fit(p["sub"], sub_w - 10, _FS_TICK),
                             cls="ch-tick", size=_FS_TICK, align="right", rtl=True))
        tip = f"{p['label']}: ממוצע {fmt(mean)} · {CI_HE}: {ci} · {_calls(p['n'])}"
        if p["sub"]:
            tip += f" · {p['sub']}"
        out.append(_row_group(row, p["filter"], tip, title=tip, hit=(0, cy - rh / 2, W, rh)))
        table_rows.append([_cell(p["label"]), _cell(p["sub"]), _num_cell(fmt(mean)),
                           _num_cell(ci), _num_cell(fmt(p["n"], 0))])
    interactive = any(p["filter"] is not None for p in prepared)
    svg = _svg(W, H, name, "".join(out), interactive=interactive)
    foot = [ref_name, "", fmt(ref), "", ""] if ref is not None else None
    table = _table(name, [ITEM_HE, "פירוט", "ממוצע", CI_HE, "שיחות"], table_rows, foot=foot)
    return _figure("dots", svg, table=table)


def diverging_bars(rows: Sequence[Mapping[str, Any]], *, label: str, unit: str = "נק'",
                   width: int = DEFAULT_WIDTH) -> Markup:
    """Signed effects around a zero axis: negative bars (ch-bar-bad) extend
    left, positive bars (ch-bar-ok) extend right, and every bar has its
    signed value.

    Both sides share one linear scale. The zero axis is placed where that
    scale puts it, so it is in the middle only when the values are balanced
    and a list of all-negative effects does not waste half the width. ``detail``
    is set as a second, muted line under the row label.
    """
    name, u = _plain(label), _plain(unit)
    prepared = [(_plain(r.get("label")), _num(r.get("value")), _plain(r.get("detail")),
                 r.get("filter")) for r in rows]
    if not prepared:
        return _empty(name)
    W = _width(width)
    vals = [v for _, v, _, _ in prepared if v is not None]
    d_lo, d_hi = min([0.0, *vals]), max([0.0, *vals])
    if d_hi - d_lo < 1e-9:
        d_hi = d_lo + 1.0
    has_detail = any(d for _, _, d, _ in prepared)
    texts = [f"{_signed(v)} {u}".strip() if v is not None else "—" for _, v, _, _ in prepared]
    val_w = max(_tw(t, bold=True) for t in texts) + 12
    lab_w = _clamp(max(max(_tw(lb), _tw(d, _FS_TICK)) for lb, _, d, _ in prepared) + 16,
                   64, W * 0.34)
    has_neg, has_pos = any(v < 0 for v in vals), any(v >= 0 for v in vals)
    plot_l = val_w if has_neg else 8.0
    plot_r = W - lab_w - (val_w if has_pos or not vals else 8.0)

    def x(v: float) -> float:
        return plot_l + (v - d_lo) / (d_hi - d_lo) * (plot_r - plot_l)

    zero = x(0.0)
    pitch = 40.0 if has_detail else 28.0
    bar_h, top = 16.0, 4.0
    H = top + pitch * len(prepared) + 4
    out = [_group([_vline(zero, top + 2, H - 4, "ch-axis")])]
    table_rows: list[list[str]] = []
    for i, (lb, v, detail, filt) in enumerate(prepared):
        cy = top + pitch * i + pitch / 2
        row: list[str] = []
        row.append(_text(W - 4, cy - 7 if detail else cy, _fit(lb, lab_w - 12), cls="ch-cat",
                         align="right", rtl=True))
        if detail:
            row.append(_text(W - 4, cy + 9, _fit(detail, lab_w - 12, _FS_TICK), cls="ch-tick",
                             size=_FS_TICK, align="right", rtl=True))
        if v is not None:
            tip_x = x(v)
            if abs(tip_x - zero) >= 0.5:
                cls = "ch-bar-bad" if v < 0 else "ch-bar-ok"
                row.append(f'<path class="{cls}" '
                           f'd="{_hbar(zero, tip_x, cy - bar_h / 2, bar_h, 3)}"/>')
            if v < 0:
                row.append(_text(tip_x - 6, cy, texts[i], cls="ch-lbl", align="right", rtl=True))
            else:
                row.append(_text(tip_x + 6, cy, texts[i], cls="ch-lbl", align="left", rtl=True))
        else:
            row.append(_text(zero + 6, cy, "—", cls="ch-tick", size=_FS_TICK))
        tip = f"{lb}: {texts[i]}" + (f" · {detail}" if detail else "")
        out.append(_row_group(row, filt, tip, title=tip, hit=(0, cy - pitch / 2, W, pitch)))
        table_rows.append([_cell(lb), _num_cell(_signed(v) if v is not None else "—"),
                           _cell(detail)])
    interactive = any(f is not None for *_, f in prepared)
    svg = _svg(W, H, name, "".join(out), interactive=interactive)
    head = [ITEM_HE, f"ערך ({u})" if u else "ערך", "פירוט"]
    return _figure("diverging", svg, table=_table(name, head, table_rows))


# -- text and numbers -------------------------------------------------------------

def _num(value: object) -> float | None:
    """A finite float, or None for missing / NaN / infinite / non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _num_range(pair: Sequence[Any] | None) -> tuple[float, float] | None:
    if not pair or len(pair) != 2:
        return None
    lo, hi = _num(pair[0]), _num(pair[1])
    if lo is None or hi is None or hi <= lo:
        return None
    return lo, hi


def _plain(value: object) -> str:
    """Caller text as one line of plain text: Markup reduced to its text,
    controls and bidi overrides removed, whitespace collapsed."""
    if value is None:
        return ""
    text = value.striptags() if isinstance(value, Markup) else str(value)
    text = _CONTROLS.sub("", _BREAKS.sub(" ", text))
    return " ".join(text.split())


def _bidi(text: str) -> str:
    """Isolate the numeric runs that RTL text would otherwise reorder."""
    return _BIDI_RISK.sub(lambda m: f"{_LRI}{m.group(0)}{_PDI}", text)


def _vis(text: str) -> str:
    """Visible text (element content): bidi-safe, then escaped."""
    return str(escape(_bidi(text)))


def _esc(text: str) -> str:
    return str(escape(text))


def _attr(text: str) -> str:
    """For attribute values read by assistive tech: escaped, no isolates."""
    return str(escape(text))


def _pct(frac: float | None) -> str:
    return "—" if frac is None else f"{fmt(frac * 100, 0)}%"


def _signed(v: float | None) -> str:
    """'+6.2' / '−4.1' / '0'. A true minus sign, as in print."""
    if v is None:
        return "—"
    text = fmt(abs(v))
    if text == "0":
        return "0"
    return f"+{text}" if v > 0 else f"{_MINUS}{text}"


def _fixed(v: float | None) -> str:
    """Exactly one decimal, for a column of values that must align on the
    decimal point ('76.0' under '75.9', where fmt() would give '76')."""
    text = fmt(v)
    return text if text == "—" or "." in text else f"{text}.0"


def _calls(n: int) -> str:
    return "שיחה אחת" if n == 1 else f"{fmt(n, 0)} שיחות"


def _tw(text: str, size: float = _FS_TEXT, bold: bool = False) -> float:
    """Estimated rendered width of text at ``size`` px.

    There are no font metrics on the server, so this is a per-character
    estimate, calibrated in Chromium against DejaVu Sans (the widest digits
    of the fonts the page stack falls back to on Linux) and the published
    Segoe UI / Arial advances. It errs wide: an underestimated label collides
    with its neighbour, an overestimated one only leaves a little air.
    """
    em = 0.0
    for ch in text:
        if ch in _ZERO_WIDTH:
            continue
        o = ord(ch)
        if 0x0590 <= o <= 0x05FF:
            em += 0.3 if ch in "וין" else 0.45 if ch in "זג׳״" else 0.6
        elif ch.isdigit():
            em += 0.63
        elif ch in " .,:;'|!":
            em += 0.32
        elif ch in "()[]-/–":
            em += 0.5
        elif ch in "%—…":
            em += 1.0
        elif ch in "+=<>−▲▼":
            em += 0.84
        elif ch.isupper():
            em += 0.7
        elif ch.isascii():
            em += 0.6
        else:
            em += 0.7
    return em * size * (_BOLD if bold else 1.0)


def _fit(text: str, max_w: float, size: float = _FS_TEXT, bold: bool = False) -> str:
    """Truncate with an ellipsis to fit ``max_w``. The full text stays in
    the tooltip and the table."""
    if _tw(text, size, bold) <= max_w:
        return text
    cut = text
    while cut and _tw(cut + _ELLIPSIS, size, bold) > max_w:
        cut = cut[:-1]
    return cut.rstrip() + _ELLIPSIS if cut else _ELLIPSIS


# -- geometry -------------------------------------------------------------------

def _c(x: float) -> str:
    """A coordinate with at most one decimal: compact and deterministic."""
    r = round(x, 1)
    if r == 0:
        return "0"
    s = f"{r:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _width(width: int) -> int:
    return max(280, int(width))


def _ticks(lo: float, hi: float, max_ticks: int, *, integer: bool = False) -> list[float]:
    """Round tick values (1/2/2.5/5 x 10^k steps) inside [lo, hi]."""
    if hi <= lo:
        return [lo]
    step = _nice_step((hi - lo) / max(1, max_ticks), integer)
    first = math.ceil(lo / step - 1e-9)
    out = []
    k = first
    while k * step <= hi + 1e-9 * step:
        out.append(round(k * step, 10))
        k += 1
    return out


def _nice_step(raw: float, integer: bool = False) -> float:
    mag = 10 ** math.floor(math.log10(raw))
    for m in ((1, 2, 5, 10) if integer else (1, 2, 2.5, 5, 10)):
        if m * mag >= raw - 1e-12:
            step = m * mag
            return max(1.0, round(step)) if integer else step
    return 10 * mag  # pragma: no cover - the loop always returns


def _nice_axis(max_value: float, max_ticks: int, *,
               integer: bool = False) -> tuple[float, list[float]]:
    """A zero-based axis whose top is the first round tick at or above max_value."""
    if max_value <= 0:
        return 1.0, [0.0, 1.0]
    step = _nice_step(max_value / max(1, max_ticks), integer)
    top = math.ceil(max_value / step - 1e-9) * step
    return top, _ticks(0.0, top, int(round(top / step)), integer=integer)


def _runs(seq: Sequence[tuple[int, Any]]) -> list[list[tuple[int, Any]]]:
    """Consecutive stretches of non-None values: a gap breaks a line."""
    runs: list[list[tuple[int, Any]]] = []
    current: list[tuple[int, Any]] = []
    for i, v in seq:
        if v is None:
            if current:
                runs.append(current)
            current = []
        else:
            current.append((i, v))
    if current:
        runs.append(current)
    return runs


def _vbar(x0: float, x1: float, top: float, base: float, r: float) -> str:
    """A column with a rounded top, square at the baseline."""
    r = max(0.0, min(r, (x1 - x0) / 2, base - top))
    return (f"M{_c(x0)},{_c(base)}V{_c(top + r)}A{_c(r)},{_c(r)} 0 0 1 {_c(x0 + r)},{_c(top)}"
            f"H{_c(x1 - r)}A{_c(r)},{_c(r)} 0 0 1 {_c(x1)},{_c(top + r)}V{_c(base)}Z")


def _hbar(base: float, tip: float, y: float, h: float, r: float) -> str:
    """A bar from ``base`` to ``tip`` (either side) with its tip end rounded."""
    r = max(0.0, min(r, abs(tip - base), h / 2))
    if tip >= base:
        return (f"M{_c(base)},{_c(y)}H{_c(tip - r)}A{_c(r)},{_c(r)} 0 0 1 {_c(tip)},{_c(y + r)}"
                f"V{_c(y + h - r)}A{_c(r)},{_c(r)} 0 0 1 {_c(tip - r)},{_c(y + h)}"
                f"H{_c(base)}Z")
    return (f"M{_c(base)},{_c(y)}H{_c(tip + r)}A{_c(r)},{_c(r)} 0 0 0 {_c(tip)},{_c(y + r)}"
            f"V{_c(y + h - r)}A{_c(r)},{_c(r)} 0 0 0 {_c(tip + r)},{_c(y + h)}H{_c(base)}Z")


def _clean_markers(markers: Sequence[tuple[float, str]], lo: float,
                   hi: float) -> list[tuple[float, str]]:
    out = []
    for m in markers:
        v = _num(m[0])
        if v is not None and lo <= v <= hi:
            out.append((v, m[1]))
    return sorted(out, key=lambda m: m[0])


def _place_labels(labels: Sequence[tuple[float, str, float]], lo: float, hi: float,
                  rows: int) -> list[tuple[int, float, float, float, str]]:
    """Reference labels as flags beside their own line: left of it (the RTL
    reading end touches the line), else right of it, on the first row where
    the label overlaps no other label and no other marker line. Markers that
    fit nowhere take the last row, centred. Returns
    (row, x0, x1, line_x, text) sorted by line position."""
    placed: list[tuple[int, float, float, float, str]] = []
    todo = sorted(labels, key=lambda m: m[0])
    line_xs = [m[0] for m in todo]
    for row in range(rows):
        left: list[tuple[float, str, float]] = []
        for lx, text, v in todo:
            w = _tw(text, _FS_TICK)
            for x0, x1 in ((lx - 5 - w, lx - 5), (lx + 5, lx + 5 + w)):
                clear = (x0 >= lo and x1 <= hi
                         and all(p[0] != row or x1 + 8 < p[1] or x0 > p[2] + 8 for p in placed)
                         and all(o == lx or not x0 - 3 <= o <= x1 + 3 for o in line_xs))
                if clear:
                    placed.append((row, x0, x1, lx, text))
                    break
            else:
                left.append((lx, text, v))
        todo = left
    for lx, text, _v in todo:
        w = _tw(text, _FS_TICK)
        cx = _clamp(lx, lo + w / 2, hi - w / 2)
        placed.append((rows - 1, cx - w / 2, cx + w / 2, lx, text))
    return sorted(placed, key=lambda p: p[3])


def _band_label_x(x0: float, x1: float, w: float, avoid: Sequence[float]) -> float | None:
    """Centre of a band name that fits inside its band and clears every
    marker line. Tries the centre, then either edge. None when it cannot fit."""
    if x1 - x0 < w + 8:
        return None
    for cx in ((x0 + x1) / 2, x0 + 4 + w / 2, x1 - 4 - w / 2):
        if all(abs(cx - a) > w / 2 + 4 for a in avoid):
            return cx
    return None


# -- SVG / HTML assembly --------------------------------------------------------

def _text(x: float, y: float, text: str, *, cls: str, size: float = _FS_TEXT,
          align: str = "left", rtl: bool = False, extra: str = "",
          raw: str | None = None) -> str:
    """A text element placed by its physical edge: ``align`` says which edge
    (or the middle) sits at x. For RTL text the SVG 'start' is the right edge,
    so the anchor is flipped to match. ``raw`` is pre-escaped markup (only
    built in this module from escaped parts)."""
    if align == "middle":
        anchor = "middle"
    elif rtl:
        anchor = "start" if align == "right" else "end"
    else:
        anchor = "end" if align == "right" else "start"
    direction = ' direction="rtl"' if rtl else ""
    body = raw if raw is not None else _vis(text)
    return (f'<text class="{cls}" x="{_c(x)}" y="{_c(y)}" dy="0.35em" font-size="{size:g}" '
            f'text-anchor="{anchor}"{direction}{extra}>{body}</text>')


def _hline(x0: float, x1: float, y: float, cls: str) -> str:
    return (f'<line class="{cls}" x1="{_c(x0)}" y1="{_c(y)}" x2="{_c(x1)}" y2="{_c(y)}" '
            f'stroke="currentColor" vector-effect="non-scaling-stroke"/>')


def _vline(x: float, y0: float, y1: float, cls: str) -> str:
    return (f'<line class="{cls}" x1="{_c(x)}" y1="{_c(y0)}" x2="{_c(x)}" y2="{_c(y1)}" '
            f'stroke="currentColor" vector-effect="non-scaling-stroke"/>')


def _polyline(pts: Sequence[tuple[float, float]], cls: str) -> str:
    coords = " ".join(f"{_c(px)},{_c(py)}" for px, py in pts)
    return (f'<polyline class="{cls}" points="{coords}" fill="none" stroke="currentColor" '
            f'vector-effect="non-scaling-stroke"/>')


def _group(parts: Sequence[str]) -> str:
    """Decoration (grid, ticks, titles): hidden from assistive tech, which
    reads the rows and the table instead."""
    return f'<g aria-hidden="true">{"".join(parts)}</g>'


def _row_group(parts: Sequence[str], filt: Any, aria: str, *, title: str,
               hit: tuple[float, float, float, float]) -> str:
    """One data row: a tooltip, a hit area as wide as the row (drawn first,
    so a CSS hover fill on .ch-hit shades behind the marks), then the marks.
    When the row has a filter it becomes a keyboard-reachable button."""
    hx, hy, hw, hh = hit
    rect = (f'<rect class="ch-hit" x="{_c(hx)}" y="{_c(hy)}" width="{_c(hw)}" '
            f'height="{_c(hh)}" fill="none" pointer-events="all"/>')
    attrs = ""
    if filt is not None:
        payload = json.dumps(filt, ensure_ascii=False, default=str)
        attrs = (f' class="ch-link" data-filter="{escape(payload)}" tabindex="0" role="button" '
                 f'aria-label="{_attr(f"{aria} — {SHOW_CALLS_HE}")}"')
    return f'<g{attrs}><title>{_vis(title)}</title>{rect}{"".join(parts)}</g>'


def _svg(w: float, h: float, name: str, body: str, *, interactive: bool) -> str:
    # A chart whose rows are buttons is a group of controls, not an image:
    # role="img" would hide those buttons from screen readers (its children
    # are presentational), so interactive charts are role="group".
    role = "group" if interactive else "img"
    return (f'<svg class="ch-svg" viewBox="0 0 {_c(w)} {_c(h)}" width="100%" '
            f'style="max-width:{_c(w)}px" role="{role}" aria-label="{_attr(name)}" '
            f'direction="ltr" fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
            f'{body}</svg>')


def _legend(items: Sequence[tuple[str, str, str]]) -> str:
    """HTML legend. Each swatch is a tiny SVG carrying the mark's own class,
    so it inherits exactly the fill or stroke the chart uses."""
    parts = []
    for kind, cls, text in items:
        if kind == "line":
            sw = (f'<svg class="ch-sw {cls}" viewBox="0 0 18 10" style="width:18px" '
                  f'aria-hidden="true"><line x1="1" y1="5" x2="17" y2="5"/></svg>')
        else:
            sw = (f'<svg class="ch-sw {cls}" viewBox="0 0 10 10" aria-hidden="true">'
                  f'<rect width="10" height="10" rx="2"/></svg>')
        parts.append(f'<span class="ch-legend-item">{sw}{_vis(text)}</span>')
    return f'<div class="ch-legend">{"".join(parts)}</div>'


def _cell(text: str) -> str:
    return f"<td>{_vis(text)}</td>"


def _num_cell(text: str) -> str:
    return f'<td class="num"><bdi dir="ltr">{_esc(text)}</bdi></td>'


def _table(caption: str, head: Sequence[str], rows: Sequence[Sequence[str]],
           foot: Sequence[str] | None = None) -> str:
    """The chart's data as a table: the accessible and printable twin."""
    ths = "".join(f'<th scope="col">{_vis(h)}</th>' for h in head)
    body = "".join(f"<tr>{''.join(r)}</tr>" for r in rows)
    tfoot = ""
    if foot:
        cells = [_cell(foot[0])] + [_num_cell(f) if f else "<td></td>" for f in foot[1:]]
        tfoot = f"<tfoot><tr>{''.join(cells)}</tr></tfoot>"
    return (f'<details class="ch-table"><summary>{TABLE_SUMMARY_HE}</summary>'
            f'<div class="tbl-wrap"><table><caption>{_vis(caption)}</caption>'
            f'<thead><tr>{ths}</tr></thead><tbody>{body}</tbody>{tfoot}</table></div></details>')


def _figure(kind: str, svg: str, *, legend: str = "", table: str = "") -> Markup:
    return Markup(f'<figure class="ch ch-{kind}">{svg}{legend}{table}</figure>')


def _empty(name: str) -> Markup:
    return Markup(f'<figure class="ch ch-none" aria-label="{_attr(name)}">'
                  f'<div class="ch-empty">{EMPTY_HE}</div></figure>')
