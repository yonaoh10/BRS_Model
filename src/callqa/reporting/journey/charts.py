"""Charts of the journey report, on the management report's SVG primitives.

Same rules as reporting/executive/charts.py: inline SVG, no colour literals
(marks carry ch-* classes the page styles, so dark mode and print restyle
them), every chart with its data as a table, Hebrew labels right-to-left,
numbers isolated so the bidi algorithm cannot reorder them.

    hbars           horizontal bars with count and share (distributions, funnels)
    stacked_bars    100% bars split by category, n on every bar
    columns         vertical bars over ordered buckets (time between contacts)
    km_chart        Kaplan-Meier: share of stories still unresolved over time
    story_timeline  one story on two lanes: the customer above, the bankers below
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from markupsafe import Markup

from callqa.reporting.executive.charts import (
    DEFAULT_WIDTH,
    _attr,
    _c,
    _cell,
    _clamp,
    _empty,
    _figure,
    _fit,
    _group,
    _hbar,
    _hline,
    _legend,
    _nice_axis,
    _num_cell,
    _pct,
    _plain,
    _row_group,
    _svg,
    _table,
    _text,
    _tick_count,
    _tw,
    _vbar,
    _vis,
    _vline,
    _width,
    fmt,
)

_FS_TICK = 10.5


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) < 1e12 else None


def hbars(items: Sequence[Mapping[str, Any]], *, label: str, unit: str = "",
          total: float | None = None, width: int = DEFAULT_WIDTH) -> Markup:
    """items: {label, value, cls?, filter?, note?}. Bars keep the given order;
    the share is of `total` (default: the sum)."""
    name = _plain(label)
    rows = [(_plain(i.get("label", "")), _num(i.get("value")), i.get("cls") or "ch-bar",
             i.get("filter"), _plain(i.get("note", ""))) for i in items]
    if not rows:
        return _empty(name)
    tot = total if total is not None else sum(max(0.0, v or 0.0) for _l, v, *_ in rows)
    vmax = max((v or 0.0) for _l, v, *_ in rows) or 1.0
    W = _width(width)
    val_texts = [f"{fmt(v, 0) if v is not None and float(v).is_integer() else fmt(v)}"
                 + (f" ({_pct(v / tot)})" if (v is not None and tot) else "") for _l, v, *_ in rows]
    lab_w = _clamp(max(_tw(lb) for lb, *_ in rows) + 16, 70, W * 0.38)
    val_w = max(_tw(t, bold=True) for t in val_texts) + 14
    bar_r, bar_l = W - lab_w, val_w
    pitch, bar_h, top = 28.0, 15.0, 4.0
    H = top + pitch * len(rows) + 4
    out = [_group([_vline(bar_r, top + 2, H - 4, "ch-axis")])]
    table_rows = []
    interactive = any(f is not None for *_x, f, _n in rows)
    for i, (lb, v, cls, filt, note) in enumerate(rows):
        cy = top + pitch * i + pitch / 2
        tip_x = bar_r - (max(0.0, v or 0.0) / vmax) * (bar_r - bar_l)
        parts = [_text(W - 4, cy, _fit(lb, lab_w - 14), cls="ch-cat", align="right", rtl=True)]
        if v and bar_r - tip_x >= 0.5:
            parts.append(f'<path class="{_attr(cls)}" d="{_hbar(bar_r, tip_x, cy - bar_h / 2, bar_h, 3)}"/>')
        parts.append(_text(tip_x - 6, cy, val_texts[i], cls="ch-lbl", align="right", rtl=True))
        tip = f"{lb}: {val_texts[i]}" + (f" · {note}" if note else "")
        out.append(_row_group(parts, filt, tip, title=tip, hit=(0, cy - pitch / 2, W, pitch)))
        table_rows.append([_cell(lb), _num_cell(fmt(v)),
                           _num_cell(_pct(v / tot) if (v is not None and tot) else "—")])
    head = ["", f"מספר{' (' + unit + ')' if unit else ''}", "חלק"]
    return _figure("hbars", _svg(W, H, name, "".join(out), interactive=interactive),
                   table=_table(name, head, table_rows))


def stacked_bars(rows: Sequence[Mapping[str, Any]], segments: Sequence[tuple[str, str, str]], *,
                 label: str, width: int = DEFAULT_WIDTH) -> Markup:
    """rows: {label, counts: {segment_key: n}, filter?}; segments: (key, label, cls)
    in drawing order from the right. Each bar is 100% of its row, with n shown."""
    name = _plain(label)
    prepared = [(_plain(r.get("label", "")), {k: max(0, int(r.get("counts", {}).get(k, 0) or 0))
                                              for k, _l, _c2 in segments}, r.get("filter"))
                for r in rows]
    prepared = [p for p in prepared if sum(p[1].values())]
    if not prepared:
        return _empty(name)
    W = _width(width)
    lab_w = _clamp(max(_tw(lb) for lb, *_ in prepared) + 16, 70, W * 0.3)
    n_w = _tw("n=00,000", _FS_TICK) + 12
    bar_r, bar_l = W - lab_w, n_w
    pitch, bar_h, top = 30.0, 17.0, 4.0
    H = top + pitch * len(prepared) + 4
    out = []
    table_rows = []
    for i, (lb, counts, filt) in enumerate(prepared):
        cy = top + pitch * i + pitch / 2
        n = sum(counts.values())
        x = bar_r
        parts = [_text(W - 4, cy, _fit(lb, lab_w - 14), cls="ch-cat", align="right", rtl=True)]
        tips = []
        for key, seg_label, cls in segments:
            k = counts[key]
            if not k:
                continue
            w = (k / n) * (bar_r - bar_l)
            parts.append(f'<rect class="{_attr(cls)}" x="{_c(x - w)}" y="{_c(cy - bar_h / 2)}" '
                         f'width="{_c(max(w, 0.5))}" height="{_c(bar_h)}"/>')
            if w >= 34:
                parts.append(_text(x - w / 2, cy, _pct(k / n), cls="ch-seg-lbl", size=_FS_TICK,
                                   align="middle"))
            tips.append(f"{_plain(seg_label)} {k} ({_pct(k / n)})")
            x -= w
        parts.append(_text(bar_l - 6, cy, f"n={n:,}", cls="ch-tick", size=_FS_TICK, align="right"))
        tip = f"{lb}: " + " · ".join(tips)
        out.append(_row_group(parts, filt, tip, title=tip, hit=(0, cy - pitch / 2, W, pitch)))
        table_rows.append([_cell(lb)] + [_num_cell(str(counts[k])) for k, *_ in segments]
                          + [_num_cell(str(n))])
    legend = _legend([("box", cls, lbl) for _k, lbl, cls in segments])
    head = [""] + [_plain(lbl) for _k, lbl, _c2 in segments] + ["סה\"כ"]
    interactive = any(f is not None for *_x, f in prepared)
    return _figure("stacked", _svg(W, H, name, "".join(out), interactive=interactive),
                   legend=legend, table=_table(name, head, table_rows))


def columns(items: Sequence[tuple[str, float]], *, label: str, y_title: str = "",
            width: int = DEFAULT_WIDTH) -> Markup:
    """Vertical bars over ordered buckets, first bucket on the right (RTL)."""
    name = _plain(label)
    vals = [(_plain(lb), max(0.0, _num(v) or 0.0)) for lb, v in items]
    if not vals or not any(v for _l, v in vals):
        return _empty(name)
    W = _width(width)
    top, bottom, left = 18.0, 34.0, 36.0
    H = 200.0
    axis_top, ticks = _nice_axis(max(v for _l, v in vals), _tick_count(H - top - bottom, 34),
                                 integer=True)
    plot_w = W - left - 8
    slot = plot_w / len(vals)
    bw = min(64.0, slot * 0.62)
    y = lambda v: H - bottom - (v / axis_top) * (H - top - bottom)  # noqa: E731
    deco = [_hline(left, W - 8, y(t), "ch-grid") for t in ticks]
    deco += [_text(left - 6, y(t), fmt(t, 0), cls="ch-tick", size=_FS_TICK, align="right") for t in ticks]
    out = [_group(deco)]
    total = sum(v for _l, v in vals)
    rows = []
    for i, (lb, v) in enumerate(vals):
        cx = W - 8 - slot * i - slot / 2
        parts = [f'<path class="ch-bar" d="{_vbar(cx - bw / 2, cx + bw / 2, y(v), H - bottom, 3)}"/>'
                 if v else "",
                 _text(cx, y(v) - 9, f"{fmt(v, 0)}", cls="ch-lbl", align="middle"),
                 _text(cx, H - bottom + 14, _fit(lb, slot - 4, _FS_TICK), cls="ch-tick",
                       size=_FS_TICK, align="middle", rtl=True)]
        tip = f"{lb}: {fmt(v, 0)} ({_pct(v / total)})"
        out.append(_row_group(parts, None, tip, title=tip, hit=(cx - slot / 2, top, slot, H - top)))
        rows.append([_cell(lb), _num_cell(fmt(v, 0)), _num_cell(_pct(v / total))])
    return _figure("columns", _svg(W, H, name, "".join(out), interactive=False),
                   table=_table(name, ["", y_title or "מספר", "חלק"], rows))


def km_chart(steps: Sequence[Mapping[str, Any]], *, label: str, width: int = DEFAULT_WIDTH,
             x_title: str = "ימים מהפנייה הראשונה") -> Markup:
    """The Kaplan-Meier curve of 'still unresolved', with censoring ticks."""
    name = _plain(label)
    pts = [(float(s["t"]), float(s["s"]), int(s.get("censored", 0))) for s in steps]
    if len(pts) < 2:
        return _empty(name)
    W = _width(width)
    top, bottom, left, right = 14.0, 36.0, 44.0, 12.0
    H = 230.0
    tmax = max(t for t, _s, _c2 in pts) or 1.0
    xmax, xticks = _nice_axis(tmax, _tick_count(W - left - right, 70), integer=True)
    # time runs right to left, as the page reads
    X = lambda t: W - right - (t / xmax) * (W - left - right)  # noqa: E731
    Y = lambda s: top + (1 - s) * (H - top - bottom)  # noqa: E731
    deco = []
    for yv in (0, 0.25, 0.5, 0.75, 1.0):
        deco.append(_hline(left, W - right, Y(yv), "ch-grid" if yv not in (0.5,) else "ch-ref"))
        deco.append(_text(left - 6, Y(yv), _pct(yv), cls="ch-tick", size=_FS_TICK, align="right"))
    for t in xticks:
        deco.append(_text(X(t), H - bottom + 14, fmt(t, 0), cls="ch-tick", size=_FS_TICK,
                          align="middle"))
    deco.append(_text(W - right, H - 6, x_title, cls="ch-title", align="right", rtl=True))
    path = []
    prev_s = 1.0
    for t, s, _c2 in pts:
        path.append((X(t), Y(prev_s)))
        path.append((X(t), Y(s)))
        prev_s = s
    d = "M" + " L".join(f"{_c(x)},{_c(y)}" for x, y in path)
    marks = [f'<path class="ch-line" d="{d}"/>']
    for t, s, c in pts:
        if c:
            marks.append(_vline(X(t), Y(s) - 4, Y(s) + 4, "ch-axis"))
    rows = [[_num_cell(fmt(t, 1)), _num_cell(_pct(s)), _num_cell(str(c))] for t, s, c in pts]
    return _figure("km", _svg(W, H, name, _group(deco) + "".join(marks), interactive=False),
                   table=_table(name, ["ימים", "עדיין פתוחים", "צונזרו"], rows))


# ------------------------------------------------------------------ timeline

CONTACT_GLYPH = {"recorded_call": "circle", "message": "square", "abandoned": "cross",
                 "unrecorded_answered": "ring", "unrecorded_unknown": "ring",
                 "branch": "ring", "other": "ring"}
CONTACT_HE = {"recorded_call": "שיחה מוקלטת", "message": "התכתבות", "abandoned": "שיחה שננטשה",
              "unrecorded_answered": "שיחה שנענתה, לא הוקלטה",
              "unrecorded_unknown": "שיחה שלא הוקלטה", "branch": "פנייה בסניף", "other": "מגע אחר"}


def _time_axis(times: list[datetime], x_right: float, x_left: float, cap_hours: float = 10.0
               ) -> tuple[Any, list[tuple[datetime, datetime]]]:
    """A time -> x map (right to left) that shrinks long quiet stretches to
    `cap_hours`, so a two-week story with a busy day stays readable; returns
    the map and the compressed gaps (for break marks)."""
    ts = sorted(set(times))
    if len(ts) < 2:
        mid = (x_left + x_right) / 2
        return (lambda _t: mid), []
    eff = [0.0]
    breaks = []
    for a, b in zip(ts, ts[1:], strict=False):
        gap = (b - a).total_seconds() / 3600
        if gap > cap_hours:
            breaks.append((a, b))
            gap = cap_hours
        eff.append(eff[-1] + gap)
    span = eff[-1] or 1.0
    pos = dict(zip(ts, eff, strict=True))

    def xmap(t: datetime) -> float:
        if t in pos:
            e = pos[t]
        else:
            # between two known moments: interpolate within the (possibly shrunk) gap
            prev = max((p for p in ts if p <= t), default=ts[0])
            nxt = min((p for p in ts if p >= t), default=ts[-1])
            if nxt == prev:
                e = pos[prev]
            else:
                frac = (t - prev).total_seconds() / max(1.0, (nxt - prev).total_seconds())
                e = pos[prev] + frac * (pos[nxt] - pos[prev])
        return x_right - (e / span) * (x_right - x_left)
    return xmap, breaks


def story_timeline(contacts: Sequence[Mapping[str, Any]], sessions: Sequence[Mapping[str, Any]],
                   *, label: str, anchor: str = "", promises: Sequence[Mapping[str, Any]] = (),
                   width: int = 1060) -> Markup:
    """One story on two lanes, time flowing right to left.

    contacts: {index, at (datetime), kind, category_cls, category, direction, title}
    sessions: {start, end, unit_kind, execute (bool), view_only (bool), contact_index|None, title}
    promises: {from_index, to (datetime), outcome}
    """
    name = _plain(label)
    if not contacts:
        return _empty(name)
    W = _width(width)
    left, right = 16.0, 90.0
    lane_c, lane_b = 46.0, 118.0
    H = 168.0
    times = [c["at"] for c in contacts] + [s["start"] for s in sessions] + [s["end"] for s in sessions]
    X, breaks = _time_axis(times, W - right, left)
    deco = [
        _text(W - 4, lane_c, "לקוח", cls="ch-lbl", align="right", rtl=True),
        _text(W - 4, lane_b, "בנקאים", cls="ch-lbl", align="right", rtl=True),
        _hline(left, W - right + 6, lane_c, "ch-grid"),
        _hline(left, W - right + 6, lane_b, "ch-grid"),
    ]
    # day ticks at the first event of each day
    seen_days: set = set()
    last_x = None
    for t in sorted(times):
        if t.date() in seen_days:
            continue
        seen_days.add(t.date())
        x = X(t)
        if last_x is not None and abs(last_x - x) < 44:
            continue
        last_x = x
        deco.append(_vline(x, lane_c - 26, H - 22, "ch-grid"))
        deco.append(_text(x, H - 10, t.strftime("%d.%m"), cls="ch-tick", size=_FS_TICK, align="middle"))
    for a, b in breaks:
        xa, xb = X(a), X(b)
        mid = (xa + xb) / 2
        hours = (b - a).total_seconds() / 3600
        text = f"≈{fmt(hours / 24, 0)} ימים" if hours >= 36 else f"≈{fmt(hours, 0)} שעות"
        deco.append(_text(mid, lane_c - 30, text, cls="ch-break", size=_FS_TICK, align="middle",
                          rtl=True))
    out = [_group(deco)]
    by_index = {c["index"]: c for c in contacts}
    # marks closer than a mark's width are spread apart (time runs right to left)
    cx: dict[int, float] = {}
    prev = None
    for c in sorted(contacts, key=lambda c: (c["at"], c["index"])):
        x = X(c["at"])
        if prev is not None and x > prev - 17:
            x = prev - 17
        cx[c["index"]] = x
        prev = x

    def XC(c: Mapping[str, Any]) -> float:
        return cx[c["index"]]
    # connectors first (under the marks)
    links = []
    for s in sessions:
        ci = s.get("contact_index")
        if ci is not None and ci in by_index:
            xs = X(s["start"])
            links.append(f'<line class="ch-link-line" x1="{_c(XC(by_index[ci]))}" '
                         f'y1="{_c(lane_c + 8)}" x2="{_c(xs)}" y2="{_c(lane_b - 8)}"/>')
    for p in promises:
        src = by_index.get(p.get("from_index"))
        if src is None or p.get("to") is None:
            continue
        x0, x1 = XC(src), X(p["to"])
        cls = {"kept": "ch-arc-ok", "broken": "ch-arc-bad"}.get(p.get("outcome"), "ch-arc")
        links.append(f'<path class="{cls}" d="M{_c(x0)},{_c(lane_c - 10)} '
                     f'Q{_c((x0 + x1) / 2)},{_c(lane_c - 34)} {_c(x1)},{_c(lane_c - 10)}"/>')
    out.append(_group(links))
    # banker sessions
    table_rows = []
    for s in sessions:
        xs, xe = X(s["start"]), X(s["end"])
        w = max(abs(xs - xe), 9.0)
        x0 = min(xs, xe) - (w - abs(xs - xe)) / 2
        cls = f"ch-sess ch-unit-{s.get('unit_kind', 'other')}" + (" ch-sess-exec" if s.get("execute")
                                                                    else " ch-sess-view")
        title = _plain(s.get("title", ""))
        out.append(f'<g><title>{_vis(title)}</title><rect class="{cls}" x="{_c(x0)}" '
                   f'y="{_c(lane_b - 9)}" width="{_c(w)}" height="18" rx="3"/></g>')
    # customer contacts
    for c in contacts:
        x = XC(c)
        glyph = CONTACT_GLYPH.get(c["kind"], "ring")
        cls = c.get("category_cls") or "ch-cat-none"
        if glyph == "circle":
            mark = f'<circle class="ch-mark {cls}" cx="{_c(x)}" cy="{_c(lane_c)}" r="8"/>'
        elif glyph == "square":
            mark = f'<rect class="ch-mark {cls}" x="{_c(x - 7.5)}" y="{_c(lane_c - 7.5)}" width="15" height="15" rx="2"/>'
        elif glyph == "cross":
            mark = (f'<path class="ch-mark-cross" d="M{_c(x - 6)},{_c(lane_c - 6)} L{_c(x + 6)},{_c(lane_c + 6)} '
                    f'M{_c(x + 6)},{_c(lane_c - 6)} L{_c(x - 6)},{_c(lane_c + 6)}"/>')
        else:
            mark = f'<circle class="ch-mark-ring {cls}" cx="{_c(x)}" cy="{_c(lane_c)}" r="7"/>'
        num = _text(x, lane_c + 19, str(c["index"] + 1), cls="ch-tick", size=_FS_TICK, align="middle")
        title = _plain(c.get("title", ""))
        href = f"#{anchor}-c{c['index']}" if anchor else ""
        hit = (f'<rect class="ch-hit" x="{_c(x - 10)}" y="{_c(lane_c - 12)}" width="20" '
               f'height="36" fill="none" pointer-events="all"/>')
        inner = f"<title>{_vis(title)}</title>{hit}{mark}{num}"
        if href:
            out.append(f'<a href="{_attr(href)}" aria-label="{_attr(title)}">{inner}</a>')
        else:
            out.append(f"<g>{inner}</g>")
        table_rows.append([_num_cell(str(c["index"] + 1)), _cell(c["at"].strftime("%d.%m %H:%M")),
                           _cell(CONTACT_HE.get(c["kind"], c["kind"])), _cell(_plain(c.get("category", "")))])
    for s in sessions:
        table_rows.append([_cell(""), _cell(s["start"].strftime("%d.%m %H:%M")),
                           _cell(_plain(s.get("title", ""))), _cell("")])
    svg = _svg(W, H, name, "".join(out), interactive=True)
    return _figure("timeline", svg, table=_table(name, ["#", "מועד", "מה", "סיווג"], table_rows))


def timeline_legend(categories: Sequence[tuple[str, str]]) -> Markup:
    """The shared legend of every story timeline: contact shapes, category
    colours and banker unit kinds."""
    shapes = [("recorded_call", "●"), ("message", "■"), ("unrecorded_answered", "○"),
              ("abandoned", "✕")]
    parts = [f'<span class="ch-legend-item"><span class="tl-glyph">{g}</span>{_vis(CONTACT_HE[k])}</span>'
             for k, g in shapes]
    cats = _legend([("box", cls, lbl) for cls, lbl in categories])
    units = _legend([("box", "ch-unit-center ch-sess-exec", "מרכז הבנקאות"),
                     ("box", "ch-unit-own_branch ch-sess-exec", "סניף החשבון"),
                     ("box", "ch-unit-branch ch-sess-exec", "סניף אחר"),
                     ("box", "ch-unit-back_office ch-sess-exec", "תפעול עורפי"),
                     ("box", "ch-unit-center ch-sess-view", "צפייה בלבד (בהיר)")])
    return Markup(f'<div class="tl-legend"><div class="ch-legend">{"".join(parts)}</div>{cats}{units}</div>')


__all__ = ["columns", "hbars", "km_chart", "stacked_bars", "story_timeline", "timeline_legend"]

