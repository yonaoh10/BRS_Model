"""Render the management report: one self-contained HTML file, plus a numbers-
only CSV and JSON beside it for Excel and BI tools.

The HTML carries everything it needs - styles, script, charts and the data
the level-3 explorer filters - because it is opened from disk on a desktop
with no network, forwarded by e-mail, and served by the dashboard, which only
serves single .html files. Text from calls enters it only through `_prose`,
which re-applies redaction and strips markup, and only for calls the dataset
gate allowed.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from markupsafe import Markup, escape

from callqa import __version__
from callqa.reporting.common import SPEAKER_HE, jinja_env, load_recommendations, safe_filename
from callqa.reporting.executive import charts
from callqa.reporting.executive.analysis import (
    BAND_LABEL,
    DRIVERS,
    DURATION_BUCKETS,
    Analysis,
    analyse,
    band_of,
    duration_bucket,
    type_bucket,
)
from callqa.reporting.executive.dataset import BatchFilters, CallRecord, load_batch
from callqa.reporting.executive.findings import SEVERITY_HE, Opinion, build_opinion
from callqa.rubric import Rubric
from callqa.state import atomic_write_text

logger = logging.getLogger(__name__)

DEFAULT_NAME = "executive"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")
MAX_PROSE = 600      # characters of one reasoning / summary line
MAX_QUOTE = 320
GRANULARITY_HE = {"day": "יום", "week": "שבוע", "month": "חודש"}
DEMO_TEXT = {
    "mock": ("דוח הדגמה — שיפוט מדומה",
             "הציונים בדוח זה הופקו על ידי מנוע בדיקה (mock) לצורכי פיתוח ואינם הערכה של "
             "שיחות אמיתיות."),
    "synthetic-demo": ("דוח הדגמה — נתונים סינתטיים",
                       "כל השיחות, הציונים והציטוטים בדוח זה הומצאו כדי להדגים את מבנה הדוח."),
}


@dataclass
class ExecutiveReport:
    html: Path
    csv: Path
    json: Path
    analysis: Analysis
    opinion: Opinion


# -- text safety ------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]{0,200}>")


def _prose(text: str | None, limit: int = MAX_PROSE) -> str:
    """Judge-written or quoted text, made safe to embed.

    Scorecards on disk can predate today's redaction rules, so the text is
    redacted again here; any tag-like run is removed (a mask token such as
    <ת"ז:████> becomes a plain placeholder first, so it survives visibly).
    """
    if not text:
        return ""
    from callqa.redaction import _MASK_TOKEN_RE, redact_text

    redacted = redact_text(text)[0]
    redacted = _MASK_TOKEN_RE.sub(lambda m: "[" + m.group(0)[1:-1] + "]", redacted)
    redacted = _TAG_RE.sub("", redacted)
    redacted = re.sub(r"\s+", " ", redacted).strip()
    return redacted[:limit] + ("…" if len(redacted) > limit else "")


def script_json(data: object) -> str:
    """JSON for a <script type="application/json"> block.

    Hebrew stays literal (readable, a third of the size, and visible to the
    repository's leak sweep); the characters that could end the script
    element or confuse an HTML parser are escaped.
    """
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return (text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


# -- template helpers ----------------------------------------------------------

def _fmt(value: float | None, digits: int = 1) -> str:
    """At most `digits` decimals; a true minus sign; never '-0'."""
    if value is None:
        return "—"
    r = round(value, digits)
    if r == 0:
        r = 0.0
    body = str(int(abs(r))) if digits <= 1 and r == int(r) else f"{abs(r):.{digits}f}"
    return ("−" if r < 0 else "") + body


def _fixed1(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _pct0(p: float | None) -> str:
    return "—" if p is None else f"{round(p * 100)}%"


def _pct1(p: float | None) -> str:
    return "—" if p is None else f"{_fmt(p * 100)}%"


def _plus(x: float | None) -> str:
    if x is None:
        return "—"
    return f"+{_fmt(x)}" if x > 0.05 else "0"


def _ci(stat, digits: int = 1) -> str:  # noqa: ANN001 - MeanCI
    if stat.low is None or stat.high is None:
        return "—"
    return f"{_fmt(stat.low, digits)}–{_fmt(stat.high, digits)}"


def _date_he(d: date | None) -> str:
    return d.strftime("%d.%m.%Y") if d else "—"


def _rich(text: str) -> Markup:
    """Escape, then isolate the ⟦numbers⟧ the findings marked for RTL layout."""
    escaped = str(escape(text))
    return Markup(re.sub(r"⟦(.*?)⟧", r'<span class="n">\1</span>', escaped))


def _fjson(obj: object) -> str:
    # Autoescaped by the template when placed in an attribute.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _pct_width(mean: float | None) -> float:
    return 0.0 if mean is None else round(max(0.0, min(1.0, (mean - 1) / 4)) * 100, 1)


# -- charts ----------------------------------------------------------------------

# Design widths (CSS px) of the places a chart sits in: labels stay 11px only
# when a chart is drawn for the column it is shown in.
W_FULL = 1110
W_HALF = 540
W_WIDE = 670

def _charts(a: Analysis) -> dict[str, Markup | None]:
    out: dict[str, Markup | None] = {}
    if not a.n_scored:
        return out
    out["stacked"] = charts.stacked_levels(
        [{"label": d.name + (" (שער)" if d.gate else ""), "counts": d.counts,
          "note": f"ממוצע {_fmt(d.mean.mean, 2)}", "filter": {"dim": d.id, "max": 2}}
         for d in a.dims],
        label="התפלגות הציונים 1–5 בכל ממד", level_labels=["1", "2", "3", "4", "5"],
        width=W_FULL)
    items = [(d.name, round(d.lost_mean, 2)) for d in a.dims]
    filters: list[dict | None] = [{"dim": d.id, "max": 2} for d in a.dims]
    if a.gate_penalty_mean > 0:
        items.append(("קנס כשלי שער", round(a.gate_penalty_mean, 2)))
        filters.append({"gate": "any"})
    order = sorted(range(len(items)), key=lambda i: -items[i][1])
    out["pareto"] = charts.pareto_chart(
        [items[i] for i in order], label="נקודות מדד שאבדו בממוצע לשיחה, לפי מקור",
        unit="נק'", total_label="סה״כ מתחת ל־100",
        filters=[filters[i] for i in order], width=W_FULL)
    bands = [(0.0, 40.0, 1), (40.0, 60.0, 2), (60.0, 80.0, 3), (80.0, 100.0, 4)]
    markers = [(a.median, f"חציון {_fmt(a.median)}")] if a.median is not None else []
    out["histogram"] = charts.histogram_chart(
        a.histogram, label="התפלגות מדד האיכות בשיחות", x_title="מדד (0–100)",
        y_title="שיחות", bands=bands, markers=markers, width=W_WIDE)
    if len(a.periods) >= 2:
        out["trend"] = charts.trend_chart(
            [{"label": p.label, "mean": p.mean.mean, "low": p.mean.low, "high": p.mean.high,
              "n": p.n, "rate": p.gate_rate} for p in a.periods],
            label="מדד האיכות לאורך זמן", y_title="מדד",
            y_range=_trend_range(a), rate_title="כשל שער", width=W_FULL)
    out["seg_type"] = _dot(a.segments.get("call_type", []), a, "המדד לפי סוג שיחה", W_HALF)
    out["seg_duration"] = _dot(a.segments.get("duration", []), a, "המדד לפי משך השיחה", W_HALF)
    out["seg_layout"] = _dot(a.segments.get("layout", []), a, "המדד לפי אופן ההקלטה", W_FULL)
    if a.bankers:
        out["bankers"] = _dot(a.bankers, a, "המדד הממוצע של כל בנקאי", W_FULL)
    # Only relationships that passed every bar: a red or green bar reads as a
    # finding, and chance-level differences do not get to look like one. The
    # table below the chart lists every behaviour, significant or not.
    rows = [{"label": d.label, "value": round(d.contrast.diff, 1),
             "detail": f"ρ={_fmt(d.corr.rho, 2)} · {d.n} שיחות",
             "filter": d.filter_high or None}
            for d in a.drivers if d.reportable]
    out["drivers"] = charts.diverging_bars(
        rows, label="הפרש המדד בין הרבעון העליון לתחתון, במדדים ההתנהגותיים המובהקים",
        width=W_FULL) if rows else None
    return out


def _trend_range(a: Analysis) -> tuple[float, float]:
    lows = [p.mean.low if p.mean.low is not None else p.mean.mean for p in a.periods
            if p.mean.mean is not None]
    highs = [p.mean.high if p.mean.high is not None else p.mean.mean for p in a.periods
             if p.mean.mean is not None]
    if not lows:
        return (0.0, 100.0)
    lo = max(0.0, (min(lows) // 10) * 10 - 5)
    hi = min(100.0, (max(highs) // 10 + 1) * 10 + 5)
    return (lo, hi)


def _dot(groups, a: Analysis, label: str, width: int):  # noqa: ANN001, ANN202
    if not groups:
        return None
    rows = []
    for g in groups:
        flag = {"bad": "bad", "ok": "ok", "few": "muted"}.get(g.flag)
        sub = f"{g.n} שיחות" + (" · מעט שיחות" if g.flag == "few" else "")
        rows.append({"label": g.label, "mean": g.mean.mean if g.mean.mean is not None else 0.0,
                     "low": g.mean.low, "high": g.mean.high, "n": g.n, "flag": flag,
                     "filter": g.filter, "sub": sub})
    values = [x for g in groups for x in (g.mean.low, g.mean.high, g.mean.mean) if x is not None]
    lo = max(0.0, (min(values) // 10) * 10 - 10) if values else 0.0
    hi = min(100.0, (max(values) // 10 + 1) * 10 + 5) if values else 100.0
    return charts.dot_plot(rows, label=label, reference=a.median, reference_label="חציון",
                           x_range=(lo, hi), width=width)


# -- context --------------------------------------------------------------------

def _kpis(a: Analysis) -> list[dict]:
    spark_mean = [p.mean.mean for p in a.periods]
    spark_high = [p.high_rate * 100 if p.high_rate is not None else None for p in a.periods]
    spark_gate = [p.gate_rate * 100 if p.gate_rate is not None else None for p in a.periods]
    has_spark = len(a.periods) >= 3

    def spark(values: list[float | None], label: str, rng: tuple[float, float] | None = None):  # noqa: ANN202
        return charts.sparkline(values, label=label, y_range=rng) if has_spark else None

    gate_p = a.gate_rate.p or 0.0
    weakest = min((d for d in a.dims if d.mean.mean is not None), key=lambda d: d.mean.mean,
                  default=None)
    n_bankers = len(a.bankers)
    ranked = [b for b in a.bankers if b.flag != "few"]
    kpis = [
        {"label": "מדד איכות ממוצע", "value": _fmt(a.total.mean), "unit": "/ 100",
         "sub": f"רווח סמך 95%: ⟦{_ci(a.total)}⟧ · חציון ⟦{_fmt(a.median)}⟧",
         "spark": spark(spark_mean, "מדד ממוצע לאורך זמן"), "href": "#sec-dist",
         "badge": BAND_LABEL[band_of(a.total.mean)] if a.total.mean is not None else None,
         "badge_cls": "muted", "cls": ""},
        {"label": "שיחות ברמה גבוהה (80+)", "value": _pct1(a.high_rate.p), "unit": "",
         "sub": f"⟦{a.band_counts.get(4, 0):,}⟧ שיחות · מתחת ל־60: ⟦{_pct1(a.low_rate.p)}⟧",
         "spark": spark(spark_high, "שיעור השיחות ברמה גבוהה לאורך זמן", (0, 100)),
         "href": "#sec-dist", "cls": ""},
        {"label": "כשל בשער חובה", "value": _pct1(a.gate_rate.p), "unit": "",
         "sub": f"⟦{round(gate_p * a.n_scored):,}⟧ שיחות · זיהוי לקוח או גילוי נאות",
         "spark": spark(spark_gate, "שיעור כשלי השער לאורך זמן",
                        (0, max([10.0] + [v for v in spark_gate if v is not None]))),
         "href": "#sec-risk", "badge": "סיכון מהותי" if gate_p >= 0.05 else
         ("דורש מעקב" if gate_p >= 0.02 else None),
         "badge_cls": "bad" if gate_p >= 0.05 else "warn",
         "cls": "risk" if gate_p >= 0.05 else ("warnk" if gate_p >= 0.02 else "")},
        {"label": "השיחות בדוח", "value": f"{a.n_scored:,}", "unit": f"מתוך {a.n_scope:,}",
         "sub": f"⟦{n_bankers}⟧ בנקאים · ⟦{len(ranked)}⟧ עם ⟦8⟧ שיחות ומעלה",
         "spark": None, "href": "#sec-coverage", "cls": ""},
        {"label": "ממתינות לבדיקה אנושית", "value": _pct1(a.review_rate.p), "unit": "",
         "sub": f"⟦{a.n_held:,}⟧ שיחות · ⟦{a.n_failed:,}⟧ לא עובדו",
         "spark": None, "href": "#sec-coverage",
         "badge": "גבוה" if (a.review_rate.p or 0) >= 0.1 else None, "badge_cls": "warn",
         "cls": "warnk" if (a.review_rate.p or 0) >= 0.1 else ""},
        {"label": "הממד החלש ביותר", "value": _fmt(weakest.mean.mean, 2) if weakest else "—",
         "unit": "/ 5", "sub": (f"„{weakest.name}” · ⟦{_pct0(weakest.low_rate)}⟧ מהשיחות בציון ⟦1–2⟧"
                                if weakest else ""),
         "spark": None, "href": "#sec-dims", "cls": ""},
    ]
    return kpis


def _heat(a: Analysis) -> list[dict]:
    rows = []
    flag_text = {"bad": ("נמוך מובהק", "bad"), "ok": ("גבוה מובהק", "ok"),
                 "none": ("בטווח", "muted"), "few": ("מעט שיחות", "muted")}
    for b in a.bankers:
        cells = []
        for d in a.dims:
            v = b.dim_means.get(d.id)
            cells.append({"v": v, "text": _fixed1(v), "name": d.name,
                          "cls": charts.level_class(v) if v is not None else "",
                          "filter": {"banker": b.key, "dim": d.id, "max": 2}})
        text, cls = flag_text[b.flag]
        rows.append({"banker": b.key, "n": b.n, "mean": b.mean.mean, "cells": cells,
                     "flag_text": text, "flag_cls": cls, "link": b.link})
    return rows


def _feature_values(record: CallRecord) -> dict[str, float]:
    if record.features is None:
        return {}
    out = {}
    for key, _, _, getter in DRIVERS:
        try:
            value = getter(record.features)
        except (TypeError, ZeroDivisionError):
            value = None
        if value is not None:
            out[key] = round(float(value), 2)
    return out


def _explorer_data(a: Analysis, with_text: bool, recs: dict[str, list[str]]) -> dict:
    dims = a.dims
    short = {d.id: _short(d.id, d.name) for d in dims}
    bucket = type_bucket(a.batch.scored)
    calls = []
    text: dict[str, dict] = {}
    for r in a.batch.records:
        card = r.card
        entry = {
            "id": r.call_id, "b": r.banker_id,
            "d": r.call_date.isoformat() if r.call_date else "",
            "t": bucket.get(r.call_id, r.call_type),
            "sec": round(r.duration_sec) if r.duration_sec is not None else None,
            "du": duration_bucket(r.duration_sec) or "",
            "ly": "" if r.mono is None else ("mono" if r.mono else "stereo"),
            "s": r.status,
            "tot": card.weighted_total if card else None,
            "g": list(card.failed_gates) if card else [],
            "sc": [card.scores[d.id].score if d.id in card.scores else None for d in dims]
            if card else None,
            "rep": r.call_report,
            "r": (", ".join(r.held_reasons) if r.held_reasons else (r.failed_reason or "")),
            "f": _feature_values(r) if card else {},
        }
        calls.append(entry)
        if card is not None and with_text and r.text_allowed:
            text[r.call_id] = {
                "sum": _prose(card.summary_he),
                "dev": _prose(card.development_area_he),
                "str": [p for p in (_prose(s) for s in card.strengths_he[:3]) if p],
                "dims": [_dim_text(card, d.id) for d in dims],
            }
    bankers = {}
    for b in a.bankers:
        bankers[b.key] = {
            "n": b.n, "mean": b.mean.mean, "lo": b.mean.low, "hi": b.mean.high, "flag": b.flag,
            "link": b.link, "weak": b.weakest,
            "coach": (recs.get(b.weakest) or [None])[0] if b.weakest else None,
        }
    return {
        "dims": [{"id": d.id, "name": d.name, "short": short[d.id], "gate": d.gate} for d in dims],
        "gd": [d.mean.mean for d in dims], "gm": a.total.mean,
        "calls": calls, "text": text, "bankers": bankers, "withText": with_text,
        "features": [{"key": k, "label": label, "unit": unit} for k, label, unit, _ in DRIVERS],
        "durations": {k: label for k, label, _, _ in DURATION_BUCKETS},
    }


def _dim_text(card, dim_id: str) -> list | None:  # noqa: ANN001 - ScoreCard
    entry = card.scores.get(dim_id)
    if entry is None:
        return None
    ev = entry.evidence[0] if entry.evidence else None
    return [_prose(entry.reasoning_he),
            _prose(ev.quote, MAX_QUOTE) if ev else "",
            ev.timestamp if ev and re.fullmatch(r"\d{1,3}:\d{2}", ev.timestamp or "") else "",
            ev.speaker if ev else ""]


SHORT_NAMES = {
    "identification": "זיהוי", "compliance": "ציות", "empathy": "אמפתיה",
    "listening": "הקשבה", "clarity": "בהירות", "resolution": "פתרון",
    "suitability": "התאמה", "closure": "סגירה",
}


def _short(dim_id: str, name: str) -> str:
    """A compact label for a dimension, for the explorer's narrow columns."""
    return SHORT_NAMES.get(dim_id) or name.split()[0]


def _period_label(a: Analysis) -> str:
    if a.first_date and a.last_date:
        if a.first_date == a.last_date:
            return f"⟦{_date_he(a.first_date)}⟧"
        return f"⟦{_date_he(a.first_date)}⟧ – ⟦{_date_he(a.last_date)}⟧"
    return "לא ידועה"


def _scope_chips(filters: BatchFilters) -> list[str]:
    chips = []
    if filters.date_from:
        chips.append(f"מתאריך {_date_he(filters.date_from)}")
    if filters.date_to:
        chips.append(f"עד תאריך {_date_he(filters.date_to)}")
    if filters.call_type:
        chips.append(f"סוג שיחה: {filters.call_type[:40]}")
    if filters.banker_id:
        chips.append(f"בנקאי: {filters.banker_id[:40]}")
    if filters.run_id:
        chips.append(f"הרצה: {filters.run_id}")
    return chips


def _trend_text(a: Analysis) -> str | None:
    t = a.trend
    if t is None or t.slope_per_30d is None:
        return None
    def signed1(x: float) -> str:
        r = round(x, 1)
        return ("+" if r > 0 else "−" if r < 0 else "") + f"{abs(r):.1f}"

    ci = (f" (רווח סמך 95%: ⟦{signed1(t.low)}⟧ עד ⟦{signed1(t.high)}⟧)"
          if t.low is not None and t.high is not None else "")
    word = "מובהק" if t.significant else "לא מובהק"
    return (f"קו מגמה על פני ⟦{t.n:,}⟧ שיחות ו־⟦{round(t.span_days)}⟧ ימים: "
            f"⟦{signed1(t.slope_per_30d)}⟧ נקודות בחודש{ci} — {word}.")


def _weakest_of(record: CallRecord) -> str:
    card = record.card
    if card is None or not card.scores:
        return ""
    return min(card.scores.items(), key=lambda kv: (kv[1].score, kv[0]))[0]


def render_html(a: Analysis, opinion: Opinion, *, with_text: bool = True,
                title: str | None = None, generated_at: datetime | None = None) -> str:
    recs = load_recommendations()
    batch = a.batch
    generated = (generated_at or datetime.now(UTC)).astimezone()
    engines = batch.judge_engines
    demo_key = next((e for e in ("synthetic-demo", "mock") if e in engines), None)
    env = jinja_env()
    env.filters["rich"] = _rich
    env.filters["fjson"] = _fjson
    bands = [{"band": b, "label": BAND_LABEL[b], "n": a.band_counts.get(b, 0),
              "share": (a.band_counts.get(b, 0) / a.n_scored) if a.n_scored else None}
             for b in (4, 3, 2, 1)]
    from callqa.redaction import ENTITY_LABELS_HE

    entities = [(ENTITY_LABELS_HE.get(k, k), n) for k, n in a.coverage["entities"]]
    gate_names = [d.name for d in a.dims if d.gate]
    sev = opinion.top_findings[0].severity if opinion.top_findings else "info"
    context = {
        "version": __version__,
        "title": title or "דוח מנהלים — איכות השירות הטלפוני",
        "generated_at": generated.strftime("%d.%m.%Y %H:%M"),
        "period_label": _period_label(a),
        "scope_chips": _scope_chips(batch.filters),
        "with_text": with_text,
        "demo": demo_key is not None,
        "demo_title": DEMO_TEXT.get(demo_key, ("", ""))[0],
        "demo_text": DEMO_TEXT.get(demo_key, ("", ""))[1],
        "a": a, "opinion": opinion, "severity_he": SEVERITY_HE,
        "verdict_class": {"critical": "sev-critical", "high": "sev-high"}.get(sev, "sev-ok"
                                                                              if sev == "positive"
                                                                              else ""),
        "band_label": BAND_LABEL[band_of(a.total.mean)] if a.total.mean is not None else "—",
        "index_delta": _index_delta(a),
        "kpis": _kpis(a), "charts": _charts(a), "heat": _heat(a), "bands": bands,
        "empty": Markup('<p class="ch-empty">אין די נתונים.</p>'),
        "granularity_he": GRANULARITY_HE.get(a.granularity or "", ""),
        "has_gate_line": any(p.gate_rate for p in a.periods),
        "trend_text": _trend_text(a),
        "cov": a.coverage, "entities": entities,
        "models": ", ".join(batch.models) or "—",
        "prompt_versions": ", ".join(batch.prompt_versions) or "—",
        "rubric_hashes": ", ".join(h[:12] for h in batch.rubric_hashes) or "—",
        "duration_buckets": DURATION_BUCKETS,
        "cases_best": {k: [_safe_example(e) for e in v] for k, v in a.best_examples.items()}
        if with_text else {},
        "cases_worst": {k: [_safe_example(e) for e in v] for k, v in a.worst_examples.items()}
        if with_text else {},
        "cases_any": with_text and any(a.best_examples.get(d.id) or a.worst_examples.get(d.id)
                                       for d in a.dims),
        "weights_text": " · ".join(f"{d.name} {round(d.weight * 100)}%" for d in a.dims),
        "gates_text": " ו".join(f"„{n}”" for n in gate_names) or "—",
        "speaker_he": SPEAKER_HE,
        "footer": _footer(a, generated),
        "data_json": Markup(script_json(_explorer_data(a, with_text, recs))),
        "fmt": _fmt, "pct0": _pct0, "pct1": _pct1, "plus": _plus, "ci": _ci,
        "date_he": _date_he, "pct_width": _pct_width, "weakest_of": _weakest_of,
        "rho": lambda v: _fmt(v, 2),
    }
    return env.get_template("executive_report.html.j2").render(**context)


def _safe_example(e):  # noqa: ANN001, ANN202 - analysis.Example
    from dataclasses import replace

    return replace(e, quote=_prose(e.quote, MAX_QUOTE), reasoning=_prose(e.reasoning),
                   timestamp=e.timestamp if re.fullmatch(r"\d{1,3}:\d{2}", e.timestamp or "")
                   else "")


def _index_delta(a: Analysis) -> dict | None:
    d = a.last_vs_prev
    if d is None or d.diff is None or len(a.periods) < 2:
        return None
    if round(abs(d.diff), 1) == 0:
        return {"cls": "flat", "text": "ללא שינוי מול התקופה הקודמת"}
    cls = "flat"
    if d.significant and abs(d.diff) >= 1:
        cls = "up" if d.diff > 0 else "down"
    sign = "+" if d.diff > 0 else "−"
    return {"cls": cls, "text": f"⟦{sign}{_fmt(abs(d.diff))}⟧ מול התקופה הקודמת"
                                + ("" if d.significant else " (לא מובהק)")}


def _footer(a: Analysis, generated: datetime) -> str:
    b = a.batch
    parts = [f"callqa {__version__}", f"הופק {generated.strftime('%d.%m.%Y %H:%M')}",
             f"{a.n_scope:,} שיחות בהיקף, {a.n_scored:,} מנוקדות"]
    if b.scored_at:
        parts.append(f"ניקוד: {b.scored_at[0][:10]} עד {b.scored_at[1][:10]}")
    parts.append(f"מנוע שיפוט: {', '.join(b.judge_engines) or '—'}")
    return " · ".join(parts)


# -- exports --------------------------------------------------------------------

def _csv_cell(value: object) -> str:
    text = "" if value is None else str(value)
    # Excel runs a cell that starts with one of these as a formula.
    if text[:1] in ("=", "+", "-", "@", "\t", "\r"):
        text = "'" + text
    return text


def render_csv(a: Analysis) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    dims = [d.id for d in a.dims]
    writer.writerow(["call_id", "call_date", "banker_id", "call_type", "duration_sec", "status",
                     "index", "band", "gate_failed", "failed_gates", *dims])
    bucket = type_bucket(a.batch.scored)
    for r in a.batch.records:
        card = r.card
        writer.writerow([_csv_cell(v) for v in (
            r.call_id, r.call_date.isoformat() if r.call_date else "", r.banker_id,
            bucket.get(r.call_id, r.call_type),
            round(r.duration_sec) if r.duration_sec is not None else "", r.status,
            card.weighted_total if card else "",
            BAND_LABEL[band_of(card.weighted_total)] if card else "",
            ("yes" if card.gate_failed else "no") if card else "",
            "|".join(card.failed_gates) if card else "",
            *[(card.scores[d].score if card and d in card.scores else "") for d in dims])])
    return "\ufeff" + out.getvalue()


def render_json(a: Analysis, opinion: Opinion) -> str:
    """Numbers only - for BI tools and for auditing what the page said."""
    def ci(s):  # noqa: ANN001, ANN202
        return {"n": s.n, "mean": s.mean, "low": s.low, "high": s.high}

    data = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": __version__,
        "scope": {"calls": a.n_scope, "scored": a.n_scored, "held": a.n_held,
                  "failed": a.n_failed,
                  "first_date": a.first_date.isoformat() if a.first_date else None,
                  "last_date": a.last_date.isoformat() if a.last_date else None},
        "index": ci(a.total), "median": a.median,
        "high_rate": a.high_rate.p, "gate_rate": a.gate_rate.p, "review_rate": a.review_rate.p,
        "dimensions": [{"id": d.id, "mean": d.mean.mean, "low": d.mean.low, "high": d.mean.high,
                        "counts": d.counts, "lost_points": round(d.lost_mean, 3),
                        "gain_if_floor3": round(d.gain_if_floor3, 3)} for d in a.dims],
        "gate_penalty": round(a.gate_penalty_mean, 3),
        "periods": [{"start": p.start.isoformat(), "n": p.n, "mean": p.mean.mean,
                     "gate_rate": p.gate_rate} for p in a.periods],
        "bankers": [{"banker_id": b.key, "n": b.n, **ci(b.mean), "flag": b.flag,
                     "gate_rate": b.gate.p} for b in a.bankers],
        "segments": {k: [{"key": g.key, "n": g.n, **ci(g.mean), "flag": g.flag} for g in v]
                     for k, v in a.segments.items()},
        "findings": [{"key": f.key, "severity": f.severity} for f in opinion.findings],
        "coverage": {"held_reasons": dict(a.coverage["held_reasons"]),
                     "failed_reasons": dict(a.coverage["failed_reasons"]),
                     "redaction_disabled": a.coverage["redaction_disabled"],
                     "excluded_other_rubric": a.coverage["excluded_other_rubric"]},
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


# -- entry point ----------------------------------------------------------------

def report_name(filters: BatchFilters, name: str | None) -> str:
    """The output file stem: executive, or executive-<scope> for a scoped report."""
    if name:
        if not _NAME_RE.match(name):
            raise ValueError("report name: letters, digits, dot, underscore, hyphen only")
        return name if name.startswith(DEFAULT_NAME) else f"{DEFAULT_NAME}-{name}"
    if not filters.active:
        return DEFAULT_NAME
    parts = []
    if filters.date_from or filters.date_to:
        parts.append(f"{filters.date_from or ''}_{filters.date_to or ''}".strip("_"))
    for value in (filters.call_type, filters.banker_id, filters.run_id):
        if value:
            parts.append(safe_filename(value))
    return f"{DEFAULT_NAME}-" + safe_filename("-".join(p for p in parts if p))[:60]


def build_executive_report(output_dir: Path, rubric: Rubric, filters: BatchFilters | None = None,
                           *, with_text: bool = True, name: str | None = None,
                           title: str | None = None) -> ExecutiveReport:
    filters = filters or BatchFilters()
    stem = report_name(filters, name)
    batch = load_batch(output_dir, rubric, filters)
    analysis = analyse(batch)
    opinion = build_opinion(analysis)
    html = render_html(analysis, opinion, with_text=with_text, title=title)
    reports = output_dir / "reports"
    html_path = reports / f"{stem}.html"
    csv_path = reports / f"{stem}_calls.csv"
    json_path = reports / f"{stem}.json"
    atomic_write_text(html_path, html)
    atomic_write_text(csv_path, render_csv(analysis))
    atomic_write_text(json_path, render_json(analysis, opinion))
    logger.info("executive report: %d call(s), %d scored -> %s", analysis.n_scope,
                analysis.n_scored, html_path.name)
    return ExecutiveReport(html=html_path, csv=csv_path, json=json_path, analysis=analysis,
                           opinion=opinion)
