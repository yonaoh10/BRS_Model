"""Render the journey report: one self-contained HTML file, with the stories
and the returns as CSV and the numbers as JSON beside it.

Like the management report it opens from disk with no network, carries its
own styles, script and data, and places text from calls only through
`_prose` (redaction re-applied, markup stripped). What it adds is the
customer's side: stories of contacts over time, why customers came back,
promises and whether they were kept, and the bankers' work behind it (Atlas).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from markupsafe import Markup

from callqa import __version__
from callqa.config import Config
from callqa.journey.analysis import (
    GAP_BUCKETS,
    KIND_HE,
    OBJECTIVE_HE,
    PENDING,
    JourneyAnalysis,
    analyse,
    shown_category,
)
from callqa.journey.findings import build_opinion
from callqa.journey.models import ContentLayer, JourneyDataset
from callqa.journey.rules import RuleSettings, StoryFacts
from callqa.journey.store import dataset_dir, load_content, load_dataset, resolve_dataset_id
from callqa.journey.vocab import Taxonomy, Units, load_taxonomy, load_units
from callqa.reporting.common import jinja_env
from callqa.reporting.executive import numfmt
from callqa.reporting.executive.charts import SHOW_CALLS_HE
from callqa.reporting.executive.dataset import shown_banker
from callqa.reporting.executive.findings import SEVERITY_HE, Opinion
from callqa.reporting.executive.render import (
    DEMO_TEXT,
    _csv_cell,
    _fjson,
    _prose,
    _rich,
    script_json,
)
from callqa.reporting.journey import charts
from callqa.state import atomic_write_text

logger = logging.getLogger(__name__)

DEFAULT_NAME = "journey"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")
# The one shape of a journey report's file name: the dashboard serves, and the
# index lists, exactly the files that match it.
JOURNEY_REPORT_FILE_RE = re.compile(r"journey(?:-[A-Za-z0-9][A-Za-z0-9._-]{0,60})?\.html")
# Each story card carries a timeline drawing (~6 KB). Past this many stories
# the rest keep their card and table, and the drawing goes to the stories a
# reader opens first: most returns, then most failures.
MAX_TIMELINES = 400
MAX_QUOTE = 320

W_FULL, W_HALF = 1110, 540

STATUS_HE = {"closed": "נסגר", "open": "פתוח", "unclear": "לא ברור"}
BASIS_HE = {"fact": "עובדה", "content": "לפי התוכן", "inference": "הסקה", "none": "—"}
DECIDED_HE = {"rule": "חוק", "llm": "מודל שפה", "llm+rule": "מודל וחוק", "none": "—"}
COVERAGE_HE = {"full": "כיסוי אטלס מלא", "partial": "כיסוי אטלס חלקי", "none": "אין נתוני אטלס"}
DIRECTION_HE = {"inbound": "נכנסת", "outbound": "יוצאת", "unknown": ""}
PROMISE_HE = {"callback": "לחזור ללקוח", "send_document": "לשלוח מסמך",
              "execute_action": "לבצע פעולה", "check_and_update": "לבדוק ולעדכן"}
OUTCOME_HE = {"kept": "קוימה", "broken": "הופרה", "unknown": "לא ידוע"}
SETTLED_HE = {"bank_contact": "הבנק חזר ללקוח", "atlas_execute": "בוצעה פעולה בחשבון",
              "customer_returned": "הלקוח חזר לפני שהבנק פעל",
              "deadline": "עבר המועד בלי פעולה"}
UNIT_KIND_HE = {"center": "מרכז הבנקאות", "own_branch": "סניף החשבון", "branch": "סניף אחר",
                "back_office": "תפעול עורפי", "other": "יחידה אחרת"}
CATEGORY_ORDER = ["unclosed_loop", "excessive_runaround", "legit_return", "new_topic",
                  "bank_initiated", "unclassifiable", PENDING]
PENDING_HE = "ממתין לסיווג לפי תוכן"


@dataclass
class JourneyReport:
    html: Path
    stories_csv: Path
    returns_csv: Path
    json: Path
    analysis: JourneyAnalysis
    opinion: Opinion


# -- helpers ----------------------------------------------------------------------

def _pct1(p: float | None) -> str:
    return numfmt.percent(p, 1)


def _fmt(v: float | None, d: int = 1) -> str:
    return numfmt.fmt(v, d)


def _cat_cls(cat: str | None) -> str:
    if not cat:
        return "ch-cat-first"
    return f"ch-cat-{cat}" if cat in CATEGORY_ORDER else "ch-cat-other"


def _story_label(no: int) -> str:
    return f"סיפור {no:03d}"


def _dt(t: datetime | None) -> str:
    return t.strftime("%d.%m.%Y %H:%M") if t else "—"


def _stories_word(n: int) -> str:
    return "סיפור אחד" if n == 1 else f"{n:,} סיפורים"


def _journey_markup(m: Markup) -> Markup:
    # The shared charts announce "show the calls"; here a row opens stories.
    return Markup(str(m).replace(SHOW_CALLS_HE, "הצגת הסיפורים"))


def _categories(tax: Taxonomy, pending: bool = False) -> list[str]:
    extra = [c for c in tax.categories if c not in CATEGORY_ORDER]
    return ([c for c in CATEGORY_ORDER if c in tax.categories] + extra
            + ([PENDING] if pending else []))


def _cat_label(tax: Taxonomy, cat: str | None) -> str:
    return PENDING_HE if cat == PENDING else tax.category_label(cat)


def _has_pending(a: JourneyAnalysis) -> bool:
    return bool(a.categories_strict.get(PENDING))


# -- charts -------------------------------------------------------------------------

def _charts(a: JourneyAnalysis, facts: list[StoryFacts], tax: Taxonomy) -> dict[str, Markup | None]:
    out: dict[str, Markup | None] = {}
    cats = _categories(tax, _has_pending(a))
    segs = [(c, _cat_label(tax, c), _cat_cls(c)) for c in cats]

    buckets = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3–4", 3, 4), ("5–9", 5, 9), ("10+", 10, 10**6)]
    out["returns_dist"] = charts.columns(
        [(lb, sum(1 for s in a.stories if lo <= s.returns <= hi)) for lb, lo, hi in buckets],
        label="סיפורים לפי מספר החזרות", y_title="סיפורים", width=W_HALF)
    kinds = Counter()
    for s in a.stories:
        kinds.update(s.kinds)
    out["kinds"] = charts.hbars(
        [{"label": KIND_HE.get(k, k), "value": v} for k, v in kinds.most_common()],
        label="המגעים לפי ערוץ", width=W_HALF)

    def cat_items(dist: dict[str, int]) -> list[dict]:
        return [{"label": _cat_label(tax, c), "value": dist.get(c, 0), "cls": _cat_cls(c),
                 "filter": {"category": c}} for c in cats if dist.get(c)]
    out["cat_strict"] = charts.hbars(cat_items(a.categories_strict),
                                     label="סיבות החזרה — סיווג קפדני", width=W_HALF)
    out["cat_extended"] = charts.hbars(cat_items(a.categories_extended),
                                       label="סיבות החזרה — כולל הסקה מרצף האירועים", width=W_HALF)
    out["objective"] = charts.hbars(
        [{"label": OBJECTIVE_HE.get(k, k), "value": v}
         for k, v in sorted(a.objective_classes.items(), key=lambda kv: -kv[1])],
        label="מה ידוע על כל חזרה (עובדות, בלי תוכן)", width=W_FULL)
    out["channel_cat"] = charts.stacked_bars(
        [{"label": ch, "counts": counts} for ch, counts in sorted(
            a.channel_by_category.items(), key=lambda kv: -sum(kv[1].values()))],
        segs, label="סיבת החזרה לפי ערוץ", width=W_FULL)

    # topic x category over the returns of each story's topic
    topic_of = {s.story_key: s.topic for s in a.stories}
    tc: dict[str, Counter] = defaultdict(Counter)
    for f in facts:
        t = topic_of.get(f.timeline.story.story_key, "other")
        for j in f.judgements.values():
            tc[t][shown_category(j)] += 1
    out["topic_cat"] = charts.stacked_bars(
        [{"label": tax.topic_label(t), "counts": dict(c), "filter": {"topic": t}}
         for t, c in sorted(tc.items(), key=lambda kv: -sum(kv[1].values()))],
        segs, label="סיבת החזרה לפי נושא", width=W_FULL)

    out["gaps"] = charts.columns([(lb, a.gaps.get(lb, 0)) for _l, lb in GAP_BUCKETS],
                                 label="הזמן בין מגע למגע הבא", y_title="מגעים", width=W_HALF)
    sd = Counter(min(s.max_same_day, 4) for s in a.stories)
    out["same_day"] = charts.columns(
        [("1", sd.get(1, 0)), ("2", sd.get(2, 0)), ("3", sd.get(3, 0)), ("4+", sd.get(4, 0))],
        label="סיפורים לפי מספר המגעים המרבי ביום אחד", y_title="סיפורים", width=W_HALF)
    out["km"] = charts.km_chart(a.km, label="שיעור הסיפורים שעדיין פתוחים, לפי ימים מהפנייה הראשונה",
                                width=W_HALF)
    status_rows = Counter((s.status, s.status_basis) for s in a.stories)
    items = []
    for (st, basis), n in sorted(status_rows.items(), key=lambda kv: -kv[1]):
        label = STATUS_HE.get(st, st) + (f" — {BASIS_HE[basis]}" if basis not in ("none",) else "")
        cls = {"closed": "ch-bar-ok", "open": "ch-bar-bad"}.get(st, "ch-bar-muted")
        items.append({"label": label, "value": n, "cls": cls, "filter": {"status": st}})
    out["status"] = charts.hbars(items, label="סטטוס הסיפורים ועל מה הוא מבוסס", width=W_HALF)

    pf = a.promise_funnel
    if pf.get("made"):
        out["promises"] = charts.hbars(
            [{"label": "הבטחות שניתנו", "value": pf.get("made", 0), "cls": "ch-bar"},
             {"label": "קוימו", "value": pf.get("kept", 0), "cls": "ch-bar-ok"},
             {"label": "הופרו", "value": pf.get("broken", 0), "cls": "ch-bar-bad",
              "filter": {"promise": "broken"}},
             {"label": "לא ידוע", "value": pf.get("unknown", 0), "cls": "ch-bar-muted"}],
            label="הבטחות הבנק ומה עלה בהן", total=pf.get("made", 0), width=W_HALF)
    ab = a.after_abandon
    if sum(ab.values()):
        out["after_abandon"] = charts.hbars(
            [{"label": "הבנק פעל ראשון", "value": ab.get("bank_first", 0), "cls": "ch-bar-ok"},
             {"label": "הלקוח חזר לפני הבנק", "value": ab.get("customer_first", 0),
              "cls": "ch-bar-bad"},
             {"label": "אף אחד לא פעל עד סוף הנתונים", "value": ab.get("nobody", 0),
              "cls": "ch-bar-muted"}],
            label="אחרי שיחה שננטשה: מי פעל ראשון", width=W_HALF)
    covered = [s for s in a.stories if s.coverage == "full"]
    if covered:
        bk = Counter(min(s.bankers, 5) for s in covered)
        out["bankers"] = charts.columns(
            [(str(i) if i < 5 else "5+", bk.get(i, 0)) for i in range(0, 6)],
            label="סיפורים לפי מספר הבנקאים שטיפלו", y_title="סיפורים", width=W_HALF)
        uk = Counter()
        for s in covered:
            uk.update(s.unit_kinds)
        out["unit_kinds"] = charts.hbars(
            [{"label": UNIT_KIND_HE.get(k, k), "value": v} for k, v in uk.most_common()],
            label="סיפורים שבהם פעלה כל יחידה", total=len(covered), width=W_HALF)
    return {k: (_journey_markup(v) if v is not None else None) for k, v in out.items()}


def _handoffs(a: JourneyAnalysis) -> dict | None:
    kinds = sorted({k for k in a.handoffs} | {k for v in a.handoffs.values() for k in v},
                   key=lambda k: list(UNIT_KIND_HE).index(k) if k in UNIT_KIND_HE else 99)
    if not kinds:
        return None
    rows = [{"label": UNIT_KIND_HE.get(src, src),
             "cells": [a.handoffs.get(src, {}).get(dst) for dst in kinds]} for src in kinds]
    return {"cols": [UNIT_KIND_HE.get(k, k) for k in kinds], "rows": rows}


# -- KPIs ---------------------------------------------------------------------------

def _metric_value(m) -> str:  # noqa: ANN001 - Metric
    if m.value is None or not m.shown:
        return "—"
    if m.unit == "share":
        return _pct1(m.value)
    if m.unit == "count":
        return f"{int(m.value):,}"
    return _fmt(m.value)


def _metric_help(m) -> dict:  # noqa: ANN001
    return {"definition": m.definition_he, "wrong_if": m.wrong_if_he,
            "basis": BASIS_HE.get(m.basis, m.basis),
            "base": (f"⟦{m.k:,}⟧ מתוך ⟦{m.n:,}⟧" if m.k is not None and m.n is not None else "")}


def _kpi(m, href: str, sub: str = "", unit: str = "", cls: str = "", badge: str | None = None,  # noqa: ANN001
         badge_cls: str = "muted") -> dict:
    value = _metric_value(m)
    if m.unit == "share" and m.n is not None and not m.shown:
        sub = (f"בסיס קטן מדי להצגת שיעור (⟦{m.k or 0:,}⟧ מתוך ⟦{m.n:,}⟧)" if m.n
               else "אין עדיין נתונים לחישוב")
    elif m.unit == "share" and m.k is not None and not sub:
        sub = f"⟦{m.k:,}⟧ מתוך ⟦{m.n:,}⟧"
        if m.low is not None and m.high is not None:
            sub += f" · רווח סמך ⟦95%⟧: ⟦{_pct1(max(0, m.low))}–{_pct1(min(1, m.high))}⟧"
    if m.preliminary and m.shown and m.value is not None:
        badge, badge_cls = badge or "ראשוני", "warn"
    return {"label": m.label_he, "value": value, "unit": unit, "sub": sub, "href": href,
            "cls": cls, "badge": badge, "badge_cls": badge_cls, "help": _metric_help(m)}


def _kpis(a: JourneyAnalysis) -> list[dict]:
    m = a.metrics
    out = [_kpi(m["returns_per_story"], "#sec-returns",
                sub=f"⟦{int(m['returns'].value or 0):,}⟧ חזרות ב־⟦{int(m['stories'].value or 0):,}⟧ "
                    "סיפורים")]
    fr = m["failure_rate"]
    k = _kpi(fr, "#sec-categories")
    if not fr.n:
        k["sub"] = "ממתין לסיווג לפי תוכן השיחות"
    elif fr.shown and (fr.low or 0) >= 0.15:
        k["cls"] = "risk"
    out.append(k)
    pb = m["promises_broken"]
    k = _kpi(pb, "#sec-promises")
    if not pb.n:
        k["sub"] = "ממתין לזיהוי הבטחות מתוך השיחות"
    elif pb.shown and (pb.value or 0) >= 0.3:
        k["cls"] = "warnk"
    out.append(k)
    if m["retold"].n:
        out.append(_kpi(m["retold"], "#sec-effort"))
    ab = m["abandoned"]
    out.append(_kpi(ab, "#sec-abandon", cls="warnk" if ab.shown and (ab.value or 0) >= 0.15 else ""))
    med = m.get("median_days_to_resolution")
    closed = m["closed"]
    k = _kpi(closed, "#sec-resolution")
    if med is not None and med.value is not None:
        k["sub"] += f" · חציון ⟦{_fmt(med.value)}⟧ ימים עד סגירה"
    out.append(k)
    if "bankers_per_story" in m:
        tb = m.get("three_bankers")
        sub = (f"⟦{_pct1(tb.value)}⟧ מהסיפורים עם ⟦3⟧ בנקאים ומעלה"
               if tb is not None and tb.shown and tb.value is not None else "")
        out.append(_kpi(m["bankers_per_story"], "#sec-effort", sub=sub,
                        unit="בסיפורים בכיסוי אטלס מלא"))
    return out[:6]


def _metric_table(a: JourneyAnalysis) -> list[dict]:
    rows = []
    for m in a.metrics.values():
        rows.append({"label": m.label_he, "value": _metric_value(m),
                     "base": f"{m.k:,} / {m.n:,}" if m.k is not None and m.n is not None else "",
                     "ci": (f"{_pct1(max(0, m.low))}–{_pct1(min(1, m.high))}"
                            if m.unit == "share" and m.low is not None and m.high is not None
                            and m.shown else ""),
                     "basis": BASIS_HE.get(m.basis, m.basis), "definition": m.definition_he,
                     "wrong_if": m.wrong_if_he, "prelim": m.preliminary and m.shown})
    return rows


# -- stories ------------------------------------------------------------------------

def _story_views(a: JourneyAnalysis, facts: list[StoryFacts], dataset: JourneyDataset,
                 tax: Taxonomy, units: Units, content: ContentLayer | None, *, with_text: bool,
                 call_reports: set[str]) -> tuple[list[dict], list[dict]]:
    """(cards for the template, rows for the explorer's JSON)."""
    summary = {s.story_key: s for s in a.stories}
    cards_in = content.cards if content else {}
    order = sorted(facts, key=lambda f: (-summary[f.timeline.story.story_key].returns,
                                         -summary[f.timeline.story.story_key].failures,
                                         f.timeline.story.story_no))
    drawn = {f.timeline.story.story_key for f in order[:MAX_TIMELINES]}
    cards, rows = [], []
    for f in sorted(facts, key=lambda f: f.timeline.story.story_no):
        tl = f.timeline
        s = summary[tl.story.story_key]
        anchor = f"s{s.story_no}"
        session_contact = {id(x): c.index for c in tl.contacts for x in c.sessions}
        promises_by_iid: dict[str, list] = defaultdict(list)
        for p in f.promises:
            promises_by_iid[p.made_in].append(p)
        index_of = {c.interaction.interaction_id: c.index for c in tl.contacts}
        contact_rows, tl_contacts = [], []
        for c in tl.contacts:
            iid = c.interaction.interaction_id
            j = f.judgements.get(iid)
            card = cards_in.get(iid)
            cat = shown_category(j) if j else None
            title = (f"#{c.index + 1} · {_dt(c.at)} · {KIND_HE.get(c.kind, c.kind)}"
                     + (f" · {DIRECTION_HE[c.direction]}" if DIRECTION_HE.get(c.direction) else "")
                     + (f" · {_cat_label(tax, cat)}" if cat else " · פנייה ראשונה"))
            tl_contacts.append({"index": c.index, "at": c.at, "kind": c.kind,
                                "category_cls": _cat_cls(cat), "category":
                                _cat_label(tax, cat) if cat else "פנייה ראשונה",
                                "direction": c.direction, "title": title})
            quote = ""
            if with_text and j is not None and j.quotes:
                quote = _prose(j.quotes[0].quote, MAX_QUOTE)
            elif with_text and card is not None:
                ev = card.retold_ev or card.outcome_ev or card.prior_ev
                quote = _prose(ev.quote, MAX_QUOTE) if ev else ""
            promises = [{"text": f"הבטחה {PROMISE_HE.get(p.kind, p.kind)}: "
                                 f"{OUTCOME_HE[p.outcome]}"
                                 + (f" ({SETTLED_HE.get(p.settled_by, '')})" if p.settled_by else ""),
                         "cls": {"kept": "ok", "broken": "bad"}.get(p.outcome, "muted")}
                        for p in promises_by_iid.get(iid, [])]
            sess = c.sessions
            call_id = c.interaction.call_id
            contact_rows.append({
                "id": f"{anchor}-c{c.index}", "n": c.index + 1, "at": _dt(c.at),
                "kind": KIND_HE.get(c.kind, c.kind), "dir": DIRECTION_HE.get(c.direction, ""),
                "ref": (call_id or c.interaction.correspondence_id or "")[:8],
                "category": _cat_label(tax, cat) if cat else "",
                "category_cls": _cat_cls(cat), "first": not c.is_return,
                "decided": DECIDED_HE.get(j.decided_by, "") if j else "",
                "basis": BASIS_HE.get(j.basis, "") if j else "",
                "objective": OBJECTIVE_HE.get(j.objective_class, "") if j else "",
                "inferred": tax.category_label(j.inferred_category)
                if j is not None and j.inferred_category else "",
                "reason": _prose(j.reason_he) if (j and with_text) else "",
                "break": bool(j and j.is_break_point),
                "break_text": _prose(j.break_he) if (j and j.is_break_point and with_text) else "",
                "quote": quote,
                "retold": {"yes": "סיפר מחדש", "partial": "סיפר מחדש חלקית"}.get(
                    card.retold, "") if card else "",
                "promises": promises,
                "sessions": (f"{len(sess)} סשנים באטלס · {_fmt(sum(x.minutes for x in sess))} דק'"
                             + (" · בוצעה פעולה" if any(x.has_execute for x in sess) else
                                " · צפייה בלבד") if sess else ""),
                "report": f"calls/{call_id}.html" if call_id and call_id in call_reports else "",
            })
        tl_sessions = []
        for x in tl.sessions:
            kind = units.kind(x.unit_code, tl.story.branch)
            tl_sessions.append({
                "start": x.start, "end": x.end, "unit_kind": kind, "execute": x.has_execute,
                "view_only": x.view_only, "contact_index": session_contact.get(id(x)),
                "title": (f"{units.label(x.unit_code, tl.story.branch)} · בנקאי "
                          f"{shown_banker(x.banker_code)} · {_fmt(x.minutes)} דק' · "
                          + ("ביצוע" if x.has_execute else "צפייה בלבד"))})
        tl_promises = [{"from_index": index_of.get(p.made_in), "to": p.settled_at or p.due,
                        "outcome": p.outcome} for p in f.promises]
        verdict = f.verdict
        svg = None
        if tl.story.story_key in drawn:
            svg = charts.story_timeline(tl_contacts, tl_sessions, label=_story_label(s.story_no),
                                        anchor=anchor, promises=tl_promises)
        cat_counts = Counter(shown_category(j) for j in f.judgements.values())
        cards.append({
            "anchor": anchor, "no": s.story_no, "label": _story_label(s.story_no),
            "branch": s.branch or "—", "topic": tax.topic_label(s.topic),
            "status": STATUS_HE.get(s.status, s.status), "status_key": s.status,
            "status_basis": BASIS_HE.get(s.status_basis, ""),
            "status_note": verdict.status_note_he if verdict else "",
            "model_status": STATUS_HE.get(verdict.model_status or "", "") if verdict else "",
            "headline": _prose(verdict.headline_he) if (verdict and with_text) else "",
            "narrative": _prose(verdict.narrative_he, 1200) if (verdict and with_text) else "",
            "coverage": COVERAGE_HE.get(s.coverage, s.coverage),
            "first": _dt(s.first_at), "last": _dt(s.last_at), "days": _fmt(s.span_days),
            "contacts": s.contacts, "returns": s.returns, "bankers": s.bankers,
            "minutes": _fmt(s.banker_minutes), "sessions": s.sessions,
            "cats": [{"label": _cat_label(tax, c), "n": n, "cls": _cat_cls(c)}
                     for c in _categories(tax, True) if (n := cat_counts.get(c))],
            "rows": contact_rows, "svg": svg,
        })
        rows.append({"no": s.story_no, "a": anchor, "t": s.topic, "tl": tax.topic_label(s.topic),
                     "br": s.branch or "", "st": s.status, "sb": s.status_basis,
                     "c": s.contacts, "r": s.returns, "f": s.failures, "rt": s.retold,
                     "p": s.promises, "pb": s.promises_broken, "ab": s.abandoned,
                     "sd": s.max_same_day, "bk": s.bankers, "mn": round(s.banker_minutes, 1),
                     "cv": s.coverage, "d": round(s.span_days, 1),
                     "fa": s.first_at.date().isoformat(),
                     "cats": sorted({shown_category(j) for j in f.judgements.values()})})
    return cards, rows


# -- page ---------------------------------------------------------------------------

def _footer(dataset: JourneyDataset, content: ContentLayer | None, generated: datetime) -> str:
    parts = [f"callqa {__version__}", f"הופק {generated.strftime('%d.%m.%Y %H:%M')}",
             f"מערך נתונים {dataset.dataset_id}"]
    parts.append(f"שכבת תוכן: {content.engine}" if content else "שכבת תוכן: לא הורצה")
    return " · ".join(parts)


def render_html(dataset: JourneyDataset, a: JourneyAnalysis, facts: list[StoryFacts],
                opinion: Opinion, tax: Taxonomy, units: Units, content: ContentLayer | None, *,
                with_text: bool = True, title: str | None = None,
                call_reports: set[str] | None = None, generated_at: datetime | None = None) -> str:
    generated = (generated_at or datetime.now(UTC)).astimezone()
    env = jinja_env()
    env.filters["rich"] = _rich
    env.filters["fjson"] = _fjson
    demo_key = None
    if content is not None and content.engine in DEMO_TEXT:
        demo_key = content.engine
    elif dataset.source.startswith("synthetic"):
        demo_key = "synthetic-demo"
    cards, rows = _story_views(a, facts, dataset, tax, units, content, with_text=with_text,
                               call_reports=call_reports or set())
    sev = opinion.top_findings[0].severity if opinion.top_findings else "info"
    cats = _categories(tax, _has_pending(a))
    context = {
        "version": __version__,
        "title": title or "דוח מסעות לקוח — פניות חוזרות",
        "generated_at": generated.strftime("%d.%m.%Y %H:%M"),
        "period": f"⟦{a.data_start.strftime('%d.%m.%Y')}⟧ – ⟦{a.data_end.strftime('%d.%m.%Y')}⟧",
        "a": a, "m": a.metrics, "opinion": opinion, "severity_he": SEVERITY_HE,
        "verdict_class": {"critical": "sev-critical", "high": "sev-high"}.get(sev, ""),
        "kpis": _kpis(a), "charts": _charts(a, facts, tax), "handoffs": _handoffs(a),
        "metric_table": _metric_table(a),
        "cards": cards, "n_drawn": sum(1 for c in cards if c.get("svg")),
        "max_timelines": MAX_TIMELINES,
        "legend": charts.timeline_legend([(_cat_cls(c), _cat_label(tax, c)) for c in cats]
                                         + [("ch-cat-first", "פנייה ראשונה")]),
        "topics": a.topics, "fmt": _fmt, "pct1": _pct1,
        "categories": [{"id": c, "label": _cat_label(tax, c),
                        "definition": tax.categories.get(c, {}).get(
                            "definition", "חזרה עם הקלטה או התכתבות שעוד לא נקראה בשלב התוכן."),
                        "failure": bool(tax.categories.get(c, {}).get("failure"))} for c in cats],
        "topic_options": [(k, tax.topic_label(k)) for k in tax.topics],
        "status_he": STATUS_HE,
        "with_text": with_text, "content": content,
        "demo": demo_key is not None,
        "demo_title": DEMO_TEXT.get(demo_key, ("", ""))[0] if demo_key else "",
        "demo_text": DEMO_TEXT.get(demo_key, ("", ""))[1] if demo_key else "",
        "import_counts": dataset.report.counts,
        "import_issues": [i for i in dataset.report.issues if i.severity != "info"],
        "coverage_he": COVERAGE_HE,
        "empty": Markup('<p class="ch-empty">אין די נתונים.</p>'),
        "footer": _footer(dataset, content, generated),
        "data_json": Markup(script_json({"stories": rows, "cats": {c: _cat_label(tax, c)
                                                                   for c in cats}})),
    }
    return env.get_template("journey_report.html.j2").render(**context)


# -- exports -------------------------------------------------------------------------

def render_stories_csv(a: JourneyAnalysis, tax: Taxonomy) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\r\n")
    w.writerow(["story_no", "branch", "topic", "first_at", "last_at", "span_days", "contacts",
                "returns", "abandoned", "failures", "content_returns", "retold", "promises",
                "promises_broken", "status", "status_basis", "atlas_coverage", "bankers",
                "units", "banker_minutes", "background_minutes"])
    for s in sorted(a.stories, key=lambda s: s.story_no):
        w.writerow([_csv_cell(v) for v in (
            s.story_no, s.branch or "", tax.topic_label(s.topic), s.first_at.isoformat(" "),
            s.last_at.isoformat(" "), round(s.span_days, 2), s.contacts, s.returns, s.abandoned,
            s.failures, s.content_returns, s.retold, s.promises, s.promises_broken, s.status,
            s.status_basis, s.coverage, s.bankers, s.units, s.banker_minutes,
            s.background_minutes)])
    return "﻿" + out.getvalue()


def render_returns_csv(facts: list[StoryFacts]) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\r\n")
    w.writerow(["story_no", "contact_no", "at", "kind", "direction", "objective_class",
                "category", "inferred_category", "basis", "decided_by", "break_point"])
    for f in sorted(facts, key=lambda f: f.timeline.story.story_no):
        for c in f.timeline.returns:
            j = f.judgements.get(c.interaction.interaction_id)
            w.writerow([_csv_cell(v) for v in (
                f.timeline.story.story_no, c.index + 1, c.at.isoformat(" "), c.kind, c.direction,
                j.objective_class if j else "", j.category if j else "",
                (j.inferred_category or "") if j else "", j.basis if j else "",
                j.decided_by if j else "", "yes" if j and j.is_break_point else "")])
    return "﻿" + out.getvalue()


def render_json(a: JourneyAnalysis, opinion: Opinion) -> str:
    data = a.model_dump(mode="json", exclude={"stories"})
    data["version"] = __version__
    data["findings"] = [{"key": f.key, "severity": f.severity} for f in opinion.findings]
    return json.dumps(data, ensure_ascii=False, indent=2)


# -- entry point ----------------------------------------------------------------------

def report_name(name: str | None) -> str:
    if not name:
        return DEFAULT_NAME
    if not _NAME_RE.match(name):
        raise ValueError("report name: letters, digits, dot, underscore, hyphen only")
    stem = name if name == DEFAULT_NAME or name.startswith(DEFAULT_NAME + "-") \
        else f"{DEFAULT_NAME}-{name}"
    if not JOURNEY_REPORT_FILE_RE.fullmatch(stem + ".html"):
        raise ValueError("report name is too long")
    return stem


def settings_of(config: Config) -> RuleSettings:
    return RuleSettings(callback_business_days=config.journey.callback_business_days,
                        quiet_days=config.journey.quiet_days)


def build_journey_report(config: Config, dataset_id: str | None = None, *,
                         with_text: bool = True, name: str | None = None,
                         title: str | None = None) -> JourneyReport:
    stem = report_name(name)
    ds_id = resolve_dataset_id(config, dataset_id)
    dataset = load_dataset(config, ds_id)
    content = load_content(config, ds_id)
    tax = load_taxonomy(config.journey.taxonomy)
    units = load_units(config.journey.units)
    analysis, facts = analyse(
        dataset, taxonomy=tax, units=units,
        cards=content.cards if content else None,
        judgements=content.judgements if content else None,
        verdicts=content.verdicts if content else None,
        settings=settings_of(config), min_rate_n=config.journey.min_rate_n,
        min_firm_n=config.journey.min_firm_n)
    opinion = build_opinion(analysis, tax)
    reports = config.paths.output_dir / "reports"
    calls_dir = reports / "calls"
    call_reports = {p.stem for p in calls_dir.glob("*.html")} if calls_dir.is_dir() else set()
    html = render_html(dataset, analysis, facts, opinion, tax, units, content,
                       with_text=with_text, title=title, call_reports=call_reports)
    paths = JourneyReport(html=reports / f"{stem}.html",
                          stories_csv=reports / f"{stem}_stories.csv",
                          returns_csv=reports / f"{stem}_returns.csv",
                          json=reports / f"{stem}.json", analysis=analysis, opinion=opinion)
    atomic_write_text(paths.html, html)
    for path, text in ((paths.stories_csv, render_stories_csv(analysis, tax)),
                       (paths.returns_csv, render_returns_csv(facts)),
                       (paths.json, render_json(analysis, opinion))):
        try:
            atomic_write_text(path, text)
        except PermissionError:
            logger.warning("could not update %s: it is open in another program (Excel?)",
                           path.name)
    # the analysis beside the dataset, for audit and for the evaluation tools
    atomic_write_text(dataset_dir(config, ds_id) / "analysis.json",
                      analysis.model_dump_json(indent=2))
    logger.info("journey report: %d stories, %d contacts -> %s", len(analysis.stories),
                sum(s.contacts for s in analysis.stories), paths.html.name)
    return paths
