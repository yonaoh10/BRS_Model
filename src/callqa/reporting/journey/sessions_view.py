"""The journey report's two analysis levels, as the page shows them.

Single contact ("שיחה בודדת"): every call and correspondence with what stood
behind it in Atlas - the table of contact types (ATL_R03 chapter 4), an
explorer with one row per contact, and in each story card the sessions each
contact had. A page of its own for one contact (`--contact STORY:N`) shows
its three layers the way ATL_R04 lays them out: the contact table, what was
said, and Atlas.

Banker session ("סשן"): the chapter the bank's Atlas report is built of
(ATL_R03): the definitions, the page of fifteen numbers with what each one
means, the tables by status, topic and unit, three stories as a table of
contacts and sessions, and the code classification; and an explorer with one
row per session.
"""

from __future__ import annotations

import re
from datetime import datetime

from markupsafe import Markup

from callqa.journey.models import AtlasRules
from callqa.journey.session_analysis import (
    FIRST_MOVE_HE,
    INT_TYPE_HE,
    ContactRow,
    SessionAnalysis,
    SessionRow,
)
from callqa.journey.sessions import OP_CATEGORY_HE, SESSION_KIND_HE, UNIT_CLASS_HE
from callqa.reporting.executive.render import _rich
from callqa.reporting.journey import charts

STATUS_HE = {"closed": "נסגר", "open": "פתוח", "unclear": "לא ברור"}
DIRECTION_HE = {"inbound": "נכנסת", "outbound": "יוצאת", "unknown": ""}
KIND_CLS = {"execute": "ch-bar-ok", "info": "ch-bar", "unclassified": "ch-bar-muted",
            "open": "ch-bar-muted", "not_customer": "ch-bar-muted"}
# rows of the explorers carried in the page; beyond, the CSV has them all
MAX_ROWS = 20000
_NUM_RE = re.compile(r"(-?\d[\d,.]*%?)")


def _f(v: float | None, d: int = 1) -> str:
    """Fixed decimals, as the bank's Atlas report prints them (1.0, not 1)."""
    return f"{v:,.{d}f}" if v is not None else "—"


def _isolated(text: str) -> Markup:
    """A value like '54 / 10 / 51 מתוך 72' read right to left: every number
    isolated, so the order on the page is the order of the words."""
    return _rich(_NUM_RE.sub(r"⟦\1⟧", text))


def _pct(v: float | None) -> str:
    return f"{v:.1f}%" if v is not None else "—"


def _hm(t: datetime) -> str:
    return t.strftime("%d.%m.%Y %H:%M")


def _hours(h: float | None) -> str:
    if h is None:
        return "—"
    if h < 1:
        return f"{h * 60:.0f} דק'"
    if h < 48:
        return f"{h:.1f} שע'"
    return f"{h / 24:.1f} ימים"


def unit_text(s: SessionRow) -> str:
    label = UNIT_CLASS_HE[s.unit_class]
    if s.unit_class in ("own", "other"):
        return f"{label} · {s.unit_label or 'יחידה ' + s.unit_code}"
    return label


def definitions(rules: AtlasRules) -> list[tuple[str, str]]:
    """The method and definitions, in the bank Atlas report's own words (ATL_R03
    chapter 2), with the windows and thresholds this dataset was built with."""
    cover = rules.coverage_from.strftime("%d.%m.%Y") if rules.coverage_from else "—"
    return [
        ("סשן", f"רצף פעולות של בנקאי אחד על החשבון באטלס. סשן חדש מתחיל כשמתחלף בנקאי, אחרי "
                f"הפסקה של יותר מ־{rules.gap_min:g} דקות, או כשהבנקאי פותח מחדש את מסך הלקוח "
                f"(קוד {rules.start_op}). זמן הסשן נמדד מהפעולה הראשונה עד האחרונה, ולכן הוא חסם "
                "תחתון לזמן העבודה. בנקאי מופיע בקוד רץ בלבד."),
        ("סוג הסשן", "לפי הפעולה החזקה בו: ביצוע או שינוי (העברה, הזמנת כרטיס, הפקדה, פתיחת "
                     "חשבון), אחריו מידע ושאילתות (תנועות, פרטי חשבון, אובליגו), אחריו פעולה שלא "
                     "סווגה. סשן של פתיחת מסך בלבד הוא הצצה. הרשימה המלאה בנספח."),
        ("הצמדה לפנייה", f"סשן נחשב מאחורי פנייה אם התחיל בחלון הזמן שלה: שיחה יוצאת - מ־"
                         f"{rules.call_out_before:g} דקות לפניה ועד סופה; שיחה נכנסת - מתחילתה "
                         f"ועד {rules.call_in_after:g} דקות אחרי סופה; הודעה יוצאת - "
                         f"{rules.msg_out_before:g} דקות לפניה; הודעה נכנסת - עד "
                         f"{rules.msg_in_after:g} דקות אחריה; כיוון לא ידוע - "
                         f"{rules.unk_before:g} דקות לפני עד {rules.unk_after:g} אחרי. כל "
                         "הודעה בהתכתבות היא נקודה בפני עצמה. סשן בכמה חלונות מוצמד לקרוב "
                         "ביותר. סשן שאינו בחלון של אף פנייה נספר כעבודה בלי פנייה מתועדת."),
        ("יחידות", "לפי טבלת היחידות: מרכז הבנקאות, תפעול עורפי, סניף החשבון, או סניף ויחידה "
                   "אחרים. מעבר בין יחידות נספר כשסוג היחידה משתנה בין שני סשנים רצופים."),
        ("כיסוי", f"היום הראשון בלוג: {cover}. הלוג שומר כשלושה חודשים אחורה, ולכן כל המדדים "
                  "נמדדים רק על סיפורים שהתחילו מהיום הזה ואילך - כדי שחשבון בלי נתונים לא "
                  "ייספר כחשבון בלי פעילות."),
    ]


# ------------------------------------------------------------------ chapter


def chapter(sa: SessionAnalysis, rules: AtlasRules, story_anchor) -> dict:  # noqa: ANN001
    """Everything the session chapter of level 2 shows."""
    kinds = charts.hbars(
        [{"label": SESSION_KIND_HE[k], "value": v, "cls": KIND_CLS[k]}
         for k, v in sa.kinds.items() if v],
        label="סשנים בסיפורים בכיסוי מלא, לפי סוג", total=sum(sa.kinds.values()), width=540)
    classes = charts.hbars(
        [{"label": UNIT_CLASS_HE[k], "value": v} for k, v in sa.classes.items() if v],
        label="סשנים בסיפורים בכיסוי מלא, לפי יחידה", total=sum(sa.classes.values()), width=540)
    cases = []
    for c in sa.cases:
        cases.append({"no": c["story_no"], "why": c["why"], "anchor": story_anchor(c["story_no"]),
                      "rows": [{"at": _hm(r["at"]), "event": r["event"], "what": r["what"],
                                "unit": r["unit"], "banker": r["banker"],
                                "minutes": _f(r["minutes"]) if r["minutes"] is not None else "",
                                "ops": r["ops"], "atlas": r["event"] == "אטלס"}
                               for r in c["rows"]]})
    return {
        "definitions": definitions(rules),
        "head": [{"item": h.item, "finding": h.finding_he, "value": _isolated(h.value_txt),
                  "explain": h.explain_he, "base": h.base_he, "n": h.n,
                  "prelim": h.preliminary} for h in sa.head],
        "kinds": Markup(kinds) if kinds else None,
        "classes": Markup(classes) if classes else None,
        "by_type": [{**r, "f_act": _pct(r["pct_activity"]), "f_do": _pct(r["pct_execute"]),
                     "f_br": _pct(r["pct_branch"]), "f_ct": _pct(r["pct_center"]),
                     "f_bank": _f(r["avg_bankers"])} for r in sa.by_type],
        "by_status": [{**r, "label": STATUS_HE.get(r["key"], r["key"]),
                       "f_bank": _f(r["avg_bankers"]), "f_3b": _pct(r["pct_3_bankers"]),
                       "f_hoff": _f(r["avg_handoffs"]), "f_cross": _pct(r["pct_cross"]),
                       "f_min": _f(r["avg_minutes"]), "f_bg": _f(r["avg_background"]),
                       "f_do": _pct(r["pct_execute"]), "f_after": _pct(r["pct_execute_after"]),
                       "f_hrs": _f(r["median_hours_to_execute"])} for r in sa.by_status],
        "by_topic": [{**r, "f_bank": _f(r["avg_bankers"]), "f_hoff": _f(r["avg_handoffs"]),
                      "f_min": _f(r["avg_minutes"]), "f_bg": _f(r["avg_background"]),
                      "f_after": _pct(r["pct_execute_after"])} for r in sa.by_topic],
        "units": [{**r, "f_link": _pct(r["pct_linked"]), "f_min": _f(r["minutes"])}
                  for r in sa.units],
        "codecat": sa.codecat,
        "checks": sa.checks,
        "cases": cases,
        "n_covered": sa.n_covered, "n_stories": sa.n_stories, "n_active": sa.n_covered_active,
        "after": [{"label": FIRST_MOVE_HE[k], "n": v} for k, v in sa.after_abandon.items()],
        "small_status": any(r["n"] < 30 for r in sa.by_status),
    }


# ------------------------------------------------------------------ explorers


def contact_rows(sa: SessionAnalysis, category_of: dict[str, tuple[str, str]],
                 anchor_of) -> list[dict]:  # noqa: ANN001
    """One row per contact for the explorer (short keys: it is carried as JSON)."""
    out = []
    for c in sa.contacts[:MAX_ROWS]:
        cat = category_of.get(c.interaction_id, ("", ""))
        units = [x for x, flag in (("מרכז", c.has_center), ("סניף", c.has_branch)) if flag]
        out.append({"s": c.story_no, "n": c.n, "a": anchor_of(c.story_no),
                    "at": c.at.strftime("%Y-%m-%d %H:%M"), "t": c.int_type,
                    "tl": INT_TYPE_HE[c.int_type], "d": DIRECTION_HE.get(c.direction, ""),
                    "c": cat[0], "cl": cat[1], "ns": c.n_sess, "nb": c.n_bankers,
                    "mn": round(c.banker_min, 1), "x": 1 if c.has_execute else 0,
                    "br": 1 if c.has_branch else 0, "ct": 1 if c.has_center else 0,
                    "u": " · ".join(units), "rh": (round(c.resp_hours, 2)
                                                   if c.resp_hours is not None else None),
                    "sg": c.segments, "cv": 1 if c.covered else 0,
                    "fm": FIRST_MOVE_HE.get(c.first_move or "", "")})
    return out


def session_rows(sa: SessionAnalysis, anchor_of) -> list[dict]:  # noqa: ANN001
    out = []
    for s in sa.sessions[:MAX_ROWS]:
        out.append({"s": s.story_no, "n": s.no, "a": anchor_of(s.story_no),
                    "at": s.start.strftime("%Y-%m-%d %H:%M"), "mn": round(s.minutes, 1),
                    "b": s.banker, "k": s.kind, "kl": SESSION_KIND_HE[s.kind],
                    "uc": s.unit_class, "ul": unit_text(s), "no": s.n_ops,
                    "cn": s.contact_n, "ops": s.ops_txt, "al": 1 if s.after_last_contact else 0,
                    "cv": 1 if s.covered else 0, "pk": 1 if s.peek else 0})
    return out


# ------------------------------------------------------------------ story cards


def story_sessions(sa_sessions: list[SessionRow], anchor: str) -> list[dict]:
    return [{"id": f"{anchor}-x{s.no}", "no": s.no, "at": _hm(s.start), "minutes": _f(s.minutes),
             "banker": s.banker, "unit": unit_text(s), "kind": SESSION_KIND_HE[s.kind],
             "kind_key": s.kind, "contact": s.contact_n, "ops": s.ops_txt, "peek": s.peek,
             "after": s.after_last_contact} for s in sa_sessions]


def contact_atlas(row: ContactRow | None, sessions: list[dict]) -> dict | None:
    """The Atlas cell of a contact in its story card."""
    if row is None:
        return None
    mine = [s for s in sessions if s["contact"] == row.n]
    strongest = next((SESSION_KIND_HE[k] for k in ("execute", "info", "unclassified", "open",
                                                   "not_customer")
                      if any(s["kind_key"] == k for s in mine)), "")
    return {"n": row.n_sess, "bankers": row.n_bankers, "minutes": _f(row.banker_min),
            "strongest": strongest, "sessions": mine,
            "resp": _hours(row.resp_hours) if row.resp_hours is not None else "",
            "first_move": FIRST_MOVE_HE.get(row.first_move or "", "")}


# ------------------------------------------------------------------ one contact


def contact_page(sa: SessionAnalysis, dataset, contact: ContactRow, *,  # noqa: ANN001
                 ops_of: dict[str, list]) -> dict:
    """The Atlas layer of the single-contact page, and the story around it."""
    story_contacts = [c for c in sa.contacts if c.story_key == contact.story_key]
    sessions = [s for s in sa.sessions if s.story_key == contact.story_key]
    prev = next((c for c in reversed(story_contacts) if c.n < contact.n), None)
    since_prev = [s for s in sessions if (prev is None or s.start > prev.at)
                  and s.start < contact.at]
    mine = [s for s in sessions if s.contact_n == contact.n]
    return {
        "layer": {"type": INT_TYPE_HE[contact.int_type],
                  "direction": DIRECTION_HE.get(contact.direction, "") or "לא ידוע",
                  "talk": (_f(contact.talk_seconds / 60) + " דק'"
                           if contact.talk_seconds else "—"),
                  "segments": contact.segments, "n_sess": contact.n_sess,
                  "bankers": contact.n_bankers, "minutes": _f(contact.banker_min),
                  "ops": contact.n_ops, "resp": _hours(contact.resp_hours),
                  "first_move": FIRST_MOVE_HE.get(contact.first_move or "", ""),
                  "covered": contact.covered},
        "sessions": [{"at": _hm(s.start), "minutes": _f(s.minutes), "banker": s.banker,
                      "unit": unit_text(s), "kind": SESSION_KIND_HE[s.kind],
                      "ops": [{"at": op.at.strftime("%H:%M:%S"), "code": op.op_code,
                               "category": OP_CATEGORY_HE.get(op.op_category, ""),
                               "description": op.description} for op in ops_of.get(s.session_id, [])]}
                     for s in mine],
        # ATL_R04's failure test: an execution between the previous contact and this one
        "since_prev": {"sessions": len(since_prev),
                       "execute": sum(1 for s in since_prev if s.kind == "execute"),
                       "prev_at": _hm(prev.at) if prev else ""},
        "timeline": [{"n": c.n, "at": _hm(c.at), "type": INT_TYPE_HE[c.int_type],
                      "dir": DIRECTION_HE.get(c.direction, ""), "sessions": c.n_sess,
                      "me": c.n == contact.n} for c in story_contacts],
    }
