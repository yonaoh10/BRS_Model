"""The two levels at which the bank's Atlas project reads a batch.

A single contact (ATLR_INT): every call and correspondence, and what the
bankers did behind it in Atlas - how many sessions, bankers, minutes and
operations, whether anything was executed, and where (a branch, the banking
centre). An incoming or abandoned contact also has the time until a banker
next opened the account.

The banker session (ATLR_SESS2 / ATLR_STORY2): every session with its kind,
unit class and the contact it served - or none - and per story the bankers,
the handoffs between unit classes, the work with no documented contact, and
whether anything was executed after the last contact. The batch figures are
the Atlas project's own (ATL_R02 blocks 4-6, the "page of numbers" of
ATL_R03): measured on stories in full log coverage only, so an account with
no data is never counted as an account with no activity.

Every formula here is the one in ATL_R02; where it names a base (covered
stories, covered stories with at least one session, all recorded calls), the
same base is used.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

from callqa.journey.models import JourneyDataset
from callqa.journey.sessions import (
    OP_CATEGORY_HE,
    SESSION_KIND_HE,
    UNIT_CLASS_HE,
    UNIT_CLASSES,
    unit_class,
)
from callqa.journey.timeline import Contact, StoryTimeline
from callqa.journey.vocab import Units

# ATL_R02 INT_TYPE, from the kind of contact
INT_TYPE = {"recorded_call": "REC", "abandoned": "ABN", "unrecorded_answered": "TALK",
            "unrecorded_unknown": "UNK", "message": "UM"}
INT_TYPE_ORDER = ("REC", "ABN", "TALK", "UNK", "UM", "OTHER")
INT_TYPE_HE = {"REC": "שיחות מוקלטות", "ABN": "שיחות שננטשו בהמתנה",
               "TALK": "שיחות שנענו בלי הקלטה", "UNK": "שיחות בלי נתון מענה",
               "UM": "התכתבות כתובה", "OTHER": "פנייה אחרת"}
FIRST_MOVE_HE = {"bank": "הבנק פעל ראשון", "customer": "הלקוח פנה שוב ראשון",
                 "none": "אף אחד"}
NO_EXECUTE = ("info", "open", "unclassified")
MAX_OPS_TXT = 3


def int_type(c: Contact) -> str:
    return INT_TYPE.get(c.kind, "OTHER")


class ContactRow(BaseModel):
    """One contact and what stood behind it in Atlas (a single call)."""

    story_key: str
    story_no: int
    n: int                          # 1 = the story's first contact
    interaction_id: str
    at: datetime
    int_type: str
    direction: str
    talk_seconds: float | None = None
    segments: int = 0               # recorded files of the call
    covered: bool = False
    n_sess: int = 0
    banker_min: float = 0.0
    n_ops: int = 0
    n_bankers: int = 0
    has_execute: bool = False
    has_branch: bool = False        # a session of the account's branch or another unit
    has_center: bool = False
    resp_hours: float | None = None  # inbound / abandoned: until the next session began
    first_move: str | None = None   # abandoned: bank / customer / none
    session_ids: list[str] = Field(default_factory=list)
    call_id: str | None = None


class SessionRow(BaseModel):
    """One banker session (ATLR_SESS2)."""

    story_key: str
    story_no: int
    session_id: str
    no: int                         # its place in the story
    start: datetime
    end: datetime
    minutes: float
    banker: str
    unit_code: str
    unit_label: str
    unit_class: str
    kind: str
    peek: bool
    n_ops: int
    ops_by_category: dict[str, int] = Field(default_factory=dict)
    ops_txt: str = ""               # up to three distinct operations, not the screen opening
    contact_n: int | None = None    # the contact it served; None = no documented contact
    after_last_contact: bool = False
    covered: bool = False


class StorySessions(BaseModel):
    """ATLR_STORY2: one story's sessions, summed."""

    story_key: str
    story_no: int
    covered: bool
    status: str
    topic: str
    n_sess: int = 0
    n_bankers: int = 0
    n_uclass: int = 0
    n_handoff: int = 0
    n_do_sess: int = 0
    n_nodo_sess: int = 0
    n_by_class: dict[str, int] = Field(default_factory=dict)
    banker_min: float = 0.0
    bg_min: float = 0.0
    n_bg: int = 0
    n_bg_br: int = 0
    first_do: datetime | None = None
    n_do_after: int = 0
    do_after: bool = False
    x_cross: bool = False
    hrs_to_do: float | None = None


class HeadItem(BaseModel):
    """A line of the page of numbers: the finding, its value, what it means."""

    item: int
    key: str
    finding_he: str
    value_txt: str
    explain_he: str
    base_he: str = ""
    n: int = 0
    preliminary: bool = False


class SessionAnalysis(BaseModel):
    coverage_from: datetime | None = None
    sessions_from: str = "none"
    n_stories: int = 0
    n_covered: int = 0
    n_covered_active: int = 0
    contacts: list[ContactRow] = Field(default_factory=list)
    sessions: list[SessionRow] = Field(default_factory=list)
    stories: list[StorySessions] = Field(default_factory=list)
    head: list[HeadItem] = Field(default_factory=list)
    by_type: list[dict] = Field(default_factory=list)
    by_status: list[dict] = Field(default_factory=list)
    by_topic: list[dict] = Field(default_factory=list)
    units: list[dict] = Field(default_factory=list)
    codecat: list[dict] = Field(default_factory=list)
    kinds: dict[str, int] = Field(default_factory=dict)          # covered sessions by kind
    classes: dict[str, int] = Field(default_factory=dict)        # covered sessions by unit class
    after_abandon: dict[str, int] = Field(default_factory=dict)
    cases: list[dict] = Field(default_factory=list)
    checks: list[dict] = Field(default_factory=list)


# ------------------------------------------------------------------ per story


def classes_of(tl: StoryTimeline, units: Units) -> list[str]:
    return [unit_class(units, s.unit_code, tl.story.branch) for s in tl.sessions]


def story_sessions(tl: StoryTimeline, units: Units, *, status: str = "unclear",
                   topic: str = "other") -> StorySessions:
    """ATL_R02 B4 for one story. Handoffs count every change of unit class
    between consecutive sessions; a story crosses when the banking centre and
    a branch (the account's own or another) both worked on it."""
    sessions = tl.sessions
    classes = classes_of(tl, units)
    linked = {id(s) for c in tl.contacts for s in c.sessions}
    first_int = tl.contacts[0].at if tl.contacts else tl.story.first_at
    last_int = tl.contacts[-1].at if tl.contacts else tl.story.last_at
    by_class = Counter(classes)
    do = [s for s in sessions if s.kind == "execute"]
    first_do = min((s.start for s in do), default=None)
    out = StorySessions(
        story_key=tl.story.story_key, story_no=tl.story.story_no,
        covered=tl.story.atlas_coverage == "full", status=status, topic=topic,
        n_sess=len(sessions), n_bankers=len({s.banker_code for s in sessions}),
        n_uclass=len(by_class),
        n_handoff=sum(1 for a, b in zip(classes, classes[1:], strict=False) if a != b),
        n_do_sess=len(do), n_nodo_sess=sum(1 for s in sessions if s.kind in NO_EXECUTE),
        n_by_class={k: by_class.get(k, 0) for k in UNIT_CLASSES},
        banker_min=sum(s.minutes for s in sessions),
        bg_min=sum(s.minutes for s in sessions if id(s) not in linked),
        n_bg=sum(1 for s in sessions if id(s) not in linked),
        n_bg_br=sum(1 for s, k in zip(sessions, classes, strict=True)
                    if id(s) not in linked and k in ("own", "other")),
        first_do=first_do,
        n_do_after=sum(1 for s in do if s.start > last_int))
    out.do_after = out.n_do_after > 0
    out.x_cross = by_class.get("center", 0) > 0 and (by_class.get("own", 0)
                                                     + by_class.get("other", 0)) > 0
    if first_do is not None:
        out.hrs_to_do = (first_do - first_int).total_seconds() / 3600
    return out


def first_move_after(tl: StoryTimeline, c: Contact) -> tuple[str, float | None]:
    """ATL_R02 B5, after an abandoned call: the bank acted first when a banker
    opened the account before the next contact of any kind (or with none
    after); the customer first when the next contact came before that; else
    nobody. Returns the hours until the bank acted when it did."""
    next_s = min((s.start for s in tl.sessions if s.start > c.at), default=None)
    next_i = min((x.at for x in tl.contacts if x.at > c.at), default=None)
    if next_s is not None and (next_i is None or next_s < next_i):
        return "bank", (next_s - c.at).total_seconds() / 3600
    if next_i is not None:
        return "customer", None
    return "none", None


# ------------------------------------------------------------------ helpers


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _pct(k: float, n: float) -> float | None:
    return 100.0 * k / n if n else None


def _f(v: float | None, digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:,.{digits}f}"


def _share_txt(k: int, n: int, min_n: int) -> str:
    """A share as a percentage once the base is big enough; below, the count."""
    if not n:
        return "—"
    if n < min_n:
        return f"{k:,} מתוך {n:,}"
    return f"{100.0 * k / n:.1f}%"


def _ops_txt(ops) -> str:
    """Up to three distinct operations, not the screen opening (ATL_R02 B6)."""
    out: list[str] = []
    for op in ops:
        if op.op_category == "open" or not op.description or len(out) >= MAX_OPS_TXT:
            continue
        if any(op.description.strip() in x for x in out):
            continue
        out.append(op.description.strip())
    return " | ".join(out)


# ------------------------------------------------------------------ batch


def analyse_sessions(dataset: JourneyDataset, timelines: list[StoryTimeline], units: Units, *,
                     status_of: dict[str, str] | None = None,
                     topic_of: dict[str, str] | None = None,
                     topic_label=lambda k: k, min_rate_n: int = 10, min_firm_n: int = 30,
                     top_units: int = 15, cases: list[int] | None = None) -> SessionAnalysis:
    status_of = status_of or {}
    topic_of = topic_of or {}
    rules = dataset.atlas_rules
    out = SessionAnalysis(coverage_from=rules.coverage_from, sessions_from=rules.sessions_from,
                          n_stories=len(timelines))
    unit_names = {code: label for code, (label, _k) in units.by_code.items()}

    for tl in timelines:
        key = tl.story.story_key
        covered = tl.story.atlas_coverage == "full"
        ss = story_sessions(tl, units, status=status_of.get(key, "unclear"),
                            topic=topic_of.get(key, "other"))
        out.stories.append(ss)
        classes = classes_of(tl, units)
        contact_of = {id(s): c.index + 1 for c in tl.contacts for s in c.sessions}
        last_int = tl.contacts[-1].at if tl.contacts else tl.story.last_at
        for n, (s, k) in enumerate(zip(tl.sessions, classes, strict=True), start=1):
            code = (s.unit_code or "").lstrip("0")
            out.sessions.append(SessionRow(
                story_key=key, story_no=tl.story.story_no, session_id=s.session_id, no=n,
                start=s.start, end=s.end, minutes=round(s.minutes, 2), banker=s.banker_code,
                unit_code=code or "?", unit_label=unit_names.get(code, ""), unit_class=k,
                kind=s.kind, peek=s.peek, n_ops=s.n_ops or len(s.ops),
                ops_by_category=dict(Counter(op.op_category for op in s.ops)),
                ops_txt=_ops_txt(s.ops), contact_n=contact_of.get(id(s)),
                after_last_contact=s.start > last_int, covered=covered))
        class_of = {id(s): k for s, k in zip(tl.sessions, classes, strict=True)}
        for c in tl.contacts:
            i = c.interaction
            row = ContactRow(
                story_key=key, story_no=tl.story.story_no, n=c.index + 1,
                interaction_id=i.interaction_id, at=c.at, int_type=int_type(c),
                direction=c.direction, talk_seconds=i.talk_seconds, covered=covered,
                n_sess=len(c.sessions), banker_min=round(sum(s.minutes for s in c.sessions), 2),
                n_ops=sum(s.n_ops or len(s.ops) for s in c.sessions),
                n_bankers=len({s.banker_code for s in c.sessions}),
                has_execute=any(s.kind == "execute" for s in c.sessions),
                has_branch=any(class_of[id(s)] in ("own", "other") for s in c.sessions),
                has_center=any(class_of[id(s)] == "center" for s in c.sessions),
                session_ids=[s.session_id for s in c.sessions], call_id=i.call_id)
            if i.recorded and i.call_key and i.call_key in dataset.calls:
                row.segments = len(dataset.calls[i.call_key].segments)
            if c.direction == "inbound" or c.kind == "abandoned":
                nxt = min((s.start for s in tl.sessions if s.start >= c.at), default=None)
                if nxt is not None:
                    row.resp_hours = (nxt - c.at).total_seconds() / 3600
            if c.kind == "abandoned":
                row.first_move = first_move_after(tl, c)[0]
            out.contacts.append(row)

    cov = [s for s in out.stories if s.covered]
    act = [s for s in cov if s.n_sess > 0]
    out.n_covered, out.n_covered_active = len(cov), len(act)
    cov_sessions = [s for s in out.sessions if s.covered]
    out.kinds = {k: sum(1 for s in cov_sessions if s.kind == k) for k in SESSION_KIND_HE}
    out.classes = {k: sum(1 for s in cov_sessions if s.unit_class == k) for k in UNIT_CLASSES}
    cov_contacts = [c for c in out.contacts if c.covered]

    # after abandoned calls (covered stories)
    after = Counter()
    bank_hours: list[float] = []
    by_key = {tl.story.story_key: tl for tl in timelines}
    for row in cov_contacts:
        if row.int_type != "ABN":
            continue
        tl = by_key[row.story_key]
        move, hours = first_move_after(tl, tl.contacts[row.n - 1])
        after[move] += 1
        if hours is not None:
            bank_hours.append(hours)
    out.after_abandon = {k: after.get(k, 0) for k in FIRST_MOVE_HE}

    out.by_type = _by_type(cov_contacts)
    out.by_status = _by_group(cov, "status")
    out.by_topic = _by_group(cov, "topic", label=topic_label)
    out.by_topic.sort(key=lambda r: (-(r["avg_bankers"] or 0), r["key"]))
    out.units = _units(cov_sessions, top_units)
    out.codecat = _codecat(dataset)
    out.head = _head(out, act, cov_contacts, bank_hours, min_rate_n, min_firm_n)
    out.cases = _cases(out, timelines, cases or [])
    out.checks = _checks(dataset, out)
    return out


def _by_type(contacts: list[ContactRow]) -> list[dict]:
    rows = []
    for t in INT_TYPE_ORDER:
        group = [c for c in contacts if c.int_type == t]
        if not group:
            continue
        n = len(group)
        rows.append({
            "type": t, "label": INT_TYPE_HE[t], "n": n,
            "with_activity": sum(c.n_sess > 0 for c in group),
            "with_execute": sum(c.has_execute for c in group),
            "with_branch": sum(c.has_branch for c in group),
            "with_center": sum(c.has_center for c in group),
            "pct_activity": _pct(sum(c.n_sess > 0 for c in group), n),
            "pct_execute": _pct(sum(c.has_execute for c in group), n),
            "pct_branch": _pct(sum(c.has_branch for c in group), n),
            "pct_center": _pct(sum(c.has_center for c in group), n),
            "avg_bankers": _mean(c.n_bankers for c in group if c.n_sess > 0)})
    return rows


def _by_group(stories: list[StorySessions], attr: str, label=None) -> list[dict]:
    """ATL_R02 by_status / by_topic: every covered story (with or without
    sessions), grouped."""
    groups: dict[str, list[StorySessions]] = defaultdict(list)
    for s in stories:
        groups[getattr(s, attr)].append(s)
    order = ("closed", "open", "unclear") if attr == "status" else sorted(groups)
    rows = []
    for key in order:
        g = groups.get(key)
        if not g:
            continue
        rows.append({
            "key": key, "label": label(key) if label else key, "n": len(g),
            "avg_bankers": _mean(s.n_bankers for s in g),
            "pct_3_bankers": _pct(sum(s.n_bankers >= 3 for s in g), len(g)),
            "avg_handoffs": _mean(s.n_handoff for s in g),
            "pct_cross": _pct(sum(s.x_cross for s in g), len(g)),
            "avg_minutes": _mean(s.banker_min for s in g),
            "avg_background": _mean(s.n_bg for s in g),
            "pct_execute": _pct(sum(s.n_do_sess > 0 for s in g), len(g)),
            "pct_execute_after": _pct(sum(s.do_after for s in g), len(g)),
            "median_hours_to_execute": _median(s.hrs_to_do for s in g)})
    return rows


def _units(sessions: list[SessionRow], top: int) -> list[dict]:
    groups: dict[tuple[str, str], list[SessionRow]] = defaultdict(list)
    for s in sessions:
        groups[(s.unit_code, s.unit_class)].append(s)
    rows = [{"unit": code, "label": g[0].unit_label, "class": cls,
             "class_label": UNIT_CLASS_HE[cls], "sessions": len(g),
             "stories": len({s.story_key for s in g}),
             "pct_linked": _pct(sum(s.contact_n is not None for s in g), len(g)),
             "minutes": sum(s.minutes for s in g)}
            for (code, cls), g in groups.items()]
    rows.sort(key=lambda r: (-r["sessions"], r["unit"]))
    return rows[:top]


def _codecat(dataset: JourneyDataset) -> list[dict]:
    """Every operation code seen, with its category (ATL_R02 CODECAT)."""
    rows: dict[str, dict] = {}
    for s in dataset.atlas_sessions:
        for op in s.ops:
            r = rows.setdefault(op.op_code, {"code": op.op_code, "category": op.op_category,
                                             "label": OP_CATEGORY_HE.get(op.op_category, ""),
                                             "description": "", "rows": 0, "stories": set()})
            r["rows"] += 1
            r["stories"].add(s.story_key)
            if op.description and op.description > r["description"]:
                r["description"] = op.description
    order = {k: n for n, k in enumerate(("open", "info", "execute", "unclassified",
                                         "not_customer"))}
    out = []
    for r in rows.values():
        r["stories"] = len(r["stories"])
        out.append(r)
    out.sort(key=lambda r: (order.get(r["category"], 9), -r["rows"], r["code"]))
    return out


def _head(a: SessionAnalysis, act: list[StorySessions], contacts: list[ContactRow],
          bank_hours: list[float], min_n: int, firm_n: int) -> list[HeadItem]:
    """The fifteen lines of ATL_R02's page of numbers, on its bases."""
    n_act = len(act)
    items: list[HeadItem] = []

    def add(key, finding, value, explain, base="", n=0):
        items.append(HeadItem(item=len(items) + 1, key=key, finding_he=finding, value_txt=value,
                              explain_he=explain, base_he=base, n=n,
                              preliminary=0 < n < firm_n))

    base_act = "סיפורים בכיסוי מלא עם פעילות בנקאי"
    add("covered", "סיפורים בכיסוי מלא של לוג אטלס / מהם עם פעילות בנקאי",
        f"{a.n_covered:,} / {n_act:,}",
        "מתוך כל הסיפורים. סיפור שהתחיל לפני היום הראשון שבלוג נשאר מחוץ למדדים, "
        "כי הלוג שומר כשלושה חודשים אחורה", "כל הסיפורים", a.n_stories)
    add("bankers", "בנקאים שונים שנגעו בסיפור - ממוצע / חציון",
        f"{_f(_mean(s.n_bankers for s in act))} / {_f(_median(s.n_bankers for s in act))}",
        "כמה בנקאים שונים פתחו את החשבון באטלס בתקופת הסיפור", base_act, n_act)
    add("three_bankers", "סיפורים שעברו 3 בנקאים ומעלה",
        _share_txt(sum(s.n_bankers >= 3 for s in act), n_act, min_n),
        "סיפורים שבהם לפחות שלושה בנקאים שונים פתחו את החשבון", base_act, n_act)
    add("cross", "סיפורים שעברו בין מרכז הבנקאות לסניפים / מעברים בין יחידות בממוצע",
        f"{_share_txt(sum(s.x_cross for s in act), n_act, min_n)} / "
        f"{_f(_mean(s.n_handoff for s in act))}",
        "הלקוח טופל גם במרכז הבנקאות וגם בסניף אחד לפחות - סניף החשבון או סניף אחר. "
        "מעבר = סוג היחידה השתנה בין שני סשנים רצופים", base_act, n_act)
    add("minutes", "דקות בנקאי לסיפור - ממוצע / חציון - חסם תחתון",
        f"{_f(_mean(s.banker_min for s in act))} / {_f(_median(s.banker_min for s in act))}",
        "זמן מול המסך בלבד, מהפעולה הראשונה בסשן עד האחרונה. הזמן האמיתי גבוה יותר",
        base_act, n_act)
    total_min = sum(s.banker_min for s in act)
    add("background_minutes", "חלק מזמן הבנקאים שנעשה בלי שום פנייה מתועדת",
        (f"{_pct(sum(s.bg_min for s in act), total_min):.1f}%" if total_min and n_act >= min_n
         else f"{_f(sum(s.bg_min for s in act))} מתוך {_f(total_min)} דק'"),
        "סשנים שלא היו בחלון הזמן של שום שיחה או הודעה של הלקוח. כולל ביקור בסניף "
        "ועבודת תפעול, שאין להם רישום פנייה", base_act, n_act)
    n_bg = sum(s.n_bg for s in act)
    add("background_branch", "מתוך הסשנים בלי פנייה מתועדת - בסניפים",
        _share_txt(sum(s.n_bg_br for s in act), n_bg, min_n),
        "סניף החשבון או סניף ויחידה אחרים. השאר במרכז הבנקאות ובתפעול העורפי",
        "סשנים בלי פנייה מתועדת", n_bg)
    n_sess = sum(s.n_sess for s in act)
    add("no_execute", "סשנים בלי שום פעולת ביצוע",
        _share_txt(sum(s.n_nodo_sess for s in act), n_sess, min_n),
        "הבנקאי פתח את המסך, שאל מידע או הפעיל קוד שלא סווג - בלי העברה, הזמנה או שינוי",
        "כל הסשנים בסיפורים האלה", n_sess)
    talk = [c for c in contacts if c.int_type == "TALK"]
    add("talk", "שיחות שנענו בלי הקלטה: עם פעילות / עם ביצוע / בסניף",
        f"{sum(c.n_sess > 0 for c in talk):,} / {sum(c.has_execute for c in talk):,} / "
        f"{sum(c.has_branch for c in talk):,} מתוך {len(talk):,}",
        "שיחות שלא הוקלטו, ולכן אין בהן תוכן לקרוא. אטלס מראה אם מישהו פעל מאחוריהן ואיפה",
        "שיחות כאלה בסיפורים בכיסוי מלא", len(talk))
    ab = a.after_abandon
    n_ab = sum(ab.values())
    add("after_abandon", "אחרי שיחות שננטשו: הבנק פעל ראשון / הלקוח חזר ראשון / אף אחד",
        f"{ab.get('bank', 0):,} / {ab.get('customer', 0):,} / {ab.get('none', 0):,}",
        "אחרי שהלקוח ניתק בהמתנה - מה קרה קודם: בנקאי פתח את החשבון, או שהייתה פנייה נוספת",
        "שיחות שננטשו בסיפורים בכיסוי מלא", n_ab)
    add("bank_hours", "כשהבנק פעל ראשון אחרי נטישה - חציון שעות",
        _f(_median(bank_hours), 2), "הזמן מהשיחה שננטשה עד שבנקאי פתח את החשבון",
        "המקרים שבהם הבנק פעל ראשון", len(bank_hours))
    uncl = [s for s in a.stories if s.covered and s.status == "unclear"]
    add("unclear_execute", "סיפורים שסטטוסם לא ברור עם פעולת ביצוע אחרי הפנייה האחרונה",
        f"{sum(s.do_after for s in uncl):,} מתוך {len(uncl):,}",
        "ביצוע אחרי הפנייה האחרונה אינו מוכיח שהסיפור נסגר - ראו את הטבלה לפי סטטוס",
        "סיפורים בכיסוי מלא שסטטוסם לא ברור", len(uncl))
    add("hours_to_execute", "חציון שעות מהפנייה הראשונה עד פעולת הביצוע הראשונה",
        _f(_median(s.hrs_to_do for s in act)),
        "עד הפעולה הראשונה שאינה צפייה בלבד. ערך שלילי: הבנק ביצע עוד לפני הפנייה",
        "סיפורים עם פעולת ביצוע", sum(s.hrs_to_do is not None for s in act))
    rec_all = [c for c in a.contacts if c.int_type == "REC"]
    add("multi_segment", "שיחות מוקלטות שמורכבות מיותר מקטע אחד",
        f"{sum(c.segments > 1 for c in rec_all):,} מתוך {len(rec_all):,}",
        "שיחות שנשמרו בכמה קבצי הקלטה - סימן להעברה או להמתנה בתוך השיחה",
        "כל השיחות המוקלטות", len(rec_all))
    rec = [c for c in contacts if c.int_type == "REC" and c.n_sess > 0]
    one = [c.n_bankers for c in rec if c.segments == 1]
    many = [c.n_bankers for c in rec if c.segments > 1]
    add("segment_bankers", "בנקאים מאחורי שיחה מוקלטת: קטע אחד / כמה קטעים",
        f"{_f(_mean(one), 2)} / {_f(_mean(many), 2)}",
        "כמה בנקאים שונים פתחו את החשבון בחלון הזמן של השיחה",
        "שיחות מוקלטות עם פעילות בסיפורים בכיסוי מלא", len(rec))
    return items


def _cases(a: SessionAnalysis, timelines: list[StoryTimeline], chosen: list[int]) -> list[dict]:
    """Three stories drawn as a table of contacts and sessions. Unless set in
    the config, chosen as ATL_R02 B6 does: the open story with the most
    bankers; the unclear one with an execution after its last contact and
    the most sessions; the one with the most minutes with no contact."""
    cov = [s for s in a.stories if s.covered]
    picks: list[tuple[int, str]] = []
    if chosen:
        picks = [(no, "") for no in chosen]
    else:
        def first(cands, key):
            cands = sorted(cands, key=key)
            return cands[0].story_no if cands else None
        taken: set[int] = set()
        a_no = first([s for s in cov if s.status == "open" and s.n_sess > 0],
                     lambda s: (-s.n_bankers, -s.n_sess, s.story_no))
        if a_no is not None:
            picks.append((a_no, "פתוח - הכי הרבה בנקאים"))
            taken.add(a_no)
        b_no = first([s for s in cov if s.status == "unclear" and s.do_after
                      and s.story_no not in taken],
                     lambda s: (-s.n_sess, -s.n_bankers, s.story_no))
        if b_no is not None:
            picks.append((b_no, "לא ברור - ביצוע אחרי הפנייה האחרונה"))
            taken.add(b_no)
        c_no = first([s for s in cov if s.story_no not in taken and s.bg_min > 0],
                     lambda s: (-s.bg_min, s.story_no))
        if c_no is not None:
            picks.append((c_no, "הכי הרבה עבודה בלי פנייה מתועדת"))
    by_no = {tl.story.story_no: tl for tl in timelines}
    out = []
    for no, why in picks:
        if no not in by_no:
            continue
        rows = [{"at": c.at, "event": "פנייה", "what": INT_TYPE_HE[c.int_type], "unit": "",
                 "banker": "", "minutes": None, "ops": "", "order": 0}
                for c in a.contacts if c.story_no == no]
        rows += [{"at": s.start, "event": "אטלס", "what": SESSION_KIND_HE[s.kind],
                  "unit": UNIT_CLASS_HE[s.unit_class], "banker": s.banker,
                  "minutes": s.minutes, "ops": s.ops_txt, "order": 1}
                 for s in a.sessions if s.story_no == no]
        rows.sort(key=lambda r: (r["at"], r["order"]))
        out.append({"story_no": no, "why": why, "rows": rows})
    return out


def _checks(dataset: JourneyDataset, a: SessionAnalysis) -> list[dict]:
    """The completeness checks printed as the last appendix (as R01-R03 do)."""
    rep = dataset.report
    issues = {i.code: i for i in rep.issues}
    out = [{"check": "סשנים / שורות אטלס שנקראו",
            "value": f"{rep.counts.atlas_sessions:,} / {rep.counts.atlas_ops:,}", "ok": True},
           {"check": "מקור הסשנים",
            "value": {"export": "טבלת הסשנים שיוצאה", "rows": "נבנו משורות הלוג",
                      "none": "אין"}[a.sessions_from], "ok": a.sessions_from != "none"},
           {"check": "היום הראשון בלוג",
            "value": a.coverage_from.strftime("%d/%m/%Y") if a.coverage_from else "—",
            "ok": a.coverage_from is not None},
           {"check": "סיפורים בכיסוי מלא", "value": f"{a.n_covered:,} מתוך {a.n_stories:,}",
            "ok": True}]
    if "atlas_sessions_differ" in issues:
        out.append({"check": "סשנים שהשורות אינן בונות מחדש", "value": issues[
            "atlas_sessions_differ"].message, "ok": False})
    elif "atlas_sessions_rebuilt" in issues:
        out.append({"check": "סשנים שהשורות בונות מחדש בדיוק",
                    "value": f"{issues['atlas_sessions_rebuilt'].count:,}", "ok": True})
    for code, label in (("atlas_ops_outside_sessions", "שורות שלא נפלו באף סשן"),
                        ("atlas_rows_unmapped", "שורות של חשבון שלא חובר לסיפור"),
                        ("atlas_contacts_unmatched", "פניות באטלס שלא נמצאו בנתונים")):
        count = issues[code].count if code in issues else 0
        out.append({"check": label, "value": f"{count:,}", "ok": count == 0})
    # information, as in ATL_R02 B2: an unclassified code decides no figure alone
    n_codes = len({op.op_code for s in dataset.atlas_sessions for op in s.ops})
    n_uncl = issues["atlas_codes_unclassified"].count if "atlas_codes_unclassified" in issues else 0
    out.append({"check": "קודי פעולה שלא סווגו", "value": f"{n_uncl:,} מתוך {n_codes:,}",
                "ok": True})
    return out


# ------------------------------------------------------------------ checking


def rows_by_category(dataset: JourneyDataset) -> dict[str, int]:
    counts = Counter(op.op_category for s in dataset.atlas_sessions for op in s.ops)
    return {k: counts.get(k, 0) for k in SESSION_KIND_HE}


def compare_expected(a: SessionAnalysis, dataset: JourneyDataset, expected: dict) -> list[str]:
    """Every difference between this analysis and the numbers the bank's
    Atlas project published (eval/atlas_r02_expected.yaml). Empty = the same."""
    out: list[str] = []
    actual_counts = {"stories": a.n_stories, "sessions": len(a.sessions),
                     "ops": sum(len(s.ops) for s in dataset.atlas_sessions),
                     "covered": a.n_covered, "covered_active": a.n_covered_active,
                     "codes": len(a.codecat)}
    for key, want in (expected.get("counts") or {}).items():
        if actual_counts.get(key) != want:
            out.append(f"counts.{key}: {actual_counts.get(key)} (expected {want})")
    head = {h.key: h.value_txt for h in a.head}
    for key, want in (expected.get("head") or {}).items():
        if head.get(key) != str(want):
            out.append(f"head.{key}: {head.get(key)} (expected {want})")
    for name, actual in (("kinds", a.kinds), ("classes", a.classes),
                         ("rows_by_category", rows_by_category(dataset))):
        for key, want in (expected.get(name) or {}).items():
            if actual.get(key, 0) != want:
                out.append(f"{name}.{key}: {actual.get(key, 0)} (expected {want})")
    by_type = {r["type"]: r for r in a.by_type}
    for t, fields in (expected.get("by_type") or {}).items():
        row = by_type.get(t)
        if row is None:
            out.append(f"by_type.{t}: missing")
            continue
        for key, want in fields.items():
            got = row.get(key)
            if got is None or abs(round(float(got), 1) - float(want)) > 0.05:
                out.append(f"by_type.{t}.{key}: {got if got is None else round(got, 2)} "
                           f"(expected {want})")
    return out
