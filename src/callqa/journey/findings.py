"""The journey report's opinion: findings and actions, written from the numbers.

As in reporting/executive/findings.py, no language model writes this: every
sentence comes from a computed figure by a fixed rule, numbers are marked
⟦like this⟧ for right-to-left layout, and a figure whose base is too small
is either left out or said to be preliminary. A finding that rests on an
inference (a promise judged from the event sequence, a status inferred from
Atlas) says so.

Links: `section` is an anchor in level 2; `filter` opens the matching stories
in the level-3 explorer (keys understood by journey_report.js.j2).
"""

from __future__ import annotations

from callqa.journey.analysis import OBJECTIVE_HE, JourneyAnalysis, Metric
from callqa.journey.vocab import Taxonomy
from callqa.reporting.executive.findings import (
    SEVERITY_HE,
    SEVERITY_ORDER,
    Action,
    Finding,
    Opinion,
    clean,
    count,
    num,
    pct,
    several,
)

__all__ = ["SEVERITY_HE", "build_opinion"]


def _ci(m: Metric) -> str:
    if m.low is None or m.high is None:
        return ""
    lo, hi = max(0.0, m.low), min(1.0, m.high)
    return f" (רווח סמך ⟦95%⟧: {pct(lo)}–{pct(hi)})"


def _usable(m: Metric | None) -> bool:
    return m is not None and m.shown and m.value is not None and bool(m.n)


def _prelim(m: Metric) -> str:
    return " הבסיס קטן, והמספר ראשוני." if m.preliminary else ""


def _stories(k: int) -> str:
    return several(k, "סיפור אחד", "סיפורים")


def _returns(k: int) -> str:
    return several(k, "חזרה אחת", "חזרות")


# -- rules ----------------------------------------------------------------------

def _burden(a: JourneyAnalysis) -> list[Finding]:
    m = a.metrics
    n_st, n_ret = int(m["stories"].value or 0), int(m["returns"].value or 0)
    if not n_st or not n_ret:
        return []
    heavy = [s for s in a.stories if s.returns >= 5]
    per = m["returns_per_story"].value or 0
    text = (f"ב־{_stories(n_st)} נרשמו {count(int(m['contacts'].value or 0))} מגעים, מהם "
            f"{_returns(n_ret)} — {num(per)} בממוצע לסיפור.")
    if heavy:
        share = sum(s.returns for s in heavy) / n_ret
        text += (f" {_stories(len(heavy))} עם ⟦5⟧ חזרות ומעלה מרכזים {pct(share, 0)} "
                 "מכל החזרות.")
    return [Finding(
        key="burden", severity="high" if per >= 3 else "medium",
        title=f"{num(per)} חזרות בממוצע לכל לקוח",
        text=text,
        meaning=("כל חזרה היא עלות טיפול נוספת וסימן שהפנייה הקודמת לא סיימה את העניין. "
                 "ריכוז החזרות במעט סיפורים מאפשר טיפול ממוקד בהם."),
        section="sec-returns", weight=per * 10,
        filter={"min_returns": 5} if heavy else None, link_label="הצג את הסיפורים")]


def _failure(a: JourneyAnalysis, tax: Taxonomy) -> list[Finding]:
    m = a.metrics["failure_rate"]
    if not _usable(m):
        return []
    rate = m.value or 0.0
    lo = m.low if m.low is not None else rate
    parts = [f"{clean(tax.category_label(c))} — {count(a.categories_strict.get(c, 0))}"
             for c in sorted(tax.failure_categories) if a.categories_strict.get(c)]
    return [Finding(
        key="failure", severity="critical" if lo >= 0.3 else ("high" if lo >= 0.15 else "medium"),
        title=f"{pct(rate)} מהחזרות שנבדקו הן כשל טיפול",
        text=(f"{count(m.k or 0)} מתוך {count(m.n or 0)} החזרות שיש להן תוכן{_ci(m)}. "
              + ("פירוט: " + "; ".join(parts) + "." if parts else "") + _prelim(m)),
        meaning=("חזרה שמקורה בכשל היא חזרה שהבנק יכול היה למנוע. רווח הסמך מחושב ברמת "
                 "הסיפור, כי חזרות של אותו לקוח תלויות זו בזו."),
        section="sec-categories", weight=rate * 100,
        filter={"failure": 1}, link_label="הצג סיפורים עם כשל")]


def _visibility(a: JourneyAnalysis) -> list[Finding]:
    m = a.metrics["classifiable"]
    n_ret = int(a.metrics["returns"].value or 0)
    if not n_ret or m.value is None:
        return []
    unseen = n_ret - (m.k or 0)
    if unseen <= 0:
        return []
    recovered = sum(a.objective_classes.get(k, 0) for k in
                    ("abandoned", "answered_execute", "answered_info", "answered_no_trace"))
    parts = [f"{clean(OBJECTIVE_HE[k])} — {count(v)}" for k, v in
             sorted(a.objective_classes.items(), key=lambda kv: -kv[1])
             if k not in ("content", "bank_initiated") and v]
    return [Finding(
        key="visibility", severity="medium" if (m.value or 0) < 0.6 else "info",
        title=f"ל־{pct(1 - (m.value or 0), 0)} מהחזרות אין תוכן שמאפשר לסווג אותן",
        text=(f"{_returns(unseen)} בלי תוכן מתועד. מתוכן, {count(recovered)} קיבלו תיאור "
              "עובדתי מנתוני המוקד והאטלס (נטישה, פעולה שבוצעה, צפייה בלבד). "
              + ("פירוט: " + "; ".join(parts) + "." if parts else "")),
        meaning=("אי אפשר לדעת מה נאמר בשיחה שלא הוקלטה. הדוח לא מנחש: חזרות אלה מתוארות "
                 "לפי מה שקרה סביבן, ומסומנות כהסקה בכל מקום שבו הן משפיעות על מדד."),
        section="sec-categories", weight=(1 - (m.value or 0)) * 40)]


def _abandon(a: JourneyAnalysis) -> list[Finding]:
    m = a.metrics["abandoned"]
    if not _usable(m) or not m.k:
        return []
    out = [Finding(
        key="abandoned", severity="high" if (m.value or 0) >= 0.15 else "medium",
        title=f"{pct(m.value)} מהשיחות ננטשו לפני מענה",
        text=f"{count(m.k)} מתוך {count(m.n or 0)} השיחות{_ci(m)}.{_prelim(m)}",
        meaning="לקוח שלא נענה חוזר — ולעיתים בערוץ יקר יותר, או לא חוזר כלל.",
        section="sec-abandon", weight=(m.value or 0) * 60,
        filter={"abandoned": 1}, link_label="הצג סיפורים עם נטישה")]
    cf = a.metrics.get("customer_first_after_abandon")
    if _usable(cf) and cf.k:
        wait = a.metrics.get("bank_first_hours")
        extra = (f" כשהבנק כן פעל ראשון, עברו עד אז {num(wait.value)} שעות (חציון)."
                 if wait is not None and wait.value is not None else "")
        out.append(Finding(
            key="abandon-callback", severity="high" if (cf.value or 0) >= 0.5 else "medium",
            title=f"אחרי {pct(cf.value, 0)} מהנטישות, הלקוח נאלץ לפנות שוב בעצמו",
            text=(f"ב־{count(cf.k)} מתוך {count(cf.n or 0)} שיחות שננטשו (בסיפורים עם כיסוי אטלס "
                  f"מלא), הלקוח פנה שוב לפני שבנקאי פתח את החשבון או חזר אליו.{extra}"
                  + _prelim(cf)),
            meaning="חזרה יזומה אחרי נטישה חוסכת את הפנייה הבאה ומראה ללקוח שראו אותו.",
            section="sec-abandon", weight=(cf.value or 0) * 50,
            filter={"abandoned": 1}, link_label="הצג את הסיפורים"))
    return out


def _promises(a: JourneyAnalysis) -> list[Finding]:
    m = a.metrics["promises_broken"]
    made = a.promise_funnel.get("made", 0)
    if not made or not _usable(m):
        return []
    rate = m.value or 0.0
    unknown = a.promise_funnel.get("unknown", 0)
    return [Finding(
        key="promises", severity="critical" if rate >= 0.4 else ("high" if rate >= 0.2 else "medium"),
        title=f"{pct(rate, 0)} מההבטחות „נחזור אליך” לא קוימו",
        text=(f"מתוך {several(made, 'הבטחה אחת', 'הבטחות')} של הבנק בשיחות, {count(m.k or 0)} "
              f"הופרו ו־{count(a.promise_funnel.get('kept', 0))} קוימו{_ci(m)}"
              + (f"; על {count(unknown)} אי אפשר לקבוע." if unknown else ".") + _prelim(m)),
        meaning=("הבטחה שלא קוימה מחזירה את הלקוח, והפעם כועס יותר. קיום נקבע מעובדות: שיחה "
                 "יוצאת או פעולה באטלס לפני שהלקוח חזר, ובתוך ⟦2⟧ ימי עסקים."),
        section="sec-promises", weight=rate * 80,
        filter={"promise": "broken"}, link_label="הצג הבטחות שהופרו")]


def _retold(a: JourneyAnalysis) -> list[Finding]:
    m = a.metrics["retold"]
    if not _usable(m) or not m.k:
        return []
    return [Finding(
        key="retold", severity="high" if (m.value or 0) >= 0.3 else "medium",
        title=f"ב־{pct(m.value, 0)} מהחזרות הלקוח נדרש לספר את הסיפור מחדש",
        text=f"{count(m.k)} מתוך {count(m.n or 0)} החזרות שיש להן תוכן{_ci(m)}.{_prelim(m)}",
        meaning="כשההקשר לא עובר בין בנקאים, הלקוח משלם בזמן ובסבלנות, והבנק בזמן טיפול כפול.",
        section="sec-effort", weight=(m.value or 0) * 60,
        filter={"retold": 1}, link_label="הצג את הסיפורים")]


def _same_day(a: JourneyAnalysis) -> list[Finding]:
    m = a.metrics["same_day_3"]
    if not _usable(m) or not m.k:
        return []
    return [Finding(
        key="same-day", severity="medium" if (m.value or 0) >= 0.1 else "info",
        title=f"ב־{pct(m.value, 0)} מהסיפורים היו ⟦3⟧ מגעים ומעלה ביום אחד",
        text=f"{_stories(m.k)} מתוך {count(m.n or 0)}{_ci(m)}.{_prelim(m)}",
        meaning="רצף פניות באותו יום הוא הסימן הברור ביותר ללקוח שלא מצליח לסגור עניין.",
        section="sec-gaps", weight=(m.value or 0) * 40,
        filter={"same_day": 3}, link_label="הצג את הסיפורים")]


def _status(a: JourneyAnalysis) -> list[Finding]:
    closed, unclear = a.metrics["closed"], a.metrics["unclear"]
    if not _usable(closed):
        return []
    out = []
    by_basis = {}
    for s in a.stories:
        if s.status == "closed":
            by_basis[s.status_basis] = by_basis.get(s.status_basis, 0) + 1
    inferred = by_basis.get("inference", 0)
    med = a.metrics.get("median_days_to_resolution")
    text = f"{_stories(closed.k or 0)} מתוך {count(closed.n or 0)} נסגרו"
    if inferred:
        text += f" ({count(inferred)} מהם לפי הסקה: פעולה בחשבון ואחריה שקט)"
    text += f"; {count(unclear.k or 0)} בסטטוס לא ברור."
    if med is not None and med.value is not None:
        text += f" חציון הזמן עד סגירה: {num(med.value)} ימים (Kaplan-Meier)."
    out.append(Finding(
        key="status", severity="medium" if (closed.value or 0) < 0.5 else "info",
        title=f"{pct(closed.value, 0)} מהסיפורים נסגרו",
        text=text + _prelim(closed),
        meaning=("סיפור שלא ברור איך הסתיים הוא סיפור שאף אחד לא סגר מולו. "
                 "Kaplan-Meier סופר סיפור פתוח כ„עדיין פתוח” ולא מוחק אותו."),
        section="sec-resolution", weight=(1 - (closed.value or 0)) * 30,
        filter={"status": "open"}, link_label="הצג סיפורים פתוחים"))
    return out


def _effort(a: JourneyAnalysis) -> list[Finding]:
    m3, cr = a.metrics.get("three_bankers"), a.metrics.get("crossed")
    out = []
    if _usable(m3) and m3.k:
        per = a.metrics["bankers_per_story"].value
        out.append(Finding(
            key="bankers", severity="high" if (m3.value or 0) >= 0.4 else "medium",
            title=f"ב־{pct(m3.value, 0)} מהסיפורים טיפלו ⟦3⟧ בנקאים ומעלה",
            text=(f"{_stories(m3.k)} מתוך {count(m3.n or 0)} הסיפורים שיש להם כיסוי אטלס מלא; "
                  f"{num(per)} בנקאים בממוצע לסיפור."
                  + (f" ב־{pct(cr.value, 0)} מהם העניין עבר הלוך ושוב בין המוקד לסניפים."
                     if _usable(cr) and cr.k else "") + _prelim(m3)),
            meaning="ככל שיותר ידיים נוגעות בעניין, גדל הסיכוי שמשהו ייפול בין הכיסאות.",
            section="sec-effort", weight=(m3.value or 0) * 40,
            filter={"min_bankers": 3}, link_label="הצג את הסיפורים"))
    bg = a.metrics.get("background_share")
    mins = a.metrics.get("banker_minutes_per_story")
    if bg is not None and bg.shown and bg.value is not None and mins is not None and mins.value:
        out.append(Finding(
            key="background", severity="info",
            title=f"{pct(bg.value, 0)} מזמן הבנקאים לא צמוד לשום פנייה מתועדת",
            text=(f"בממוצע {num(mins.value)} דקות עבודה באטלס לכל סיפור (חסם תחתון). חלק "
                  "מהזמן הזה הוא טיפול בסניף או בתפעול העורפי, שאין לו רישום פנייה."),
            meaning="זו העלות הנסתרת של החזרות: העבודה שנעשית מאחורי הקלעים.",
            section="sec-effort", weight=(bg.value or 0) * 20))
    return out


def _topics(a: JourneyAnalysis) -> list[Finding]:
    rows = [r for r in a.topics if r.get("rate") is not None]
    if len(rows) < 2:
        return []
    worst = max(rows, key=lambda r: r["rate"])
    others_k = sum(r["failures"] for r in rows if r is not worst)
    others_n = sum(r["content_returns"] for r in rows if r is not worst)
    if not others_n:
        return []
    other_rate = others_k / others_n
    if worst["low"] is None or worst["low"] <= other_rate:
        return []
    return [Finding(
        key="topic", severity="high",
        title=f"„{clean(worst['label'])}” — שיעור הכשל הגבוה ביותר",
        text=(f"{pct(worst['rate'])} מהחזרות בנושא זה הן כשל (רווח סמך ⟦95%⟧: "
              f"{pct(worst['low'])}–{pct(worst['high'])}), מול {pct(other_rate)} בשאר הנושאים. "
              f"{_stories(worst['stories'])}, {_returns(worst['returns'])}."),
        meaning="נושא שבו הכשל מרוכז הוא נקודת כניסה לשינוי תהליך, לא רק לאימון בנקאים.",
        section="sec-topics", weight=worst["rate"] * 70,
        filter={"topic": worst["topic"]}, link_label="הצג את הסיפורים")]


# -- actions ----------------------------------------------------------------------

def _actions(a: JourneyAnalysis, findings: dict[str, Finding]) -> list[Action]:
    acts: list[Action] = []
    m = a.metrics
    if "promises" in findings:
        broken = m["promises_broken"].k or 0
        acts.append(Action(
            title="רישום ומעקב לכל הבטחה „נחזור אליך”",
            text=("כל הבטחה בשיחה נרשמת עם מועד ואחראי, ומי שלא חזר עד המועד מקבל התראה. "
                  "בסוף יום: רשימת הבטחות פתוחות לכל צוות."),
            impact=f"עד {_returns(broken)} פחות, לפי ההבטחות שהופרו בדוח זה",
            owner="מנהלי צוותים במרכז הבנקאות", section="sec-promises",
            score=broken * 3.0, filter={"promise": "broken"}))
    if "abandon-callback" in findings:
        cf = m["customer_first_after_abandon"]
        acts.append(Action(
            title="חזרה יזומה אחרי כל שיחה שננטשה",
            text=("שיחה שננטשה נכנסת לתור חזרה עם יעד זמן. לקוח שנטש פעמיים באותו יום עולה "
                  "לראש התור."),
            impact=f"עד {_returns(cf.k or 0)} שהלקוח יזם בעצמו אחרי נטישה",
            owner="ניהול המוקד", section="sec-abandon", score=(cf.k or 0) * 2.0,
            filter={"abandoned": 1}))
    if "retold" in findings or "bankers" in findings:
        k = (m["retold"].k or 0) if "retold" in findings else (m["three_bankers"].k or 0)
        acts.append(Action(
            title="סיכום פנייה קצר שעובר עם הלקוח",
            text=("בסוף כל מגע נרשם משפט אחד: מה הלקוח ביקש, מה נעשה, מה נשאר פתוח. הבנקאי הבא "
                  "פותח בו ולא מבקש מהלקוח להתחיל מההתחלה."),
            impact=(f"{_returns(k)} שבהן הלקוח סיפר מחדש" if "retold" in findings
                    else f"{_stories(k)} עם ⟦3⟧ בנקאים ומעלה"),
            owner="מרכז הבנקאות והסניפים", section="sec-effort", score=k * 1.5,
            filter={"retold": 1} if "retold" in findings else {"min_bankers": 3}))
    if "topic" in findings:
        f = findings["topic"]
        acts.append(Action(
            title="בדיקת תהליך בנושא עם הכשל הגבוה",
            text="מיפוי השלבים שבהם העניין נתקע בנושא זה, מול הסיפורים שבדוח, ותיקון בתהליך.",
            impact=f.title, owner="בעלי התהליך", section="sec-topics", score=f.weight,
            filter=f.filter))
    if "visibility" in findings and (m["classifiable"].value or 1) < 0.6:
        unseen = int(m["returns"].value or 0) - (m["classifiable"].k or 0)
        acts.append(Action(
            title="הקלטה של שיחות הסניפים",
            text="חזרות רבות עוברות דרך שיחות שלא הוקלטו, ולכן אי אפשר לדעת למה הלקוח חזר.",
            impact=f"{_returns(unseen)} שהיום לא ניתן לסווג",
            owner="מערכות מידע ותפעול", section="sec-categories", score=unseen * 0.5))
    acts.sort(key=lambda x: -x.score)
    return acts[:4]


# -- the opinion --------------------------------------------------------------------

def assess(a: JourneyAnalysis, findings: list[Finding]) -> str:
    n = int(a.metrics["stories"].value or 0)
    if not n:
        return "אין בנתונים סיפורי לקוח"
    worst = min((SEVERITY_ORDER[f.severity] for f in findings), default=4)
    if n < 30:
        head = f"הערכה ראשונית בלבד — {_stories(n)}"
    elif worst == 0:
        head = "חזרות רבות שמקורן בכשל טיפול — נדרשת התערבות"
    elif worst == 1:
        head = "עומס חזרות משמעותי, עם מוקדי כשל ברורים"
    else:
        head = "עומס חזרות מתון"
    fr = a.metrics["failure_rate"]
    if _usable(fr):
        head += f"; {pct(fr.value, 0)} מהחזרות שנבדקו הן כשל"
    return head


def bottom_line(a: JourneyAnalysis, findings: list[Finding], actions: list[Action]) -> list[str]:
    m = a.metrics
    n_st = int(m["stories"].value or 0)
    if not n_st:
        return ["לא נמצאו בנתונים סיפורי לקוח להצגה."]
    lines = [f"הדוח עוקב אחרי {_stories(n_st)}: {count(int(m['contacts'].value or 0))} מגעים, "
             f"מהם {_returns(int(m['returns'].value or 0))}, לאורך התקופה שבנתונים."]
    fr = m["failure_rate"]
    if _usable(fr):
        lines.append(f"מתוך החזרות שיש להן תוכן, {pct(fr.value)} הן כשל טיפול{_ci(fr)} — "
                     "חזרות שהבנק יכול היה למנוע.")
    elif fr.n == 0:
        lines.append("סיווג החזרות לפי תוכן השיחות עוד לא הורץ על batch זה; המדדים כאן "
                     "מבוססים על עובדות: זמנים, ערוצים, נטישה ופעולות בנקאים.")
    for key in ("promises", "abandon-callback", "retold", "bankers"):
        f = next((x for x in findings if x.key == key), None)
        if f is not None:
            lines.append(f.title + ".")
            break
    if actions:
        lines.append(f"הפעולה בעלת ההשפעה הגדולה ביותר: {actions[0].title}.")
    if n_st < 30:
        lines.append("מספר הסיפורים קטן; המספרים מוצגים לידיעה ואינם מאפשרים מסקנה נחרצת.")
    return lines


def build_opinion(a: JourneyAnalysis, taxonomy: Taxonomy) -> Opinion:
    findings: list[Finding] = []
    for rule in (_burden, _abandon, _promises, _retold, _same_day, _status, _effort,
                 _topics, _visibility):
        findings.extend(rule(a))
    findings.extend(_failure(a, taxonomy))
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], -f.weight, f.key))
    by_key = {f.key: f for f in findings}
    by_section: dict[str, list[Finding]] = {}
    for f in findings:
        by_section.setdefault(f.section, []).append(f)
    actions = _actions(a, by_key)
    top = [f for f in findings if f.severity != "info"][:5]
    if len(top) < 5:
        top += [f for f in findings if f.severity == "info"][: 5 - len(top)]
    return Opinion(assessment=assess(a, findings), bottom_line=bottom_line(a, findings, actions),
                   findings=findings, top_findings=top, actions=actions, by_section=by_section)
