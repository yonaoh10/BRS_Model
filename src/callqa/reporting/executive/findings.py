"""The professional opinion: findings and actions, written from the numbers.

Every sentence here is generated from a computed statistic by a fixed rule -
no language model writes the management summary - so every number in it can
be traced to a chart, and the same batch always yields the same opinion.
Numbers are marked ⟦like this⟧ so the template can isolate them for
right-to-left layout; data values put into a sentence are stripped of those
marks first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from callqa.reporting.common import load_recommendations
from callqa.reporting.executive.analysis import (
    BAND_LABEL,
    MIN_PRACTICAL_DIFF,
    Analysis,
    Driver,
    band_of,
)

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "positive": 3, "info": 4}
SEVERITY_HE = {"critical": "קריטי", "high": "גבוה", "medium": "בינוני",
               "positive": "חיובי", "info": "לידיעה"}


@dataclass
class Finding:
    key: str
    severity: str
    title: str
    text: str
    meaning: str
    section: str
    weight: float
    filter: dict | None = None
    link_label: str = "הצג את השיחות"


@dataclass
class Action:
    title: str
    text: str
    impact: str
    owner: str
    section: str
    score: float
    filter: dict | None = None
    coaching: str | None = None


@dataclass
class Opinion:
    assessment: str                       # one-line overall verdict
    bottom_line: list[str]                # 2-5 sentences
    findings: list[Finding]               # all, ranked
    top_findings: list[Finding]           # the executive five
    actions: list[Action]                 # the top three
    by_section: dict[str, list[Finding]] = field(default_factory=dict)


# -- formatting -----------------------------------------------------------------

def clean(value: object) -> str:
    """A data value about to go into a generated sentence."""
    return str(value).replace("⟦", "").replace("⟧", "")


def n1(x: float | None) -> str:
    if x is None:
        return "—"
    r = round(x, 1)
    body = str(int(abs(r))) if r == int(r) else f"{abs(r):.1f}"
    return ("−" if r < 0 else "") + body


def num(x: float | None) -> str:
    return f"⟦{n1(x)}⟧"


def pct(p: float | None, digits: int = 1) -> str:
    if p is None:
        return "⟦—⟧"
    v = round(p * 100, digits)
    text = str(int(v)) if v == int(v) else f"{v:.{digits}f}"
    return f"⟦{text}%⟧"


def signed(x: float | None) -> str:
    if x is None:
        return "⟦—⟧"
    r = round(x, 1)
    body = n1(abs(r))
    return f"⟦{'+' if r > 0 else '−' if r < 0 else ''}{body}⟧"


def count(k: int) -> str:
    return f"⟦{k:,}⟧"


def several(k: int, one: str, many: str) -> str:
    """'בנקאי אחד' / '⟦3⟧ בנקאים': Hebrew counts one differently."""
    return one if k == 1 else f"{count(k)} {many}"


def names(items: list[str], limit: int = 5) -> str:
    shown = [f"⟦{clean(i)}⟧" for i in items[:limit]]
    if len(items) > limit:
        shown.append(f"ועוד {count(len(items) - limit)}")
    if len(shown) <= 1:
        return "".join(shown)
    return ", ".join(shown[:-1]) + " ו" + shown[-1]


def dim_word(analysis: Analysis, dim_id: str) -> str:
    return f"„{clean(analysis.dim_names.get(dim_id, dim_id))}”"


# -- the opinion ----------------------------------------------------------------

def assess(analysis: Analysis) -> str:
    mean = analysis.total.mean
    if mean is None:
        return "אין די שיחות מנוקדות לגיבוש הערכה"
    gate = analysis.gate_rate.p or 0.0
    if mean >= 80 and gate < 0.02:
        verdict = "רמת שירות גבוהה"
    elif mean >= 70:
        verdict = "רמת שירות טובה, עם מוקדי שיפור ברורים"
    elif mean >= 60:
        verdict = "רמת שירות בינונית — נדרש מיקוד בשיפור"
    else:
        verdict = "רמת שירות נמוכה — נדרשת התערבות"
    if gate >= 0.05:
        verdict += "; סיכון ציות מהותי"
    elif gate >= 0.02:
        verdict += "; סיכון ציות הדורש מעקב"
    return verdict


def bottom_line(analysis: Analysis) -> list[str]:
    a = analysis
    if a.n_scored == 0:
        return ["לא נמצאו בתקופה זו שיחות שנוקדו ופורסמו, ולכן אין מדד איכות להצגה. "
                "פירוט מצב השיחות מופיע בפרק הכיסוי והאיכות."]
    lines = []
    band = BAND_LABEL[band_of(a.total.mean)]
    ci = ""
    if a.total.low is not None and a.total.high is not None:
        ci = f" (רווח סמך 95%: ⟦{n1(a.total.low)}–{n1(a.total.high)}⟧)"
    lines.append(
        f"מדד האיכות הממוצע של {count(a.n_scored)} השיחות שנותחו הוא {num(a.total.mean)} "
        f"מתוך ⟦100⟧{ci} — רמה „{band}”. {pct(a.high_rate.p)} מהשיחות ברמה גבוהה "
        f"(⟦80⟧ ומעלה) ו־{pct(a.low_rate.p)} מתחת ל־⟦60⟧.")
    fails = round((a.gate_rate.p or 0) * a.n_scored)
    if fails:
        who = ("שיחה אחת" if fails == 1 else f"{count(fails)} שיחות") + f" ({pct(a.gate_rate.p)})"
        verb = "נכשלה" if fails == 1 else "נכשלו"
        lines.append(
            f"{who} {verb} בשער חובה — זיהוי הלקוח או גילוי נאות — ובהן הציון מוגבל "
            f"ל־⟦59⟧ לכל היותר; זהו הסיכון המרכזי בדוח.")
    else:
        lines.append("לא נמצאו כשלים בשערי החובה (זיהוי הלקוח וגילוי נאות).")
    top = sorted(a.dims, key=lambda d: -d.lost_share)[:2]
    if top and top[0].lost_share > 0:
        shares = sum(d.lost_share for d in top)
        lines.append(
            f"הפער בין המדד ל־⟦100⟧ נובע בעיקר מ־{dim_word(a, top[0].id)}"
            + (f" ומ־{dim_word(a, top[1].id)}" if len(top) > 1 and top[1].lost_share > 0 else "")
            + f" ({pct(shares, 0)} מהנקודות שאבדו).")
    if a.trend is not None and a.trend.slope_per_30d is not None:
        if a.trend.significant and abs(a.trend.slope_per_30d) >= 1:
            word = "שיפור" if a.trend.slope_per_30d > 0 else "ירידה"
            lines.append(f"לאורך התקופה נרשמה {word} מובהקת של "
                         f"{signed(a.trend.slope_per_30d)} נקודות בחודש בממוצע.")
        else:
            lines.append("המדד יציב לאורך התקופה — ללא מגמה מובהקת.")
    if a.review_rate.p and a.review_rate.p >= 0.05:
        lines.append(f"שים לב: {pct(a.review_rate.p)} מהשיחות ממתינות לבדיקה אנושית "
                     f"ואינן כלולות במדד.")
    return lines


def build_opinion(analysis: Analysis, recommendations_path: str | None = None) -> Opinion:
    recs = load_recommendations(recommendations_path)
    findings: list[Finding] = []
    for rule in (_gate_findings, _points_findings, _dimension_findings, _trend_findings,
                 _segment_findings, _banker_findings, _driver_findings, _coverage_findings):
        findings.extend(rule(analysis))
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], -f.weight, f.key))
    by_section: dict[str, list[Finding]] = {}
    for f in findings:
        by_section.setdefault(f.section, []).append(f)
    return Opinion(
        assessment=assess(analysis), bottom_line=bottom_line(analysis), findings=findings,
        top_findings=_executive_five(findings), actions=_actions(analysis, recs),
        by_section=by_section,
    )


def _executive_five(findings: list[Finding]) -> list[Finding]:
    """Four problems and the strongest good news: a management summary that is
    all red is not believed, and one that hides the red is not useful."""
    problems = [f for f in findings if f.severity in ("critical", "high", "medium")]
    positives = [f for f in findings if f.severity == "positive"]
    picked = problems[:4]
    if positives:
        picked.append(positives[0])
    for f in findings:
        if len(picked) >= 5:
            break
        if f not in picked and f.severity != "info":
            picked.append(f)
    return sorted(picked, key=lambda f: (SEVERITY_ORDER[f.severity], -f.weight))


# -- rules ----------------------------------------------------------------------

def _gate_findings(a: Analysis) -> list[Finding]:
    if a.n_scored == 0:
        return []
    fails = round((a.gate_rate.p or 0) * a.n_scored)
    if fails == 0:
        return [Finding(
            key="gates-clean", severity="positive",
            title="אין כשלים בשערי החובה",
            text=f"בכל {count(a.n_scored)} השיחות בוצעו זיהוי הלקוח וגילוי נאות ברמה "
                 "העומדת בסף.",
            meaning="אין בתקופה זו ממצא רגולטורי בתחומי הזיהוי והגילוי הנאות.",
            section="sec-risk", weight=1.0)]
    rate = a.gate_rate.p or 0
    parts = [f"{dim_word(a, g.id)}: {count(g.fails)} ({pct(g.rate.p)})"
             for g in a.gates if g.fails]
    text = f"פירוט: {'; '.join(parts)}."
    conc = [g for g in a.gates if g.concentration is not None and g.concentration >= 0.5]
    if conc:
        g = max(conc, key=lambda g: g.concentration)
        text += (f" ב{dim_word(a, g.id)} הכשלים מרוכזים: {pct(g.concentration, 0)} מהם אצל "
                 f"⟦20%⟧ מהבנקאים.")
    return [Finding(
        key="gates", severity="critical" if rate >= 0.05 else "high",
        title=(f"שיחה אחת ({pct(rate)}) נכשלה בשער חובה" if fails == 1 else
               f"{count(fails)} שיחות ({pct(rate)}) נכשלו בשער חובה"),
        text=text,
        meaning="כשל שער הוא סיכון רגולטורי ישיר — מסירת מידע ללא זיהוי מלא או ללא גילוי "
                "נאות — ולא רק שאלה של איכות שירות.",
        section="sec-risk", weight=rate * 100, filter={"gate": "any"})]


def _points_findings(a: Analysis) -> list[Finding]:
    if a.n_scored == 0:
        return []
    top = max(a.dims, key=lambda d: d.lost_share, default=None)
    if top is None or top.lost_share <= 0:
        return []
    return [Finding(
        key="pareto", severity="high" if top.lost_share >= 0.25 else "medium",
        title=f"{dim_word(a, top.id)} — המקור הגדול ביותר לאובדן נקודות",
        text=f"הממד אחראי ל־{pct(top.lost_share, 0)} מכלל הנקודות שאבדו "
             f"({num(top.lost_mean)} נק' לשיחה בממוצע, במשקל {pct(top.weight, 0)} מהמדד). "
             f"{pct(top.low_rate, 0)} מהשיחות קיבלו בו ציון ⟦1–2⟧.",
        meaning="שיפור בממד זה ישפיע על המדד הכולל יותר מכל שיפור נקודתי אחר.",
        section="sec-dims", weight=top.lost_share * 100,
        filter={"dim": top.id, "max": 2})]


def _dimension_findings(a: Analysis) -> list[Finding]:
    scored = [d for d in a.dims if d.mean.mean is not None]
    if len(scored) < 2:
        return []
    out = []
    pareto_top = max(a.dims, key=lambda d: d.lost_share).id
    weakest = min(scored, key=lambda d: d.mean.mean)
    if weakest.id != pareto_top and weakest.low_rate >= 0.15:
        out.append(Finding(
            key=f"weak-{weakest.id}", severity="medium",
            title=f"הממד החלש ביותר: {dim_word(a, weakest.id)}",
            text=f"ציון ממוצע {num(weakest.mean.mean)} מתוך ⟦5⟧; {pct(weakest.low_rate, 0)} "
                 f"מהשיחות קיבלו ⟦1–2⟧.",
            meaning="פער התנהגותי רחב — מתאים לנושא מרכזי בתוכנית ההדרכה הבאה.",
            section="sec-dims", weight=weakest.low_rate * 50,
            filter={"dim": weakest.id, "max": 2}))
    strongest = max(scored, key=lambda d: d.mean.mean)
    if strongest.high_rate >= 0.6:
        out.append(Finding(
            key=f"strong-{strongest.id}", severity="positive",
            title=f"חוזקה בולטת: {dim_word(a, strongest.id)}",
            text=f"{pct(strongest.high_rate, 0)} מהשיחות ברמה ⟦4–5⟧ בממד זה "
                 f"(ממוצע {num(strongest.mean.mean)}).",
            meaning="התנהגות מבוססת שכדאי לשמר ולהציג כדוגמה בהדרכות.",
            section="sec-dims", weight=strongest.high_rate * 10,
            filter={"dim": strongest.id, "min": 4}))
    return out


def _trend_findings(a: Analysis) -> list[Finding]:
    t = a.trend
    out = []
    if t is not None and t.slope_per_30d is not None and t.significant \
            and abs(t.slope_per_30d) >= 1:
        up = t.slope_per_30d > 0
        out.append(Finding(
            key="trend", severity="positive" if up else "high",
            title=("המדד משתפר לאורך התקופה" if up else "המדד יורד לאורך התקופה"),
            text=f"שינוי של {signed(t.slope_per_30d)} נק' בחודש בממוצע "
                 f"(רווח סמך 95%: ⟦{n1(t.low)} עד {n1(t.high)}⟧), על פני {count(t.n)} שיחות.",
            meaning=("מגמה עקבית — כדאי לזהות מה השתנה ולהרחיב אותו." if up else
                     "מגמה עקבית של הידרדרות — מומלץ לבדוק שינויים בנהלים, בעומס או בצוות."),
            section="sec-trend", weight=abs(t.slope_per_30d) * 3))
    d = a.last_vs_prev
    if d is not None and d.diff is not None and d.significant \
            and abs(d.diff) >= MIN_PRACTICAL_DIFF and len(a.periods) >= 2:
        up = d.diff > 0
        out.append(Finding(
            key="last-period", severity="positive" if up else "medium",
            title=("התקופה האחרונה טובה מקודמתה" if up else "התקופה האחרונה חלשה מקודמתה"),
            text=f"בתקופה {num_label(a.periods[-1].label)} המדד {'גבוה' if up else 'נמוך'} "
                 f"ב־{num(abs(d.diff))} נק' מהתקופה {num_label(a.periods[-2].label)}.",
            meaning="שינוי חד בין תקופות סמוכות — כדאי לבדוק אם הוא נמשך.",
            section="sec-trend", weight=abs(d.diff)))
    return out


def num_label(label: str) -> str:
    return f"⟦{clean(label)}⟧"


def _segment_findings(a: Analysis) -> list[Finding]:
    out = []
    names_he = {"call_type": "שיחות", "duration": "שיחות באורך", "layout": "הקלטות"}
    for kind, groups in a.segments.items():
        for g in groups:
            if g.flag not in ("bad", "ok") or g.diff is None or g.diff.diff is None:
                continue
            # With two groups "A vs the rest" and "B vs the rest" are one
            # comparison; say it once, from the side that needs attention.
            if len(groups) == 2 and g.flag == "ok" and any(x.flag == "bad" for x in groups):
                continue
            bad = g.flag == "bad"
            lead = f"{names_he[kind]} {_quoted(g.label)}"
            out.append(Finding(
                key=f"seg-{kind}-{g.key}", severity="high" if bad and kind == "call_type"
                else ("medium" if bad else "positive"),
                title=f"{lead} {'נמוכות' if bad else 'גבוהות'} ב־{num(abs(g.diff.diff))} נק' "
                      f"מיתר השיחות",
                text=f"ממוצע {num(g.mean.mean)} ב־{count(g.n)} שיחות"
                     + (f"; הממד החלש בהן: {dim_word(a, g.weakest)}" if g.weakest else "")
                     + (f"; כשל שער ב־{pct(g.gate.p)}" if g.gate.p else "") + ".",
                meaning=("פער מובהק (רמת ביטחון ⟦99%⟧) — כדאי לבחון הדרכה ייעודית או תסריט "
                         "מותאם לסוג שיחה זה." if bad else
                         "ביצועים עודפים מובהקים — אפשר ללמוד מהם לשאר סוגי השיחות."),
                section="sec-segments", weight=abs(g.diff.diff), filter=g.filter))
    return out


def _quoted(label: str) -> str:
    return f"„{clean(label)}”"


def _banker_findings(a: Analysis) -> list[Finding]:
    ranked = [b for b in a.bankers if b.flag != "few"]
    out = []
    bad = [b for b in ranked if b.flag == "bad"]
    ok = [b for b in ranked if b.flag == "ok"]
    if bad:
        gaps = sorted(-b.diff.diff for b in bad)
        gap = (f"פער של {num(gaps[0])} נק'" if len(gaps) == 1 else
               f"פער של ⟦{n1(gaps[0])}–{n1(gaps[-1])}⟧ נק'")
        out.append(Finding(
            key="bankers-bad", severity="high",
            title=("בנקאי אחד נמוך באופן מובהק משאר הבנקאים" if len(bad) == 1 else
                   f"{count(len(bad))} בנקאים נמוכים באופן מובהק משאר הבנקאים"),
            text=f"{names([b.key for b in bad])} — {gap} מתחת לשאר הבנקאים "
                 f"(רמת ביטחון ⟦99%⟧, לפחות ⟦8⟧ שיחות לכל אחד).",
            meaning="פער אישי עקבי ולא מקרי — מתאים לליווי אישי ממוקד בממד החלש של כל בנקאי.",
            section="sec-bankers", weight=len(bad) * 5,
            filter={"bankers": [b.key for b in bad]}))
    if ok:
        out.append(Finding(
            key="bankers-ok", severity="positive",
            title=("בנקאי אחד בולט לטובה" if len(ok) == 1 else
                   f"{count(len(ok))} בנקאים בולטים לטובה"),
            text=f"{names([b.key for b in ok])} — "
                 + ("גבוה" if len(ok) == 1 else "גבוהים") + " באופן מובהק משאר הבנקאים.",
            meaning="מקור לשיתוף ידע: הקלטות שלהם יכולות לשמש דוגמה בהדרכות עמיתים.",
            section="sec-bankers", weight=len(ok) * 2,
            filter={"bankers": [b.key for b in ok]}))
    means = sorted(b.mean.mean for b in ranked if b.mean.mean is not None)
    if len(means) >= 8:
        k = max(1, len(means) // 4)
        spread = sum(means[-k:]) / k - sum(means[:k]) / k
        if spread >= 10:
            out.append(Finding(
                key="bankers-spread", severity="medium",
                title=f"פער של {num(spread)} נק' בין הרבעון העליון לתחתון של הבנקאים",
                text=f"בנקאי ברבעון העליון מקבל בממוצע {num(sum(means[-k:]) / k)}, "
                     f"וברבעון התחתון {num(sum(means[:k]) / k)}.",
                meaning="השונות בין בנקאים גדולה — סטנדרטיזציה של שיטות העבודה יכולה להעלות "
                        "את המדד הכולל.",
                section="sec-bankers", weight=spread / 2))
    return out


DRIVER_MEANING = {
    "talk_ratio": "בנקאי שמדבר רוב השיחה מקשיב פחות — כדאי לעודד שאלות פתוחות והקשבה.",
    "interruptions": "קטיעת הלקוח פוגעת בתחושת ההקשבה — מומלץ לתרגל המתנה לסיום דברי הלקוח.",
    "patience": "הזמן שהבנקאי ממתין לפני שהוא עונה קשור לאיכות השיחה.",
    "questions": "שאלות בירור מאפשרות התאמת מענה — מומלץ לעגן שאלות חובה בתסריט.",
    "monologue": "הסברים ארוכים ברצף מקשים על הלקוח — מומלץ לפרק הסברים לשלבים עם בדיקת הבנה.",
    "dead_air": "שקט ממושך מעיד על חוסר זמינות מידע — כדאי לבדוק כלי עבודה ונגישות מידע.",
    "duration": "משך השיחה קשור לאיכותה — כדאי לבדוק את מקור השיחות הארוכות.",
}


def _driver_findings(a: Analysis) -> list[Finding]:
    out = []
    for d in [d for d in a.drivers if d.reportable][:2]:
        out.append(_driver_finding(d))
    return out


def _driver_finding(d: Driver) -> Finding:
    diff = d.contrast.diff
    higher = diff > 0
    return Finding(
        key=f"driver-{d.key}", severity="medium",
        title=f"{clean(d.label)}: {'קשור לציון גבוה יותר' if higher else 'קשור לציון נמוך יותר'}",
        text=f"כשהערך גבוה (⟦{n1(d.q3)}⟧ {clean(d.unit)} ומעלה) המדד הממוצע "
             f"{num(d.high_mean)}, וכשהוא נמוך (עד ⟦{n1(d.q1)}⟧ {clean(d.unit)}) — "
             f"{num(d.low_mean)}: הפרש של {signed(diff)} נק' (ρ={num(d.corr.rho)}, "
             f"{count(d.n)} שיחות).",
        meaning=DRIVER_MEANING.get(d.key, "") + " מדובר בקשר סטטיסטי, לא בהכרח סיבתי.",
        section="sec-drivers", weight=abs(diff), filter=d.filter_high)


def _coverage_findings(a: Analysis) -> list[Finding]:
    out = []
    cov = a.coverage
    if cov["redaction_disabled"]:
        out.append(Finding(
            key="redaction-off", severity="critical",
            title="הסתרת הפרטים כובתה " + ("בשיחה אחת" if cov["redaction_disabled"] == 1
                                            else f"ב־{count(cov['redaction_disabled'])} שיחות"),
            text="בשיחות אלה התמליל ודוח השיחה מכילים פרטים מזהים של לקוחות. בדוח זה מוצגים "
                 "עבורן מספרים בלבד.",
            meaning="חריגה מנוהל הגנת הפרטיות — יש לבדוק מדוע כובתה ההסתרה.",
            section="sec-coverage", weight=100))
    if a.review_rate.p and a.review_rate.p >= 0.05:
        out.append(Finding(
            key="review-rate", severity="high" if a.review_rate.p >= 0.15 else "medium",
            title=f"{pct(a.review_rate.p)} מהשיחות ממתינות לבדיקה אנושית",
            text=(("שיחה אחת לא נוקדה אוטומטית ואינה כלולה" if a.n_held == 1 else
                   f"{count(a.n_held)} שיחות לא נוקדו אוטומטית ואינן כלולות") + " במדד. ")
                 + (f"הסיבה העיקרית: {clean(cov['held_reasons'][0][0])}."
                    if cov["held_reasons"] else ""),
            meaning="כל עוד השיחות לא נבדקו, המדד מייצג רק את השיחות שנוקדו.",
            section="sec-coverage", weight=a.review_rate.p * 40,
            filter={"status": "needs_human_review"}))
    if a.n_scope and a.n_failed / a.n_scope >= 0.02:
        out.append(Finding(
            key="failed", severity="medium",
            title=("שיחה אחת לא עובדה" if a.n_failed == 1 else
                   f"{count(a.n_failed)} שיחות לא עובדו"),
            text="השיחות נכשלו בשלב טכני ואינן כלולות בדוח. פירוט השלבים בפרק הכיסוי.",
            meaning="כדאי לבדוק את איכות ההקלטות ואת תקינות התהליך.",
            section="sec-coverage", weight=a.n_failed / a.n_scope * 30,
            filter={"status": "failed"}))
    cal = cov.get("calibration")
    if a.n_scored and (not cal or not cal.get("overall_pass")):
        out.append(Finding(
            key="calibration", severity="info",
            title="הציונים טרם אומתו מול מעריכים אנושיים" if not cal else
                  "הכיול מול מעריכים אנושיים טרם עבר",
            text="מדדים קבוצתיים (ממוצעים, מגמות, פילוחים) אמינים גם כך; ציון של שיחה בודדת "
                 "ושל בנקאי עם מעט שיחות — פחות.",
            meaning="אין להשתמש בדוח זה כבסיס יחיד להערכת עובד.",
            section="sec-coverage", weight=1))
    return out


# -- actions --------------------------------------------------------------------

OWNER_GATE = "ציות ונהלים, בשיתוף ראשי הצוותים"
OWNER_TRAINING = "הדרכה ופיתוח מקצועי"
OWNER_MANAGERS = "ראשי צוותים"


def _actions(a: Analysis, recs: dict[str, list[str]]) -> list[Action]:
    if a.n_scored == 0:
        return []
    candidates: list[Action] = []
    for d in a.dims:
        if d.gain_if_floor3 <= 0.05 and d.gates_removed == 0:
            continue
        lows = d.counts[0] + d.counts[1]
        focus = [b for b, _, _ in d.focus_bankers]
        text = f"{count(lows)} שיחות קיבלו ⟦1–2⟧ בממד זה"
        if focus:
            share = sum(k for _, k, _ in d.focus_bankers) / lows if lows else 0
            text += f"; {pct(share, 0)} מהן אצל {names(focus)}"
        text += "."
        impact = f"העלאת כל הציונים הנמוכים ל־⟦3⟧ תוסיף למדד {signed(d.gain_if_floor3)} נק'"
        if d.gates_removed:
            impact += f" ותבטל {count(d.gates_removed)} כשלי שער"
        # Gate failures are a regulatory exposure: they rank above a larger
        # but purely qualitative gain.
        score = d.gain_if_floor3 + (d.gates_removed / a.n_scored) * 100 * (2 if d.gate else 0)
        coaching = (recs.get(d.id) or [None])[0]
        candidates.append(Action(
            title=(f"ריענון נוהל {dim_word(a, d.id)}" if d.gate
                   else f"תוכנית חיזוק ל{dim_word(a, d.id)}"),
            text=text, impact=impact + ".",
            owner=OWNER_GATE if d.gate else OWNER_TRAINING,
            section="sec-risk" if d.gate else "sec-dims", score=score,
            filter={"dim": d.id, "max": 2}, coaching=coaching))
    bad = [b for b in a.bankers if b.flag == "bad"]
    if bad and a.median is not None:
        gain = sum(b.n * max(0.0, a.median - (b.mean.mean or 0)) for b in bad) / a.n_scored
        candidates.append(Action(
            title=(f"ליווי אישי לבנקאי {names([bad[0].key])}" if len(bad) == 1 else
                   f"ליווי אישי ל־{count(len(bad))} בנקאים"),
            text=f"{names([b.key for b in bad])}: לכל אחד — משוב על הממד החלש שלו, עם הקלטות "
                 "מהדוח (רמה 3) כבסיס לשיחת המשוב.",
            impact=("הגעה של הבנקאי" if len(bad) == 1 else "הגעה של בנקאים אלה")
            + f" לחציון תוסיף למדד {signed(gain)} נק'.",
            owner=OWNER_MANAGERS, section="sec-bankers", score=gain,
            filter={"bankers": [b.key for b in bad]}))
    candidates.sort(key=lambda x: -x.score)
    picked: list[Action] = []
    seen_sections: dict[str, int] = {}
    for c in candidates:
        # At most two actions from the same area, so the plan is not three
        # variations of one idea.
        if seen_sections.get(c.section, 0) >= 2:
            continue
        picked.append(c)
        seen_sections[c.section] = seen_sections.get(c.section, 0) + 1
        if len(picked) == 3:
            break
    return picked
