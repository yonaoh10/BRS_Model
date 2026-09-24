"""The professional opinion: findings and actions, written from the numbers.

Every sentence here is generated from a computed statistic by a fixed rule -
no language model writes the management summary - so every number in it can
be traced to a chart, and the same batch always yields the same opinion.

Numbers are marked ⟦like this⟧ so the template can isolate them for
right-to-left layout. A marker holds numbers only - a Hebrew word inside one
would be laid out left to right with them - and data values put into a
sentence are stripped of the marker characters first.

The wording follows three rules: say "one call" and not "1 calls"; claim no
more than the test showed (a significant gap is not "consistent and not
random"); and never give a firm verdict on a batch too small to carry one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from callqa.reporting.common import load_recommendations
from callqa.reporting.executive import numfmt
from callqa.reporting.executive.analysis import (
    BAND_LABEL,
    MIN_GROUP_N,
    MIN_PRACTICAL_DIFF,
    Analysis,
    Driver,
    band_of,
)

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "positive": 3, "info": 4}
SEVERITY_HE = {"critical": "קריטי", "high": "גבוה", "medium": "בינוני",
               "positive": "חיובי", "info": "לידיעה"}
MIN_FIRM_N = 30          # below this the opinion is labelled preliminary
PERIOD_WORDS = {         # (this period, the one before, "the last period" phrase)
    "day": ("ביום", "ביום שקדם לו", "היום האחרון"),
    "week": ("בשבוע שהחל ב־", "בשבוע שקדם לו", "השבוע האחרון"),
    "month": ("בחודש", "בחודש שקדם לו", "החודש האחרון"),
}


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
    bottom_line: list[str]                # 2-6 sentences
    findings: list[Finding]               # all, ranked
    top_findings: list[Finding]           # the executive five
    actions: list[Action]                 # up to three
    by_section: dict[str, list[Finding]] = field(default_factory=dict)


# -- formatting -----------------------------------------------------------------

def clean(value: object) -> str:
    """A data value about to go into a generated sentence."""
    return str(value).replace("⟦", "").replace("⟧", "")


def num(x: float | None, digits: int = 1) -> str:
    return f"⟦{numfmt.fmt(x, digits)}⟧"


def mean5(x: float | None) -> str:
    """A 1-5 mean, always with two decimals, as the tables show it."""
    return f"⟦{numfmt.fixed(x, 2)}⟧"


def pct(p: float | None, digits: int = 1) -> str:
    return f"⟦{numfmt.percent(p, digits)}⟧"


def signed(x: float | None) -> str:
    return f"⟦{numfmt.signed(x)}⟧"


def count(k: int) -> str:
    return f"⟦{k:,}⟧"


def several(k: int, one: str, many: str) -> str:
    """'בנקאי אחד' / '⟦3⟧ בנקאים': Hebrew counts one differently."""
    return one if k == 1 else f"{count(k)} {many}"


def names(items: list[str], limit: int = 5) -> str:
    shown = [f"⟦{clean(i)}⟧" for i in items[:limit]]
    rest = len(items) - limit
    if rest > 0:
        return ", ".join(shown) + f" ועוד {count(rest)}"
    if len(shown) <= 1:
        return "".join(shown)
    # A maqaf joins the vav to a Latin id ("ו־B004"), which Hebrew typesetting
    # would otherwise glue on as "וB004".
    return ", ".join(shown[:-1]) + " ו־" + shown[-1]


def dim_word(analysis: Analysis, dim_id: str) -> str:
    return f"„{clean(analysis.dim_names.get(dim_id, dim_id))}”"


def gates_text(analysis: Analysis) -> str:
    gates = [f"„{clean(d.name)}”" for d in analysis.dims if d.gate]
    return " או ".join(gates) if gates else "שערי החובה"


def _range(low: float | None, high: float | None, lo: float, hi: float) -> str:
    """A confidence interval clamped to the scale it lives on."""
    if low is None or high is None:
        return ""
    return f"⟦{numfmt.fmt(max(lo, low))}–{numfmt.fmt(min(hi, high))}⟧"


def trend_change(analysis: Analysis) -> float | None:
    """The trend's change over the whole span, in index points - the practical
    size of a trend is what it adds up to, not its monthly rate."""
    t = analysis.trend
    if t is None or t.slope_per_30d is None:
        return None
    return t.slope_per_30d * t.span_days / 30.0


# -- the opinion ----------------------------------------------------------------

def assess(analysis: Analysis) -> str:
    a = analysis
    if a.n_scored == 0 or a.total.mean is None:
        return "אין בתקופה זו שיחות שקיבלו ציון"
    fails = round((a.gate_rate.p or 0) * a.n_scored)
    if a.n_scored < MIN_FIRM_N:
        return (f"הערכה ראשונית בלבד — מבוססת על {several(a.n_scored, 'שיחה אחת', 'שיחות')}"
                + ("; נמצא כשל בשער חובה" if fails else ""))
    verdict = {4: "רמת שירות גבוהה", 3: "רמת שירות בינונית",
               2: "רמת שירות גבולית — נדרש מיקוד בשיפור",
               1: "רמת שירות נמוכה — נדרשת התערבות"}[band_of(a.total.mean)]
    change = trend_change(a)
    if a.trend is not None and a.trend.significant and change is not None \
            and abs(change) >= MIN_PRACTICAL_DIFF:
        verdict += ", במגמת שיפור" if change > 0 else ", במגמת ירידה"
    # By the lower confidence bound: "material" should hold even if the rate
    # is at the low end of what the data allow.
    low = a.gate_rate.low or 0.0
    if low >= 0.05:
        verdict += "; סיכון ציות מהותי"
    elif low >= 0.02 or (a.gate_rate.p or 0) >= 0.05:
        verdict += "; סיכון ציות הדורש מעקב"
    return verdict


def bottom_line(analysis: Analysis, actions: list[Action]) -> list[str]:
    a = analysis
    if a.n_scored == 0:
        return ["לא נמצאו בתקופה זו שיחות שקיבלו ציון, ולכן אין מדד איכות להצגה. "
                "פירוט מצב השיחות מופיע בפרק „כיסוי, איכות הנתונים ומקור”."]
    lines = []
    band = BAND_LABEL[band_of(a.total.mean)]
    ci = _range(a.total.low, a.total.high, 0, 100)
    lines.append(
        f"מדד האיכות הממוצע עומד על {num(a.total.mean)} מתוך ⟦100⟧"
        + (f" (רווח סמך 95%: {ci})" if ci else "")
        + f" — בטווח „{band}”, על בסיס {count(a.n_scored)} מתוך {count(a.n_scope)} השיחות "
        f"בתקופה. {pct(a.high_rate.p)} מהשיחות ברמה גבוהה (⟦80⟧ ומעלה) ו־{pct(a.low_rate.p)} "
        "מתחת ל־⟦60⟧.")
    if a.n_scored < MIN_FIRM_N:
        lines.append("מספר השיחות קטן מכדי לקבוע רמת שירות; המספרים מוצגים לידיעה בלבד.")
    fails = round((a.gate_rate.p or 0) * a.n_scored)
    if fails:
        one = fails == 1
        sentence = (("שיחה אחת" if one else f"{count(fails)} שיחות") + f" ({pct(a.gate_rate.p)}) "
                    + ("נכשלה" if one else "נכשלו") + f" בשער חובה — {gates_text(a)}; "
                    + ("בה" if one else "בהן") + " המדד מוגבל ל־⟦59⟧ לכל היותר.")
        if (a.gate_rate.p or 0) >= 0.05:
            sentence += " זהו הסיכון המרכזי בדוח, והוא מחייב טיפול מיידי."
        else:
            sentence += " יש לבחון " + ("אותה" if one else "אותן") + " פרטנית."
        lines.append(sentence)
    else:
        lines.append(f"לא נמצאו כשלים בשערי החובה ({gates_text(a)}).")
    lines.extend(_trend_sentence(a))
    ranked = sorted(a.dims, key=lambda d: -d.lost_share_with_gate)
    top = [d for d in ranked[:2] if d.lost_share_with_gate > 0]
    if len(top) == 2:
        share = top[0].lost_share_with_gate + top[1].lost_share_with_gate
        lines.append(f"שני המקורות הגדולים לפער בין המדד ל־⟦100⟧ הם {dim_word(a, top[0].id)} "
                     f"ו־{dim_word(a, top[1].id)} (יחד {pct(share, 0)} מהנקודות שאבדו"
                     + (", כולל ההגבלה בכשלי שער" if any(d.gate for d in top)
                        and a.gate_penalty_mean > 0 else "") + ").")
    elif top:
        lines.append(f"המקור הגדול ביותר לפער בין המדד ל־⟦100⟧ הוא {dim_word(a, top[0].id)} "
                     f"({pct(top[0].lost_share_with_gate, 0)} מהנקודות שאבדו).")
    if actions and a.n_scored >= MIN_FIRM_N:
        first = actions[0]
        lines.append(f"הפעולה בעלת ההשפעה הגדולה ביותר: {first.title} — {first.impact}")
    if a.review_rate.p and a.review_rate.p >= 0.05:
        lines.append(f"{pct(a.review_rate.p)} מהשיחות בתקופה ממתינות לבדיקה אנושית ואינן "
                     "כלולות במדד.")
    return lines


def _trend_sentence(a: Analysis) -> list[str]:
    t = a.trend
    if t is None or t.slope_per_30d is None:
        return []
    basis = " (לפי תאריך העיבוד, כי תאריכי השיחות חסרים)" if a.date_basis == "processed" else ""
    change = trend_change(a) or 0.0
    slope = t.slope_per_30d
    if t.significant and abs(change) >= MIN_PRACTICAL_DIFF:
        verb = "משתפר" if slope > 0 else "יורד"
        return [f"המדד {verb} בקצב מובהק של {num(abs(slope))} נקודות בחודש — "
                f"כ־{num(abs(change))} נקודות לאורך התקופה{basis}."]
    if t.significant:
        return [f"נרשם שינוי מובהק אך קטן לאורך התקופה: {signed(change)} נקודות{basis}."]
    if t.low is not None and t.high is not None and max(abs(t.low), abs(t.high)) < 1:
        return [f"המדד יציב לאורך התקופה{basis}."]
    return [f"לא זוהתה מגמה מובהקת לאורך התקופה{basis}."]


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
    actions = _actions(analysis, recs)
    return Opinion(
        assessment=assess(analysis), bottom_line=bottom_line(analysis, actions),
        findings=findings, top_findings=_executive_five(findings), actions=actions,
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
            text=f"בכל {several(a.n_scored, 'השיחה', 'השיחות')} בוצעו זיהוי הלקוח והגילוי "
                 "הנאות ברמה העומדת בסף.",
            meaning="אין בתקופה זו ממצא רגולטורי בתחומי הזיהוי והגילוי הנאות.",
            section="sec-risk", weight=1.0)]
    rate = a.gate_rate.p or 0
    parts = [f"{dim_word(a, g.id)} — {count(g.fails)} ({pct(g.rate.p)})"
             for g in a.gates if g.fails]
    text = f"פירוט: {'; '.join(parts)}."
    conc = [g for g in a.gates if g.concentration is not None]
    if conc:
        g = max(conc, key=lambda g: g.concentration)
        text += (f" ב{dim_word(a, g.id)} הכשלים מרוכזים: {pct(g.concentration, 0)} מהם אצל "
                 f"{several(g.top_n, 'בנקאי אחד', 'בנקאים')} מתוך {count(g.n_bankers)}, "
                 f"שמטפלים ב־{pct(g.call_share, 0)} מהשיחות.")
    one = fails == 1
    return [Finding(
        key="gates",
        severity="critical" if rate >= 0.05 else ("high" if rate >= 0.02 else "medium"),
        title=(f"שיחה אחת ({pct(rate)}) נכשלה בשער חובה" if one else
               f"{count(fails)} שיחות ({pct(rate)}) נכשלו בשער חובה"),
        text=text,
        meaning="כשל שער הוא חשיפה רגולטורית — מסירת מידע ללא זיהוי מלא או ללא גילוי "
                "נאות — ולא רק שאלה של איכות שירות.",
        section="sec-risk", weight=rate * 100, filter={"gate": "any"})]


def _points_findings(a: Analysis) -> list[Finding]:
    if a.n_scored == 0:
        return []
    top = max(a.dims, key=lambda d: d.lost_share_with_gate, default=None)
    if top is None or top.lost_share_with_gate <= 0:
        return []
    caused = top.lost_with_gate - top.lost_mean
    return [Finding(
        key="pareto", severity="high" if top.lost_share_with_gate >= 0.25 else "medium",
        title=f"{dim_word(a, top.id)} — המקור הגדול ביותר לאובדן נקודות",
        text=f"הממד אחראי ל־{pct(top.lost_share_with_gate, 0)} מכלל הנקודות שאבדו "
             f"({num(top.lost_with_gate)} נקודות לשיחה בממוצע"
             + (f", מתוכן {num(caused)} בגלל הגבלת המדד בשיחות שנכשלו בו בשער" if caused > 0.05
                else "")
             + f"; משקלו במדד {pct(top.weight, 0)}). {pct(top.low_rate, 0)} מהשיחות קיבלו בו "
             "ציון ⟦1–2⟧.",
        meaning="שיפור בממד זה ישפיע על המדד הכולל יותר מכל שיפור אחר.",
        section="sec-dims", weight=top.lost_share_with_gate * 100,
        filter={"dim": top.id, "max": 2})]


def _dimension_findings(a: Analysis) -> list[Finding]:
    scored = [d for d in a.dims if d.mean.mean is not None]
    if len(scored) < 2:
        return []
    out = []
    pareto_top = max(a.dims, key=lambda d: d.lost_share_with_gate).id
    weakest = min(scored, key=lambda d: d.mean.mean)
    if weakest.id != pareto_top and weakest.low_rate >= 0.15:
        out.append(Finding(
            key=f"weak-{weakest.id}", severity="medium",
            title=f"הממד החלש ביותר: {dim_word(a, weakest.id)}",
            text=f"ציון ממוצע {mean5(weakest.mean.mean)} מתוך ⟦5⟧; {pct(weakest.low_rate, 0)} "
                 "מהשיחות קיבלו בו ציון ⟦1–2⟧.",
            meaning="פער התנהגותי רחב — נושא מתאים למוקד בתוכנית ההדרכה הבאה.",
            section="sec-dims", weight=weakest.low_rate * 50,
            filter={"dim": weakest.id, "max": 2}))
    strongest = max(scored, key=lambda d: d.mean.mean)
    if strongest.high_rate >= 0.6:
        out.append(Finding(
            key=f"strong-{strongest.id}", severity="positive",
            title=f"חוזקה בולטת: {dim_word(a, strongest.id)}",
            text=f"{pct(strongest.high_rate, 0)} מהשיחות קיבלו ציון ⟦4–5⟧ בממד זה "
                 f"(ממוצע {mean5(strongest.mean.mean)}).",
            meaning="התנהגות מבוססת — כדאי לשמר אותה ולהציג אותה כדוגמה בהדרכות.",
            section="sec-dims", weight=strongest.high_rate * 10,
            filter={"dim": strongest.id, "min": 4}))
    return out


def _trend_findings(a: Analysis) -> list[Finding]:
    t = a.trend
    out = []
    change = trend_change(a)
    if t is not None and t.slope_per_30d is not None and t.significant \
            and change is not None and abs(change) >= MIN_PRACTICAL_DIFF:
        up = t.slope_per_30d > 0
        out.append(Finding(
            key="trend", severity="positive" if up else "high",
            title="המדד משתפר לאורך התקופה" if up else "המדד יורד לאורך התקופה",
            text=f"שינוי של {signed(t.slope_per_30d)} נקודות בחודש בממוצע (רווח סמך 95%: "
                 f"{signed(t.low)} עד {signed(t.high)}), כ־{signed(change)} נקודות על פני "
                 f"התקופה, על בסיס {count(t.n)} שיחות.",
            meaning=("מגמה עקבית — כדאי לזהות מה השתנה ולהרחיב את זה." if up else
                     "הידרדרות עקבית — מומלץ לבדוק שינויים בנהלים, בעומס או בצוות."),
            section="sec-trend", weight=abs(change)))
    d = a.last_vs_prev
    words = PERIOD_WORDS.get(a.granularity or "")
    if d is not None and d.diff is not None and d.significant and words \
            and abs(d.diff) >= MIN_PRACTICAL_DIFF and len(a.periods) >= 2:
        up = d.diff > 0
        this, before, the_last = words
        out.append(Finding(
            key="last-period", severity="positive" if up else "medium",
            title=f"{the_last} {'טוב' if up else 'חלש'} מקודמו",
            text=f"{this}⟦{clean(a.periods[-1].label)}⟧ המדד {'גבוה' if up else 'נמוך'} "
                 f"ב־{num(abs(d.diff))} נקודות מאשר {before} (⟦{clean(a.periods[-2].label)}⟧).",
            meaning="שינוי חד בין תקופות סמוכות, וייתכן שהתקופה האחרונה חלקית — כדאי לבדוק "
                    "אם הוא נמשך.",
            section="sec-trend", weight=abs(d.diff)))
    return out


LAYOUT_SHORT = {"mono": "חד־ערוציות", "stereo": "דו־ערוציות"}


def _segment_findings(a: Analysis) -> list[Finding]:
    out = []
    for kind, groups in a.segments.items():
        ranked = [x for x in groups if x.flag != "few"]
        for g in groups:
            if g.flag not in ("bad", "ok") or g.diff is None or g.diff.diff is None:
                continue
            # With two compared groups, "A vs the rest" and "B vs the rest" are
            # one comparison; say it once, from the side that needs attention.
            if len(ranked) == 2 and g.flag == "ok" and any(x.flag == "bad" for x in ranked):
                continue
            bad = g.flag == "bad"
            gap = f"ב־{num(abs(g.diff.diff))} נקודות"
            detail = (f"ממוצע {num(g.mean.mean)} ב־{several(g.n, 'שיחה אחת', 'שיחות')}"
                      + (f"; הממד החלש בהן: {dim_word(a, g.weakest)}" if g.weakest else "")
                      + (f"; כשל שער ב־{pct(g.gate.p)}" if g.gate.p else "") + ".")
            if kind == "layout":
                out.append(Finding(
                    key=f"seg-layout-{g.key}", severity="info",
                    title=f"הקלטות {LAYOUT_SHORT.get(g.key, clean(g.label))} "
                          f"{'נמוכות' if bad else 'גבוהות'} {gap} משאר ההקלטות",
                    text=detail,
                    meaning="ייתכן שהפער נובע מזיהוי הדוברים האוטומטי בהקלטה חד־ערוצית ולא "
                            "מהתנהגות הבנקאים — יש לבדוק את איכות ההקלטות לפני הסקת מסקנות.",
                    section="sec-segments", weight=abs(g.diff.diff), filter=g.filter))
                continue
            if kind == "duration":
                title = (f"שיחות באורך {clean(g.label)} {'נמוכות' if bad else 'גבוהות'} {gap} "
                         "משאר השיחות")
                meaning = ("כדאי לבדוק אילו פניות מגיעות לשיחות באורך זה ומה מקשה עליהן."
                           if bad else "כדאי לבדוק מה מאפיין שיחות באורך זה.")
                severity = "medium" if bad else "positive"
            else:
                title = (f"שיחות {_quoted(g.label)} {'נמוכות' if bad else 'גבוהות'} {gap} "
                         "משאר השיחות")
                meaning = ("פער מובהק (רמת ביטחון ⟦99%⟧) — כדאי לבחון הדרכה ייעודית או תסריט "
                           "מותאם לסוג שיחה זה." if bad else
                           "ביצועים טובים מהממוצע באופן מובהק — כדאי לזהות מה עובד בסוג "
                           "שיחה זה וליישם אותו בסוגים האחרים.")
                severity = "high" if bad else "positive"
            out.append(Finding(key=f"seg-{kind}-{g.key}", severity=severity, title=title,
                               text=detail, meaning=meaning, section="sec-segments",
                               weight=abs(g.diff.diff), filter=g.filter))
    return out


def _quoted(label: str) -> str:
    return f"„{clean(label)}”"


def _banker_findings(a: Analysis) -> list[Finding]:
    ranked = [b for b in a.bankers if b.flag != "few"]
    out = []
    bad = [b for b in ranked if b.flag == "bad"]
    ok = [b for b in ranked if b.flag == "ok"]
    ref = (f" (⟦{numfmt.fmt(a.banker_reference)}⟧)" if a.banker_reference is not None else "")
    if bad:
        gaps = sorted(-b.diff.diff for b in bad)
        lo, hi = numfmt.fmt(gaps[0]), numfmt.fmt(gaps[-1])
        gap = f"פער של ⟦{lo}⟧ נקודות" if lo == hi else f"פער של ⟦{lo}–{hi}⟧ נקודות"
        one = len(bad) == 1
        out.append(Finding(
            key="bankers-bad", severity="high",
            title=("בנקאי אחד נמוך באופן מובהק מהבנקאי הטיפוסי" if one else
                   f"{count(len(bad))} בנקאים נמוכים באופן מובהק מהבנקאי הטיפוסי"),
            text=f"{names([b.key for b in bad])} — {gap} מתחת לחציון הבנקאים{ref}, ברמת "
                 f"ביטחון ⟦99%⟧ ועם ⟦{MIN_GROUP_N}⟧ שיחות לפחות "
                 + ("לבנקאי." if one else "לכל בנקאי."),
            meaning="הפער מובהק סטטיסטית. מומלץ לאמת אותו בהאזנה לשיחות לפני שיחת משוב, "
                    "ולהתמקד בממד החלש של כל בנקאי.",
            section="sec-bankers", weight=len(bad) * 5,
            filter={"bankers": [b.key for b in bad]}))
    if ok and not (len(ranked) == 2 and bad):
        one = len(ok) == 1
        out.append(Finding(
            key="bankers-ok", severity="positive",
            title="בנקאי אחד בולט לטובה" if one else f"{count(len(ok))} בנקאים בולטים לטובה",
            text=f"{names([b.key for b in ok])} — "
                 + ("גבוה" if one else "גבוהים") + f" באופן מובהק מחציון הבנקאים{ref}.",
            meaning="מקור לשיתוף ידע: שיחות שלהם יכולות לשמש דוגמה בהדרכות עמיתים.",
            section="sec-bankers", weight=len(ok) * 2,
            filter={"bankers": [b.key for b in ok]}))
    means = sorted(b.mean.mean for b in ranked if b.mean.mean is not None)
    if len(means) >= 8:
        k = max(1, len(means) // 4)
        top_mean, bottom_mean = sum(means[-k:]) / k, sum(means[:k]) / k
        spread = top_mean - bottom_mean
        if spread >= 10:
            out.append(Finding(
                key="bankers-spread", severity="medium",
                title=f"פער של {num(spread)} נקודות בין רבע הבנקאים החזקים לרבע החלש",
                text=f"ממוצע המדד ברבע הבנקאים החזקים הוא {num(top_mean)}, לעומת "
                     f"{num(bottom_mean)} ברבע החלש.",
                meaning="השונות בין הבנקאים גדולה — אחידות בשיטות העבודה יכולה להעלות את "
                        "המדד הכולל.",
                section="sec-bankers", weight=spread / 2))
    return out


DRIVER_MEANING = {
    "talk_ratio": "בנקאי שמדבר רוב השיחה מקשיב פחות — כדאי לעודד שאלות פתוחות והקשבה.",
    "interruptions": "קטיעת הלקוח פוגעת בתחושת ההקשבה — מומלץ לתרגל המתנה לסיום דברי הלקוח.",
    "patience": "הזמן שהבנקאי ממתין לפני שהוא עונה קשור לאיכות השיחה.",
    "questions": "שאלות בירור מאפשרות להתאים את המענה — מומלץ לעגן שאלות חובה בתסריט.",
    "monologue": "הסבר ארוך ברצף מקשה על הלקוח — מומלץ לפרק הסברים לשלבים ולוודא הבנה.",
    "dead_air": "שקט ממושך מעיד לרוב על קושי לאתר מידע — כדאי לבדוק את כלי העבודה ואת "
                "נגישות המידע.",
    "duration": "משך השיחה קשור לאיכותה — כדאי לבדוק מה מאפיין את השיחות הארוכות.",
}


def _driver_findings(a: Analysis) -> list[Finding]:
    return [_driver_finding(d) for d in [d for d in a.drivers if d.reportable][:2]]


def _value(v: float | None, unit: str) -> str:
    if unit == "%":
        return f"⟦{numfmt.fmt(v)}%⟧"
    return f"⟦{numfmt.fmt(v)}⟧ {clean(unit)}"


def _driver_finding(d: Driver) -> Finding:
    diff = d.contrast.diff
    higher = diff > 0
    return Finding(
        key=f"driver-{d.key}", severity="medium",
        title=f"{clean(d.label)}: {'קשור למדד גבוה יותר' if higher else 'קשור למדד נמוך יותר'}",
        text=f"בשיחות שבהן הערך גבוה ({_value(d.q3, d.unit)} ומעלה) המדד הממוצע הוא "
             f"{num(d.high_mean)}, ובשיחות שבהן הוא נמוך (עד {_value(d.q1, d.unit)}) — "
             f"{num(d.low_mean)}: הפרש של {signed(diff)} נקודות (מתאם {num(d.corr.rho, 2)}, "
             f"{count(d.n)} שיחות).",
        meaning=DRIVER_MEANING.get(d.key, "") + " מדובר בקשר סטטיסטי, לא בהכרח סיבתי.",
        section="sec-drivers", weight=abs(diff), filter=d.filter_high)


def _coverage_findings(a: Analysis) -> list[Finding]:
    out = []
    cov = a.coverage
    off = cov["redaction_disabled"]
    if off:
        one = off == 1
        out.append(Finding(
            key="redaction-off", severity="critical",
            title="הסתרת הפרטים כובתה " + ("בשיחה אחת" if one else f"ב־{count(off)} שיחות"),
            text=("בשיחה זו התמליל ודוח השיחה מכילים פרטים מזהים של לקוחות. בדוח זה מוצגים "
                  "עבורה מספרים בלבד." if one else
                  "בשיחות אלה התמליל ודוחות השיחה מכילים פרטים מזהים של לקוחות. בדוח זה "
                  "מוצגים עבורן מספרים בלבד."),
            meaning="חריגה מנוהל הגנת הפרטיות — יש לבדוק מדוע כובתה ההסתרה.",
            section="sec-coverage", weight=100))
    if a.review_rate.p and a.review_rate.p >= 0.05:
        one = a.n_held == 1
        out.append(Finding(
            key="review-rate", severity="high" if a.review_rate.p >= 0.15 else "medium",
            title=f"{pct(a.review_rate.p)} מהשיחות בתקופה ממתינות לבדיקה אנושית",
            text=("שיחה אחת לא קיבלה ציון אוטומטי ואינה כלולה" if one else
                  f"{count(a.n_held)} שיחות לא קיבלו ציון אוטומטי ואינן כלולות") + " במדד. "
                 + (f"הסיבה העיקרית: {clean(cov['held_reasons'][0][0])}."
                    if cov["held_reasons"] else ""),
            meaning="כל עוד השיחות לא נבדקו, המדד מייצג רק את השיחות שקיבלו ציון.",
            section="sec-coverage", weight=a.review_rate.p * 40,
            filter={"status": "needs_human_review"}))
    if a.n_scope and a.n_failed / a.n_scope >= 0.02:
        one = a.n_failed == 1
        out.append(Finding(
            key="failed", severity="medium",
            title="שיחה אחת לא עובדה" if one else f"{count(a.n_failed)} שיחות לא עובדו",
            text=("השיחה נכשלה בשלב טכני ואינה כלולה" if one else
                  "השיחות נכשלו בשלב טכני ואינן כלולות") + " בדוח. פירוט השלבים בפרק הכיסוי.",
            meaning="כדאי לבדוק את איכות ההקלטות ואת תקינות התהליך.",
            section="sec-coverage", weight=a.n_failed / a.n_scope * 30,
            filter={"status": "failed"}))
    cal = cov.get("calibration")
    if a.n_scored and (not cal or not cal.get("overall_pass")):
        out.append(Finding(
            key="calibration", severity="info",
            title=("הציונים טרם אומתו מול מעריכים אנושיים" if not cal else
                   "הכיול מול מעריכים אנושיים טרם עבר"),
            text="ממוצעים, מגמות ופילוחים רגישים פחות לטעות בשיחה בודדת, אך הטיה שיטתית של "
                 "מודל השיפוט אפשרית עד שיבוצע כיול מול מעריכים אנושיים. ציון של שיחה בודדת, "
                 "ושל בנקאי עם מעט שיחות, אמין עוד פחות.",
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
    n_bankers = len(a.bankers)
    candidates: list[Action] = []
    for d in a.dims:
        if d.gain_if_floor3 <= 0.05 and d.gates_removed == 0:
            continue
        lows = d.counts[0] + d.counts[1]
        text = several(lows, "שיחה אחת קיבלה", "שיחות קיבלו") + " ציון ⟦1–2⟧ בממד זה"
        focus = [b for b, _, _ in d.focus_bankers]
        # Name the bankers only when they are a minority that holds clearly
        # more of the low scores than of the calls - otherwise the sentence
        # would just restate the team.
        if focus and lows >= 5 and d.focus_share - d.focus_call_share >= 0.15:
            text += (f"; {pct(d.focus_share, 0)} מהן אצל {names(focus)} — "
                     f"{several(len(focus), 'בנקאי אחד', 'בנקאים')} מתוך {count(n_bankers)}, "
                     f"שמטפלים ב־{pct(d.focus_call_share, 0)} מהשיחות")
        text += "."
        impact = f"העלאת כל הציונים הנמוכים ל־⟦3⟧ תוסיף למדד {signed(d.gain_if_floor3)} נקודות"
        if d.gates_removed:
            impact += " ותבטל " + several(d.gates_removed, "כשל שער אחד", "כשלי שער")
        # Gate failures are a regulatory exposure: they rank above a larger
        # but purely qualitative gain.
        score = d.gain_if_floor3 + (d.gates_removed / a.n_scored) * 100 * (2 if d.gate else 0)
        candidates.append(Action(
            title=(f"ריענון נוהל {dim_word(a, d.id)}" if d.gate
                   else f"תוכנית חיזוק ל{dim_word(a, d.id)}"),
            text=text, impact=impact + ".",
            owner=OWNER_GATE if d.gate else OWNER_TRAINING,
            section="sec-risk" if d.gate else "sec-dims", score=score,
            filter={"dim": d.id, "max": 2}, coaching=(recs.get(d.id) or [None])[0]))
    bad = [b for b in a.bankers if b.flag == "bad"]
    ref = a.banker_reference
    if bad and ref is not None:
        gain = sum(b.n * max(0.0, ref - (b.mean.mean or 0)) for b in bad) / a.n_scored
        one = len(bad) == 1
        if one:
            b = bad[0]
            text = (f"משוב ממוקד על הממד החלש שלו, {dim_word(a, b.weakest)}, עם שיחות מהדוח "
                    "(רמה 3) כבסיס לשיחת המשוב." if b.weakest else
                    "משוב ממוקד על הממד החלש שלו, עם שיחות מהדוח (רמה 3) כבסיס לשיחת המשוב.")
        else:
            text = (f"{names([b.key for b in bad])}: לכל אחד — משוב ממוקד על הממד החלש שלו, "
                    "עם שיחות מהדוח (רמה 3) כבסיס לשיחת המשוב.")
        candidates.append(Action(
            title=(f"ליווי אישי לבנקאי {names([bad[0].key])}" if one else
                   f"ליווי אישי ל־{count(len(bad))} בנקאים"),
            text=text,
            impact=("הגעת הבנקאי" if one else "הגעת בנקאים אלה")
            + f" לחציון הבנקאים תוסיף למדד {signed(gain)} נקודות.",
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
