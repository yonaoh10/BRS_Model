"""The batch analysis: every number the journey report shows, with its meaning.

Each figure is a Metric carrying its count and base (k of n), a confidence
interval where one makes sense, a Hebrew definition, and the line "what would
make this wrong" - the report prints both next to the number. A rate on
fewer than `min_rate_n` cases is not shown as a rate; below `min_firm_n` it
is marked preliminary.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

from callqa.journey.models import InteractionCard, JourneyDataset, StoryVerdict
from callqa.journey.rules import RuleSettings, StoryFacts, apply_rules
from callqa.journey.stats import cluster_bootstrap_share, kaplan_meier, km_median
from callqa.journey.timeline import StoryTimeline, build_timelines
from callqa.journey.vocab import Taxonomy, Units
from callqa.reporting.executive.stats import proportion_ci, quantile

GAP_BUCKETS = [(1, "עד שעה"), (4, "1–4 שעות"), (24, "באותה יממה"), (72, "1–3 ימים"),
               (168, "3–7 ימים"), (float("inf"), "יותר משבוע")]
KIND_HE = {
    "recorded_call": "שיחה מוקלטת", "message": "התכתבות", "abandoned": "שיחה שננטשה",
    "unrecorded_answered": "שיחה שנענתה בלי הקלטה", "unrecorded_unknown": "שיחה בלי הקלטה",
    "branch": "פנייה בסניף", "other": "אחר",
}
OBJECTIVE_HE = {
    "content": "יש תוכן (הקלטה או התכתבות)", "abandoned": "שיחה שננטשה",
    "bank_initiated": "ביוזמת הבנק", "answered_execute": "נענתה בלי הקלטה, ובוצעה פעולה בחשבון",
    "answered_info": "נענתה בלי הקלטה, ובנקאי רק צפה בחשבון",
    "answered_no_trace": "נענתה בלי הקלטה, בלי פעילות בנקאי",
    "unrecorded_no_cover": "בלי הקלטה ובלי כיסוי אטלס",
}


PENDING = "pending"     # a return with content the content stage has not read yet


def shown_category(j) -> str:  # noqa: ANN001 - ReturnJudgement
    """The category a return is shown and counted under: a content return the
    language model has not judged yet is 'pending', not 'unclassifiable'."""
    if j.objective_class == "content" and j.decided_by == "none":
        return PENDING
    return j.category


class Metric(BaseModel):
    model_config = {"extra": "forbid"}

    key: str
    label_he: str
    value: float | None = None
    k: int | None = None
    n: int | None = None
    low: float | None = None
    high: float | None = None
    unit: str = "number"             # share | count | hours | days | minutes | number
    definition_he: str = ""
    wrong_if_he: str = ""
    basis: str = "fact"              # fact | content | inference
    shown: bool = True               # False when the base is below min_rate_n
    preliminary: bool = False


class StorySummary(BaseModel):
    story_key: str
    story_no: int
    branch: str | None
    topic: str
    first_at: datetime
    last_at: datetime
    span_days: float
    contacts: int
    returns: int
    kinds: dict[str, int]
    abandoned: int
    max_same_day: int
    failures: int
    content_returns: int
    retold: int
    promises: int
    promises_broken: int
    status: str
    status_basis: str
    coverage: str
    bankers: int = 0
    units: int = 0
    unit_kinds: list[str] = Field(default_factory=list)
    crossings: int = 0
    banker_minutes: float = 0.0
    background_minutes: float = 0.0
    sessions: int = 0
    view_only_sessions: int = 0


class JourneyAnalysis(BaseModel):
    dataset_id: str
    generated_at: datetime
    data_start: datetime
    data_end: datetime
    metrics: dict[str, Metric]
    stories: list[StorySummary]
    objective_classes: dict[str, int]
    categories_strict: dict[str, int]
    categories_extended: dict[str, int]
    channel_by_category: dict[str, dict[str, int]]
    topics: list[dict]
    gaps: dict[str, int]
    km: list[dict]
    handoffs: dict[str, dict[str, int]]
    after_abandon: dict[str, int]
    promise_funnel: dict[str, int]
    judged_by: dict[str, int]
    coverage: dict[str, int]
    taxonomy_sha: str = ""


def _share_metric(key, label, k, n, *, definition, wrong_if, basis="fact", min_n=10, firm_n=30,
                  clusters: list[tuple[int, int]] | None = None) -> Metric:
    m = Metric(key=key, label_he=label, k=k, n=n, unit="share", definition_he=definition,
               wrong_if_he=wrong_if, basis=basis)
    if not n:
        m.shown = False
        return m
    m.value = k / n
    if clusters is not None:
        cs = cluster_bootstrap_share(clusters)
        m.low, m.high = cs.low, cs.high
    else:
        ci = proportion_ci(k, n)
        m.low, m.high = ci.low, ci.high
    m.shown = n >= min_n
    m.preliminary = n < firm_n
    return m


def _unit_kinds(units: Units, tl: StoryTimeline) -> list[str]:
    return [units.kind(s.unit_code, tl.story.branch) for s in tl.sessions]


def _crossings(kinds: list[str]) -> int:
    center = {"center"}
    n = 0
    for a, b in zip(kinds, kinds[1:], strict=False):
        if (a in center) != (b in center):
            n += 1
    return n


def analyse(dataset: JourneyDataset, *, taxonomy: Taxonomy, units: Units,
            cards: dict[str, InteractionCard] | None = None,
            judgements: dict | None = None, verdicts: dict[str, StoryVerdict] | None = None,
            settings: RuleSettings | None = None, min_rate_n: int = 10, min_firm_n: int = 30,
            now: datetime | None = None) -> tuple[JourneyAnalysis, list[StoryFacts]]:
    """Rules over every story, then the batch figures. `judgements` and
    `verdicts` (from the language model) override the rules' placeholders
    for content returns; facts decided by rule are never overridden."""
    cards = cards or {}
    judgements = judgements or {}
    verdicts = verdicts or {}
    timelines = build_timelines(dataset)
    data_end = max((i.at for i in dataset.interactions), default=datetime(1970, 1, 1))
    data_start = min((i.at for i in dataset.interactions), default=data_end)
    ops_end = max((op.at for s in dataset.atlas_sessions for op in s.ops), default=None)
    if ops_end and ops_end > data_end:
        data_end = ops_end
    settings = settings or RuleSettings()
    settings.data_end = settings.data_end or data_end
    failure_cats = taxonomy.failure_categories

    all_facts: list[StoryFacts] = []
    for tl in timelines:
        facts = apply_rules(tl, cards, settings)
        for iid, j in facts.judgements.items():
            llm = judgements.get(iid)
            if llm is not None and j.objective_class == "content":
                facts.judgements[iid] = llm.model_copy(update={"objective_class": "content"})
        v = verdicts.get(tl.story.story_key)
        if v is not None and facts.verdict is not None:
            rule_v = facts.verdict
            if rule_v.status_basis in ("fact", "inference") and v.status != rule_v.status:
                v = v.model_copy(update={"model_status": v.status, "status": rule_v.status,
                                         "status_basis": rule_v.status_basis,
                                         "status_note_he": rule_v.status_note_he})
            elif v.status_basis == "none":
                v = v.model_copy(update={"status_basis": "content"})
            facts.verdict = v
        all_facts.append(facts)

    # ---- per story
    summaries: list[StorySummary] = []
    story_clusters_fail: list[tuple[int, int]] = []
    for facts in all_facts:
        tl = facts.timeline
        topic = (facts.verdict.topic if facts.verdict and facts.verdict.topic != "other" else None)
        if topic is None:
            topic_votes = Counter(cards[c.interaction.interaction_id].topic for c in tl.contacts
                                  if c.interaction.interaction_id in cards)
            topic = topic_votes.most_common(1)[0][0] if topic_votes else "other"
        per_day = Counter(c.at.date() for c in tl.contacts)
        content = [j for j in facts.judgements.values() if j.objective_class == "content"
                   and j.decided_by != "none"]
        fails = sum(1 for j in facts.judgements.values() if j.category in failure_cats)
        # the same base as the failure rate: returns with content that were read
        judged = {j.interaction_id for j in content}
        retold = sum(1 for c in tl.returns if c.interaction.interaction_id in judged
                     and cards.get(c.interaction.interaction_id)
                     and cards[c.interaction.interaction_id].retold in ("yes", "partial"))
        kinds = _unit_kinds(units, tl)
        linked = {id(s) for c in tl.contacts for s in c.sessions}
        summaries.append(StorySummary(
            story_key=tl.story.story_key, story_no=tl.story.story_no, branch=tl.story.branch,
            topic=topic, first_at=tl.story.first_at, last_at=tl.story.last_at,
            span_days=(tl.story.last_at - tl.story.first_at).total_seconds() / 86400,
            contacts=len(tl.contacts), returns=len(tl.returns),
            kinds=dict(Counter(c.kind for c in tl.contacts)),
            abandoned=sum(1 for c in tl.contacts if c.kind == "abandoned"),
            max_same_day=max(per_day.values(), default=0),
            failures=fails, content_returns=len(content), retold=retold,
            promises=len(facts.promises),
            promises_broken=sum(1 for p in facts.promises if p.outcome == "broken"),
            status=facts.verdict.status if facts.verdict else "unclear",
            status_basis=facts.verdict.status_basis if facts.verdict else "none",
            coverage=tl.story.atlas_coverage,
            bankers=len({s.banker_code for s in tl.sessions}),
            units=len({s.unit_code for s in tl.sessions}),
            unit_kinds=sorted(set(kinds)), crossings=_crossings(kinds),
            banker_minutes=round(sum(s.minutes for s in tl.sessions), 2),
            background_minutes=round(sum(s.minutes for s in tl.sessions if id(s) not in linked), 2),
            sessions=len(tl.sessions),
            view_only_sessions=sum(1 for s in tl.sessions if s.view_only)))
        story_clusters_fail.append((fails, len(content)))

    # ---- distributions
    objective = Counter()
    strict = Counter()
    extended = Counter()
    channel_cat: dict[str, Counter] = defaultdict(Counter)
    judged_by = Counter()
    gaps = Counter()
    gap_hours: list[float] = []
    for facts in all_facts:
        tl = facts.timeline
        prev = None
        for c in tl.contacts:
            if prev is not None:
                h = (c.at - prev.at).total_seconds() / 3600
                gap_hours.append(h)
                gaps[next(label for limit, label in GAP_BUCKETS if h <= limit)] += 1
            prev = c
        for c in tl.returns:
            j = facts.judgements[c.interaction.interaction_id]
            cat = shown_category(j)
            objective[j.objective_class] += 1
            strict[cat] += 1
            extended[j.inferred_category or cat] += 1
            channel_cat[KIND_HE.get(c.kind, c.kind)][cat] += 1
            judged_by[j.decided_by] += 1

    # topics
    topic_rows = []
    by_topic: dict[str, list[StorySummary]] = defaultdict(list)
    for s in summaries:
        by_topic[s.topic].append(s)
    for key in list(taxonomy.topics):
        group = by_topic.get(key, [])
        if not group:
            continue
        returns = sum(s.returns for s in group)
        content = sum(s.content_returns for s in group)
        fails = sum(s.failures for s in group)
        row = {"topic": key, "label": taxonomy.topic_label(key), "stories": len(group),
               "returns": returns, "content_returns": content, "failures": fails,
               "rate": None, "low": None, "high": None}
        if content >= min_rate_n:
            ci = cluster_bootstrap_share([(s.failures, s.content_returns) for s in group])
            row.update(rate=ci.p, low=ci.low, high=ci.high)
        topic_rows.append(row)
    topic_rows.sort(key=lambda r: (-r["stories"], r["topic"]))

    # time to resolution: a closed story resolves at its last contact; any other
    # story was followed until the end of the data and is censored THERE - not
    # at its last contact, which would count a quiet open story as a short one
    durations, resolved = [], []
    for s in summaries:
        if s.status == "closed":
            durations.append(max(0.0, s.span_days))
            resolved.append(True)
        else:
            durations.append(max(0.0, (data_end - s.first_at).total_seconds() / 86400))
            resolved.append(False)
    km = kaplan_meier(durations, resolved) if summaries else []

    # banker handoffs between unit kinds (consecutive sessions of a story)
    handoffs: dict[str, Counter] = defaultdict(Counter)
    for facts in all_facts:
        kinds = _unit_kinds(units, facts.timeline)
        for a, b in zip(kinds, kinds[1:], strict=False):
            if a != b:
                handoffs[a][b] += 1

    # after an abandoned call: who acted first
    after = Counter()
    wait_hours: list[float] = []
    for facts in all_facts:
        tl = facts.timeline
        if tl.story.atlas_coverage != "full":
            continue
        for c in tl.contacts:
            if c.kind != "abandoned":
                continue
            nxt = tl.next_contact_after(c.at, inbound_only=True)
            bank = [s.start for s in tl.sessions if s.start > c.at]
            bank += [x.at for x in tl.contacts if x.direction == "outbound" and x.at > c.at]
            first_bank = min(bank) if bank else None
            if first_bank and (nxt is None or first_bank < nxt.at):
                after["bank_first"] += 1
                wait_hours.append((first_bank - c.at).total_seconds() / 3600)
            elif nxt is not None:
                after["customer_first"] += 1
            else:
                after["nobody"] += 1

    promise_funnel = Counter()
    for facts in all_facts:
        for p in facts.promises:
            promise_funnel["made"] += 1
            promise_funnel[p.outcome] += 1

    # ---- headline metrics
    n_stories = len(summaries)
    n_contacts = sum(s.contacts for s in summaries)
    n_returns = sum(s.returns for s in summaries)
    content_n = sum(s.content_returns for s in summaries)
    fail_k = sum(s.failures for s in summaries)
    m: dict[str, Metric] = {}

    def count(key, label, value, definition, wrong_if="", basis="fact", unit="count"):
        m[key] = Metric(key=key, label_he=label, value=value, unit=unit, definition_he=definition,
                        wrong_if_he=wrong_if, basis=basis)

    count("stories", "סיפורי לקוח", n_stories,
          "חשבונות שנכללו ב-batch; כל חשבון הוא סיפור אחד.",
          "חשבון עם כמה בעלים שפנו בנפרד נספר פעם אחת.")
    count("contacts", "אינטראקציות", n_contacts,
          "כל מגע בין הלקוח לבנק: שיחה (מוקלטת או לא, כולל שיחה שננטשה) או התכתבות.",
          "מגע שלא הגיע לקובץ המקור (למשל ביקור בסניף בלי רישום) אינו נספר.")
    count("returns", "חזרות", n_returns,
          "כל מגע אחרי הראשון בסיפור.",
          "מגע ראשון שהתרחש לפני תחילת התקופה נראה כאן כחזרה ראשונה.")
    mean_returns = n_returns / n_stories if n_stories else None
    count("returns_per_story", "חזרות לסיפור (ממוצע)", mean_returns,
          "מספר החזרות חלקי מספר הסיפורים.", "", unit="number")
    m["failure_rate"] = _share_metric(
        "failure_rate", "שיעור כשל טיפול", fail_k, content_n,
        definition=("מתוך החזרות שיש להן תוכן (הקלטה או התכתבות) ושסווגו: כמה מקורן באי-סגירת "
                    "מעגל טיפול או בטרטור יתר."),
        wrong_if=("חזרות בלי תוכן אינן במכנה; אם הן שונות מהחזרות המוקלטות, השיעור הכללי שונה. "
                  "הסיווג נעשה מתוך התוכן ואומת מול ציטוטים."),
        basis="content", min_n=min_rate_n, firm_n=min_firm_n, clusters=story_clusters_fail)
    content_share_k = objective.get("content", 0)
    classifiable_k = content_share_k + objective.get("bank_initiated", 0)
    m["classifiable"] = _share_metric(
        "classifiable", "חזרות שאפשר לסווג", classifiable_k, n_returns,
        definition="חזרות עם תוכן, או שהבנק יזם אותן (עובדה).",
        wrong_if="שיחה שנענתה בלי הקלטה נספרת כלא ניתנת לסיווג גם אם יש לה תיעוד אחר.",
        min_n=min_rate_n, firm_n=min_firm_n)
    retold_k = sum(s.retold for s in summaries)
    m["retold"] = _share_metric(
        "retold", "הלקוח נדרש לספר מחדש", retold_k, content_n,
        definition="חזרות עם תוכן שבהן הלקוח חזר ותיאר את עניינו מההתחלה, במלואו או בחלקו.",
        wrong_if="לקוח שהזכיר בקצרה פנייה קודמת עשוי להיספר כ'חלקית'.",
        basis="content", min_n=min_rate_n, firm_n=min_firm_n,
        clusters=[(s.retold, s.content_returns) for s in summaries])
    m["promises_broken"] = _share_metric(
        "promises_broken", "הבטחות שהופרו", promise_funnel.get("broken", 0),
        promise_funnel.get("kept", 0) + promise_funnel.get("broken", 0),
        definition=("הבטחות של הבנק (לחזור, לשלוח, לבצע) שהלקוח חזר לפני שהבנק פעל, או שעבר "
                    "המועד בלי פעולה. מכנה: הבטחות שאפשר היה להכריע לגביהן."),
        wrong_if=("פעולה שנעשתה בלי תיעוד באטלס או בשיחה יוצאת (למשל SMS) נספרת כהפרה."),
        basis="inference", min_n=min_rate_n, firm_n=min_firm_n)
    same_day = sum(1 for s in summaries if s.max_same_day >= 3)
    m["same_day_3"] = _share_metric(
        "same_day_3", "סיפורים עם 3 מגעים ומעלה ביום אחד", same_day, n_stories,
        definition="סיפורים שבהם הלקוח והבנק היו במגע שלוש פעמים ומעלה באותו יום.",
        wrong_if="שיחה שהתנתקה ונפתחה מחדש נספרת כשני מגעים.",
        min_n=min_rate_n, firm_n=min_firm_n)
    abandoned = sum(s.abandoned for s in summaries)
    calls = sum(v for s in summaries for k, v in s.kinds.items()
                if k in ("recorded_call", "abandoned", "unrecorded_answered", "unrecorded_unknown"))
    m["abandoned"] = _share_metric(
        "abandoned", "שיחות שננטשו", abandoned, calls,
        definition="שיחות נכנסות שהסתיימו לפני מענה, מתוך כל השיחות.",
        wrong_if="דורש נתוני מוקד (משך ודגל נטישה); בלעדיהם שיחה לא מוקלטת נספרת כלא ידועה.",
        min_n=min_rate_n, firm_n=min_firm_n)
    closed = sum(1 for s in summaries if s.status == "closed")
    unclear = sum(1 for s in summaries if s.status == "unclear")
    m["closed"] = _share_metric(
        "closed", "סיפורים שנסגרו", closed, n_stories,
        definition=("סיפורים שהסתיימו בפתרון לפי התוכן, או (הסקה) שבוצעה בהם פעולה אחרי הפנייה "
                    "האחרונה ולא הייתה פנייה נוספת."),
        wrong_if="לקוח שהתייאש ועזב נראה כסיפור שקט; הסקה אינה ודאות.",
        basis="content", min_n=min_rate_n, firm_n=min_firm_n)
    m["unclear"] = _share_metric(
        "unclear", "סיפורים בסטטוס לא ברור", unclear, n_stories,
        definition="סיפורים שאין תוכן או פעולה מתועדת שמאפשרים לקבוע איך הסתיימו.",
        wrong_if="", min_n=min_rate_n, firm_n=min_firm_n)
    med_km = km_median(km) if km else None
    count("median_days_to_resolution", "חציון ימים עד סגירה", med_km,
          ("לפי עקומת Kaplan-Meier: סיפור שלא נסגר עד סוף הנתונים נחשב 'עדיין פתוח' "
           "עד סוף הנתונים, ולא מושמט."),
          "אם רוב הסיפורים לא נסגרו, החציון לא מוגדר.", basis="inference", unit="days")
    m["median_days_to_resolution"].n = n_stories
    m["median_days_to_resolution"].shown = n_stories >= min_rate_n and med_km is not None
    m["median_days_to_resolution"].preliminary = n_stories < min_firm_n
    if gap_hours:
        count("median_gap_hours", "חציון זמן בין מגעים (שעות)", quantile(gap_hours, 0.5),
              "הזמן בין מגע למגע הבא באותו סיפור.", "", unit="hours")

    covered = [s for s in summaries if s.coverage == "full"]
    if covered:
        n_cov = len(covered)
        count("bankers_per_story", "בנקאים שונים לסיפור (ממוצע)",
              statistics.fmean(s.bankers for s in covered),
              "בנקאים שונים שפתחו את החשבון באטלס, בסיפורים בכיסוי אטלס מלא.",
              "בנקאי שטיפל בלי לפתוח את החשבון באטלס אינו נספר.", unit="number")
        m["three_bankers"] = _share_metric(
            "three_bankers", "סיפורים עם 3 בנקאים ומעלה", sum(1 for s in covered if s.bankers >= 3),
            n_cov, definition="מתוך הסיפורים בכיסוי אטלס מלא.",
            wrong_if="ריבוי בנקאים במוקד הוא בחלקו מבני (מי שעונה); הבעיה היא כשההקשר לא עובר.",
            min_n=min_rate_n, firm_n=min_firm_n)
        m["crossed"] = _share_metric(
            "crossed", "סיפורים שעברו בין המוקד לסניפים",
            sum(1 for s in covered if s.crossings > 0), n_cov,
            definition="סיפורים שבהם פעלו גם מרכז הבנקאות וגם יחידה אחרת, לסירוגין.",
            wrong_if="", min_n=min_rate_n, firm_n=min_firm_n)
        minutes = sum(s.banker_minutes for s in covered)
        bg = sum(s.background_minutes for s in covered)
        count("banker_minutes_per_story", "דקות בנקאי לסיפור (ממוצע, חסם תחתון)",
              minutes / n_cov, "משך הסשנים באטלס מפתיחת החשבון עד הפעולה האחרונה.",
              "הזמן שבין פעולות בתוך סשן נספר; עבודה מחוץ לאטלס לא נספרת.", unit="minutes")
        m["background_share"] = Metric(
            key="background_share", label_he="זמן בנקאי בלי פנייה מתועדת", value=bg / minutes
            if minutes else None, k=None, unit="share",
            definition_he="חלק מזמן הבנקאים שלא היה צמוד לשום פנייה מתועדת של הלקוח.",
            wrong_if_he="כולל ביקור פיזי בסניף ועבודת תפעול עורפי, שאין להם רישום פנייה.",
            n=n_cov, shown=minutes > 0 and n_cov >= min_rate_n,
            preliminary=n_cov < min_firm_n)
        sessions = sum(s.sessions for s in covered)
        m["view_only"] = _share_metric(
            "view_only", "סשנים של צפייה בלבד", sum(s.view_only_sessions for s in covered),
            sessions, definition="סשנים שבהם הבנקאי רק פתח את המסך או שאל מידע, בלי לבצע פעולה.",
            wrong_if="פעולה שנרשמה בקוד שלא סווג נספרת כלא-צפייה.",
            min_n=min_rate_n, firm_n=min_firm_n)
    ab_total = sum(after.values())
    if ab_total:
        m["customer_first_after_abandon"] = _share_metric(
            "customer_first_after_abandon", "אחרי נטישה: הלקוח חזר לפני שהבנק פעל",
            after.get("customer_first", 0), ab_total,
            definition=("שיחות שננטשו, שאחריהן הלקוח פנה שוב לפני שבנקאי פתח את החשבון או חזר "
                        "אליו (בסיפורים בכיסוי אטלס מלא)."),
            wrong_if="חזרה ללקוח שלא נרשמה באטלס או כשיחה יוצאת לא נראית.",
            min_n=min_rate_n, firm_n=min_firm_n)
        if wait_hours:
            count("bank_first_hours", "כשהבנק פעל ראשון - חציון שעות", quantile(wait_hours, 0.5),
                  "הזמן מהשיחה שננטשה עד שבנקאי פתח את החשבון.", "", unit="hours")

    analysis = JourneyAnalysis(
        dataset_id=dataset.dataset_id, generated_at=now or datetime.now(),
        data_start=data_start, data_end=data_end, metrics=m, stories=summaries,
        objective_classes=dict(objective), categories_strict=dict(strict),
        categories_extended=dict(extended),
        channel_by_category={k: dict(v) for k, v in channel_cat.items()},
        topics=topic_rows, gaps={label: gaps.get(label, 0) for _l, label in GAP_BUCKETS},
        km=[{"t": s.t, "s": s.survival, "at_risk": s.at_risk, "events": s.events,
             "censored": s.censored} for s in km],
        handoffs={k: dict(v) for k, v in handoffs.items()}, after_abandon=dict(after),
        promise_funnel=dict(promise_funnel), judged_by=dict(judged_by),
        coverage=dict(Counter(s.coverage for s in summaries)), taxonomy_sha=taxonomy.sha256)
    return analysis, all_facts
