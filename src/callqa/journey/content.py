"""The content stage: read every recorded call and correspondence of a batch,
and turn what was said into cards, return judgements and story verdicts.

    for each story (in parallel on the gpu profile)
      A  a card per contact that has text (a redacted transcript, or messages)
         rules: promises kept or broken, the facts of every return
      B  why each return with content happened, over the story's fact table
      C  the story's headline, paragraph and end status

Every answer is parsed strictly, its quotes verified against the lines they
name, and retried with feedback up to journey.llm.max_retries times; a quote
that never verifies is dropped with the claim it supported, and the rest of
the card stands. A contact whose text is not there yet (not transcribed) is
left for later - its return stays "pending" in the report - and a contact the
model could not answer is counted, never guessed.

Only redacted text is read: transcripts from output/redacted/, message bodies
as the import stored them (redacted unless journey.redact_messages is off).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from callqa.config import Config
from callqa.journey.analysis import KIND_HE, OBJECTIVE_HE
from callqa.journey.llm.cache import AnswerCache
from callqa.journey.llm.client import ContentEngine, Profile, make_engine, resolve_profile
from callqa.journey.llm.tasks import (
    COMMIT_HE,
    VERSIONS,
    Prompt,
    QuoteRef,
    TaskError,
    card_prompt,
    fact_table,
    parse_card,
    parse_returns,
    parse_story,
    returns_prompt,
    story_prompt,
)
from callqa.journey.models import ContentLayer, InteractionCard, ReturnJudgement, StoryVerdict
from callqa.journey.rules import RuleSettings, apply_rules
from callqa.journey.store import dataset_dir, load_dataset, resolve_dataset_id, save_content
from callqa.journey.timeline import Contact, StoryTimeline, build_timelines
from callqa.journey.transcript_view import (
    ContentView,
    Lexicon,
    call_view,
    compress,
    load_lexicon,
    message_view,
)
from callqa.journey.vocab import Taxonomy, Units, load_taxonomy, load_units
from callqa.models import RedactedTranscript

logger = logging.getLogger(__name__)

DIRECTION_HE = {"inbound": "נכנסת", "outbound": "יוצאת"}
OUTCOME_HE = {"kept": "קוימה", "broken": "הופרה", "unknown": "לא ידוע"}
SETTLED_HE = {"bank_contact": "הבנק חזר", "atlas_execute": "בוצעה פעולה בחשבון",
              "customer_returned": "הלקוח חזר לפני שהבנק פעל", "deadline": "עבר המועד"}


class TaskFailed(RuntimeError):
    pass


@dataclass
class ContentStats:
    stories: int = 0
    contacts_with_text: int = 0
    cards: int = 0
    no_text: int = 0
    failed: int = 0
    retries: int = 0
    dropped_quotes: int = 0
    cache_hits: int = 0
    engine_calls: int = 0
    skipped: int = 0
    problems: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **kw: int) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def problem(self, key: str) -> None:
        with self._lock:
            self.problems[key] += 1


@dataclass
class Context:
    config: Config
    output_dir: Path
    taxonomy: Taxonomy
    units: Units
    lexicon: Lexicon
    profile: Profile
    engine: ContentEngine
    cache: AnswerCache
    settings: RuleSettings
    stats: ContentStats


# -- asking -----------------------------------------------------------------------------

def ask(ctx: Context, prompt: Prompt, parse, soft=None):  # noqa: ANN001, ANN201
    """The answer to `prompt`, parsed; retried with feedback on a hard error
    (TaskError) and, while retries remain, on soft failures (`soft(result)`
    returns their messages). A soft failure on the last try is accepted - the
    parser already dropped what it could not verify."""
    feedback: str | None = None
    last = None
    last_error = "no answer"
    for attempt in range(ctx.profile.max_retries + 1):
        if attempt:
            ctx.stats.add(retries=1)
        key = ctx.cache.key(ctx.engine.name, ctx.engine.model, prompt, feedback or "")
        raw = ctx.cache.get(prompt, key)
        if raw is None:
            ctx.stats.add(engine_calls=1)
            raw = ctx.engine.answer(prompt, feedback)
            ctx.cache.put(prompt, key, raw)
        else:
            ctx.stats.add(cache_hits=1)
        try:
            result = parse(raw)
        except TaskError as exc:
            feedback = last_error = str(exc)
            continue
        issues = soft(result) if soft else []
        if issues and attempt < ctx.profile.max_retries:
            last, feedback = result, "; ".join(issues)
            continue
        return result
    if last is not None:
        return last
    raise TaskFailed(last_error)


# -- one story --------------------------------------------------------------------------

def _view(ctx: Context, c: Contact) -> ContentView | None:
    return contact_view(ctx.output_dir, c, ctx.config.journey.uncertain_word_prob)


def contact_view(output_dir: Path, c: Contact, uncertain_prob: float = 0.5) -> ContentView | None:
    """The numbered text of one contact, or None when it has none (yet)."""
    i = c.interaction
    if c.kind == "message":
        if not c.messages:
            return None
        return message_view(i.interaction_id, c.messages)
    if c.kind != "recorded_call" or not i.call_id:
        return None
    path = output_dir / "redacted" / f"{i.call_id}.json"
    if not path.exists():
        return None
    transcript = RedactedTranscript.model_validate_json(path.read_text(encoding="utf-8"))
    if not transcript.enabled or not transcript.turns:
        return None          # never read text the redaction stage did not clean
    dialog = output_dir / "transcripts" / f"{i.call_id}.dialog.json"
    role_conf = None
    if dialog.exists():
        try:
            role_conf = float(json.loads(dialog.read_text(encoding="utf-8"))
                              .get("role_confidence", 1.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            role_conf = None
    return call_view(i.interaction_id, transcript,
                     segmap_path=output_dir / "audio" / "assembled" / f"{i.call_id}.segmap.json",
                     dialog_path=dialog, role_confidence=role_conf, uncertain_prob=uncertain_prob)


def _when(c: Contact) -> str:
    return c.at.strftime("%d.%m %H:%M")


def _kind_he(c: Contact) -> str:
    return KIND_HE.get(c.kind, c.kind)


def read_story(ctx: Context, tl: StoryTimeline) -> tuple[dict[str, InteractionCard],
                                                         dict[str, ReturnJudgement],
                                                         StoryVerdict | None]:
    tax = ctx.taxonomy
    cards: dict[str, InteractionCard] = {}
    previous: list[dict] = []
    total = len(tl.contacts)
    for c in tl.contacts:
        iid = c.interaction.interaction_id
        view = _view(ctx, c)
        if view is None:
            if c.has_content:
                ctx.stats.add(no_text=1)
        else:
            ctx.stats.add(contacts_with_text=1)
            if ctx.profile.transcript_mode == "compressed":
                view = compress(view, ctx.lexicon, ctx.profile.transcript_chars)
            prompt = card_prompt(view, taxonomy=tax, position=c.index + 1, total=total,
                                 channel_he=_kind_he(c) + (f", {DIRECTION_HE[c.direction]}"
                                                           if c.direction in DIRECTION_HE else ""),
                                 when=_when(c), previous=previous)
            try:
                res = ask(ctx, prompt,
                          lambda raw, v=view, first=not c.is_return: parse_card(raw, v, tax,
                                                                                first=first),
                          soft=lambda r: r.failed_quotes)
            except Exception as exc:  # noqa: BLE001 - one contact never stops the batch
                ctx.stats.add(failed=1)
                ctx.stats.problem(f"card: {type(exc).__name__}")
                logger.warning("story %03d contact %d: no card (%s)", tl.story.story_no,
                               c.index + 1, type(exc).__name__)
            else:
                cards[iid] = res.card
                ctx.stats.add(cards=1, dropped_quotes=len(res.card.problems))
                for p in res.card.problems:
                    ctx.stats.problem(p.split("[")[0])
        card = cards.get(iid)
        previous.append({"no": c.index + 1, "when": _when(c), "kind": _kind_he(c),
                         "topic": tax.topic_label(card.topic) if card else "",
                         "topic_id": card.topic if card else "",
                         "issue": card.issue_he if card else ""})

    facts = apply_rules(tl, cards, ctx.settings)
    if not cards:
        return cards, {}, None

    # the story's fact table and its verified quotes
    quotes: list[QuoteRef] = []
    for c in tl.contacts:
        card = cards.get(c.interaction.interaction_id)
        if card is None:
            continue
        evs = [card.retold_ev, card.prior_ev, card.outcome_ev] + [x.evidence for x in card.commitments]
        for ev in evs:
            if ev is not None and not any(q.evidence.quote == ev.quote and q.contact_no == c.index + 1
                                          for q in quotes):
                ev = ev.model_copy(update={"quote_id": f"Q{len(quotes) + 1}"})
                quotes.append(QuoteRef(f"Q{len(quotes) + 1}", c.index + 1, ev))
    rows = []
    for c in tl.contacts:
        iid = c.interaction.interaction_id
        card = cards.get(iid)
        j = facts.judgements.get(iid)
        made = [p for p in facts.promises if p.made_in == iid]
        broken_here = any(p.outcome == "broken" and p.settled_by == "customer_returned"
                          and p.settled_at == c.at for p in facts.promises)
        units = sorted({ctx.units.label(s.unit_code, tl.story.branch) for s in c.sessions})
        rows.append({
            "no": c.index + 1, "when": _when(c), "kind_he": _kind_he(c),
            "direction_he": DIRECTION_HE.get(c.direction, ""), "is_return": c.is_return,
            "kind": c.kind, "direction": c.direction, "broken_here": broken_here,
            "card": ({"topic": card.topic, "topic_he": tax.topic_label(card.topic),
                      "issue": card.issue_he, "outcome": card.outcome, "retold": card.retold,
                      "redirect": card.redirect,
                      "prior_contact_mentioned": card.prior_contact_mentioned,
                      "commitments": [{"kind": x.kind, "by": x.by} for x in card.commitments]}
                     if card else None),
            "known_he": (OBJECTIVE_HE.get(j.objective_class, "")
                         if j is not None and j.objective_class != "content" else ""),
            "promises_he": "; ".join(f"{COMMIT_HE.get(p.kind, p.kind)} — {OUTCOME_HE[p.outcome]}"
                                     + (f" ({SETTLED_HE.get(p.settled_by, '')})"
                                        if p.settled_by else "") for p in made),
            "atlas_he": (f"{len(c.sessions)} סשנים, {', '.join(units)}, "
                         + ("בוצעה פעולה" if any(s.has_execute for s in c.sessions)
                            else "צפייה בלבד")) if c.sessions else "",
        })
    quote_contacts = {q.qid: q.contact_no for q in quotes}
    by_qid = {q.qid: q for q in quotes}

    # B: why each return with content happened
    judgements: dict[str, ReturnJudgement] = {}
    to_classify = [c.index + 1 for c in tl.returns
                   if c.interaction.interaction_id in cards
                   and facts.judgements[c.interaction.interaction_id].objective_class == "content"]
    categories = {k: str(v.get("definition", v.get("label", k)))
                  for k, v in tax.categories.items() if k != "bank_initiated"}
    if to_classify:
        prompt = returns_prompt(fact_table(rows, quotes), to_classify=to_classify,
                                categories=categories, quotes=quotes,
                                inputs={"rows": rows, "quote_contacts": quote_contacts})
        try:
            answers = ask(ctx, prompt, lambda raw: parse_returns(
                raw, to_classify=to_classify, categories=list(categories), qids=list(by_qid)))
        except Exception as exc:  # noqa: BLE001
            ctx.stats.add(failed=1)
            ctx.stats.problem(f"returns: {type(exc).__name__}")
            answers = []
        contact_by_no = {c.index + 1: c for c in tl.contacts}
        for a in answers:
            iid = contact_by_no[a.contact_no].interaction.interaction_id
            judgements[iid] = ReturnJudgement(
                interaction_id=iid, category=a.category, basis="content", decided_by="llm",
                objective_class="content", reason_he=a.reason_he,
                is_break_point=a.is_break_point,
                break_he=a.reason_he if a.is_break_point else "",
                quotes=[by_qid[q].evidence for q in a.quote_ids])
            for r in rows:
                if r["no"] == a.contact_no:
                    r["category"] = a.category
                    r["is_break"] = a.is_break_point
                    r["known_he"] = f"סיבת החזרה: {tax.category_label(a.category)}"

    # C: the story
    topic_votes = Counter(card.topic for card in cards.values() if card.topic != "other")
    topic = topic_votes.most_common(1)[0][0] if topic_votes else "other"
    prompt = story_prompt(fact_table(rows, quotes), contacts=[r["no"] for r in rows],
                          quotes=quotes, inputs={"rows": rows, "quote_contacts": quote_contacts})
    try:
        s = ask(ctx, prompt, lambda raw: parse_story(raw, contacts=[r["no"] for r in rows],
                                                     qids=list(by_qid)))
    except Exception as exc:  # noqa: BLE001
        ctx.stats.add(failed=1)
        ctx.stats.problem(f"story: {type(exc).__name__}")
        return cards, judgements, None
    break_iid = next((c.interaction.interaction_id for c in tl.contacts
                      if c.index + 1 == s.break_contact), None) if s.break_contact else None
    verdict = StoryVerdict(story_key=tl.story.story_key, topic=topic, status=s.status,
                           status_basis="content", status_note_he=s.status_note_he,
                           headline_he=s.headline_he, narrative_he=s.narrative_he,
                           break_point_interaction_id=break_iid,
                           quotes=[by_qid[q].evidence for q in s.quote_ids])
    return cards, judgements, verdict


# -- the batch -----------------------------------------------------------------------------

def run_content(config: Config, dataset_id: str | None = None, *, mock: bool = False,
                profile: str | None = None, limit_stories: int | None = None,
                engine: ContentEngine | None = None, progress=None,  # noqa: ANN001
                stop=None) -> tuple[ContentLayer, ContentStats]:  # noqa: ANN001
    """Read the batch. `stop()` is asked before each story; stories not read
    this time keep what an earlier run of the same engine and model read."""
    ds_id = resolve_dataset_id(config, dataset_id)
    dataset = load_dataset(config, ds_id)
    prof = resolve_profile(config, profile)
    eng = engine or make_engine(config, prof, mock=mock)
    if hasattr(eng, "check"):
        eng.check()
    stats = ContentStats()
    ctx = Context(config=config, output_dir=config.paths.output_dir,
                  taxonomy=load_taxonomy(config.journey.taxonomy),
                  units=load_units(config.journey.units),
                  lexicon=load_lexicon(config.journey.lexicon), profile=prof, engine=eng,
                  cache=AnswerCache(dataset_dir(config, ds_id) / "content_cache"),
                  settings=RuleSettings(callback_business_days=config.journey.callback_business_days,
                                        quiet_days=config.journey.quiet_days),
                  stats=stats)
    data_end = max((i.at for i in dataset.interactions), default=None)
    ops_end = max((op.at for s in dataset.atlas_sessions for op in s.ops), default=None)
    ctx.settings.data_end = max(x for x in (data_end, ops_end) if x is not None) \
        if (data_end or ops_end) else None
    timelines = build_timelines(dataset)
    if limit_stories:
        timelines = timelines[:limit_stories]
    layer = ContentLayer(engine=eng.name, model=eng.model, prompt_versions=dict(VERSIONS),
                         created_at=datetime.now(UTC))
    from callqa.journey.store import load_content
    earlier = load_content(config, ds_id)
    if earlier is not None and (earlier.engine, earlier.model) == (eng.name, eng.model):
        layer.cards.update(earlier.cards)
        layer.judgements.update(earlier.judgements)
        layer.verdicts.update(earlier.verdicts)
    lock = threading.Lock()
    done = 0

    def one(tl: StoryTimeline) -> None:
        nonlocal done
        if stop is not None and stop():
            stats.add(skipped=1)
            return
        cards, judgements, verdict = read_story(ctx, tl)
        ids = {c.interaction.interaction_id for c in tl.contacts}
        with lock:
            for store in (layer.cards, layer.judgements):
                for iid in ids & set(store):
                    store.pop(iid)
            layer.verdicts.pop(tl.story.story_key, None)
            layer.cards.update(cards)
            layer.judgements.update(judgements)
            if verdict is not None:
                layer.verdicts[tl.story.story_key] = verdict
            done += 1
            stats.stories += 1
            if progress:
                progress(done, len(timelines))

    if prof.concurrency > 1 and len(timelines) > 1:
        with ThreadPoolExecutor(max_workers=prof.concurrency) as pool:
            list(pool.map(one, timelines))
    else:
        for tl in timelines:
            one(tl)
    save_content(config, ds_id, layer)
    return layer, stats
