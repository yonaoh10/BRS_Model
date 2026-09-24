"""The three reading tasks, as prompts with a JSON schema and a strict parser.

    A  card      one recorded call or correspondence: topic, request, outcome,
                 the bank's promises, whether the customer had to tell it again
    B  returns   one story: why each return with content happened, and where
                 the story broke
    C  story     one story: a headline, a paragraph, and how it ended

Rules the prompts state and the parsers enforce:
- evidence is a line number plus the words on that line; the parser checks
  them against the line (verify.py) and drops a claim whose evidence fails;
- tasks B and C never quote: they point at quotes task A already verified
  (Q1..Qn), so no text reaches a report that was not said;
- every closed field is an enum; anything else is a retry with feedback.

Each prompt carries a structured copy of its inputs: the mock engine answers
from it, and the cache key is built from the prompt text itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from callqa.journey.models import Commitment, Evidence, InteractionCard
from callqa.journey.transcript_view import ContentView
from callqa.journey.vocab import Taxonomy

VERSIONS = {"card": "card-1", "returns": "returns-1", "story": "story-1"}

OUTCOMES = ["resolved", "partially_resolved", "not_resolved", "info_only", "unknown"]
COMMIT_KINDS = ["callback", "send_document", "execute_action", "check_and_update",
                "customer_to_act"]
RETOLD = ["yes", "partial", "no", "first_contact", "unknown"]
AWARE = ["yes", "no", "unclear"]
REDIRECT = ["none", "internal_transfer", "sent_to_branch", "sent_to_other_channel"]
CONFIDENCE = ["high", "medium", "low"]
STATUS = ["closed", "open", "unclear"]

OUTCOME_HE = {"resolved": "נפתר", "partially_resolved": "נפתר חלקית", "not_resolved": "לא נפתר",
              "info_only": "מסירת מידע", "unknown": "לא ידוע"}
COMMIT_HE = {"callback": "לחזור ללקוח", "send_document": "לשלוח מסמך",
             "execute_action": "לבצע פעולה", "check_and_update": "לבדוק ולעדכן",
             "customer_to_act": "הלקוח יפעל (יביא/ישלח)"}


class TaskError(ValueError):
    """The answer cannot be used; str() is fed back to the model on retry."""


@dataclass
class Prompt:
    task: str
    system: str
    user: str
    schema: dict
    inputs: dict[str, Any]

    @property
    def version(self) -> str:
        return VERSIONS[self.task]

    def sha256(self) -> str:
        return hashlib.sha256((self.system + "\n\n" + self.user).encode("utf-8")).hexdigest()


# -- schema pieces -------------------------------------------------------------------

_EV = {"type": "object",
       "properties": {"line": {"type": "integer", "minimum": 1},
                      "quote": {"type": "string", "minLength": 8, "maxLength": 160}},
       "required": ["line", "quote"], "additionalProperties": False}
_EV1 = {"type": "array", "items": _EV, "maxItems": 1}


def _str(n: int) -> dict:
    return {"type": "string", "maxLength": n}


def _enum(values: list[str]) -> dict:
    return {"type": "string", "enum": list(values)}


# -- A: the card ---------------------------------------------------------------------

SYSTEM_A = """אתה אנליסט שירות בבנק. אתה קורא שיחה מוקלטת (תמלול) או התכתבות בין לקוח לבנק, וממלא כרטיס עובדתי קצר.

כללים:
1. כתוב רק מה שנאמר בטקסט. אל תנחש.
2. כל ראיה היא מספר שורה (L..) ומילים שמופיעות בשורה הזו בדיוק, בלי שינוי. אל תחבר מילים משתי שורות.
3. אסור לצטט שורה שמסומנת ⚠ (התמלול בה לא ודאי).
4. "הבטחה" היא התחייבות מפורשת לפעולה עתידית ("אחזור אלייך", "נשלח", "נבצע", "נבדוק ונעדכן"). מי התחייב: bank או customer.
5. retold: האם הלקוח נאלץ לתאר שוב את עניינו מההתחלה בגלל פנייה קודמת (yes / partial / no). בפנייה הראשונה בסיפור: first_contact.
6. פרטים מזהים מוסתרים בטקסט (למשל <שם:████>) — אל תנסה לשחזר אותם.
7. החזר JSON בלבד, לפי הסכמה."""


def card_prompt(view: ContentView, *, taxonomy: Taxonomy, position: int, total: int,
                channel_he: str, when: str, previous: list[dict]) -> Prompt:
    topics = "\n".join(f"- {k}: {v.get('label', '')} — {v.get('covers', '')}"
                       for k, v in taxonomy.topics.items())
    prev = "\n".join(f"- #{p['no']} {p['when']} {p['kind']}"
                     + (f" · נושא: {p['topic']}" if p.get("topic") else "")
                     + (f" · {p['issue']}" if p.get("issue") else "") for p in previous) or "אין"
    user = (f"מגע {position} מתוך {total} בסיפור של הלקוח. ערוץ: {channel_he}. מועד: {when}.\n"
            f"מגעים קודמים בסיפור:\n{prev}\n\n"
            f"נושאים אפשריים (topic):\n{topics}\n\n"
            f"הטקסט:\n{view.render()}\n")
    schema = {
        "type": "object",
        "properties": {
            "topic": _enum(list(taxonomy.topics)),
            "issue_he": _str(200), "customer_request_he": _str(200),
            "outcome": _enum(OUTCOMES), "outcome_ev": _EV1,
            "commitments": {"type": "array", "maxItems": 4, "items": {
                "type": "object",
                "properties": {"kind": _enum(COMMIT_KINDS), "by": _enum(["bank", "customer"]),
                               "when_he": _str(60), "ev": _EV1},
                "required": ["kind", "by", "when_he", "ev"], "additionalProperties": False}},
            "prior_contact_mentioned": {"type": "boolean"}, "prior_ev": _EV1,
            "retold": _enum(RETOLD), "retold_ev": _EV1,
            "banker_aware_of_history": _enum(AWARE), "redirect": _enum(REDIRECT),
            "frustration": {"type": "integer", "minimum": 1, "maximum": 3},
            "confidence": _enum(CONFIDENCE),
        },
        "required": ["topic", "issue_he", "customer_request_he", "outcome", "outcome_ev",
                     "commitments", "prior_contact_mentioned", "prior_ev", "retold", "retold_ev",
                     "banker_aware_of_history", "redirect", "frustration", "confidence"],
        "additionalProperties": False,
    }
    inputs = {"position": position, "total": total, "previous": previous,
              "lines": [{"no": ln.no, "who": ln.who, "text": ln.text, "uncertain": ln.uncertain}
                        for ln in view.lines],
              "topics": list(taxonomy.topics)}
    return Prompt("card", SYSTEM_A, user, schema, inputs)


def _load(raw: str) -> dict:
    from callqa.judge.validation import extract_json
    try:
        data = json.loads(extract_json(raw or ""))
    except json.JSONDecodeError as exc:
        raise TaskError(f"התשובה אינה JSON תקין: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskError("התשובה אינה אובייקט JSON")
    return data


def _choice(data: dict, key: str, allowed: list[str]) -> str:
    value = data.get(key)
    if value not in allowed:
        raise TaskError(f"השדה {key} חייב להיות אחד מ: {', '.join(allowed)} (התקבל {value!r})")
    return value


def _text(data: dict, key: str, limit: int) -> str:
    from callqa.redaction import redact_text
    value = data.get(key) or ""
    if not isinstance(value, str):
        raise TaskError(f"השדה {key} חייב להיות מחרוזת")
    return redact_text(" ".join(value.split())[:limit])[0]


@dataclass
class CardResult:
    card: InteractionCard
    failed_quotes: list[str] = field(default_factory=list)   # feedback for a retry


def parse_card(raw: str, view: ContentView, taxonomy: Taxonomy, *, first: bool) -> CardResult:
    from callqa.journey.llm.verify import verify_evidence

    data = _load(raw)
    failed: list[str] = []
    problems: list[str] = []

    def ev(key_or_list: Any, label: str) -> Evidence | None:
        if isinstance(key_or_list, dict):          # one object where a list of one was due
            key_or_list = [key_or_list]
        if key_or_list is not None and not isinstance(key_or_list, list):
            failed.append(f"{label}: ראיה חייבת להיות מערך של {{line, quote}}")
            problems.append(label)
            return None
        items = key_or_list or []
        if not items:
            return None
        item = items[0] if isinstance(items[0], dict) else {}
        checked, why = verify_evidence(view, item.get("line"), item.get("quote"))
        if checked is None:
            failed.append(f"{label}: {why}")
            problems.append(label)
        return checked

    topic = _choice(data, "topic", list(taxonomy.topics))
    outcome = _choice(data, "outcome", OUTCOMES)
    retold = _choice(data, "retold", RETOLD)
    aware = _choice(data, "banker_aware_of_history", AWARE)
    redirect = _choice(data, "redirect", REDIRECT)
    confidence = _choice(data, "confidence", CONFIDENCE)
    frustration = data.get("frustration")
    if not isinstance(frustration, int) or not 1 <= frustration <= 3:
        raise TaskError("frustration חייב להיות מספר שלם 1–3")
    commitments = []
    raw_commitments = data.get("commitments") or []
    if not isinstance(raw_commitments, list):
        raise TaskError("commitments חייב להיות מערך")
    for i, c in enumerate(raw_commitments[:4]):
        if not isinstance(c, dict):
            raise TaskError("כל פריט ב-commitments חייב להיות אובייקט")
        kind = _choice(c, "kind", COMMIT_KINDS)
        by = _choice(c, "by", ["bank", "customer"])
        evidence = ev(c.get("ev"), f"commitments[{i}]")
        if evidence is None:
            # a promise nobody can point at is not counted as a promise
            continue
        commitments.append(Commitment(kind=kind, by=by, when_he=_text(c, "when_he", 60),
                                      evidence=evidence))
    retold_ev = ev(data.get("retold_ev"), "retold_ev")
    if first:
        retold = "first_contact"
    elif retold in ("yes", "partial") and retold_ev is None:
        retold = "unknown"
    prior_ev = ev(data.get("prior_ev"), "prior_ev")
    prior = bool(data.get("prior_contact_mentioned")) and prior_ev is not None
    card = InteractionCard(
        interaction_id=view.interaction_id, topic=topic,
        issue_he=_text(data, "issue_he", 200),
        customer_request_he=_text(data, "customer_request_he", 200),
        outcome=outcome, outcome_ev=ev(data.get("outcome_ev"), "outcome_ev"),
        commitments=commitments, prior_contact_mentioned=prior, prior_ev=prior_ev,
        retold=retold, retold_ev=retold_ev, banker_aware_of_history=aware, redirect=redirect,
        frustration=frustration, confidence=confidence, problems=problems)
    return CardResult(card=card, failed_quotes=failed)


# -- the story's fact table (input of B and C) ------------------------------------------

@dataclass
class QuoteRef:
    qid: str
    contact_no: int
    evidence: Evidence


def fact_table(rows: list[dict], quotes: list[QuoteRef]) -> str:
    """rows: {no, when, kind_he, direction_he, card?: {...}, known_he, promises_he, atlas_he}"""
    out = []
    for r in rows:
        parts = [f"#{r['no']}", r["when"], r["kind_he"] + (f", {r['direction_he']}"
                                                            if r.get("direction_he") else "")]
        c = r.get("card")
        if c:
            parts.append(f"נושא: {c['topic_he']}")
            if c.get("issue"):
                parts.append(f"עניין: {c['issue']}")
            parts.append(f"תוצאה: {OUTCOME_HE.get(c['outcome'], c['outcome'])}")
            if c.get("retold") in ("yes", "partial"):
                parts.append("הלקוח סיפר מחדש" + (" (חלקית)" if c["retold"] == "partial" else ""))
            if c.get("redirect") and c["redirect"] != "none":
                parts.append("הופנה/הועבר")
        if r.get("known_he"):
            parts.append(r["known_he"])
        if r.get("promises_he"):
            parts.append("הבטחות: " + r["promises_he"])
        if r.get("atlas_he"):
            parts.append("אטלס: " + r["atlas_he"])
        out.append(" | ".join(parts))
    qs = [f"{q.qid} (#{q.contact_no}, L{q.evidence.line}, "
          f"{'בנק' if q.evidence.speaker == 'bank' else 'לקוח'}): „{q.evidence.quote}”"
          for q in quotes]
    return "\n".join(out) + ("\n\nציטוטים מאומתים:\n" + "\n".join(qs) if qs else "")


# -- B: why each return happened ---------------------------------------------------------

SYSTEM_B = """אתה אנליסט שירות בבנק. לפניך סיפור של לקוח: כל המגעים שלו עם הבנק לפי הסדר, עם מה שידוע על כל אחד.
לכל חזרה ברשימה "לסווג" קבע למה הלקוח חזר, לפי ההגדרות. נמק במשפט אחד.
כללים:
1. הסתמך רק על הטבלה ועל הציטוטים. אל תמציא.
2. אל תכתוב ציטוטים משלך — בחר עד שני מזהי ציטוט (Q..) מהרשימה.
3. נקודת שבר: המגע שבו הטיפול התחיל להשתבש. לכל היותר אחת בסיפור.
4. החזר JSON בלבד, לפי הסכמה."""


def returns_prompt(table: str, *, to_classify: list[int], categories: dict[str, str],
                   quotes: list[QuoteRef], inputs: dict) -> Prompt:
    defs = "\n".join(f"- {k}: {v}" for k, v in categories.items())
    user = (f"הגדרות (category):\n{defs}\n\n"
            f"הסיפור:\n{table}\n\n"
            f"לסווג: {', '.join(f'#{n}' for n in to_classify)}\n")
    qids = [q.qid for q in quotes] or ["-"]
    item = {"type": "object",
            "properties": {"contact": {"type": "integer", "enum": list(to_classify)},
                           "category": _enum(list(categories)), "reason_he": _str(250),
                           "is_break_point": {"type": "boolean"},
                           "quote_ids": {"type": "array", "maxItems": 2, "items": _enum(qids)}},
            "required": ["contact", "category", "reason_he", "is_break_point", "quote_ids"],
            "additionalProperties": False}
    schema = {"type": "object",
              "properties": {"returns": {"type": "array", "items": item,
                                         "minItems": len(to_classify),
                                         "maxItems": len(to_classify)}},
              "required": ["returns"], "additionalProperties": False}
    return Prompt("returns", SYSTEM_B, user, schema,
                  dict(inputs, to_classify=list(to_classify), qids=[q.qid for q in quotes]))


@dataclass
class ReturnAnswer:
    contact_no: int
    category: str
    reason_he: str
    is_break_point: bool
    quote_ids: list[str]


def parse_returns(raw: str, *, to_classify: list[int], categories: list[str],
                  qids: list[str]) -> list[ReturnAnswer]:
    data = _load(raw)
    items = data.get("returns")
    if not isinstance(items, list):
        raise TaskError("returns חייב להיות מערך")
    seen: dict[int, ReturnAnswer] = {}
    for it in items:
        if not isinstance(it, dict):
            raise TaskError("כל פריט ב-returns חייב להיות אובייקט")
        no = it.get("contact")
        if no not in to_classify:
            raise TaskError(f"contact {no!r} אינו ברשימה לסיווג ({to_classify})")
        if no in seen:
            raise TaskError(f"contact {no} מופיע פעמיים")
        cat = _choice(it, "category", categories)
        ids = [q for q in (it.get("quote_ids") or []) if isinstance(q, str)]
        bad = [q for q in ids if q not in qids]
        if bad:
            raise TaskError(f"מזהי ציטוט לא קיימים: {bad}; השתמש רק ב: {qids or 'אין'}")
        seen[no] = ReturnAnswer(no, cat, _text(it, "reason_he", 250),
                                bool(it.get("is_break_point")), ids[:2])
    missing = [n for n in to_classify if n not in seen]
    if missing:
        raise TaskError(f"חסר סיווג ל: {missing}")
    breaks = [a for a in seen.values() if a.is_break_point]
    for extra in breaks[1:]:
        extra.is_break_point = False
    return [seen[n] for n in to_classify]


# -- C: the story ---------------------------------------------------------------------------

SYSTEM_C = """אתה אנליסט שירות בבנק. לפניך סיפור של לקוח: כל המגעים שלו עם הבנק לפי הסדר, וסיבת כל חזרה.
כתוב כותרת קצרה (עד 12 מילים) ופסקה של 2–4 משפטים: מה הלקוח רצה, מה השתבש (אם השתבש), ואיך זה נגמר.
קבע סטטוס: closed (העניין נפתר), open (לא נפתר), unclear (אי אפשר לדעת מהנתונים).
כללים: הסתמך רק על הטבלה. אל תכתוב ציטוטים — בחר עד שני מזהי ציטוט (Q..). אל תזכיר שמות או מספרים מזהים. החזר JSON בלבד."""


def story_prompt(table: str, *, contacts: list[int], quotes: list[QuoteRef], inputs: dict) -> Prompt:
    qids = [q.qid for q in quotes] or ["-"]
    schema = {"type": "object",
              "properties": {"headline_he": _str(90), "narrative_he": _str(700),
                             "status": _enum(STATUS), "status_note_he": _str(160),
                             "break_contact": {"type": "integer", "minimum": 0},
                             "quote_ids": {"type": "array", "maxItems": 2, "items": _enum(qids)}},
              "required": ["headline_he", "narrative_he", "status", "status_note_he",
                           "break_contact", "quote_ids"],
              "additionalProperties": False}
    user = f"הסיפור:\n{table}\n\n(break_contact: מספר המגע שבו הטיפול השתבש, או 0)\n"
    return Prompt("story", SYSTEM_C, user, schema,
                  dict(inputs, contacts=list(contacts), qids=[q.qid for q in quotes]))


@dataclass
class StoryAnswer:
    headline_he: str
    narrative_he: str
    status: str
    status_note_he: str
    break_contact: int
    quote_ids: list[str]


def parse_story(raw: str, *, contacts: list[int], qids: list[str]) -> StoryAnswer:
    data = _load(raw)
    status = _choice(data, "status", STATUS)
    bc = data.get("break_contact") or 0
    if not isinstance(bc, int) or (bc and bc not in contacts):
        raise TaskError(f"break_contact חייב להיות 0 או אחד מ: {contacts}")
    ids = [q for q in (data.get("quote_ids") or []) if isinstance(q, str)]
    bad = [q for q in ids if q not in qids]
    if bad:
        raise TaskError(f"מזהי ציטוט לא קיימים: {bad}")
    headline = _text(data, "headline_he", 90)
    if not headline:
        raise TaskError("headline_he ריק")
    return StoryAnswer(headline, _text(data, "narrative_he", 700), status,
                       _text(data, "status_note_he", 160), bc, ids[:2])
