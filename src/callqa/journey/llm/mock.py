"""A deterministic stand-in for the language model: CI, demos, and dry runs.

It answers the same prompts with the same JSON a model would, from cue
phrases (journey_lexicon_he.yaml) and simple rules, so the whole chain -
parsing, quote verification, rules, the report - runs with no model at all.
Its "understanding" is shallow on purpose, and a report built from it is
labelled as a demo.
"""

from __future__ import annotations

import json

from callqa.journey.llm.tasks import Prompt
from callqa.journey.transcript_view import load_lexicon

BANK_PROMISE_CUES = (("callback_promise", "callback"), ("send_document", "send_document"))


def _quote(text: str, limit: int = 120) -> str:
    """Words from the start of a line, within the verifier's length cap."""
    out = []
    for w in text.split():
        if len(" ".join(out + [w])) > limit:
            break
        out.append(w)
    return " ".join(out) or text[:limit]


class MockContentEngine:
    name = "mock"
    model = "mock"

    def __init__(self) -> None:
        self.lex = load_lexicon()

    def answer(self, prompt: Prompt, feedback: str | None = None) -> str:
        fn = {"card": self._card, "returns": self._returns, "story": self._story}[prompt.task]
        return json.dumps(fn(prompt.inputs), ensure_ascii=False)

    # -- A
    def _card(self, x: dict) -> dict:
        lines = [ln for ln in x["lines"] if not ln["uncertain"]]
        hits = [(ln, self.lex.hits(ln["text"])) for ln in lines]

        def first(group: str, who: str | None = None):  # noqa: ANN202
            for ln, h in hits:
                if group in h and (who is None or ln["who"] == who):
                    return ln
            return None

        def ev(ln) -> list:  # noqa: ANN001
            return [{"line": ln["no"], "quote": _quote(ln["text"])}] if ln else []

        votes: dict[str, int] = {}
        for ln in lines:
            if ln["who"] == "customer":
                for t, n in self.lex.topic_votes(ln["text"]).items():
                    votes[t] = votes.get(t, 0) + n
        prev_topic = next((p["topic_id"] for p in reversed(x["previous"]) if p.get("topic_id")), None)
        topic = max(votes, key=lambda t: (votes[t], t)) if votes else (prev_topic or "other")
        if topic not in x["topics"]:
            topic = "other"
        commitments = []
        for group, kind in BANK_PROMISE_CUES:
            ln = first(group, "bank")
            if ln:
                commitments.append({"kind": kind, "by": "bank", "when_he": "", "ev": ev(ln)})
        act = first("customer_to_act", "bank")
        if act:
            commitments.append({"kind": "customer_to_act", "by": "customer", "when_he": "",
                                "ev": ev(act)})
        resolved = first("resolved", "bank")
        transfer = first("transfer")
        if resolved and not commitments:
            outcome, out_ln = "resolved", resolved
        elif any(c["by"] == "bank" for c in commitments) or transfer:
            outcome, out_ln = "not_resolved", transfer or None
        elif act:
            outcome, out_ln = "partially_resolved", act
        else:
            outcome, out_ln = "info_only", None
        prior = first("prior_contact", "customer") or first("broken_promise", "customer")
        retold_ln = first("retold", "customer")
        if x["position"] == 1:
            retold = "first_contact"
        elif retold_ln:
            retold = "yes"
        elif prior and first("no_history", "bank"):
            retold, retold_ln = "partial", prior
        else:
            retold = "no"
        aware = "yes" if first("history_known", "bank") else (
            "no" if first("no_history", "bank") else "unclear")
        redirect = "none"
        if transfer:
            redirect = "sent_to_branch" if "סניף" in transfer["text"] else "internal_transfer"
        anger = sum(1 for _ln, h in hits if h & {"frustration", "broken_promise"})
        return {"topic": topic, "issue_he": "", "customer_request_he": "",
                "outcome": outcome, "outcome_ev": ev(out_ln), "commitments": commitments,
                "prior_contact_mentioned": prior is not None, "prior_ev": ev(prior),
                "retold": retold, "retold_ev": ev(retold_ln) if retold in ("yes", "partial") else [],
                "banker_aware_of_history": aware, "redirect": redirect,
                "frustration": min(3, 1 + anger), "confidence": "medium"}

    # -- B
    def _returns(self, x: dict) -> dict:
        rows = {r["no"]: r for r in x["rows"]}
        out = []
        broke = False
        for no in x["to_classify"]:
            r = rows[no]
            card = r.get("card") or {}
            prev = [rows[n] for n in sorted(rows) if n < no]
            prev_card = next((p["card"] for p in reversed(prev) if p.get("card")), None) or {}
            prev_bank_promise = any(c["by"] == "bank" for c in prev_card.get("commitments", []))
            if r.get("broken_here") or (card.get("prior_contact_mentioned") and prev_bank_promise):
                cat, why = "unclosed_loop", "הלקוח חזר כי ההבטחה לחזור אליו לא קוימה."
            elif card.get("retold") in ("yes", "partial") or prev_card.get("redirect", "none") != "none":
                cat, why = "excessive_runaround", "הלקוח הועבר בין גורמים ונאלץ לחזור על עניינו."
            elif prev_card and card.get("topic") != prev_card.get("topic") \
                    and card.get("topic") != "other":
                cat, why = "new_topic", "הפנייה עוסקת בעניין אחר מזה של הפניות הקודמות."
            else:
                cat, why = "legit_return", "המשך טבעי של הטיפול בעניין."
            is_break = cat in ("unclosed_loop", "excessive_runaround") and not broke
            broke = broke or is_break
            qids = [q for q, c in x.get("quote_contacts", {}).items() if c == no][:1]
            out.append({"contact": no, "category": cat, "reason_he": why,
                        "is_break_point": is_break, "quote_ids": qids})
        return {"returns": out}

    # -- C
    def _story(self, x: dict) -> dict:
        rows = x["rows"]
        cats = [r.get("category") for r in rows if r.get("category")]
        last_card = next((r["card"] for r in reversed(rows) if r.get("card")), None)
        if "unclosed_loop" in cats:
            head = "הבטחה לחזור ללקוח שלא קוימה"
        elif "excessive_runaround" in cats:
            head = "לקוח שהועבר בין גורמים"
        elif "new_topic" in cats:
            head = "כמה עניינים נפרדים"
        elif cats:
            head = "תהליך שהתקדם שלב אחר שלב"
        else:
            head = "פנייה אחת"
        status = "unclear"
        if last_card and rows[-1].get("card"):
            status = {"resolved": "closed", "not_resolved": "open"}.get(last_card["outcome"],
                                                                       "unclear")
        n_ret = sum(1 for r in rows if r["no"] > 1)
        breaks = [r["no"] for r in rows if r.get("is_break")]
        text = (f"הלקוח פנה {len(rows)} פעמים" + (f", מהן {n_ret} חזרות" if n_ret else "")
                + ". " + {"closed": "העניין נפתר בסוף.", "open": "העניין נשאר פתוח.",
                          "unclear": "לא ברור מהנתונים איך העניין הסתיים."}[status])
        qids = [q for q, c in x.get("quote_contacts", {}).items() if breaks and c == breaks[0]][:1]
        return {"headline_he": head, "narrative_he": text, "status": status,
                "status_note_he": "", "break_contact": breaks[0] if breaks else 0,
                "quote_ids": qids}
