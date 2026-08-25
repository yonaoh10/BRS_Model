"""Deterministic mock judge.

Scores are derived from a stable hash of (call_id, dimension_id) so the same
call always gets the same scorecard, across processes. Evidence quotes are
taken verbatim from the redacted transcript so evidence verification passes.

`fail_first_for` makes the first attempt for the listed call_ids return an
unverifiable quote - used by tests to exercise the retry path.
"""

from __future__ import annotations

import hashlib
import json

from callqa.judge.base import JudgeRequest
from callqa.models import RedactedTurn

REASONING_TEMPLATES_HE = {
    "identification": "הבנקאי ביקש פרטים מזהים בתחילת השיחה ואישר את הזיהוי לפני מסירת מידע.",
    "compliance": "הבנקאי ציין עמלות ותנאים רלוונטיים במהלך השיחה, בהתאם לעוגני המחוון.",
    "empathy": "הבנקאי הגיב לדברי הלקוח בטון מותאם והתייחס לקושי שהוצג.",
    "listening": "בהתבסס על המדדים האובייקטיביים (יחס דיבור, קטיעות והמתנה) ועל מהלך השיחה.",
    "clarity": "ההסברים נמסרו בשפה ברורה ובשלבים, כולל סכומים ומועדים.",
    "resolution": "הפנייה טופלה במהלך השיחה או שהוגדר מסלול המשך ברור.",
    "suitability": "המענה שהוצע נקשר לצורך שהלקוח תיאר בשיחה.",
    "closure": "השיחה נסגרה עם סיכום וצעד הבא, בהתאם לעוגני המחוון.",
}


def _stable_int(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class MockJudge:
    name = "mock"

    def __init__(self, fail_first_for: set[str] | None = None) -> None:
        self.fail_first_for = fail_first_for or set()
        self._attempts: dict[str, int] = {}

    def check_connectivity(self) -> None:  # parity with VLLMJudge
        return

    def _pick_turn(self, turns: list[RedactedTurn], seed: int, speaker: str) -> RedactedTurn | None:
        candidates = [t for t in turns if t.speaker == speaker and len(t.text) > 10]
        if not candidates:
            candidates = [t for t in turns if len(t.text) > 0]
        if not candidates:
            return None
        return candidates[seed % len(candidates)]

    def complete(self, request: JudgeRequest) -> str:
        call_id = request.call_id
        attempt_key = f"{call_id}:{len(request.dimensions)}"
        attempt = self._attempts.get(attempt_key, 0)
        self._attempts[attempt_key] = attempt + 1
        inject_bad_quote = call_id in self.fail_first_for and attempt == 0

        from callqa.judge.prompts import mmss  # local import avoids cycle at module load

        scores: dict[str, dict] = {}
        dim_scores: dict[str, int] = {}
        for dim in request.dimensions:
            seed = _stable_int(call_id, dim.id)
            score = 2 + seed % 4  # 2..5, deterministic; some calls fail a gate
            dim_scores[dim.id] = score
            speaker = "banker" if seed % 3 != 0 else "customer"
            turn = self._pick_turn(request.redacted.turns, seed, speaker)
            evidence = []
            if turn is not None:
                quote = turn.text if not inject_bad_quote else "ציטוט שאינו קיים בתמליל כלל"
                evidence.append(
                    {"quote": quote, "timestamp": mmss(turn.start), "speaker": turn.speaker}
                )
                inject_bad_quote = False  # one bad quote is enough to fail validation
            scores[dim.id] = {
                "score": score,
                "reasoning_he": REASONING_TEMPLATES_HE.get(
                    dim.id, "הציון נקבע על סמך עוגני המחוון והתמליל."
                ),
                "evidence": evidence,
            }

        best = max(dim_scores, key=lambda k: dim_scores[k])
        worst = min(dim_scores, key=lambda k: dim_scores[k])
        name_he = {d.id: d.name_he for d in request.dimensions}
        response = {
            "scores": scores,
            "strengths_he": [f"ביצוע טוב בממד {name_he[best]}."],
            "development_area_he": f"נדרש חיזוק בממד {name_he[worst]}.",
            "summary_he": "שיחה עניינית; הבנקאי טיפל בפניית הלקוח בהתאם לנהלים, עם נקודות לשיפור.",
        }
        return json.dumps(response, ensure_ascii=False)
