"""Prompt builder for the LLM judge (versioned, Hebrew).

The prompt contains: role framing, the full rubric with anchors, the
objective features block, and the redacted dialog transcript with
[mm:ss] speaker prefixes. The model must reason per dimension first
(chain-of-thought) and then emit strict JSON only.
"""

from __future__ import annotations

import hashlib

from callqa.models import Features, RedactedTranscript
from callqa.rubric import RubricDimension

PROMPT_VERSION = "1.0.0"

SPEAKER_HE = {"banker": "בנקאי", "customer": "לקוח"}

SYSTEM_PROMPT_HE = (
    "אתה מעריך איכות שיחות שירות בבנק. אתה מקבל תמליל שיחה בין בנקאי ללקוח, "
    "מחוון הערכה עם עוגנים התנהגותיים, ונתונים אובייקטיביים שחושבו מהשיחה. "
    "עליך להעריך את הבנקאי בלבד, על סמך המחוון בלבד, בהתבסס אך ורק על מה שנאמר בתמליל. "
    "כל ציטוט (quote) חייב להופיע בתמליל מילה במילה - אסור להמציא, לקצר או לנסח מחדש ציטוטים. "
    "כתוב תחילה ניתוח קצר לכל ממד בשדה reasoning_he, ולאחר מכן החזר JSON תקין בלבד, "
    "ללא טקסט נוסף לפני או אחרי."
)


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def format_transcript(redacted: RedactedTranscript, max_chars: int | None = None,
                      max_seconds: float | None = None) -> str:
    lines = []
    for turn in redacted.turns:
        if max_seconds is not None and turn.start > max_seconds:
            break
        lines.append(f"[{mmss(turn.start)}] {SPEAKER_HE[turn.speaker]}: {turn.text}")
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0]
    return text


def format_rubric(dimensions: list[RubricDimension]) -> str:
    blocks = []
    for d in dimensions:
        gate_note = " (ממד שער: ציון 2 ומטה מגביל את הציון הכולל)" if d.gate else ""
        anchor_lines = "\n".join(
            f"  ציון {level}: {text}" for level, text in sorted(d.anchors.items())
        )
        feature_note = (
            f"\n  ממד משולב: בסס את הציון גם על המדדים האובייקטיביים: {', '.join(d.features)}."
            if d.signal == "hybrid" and d.features
            else ""
        )
        blocks.append(f"- {d.id} | {d.name_he} | משקל {d.weight}{gate_note}\n{anchor_lines}{feature_note}")
    return "\n".join(blocks)


def format_features(features: Features) -> str:
    patience = (
        f"{features.patience_median_sec:.1f} שניות"
        if features.patience_median_sec is not None
        else "לא נמדד"
    )
    return "\n".join(
        [
            f"- יחס דיבור של הבנקאי (talk_ratio): {features.talk_ratio:.2f}",
            f"- המונולוג הארוך ביותר של הבנקאי: {features.longest_banker_monologue_sec:.1f} שניות",
            f"- קטיעות של הבנקאי את הלקוח: {features.interruptions_by_banker}",
            f"- קטיעות של הלקוח את הבנקאי: {features.interruptions_by_customer}",
            f"- חציון זמן המתנה לפני מענה (patience): {patience}",
            f"- שאלות של הבנקאי: {features.banker_question_count}"
            f" ({features.banker_questions_per_minute:.2f} לדקה)",
            f"- קצב דיבור (מילים לדקה): בנקאי {features.speech_rate_wpm.banker:.0f},"
            f" לקוח {features.speech_rate_wpm.customer:.0f}",
            f"- זמן שקט כולל (dead air): {features.dead_air_total_sec:.1f} שניות",
        ]
    )


def _json_schema_block(dimensions: list[RubricDimension]) -> str:
    dim_ids = ", ".join(f'"{d.id}"' for d in dimensions)
    return (
        "החזר JSON יחיד במבנה הבא (מפתחות scores חייבים להיות בדיוק: "
        f"{dim_ids}):\n"
        "{\n"
        '  "scores": {"<dim_id>": {"score": 1-5, "reasoning_he": "...",\n'
        '     "evidence": [{"quote": "ציטוט מילה במילה מהתמליל", "timestamp": "mm:ss",\n'
        '                    "speaker": "banker|customer"}]}},\n'
        '  "strengths_he": ["..."],\n'
        '  "development_area_he": "...",\n'
        '  "summary_he": "..."\n'
        "}"
    )


def build_user_prompt(
    dimensions: list[RubricDimension],
    features: Features,
    redacted: RedactedTranscript,
    *,
    max_chars: int | None = None,
    max_seconds: float | None = None,
    validation_error: str | None = None,
) -> str:
    parts = [
        "## מחוון ההערכה",
        format_rubric(dimensions),
        "",
        "## מדדים אובייקטיביים שחושבו מהשיחה",
        format_features(features),
        "",
        "## תמליל השיחה (לאחר הסרת פרטים מזהים)",
        format_transcript(redacted, max_chars=max_chars, max_seconds=max_seconds),
        "",
        "## הנחיות פלט",
        _json_schema_block(dimensions),
    ]
    if validation_error:
        parts += [
            "",
            "## שגיאת אימות בניסיון הקודם - חובה לתקן",
            validation_error,
            "זכור: כל ציטוט חייב להופיע בתמליל למעלה מילה במילה.",
        ]
    return "\n".join(parts)


def prompt_sha256(*prompts: str) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()
