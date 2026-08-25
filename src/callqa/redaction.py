"""PII redaction (stage 5), Hebrew-aware.

Two engines behind one interface:
- RegexRedactor: dependency-free, implements ALL mandatory recognizers
  (Israeli ID with checksum, Israeli phones, payment cards via Luhn,
  aggressive ACCOUNT_LIKE, metadata-based names). Used in mock mode and as
  a fallback when presidio is unavailable.
- PresidioRedactor: presidio analyzer/anonymizer wired with the same custom
  recognizers (lazy import; server deployment).

Replacement format: <סוג:████> e.g. <ת"ז:████>. Turn-level timestamps are
untouched, so evidence-quote timestamps stay valid.

Only redacted text may reach the judge, reports, or logs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from callqa.config import RedactionConfig
from callqa.models import DialogTranscript, RedactedTranscript, RedactedTurn

logger = logging.getLogger(__name__)

# Entity type -> Hebrew label used in the replacement token.
ENTITY_LABELS_HE = {
    "ISRAELI_ID": 'ת"ז',
    "PHONE": "טלפון",
    "CREDIT_CARD": "כרטיס אשראי",
    "ACCOUNT_LIKE": "חשבון",
    "PERSON": "שם",
}

MASK = "████"


def replacement_token(entity_type: str) -> str:
    return f"<{ENTITY_LABELS_HE.get(entity_type, entity_type)}:{MASK}>"


# -- validators --------------------------------------------------------------

def is_valid_israeli_id(value: str) -> bool:
    """Israeli ID (teudat zehut) checksum.

    Left-pad to 9 digits; multiply digits alternately x1, x2; if a product
    exceeds 9, sum its digits (equivalent to product - 9); valid iff the
    total is divisible by 10.
    """
    digits = re.sub(r"\D", "", value)
    if not 8 <= len(digits) <= 9:
        return False
    digits = digits.zfill(9)
    total = 0
    for i, ch in enumerate(digits):
        product = int(ch) * (1 if i % 2 == 0 else 2)
        if product > 9:
            product -= 9
        total += product
    return total % 10 == 0


def is_valid_luhn(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# -- pattern recognizers (shared by both engines) ----------------------------

# Israeli mobile (05X) and landline (0X / 07X), optional +972, hyphens/spaces.
PHONE_RE = re.compile(r"(?<!\d)(?:\+972[-\s]?0?|0)(?:5\d|7\d|[23489])(?:[-\s]?\d){7}(?!\d)")
ISRAELI_ID_RE = re.compile(r"(?<!\d)\d{8,9}(?!\d)")
CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[-\s]?){12,18}\d(?!\d)")
ACCOUNT_LIKE_RE = re.compile(r"(?<!\d)\d{6,}(?!\d)")


@dataclass
class PIIMatch:
    entity_type: str
    start: int
    end: int


def find_pii(text: str) -> list[PIIMatch]:
    """Find PII spans; earlier recognizers win on overlap.

    Priority: PHONE > ISRAELI_ID > CREDIT_CARD > ACCOUNT_LIKE. ACCOUNT_LIKE is
    aggressive on purpose (6+ digits) - PoC bias is privacy.
    """
    recognizers: list[tuple[str, re.Pattern[str], object]] = [
        ("PHONE", PHONE_RE, None),
        ("ISRAELI_ID", ISRAELI_ID_RE, is_valid_israeli_id),
        ("CREDIT_CARD", CREDIT_CARD_RE, is_valid_luhn),
        ("ACCOUNT_LIKE", ACCOUNT_LIKE_RE, None),
    ]
    accepted: list[PIIMatch] = []
    for entity_type, pattern, validator in recognizers:
        for m in pattern.finditer(text):
            if validator is not None and not validator(m.group()):  # type: ignore[operator]
                continue
            if any(m.start() < a.end and a.start < m.end() for a in accepted):
                continue  # overlaps a higher-priority match
            accepted.append(PIIMatch(entity_type, m.start(), m.end()))
    accepted.sort(key=lambda a: a.start)
    return accepted


def redact_text(text: str, extra_names: list[str] | None = None) -> tuple[str, dict[str, int]]:
    """Redact one string; returns (redacted_text, counts_by_entity_type)."""
    counts: dict[str, int] = {}
    matches = find_pii(text)
    out: list[str] = []
    cursor = 0
    for m in matches:
        out.append(text[cursor:m.start])
        out.append(replacement_token(m.entity_type))
        counts[m.entity_type] = counts.get(m.entity_type, 0) + 1
        cursor = m.end
    out.append(text[cursor:])
    redacted = "".join(out)
    for name in extra_names or []:
        name = name.strip()
        if len(name) >= 2 and name in redacted:
            occurrences = redacted.count(name)
            redacted = redacted.replace(name, replacement_token("PERSON"))
            counts["PERSON"] = counts.get("PERSON", 0) + occurrences
    return redacted, counts


class Redactor(Protocol):
    name: str

    def redact_dialog(
        self, dialog: DialogTranscript, extra_names: list[str] | None = None
    ) -> RedactedTranscript: ...


class RegexRedactor:
    """Dependency-free redactor implementing all mandatory recognizers."""

    name = "regex"

    def __init__(self, config: RedactionConfig | None = None) -> None:
        self.config = config or RedactionConfig()

    def redact_dialog(
        self, dialog: DialogTranscript, extra_names: list[str] | None = None
    ) -> RedactedTranscript:
        turns: list[RedactedTurn] = []
        totals: dict[str, int] = {}
        for turn in dialog.turns:
            if self.config.enabled:
                text, counts = redact_text(turn.text, extra_names)
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
            else:
                text = turn.text
            turns.append(
                RedactedTurn(speaker=turn.speaker, start=turn.start, end=turn.end, text=text)
            )
        logger.info(
            "redaction done: call_id=%s entities=%d", dialog.call_id, sum(totals.values())
        )
        return RedactedTranscript(
            call_id=dialog.call_id, engine=self.name, turns=turns, redaction_counts=totals
        )


class PresidioRedactor:
    """Presidio analyzer+anonymizer with the same custom recognizers.

    Lazy import - only instantiated on the server (never in mock mode).
    Optional NER person redaction (DictaBERT-NER) when redaction.ner is true.
    """

    name = "presidio"

    def __init__(self, config: RedactionConfig) -> None:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

        self.config = config
        self._fallback = RegexRedactor(config)
        registry_recognizers = [
            PatternRecognizer(
                supported_entity="PHONE",
                patterns=[Pattern("il_phone", PHONE_RE.pattern, 0.6)],
            ),
            PatternRecognizer(
                supported_entity="ISRAELI_ID",
                patterns=[Pattern("il_id", ISRAELI_ID_RE.pattern, 0.5)],
            ),
            PatternRecognizer(
                supported_entity="ACCOUNT_LIKE",
                patterns=[Pattern("account_like", ACCOUNT_LIKE_RE.pattern, 0.3)],
            ),
        ]
        self.analyzer = AnalyzerEngine()
        for rec in registry_recognizers:
            self.analyzer.registry.add_recognizer(rec)
        self._ner = None
        if config.ner:
            self._ner = self._load_ner()

    def _load_ner(self):  # noqa: ANN202
        try:
            from transformers import pipeline as hf_pipeline  # lazy import

            return hf_pipeline("ner", model="models/dictabert-ner", aggregation_strategy="simple")
        except Exception as exc:  # pragma: no cover - server-only path
            logger.warning("DictaBERT-NER unavailable (%s); NER redaction disabled", exc)
            return None

    def redact_dialog(
        self, dialog: DialogTranscript, extra_names: list[str] | None = None
    ) -> RedactedTranscript:
        # Presidio analyzer results feed the same span-replacement logic; the
        # checksum/Luhn validation from the regex engine still applies via
        # find_pii, and presidio's CREDIT_CARD recognizer adds coverage.
        turns: list[RedactedTurn] = []
        totals: dict[str, int] = {}
        for turn in dialog.turns:
            if not self.config.enabled:
                turns.append(RedactedTurn(speaker=turn.speaker, start=turn.start,
                                          end=turn.end, text=turn.text))
                continue
            text, counts = redact_text(turn.text, extra_names)
            results = self.analyzer.analyze(text=text, language="en",
                                            entities=["CREDIT_CARD"])
            for res in sorted(results, key=lambda r: r.start, reverse=True):
                text = text[: res.start] + replacement_token("CREDIT_CARD") + text[res.end:]
                counts["CREDIT_CARD"] = counts.get("CREDIT_CARD", 0) + 1
            if self._ner is not None:
                for ent in sorted(self._ner(text), key=lambda e: e["start"], reverse=True):
                    if ent.get("entity_group") in ("PER", "PERSON"):
                        text = text[: ent["start"]] + replacement_token("PERSON") + text[ent["end"]:]
                        counts["PERSON"] = counts.get("PERSON", 0) + 1
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            turns.append(
                RedactedTurn(speaker=turn.speaker, start=turn.start, end=turn.end, text=text)
            )
        return RedactedTranscript(
            call_id=dialog.call_id, engine=self.name, turns=turns, redaction_counts=totals
        )


def build_redactor(config: RedactionConfig, mock: bool) -> Redactor:
    if mock:
        return RegexRedactor(config)
    try:
        return PresidioRedactor(config)
    except ImportError:
        logger.warning("presidio not installed; using built-in regex redactor")
        return RegexRedactor(config)
