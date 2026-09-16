"""Judge output validation: strict JSON schema + evidence verification.

Anti-hallucination rule: every evidence quote must exist verbatim in the
redacted transcript, in a turn spoken by the speaker the judge attributes it
to. Both halves matter. A quote that exists somewhere in the call but not in
that speaker's mouth is how a customer's complaint gets published as the
banker's words, which is a worse error than an invented quote because it reads
as evidence.

A response with any unverifiable quote is rejected and the model is asked
again with the reason.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from callqa.models import JudgeResponse, RedactedTranscript

_NORMALIZE_RE = re.compile(r"[\s\.,;:!\?\-–—\'\"״׳\(\)\[\]<>]+")
# Niqqud and the bidi marks Hebrew text carries: present in one copy of a
# string and absent from the other, they make an identical quote look invented.
_INVISIBLE_RE = re.compile(r"[\u0591-\u05C7\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
# 1-4 minute digits to match the server-side schema pattern; the real bound on
# a fabricated timestamp is the call_end+60 check in verify_evidence.
_TIMESTAMP_RE = re.compile(r"^\d{1,4}:[0-5]\d$")
MIN_QUOTE_CHARS = 8


class JudgeValidationError(ValueError):
    """Validation failure; str(err) is fed back to the model on retry."""


def normalize_for_match(text: str) -> str:
    return _NORMALIZE_RE.sub("", _INVISIBLE_RE.sub("", text))


def extract_json(raw: str) -> str:
    """Tolerate markdown fences / stray text around the JSON object."""
    raw = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence:
        return fence.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return raw


def parse_judge_response(raw: str) -> JudgeResponse:
    if not isinstance(raw, str) or not raw.strip():
        raise JudgeValidationError("judge returned an empty response")
    try:
        data = json.loads(extract_json(raw))
    except json.JSONDecodeError as exc:
        raise JudgeValidationError(f"output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeValidationError("output is not a JSON object")
    try:
        return JudgeResponse.model_validate(data)
    except ValidationError as exc:
        raise JudgeValidationError(f"output does not match the required schema: {exc}") from exc


SNAP_MIN_SIMILARITY = 0.90

_DIGIT_SEQ_RE = re.compile(r"\d+")
_HEBREW_WORD_RE = re.compile(r"[֐-׿]+")
# Negation/polarity words: dropping or adding one of these inverts the meaning
# of a sentence, and a ~10% edit (all the 0.90 threshold requires) is enough to
# drop a short word like "לא". These must be identical on both sides of a snap.
_POLARITY_WORDS = frozenset({"לא", "אין", "אל", "ללא", "בלי", "אף", "שום", "לעולם"})


def _meaning_tokens(text: str) -> tuple[list[str], list[str]]:
    """(ordered digit sequences, sorted polarity words) - the parts of a quote
    a snap must not silently change."""
    digits = _DIGIT_SEQ_RE.findall(text)
    polarity = sorted(w for w in _HEBREW_WORD_RE.findall(text) if w in _POLARITY_WORDS)
    return digits, polarity


def _snap_preserves_meaning(quote: str, span: str) -> bool:
    """A snap may fix wording (a dropped conjunction, a reorder) but must never
    change a NUMBER or flip a NEGATION - those invert what the banker was
    quoted as saying while still clearing the fuzzy similarity threshold."""
    return _meaning_tokens(quote) == _meaning_tokens(span)


def _snap_quote_to_turn(quote: str, raw_turn: str) -> str | None:
    """Return the verbatim span of raw_turn best matching the quote, or None.

    Judges at the 7-14B tier produce NEAR-quotes: a dropped conjunction, a
    reordered word ("אני מזמין" for "ומזמין"). Rejecting those outright made
    an otherwise sound scorecard unobtainable, but accepting the model's text
    would put words in the transcript's mouth. The compromise keeps the
    guarantee: slide a word window over the REAL turn text, and if some
    window is >= SNAP_MIN_SIMILARITY similar (after match-normalisation),
    the evidence quote is REPLACED with that real window - the scorecard
    only ever contains text that was actually said. Cross-turn splices stay
    rejected because the window never leaves one turn.

    The similarity threshold alone cannot tell a harmless wording fix from a
    meaning INVERSION (a flipped fee "90"->"10", a dropped "לא") - both are
    small edits. So a candidate span is accepted only if it also preserves the
    quote's numbers and negations verbatim; otherwise the near-miss is treated
    as unverifiable and the call goes to human review, which is the safe
    outcome for a judge that misread the transcript.
    """
    from difflib import SequenceMatcher

    words = raw_turn.split()
    target = normalize_for_match(quote)
    if not words or not target:
        return None
    n_quote = max(1, len(quote.split()))
    best_ratio, best_span = 0.0, None
    for size in range(max(1, n_quote - 2), min(len(words), n_quote + 2) + 1):
        for start in range(0, len(words) - size + 1):
            span = " ".join(words[start:start + size])
            ratio = SequenceMatcher(None, target, normalize_for_match(span)).ratio()
            if ratio > best_ratio:
                best_ratio, best_span = ratio, span
    if best_span is None or best_ratio < SNAP_MIN_SIMILARITY:
        return None
    if not _snap_preserves_meaning(quote, best_span):
        return None
    return best_span


_SPEAKER_LABEL_RE = re.compile(r"\s*(?:בנקאי|לקוח)\s*:\s*")


def _same_speaker_blocks(redacted: RedactedTranscript) -> list[tuple[str, str]]:
    """(speaker, raw text) for each run of consecutive same-speaker turns.

    Verification used to check quotes per TURN, but ASR segmentation splits
    one utterance into consecutive turns mid-sentence ("...שביצעת ב-10
    לחודש," / "30 שקלים...") and the judge rightly quotes the natural
    sentence - which then failed as a "cross-turn splice". Text spanning
    consecutive turns of the SAME speaker is something that speaker actually
    said, contiguously; only a splice across a speaker change is invented.
    """
    blocks: list[tuple[str, str]] = []
    for turn in redacted.turns:
        if blocks and blocks[-1][0] == turn.speaker:
            blocks[-1] = (turn.speaker, blocks[-1][1] + " " + turn.text)
        else:
            blocks.append((turn.speaker, turn.text))
    return blocks


def verify_evidence(response: JudgeResponse, redacted: RedactedTranscript) -> list[str]:
    """Return a list of human-readable problems (empty = all quotes verified)."""
    # Per same-speaker block, never joined across a speaker change: a quote
    # spliced across two speakers is not something anybody said.
    raw_blocks = _same_speaker_blocks(redacted)
    turns = [(speaker, normalize_for_match(text)) for speaker, text in raw_blocks]
    call_end = max((t.end for t in redacted.turns), default=0.0)
    problems: list[str] = []
    for dim_id, dim_score in response.scores.items():
        if not dim_score.evidence:
            problems.append(f"dimension '{dim_id}': no evidence quote provided")
            continue
        for ev in dim_score.evidence:
            # The prompt renders turns as "[mm:ss] בנקאי: text", and models
            # copy the speaker labels into quotes - leading, and mid-quote
            # when they quote across our line breaks. The labels are a
            # formatting artifact of our own prompt, not invented content -
            # strip them everywhere rather than reject the quote.
            ev.quote = _SPEAKER_LABEL_RE.sub(" ", ev.quote).strip()
            needle = normalize_for_match(ev.quote)
            if len(needle) < MIN_QUOTE_CHARS:
                problems.append(
                    f"dimension '{dim_id}': evidence quote is too short to verify: "
                    f"'{ev.quote[:40]}'"
                )
                continue
            holders = [speaker for speaker, text in turns if needle in text]
            if not holders:
                # Near-verbatim rescue: snap to the real span if one exists.
                for _speaker, block_text in raw_blocks:
                    snapped = _snap_quote_to_turn(ev.quote, block_text)
                    if snapped is not None:
                        logging.getLogger(__name__).info(
                            "evidence quote snapped to transcript: %r -> %r",
                            ev.quote[:60], snapped[:60])
                        ev.quote = snapped
                        holders = [s for s, text in turns
                                   if normalize_for_match(snapped) in text]
                        break
            if not holders:
                problems.append(
                    f"dimension '{dim_id}': quote not found verbatim in transcript: "
                    f"'{ev.quote[:80]}'"
                )
            elif ev.speaker not in holders:
                problems.append(
                    f"dimension '{dim_id}': quote is attributed to {ev.speaker} but was "
                    f"said by {holders[0]}: '{ev.quote[:80]}'"
                )
            if not _TIMESTAMP_RE.match(ev.timestamp or ""):
                problems.append(
                    f"dimension '{dim_id}': timestamp '{ev.timestamp}' is not mm:ss"
                )
            elif _timestamp_seconds(ev.timestamp) > call_end + 60:
                problems.append(
                    f"dimension '{dim_id}': timestamp '{ev.timestamp}' is after the end "
                    f"of the call"
                )
    return problems


def _timestamp_seconds(value: str) -> float:
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + int(seconds)


def validate_judge_output(
    raw: str, expected_dimensions: list[str], redacted: RedactedTranscript
) -> JudgeResponse:
    """Parse + schema-validate + evidence-verify. Raises JudgeValidationError."""
    response = parse_judge_response(raw)
    missing = [d for d in expected_dimensions if d not in response.scores]
    extra = [d for d in response.scores if d not in expected_dimensions]
    if missing or extra:
        raise JudgeValidationError(
            f"scores keys mismatch: missing={missing} unexpected={extra}"
        )
    problems = verify_evidence(response, redacted)
    if problems:
        raise JudgeValidationError("evidence verification failed: " + " | ".join(problems))
    return response
