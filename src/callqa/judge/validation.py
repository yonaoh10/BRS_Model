"""Judge output validation: strict JSON schema + evidence-quote verification.

Anti-hallucination rule: every evidence quote must exist verbatim in the
redacted transcript (substring match after whitespace/punctuation
normalization). A response with any unverifiable quote is rejected.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from callqa.models import JudgeResponse, RedactedTranscript

_NORMALIZE_RE = re.compile(r"[\s\.,;:!\?\-–—'\"״׳\(\)\[\]<>]+")


class JudgeValidationError(ValueError):
    """Validation failure; str(err) is fed back to the model on retry."""


def normalize_for_match(text: str) -> str:
    return _NORMALIZE_RE.sub("", text)


def extract_json(raw: str) -> str:
    """Tolerate markdown fences / stray text around the JSON object."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence:
        return fence.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return raw


def parse_judge_response(raw: str) -> JudgeResponse:
    try:
        data = json.loads(extract_json(raw))
    except json.JSONDecodeError as exc:
        raise JudgeValidationError(f"output is not valid JSON: {exc}") from exc
    try:
        return JudgeResponse.model_validate(data)
    except ValidationError as exc:
        raise JudgeValidationError(f"output does not match the required schema: {exc}") from exc


def verify_evidence(response: JudgeResponse, redacted: RedactedTranscript) -> list[str]:
    """Return a list of human-readable problems (empty = all quotes verified)."""
    haystack = normalize_for_match("".join(turn.text for turn in redacted.turns))
    problems: list[str] = []
    for dim_id, dim_score in response.scores.items():
        for ev in dim_score.evidence:
            needle = normalize_for_match(ev.quote)
            if not needle:
                problems.append(f"dimension '{dim_id}': empty evidence quote")
            elif needle not in haystack:
                problems.append(
                    f"dimension '{dim_id}': quote not found verbatim in transcript: "
                    f"'{ev.quote[:80]}'"
                )
    return problems


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
