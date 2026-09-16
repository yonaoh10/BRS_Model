"""Judge orchestration: retries, self-consistency sampling, long-call
chunking, and scorecard assembly."""

from __future__ import annotations

import logging
import re
import statistics
import time
from datetime import UTC, datetime

from callqa.config import JudgeConfig
from callqa.judge.base import Judge, JudgeRequest
from callqa.judge.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT_HE,
    build_user_prompt,
    format_transcript,
    prompt_sha256,
)
from callqa.judge.validation import JudgeValidationError, validate_judge_output
from callqa.models import Features, JudgeResponse, RedactedTranscript, ScoreCard
from callqa.redaction import redact_text, sanitize_error
from callqa.rubric import Rubric, RubricDimension, weighted_total

logger = logging.getLogger(__name__)

GATE_CHUNK_SECONDS = 300.0  # identification/compliance occur early in the call


class NeedsHumanReviewError(RuntimeError):
    """Judge output could not be validated after all retries."""


def _judge_once_with_retries(
    judge: Judge,
    config: JudgeConfig,
    call_id: str,
    dimensions: list[RubricDimension],
    redacted: RedactedTranscript,
    features: Features,
    *,
    max_chars: int | None,
    max_seconds: float | None,
) -> tuple[JudgeResponse, str, int]:
    """Run one judging call with the validation-retry loop.

    Returns (validated response, user_prompt of the first attempt, retries used).
    """
    expected = [d.id for d in dimensions]
    validation_error: str | None = None
    first_prompt: str | None = None
    last_error: JudgeValidationError | None = None
    for attempt in range(config.max_retries + 1):
        user_prompt = build_user_prompt(
            dimensions,
            features,
            redacted,
            max_chars=max_chars,
            max_seconds=max_seconds,
            validation_error=validation_error,
        )
        if first_prompt is None:
            first_prompt = user_prompt
        request = JudgeRequest(
            call_id=call_id,
            system_prompt=SYSTEM_PROMPT_HE,
            user_prompt=user_prompt,
            dimensions=dimensions,
            redacted=redacted,
            features=features,
            attempt=attempt,
        )
        try:
            raw = judge.complete(request)
            response = validate_judge_output(raw, expected, redacted)
            return _scrub(response), first_prompt, attempt
        except JudgeValidationError as exc:
            last_error = exc
            # Fed back to the model on the next attempt, so it must not carry
            # transcript text that the redactor has not seen.
            validation_error = sanitize_error(exc, limit=2000)
            logger.info("judge validation failed: call_id=%s attempt=%d", call_id, attempt)
        except Exception as exc:  # noqa: BLE001
            # A refused connection, a 500, a truncated body: transport failures
            # used to escape the retry loop entirely and fail the call outright,
            # when the right outcome is the same held-for-review status as a
            # model that cannot produce valid output.
            last_error = exc
            validation_error = None
            logger.warning("judge call failed: call_id=%s attempt=%d: %s",
                           call_id, attempt, sanitize_error(exc, limit=300))
    raise NeedsHumanReviewError(
        f"judge output failed after {config.max_retries + 1} attempts: "
        f"{sanitize_error(last_error, limit=500)}"
    )


# Any "<...>" tag-like run in model prose: stripped so an unescaping downstream
# consumer of the stored JSON cannot be driven by it. Hebrew prose never
# legitimately contains HTML tags.
_TAG_RE = re.compile(r"<[^>]{0,200}>")


def _scrub(response: JudgeResponse) -> JudgeResponse:
    """Re-redact everything the model wrote in free text.

    Evidence quotes are verified against the redacted transcript, but the
    model's own prose is not: a judge that repeats an identifier it inferred,
    or that is steered into doing so, would otherwise put it straight into the
    report and the dashboard. The prose is also un-schema'd free text, so a
    model can emit HTML/markup into it; the report autoescapes, but the stored
    JSON feeds other consumers (exports, a summariser) that may not, so any
    angle-bracket tag-like run is stripped here at the source.
    """
    def clean(text: str) -> str:
        return _TAG_RE.sub("", redact_text(text or "")[0])

    for dim in response.scores.values():
        dim.reasoning_he = clean(dim.reasoning_he)
        for ev in dim.evidence:
            ev.quote = clean(ev.quote)
    response.strengths_he = [clean(s) for s in response.strengths_he]
    response.development_area_he = clean(response.development_area_he)
    response.summary_he = clean(response.summary_he)
    return response


def _merge_samples(samples: list[JudgeResponse]) -> JudgeResponse:
    """Self-consistency: median score per dimension; reasoning/evidence taken
    from the sample whose score is closest to the median for that dimension."""
    if len(samples) == 1:
        return samples[0]
    merged_scores = {}
    for dim_id in samples[0].scores:
        values = [s.scores[dim_id].score for s in samples]
        # median_low, not round(median): on an even number of samples a tie
        # otherwise rounds in whichever direction the pair happens to sit,
        # so [2,3] capped the call at 59 while [3,4] rounded up to 4. The low
        # value is both deterministic and the safer reading of a disagreement.
        median = int(statistics.median_low(values))
        closest = min(samples, key=lambda s: abs(s.scores[dim_id].score - median))
        chosen = closest.scores[dim_id].model_copy(deep=True)
        if chosen.score != median:
            # Never show a reasoning that argues for a different number than
            # the one printed beside it.
            chosen.reasoning_he = (
                f"{chosen.reasoning_he} [הציון הסופי {median} הוא החציון של "
                f"{len(values)} הערכות: {sorted(values)}]"
            )
        chosen.score = median
        merged_scores[dim_id] = chosen
    base = samples[0]
    return JudgeResponse(
        scores=merged_scores,
        strengths_he=base.strengths_he,
        development_area_he=base.development_area_he,
        summary_he=base.summary_he,
    )


def run_judge(
    judge: Judge,
    config: JudgeConfig,
    rubric: Rubric,
    call_id: str,
    banker_id: str,
    redacted: RedactedTranscript,
    features: Features,
) -> ScoreCard:
    """Score one call. Raises NeedsHumanReviewError on unrecoverable validation
    failure (the pipeline maps it to the needs_human_review status)."""
    start_time = time.monotonic()
    full_text = format_transcript(redacted)
    truncated = len(full_text) > config.max_transcript_chars

    total_retries = 0
    prompts_for_hash: list[str] = [SYSTEM_PROMPT_HE]
    samples: list[JudgeResponse] = []

    for _sample in range(config.n_samples):
        if truncated:
            # Long-call rule: gates scored on the first 5 minutes of the full
            # transcript; other dimensions on a length-capped transcript.
            gate_dims = [d for d in rubric.dimensions if d.gate]
            other_dims = [d for d in rubric.dimensions if not d.gate]
            gate_resp, gate_prompt, r1 = _judge_once_with_retries(
                judge, config, call_id, gate_dims, redacted, features,
                max_chars=None, max_seconds=GATE_CHUNK_SECONDS,
            )
            other_resp, other_prompt, r2 = _judge_once_with_retries(
                judge, config, call_id, other_dims, redacted, features,
                max_chars=config.max_transcript_chars, max_seconds=None,
            )
            total_retries += r1 + r2
            if not samples:
                prompts_for_hash += [gate_prompt, other_prompt]
            merged = JudgeResponse(
                scores={**gate_resp.scores, **other_resp.scores},
                strengths_he=other_resp.strengths_he,
                development_area_he=other_resp.development_area_he,
                summary_he=other_resp.summary_he,
            )
            samples.append(merged)
        else:
            resp, prompt, retries = _judge_once_with_retries(
                judge, config, call_id, rubric.dimensions, redacted, features,
                max_chars=None, max_seconds=None,
            )
            total_retries += retries
            if not samples:
                prompts_for_hash.append(prompt)
            samples.append(resp)

    response = _merge_samples(samples)
    total, gate_failed, failed_gates = weighted_total(rubric, response.scores)
    scorecard = ScoreCard(
        call_id=call_id,
        banker_id=banker_id,
        scores=response.scores,
        weighted_total=total,
        gate_failed=gate_failed,
        failed_gates=failed_gates,
        strengths_he=response.strengths_he,
        development_area_he=response.development_area_he,
        summary_he=response.summary_he,
        judge_engine=judge.name,
        model=config.model if judge.name == "vllm" else f"mock:{judge.name}",
        prompt_sha256=prompt_sha256(*prompts_for_hash),
        prompt_version=PROMPT_VERSION,
        rubric_sha256=rubric.sha256,
        retries=total_retries,
        latency_sec=round(time.monotonic() - start_time, 3),
        n_samples=config.n_samples,
        truncated=truncated,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    logger.info(
        "judge done: call_id=%s total=%.1f gate_failed=%s retries=%d",
        call_id, total, gate_failed, total_retries,
    )
    return scorecard
