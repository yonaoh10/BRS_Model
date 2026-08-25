"""Unit tests: judge validation, evidence verification, retries, mock judge."""

from __future__ import annotations

import json

import pytest

from callqa.config import JudgeConfig
from callqa.judge.mock_judge import MockJudge
from callqa.judge.runner import NeedsHumanReviewError, run_judge
from callqa.judge.validation import (
    JudgeValidationError,
    normalize_for_match,
    parse_judge_response,
    validate_judge_output,
    verify_evidence,
)
from callqa.models import (
    DialogTranscript,
    DialogTurn,
    RedactedTranscript,
    RedactedTurn,
)
from callqa.rubric import load_rubric


@pytest.fixture(scope="module")
def rubric():
    return load_rubric("config/rubric.yaml")


def make_redacted(call_id: str = "T1") -> RedactedTranscript:
    turns = [
        RedactedTurn(speaker="banker", start=0.0, end=4.0,
                     text="שלום, במה אפשר לעזור לך היום?"),
        RedactedTurn(speaker="customer", start=5.0, end=9.0,
                     text="אני רוצה לבדוק את החשבון שלי."),
        RedactedTurn(speaker="banker", start=10.0, end=15.0,
                     text="בשמחה, הזיהוי הושלם ואפשר להתקדם."),
    ]
    return RedactedTranscript(call_id=call_id, engine="regex", turns=turns)


def make_features(call_id: str = "T1"):
    from callqa.features import compute_features

    dialog = DialogTranscript(
        call_id=call_id,
        attribution_mode="mock",
        turns=[DialogTurn(**t.model_dump()) for t in make_redacted(call_id).turns],
    )
    return compute_features(call_id, [], [], dialog, 60.0)


def response_json(rubric, quote: str) -> str:
    return json.dumps(
        {
            "scores": {
                d.id: {
                    "score": 4,
                    "reasoning_he": "נימוק",
                    "evidence": [{"quote": quote, "timestamp": "00:10", "speaker": "banker"}],
                }
                for d in rubric.dimensions
            },
            "strengths_he": ["חוזק"],
            "development_area_he": "פיתוח",
            "summary_he": "סיכום",
        },
        ensure_ascii=False,
    )


def test_verifier_accepts_verbatim_quote(rubric) -> None:
    redacted = make_redacted()
    raw = response_json(rubric, "הזיהוי הושלם ואפשר להתקדם")
    response = validate_judge_output(raw, [d.id for d in rubric.dimensions], redacted)
    assert response.scores["empathy"].score == 4


def test_verifier_tolerates_whitespace_and_punctuation(rubric) -> None:
    redacted = make_redacted()
    # Same words, different punctuation/spacing - normalization should match.
    raw = response_json(rubric, "שלום  במה אפשר לעזור לך היום")
    validate_judge_output(raw, [d.id for d in rubric.dimensions], redacted)


def test_verifier_rejects_fabricated_quote(rubric) -> None:
    redacted = make_redacted()
    raw = response_json(rubric, "ציטוט מומצא שלא נאמר בשיחה")
    with pytest.raises(JudgeValidationError, match="not found verbatim"):
        validate_judge_output(raw, [d.id for d in rubric.dimensions], redacted)


def test_verifier_rejects_missing_dimension(rubric) -> None:
    redacted = make_redacted()
    data = json.loads(response_json(rubric, "שלום"))
    del data["scores"]["closure"]
    with pytest.raises(JudgeValidationError, match="missing"):
        validate_judge_output(json.dumps(data), [d.id for d in rubric.dimensions], redacted)


def test_parse_rejects_non_json() -> None:
    with pytest.raises(JudgeValidationError, match="not valid JSON"):
        parse_judge_response("this is not json at all")


def test_parse_tolerates_markdown_fences(rubric) -> None:
    raw = "```json\n" + response_json(rubric, "שלום") + "\n```"
    parse_judge_response(raw)


def test_verify_evidence_reports_all_problems(rubric) -> None:
    redacted = make_redacted()
    data = json.loads(response_json(rubric, "ציטוט שגוי לחלוטין"))
    response = parse_judge_response(json.dumps(data))
    problems = verify_evidence(response, redacted)
    assert len(problems) == len(rubric.dimensions)


def test_normalize_for_match() -> None:
    assert normalize_for_match("שלום,  עולם!") == normalize_for_match("שלום עולם")


def test_mock_judge_deterministic(rubric) -> None:
    config = JudgeConfig(engine="mock", model="mock")
    redacted = make_redacted("CALL_X")
    features = make_features("CALL_X")
    card1 = run_judge(MockJudge(), config, rubric, "CALL_X", "B1", redacted, features)
    card2 = run_judge(MockJudge(), config, rubric, "CALL_X", "B1", redacted, features)
    assert {k: v.score for k, v in card1.scores.items()} == {
        k: v.score for k, v in card2.scores.items()
    }
    assert card1.weighted_total == card2.weighted_total
    assert card1.prompt_sha256 == card2.prompt_sha256


def test_retry_path_recovers_from_bad_quote(rubric) -> None:
    config = JudgeConfig(engine="mock", model="mock", max_retries=3)
    redacted = make_redacted("RETRY_CALL")
    features = make_features("RETRY_CALL")
    judge = MockJudge(fail_first_for={"RETRY_CALL"})
    card = run_judge(judge, config, rubric, "RETRY_CALL", "B1", redacted, features)
    assert card.retries == 1  # first attempt rejected, second verified


def test_needs_human_review_after_exhausted_retries(rubric) -> None:
    class AlwaysBadJudge:
        name = "bad"

        def complete(self, request):  # noqa: ANN001
            return "not json {{{"

    config = JudgeConfig(engine="mock", model="mock", max_retries=1)
    redacted = make_redacted("BAD_CALL")
    features = make_features("BAD_CALL")
    with pytest.raises(NeedsHumanReviewError):
        run_judge(AlwaysBadJudge(), config, rubric, "BAD_CALL", "B1", redacted, features)


def test_self_consistency_median(rubric) -> None:
    config = JudgeConfig(engine="mock", model="mock", n_samples=3)
    redacted = make_redacted("SC_CALL")
    features = make_features("SC_CALL")
    card = run_judge(MockJudge(), config, rubric, "SC_CALL", "B1", redacted, features)
    assert card.n_samples == 3
    # Mock is deterministic, so the median equals the single-sample score.
    single = run_judge(
        MockJudge(), JudgeConfig(engine="mock", model="mock"), rubric,
        "SC_CALL", "B1", redacted, features,
    )
    assert {k: v.score for k, v in card.scores.items()} == {
        k: v.score for k, v in single.scores.items()
    }
