"""Unit tests: feature math on hand-built segment fixtures (exact values)."""

from __future__ import annotations

from callqa.features import (
    compute_features,
    count_banker_questions,
    count_interruptions,
    dead_air_total,
    longest_monologue,
    patience_median,
)
from callqa.models import DialogTranscript, DialogTurn, VADSegment


def seg(start: float, end: float) -> VADSegment:
    return VADSegment(start=start, end=end)


# Hand-built fixture:
# banker:   [0,10], [11,20], [30,40]           -> 29s speech
# customer: [10.2,10.8], [20,25], [41,45]      -> 9.6s speech
BANKER = [seg(0, 10), seg(11, 20), seg(30, 40)]
CUSTOMER = [seg(10.2, 10.8), seg(20, 25), seg(41, 45)]


def test_talk_ratio_exact() -> None:
    dialog = DialogTranscript(call_id="T", attribution_mode="mock", turns=[])
    f = compute_features("T", BANKER, CUSTOMER, dialog, call_duration_sec=45.0)
    assert f.banker_speech_sec == 29.0
    assert f.customer_speech_sec == 9.6
    assert f.talk_ratio == round(29.0 / 38.6, 4)  # 0.7513


def test_longest_monologue_exact() -> None:
    # Gap [10,11] contains 0.6s of customer speech (<1.0s) -> run continues;
    # gap [20,30] contains 5s (>=1.0s) -> run breaks.
    # Run 1 spans 0..20 = 20s; run 2 spans 30..40 = 10s.
    assert longest_monologue(BANKER, CUSTOMER) == 20.0


def test_interruptions_exact() -> None:
    assert count_interruptions(BANKER, CUSTOMER) == 0
    assert count_interruptions(CUSTOMER, BANKER) == 0
    # Customer starts at 15 while banker has been speaking since 11 (4s >= 1s)
    # and overlap is 3s (>= 0.5s) -> one interruption by customer.
    customer = [seg(15, 18)]
    assert count_interruptions(customer, BANKER) == 1
    # Onset too early into the other's speech (0.5s < 1.0s): not counted.
    assert count_interruptions([seg(11.5, 14)], BANKER) == 0
    # Overlap too short (0.3s < 0.5s): not counted.
    assert count_interruptions([seg(19.7, 25)], BANKER) == 0


def test_patience_median_exact() -> None:
    # customer->banker gaps: 10.8->11 = 0.2s, 25->30 = 5s -> median 2.6.
    assert patience_median(BANKER, CUSTOMER) == 2.6
    # Gaps are capped at 10s.
    assert patience_median([seg(50, 55)], [seg(0, 10)]) == 10.0
    # No customer->banker pair -> None.
    assert patience_median([seg(0, 5)], [seg(6, 8)]) is None


def test_dead_air_exact() -> None:
    # Union: [0,10],[10.2,10.8],[11,25],[30,40],[41,45]
    # Gaps: 0.2, 0.2, 5, 1 -> only the 5s gap exceeds 3s.
    assert dead_air_total(BANKER, CUSTOMER) == 5.0


def test_banker_question_count() -> None:
    dialog = DialogTranscript(
        call_id="T",
        attribution_mode="mock",
        turns=[
            DialogTurn(speaker="banker", start=0, end=5,
                       text="מה השעה? אני כאן. כמה זה עולה."),
            DialogTurn(speaker="customer", start=5, end=8, text="מה זה? למה?"),
            DialogTurn(speaker="banker", start=8, end=12, text="תודה רבה לך."),
        ],
    )
    # banker: "מה השעה?" (ends with ?), "כמה זה עולה." (interrogative start) -> 2.
    # Customer questions never counted.
    assert count_banker_questions(dialog) == 2


def test_speech_rate_and_questions_per_minute() -> None:
    dialog = DialogTranscript(
        call_id="T",
        attribution_mode="mock",
        turns=[
            DialogTurn(speaker="banker", start=0, end=29, text=" ".join(["מילה"] * 58)),
            DialogTurn(speaker="customer", start=30, end=40, text=" ".join(["מילה"] * 16)),
        ],
    )
    f = compute_features("T", BANKER, CUSTOMER, dialog, call_duration_sec=120.0)
    # banker: 58 words / (29s / 60) = 120 wpm; customer: 16 / (9.6/60) = 100 wpm.
    assert f.speech_rate_wpm.banker == 120.0
    assert f.speech_rate_wpm.customer == 100.0
    assert f.banker_question_count == 0
    assert f.banker_questions_per_minute == 0.0
    assert f.call_duration_sec == 120.0
