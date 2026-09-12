"""Deciding which anonymous speaker is the banker.

A swapped decision does not look like a failure. It produces a complete,
internally consistent report about the wrong person, so these tests check both
that the right speaker wins and that an undecidable call says so instead of
guessing confidently.
"""

from __future__ import annotations

from callqa.models import TranscriptSegment
from callqa.speakers.roles import infer_roles


def dialog(*turns: tuple[int, float, float, str]) -> list[tuple[int, TranscriptSegment]]:
    return [(idx, TranscriptSegment(start=start, end=end, text=text))
            for idx, start, end, text in turns]


A_REAL_CALL = (
    (0, 0.0, 5.0, "שלום, הגעת למוקד הבנק, מדבר דני. השיחה מוקלטת. במה אפשר לעזור?"),
    (1, 5.5, 11.0, "שלום, ירדה לי עמלה שאני לא מזהה."),
    (0, 11.5, 16.0, "בשמחה אבדוק. מה מספר תעודת הזהות שלך?"),
    (1, 16.5, 21.0, "המספר הוא 314159260."),
    (0, 21.5, 27.0, "תודה, הזיהוי הושלם. אני בודק את החיוב."),
    (1, 27.5, 31.0, "אוקיי, אני מחכה."),
    (0, 31.5, 38.0, "לסיכום, העברתי אותך למסלול החודשי. תודה שפנית אלינו, יום נעים."),
)


def test_the_banker_is_identified_in_a_normal_call() -> None:
    decision = infer_roles(dialog(*A_REAL_CALL))
    assert decision.banker_index == 0
    assert decision.confidence > 0.5
    assert {s.name for s in decision.signals} >= {
        "opening", "identity_request", "identity_supply", "question_rate",
    }


def test_the_decision_follows_the_speaker_not_the_index() -> None:
    """The same call with the two speakers swapped must invert the answer.
    If it does not, something is keying off the index instead of the words."""
    swapped = dialog(*[(1 - idx, s, e, t) for idx, s, e, t in A_REAL_CALL])
    decision = infer_roles(swapped)
    assert decision.banker_index == 1
    assert decision.confidence > 0.5


def test_the_speaker_reading_out_an_id_is_the_customer() -> None:
    """A long run of digits is identification being supplied, not requested."""
    decision = infer_roles(dialog(
        (0, 0.0, 4.0, "רגע אחד."),
        (1, 4.0, 9.0, "המספר שלי הוא 314159260 והנייד 0548765432."),
    ))
    supply = [s for s in decision.signals if s.name == "identity_supply"]
    assert supply and supply[0].votes_for == 0, "the digits point AWAY from their speaker"


def test_an_evenly_split_call_reports_no_confidence() -> None:
    """Two speakers, nothing to tell them apart: the honest answer is that the
    evidence was even, not a 50% confident guess."""
    decision = infer_roles(dialog(
        (0, 0.0, 3.0, "אהה"),
        (1, 3.0, 6.0, "אהה"),
    ))
    assert decision.confidence < 0.34
    assert not decision.decided or decision.confidence < 0.34


def test_a_single_speaker_call_is_not_decided() -> None:
    decision = infer_roles(dialog((0, 0.0, 5.0, "שלום, מדבר דני מהמוקד.")))
    assert decision.confidence == 0.0
    assert decision.signals == []


def test_no_transcript_is_not_a_crash() -> None:
    decision = infer_roles([])
    assert decision.confidence == 0.0


def test_every_signal_is_recorded_for_the_audit_trail() -> None:
    decision = infer_roles(dialog(*A_REAL_CALL))
    for signal in decision.signals:
        assert signal.name and signal.weight > 0
        assert signal.votes_for in (0, 1)
        assert signal.detail


def test_an_outbound_call_still_resolves_against_the_first_speaker_prior() -> None:
    """The customer answers an outbound call, so 'who spoke first' points the
    wrong way. The scripted content must outweigh it."""
    decision = infer_roles(dialog(
        (1, 0.0, 2.0, "הלו?"),
        (0, 2.0, 9.0, "שלום, מדבר דני מהבנק, השיחה מוקלטת. אפשר לאמת את תעודת הזהות?"),
        (1, 9.0, 13.0, "כן, 314159260."),
        (0, 13.0, 18.0, "תודה. אשמח לעדכן אותך, לסיכום נשלח לך הודעה. יום נעים."),
    ))
    assert decision.banker_index == 0
    first = [s for s in decision.signals if s.name == "first_speaker"][0]
    assert first.votes_for == 1, "the prior did point the wrong way"
    assert decision.confidence > 0.5
