"""Word-level speaker attribution on a single-channel recording.

The mono path carries every metric this system reports about a banker, so the
cases here are the ones that silently corrupt those metrics: a Whisper segment
that straddles a turn boundary, a diarizer that returns a third speaker, and a
backchannel word landing inside the other party's turn.
"""

from __future__ import annotations

from callqa.models import TranscriptSegment, Word
from callqa.speakers.diarization import (
    DiarizedSegment,
    attribute_segments,
    index_labels,
    reduce_to_two_speakers,
)


def w(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end)


# ------------------------------------------------------------- speaker count

def test_two_speakers_pass_through_untouched() -> None:
    segs = [DiarizedSegment("A", 0, 5), DiarizedSegment("B", 5, 9)]
    reduced, quality = reduce_to_two_speakers(segs)
    assert [s.label for s in reduced] == ["A", "B"]
    assert quality.speakers_found == 2
    assert quality.reassigned_sec == 0.0


def test_a_third_speaker_is_folded_into_the_nearest_one() -> None:
    """Asking for two speakers does not guarantee two. The old code mapped
    labels with index % 2, which merged a third speaker into a ROLE."""
    segs = [
        DiarizedSegment("A", 0, 20),
        DiarizedSegment("B", 20, 38),
        DiarizedSegment("C", 38, 40),      # two seconds of somebody else
        DiarizedSegment("A", 40, 50),
    ]
    reduced, quality = reduce_to_two_speakers(segs)
    assert quality.speakers_found == 3
    assert quality.reassigned_sec == 2.0
    assert {s.label for s in reduced} == {"A", "B"}
    # 38-40 sits next to B's turn, so it joins B rather than a random role,
    # and the two touching B regions are fused into one.
    assert [(s.label, s.start, s.end) for s in reduced] == [
        ("A", 0, 20), ("B", 20, 40), ("A", 40, 50),
    ]


def test_folding_never_leaves_two_overlapping_regions_of_one_speaker() -> None:
    """Overlapping same-label regions double-count that speaker's time and let
    it win words that belong to the other party."""
    segs = [
        DiarizedSegment("A", 0, 30),
        DiarizedSegment("B", 30, 60),
        DiarizedSegment("C", 10, 25),          # folds into A, inside A
    ]
    reduced, _ = reduce_to_two_speakers(segs)
    for first, second in zip(reduced, reduced[1:], strict=False):
        assert not (first.label == second.label and second.start < first.end)


def test_labels_are_indexed_in_order_of_appearance() -> None:
    segs = [DiarizedSegment("SPEAKER_07", 0, 2), DiarizedSegment("SPEAKER_02", 2, 4)]
    assert index_labels(segs) == {"SPEAKER_07": 0, "SPEAKER_02": 1}


def test_empty_diarization_is_not_a_crash() -> None:
    reduced, quality = reduce_to_two_speakers([])
    assert reduced == []
    assert quality.speakers_found == 0


# ------------------------------------------------------------- attribution

def test_a_segment_spanning_a_turn_change_is_split() -> None:
    """The case that matters. Whisper emits one segment across the handover;
    attributing all of it to one speaker puts the other party's words in the
    banker's mouth and corrupts talk ratio, monologue and interruptions."""
    diarized = [DiarizedSegment("A", 0, 5), DiarizedSegment("B", 5, 10)]
    segment = TranscriptSegment(
        start=3.0, end=7.0, text="בבקשה תודה רבה",
        words=[w(" בבקשה", 3.0, 4.0), w(" תודה", 5.5, 6.0), w(" רבה", 6.1, 6.6)],
    )
    result = attribute_segments([segment], diarized)
    assert [idx for idx, _ in result.segments] == [0, 1]
    assert [seg.text for _, seg in result.segments] == ["בבקשה", "תודה רבה"]
    assert result.segments[1][1].start == 5.5


def test_a_segment_inside_one_turn_is_not_split() -> None:
    diarized = [DiarizedSegment("A", 0, 30)]
    segment = TranscriptSegment(
        start=1.0, end=3.0, text="שלום וברוך הבא",
        words=[w(" שלום", 1.0, 1.5), w(" וברוך", 1.6, 2.2), w(" הבא", 2.3, 2.9)],
    )
    result = attribute_segments([segment], diarized)
    assert len(result.segments) == 1
    assert result.segments[0][1].text == "שלום וברוך הבא"


def test_a_short_word_island_is_smoothed_away() -> None:
    """A one-word backchannel landing inside the other speaker's turn is a
    boundary artefact far more often than a real turn."""
    diarized = [DiarizedSegment("A", 0, 4), DiarizedSegment("B", 4, 4.3),
                DiarizedSegment("A", 4.3, 10)]
    segment = TranscriptSegment(
        start=3.0, end=6.0, text="אני בודק כן ומוצא",
        words=[w(" אני", 3.0, 3.4), w(" בודק", 3.5, 3.9), w(" כן", 4.05, 4.25),
               w(" ומוצא", 4.5, 5.0)],
    )
    result = attribute_segments([segment], diarized)
    assert len(result.segments) == 1, "the island should not create two turns"
    assert result.quality.smoothed_islands == 1


def test_segments_without_word_timestamps_fall_back_to_whole_segment() -> None:
    diarized = [DiarizedSegment("A", 0, 5), DiarizedSegment("B", 5, 10)]
    segment = TranscriptSegment(start=6.0, end=9.0, text="בלי חותמות זמן", words=[])
    result = attribute_segments([segment], diarized)
    assert len(result.segments) == 1
    assert result.segments[0][0] == 1


def test_a_word_outside_every_diarized_region_takes_the_nearest_speaker() -> None:
    """Diarization misses roughly a third of a second at turn edges; dropping
    those words would lose real speech, so they take the nearest speaker and
    the count is reported."""
    diarized = [DiarizedSegment("A", 0, 2), DiarizedSegment("B", 8, 12)]
    segment = TranscriptSegment(
        start=4.0, end=5.0, text="באמצע",
        words=[w(" באמצע", 4.0, 5.0)],
    )
    result = attribute_segments([segment], diarized)
    assert result.quality.words_by_nearest == 1
    assert result.quality.nearest_ratio == 1.0


def test_a_long_earlier_turn_still_wins_the_overlap() -> None:
    """The lookup scans backwards from the query; a long turn that started well
    before the word must not be missed by that scan."""
    diarized = [DiarizedSegment("A", 0, 60), DiarizedSegment("B", 60, 61)]
    segment = TranscriptSegment(
        start=50.0, end=51.0, text="עדיין הוא",
        words=[w(" עדיין", 50.0, 50.4), w(" הוא", 50.5, 51.0)],
    )
    result = attribute_segments([segment], diarized)
    assert all(idx == 0 for idx, _ in result.segments)
    assert result.quality.words_by_nearest == 0


def test_empty_text_segments_are_dropped() -> None:
    diarized = [DiarizedSegment("A", 0, 5)]
    segments = [TranscriptSegment(start=0, end=1, text="   ", words=[])]
    assert attribute_segments(segments, diarized).segments == []
