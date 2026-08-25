"""Objective conversational features (stage 6), implemented exactly per spec.

Inputs: per-speaker speech segments (stereo: per-channel VAD; mono: diarized
segments) and the merged dialog transcript (for text-based metrics).

Definitions (documented for auditability):
- talk_ratio: banker speech seconds / (banker + customer speech seconds).
- longest_banker_monologue_sec: wall-clock length of the longest run of
  banker segments where, between consecutive banker segments, customer
  speech within the gap totals < 1.0s.
- interruptions_by_banker: banker speech onsets while the customer has
  already been speaking >= 1.0s and the resulting overlap lasts >= 0.5s.
  Mirrored for the customer.
- patience_median_sec: median gap between the end of a customer segment and
  the start of the immediately following banker segment (negative gaps
  excluded; gaps capped at 10s). None when no such pairs exist.
- banker_question_count: banker sentences ending with '?' OR starting with a
  Hebrew interrogative (each sentence counted once). Also reported per minute.
- speech_rate_wpm: words / speech minutes, per speaker.
- dead_air_total_sec: total duration of gaps > 3s between speech intervals
  (union of both speakers); leading/trailing silence is not counted.
"""

from __future__ import annotations

import re
import statistics

from callqa.models import DialogTranscript, Features, SpeechRateWPM, VADSegment

HEBREW_INTERROGATIVES = ("האם", "מה", "מתי", "איך", "כמה", "למה", "איפה", "מי")

INTERRUPTION_MIN_SPEAKING_SEC = 1.0
INTERRUPTION_MIN_OVERLAP_SEC = 0.5
MONOLOGUE_BREAK_SEC = 1.0
PATIENCE_CAP_SEC = 10.0
DEAD_AIR_MIN_GAP_SEC = 3.0


def _total(segments: list[VADSegment]) -> float:
    return sum(s.end - s.start for s in segments)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _speech_in_window(segments: list[VADSegment], start: float, end: float) -> float:
    if end <= start:
        return 0.0
    return sum(_overlap(s.start, s.end, start, end) for s in segments)


def longest_monologue(
    speaker_segments: list[VADSegment], other_segments: list[VADSegment]
) -> float:
    """Longest wall-clock run of speaker segments not broken by >= 1.0s of the
    other party's speech between consecutive segments."""
    segs = sorted(speaker_segments, key=lambda s: s.start)
    if not segs:
        return 0.0
    longest = 0.0
    run_start = segs[0].start
    run_end = segs[0].end
    for prev, nxt in zip(segs, segs[1:], strict=False):
        interruption = _speech_in_window(other_segments, prev.end, nxt.start)
        if interruption >= MONOLOGUE_BREAK_SEC:
            longest = max(longest, run_end - run_start)
            run_start = nxt.start
        run_end = nxt.end
    return round(max(longest, run_end - run_start), 3)


def count_interruptions(
    interrupter: list[VADSegment], interrupted: list[VADSegment]
) -> int:
    """Onsets of `interrupter` while `interrupted` has been speaking >= 1.0s,
    with overlap >= 0.5s."""
    count = 0
    for seg in sorted(interrupter, key=lambda s: s.start):
        for other in interrupted:
            if other.start <= seg.start < other.end:
                speaking_for = seg.start - other.start
                overlap = _overlap(seg.start, seg.end, other.start, other.end)
                if (
                    speaking_for >= INTERRUPTION_MIN_SPEAKING_SEC
                    and overlap >= INTERRUPTION_MIN_OVERLAP_SEC
                ):
                    count += 1
                break
    return count


def patience_median(
    banker_segments: list[VADSegment], customer_segments: list[VADSegment]
) -> float | None:
    """Median gap between a customer segment end and the next banker onset."""
    events: list[tuple[float, float, str]] = [
        (s.start, s.end, "banker") for s in banker_segments
    ] + [(s.start, s.end, "customer") for s in customer_segments]
    events.sort(key=lambda e: (e[0], e[1]))
    gaps: list[float] = []
    for (_, prev_end, prev_role), (next_start, _, next_role) in zip(
        events, events[1:], strict=False
    ):
        if prev_role == "customer" and next_role == "banker":
            gap = next_start - prev_end
            if gap >= 0:
                gaps.append(min(gap, PATIENCE_CAP_SEC))
    if not gaps:
        return None
    return round(statistics.median(gaps), 3)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


def count_banker_questions(dialog: DialogTranscript) -> int:
    count = 0
    for turn in dialog.turns:
        if turn.speaker != "banker":
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(turn.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            first_word = re.sub(r"^[^\w֐-׿]+", "", sentence).split(" ")[0]
            if sentence.endswith("?") or first_word in HEBREW_INTERROGATIVES:
                count += 1
    return count


def dead_air_total(
    banker_segments: list[VADSegment], customer_segments: list[VADSegment]
) -> float:
    """Sum of >3s gaps in the union of both speakers' speech intervals."""
    intervals = sorted(
        [(s.start, s.end) for s in banker_segments + customer_segments], key=lambda i: i[0]
    )
    if not intervals:
        return 0.0
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    total = 0.0
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:], strict=False):
        gap = next_start - prev_end
        if gap > DEAD_AIR_MIN_GAP_SEC:
            total += gap
    return round(total, 3)


def _word_count(dialog: DialogTranscript, speaker: str) -> int:
    return sum(len(t.text.split()) for t in dialog.turns if t.speaker == speaker)


def compute_features(
    call_id: str,
    banker_segments: list[VADSegment],
    customer_segments: list[VADSegment],
    dialog: DialogTranscript,
    call_duration_sec: float,
) -> Features:
    banker_speech = _total(banker_segments)
    customer_speech = _total(customer_segments)
    denom = banker_speech + customer_speech
    talk_ratio = round(banker_speech / denom, 4) if denom > 0 else 0.0

    question_count = count_banker_questions(dialog)
    duration_min = call_duration_sec / 60.0 if call_duration_sec > 0 else 0.0

    banker_minutes = banker_speech / 60.0
    customer_minutes = customer_speech / 60.0
    wpm = SpeechRateWPM(
        banker=round(_word_count(dialog, "banker") / banker_minutes, 1) if banker_minutes > 0 else 0.0,
        customer=round(_word_count(dialog, "customer") / customer_minutes, 1)
        if customer_minutes > 0
        else 0.0,
    )

    return Features(
        call_id=call_id,
        talk_ratio=talk_ratio,
        longest_banker_monologue_sec=longest_monologue(banker_segments, customer_segments),
        interruptions_by_banker=count_interruptions(banker_segments, customer_segments),
        interruptions_by_customer=count_interruptions(customer_segments, banker_segments),
        patience_median_sec=patience_median(banker_segments, customer_segments),
        banker_question_count=question_count,
        banker_questions_per_minute=round(question_count / duration_min, 3)
        if duration_min > 0
        else 0.0,
        speech_rate_wpm=wpm,
        dead_air_total_sec=dead_air_total(banker_segments, customer_segments),
        call_duration_sec=round(call_duration_sec, 3),
        banker_speech_sec=round(banker_speech, 3),
        customer_speech_sec=round(customer_speech, 3),
    )
