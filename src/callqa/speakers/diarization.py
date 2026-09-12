"""Turning raw diarizer output into speaker-attributed dialog turns.

This is the part of the mono path that decides who said which WORD, and it is
where most of the accuracy of the whole mono pipeline is won or lost. Three
problems have to be solved, and each of them was previously handled by a line
of code that quietly did the wrong thing:

1. **Extra speakers.** A diarizer asked for two speakers can still return
   three or four (hold music, a supervisor, a bad crossfade). Mapping labels
   with `index % 2` merges an unrelated speaker into a role. Instead the two
   labels with the most speech win, and every other segment is folded into the
   neighbouring one, with the amount of reassigned audio reported.

2. **Segment-level attribution.** Whisper segments are not turns: one segment
   routinely straddles a speaker change ("...בסדר גמור" / "תודה, אז מה מספר
   החשבון"). Attributing the whole segment to whoever overlaps it most puts
   the other speaker's words in the wrong mouth, which then corrupts talk
   ratio, monologue length and the interruption counts. Words carry their own
   timestamps, so each word is attributed separately and the segment is split
   at the speaker changes.

3. **One-word islands.** Word-level attribution flickers around turn
   boundaries and on backchannels ("כן", "אוקיי"). A single short word
   surrounded by the other speaker is far more often a boundary artefact than
   a real turn, so those are smoothed away before the split.

The per-word attribution itself is the standard maximum-overlap rule (the same
rule WhisperX uses): a word belongs to the diarization segment it shares the
most time with, and to the nearest segment when it overlaps none.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field

from callqa.models import TranscriptSegment, Word

logger = logging.getLogger(__name__)

# A word shorter than this, alone between two runs of the other speaker, is
# treated as a boundary artefact rather than a turn of its own.
ISLAND_MAX_SEC = 0.6


@dataclass(frozen=True)
class DiarizedSegment:
    """One speech region attributed to one anonymous speaker label."""

    label: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class DiarizationQuality:
    """What had to be cleaned up, so the operator can see it."""

    speakers_found: int = 2
    reassigned_sec: float = 0.0
    words_attributed: int = 0
    words_by_nearest: int = 0      # attributed with no overlap at all
    smoothed_islands: int = 0
    words_dropped: int = 0         # words whose timestamps were unusable

    @property
    def nearest_ratio(self) -> float:
        return self.words_by_nearest / self.words_attributed if self.words_attributed else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "speakers_found": self.speakers_found,
            "reassigned_sec": round(self.reassigned_sec, 2),
            "words_attributed": self.words_attributed,
            "words_by_nearest": self.words_by_nearest,
            "smoothed_islands": self.smoothed_islands,
            "words_dropped": self.words_dropped,
        }


@dataclass
class AttributionResult:
    segments: list[tuple[int, TranscriptSegment]] = field(default_factory=list)
    quality: DiarizationQuality = field(default_factory=DiarizationQuality)


# ---------------------------------------------------------------- speakers

def speech_by_label(segments: list[DiarizedSegment]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for seg in segments:
        totals[seg.label] = totals.get(seg.label, 0.0) + seg.duration
    return totals


def reduce_to_two_speakers(
    segments: list[DiarizedSegment], call_id: str = "-"
) -> tuple[list[DiarizedSegment], DiarizationQuality]:
    """Collapse a diarization down to exactly two labels.

    The two labels with the most speech are kept. Any other segment is given
    the label of the nearest kept segment in time, which is almost always the
    speaker whose turn it interrupted.
    """
    quality = DiarizationQuality()
    if not segments:
        quality.speakers_found = 0
        return [], quality

    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    totals = speech_by_label(ordered)
    quality.speakers_found = len(totals)
    if len(totals) <= 2:
        return ordered, quality

    keep = {label for label, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:2]}
    kept = [s for s in ordered if s.label in keep]

    result: list[DiarizedSegment] = []
    for seg in ordered:
        if seg.label in keep or not kept:
            result.append(seg)
            continue
        # Nearest by the gap between the regions, not between their midpoints:
        # a long turn ending right before this one is adjacent to it however
        # far away its centre happens to be. Ties go to the earlier turn, the
        # one this segment interrupted.
        idx = min(range(len(kept)), key=lambda i: (_gap(kept[i], seg), kept[i].start))
        result.append(DiarizedSegment(kept[idx].label, seg.start, seg.end))
        quality.reassigned_sec += seg.duration

    logger.warning(
        "call_id=%s: diarizer returned %d speakers; kept the two with most speech "
        "and folded %.1fs of the rest into them",
        call_id, len(totals), quality.reassigned_sec,
    )
    return _merge_adjacent(sorted(result, key=lambda s: (s.start, s.end))), quality


def _merge_adjacent(segments: list[DiarizedSegment]) -> list[DiarizedSegment]:
    """Fuse touching or overlapping regions that now carry the same label.

    Folding a third speaker in can leave two same-label regions overlapping,
    which double-counts that speaker's time in the overlap and lets it win
    words that belong to the other party.
    """
    merged: list[DiarizedSegment] = []
    for seg in segments:
        if merged and merged[-1].label == seg.label and seg.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = DiarizedSegment(last.label, last.start, max(last.end, seg.end))
        else:
            merged.append(seg)
    return merged


def _gap(a: DiarizedSegment, b: DiarizedSegment) -> float:
    """Silence between two regions; 0 when they touch or overlap."""
    return max(0.0, a.start - b.end, b.start - a.end)


def index_labels(segments: list[DiarizedSegment]) -> dict[str, int]:
    """Map the (at most two) labels to indices 0 and 1, in order of appearance."""
    mapping: dict[str, int] = {}
    for seg in segments:
        if seg.label not in mapping:
            mapping[seg.label] = len(mapping)
    return mapping


# ---------------------------------------------------------------- attribution

class _SpeakerLookup:
    """Maximum-overlap speaker lookup over time-sorted diarization segments."""

    def __init__(self, segments: list[DiarizedSegment]) -> None:
        self.segments = sorted(segments, key=lambda s: (s.start, s.end))
        self.starts = [s.start for s in self.segments]
        self.ends = [s.end for s in self.segments]
        # The longest segment bounds how far back the scan must look: no
        # earlier-starting segment can reach a query that this one cannot.
        self.max_duration = max((s.duration for s in self.segments), default=0.0)

    def __bool__(self) -> bool:
        return bool(self.segments)

    def label_for(self, start: float, end: float) -> tuple[str | None, bool]:
        """Return (label, used_nearest). used_nearest means there was no overlap."""
        if not self.segments:
            return None, False
        # Candidates start before `end`; scan back far enough to catch any that
        # started earlier and are still running.
        hi = bisect.bisect_right(self.starts, end)
        overlaps: dict[str, float] = {}
        for i in range(hi - 1, -1, -1):
            seg = self.segments[i]
            if seg.start + self.max_duration <= start:
                break          # nothing earlier can reach this query
            if seg.end <= start:
                continue
            overlap = min(seg.end, end) - max(seg.start, start)
            if overlap > 0:
                overlaps[seg.label] = overlaps.get(seg.label, 0.0) + overlap
        if overlaps:
            return max(overlaps.items(), key=lambda kv: kv[1])[0], False

        # Nearest by distance. Scanning only the start-ordered neighbours picks
        # a short segment that merely STARTS nearby over a long one the word
        # actually sits beside.
        mid = (start + end) / 2
        best, best_dist = None, float("inf")
        for seg in self.segments:
            dist = 0.0 if seg.start <= mid <= seg.end else min(
                abs(seg.start - mid), abs(seg.end - mid)
            )
            if dist < best_dist:
                best, best_dist = seg.label, dist
                if dist == 0.0:
                    break
        return best, True


def _smooth_islands(
    labels: list[str | None], words: list[Word], quality: DiarizationQuality
) -> list[str | None]:
    """Flip a single short word wedged between two runs of the other speaker."""
    out = list(labels)
    for i in range(1, len(labels) - 1):
        # Read the ORIGINAL labels, write the copy. Reading back what this
        # loop just wrote lets one flip create the next one's context, so a
        # correctly alternating sequence of short words collapses into a
        # single turn.
        before, here, after = labels[i - 1], labels[i], labels[i + 1]
        if here is None or before is None or before != after or here == before:
            continue
        if (words[i].end - words[i].start) <= ISLAND_MAX_SEC:
            out[i] = before
            quality.smoothed_islands += 1
    return out


def join_words(words: list[Word]) -> str:
    """Rebuild a turn's text from its words.

    Whisper emits word tokens with a leading space; this project's engines
    strip it. Joining on "" is right for the first convention and glues the
    whole turn into one word for the second, which then silently breaks
    redaction of names, every multi-word role marker and every word count.
    So the separator is decided per word, from the token itself.
    """
    parts: list[str] = []
    for word in words:
        token = word.word
        if parts and token[:1] not in (" ", "\t", "\u00a0"):
            parts.append(" ")
        parts.append(token)
    return "".join(parts).strip()


def _segment_from_words(
    template: TranscriptSegment, words: list[Word]
) -> TranscriptSegment:
    return TranscriptSegment(
        start=min(w.start for w in words),
        end=max(w.end for w in words),
        text=join_words(words),
        words=words,
        avg_logprob=template.avg_logprob,
    )


def attribute_segments(
    segments: list[TranscriptSegment],
    diarized: list[DiarizedSegment],
) -> AttributionResult:
    """Attribute transcript segments to speaker indices, splitting at changes.

    Segments that carry word timestamps are attributed word by word and split
    wherever the speaker changes. Segments without word timestamps fall back to
    whole-segment maximum overlap, which is all that is available for them.
    """
    reduced, quality = reduce_to_two_speakers(diarized)
    lookup = _SpeakerLookup(reduced)
    label_index = index_labels(reduced)
    result = AttributionResult(quality=quality)

    for seg in segments:
        if not seg.text.strip():
            continue
        usable = sorted((w for w in seg.words if w.end > w.start),
                        key=lambda w: (w.start, w.end))
        dropped = len(seg.words) - len(usable)
        if dropped:
            quality.words_dropped += dropped
        if not usable:
            label, nearest = lookup.label_for(seg.start, seg.end)
            quality.words_attributed += 1
            quality.words_by_nearest += int(nearest)
            result.segments.append((label_index.get(label or "", 0), seg))
            continue

        labels: list[str | None] = []
        for word in usable:
            label, nearest = lookup.label_for(word.start, word.end)
            labels.append(label)
            quality.words_attributed += 1
            quality.words_by_nearest += int(nearest)
        labels = _smooth_islands(labels, usable, quality)

        run_start = 0
        for i in range(1, len(usable) + 1):
            if i < len(usable) and labels[i] == labels[run_start]:
                continue
            run = usable[run_start:i]
            piece = _segment_from_words(seg, run)
            if piece.text:
                result.segments.append((label_index.get(labels[run_start] or "", 0), piece))
            run_start = i

    result.segments.sort(key=lambda item: (item[1].start, item[1].end))
    return result
