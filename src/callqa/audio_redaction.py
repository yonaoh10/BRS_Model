"""Redacted audio, produced alongside the redacted transcript.

The project's hardest rule is that raw customer identifiers must not survive
the redaction stage. A play button in the dashboard would defeat that on its
own: the recording has the customer reading their national ID aloud. So the
only audio the browser can reach is a copy that is SILENCED wherever the
transcript was masked. The raw WAVs under ``audio/wav/`` have no route to the
browser; only the artifact this module writes does.

The map from a masked character span to a stretch of audio runs through the
per-word timestamps the ASR already produced::

    masked characters -> the words they overlap -> (min start, max end) seconds

Detection is re-run here over a document rebuilt from the WORDS (the same
inter-word spacing :func:`join_words` uses), not over the redacted turn text,
so every character maps to a known word with a known time - even on the stereo
path where ``turn.text`` and ``join_words(turn.words)`` differ. Each range is
padded at both ends and overlapping ranges are merged, biased toward silencing
too much: a fraction of a spoken digit is still a leak.

The silence is exactly the text redaction's silence: this artifact is only as
trustworthy as the mask it mirrors, no more. Where the text redactor misses
something, the audio misses it too.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from callqa.audio import _read_wav, _write_wav
from callqa.models import AudioArtifact, DialogTranscript
from callqa.redaction import TURN_SEPARATOR, TurnSpan, _name_pattern, find_pii

logger = logging.getLogger(__name__)

# Pad every silenced region at both ends. Word timestamps are approximate and a
# fraction of a spoken identifier is still a leak, so err toward silencing more.
AUDIO_PAD_SEC = 0.25

_LEADING_SPACE = (" ", "\t", "\u00a0")


def _spoken_document(
    dialog: DialogTranscript,
) -> tuple[str, list[tuple[int, int, float, float]], list[TurnSpan]]:
    """Rebuild the spoken text from words, tracking each word's char span + time.

    Returns ``(document, word_spans, turn_spans)`` where each word entry is
    ``(char_start, char_end, t_start, t_end)``. Turns are joined with
    ``TURN_SEPARATOR`` so a number the speaker paused in the middle of - two
    turns once split - is still one detectable digit run.

    ``turn_spans`` carries the same boundaries in THIS document's coordinates,
    which differ from the turn-text document's: detection needs them to tell an
    identity question from the answer somebody else gave it.
    """
    pieces: list[str] = []
    word_spans: list[tuple[int, int, float, float]] = []
    turn_spans: list[TurnSpan] = []
    cursor = 0
    for t_index, turn in enumerate(dialog.turns):
        if t_index:
            pieces.append(TURN_SEPARATOR)
            cursor += len(TURN_SEPARATOR)
        turn_start = cursor
        started = False
        for word in turn.words:
            token = word.word
            if started and token[:1] not in _LEADING_SPACE:
                pieces.append(" ")
                cursor += 1
            started = True
            start = cursor
            pieces.append(token)
            cursor += len(token)
            word_spans.append((start, cursor, word.start, word.end))
        turn_spans.append(TurnSpan(turn_start, cursor, turn.speaker))
    return "".join(pieces), word_spans, turn_spans


def _text_document(dialog: DialogTranscript) -> list[tuple[int, int, object]]:
    """Join turn TEXT exactly as RegexRedactor._redact does, tracking each turn's
    char span. This is the ground truth of what the text redactor masks, so
    detecting over it lets the audio mirror the text mask precisely - including
    a number split across a turn boundary, which per-turn detection would miss.
    """
    spans: list[tuple[int, int, object]] = []
    cursor = 0
    for i, turn in enumerate(dialog.turns):
        if i:
            cursor += len(TURN_SEPARATOR)
        start = cursor
        cursor += len(turn.text)
        spans.append((start, cursor, turn))
    return spans


def _all_pii_spans(
    document: str, names: list[str], turns: list[TurnSpan] | None = None
) -> list[tuple[int, int]]:
    return [(m.start, m.end) for m in find_pii(document, turns)] + _name_spans(document, names)


def _name_spans(document: str, extra_names: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for name in extra_names:
        cleaned = " ".join((name or "").split())
        if len(cleaned) < 2:
            continue
        pattern = _name_pattern(cleaned)
        if pattern is None:
            continue
        spans.extend((m.start(), m.end()) for m in pattern.finditer(document))
    return spans


def _merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = sorted((max(0.0, a), b) for a, b in ranges if b > a)
    merged: list[tuple[float, float]] = []
    for a, b in clean:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def masked_time_ranges(
    dialog: DialogTranscript, extra_names: list[str] | None = None
) -> list[tuple[float, float]]:
    """Second-ranges of the audio to silence: every masked identifier + name.

    Two passes, so the audio mirrors the text mask exactly:

    1. **Word pass** - detect over a document rebuilt from ``turn.words`` and
       map each masked span to its words' precise (start, end). This gives tight
       silence where the ASR gave us word timings, which is the common case.
    2. **Text safety net** - detect over the SAME joined-text document the
       redactor masks (cross-turn), and for every masked span silence the WHOLE
       of each overlapping turn that the word pass did not already cover. This
       closes the gap when a turn's ``words`` do not faithfully reproduce its
       ``text`` (empty or partial word timings, or a number split across a
       wordless turn boundary): without it, such an identifier would play in the
       clear even though the transcript masks it.
    """
    names = extra_names or []

    # Pass 1: precise word-level ranges. word_hits keeps the UNPADDED spans for
    # the coverage test below, so a neighbouring turn's padding cannot make a
    # turn look covered when it is not.
    word_doc, word_spans, word_turns = _spoken_document(dialog)
    ranges: list[tuple[float, float]] = []
    word_hits: list[tuple[float, float]] = []
    for s0, s1 in _all_pii_spans(word_doc, names, word_turns):
        hits = [(ts, te) for (c0, c1, ts, te) in word_spans if c0 < s1 and s0 < c1]
        if hits:
            lo, hi = min(t for t, _ in hits), max(t for _, t in hits)
            word_hits.append((lo, hi))
            ranges.append((lo - AUDIO_PAD_SEC, hi + AUDIO_PAD_SEC))

    # Pass 2: the cross-turn safety net over the redactor's own document.
    turn_spans = _text_document(dialog)
    text_turns = [TurnSpan(c0, c1, turn.speaker) for c0, c1, turn in turn_spans]
    for s0, s1 in _all_pii_spans(_joined_text(dialog), names, text_turns):
        for c0, c1, turn in turn_spans:
            if c0 < s1 and s0 < c1 and not _covered(turn, word_hits):
                ranges.append((turn.start - AUDIO_PAD_SEC, turn.end + AUDIO_PAD_SEC))

    return _merge(ranges)


def _joined_text(dialog: DialogTranscript) -> str:
    return TURN_SEPARATOR.join(turn.text for turn in dialog.turns)


def _covered(turn: object, word_hits: list[tuple[float, float]]) -> bool:
    """True if a precise word-range already lands inside this turn's time span."""
    start, end = turn.start, turn.end  # type: ignore[attr-defined]
    return any(a < end and start < b for a, b in word_hits)


def _load_source(audio_art: AudioArtifact) -> tuple[np.ndarray, int]:
    """Load the source recording as a mono float32 signal, plus its rate.

    Stereo is downmixed to the same single timeline the transcript timestamps
    live on, so a silence range in seconds lands on the same samples in both
    layouts.
    """
    if audio_art.is_stereo and audio_art.banker_wav and audio_art.customer_wav:
        banker, rate = _read_wav(Path(audio_art.banker_wav))
        customer, _ = _read_wav(Path(audio_art.customer_wav))
        n = min(len(banker), len(customer))
        mono = (banker[:n, 0] + customer[:n, 0]) / 2.0
        return mono.astype(np.float32), rate
    if audio_art.mono_wav:
        data, rate = _read_wav(Path(audio_art.mono_wav))
        return data[:, 0].astype(np.float32), rate
    raise FileNotFoundError("audio artifact has no source WAV to redact")


def _tmp_path(path: Path) -> Path:
    # Per-process suffix so two workers on one call_id never share a tmp file
    # and race a corrupt partial into place via os.replace.
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def _atomic_write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:  # pragma: no cover - unusual filesystems
        pass
    tmp = _tmp_path(path)
    try:
        _write_wav(tmp, samples, rate)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)  # never orphan a partial tmp on failure
        raise
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover
        pass


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover
        pass


def write_redacted_wav(
    audio_art: AudioArtifact, ranges: list[tuple[float, float]], out_path: Path
) -> tuple[float, int]:
    """Write a copy of the recording with ``ranges`` zeroed. Returns (dur, rate)."""
    samples, rate = _load_source(audio_art)
    for a, b in ranges:
        i0 = max(0, int(a * rate))
        i1 = min(len(samples), int(b * rate))
        if i1 > i0:
            samples[i0:i1] = 0.0
    _atomic_write_wav(out_path, samples, rate)
    duration = len(samples) / rate if rate else 0.0
    return duration, rate


def produce_redacted_audio(
    audio_art: AudioArtifact,
    dialog: DialogTranscript,
    extra_names: list[str] | None,
    out_wav: Path,
) -> list[tuple[float, float]]:
    """Write the redacted WAV and its sidecar metadata. Returns the silences.

    The sidecar ``<id>.json`` next to the WAV carries the duration, sample rate
    and silence ranges the dashboard draws on the timeline.
    """
    ranges = masked_time_ranges(dialog, extra_names)
    duration, rate = write_redacted_wav(audio_art, ranges, out_wav)
    meta = {
        "duration": round(duration, 3),
        "sample_rate": rate,
        "silences": [[round(a, 3), round(b, 3)] for a, b in ranges],
    }
    _atomic_write_json(out_wav.with_suffix(".json"), meta)
    logger.info("redacted audio: call_id=%s silences=%d duration=%.1fs",
                dialog.call_id, len(ranges), duration)
    return ranges
