"""One call from its recorded parts.

A transferred or held call is stored by the recorder as several files that
share one call id. They are decoded, put in order and joined into a single
WAV, with a short silence between parts, plus a segment map recording where
each part sits in the call. From then on it is an ordinary call: the
existing pipeline transcribes, attributes speakers and scores it as one.

When every part carries two streams (one per party), the call is written as
stereo - the pipeline's channel path then needs no speaker diarization,
which is both more reliable on 8 kHz audio and far cheaper on a CPU.
"""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from callqa.journey import nmf
from callqa.journey.importers.common import AudioSource, shown_file
from callqa.journey.models import CallAudio

RATE = nmf.SAMPLE_RATE


class SegmentSpan(BaseModel):
    seq: int
    source: str                      # the file, as it may be printed (a digest)
    start_in_call: float
    end_in_call: float
    streams: int
    codec: str
    alignment: str


class SegmentMap(BaseModel):
    call_id: str
    layout: str                      # stereo | mono
    sample_rate: int = RATE
    gap_sec: float
    banker_stream: str = "first"     # which stream went to the left channel
    segments: list[SegmentSpan] = Field(default_factory=list)

    def segment_at(self, t: float) -> int | None:
        for s in self.segments:
            if s.start_in_call - 1e-6 <= t <= s.end_in_call + 1e-6:
                return s.seq
        return None


@dataclass
class Assembled:
    wav_path: Path
    map_path: Path
    segmap: SegmentMap


class AssembleError(RuntimeError):
    pass


def _decode_other(data: bytes, suffix: str) -> list[np.ndarray]:
    """A WAV/MP3 part: its channels as int16 at 8 kHz."""
    if suffix == ".wav":
        import io
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if width != 2:
            raise AssembleError(f"unsupported WAV sample width {width}")
        pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
        if rate != RATE:
            n = int(round(len(pcm) * RATE / rate))
            x = np.linspace(0, len(pcm) - 1, n)
            pcm = np.stack([np.interp(x, np.arange(len(pcm)), pcm[:, c]) for c in range(channels)],
                           axis=1).astype(np.int16)
        return [pcm[:, c].copy() for c in range(min(channels, 2))]
    from callqa.portable import CHILD_FLAGS, find_executable
    ffmpeg = find_executable("ffmpeg")
    if ffmpeg is None:
        raise AssembleError(f"decoding {suffix} needs ffmpeg")
    proc = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", "pipe:0", "-f", "s16le",
                           "-ac", "2", "-ar", str(RATE), "pipe:1"], input=data,
                          capture_output=True, timeout=600, creationflags=CHILD_FLAGS)
    if proc.returncode != 0:
        raise AssembleError(f"ffmpeg could not decode a {suffix} part")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).reshape(-1, 2)
    if np.array_equal(pcm[:, 0], pcm[:, 1]):
        return [pcm[:, 0].copy()]
    return [pcm[:, 0].copy(), pcm[:, 1].copy()]


def decode_part(data: bytes, name: str, codec_overrides: dict[int, str] | None = None
                ) -> tuple[list[np.ndarray], str, str]:
    """(channels, codec, alignment) of one recorded part."""
    suffix = Path(name).suffix.lower()
    if suffix in (".wav", ".mp3"):
        return _decode_other(data, suffix), suffix[1:], "file"
    streams, info = nmf.decode_file(data, codec_overrides)
    if not streams:
        raise AssembleError("the part holds no audio")
    code = info.streams[0].codes.most_common(1) if info.streams else []
    codec = nmf.CODECS.get(code[0][0], ("?",))[0] if code else "?"
    return [s.pcm for s in streams[:2]], codec, streams[0].alignment


def _write(path: Path, channels: list[np.ndarray]) -> None:
    from callqa.portable import make_private_dir
    make_private_dir(path.parent)
    n = max(len(c) for c in channels)
    stacked = np.zeros((n, len(channels)), dtype=np.int16)
    for i, c in enumerate(channels):
        stacked[: len(c), i] = c
    tmp = path.with_suffix(".tmp.wav")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(len(channels))
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(stacked.tobytes())
    from callqa.portable import replace
    replace(tmp, path)


def assemble_call(call: CallAudio, source: AudioSource, out_dir: Path, *, gap_sec: float = 1.0,
                  codec_overrides: dict[int, str] | None = None,
                  banker_stream: str = "first") -> Assembled:
    """Decode a call's parts in order and write <out_dir>/<call_id>.wav plus
    <call_id>.segmap.json. Raises AssembleError for a call that cannot be
    assembled whole (a missing part is never skipped silently)."""
    if not call.segments:
        raise AssembleError("the call has no audio files")
    if call.missing_segments:
        raise AssembleError(f"parts {call.missing_segments} are missing from the audio source")
    parts: list[tuple[list[np.ndarray], str, str]] = []
    for seg in call.segments:
        member = source.find(seg.file_name)
        if member is None:
            raise AssembleError(f"part {seg.seq} is not in the audio source")
        try:
            parts.append(decode_part(source.read(member), seg.file_name, codec_overrides))
        except (nmf.NMFError, AssembleError) as exc:
            raise AssembleError(f"part {seg.seq}: {exc}") from None
    stereo = all(len(p[0]) == 2 for p in parts)
    gap = np.zeros(int(round(gap_sec * RATE)), dtype=np.int16)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    spans: list[SegmentSpan] = []
    cursor = 0
    for seg, (channels, codec, alignment) in zip(call.segments, parts, strict=True):
        if stereo:
            a, b = channels
            if banker_stream == "second":
                a, b = b, a
            n = max(len(a), len(b))
            a = np.pad(a, (0, n - len(a)))
            b = np.pad(b, (0, n - len(b)))
        else:
            if len(channels) == 2:
                n = max(len(channels[0]), len(channels[1]))
                mix = (np.pad(channels[0], (0, n - len(channels[0]))).astype(np.int32)
                       + np.pad(channels[1], (0, n - len(channels[1]))).astype(np.int32)) // 2
                a = mix.astype(np.int16)
            else:
                a = channels[0]
            b = None
            n = len(a)
        if spans:
            left.append(gap)
            if stereo:
                right.append(gap)
            cursor += len(gap)
        spans.append(SegmentSpan(seq=seg.seq, source=shown_file(seg.file_name),
                                 start_in_call=cursor / RATE, end_in_call=(cursor + n) / RATE,
                                 streams=len(channels), codec=codec, alignment=alignment))
        left.append(a)
        if stereo:
            right.append(b)
        cursor += n
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{call.call_id}.wav"
    channels_out = [np.concatenate(left)] + ([np.concatenate(right)] if stereo else [])
    _write(wav_path, channels_out)
    segmap = SegmentMap(call_id=call.call_id, layout="stereo" if stereo else "mono",
                        gap_sec=gap_sec, banker_stream=banker_stream, segments=spans)
    map_path = out_dir / f"{call.call_id}.segmap.json"
    from callqa.state import atomic_write_model
    atomic_write_model(map_path, segmap)
    return Assembled(wav_path=wav_path, map_path=map_path, segmap=segmap)


def load_segmap(path: Path | None) -> SegmentMap | None:
    if path is None or not Path(path).exists():
        return None
    return SegmentMap.model_validate_json(Path(path).read_text(encoding="utf-8"))
