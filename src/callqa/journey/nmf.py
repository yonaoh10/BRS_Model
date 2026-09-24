"""NICE recorder files (.nmf): structure, streams, and decoded audio.

NICE does not publish the format. Two public projects read it (by
decompiling NICE's player); this is an independent implementation of the
same structure, written for this pipeline and checked on real files with
`callqa nmf-info`, which prints structure only - never audio or content.

A file is a sequence of packets, each a 28-byte header:

    int8    packet type          4/5 = audio, 7 = end of file
    int16   packet subtype       (4,0) (4,3) (5,300) carry audio
    int8    stream id            one per recorded party or channel
    float64 start time           units vary between versions (checked, not assumed)
    float64 end time
    uint32  packet size          bytes after the header, parameters included
    uint32  parameters size

then the parameters - 22-byte records {int16 type, int32 size, 16 bytes of
data}; type 10 holds the compression code - then the audio payload.

Byte order is little-endian in every known file; a file that reads as
nonsense that way is tried big-endian before it is declared unreadable.
"""

from __future__ import annotations

import math
import struct
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from callqa.journey import g711

HEADER = 28
PARAM = 22
AUDIO_TYPES = {(4, 0), (4, 3), (5, 300)}
END_TYPE = 7
COMPRESSION_PARAM = 10
SAMPLE_RATE = 8000

# compression code -> (ffmpeg demuxer, output rate, samples per payload byte)
CODECS: dict[int, tuple[str, int, float]] = {
    0: ("g729", 8000, 8.0),        # 10 bytes = 10 ms = 80 samples
    1: ("g726", 8000, 2.0),        # 32 kbit/s: 4 bits per sample (bit rate decided at decode)
    2: ("g726", 8000, 2.0),
    3: ("alaw", 8000, 1.0),
    7: ("mulaw", 8000, 1.0),
    8: ("g729", 8000, 8.0),
    9: ("g723_1", 8000, 10.0),     # 24 bytes = 30 ms = 240 samples (6.3 kbit/s)
    10: ("g723_1", 8000, 10.0),
    19: ("g722", 16000, 2.0),      # 1 byte = 2 samples at 16 kHz
}


class NMFError(ValueError):
    pass


@dataclass
class Packet:
    ptype: int
    subtype: int
    stream: int
    start: float
    end: float
    size: int
    params_size: int
    offset: int                     # where the header starts
    compression: int | None = None

    @property
    def payload_offset(self) -> int:
        return self.offset + HEADER + self.params_size

    @property
    def payload_len(self) -> int:
        return max(0, self.size - self.params_size)

    @property
    def is_audio(self) -> bool:
        return (self.ptype, self.subtype) in AUDIO_TYPES


@dataclass
class ParsedNMF:
    packets: list[Packet]
    byte_order: str
    truncated: bool = False
    trailing_bytes: int = 0


@dataclass
class StreamInfo:
    stream: int
    packets: int = 0
    payload_bytes: int = 0
    codes: Counter = field(default_factory=Counter)
    first_start: float | None = None
    last_end: float | None = None


@dataclass
class NMFInfo:
    """Structure of one file - nothing in it is audio or content."""

    size: int
    byte_order: str
    packets: int
    by_type: dict[str, int]
    streams: list[StreamInfo]
    param_types: dict[int, int]
    truncated: bool
    trailing_bytes: int
    time_unit: str
    duration_sec: float | None
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"size {self.size:,} bytes; byte order {self.byte_order}; "
               f"{self.packets:,} packets"
               + ("; TRUNCATED at the end" if self.truncated else "")]
        out.append("packet types: " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_type.items())))
        out.append("parameter types: " + ", ".join(f"{k}={v}" for k, v in sorted(self.param_types.items())))
        for s in self.streams:
            codes = ", ".join(f"{c}({CODECS.get(c, ('?',))[0]})x{n}" for c, n in s.codes.items())
            span = ""
            if s.first_start is not None and s.last_end is not None:
                span = f"; time {s.first_start:.6g} .. {s.last_end:.6g}"
            out.append(f"stream {s.stream}: {s.packets:,} audio packets, "
                       f"{s.payload_bytes:,} bytes; compression {codes or 'none'}{span}")
        out.append(f"time unit: {self.time_unit}"
                   + (f"; about {self.duration_sec:.1f} s of audio" if self.duration_sec else ""))
        out.extend(f"note: {n}" for n in self.notes)
        return out


def _parse(data: bytes, order: str) -> ParsedNMF:
    fmt = order + "bhbddII"
    packets: list[Packet] = []
    pos = 0
    truncated = False
    while pos + HEADER <= len(data):
        ptype, sub, stream, start, end, size, psize = struct.unpack_from(fmt, data, pos)
        if size > len(data) - pos - HEADER or psize > size:
            truncated = True
            break
        pkt = Packet(ptype, sub, stream, start, end, size, psize, pos)
        if psize >= PARAM:
            base = pos + HEADER
            for off in range(0, psize - PARAM + 1, PARAM):
                ptype_id, _dsize = struct.unpack_from(order + "hi", data, base + off)
                if ptype_id == COMPRESSION_PARAM:
                    pkt.compression = struct.unpack_from("b", data, base + off + 6)[0]
        packets.append(pkt)
        pos += HEADER + size
        if ptype == END_TYPE:
            break
    return ParsedNMF(packets=packets, byte_order="little" if order == "<" else "big",
                     truncated=truncated, trailing_bytes=max(0, len(data) - pos))


def _plausible(parsed: ParsedNMF) -> bool:
    if not parsed.packets:
        return False
    audio = sum(p.is_audio for p in parsed.packets)
    finite = all(math.isfinite(p.start) and math.isfinite(p.end) for p in parsed.packets)
    return audio > 0 and finite and audio >= 0.3 * len(parsed.packets)


def parse(data: bytes) -> ParsedNMF:
    little = _parse(data, "<")
    if _plausible(little):
        return little
    big = _parse(data, ">")
    if _plausible(big):
        return big
    raise NMFError(f"not a readable NMF file ({len(little.packets)} packets read, "
                   f"{sum(p.is_audio for p in little.packets)} audio)")


def _time_unit(packets: list[Packet], samples_per_byte: float, rate: int) -> tuple[str, float]:
    """Seconds per unit of the header times, judged from the audio itself: a
    packet of N payload bytes holds N * samples_per_byte / rate seconds."""
    ratios = []
    for p in packets:
        span = p.end - p.start
        if p.payload_len and span > 0:
            ratios.append((p.payload_len * samples_per_byte / rate) / span)
    if len(ratios) < 3:
        return "unknown", 0.0
    r = float(np.median(ratios))
    for name, scale in (("seconds", 1.0), ("milliseconds", 1e-3), ("days", 86400.0),
                        ("100-nanosecond ticks", 1e-7), ("microseconds", 1e-6)):
        if abs(r / scale - 1.0) < 0.15:
            return name, scale
    return f"unrecognised (x{r:.3g})", 0.0


def inspect_bytes(data: bytes) -> NMFInfo:
    parsed = parse(data)
    by_type = Counter(f"{p.ptype}/{p.subtype}" for p in parsed.packets)
    streams: dict[int, StreamInfo] = {}
    params: Counter = Counter()
    for p in parsed.packets:
        if p.params_size >= PARAM:
            params[COMPRESSION_PARAM if p.compression is not None else -1] += 1
        if not p.is_audio:
            continue
        s = streams.setdefault(p.stream, StreamInfo(stream=p.stream))
        s.packets += 1
        s.payload_bytes += p.payload_len
        if p.compression is not None:
            s.codes[p.compression] += 1
        s.first_start = p.start if s.first_start is None else min(s.first_start, p.start)
        s.last_end = p.end if s.last_end is None else max(s.last_end, p.end)
    notes: list[str] = []
    audio = [p for p in parsed.packets if p.is_audio]
    code = Counter(p.compression for p in audio if p.compression is not None).most_common(1)
    unit, scale = "unknown", 0.0
    duration = None
    if code:
        c = code[0][0]
        if c in CODECS:
            _fmt, rate, spb = CODECS[c]
            unit, scale = _time_unit(audio, spb, rate)
            per_stream = [s.payload_bytes * spb / rate for s in streams.values()]
            duration = max(per_stream) if per_stream else None
        else:
            notes.append(f"compression code {c} is not known; set journey.nmf.codec_overrides")
    else:
        notes.append("no compression parameter found; G.729 would be assumed")
    if len(streams) == 2:
        notes.append("two streams: likely one per party (stereo)")
    if parsed.trailing_bytes:
        notes.append(f"{parsed.trailing_bytes} bytes after the last packet")
    sample = b"".join(data[p.payload_offset:p.payload_offset + p.payload_len] for p in audio[:50])
    if sample and _entropy(sample) > 7.95:
        notes.append("payload looks random (encrypted?); if decoding fails, export WAV from NICE")
    return NMFInfo(size=len(data), byte_order=parsed.byte_order, packets=len(parsed.packets),
                   by_type=dict(by_type), streams=sorted(streams.values(), key=lambda s: s.stream),
                   param_types=dict(params), truncated=parsed.truncated,
                   trailing_bytes=parsed.trailing_bytes, time_unit=unit, duration_sec=duration,
                   notes=notes)


def _entropy(data: bytes) -> float:
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(float)
    p = counts[counts > 0] / len(data)
    return float(-(p * np.log2(p)).sum())


def inspect(path: str | Path) -> NMFInfo:
    return inspect_bytes(Path(path).read_bytes())


# ------------------------------------------------------------------ decoding


@dataclass
class Stream:
    stream: int
    compression: int
    chunks: list[tuple[float, float, bytes]]   # (start, end, payload)


def extract_streams(data: bytes, codec_overrides: dict[int, str] | None = None) -> list[Stream]:
    parsed = parse(data)
    by_stream: dict[int, list[Packet]] = defaultdict(list)
    for p in parsed.packets:
        if p.is_audio and p.payload_len:
            by_stream[p.stream].append(p)
    out = []
    for sid, packets in sorted(by_stream.items()):
        codes = Counter(p.compression for p in packets if p.compression is not None)
        code = codes.most_common(1)[0][0] if codes else 0
        out.append(Stream(stream=sid, compression=code, chunks=[
            (p.start, p.end, data[p.payload_offset:p.payload_offset + p.payload_len])
            for p in packets]))
    return out


def _decode_external(raw: bytes, demuxer: str, rate: int, extra: list[str] | None = None
                     ) -> np.ndarray:
    """Raw codec bytes -> int16 PCM at SAMPLE_RATE through ffmpeg (the one the
    pipeline already finds, including tools/ffmpeg)."""
    from callqa.portable import CHILD_FLAGS, find_executable

    ffmpeg = find_executable("ffmpeg")
    if ffmpeg is None:
        raise NMFError(f"decoding {demuxer} needs ffmpeg (tools/ffmpeg or CALLQA_FFMPEG_DIR)")
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-f", demuxer, *(extra or []),
           "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=600,
                          creationflags=CHILD_FLAGS)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["?"]
        raise NMFError(f"ffmpeg could not decode {demuxer}: {msg[0][:200]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).copy()


def _speech_share(pcm: np.ndarray) -> float:
    """Share of 20 ms frames above a quiet floor, penalising clipping - a
    cheap test of whether a decode produced speech or noise."""
    if len(pcm) < 1600:
        return 0.0
    frames = pcm[: len(pcm) // 160 * 160].reshape(-1, 160).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    clipped = float((np.abs(pcm) >= 32000).mean())
    active = float(((rms > 300) & (rms < 12000)).mean())
    return active - 5 * clipped


def decode_payload(raw: bytes, compression: int, codec_overrides: dict[int, str] | None = None
                   ) -> np.ndarray:
    """One stream's concatenated payload -> int16 PCM at 8 kHz."""
    name = (codec_overrides or {}).get(compression)
    demuxer = name or CODECS.get(compression, ("g729",))[0]
    if demuxer == "alaw":
        return g711.decode_alaw(raw)
    if demuxer == "mulaw":
        return g711.decode_ulaw(raw)
    if demuxer in ("g726", "g726le"):
        # bit order and rate vary by recorder; keep the decode that sounds like speech
        best, best_score = None, -1e9
        for mux in ("g726le", "g726"):
            for bits in (4, 2, 3, 5):
                try:
                    probe = _decode_external(raw[:40_000], mux, SAMPLE_RATE,
                                             ["-code_size", str(bits)])
                except NMFError:
                    continue
                score = _speech_share(probe)
                if score > best_score:
                    best, best_score = (mux, bits), score
        if best is None:
            raise NMFError("no G.726 variant decoded")
        return _decode_external(raw, best[0], SAMPLE_RATE, ["-code_size", str(best[1])])
    return _decode_external(raw, demuxer, SAMPLE_RATE)


def _samples_per_byte(compression: int, overrides: dict[int, str] | None) -> float | None:
    name = (overrides or {}).get(compression)
    if name in ("alaw", "mulaw"):
        return 1.0
    spec = CODECS.get(compression)
    if spec is None:
        return None
    # output is resampled to 8 kHz, so G.722's 2 samples per byte at 16 kHz are 1 at 8 kHz
    return spec[2] * SAMPLE_RATE / spec[1]


@dataclass
class DecodedStream:
    stream: int
    pcm: np.ndarray
    alignment: str                  # timed | sequential
    start: float | None             # header time of the first chunk, in seconds when known


def decode_stream(stream: Stream, codec_overrides: dict[int, str] | None = None,
                  time_scale: float = 0.0, gap_min_sec: float = 0.1) -> DecodedStream:
    """Decode a stream. When the header times are understood (time_scale > 0)
    and the decoded length matches the payload arithmetic, chunks are placed at
    their recorded times, so silences the recorder skipped come back and two
    streams stay in step; otherwise the audio is simply concatenated."""
    raw = b"".join(c[2] for c in stream.chunks)
    pcm = decode_payload(raw, stream.compression, codec_overrides)
    spb = _samples_per_byte(stream.compression, codec_overrides)
    first = stream.chunks[0][0] * time_scale if time_scale and stream.chunks else None
    if not time_scale or spb is None:
        return DecodedStream(stream.stream, pcm, "sequential", first)
    expected = [int(round(len(c[2]) * spb)) for c in stream.chunks]
    if abs(sum(expected) - len(pcm)) > max(0.01 * len(pcm), 160):
        return DecodedStream(stream.stream, pcm, "sequential", first)
    pieces: list[np.ndarray] = []
    cursor = 0
    t0 = stream.chunks[0][0]
    placed = 0
    for (start, _end, _payload), n in zip(stream.chunks, expected, strict=True):
        at = int(round((start - t0) * time_scale * SAMPLE_RATE))
        gap = at - placed
        if gap > gap_min_sec * SAMPLE_RATE:
            pieces.append(np.zeros(gap, dtype=np.int16))
            placed += gap
        chunk = pcm[cursor:cursor + n]
        cursor += n
        pieces.append(chunk)
        placed += len(chunk)
    return DecodedStream(stream.stream, np.concatenate(pieces) if pieces else pcm, "timed", first)


def decode_file(data: bytes, codec_overrides: dict[int, str] | None = None
                ) -> tuple[list[DecodedStream], NMFInfo]:
    info = inspect_bytes(data)
    scale = {"seconds": 1.0, "milliseconds": 1e-3, "days": 86400.0,
             "100-nanosecond ticks": 1e-7, "microseconds": 1e-6}.get(info.time_unit, 0.0)
    streams = [decode_stream(s, codec_overrides, time_scale=scale)
               for s in extract_streams(data, codec_overrides)]
    if len(streams) == 2 and all(s.alignment == "timed" and s.start is not None for s in streams):
        # put both streams on one clock
        base = min(s.start for s in streams)
        for s in streams:
            lead = int(round((s.start - base) * SAMPLE_RATE))
            if lead > 0:
                s.pcm = np.concatenate([np.zeros(lead, dtype=np.int16), s.pcm])
    return streams, info


def write_nmf(streams: dict[int, list[tuple[float, float, bytes]]], compression: int = 3,
              order: str = "<") -> bytes:
    """A synthetic NMF file (tests and demo data): packets of each stream,
    interleaved by time, each with a compression parameter, then an end
    packet."""
    out = bytearray()
    items = sorted(((start, sid, end, payload) for sid, chunks in streams.items()
                    for start, end, payload in chunks), key=lambda t: (t[0], t[1]))
    param = struct.pack(order + "hi", COMPRESSION_PARAM, 1) + struct.pack("b", compression) + b"\0" * 15
    for start, sid, end, payload in items:
        size = len(param) + len(payload)
        out += struct.pack(order + "bhbddII", 4, 0, sid, start, end, size, len(param))
        out += param + payload
    out += struct.pack(order + "bhbddII", END_TYPE, 0, 0, 0.0, 0.0, 0, 0)
    return bytes(out)
