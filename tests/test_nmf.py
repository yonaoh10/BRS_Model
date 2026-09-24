"""NICE .nmf recordings: parsing, inspection (structure only) and decoding."""

from __future__ import annotations

import random
import shutil
import struct
import subprocess

import numpy as np
import pytest

from callqa.journey import g711, nmf

RATE = 8000


def tone(freq: float, seconds: float, amp: int = 8000) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.int16)


def chunked(pcm: np.ndarray, t0: float, encode=g711.encode_alaw, step: int = 1600,
            time_scale: float = 1.0) -> list[tuple[float, float, bytes]]:
    out = []
    for i in range(0, len(pcm), step):
        piece = pcm[i:i + step]
        start = t0 + i / RATE
        out.append((start / time_scale, (start + len(piece) / RATE) / time_scale, encode(piece)))
    return out


def two_party(seconds: float = 3.0, lead: float = 0.5, compression: int = 3,
              encode=g711.encode_alaw, order: str = "<") -> bytes:
    return nmf.write_nmf({0: chunked(tone(300, seconds), 100.0, encode),
                          1: chunked(tone(500, seconds, 6000), 100.0 + lead, encode)},
                         compression=compression, order=order)


def test_inspect_reports_structure_only():
    data = two_party()
    info = nmf.inspect_bytes(data)
    assert info.byte_order == "little"
    assert [s.stream for s in info.streams] == [0, 1]
    assert all(s.codes == {3: 15} for s in info.streams)
    assert info.time_unit == "seconds"
    assert info.duration_sec == pytest.approx(3.0)
    text = "\n".join(info.lines())
    assert "alaw" in text and "two streams" in text
    # nothing of the payload is printed
    payload = g711.encode_alaw(tone(300, 0.01))
    assert payload.hex()[:16] not in text


def test_decode_keeps_two_streams_on_one_clock():
    streams, info = nmf.decode_file(two_party(lead=0.5))
    a, b = streams
    assert a.alignment == b.alignment == "timed"
    assert len(a.pcm) == 3 * RATE
    assert len(b.pcm) == int(3.5 * RATE)              # 0.5 s of lead silence
    assert not b.pcm[: int(0.4 * RATE)].any()
    corr = np.corrcoef(a.pcm.astype(float), tone(300, 3.0).astype(float))[0, 1]
    assert corr > 0.99


def test_gaps_the_recorder_skipped_come_back_as_silence():
    pcm = tone(300, 2.0)
    first = chunked(pcm[:RATE], 50.0)
    second = chunked(pcm[RATE:], 53.0)                 # 2 s pause not stored
    streams, _ = nmf.decode_file(nmf.write_nmf({0: first + second}))
    assert len(streams[0].pcm) == pytest.approx(4 * RATE, abs=10)


def test_time_units_are_recognised_from_the_audio():
    for scale, name in ((1e-3, "milliseconds"), (86400.0, "days")):
        data = nmf.write_nmf({0: chunked(tone(300, 2.0), 1000.0, time_scale=scale)})
        assert nmf.inspect_bytes(data).time_unit == name


def test_mulaw_and_big_endian():
    data = two_party(compression=7, encode=g711.encode_ulaw, order=">")
    info = nmf.inspect_bytes(data)
    assert info.byte_order == "big"
    streams, _ = nmf.decode_file(data)
    assert np.corrcoef(streams[0].pcm[:RATE].astype(float),
                       tone(300, 1.0).astype(float))[0, 1] > 0.99


def test_truncated_file_is_read_up_to_the_break():
    data = two_party()
    info = nmf.inspect_bytes(data[:-500])
    assert info.truncated
    assert sum(s.packets for s in info.streams) >= 25


def test_random_bytes_never_crash():
    rng = random.Random(7)
    for _ in range(300):
        blob = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 400)))
        try:
            nmf.inspect_bytes(blob)
        except nmf.NMFError:
            pass


def test_unknown_codec_is_reported_not_guessed_silently():
    data = nmf.write_nmf({0: chunked(tone(300, 1.0), 10.0)}, compression=42)
    info = nmf.inspect_bytes(data)
    assert any("not known" in n for n in info.notes)


def test_codec_override():
    data = nmf.write_nmf({0: chunked(tone(300, 1.0), 10.0)}, compression=42)
    streams, _ = nmf.decode_file(data, codec_overrides={42: "alaw"})
    assert len(streams[0].pcm) == RATE


def test_random_payload_is_flagged_as_possibly_encrypted():
    rng = np.random.default_rng(1)
    chunks = [(i * 0.2, (i + 1) * 0.2, rng.bytes(1600)) for i in range(20)]
    info = nmf.inspect_bytes(nmf.write_nmf({0: chunks}))
    assert any("encrypted" in n for n in info.notes)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_g722_goes_through_ffmpeg(tmp_path):
    src = tmp_path / "t.wav"
    import wave
    with wave.open(str(src), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        t = np.arange(32000) / 16000
        wf.writeframes((np.sin(2 * np.pi * 400 * t) * 8000).astype(np.int16).tobytes())
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "g722", "-"],
                         capture_output=True, check=True).stdout
    pcm = nmf.decode_payload(raw, 19)
    assert len(pcm) == pytest.approx(2 * RATE, rel=0.02)   # resampled to 8 kHz


def test_header_layout_is_28_bytes():
    assert struct.calcsize("<bhbddII") == nmf.HEADER
