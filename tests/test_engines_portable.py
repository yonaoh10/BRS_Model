"""Engine choices that used to depend on the machine being a Linux GPU server.

The bank's desktops are Windows VDI with no GPU. Two defaults broke there
without any code being wrong on Linux: the ASR compute type, and pyannote's
own audio decoding.
"""

from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest


def _fake_ct2(monkeypatch, gpus: int) -> None:  # noqa: ANN001
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        types.SimpleNamespace(get_cuda_device_count=lambda: gpus))


def test_gpu_only_compute_type_falls_back_to_int8_without_a_gpu(monkeypatch) -> None:  # noqa: ANN001
    from callqa.asr.faster_whisper_engine import _usable_compute_type

    _fake_ct2(monkeypatch, gpus=0)
    assert _usable_compute_type("float16") == "int8"
    assert _usable_compute_type("int8_float16") == "int8"
    assert _usable_compute_type("int8") == "int8"          # already CPU-safe
    assert _usable_compute_type("float32") == "float32"


def test_gpu_compute_type_is_kept_when_there_is_a_gpu(monkeypatch) -> None:  # noqa: ANN001
    from callqa.asr.faster_whisper_engine import _usable_compute_type

    _fake_ct2(monkeypatch, gpus=1)
    assert _usable_compute_type("float16") == "float16"


def test_pyannote_gets_the_audio_in_memory_not_a_path(tmp_path: Path) -> None:
    """A path makes pyannote decode with torchcodec, which needs FFmpeg DLLs a
    Windows desktop does not have."""
    torch = pytest.importorskip("torch")
    from callqa.config import SpeakersConfig
    from callqa.speakers.pyannote_engine import LazyPyannoteDiarizer

    wav = tmp_path / "mono.wav"
    pcm = (np.sin(np.linspace(0, 2000, 16000)) * 12000).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())

    seen = {}

    class _Pipeline:
        def __call__(self, audio, **kwargs):  # noqa: ANN001,ANN003,ANN204
            seen["audio"] = audio
            return []

    diarizer = LazyPyannoteDiarizer(SpeakersConfig())
    diarizer._pipeline = _Pipeline()                       # noqa: SLF001
    diarizer.diarize(wav, "C1")
    audio = seen["audio"]
    assert isinstance(audio, dict) and audio["sample_rate"] == 16000
    assert isinstance(audio["waveform"], torch.Tensor)
    assert tuple(audio["waveform"].shape) == (1, 16000)


def _mulaw_stereo_wav(path: Path) -> Path:
    """A G.711 mu-law stereo WAV - what call recorders export, and a format the
    standard library's wave module refuses ("unknown format: 7")."""
    av = pytest.importorskip("av")
    t = np.arange(8000 * 3) / 8000
    left = (np.sin(2 * np.pi * 300 * t) * 12000).astype(np.int16)
    right = (np.sin(2 * np.pi * 700 * t) * 12000).astype(np.int16) * (t > 1.5)
    interleaved = np.stack([left, right.astype(np.int16)], axis=1).reshape(1, -1)
    with av.open(str(path), "w", format="wav") as out:
        stream = out.add_stream("pcm_mulaw", rate=8000, layout="stereo")
        frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
        frame.sample_rate = 8000
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return path


def test_telephony_wav_is_read_without_ffmpeg(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """No ffmpeg.exe on a VDI desktop: PyAV, installed with the engines, decodes."""
    from callqa import audio as audio_mod
    from callqa.ingestion import _probe_with_pyav

    src = _mulaw_stereo_wav(tmp_path / "call.wav")
    with pytest.raises(wave.Error):
        wave.open(str(src), "rb")                       # the stdlib alone cannot

    probe = _probe_with_pyav(src)
    assert probe["streams"][0]["channels"] == 2
    assert abs(probe["format"]["duration"] - 3.0) < 0.1

    monkeypatch.setattr(audio_mod, "_have_ffmpeg", lambda: False)
    dst = tmp_path / "left.wav"
    audio_mod._extract_channel(src, dst, 0, 16000)
    data, rate = audio_mod._read_wav(dst)
    assert rate == 16000 and data.shape[1] == 1 and abs(len(data) / rate - 3.0) < 0.1
    window = audio_mod._decode_stereo_window(src, 0.0, 3.0)
    assert window is not None and window.shape[1] == 2
