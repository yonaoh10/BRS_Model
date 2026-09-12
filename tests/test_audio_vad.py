"""VAD selection and fallback.

Silero is the default VAD and it depends on torch. A VAD that cannot load is
an accuracy problem; a VAD that raises is a lost call. The second must never
happen, so every failure mode falls back to the energy VAD.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from callqa import audio as audio_mod
from callqa.config import Config

RATE = 16000


def _speech_like_wav(path: Path) -> Path:
    """Two bursts of tone separated by silence - enough for any VAD to segment."""
    rng = np.random.default_rng(0)
    t = np.arange(int(RATE * 1.5)) / RATE
    burst = (0.3 * np.sin(2 * np.pi * 180 * t) + 0.02 * rng.normal(size=len(t))).astype(np.float32)
    gap = np.zeros(int(RATE * 1.0), dtype=np.float32)
    signal = np.concatenate([gap, burst, gap, burst, gap])
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    return path


def _config(vad: str) -> Config:
    config = Config()
    config.audio.vad = vad          # type: ignore[assignment]
    config.run.mock = False
    return config


def test_energy_vad_finds_the_bursts(tmp_path: Path) -> None:
    segments = audio_mod.energy_vad(_speech_like_wav(tmp_path / "a.wav"))
    assert len(segments) == 2
    assert all(s.duration > 1.0 for s in segments)


def test_missing_silero_falls_back(tmp_path: Path, monkeypatch) -> None:
    def boom(*_a, **_k):
        raise ImportError("no silero-vad")
    monkeypatch.setattr(audio_mod, "silero_vad", boom)
    engine, segments = audio_mod.run_vad(_speech_like_wav(tmp_path / "a.wav"), _config("silero"))
    assert engine == "energy"
    assert segments


def test_broken_silero_falls_back_rather_than_failing_the_call(
    tmp_path: Path, monkeypatch
) -> None:
    """torchaudio has refused to decode without torchcodec installed, which
    surfaces as a RuntimeError, not an ImportError. Losing the call over a VAD
    is worse than losing VAD accuracy."""
    def boom(*_a, **_k):
        raise RuntimeError("torchaudio requires torchcodec for audio I/O")
    monkeypatch.setattr(audio_mod, "silero_vad", boom)
    engine, segments = audio_mod.run_vad(_speech_like_wav(tmp_path / "a.wav"), _config("silero"))
    assert engine == "energy"
    assert segments


def test_mock_mode_never_uses_silero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(audio_mod, "silero_vad", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("mock mode must not touch silero")))
    config = _config("silero")
    config.run.mock = True
    engine, _ = audio_mod.run_vad(_speech_like_wav(tmp_path / "a.wav"), config)
    assert engine == "energy"
