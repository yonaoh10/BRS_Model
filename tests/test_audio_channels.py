"""A file that declares two channels does not necessarily hold two recordings.

A call recorded on one phone and saved as stereo carries the same signal on
both channels; some recorders leave one channel dead. Both look like stereo in
the metadata, and both destroy speaker attribution silently if the pipeline
takes the channel-split path: duplicated channels put both speakers on both
sides, a dead channel leaves one party mute.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from callqa.audio import prepare_audio, probe_channels
from callqa.config import Config
from callqa.models import CallInput, CallMeta

RATE = 16000


def _write(path: Path, left: np.ndarray, right: np.ndarray) -> Path:
    stereo = np.stack([left, right], axis=1)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    return path


def _tone(freq: float, seconds: float = 3.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_genuine_stereo_is_recognised(tmp_path: Path) -> None:
    path = _write(tmp_path / "real.wav", _tone(220), _tone(700))
    assert probe_channels(path) == "stereo"


def test_duplicated_channels_are_recognised(tmp_path: Path) -> None:
    """What a single-device phone recording saved as stereo looks like."""
    voice = _tone(300)
    assert probe_channels(_write(tmp_path / "dual.wav", voice, voice.copy())) == "dual_mono"


def test_duplicated_channels_survive_lossy_encoding(tmp_path: Path) -> None:
    """AAC/MP3 leave the two channels close but not bit-identical, so the test
    has to be a tolerance, not equality."""
    voice = _tone(300)
    noise = np.random.default_rng(0).normal(0, 0.001, len(voice)).astype(np.float32)
    assert probe_channels(_write(tmp_path / "lossy.wav", voice, voice + noise)) == "dual_mono"


def test_dead_channel_is_recognised(tmp_path: Path) -> None:
    silence = np.zeros(int(RATE * 3), dtype=np.float32)
    assert probe_channels(_write(tmp_path / "half.wav", _tone(300), silence)) == "single_channel"


def test_prepare_audio_downmixes_a_dual_mono_file(tmp_path: Path) -> None:
    """The important consequence: no channel split, so no phantom second
    speaker, and the artifact says why."""
    voice = _tone(300, seconds=4.0)
    audio_path = _write(tmp_path / "CALL.wav", voice, voice.copy())
    call = CallInput(call_id="CALL", audio_path=audio_path, banker_id="B1", banker_channel="L")
    meta = CallMeta(call_id="CALL", banker_id="B1", file_name=audio_path.name,
                    duration_sec=4.0, sample_rate=RATE, channels=2,
                    codec="pcm_s16le", banker_channel="L")
    config = Config()
    config.audio.vad = "energy"

    artifact = prepare_audio(call, meta, config, tmp_path / "wav")

    assert artifact.channel_layout == "dual_mono"
    assert artifact.is_stereo is False
    assert artifact.mono_wav and not artifact.banker_wav
    assert artifact.banker_segments == [] and artifact.customer_segments == []


def test_prepare_audio_still_splits_a_real_stereo_file(tmp_path: Path) -> None:
    audio_path = _write(tmp_path / "CALL.wav", _tone(220, 4.0), _tone(700, 4.0))
    call = CallInput(call_id="CALL", audio_path=audio_path, banker_id="B1", banker_channel="L")
    meta = CallMeta(call_id="CALL", banker_id="B1", file_name=audio_path.name,
                    duration_sec=4.0, sample_rate=RATE, channels=2,
                    codec="pcm_s16le", banker_channel="L")
    config = Config()
    config.audio.vad = "energy"

    artifact = prepare_audio(call, meta, config, tmp_path / "wav")

    assert artifact.channel_layout == "stereo"
    assert artifact.is_stereo is True
    assert artifact.banker_wav and artifact.customer_wav


@pytest.mark.parametrize("channels", [1])
def test_mono_file_is_labelled_mono(tmp_path: Path, channels: int) -> None:
    path = tmp_path / "mono.wav"
    pcm = (_tone(300, 3.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    call = CallInput(call_id="M", audio_path=path, banker_id="B1")
    meta = CallMeta(call_id="M", banker_id="B1", file_name=path.name,
                    duration_sec=3.0, sample_rate=RATE, channels=channels,
                    codec="pcm_s16le")
    config = Config()
    config.audio.vad = "energy"
    assert prepare_audio(call, meta, config, tmp_path / "wav").channel_layout == "mono"
