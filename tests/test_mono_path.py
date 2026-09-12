"""The single-recording path, end to end.

Most calls the bank will hand this system are one file with both people on it.
That path has to produce the same artifacts as the stereo path, say clearly
that the roles were inferred, and refuse to present a near-coin-flip as a
finished score.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from callqa.models import CallInput
from callqa.pipeline import process_call
from callqa.state import StateDB

RATE = 16000


def _mono_call(tmp_path: Path, call_id: str = "MONO001", seconds: float = 120.0) -> Path:
    """A single-channel file with alternating bursts, like a phone recording."""
    rng = np.random.default_rng(7)
    t = np.arange(int(RATE * seconds)) / RATE
    signal = np.zeros_like(t, dtype=np.float32)
    for i in range(int(seconds // 8)):
        start, end = int(i * 8 * RATE), int((i * 8 + 7) * RATE)
        freq = 160 if i % 2 == 0 else 220        # two "voices"
        chunk = t[start:end]
        signal[start:end] = (0.3 * np.sin(2 * np.pi * freq * chunk)
                             + 0.01 * rng.normal(size=len(chunk))).astype(np.float32)
    path = tmp_path / f"{call_id}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes((np.clip(signal, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


@pytest.fixture()
def mono_result(tmp_path: Path, workspace, engines):  # noqa: ANN001
    call = CallInput(call_id="MONO001", audio_path=_mono_call(tmp_path),
                     banker_id="B900")
    state = StateDB(workspace.paths.state_db)
    result = process_call(call, engines, state)
    return result, workspace


def test_a_single_channel_call_runs_end_to_end(mono_result) -> None:  # noqa: ANN001
    result, workspace = mono_result
    assert result.status in ("success", "needs_human_review"), result.error
    out = workspace.paths.output_dir
    for name in ("audio", "transcripts", "redacted", "features", "scores"):
        assert list((out / name).glob("MONO001*")), f"no {name} artifact"
    assert (out / "reports" / "calls" / "MONO001.html").exists()


def test_the_dialog_records_how_the_roles_were_decided(mono_result) -> None:  # noqa: ANN001
    from callqa.models import DialogTranscript

    _, workspace = mono_result
    path = workspace.paths.output_dir / "transcripts" / "MONO001.dialog.json"
    dialog = DialogTranscript.model_validate_json(path.read_text(encoding="utf-8"))

    assert dialog.attribution_mode == "mono_diarized"
    assert dialog.role_signals, "the role decision must be auditable"
    assert dialog.diarization is not None
    assert dialog.diarization.words_attributed > 0
    assert {t.speaker for t in dialog.turns} == {"banker", "customer"}


def test_the_report_says_the_roles_were_inferred(mono_result) -> None:  # noqa: ANN001
    _, workspace = mono_result
    html = (workspace.paths.output_dir / "reports" / "calls" / "MONO001.html").read_text(
        encoding="utf-8")
    assert "ייחוס דוברים משוער" in html, "a mono call must not read as certain"


def test_a_near_coin_flip_is_held_for_a_human(tmp_path: Path, workspace, engines) -> None:  # noqa: ANN001
    """Raising the bar above what the evidence supports must stop the call
    being reported as a finished score - that is the whole point of the gate."""
    workspace.speakers.min_role_confidence = 0.99
    call = CallInput(call_id="MONO002", audio_path=_mono_call(tmp_path, "MONO002"),
                     banker_id="B900")
    result = process_call(call, engines, StateDB(workspace.paths.state_db))

    assert result.status == "needs_human_review"
    assert "role" in (result.error or "").lower()


def test_stereo_calls_are_unaffected(workspace, engines) -> None:  # noqa: ANN001
    """The mono work must not have touched the path that has real channels."""
    from callqa.models import DialogTranscript

    call = CallInput(call_id="CALL001",
                     audio_path=workspace.paths.input_dir / "calls" / "CALL001.wav",
                     banker_id="B001", banker_channel="L")
    result = process_call(call, engines, StateDB(workspace.paths.state_db))
    assert result.status == "success"
    dialog = DialogTranscript.model_validate_json(
        (workspace.paths.output_dir / "transcripts" / "CALL001.dialog.json")
        .read_text(encoding="utf-8"))
    assert dialog.attribution_mode == "stereo"
    assert dialog.role_confidence == 1.0
    assert dialog.role_signals == []
