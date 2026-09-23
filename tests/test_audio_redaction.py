"""The redacted-audio door.

The dashboard lets a supervisor play a call. The recording has the customer
reading identifiers aloud, so the audio served to the browser must itself be
redacted: silenced wherever the transcript was masked. These tests prove the
silence is real - that an identifier's samples are actually zero - which is the
whole justification for serving audio at all.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from callqa.audio import _read_wav
from callqa.audio_redaction import AUDIO_PAD_SEC, produce_redacted_audio
from callqa.models import AudioArtifact, CallInput, DialogTranscript, DialogTurn, Word
from callqa.pipeline import process_call
from callqa.portable import acl_summary, private_to_owner
from callqa.state import StateDB

RATE = 16000


def _words(text: str, start: float, step: float) -> list[Word]:
    """One Word per whitespace token, evenly spaced from `start`."""
    return [
        Word(word=tok, start=round(start + i * step, 3), end=round(start + (i + 1) * step, 3))
        for i, tok in enumerate(text.split())
    ]


def _tone_wav(path: Path, seconds: float) -> Path:
    """A continuous non-silent tone, so any zeroed region stands out."""
    t = np.arange(int(RATE * seconds)) / RATE
    signal = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes((np.clip(signal, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _dialog(turns: list[DialogTurn]) -> DialogTranscript:
    return DialogTranscript(call_id="AUD001", attribution_mode="mock", turns=turns)


def test_a_spoken_identifier_is_silenced(tmp_path: Path) -> None:
    """The digits of a national ID are spoken at [4.0, 5.0); those samples,
    padded, must be exactly zero, and audio on either side must survive."""
    # "המספר הוא <ID>" - the ID is the 3rd token, at 4.0-5.0s.
    words = _words("המספר הוא 123456782 תודה", start=2.0, step=1.0)
    id_word = next(w for w in words if w.word == "123456782")
    dialog = _dialog([DialogTurn(speaker="customer", start=2.0, end=6.0,
                                 text="המספר הוא 123456782 תודה", words=words)])
    art = AudioArtifact(call_id="AUD001", is_stereo=False,
                        mono_wav=str(_tone_wav(tmp_path / "src.wav", 8.0)),
                        sample_rate=RATE, vad_engine="energy")

    out = tmp_path / "redacted_audio" / "AUD001.wav"
    ranges = produce_redacted_audio(art, dialog, [], out)

    assert ranges, "the ID must have produced a silenced range"
    assert any(a <= id_word.start and b >= id_word.end for a, b in ranges)

    samples, rate = _read_wav(out)
    mono = samples[:, 0]
    lo, hi = int((id_word.start - AUDIO_PAD_SEC) * rate), int((id_word.end + AUDIO_PAD_SEC) * rate)
    assert float(np.max(np.abs(mono[lo:hi]))) == 0.0, "the identifier plays back as silence"
    # a control region well before the ID keeps its audio
    assert float(np.max(np.abs(mono[: int(1.0 * rate)]))) > 0.05


def test_dictated_number_words_are_silenced(tmp_path: Path) -> None:
    """Customers dictate digits as Hebrew words; the fold path must silence the
    whole spoken run, not leave it audible because no digit was ever written."""
    text = "החשבון הוא ארבע חמש שמונה אפס בסניף"
    words = _words(text, start=1.0, step=0.5)
    run = [w for w in words if w.word in ("ארבע", "חמש", "שמונה", "אפס")]
    dialog = _dialog([DialogTurn(speaker="customer", start=1.0, end=5.0, text=text, words=words)])
    art = AudioArtifact(call_id="AUD001", is_stereo=False,
                        mono_wav=str(_tone_wav(tmp_path / "src.wav", 6.0)),
                        sample_rate=RATE, vad_engine="energy")

    out = tmp_path / "redacted_audio" / "AUD001.wav"
    ranges = produce_redacted_audio(art, dialog, [], out)

    lo_t, hi_t = run[0].start, run[-1].end
    assert any(a <= lo_t and b >= hi_t for a, b in ranges), "the whole dictated run is silenced"
    samples, rate = _read_wav(out)
    mono = samples[:, 0]
    assert float(np.max(np.abs(mono[int(lo_t * rate):int(hi_t * rate)]))) == 0.0


def test_a_banker_name_is_silenced(tmp_path: Path) -> None:
    """A banker's name is redacted in text (never reported); it must be silenced
    in audio too, so the recording does not read it back in the clear."""
    text = "מדבר דוד מהמוקד שלום"
    words = _words(text, start=0.0, step=1.0)
    name_word = next(w for w in words if w.word == "דוד")
    dialog = _dialog([DialogTurn(speaker="banker", start=0.0, end=5.0, text=text, words=words)])
    art = AudioArtifact(call_id="AUD001", is_stereo=False,
                        mono_wav=str(_tone_wav(tmp_path / "src.wav", 6.0)),
                        sample_rate=RATE, vad_engine="energy")

    out = tmp_path / "redacted_audio" / "AUD001.wav"
    ranges = produce_redacted_audio(art, dialog, ["דוד"], out)
    assert any(a <= name_word.start and b >= name_word.end for a, b in ranges)
    samples, rate = _read_wav(out)
    mono = samples[:, 0]
    lo, hi = int(name_word.start * rate), int(name_word.end * rate)
    assert float(np.max(np.abs(mono[lo:hi]))) == 0.0


def test_a_clean_call_is_left_untouched(tmp_path: Path) -> None:
    """No identifiers, no names: nothing is silenced and the audio is intact."""
    text = "שלום מה שלומך תודה רבה על העזרה"
    dialog = _dialog([DialogTurn(speaker="customer", start=0.0, end=4.0, text=text,
                                 words=_words(text, start=0.0, step=0.5))])
    art = AudioArtifact(call_id="AUD001", is_stereo=False,
                        mono_wav=str(_tone_wav(tmp_path / "src.wav", 5.0)),
                        sample_rate=RATE, vad_engine="energy")
    out = tmp_path / "redacted_audio" / "AUD001.wav"
    ranges = produce_redacted_audio(art, dialog, [], out)
    assert ranges == []
    samples, _ = _read_wav(out)
    assert float(np.max(np.abs(samples[:, 0]))) > 0.05


def test_a_wordless_masked_turn_silences_the_whole_turn(tmp_path: Path) -> None:
    """If a turn holds an identifier but carries no word timings, the fallback
    silences the whole turn rather than let it play in the clear."""
    dialog = _dialog([
        DialogTurn(speaker="customer", start=0.0, end=2.0, text="שלום", words=[]),
        DialogTurn(speaker="customer", start=2.0, end=4.0,
                   text="תעודת הזהות שלי 123456782", words=[]),
    ])
    art = AudioArtifact(call_id="AUD001", is_stereo=False,
                        mono_wav=str(_tone_wav(tmp_path / "src.wav", 5.0)),
                        sample_rate=RATE, vad_engine="energy")
    out = tmp_path / "redacted_audio" / "AUD001.wav"
    ranges = produce_redacted_audio(art, dialog, [], out)
    assert any(a <= 2.0 and b >= 4.0 for a, b in ranges), "the masked turn is silenced whole"


def test_number_split_across_wordless_turns_is_silenced(tmp_path: Path) -> None:
    """A number the speaker paused in the middle of arrives as two turns; if
    neither carries word timings, per-turn detection misses it but the text
    redactor (which joins turns) masks it. Both turns must be silenced."""
    dialog = _dialog([
        DialogTurn(speaker="customer", start=0.0, end=2.0,
                   text="תעודת הזהות שלי 12345", words=[]),
        DialogTurn(speaker="customer", start=2.5, end=3.0, text="6782", words=[]),
    ])
    art = AudioArtifact(call_id="AUD001", is_stereo=False,
                        mono_wav=str(_tone_wav(tmp_path / "src.wav", 4.0)),
                        sample_rate=RATE, vad_engine="energy")
    out = tmp_path / "redacted_audio" / "AUD001.wav"
    ranges = produce_redacted_audio(art, dialog, [], out)
    # the joined text "…12345\n6782" is the valid ID 123456782 → both turns silenced
    assert any(a <= 0.0 and b >= 2.0 for a, b in ranges)
    assert any(a <= 2.5 and b >= 3.0 for a, b in ranges)
    samples, rate = _read_wav(out)
    assert float(np.max(np.abs(samples[:, 0][int(2.5 * rate):int(3.0 * rate)]))) == 0.0


def test_partial_word_list_missing_the_identifier_silences_the_turn(tmp_path: Path) -> None:
    """A turn whose words cover the surrounding speech but NOT the spoken number
    (the identifier is absent from the word stream) must still be silenced —
    the whole turn, since we cannot trust word positions for the gap."""
    text = "תעודת הזהות שלי 123456782"
    words = _words("תעודת הזהות שלי", start=0.0, step=1.0)  # number absent
    dialog = _dialog([DialogTurn(speaker="customer", start=0.0, end=5.0, text=text, words=words)])
    art = AudioArtifact(call_id="AUD001", is_stereo=False,
                        mono_wav=str(_tone_wav(tmp_path / "src.wav", 6.0)),
                        sample_rate=RATE, vad_engine="energy")
    out = tmp_path / "redacted_audio" / "AUD001.wav"
    ranges = produce_redacted_audio(art, dialog, [], out)
    assert any(a <= 0.0 and b >= 5.0 for a, b in ranges), "whole turn silenced"


def test_stereo_downmix_is_silenced(tmp_path: Path) -> None:
    """The stereo path downmixes banker+customer to one timeline; a masked
    identifier in a customer turn must be silent in the mix."""
    text = "המספר הוא 123456782 תודה"
    words = _words(text, start=1.0, step=1.0)
    id_word = next(w for w in words if w.word == "123456782")
    dialog = _dialog([DialogTurn(speaker="customer", start=1.0, end=6.0, text=text, words=words)])
    art = AudioArtifact(
        call_id="AUD001", is_stereo=True,
        banker_wav=str(_tone_wav(tmp_path / "b.wav", 7.0)),
        customer_wav=str(_tone_wav(tmp_path / "c.wav", 7.0)),
        sample_rate=RATE, vad_engine="energy")
    out = tmp_path / "redacted_audio" / "AUD001.wav"
    produce_redacted_audio(art, dialog, [], out)
    samples, rate = _read_wav(out)
    lo, hi = int(id_word.start * rate), int(id_word.end * rate)
    assert float(np.max(np.abs(samples[:, 0][lo:hi]))) == 0.0
    assert float(np.max(np.abs(samples[:, 0][: int(0.5 * rate)]))) > 0.05  # audio survives


def test_regeneration_overwrites_previous_audio(tmp_path: Path) -> None:
    """When the transcript is re-masked (e.g. --force), re-producing to the same
    path must overwrite: a formerly-clean recording that now contains an ID must
    end up silenced, never keep the stale un-silenced audio."""
    src = _tone_wav(tmp_path / "src.wav", 6.0)
    art = AudioArtifact(call_id="AUD001", is_stereo=False, mono_wav=str(src),
                        sample_rate=RATE, vad_engine="energy")
    out = tmp_path / "redacted_audio" / "AUD001.wav"
    clean = _dialog([DialogTurn(speaker="customer", start=0.0, end=4.0,
                                text="שלום מה שלומך", words=_words("שלום מה שלומך", 0.0, 1.0))])
    assert produce_redacted_audio(art, clean, [], out) == []
    text = "המספר הוא 123456782 תודה"
    words = _words(text, start=0.0, step=1.0)
    id_word = next(w for w in words if w.word == "123456782")
    pii = _dialog([DialogTurn(speaker="customer", start=0.0, end=4.0, text=text, words=words)])
    produce_redacted_audio(art, pii, [], out)  # same path — must overwrite
    samples, rate = _read_wav(out)
    assert float(np.max(np.abs(samples[:, 0][int(id_word.start * rate):int(id_word.end * rate)]))) == 0.0


def test_sidecar_metadata_matches_the_silences(tmp_path: Path) -> None:
    """The dashboard draws the silences from a sidecar JSON; it must agree with
    the ranges actually zeroed, and never carry raw digits."""
    import json

    text = "המספר הוא 123456782 תודה"
    dialog = _dialog([DialogTurn(speaker="customer", start=0.0, end=4.0, text=text,
                                 words=_words(text, start=0.0, step=1.0))])
    art = AudioArtifact(call_id="AUD001", is_stereo=False,
                        mono_wav=str(_tone_wav(tmp_path / "src.wav", 5.0)),
                        sample_rate=RATE, vad_engine="energy")
    out = tmp_path / "redacted_audio" / "AUD001.wav"
    ranges = produce_redacted_audio(art, dialog, [], out)

    meta = json.loads((out.with_suffix(".json")).read_text(encoding="utf-8"))
    assert meta["sample_rate"] == RATE
    assert len(meta["silences"]) == len(ranges)
    assert "123456782" not in json.dumps(meta)


# ------------------------------------------------- pipeline hook (idempotency)

def _speech_wav(path: Path, seconds: float = 45.0) -> Path:
    """A mono file with alternating bursts so energy VAD finds real speech."""
    t = np.arange(int(RATE * seconds)) / RATE
    sig = np.zeros_like(t, dtype=np.float32)
    for i in range(int(seconds // 6)):
        lo, hi = int(i * 6 * RATE), int((i * 6 + 5) * RATE)
        freq = 160 if i % 2 == 0 else 220
        sig[lo:hi] = (0.3 * np.sin(2 * np.pi * freq * t[lo:hi])).astype(np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes((np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


@pytest.fixture()
def processed(workspace, engines, tmp_path):  # noqa: ANN001
    call = CallInput(call_id="AUDIDEM1", audio_path=_speech_wav(tmp_path / "call.wav"),
                     banker_id="B1")
    process_call(call, engines, StateDB(workspace.paths.state_db))
    return workspace, call, engines


def _audio_paths(workspace, call):  # noqa: ANN001
    wav = workspace.paths.output_dir / "redacted_audio" / f"{call.call_id}.wav"
    return wav, wav.with_suffix(".json")


def test_pipeline_writes_redacted_audio_and_sidecar(processed) -> None:  # noqa: ANN001
    workspace, call, _ = processed
    wav, sidecar = _audio_paths(workspace, call)
    assert wav.is_file() and sidecar.is_file()
    assert wav.read_bytes()[:4] == b"RIFF"
    if os.name == "nt":
        # chmod means nothing there: the folder's permissions are replaced by
        # owner + SYSTEM + Administrators, and the WAV inherits them.
        assert private_to_owner(wav.parent), acl_summary(wav.parent)
        assert private_to_owner(wav), acl_summary(wav)
    else:
        # written 0600 inside a 0700 dir
        assert (wav.stat().st_mode & 0o777) == 0o600
        assert (wav.parent.stat().st_mode & 0o777) == 0o700


def test_missing_sidecar_is_regenerated_without_force(processed) -> None:  # noqa: ANN001
    """A crash between the WAV and its sidecar must self-heal on a normal
    re-run, not leave the call permanently audio-less until --force."""
    workspace, call, engines = processed
    _, sidecar = _audio_paths(workspace, call)
    sidecar.unlink()
    process_call(call, engines, StateDB(workspace.paths.state_db))  # non-force resume
    assert sidecar.is_file(), "the sidecar must be regenerated on a normal re-run"


def test_recomputed_redaction_regenerates_stale_audio(processed) -> None:  # noqa: ANN001
    """If the redaction stage recomputes (the mask may have changed), the audio
    must be regenerated, never left as the previous run's file."""
    workspace, call, engines = processed
    wav, _ = _audio_paths(workspace, call)
    wav.write_bytes(b"STALE-NOT-A-WAV")                       # stand-in for stale audio
    (workspace.paths.output_dir / "redacted" / f"{call.call_id}.json").unlink()  # force recompute
    process_call(call, engines, StateDB(workspace.paths.state_db))
    assert wav.read_bytes()[:4] == b"RIFF", "recomputed redaction must rewrite the audio"


# -- coverage is accounted per TURN, never by time ----------------------------
#
# The safety net used to ask "does any silenced word overlap this turn's time
# span?". That leaks on the most ordinary recording there is: stereo turns
# overlap in time, so an identifier silenced in one speaker's turn marked the
# other speaker's overlapping turn as covered. Found by an adversarial review,
# reproduced, fixed; these pin every shape of it.

def _silenced(ranges: list[tuple[float, float]], start: float, end: float) -> bool:
    return any(a <= start and b >= end for a, b in ranges)


def test_an_overlapping_turn_of_the_other_speaker_is_not_covered_by_time() -> None:
    from callqa.audio_redaction import masked_time_ranges
    from callqa.redaction import RegexRedactor

    banker = DialogTurn(speaker="banker", start=0.0, end=10.0,
                        text="אפשר לחזור אליך לטלפון 052-1234567 תודה",
                        words=_words("אפשר לחזור אליך לטלפון 052-1234567 תודה", 0.5, 1.5))
    # The customer speaks over the banker, and the ASR produced no word timings.
    customer = DialogTurn(speaker="customer", start=2.0, end=8.0,
                          text="תעודת זהות 123456782", words=[])
    dialog = DialogTranscript(call_id="OV", attribution_mode="stereo",
                              turns=[banker, customer])
    redacted = RegexRedactor().redact_dialog(dialog)

    ranges = masked_time_ranges(dialog, [], redacted)
    assert _silenced(ranges, 2.0, 8.0), f"the customer's national ID stays audible: {ranges}"


def test_a_second_identifier_in_a_partially_timed_turn_is_not_hidden_by_the_first() -> None:
    """Locating identifier A in a turn used to mark the whole turn covered, so
    identifier B - present in the text, absent from a partial word list -
    played in the clear. Counting, not overlap, is what closes it."""
    from callqa.audio_redaction import masked_time_ranges
    from callqa.redaction import RegexRedactor

    text = "הטלפון 052-1234567 ותעודת הזהות 123456782"
    turn = DialogTurn(speaker="customer", start=0.0, end=12.0, text=text,
                      # the word list stops before the national ID
                      words=_words("הטלפון 052-1234567", 0.0, 1.0))
    dialog = DialogTranscript(call_id="PART", attribution_mode="stereo", turns=[turn])
    redacted = RegexRedactor().redact_dialog(dialog)

    ranges = masked_time_ranges(dialog, [], redacted)
    assert _silenced(ranges, 0.0, 12.0), f"the second identifier stays audible: {ranges}"


def test_whatever_the_text_masked_is_silent_even_if_audio_detection_cannot_see_it() -> None:
    """The redacted transcript is the ground truth. A mask written by a rule or
    model the audio stage does not re-run - NER, presidio, anything added
    later - must still silence that turn."""
    from callqa.audio_redaction import masked_time_ranges
    from callqa.models import RedactedTranscript, RedactedTurn

    # A third party named in passing: no rule in find_pii recognises it (no
    # question, no self-naming phrase), only a model would.
    spoken = "תמסור בבקשה למיכל אברמוביץ שהתקשרתי"
    turn = DialogTurn(speaker="customer", start=3.0, end=7.0,
                      text=spoken, words=_words(spoken, 3.0, 0.8))
    dialog = DialogTranscript(call_id="NER", attribution_mode="stereo", turns=[turn])
    # As if an NER model had masked the name; nothing in find_pii knows it.
    redacted = RedactedTranscript(
        call_id="NER", engine="ner", enabled=True, redaction_counts={"PERSON": 1},
        turns=[RedactedTurn(speaker="customer", start=3.0, end=7.0,
                            text="תמסור בבקשה ל<שם:████> שהתקשרתי")])

    assert _silenced(masked_time_ranges(dialog, [], redacted), 3.0, 7.0)


def test_a_clean_turn_is_not_silenced() -> None:
    """The other direction: accounting must not turn into silencing everything."""
    from callqa.audio_redaction import masked_time_ranges
    from callqa.redaction import RegexRedactor

    turn = DialogTurn(speaker="banker", start=0.0, end=5.0, text="בוקר טוב, איך אפשר לעזור?",
                      words=_words("בוקר טוב, איך אפשר לעזור?", 0.0, 1.0))
    dialog = DialogTranscript(call_id="CLEAN", attribution_mode="stereo", turns=[turn])
    assert masked_time_ranges(dialog, [], RegexRedactor().redact_dialog(dialog)) == []
