"""Regressions from the adversarial QA sweep.

Every test here reproduces a defect that was found by trying to break the
system rather than by trying to use it. They are grouped by what an operator
would have seen: a leak, a wrong number, a lost call, or an open door.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from callqa.audio import energy_vad, probe_channels
from callqa.ingestion import load_metadata, sanitize_call_id
from callqa.models import CallInput, DimensionScore, RedactedTranscript, RedactedTurn
from callqa.redaction import redact_text, sanitize_error
from callqa.reporting.common import safe_filename

RATE = 16000


# ------------------------------------------------------- PII must not escape

class TestRedactionLeaks:
    """A number spoken aloud arrives in whatever shape the speaker used."""

    @pytest.mark.parametrize("spoken", [
        "123456782",
        "123-456-782",
        "123 456 782",
        "123.456.782",
        "1 2 3 4 5 6 7 8 2",
    ])
    def test_an_id_is_masked_however_it_was_spoken(self, spoken: str) -> None:
        redacted, counts = redact_text(f"תעודת הזהות שלי היא {spoken}")
        assert "████" in redacted
        assert spoken not in redacted
        assert counts

    def test_an_invisible_character_inside_a_number_does_not_hide_it(self) -> None:
        """Hebrew text carries bidi marks constantly; one inside a digit run
        used to defeat every recognizer at once."""
        for invisible in ("​", "‏", "‫", "﻿"):
            spiked = f"12345{invisible}6782"
            redacted, _ = redact_text(f"תעודת זהות {spiked}")
            assert "123456782" not in redacted.replace(invisible, "")
            assert "████" in redacted

    def test_non_ascii_digits_are_masked(self) -> None:
        arabic = "١٢٣٤٥٦٧٨٢"
        redacted, _ = redact_text(f"תעודת זהות {arabic}")
        assert "████" in redacted
        assert arabic not in redacted

    def test_a_number_split_across_two_turns_is_caught(self) -> None:
        """The speaker pauses mid-number, so the ASR emits two segments and
        neither half is recognisable on its own."""
        from callqa.models import DialogTranscript, DialogTurn
        from callqa.redaction import RegexRedactor

        dialog = DialogTranscript(
            call_id="SPLIT", attribution_mode="stereo",
            turns=[
                DialogTurn(speaker="customer", start=0, end=2, text="המספר שלי הוא 1234"),
                DialogTurn(speaker="customer", start=2, end=4, text="56782, בבקשה"),
            ],
        )
        result = RegexRedactor().redact_dialog(dialog)
        joined = " ".join(t.text for t in result.turns)
        assert "1234" not in joined and "56782" not in joined
        assert "████" in joined

    @pytest.mark.parametrize("text,must_go", [
        ("שלח לי לכתובת yossi.cohen@gmail.com בבקשה", "yossi.cohen@gmail.com"),
        ("החשבון הוא IL620108000000099999999", "IL620108000000099999999"),
        ("אפשר להתקשר ל*2121", "*2121"),
    ])
    def test_other_identifiers_are_recognised(self, text: str, must_go: str) -> None:
        redacted, _ = redact_text(text)
        assert must_go not in redacted

    def test_the_mothers_name_is_masked(self) -> None:
        redacted, counts = redact_text("מה שם האם שלך? שם האם הוא רבקה")
        assert "רבקה" not in redacted
        assert counts.get("PERSON")

    def test_an_amount_is_not_an_account_number(self) -> None:
        """Masking every price blinds the compliance and clarity dimensions,
        which are scored on what the banker actually quoted."""
        redacted, counts = redact_text("הפקדת 120000 שקל בחיסכון")
        assert "120000" in redacted
        assert not counts

    def test_a_banker_name_is_matched_as_a_whole_word(self) -> None:
        """Substring replacement shredded ordinary Hebrew: the name אור
        turned האורח and באורך into masks."""
        redacted, counts = redact_text(
            "האורח חיכה באורך רוח, ואז אור ענה", extra_names=["אור"])
        assert "האורח" in redacted and "באורך" in redacted
        assert counts["PERSON"] == 1

    def test_an_error_message_carrying_transcript_text_is_scrubbed(self) -> None:
        """Exceptions raised before redaction quote the text that caused them,
        and that message is written to results/<call>.json and to the log."""
        message = sanitize_error(ValueError("bad turn: 'המספר הוא 123456782'"))
        assert "123456782" not in message
        assert "████" in message


# -------------------------------------------------- identifiers are not paths

class TestIdentifiersAreNotPaths:
    @pytest.mark.parametrize("hostile", [
        "../../../etc/passwd",
        "/tmp/evil",
        'A" onmouseover="alert(1)" x="',
        "call\nid",
    ])
    def test_a_hostile_call_id_is_refused(self, hostile: str) -> None:
        with pytest.raises(ValueError, match="safe identifier"):
            CallInput(call_id=hostile, audio_path=Path("x.wav"))

    def test_a_filename_is_turned_into_a_safe_call_id(self) -> None:
        assert sanitize_call_id("../../../etc/passwd") == "etc_passwd"
        assert sanitize_call_id("CALL001") == "CALL001"
        assert sanitize_call_id("") == "call"

    def test_a_banker_id_cannot_choose_where_its_report_is_written(self) -> None:
        assert "/" not in safe_filename("../../../BANKERPWN")
        assert safe_filename("B001") == "B001"

    def test_metadata_rejects_a_file_name_that_is_a_path(self, tmp_path: Path) -> None:
        """An absolute or traversing file_name reads audio from outside the
        recordings directory - and then writes its RAW transcript to output."""
        csv = tmp_path / "metadata.csv"
        csv.write_text("call_id,banker_id,file_name\n"
                       "C1,B1,../../../etc/passwd\n"
                       "C2,B1,/etc/hosts\n", encoding="utf-8")
        problems = load_metadata(csv).problems
        assert len(problems) == 2
        assert all(p.column == "file_name" for p in problems)

    def test_metadata_rejects_an_unsafe_call_id(self, tmp_path: Path) -> None:
        csv = tmp_path / "metadata.csv"
        csv.write_text("call_id,banker_id,file_name\n../x,B1,a.wav\n", encoding="utf-8")
        assert any(p.column == "call_id" for p in load_metadata(csv).problems)

    def test_a_ragged_row_is_reported_not_crashed(self, tmp_path: Path) -> None:
        csv = tmp_path / "metadata.csv"
        csv.write_text("call_id,banker_id,file_name\nC1,B1,a.wav,surplus\n", encoding="utf-8")
        problems = load_metadata(csv).problems
        assert problems and "more fields" in problems[0].problem

    def test_a_hebrew_windows_csv_is_readable(self, tmp_path: Path) -> None:
        """Excel on a Hebrew machine writes cp1255, and the loader died on it
        before it could report anything at all."""
        csv = tmp_path / "metadata.csv"
        csv.write_bytes("call_id,banker_id,file_name,banker_name\n"
                        "C1,B1,a.wav,רון לוי\n".encode("cp1255"))
        result = load_metadata(csv)
        assert result.rows["C1"]["banker_name"] == "רון לוי"


# --------------------------------------------------------- audio and channels

class TestChannelDetection:
    def _write(self, path: Path, left: np.ndarray, right: np.ndarray) -> Path:
        pcm = (np.clip(np.stack([left, right], axis=1), -1, 1) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(pcm.tobytes())
        return path

    def _voice(self, freq: float = 300.0, seconds: float = 4.0) -> np.ndarray:
        t = np.arange(int(RATE * seconds)) / RATE
        rng = np.random.default_rng(3)
        return (0.3 * np.sin(2 * np.pi * freq * t)
                + 0.02 * rng.normal(size=len(t))).astype(np.float32)

    def test_a_gain_trimmed_copy_is_still_one_recording(self, tmp_path: Path) -> None:
        """A bit-exact test only catches the easiest case; re-encoding, a gain
        trim or a DC offset all leave a residual that defeats it."""
        voice = self._voice()
        assert probe_channels(self._write(tmp_path / "g.wav", voice, voice * 0.98)) == "dual_mono"

    def test_a_phase_inverted_copy_is_still_one_recording(self, tmp_path: Path) -> None:
        voice = self._voice()
        assert probe_channels(self._write(tmp_path / "p.wav", voice, -voice)) == "dual_mono"

    def test_a_dc_offset_is_not_a_second_speaker(self, tmp_path: Path) -> None:
        voice = self._voice()
        assert probe_channels(self._write(tmp_path / "d.wav", voice, voice + 0.01)) == "dual_mono"

    def test_two_real_speakers_are_still_stereo(self, tmp_path: Path) -> None:
        assert probe_channels(
            self._write(tmp_path / "s.wav", self._voice(220), self._voice(700))) == "stereo"

    def test_an_undecodable_file_does_not_fail_open(self, tmp_path: Path) -> None:
        """Taking the channel split on a file nothing could read attributes the
        call by guesswork; the diarization path at least says that it inferred."""
        broken = tmp_path / "broken.wav"
        broken.write_bytes(b"RIFF....WAVEfmt not really audio")
        assert probe_channels(broken) == "dual_mono"


class TestVoiceActivity:
    def _mono(self, path: Path, samples: np.ndarray) -> Path:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
        return path

    def test_room_tone_is_not_ten_seconds_of_speech(self, tmp_path: Path) -> None:
        """A purely relative threshold scales itself to whatever is there, so
        a recording of an empty room reported speech throughout."""
        rng = np.random.default_rng(1)
        quiet = (rng.normal(0, 0.0005, RATE * 10)).astype(np.float32)
        assert energy_vad(self._mono(tmp_path / "quiet.wav", quiet)) == []

    def test_a_steady_tone_is_not_speech(self, tmp_path: Path) -> None:
        """Hold music and a stuck line have no dynamic range; speech does."""
        t = np.arange(RATE * 10) / RATE
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        assert energy_vad(self._mono(tmp_path / "tone.wav", tone)) == []

    def test_real_speech_is_still_found(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2)
        signal = np.zeros(RATE * 10, dtype=np.float32)
        for start in (1, 4, 7):
            chunk = slice(start * RATE, (start + 2) * RATE)
            t = np.arange(2 * RATE) / RATE
            signal[chunk] = (0.3 * np.sin(2 * np.pi * 180 * t)
                             + 0.02 * rng.normal(size=2 * RATE)).astype(np.float32)
        assert len(energy_vad(self._mono(tmp_path / "speech.wav", signal))) == 3


# ------------------------------------------------------------ judging a call

class TestJudgeOutput:
    def _redacted(self) -> RedactedTranscript:
        return RedactedTranscript(
            call_id="J1", engine="regex",
            turns=[
                RedactedTurn(speaker="banker", start=0, end=5,
                             text="שלום, הגעת לבנק, במה אפשר לעזור?"),
                RedactedTurn(speaker="customer", start=5, end=10,
                             text="אני לא מסכים עם החיוב הזה בכלל."),
            ],
        )

    def test_a_boolean_is_not_a_score(self) -> None:
        """bool is a subclass of int, so `"score": true` was coerced to 1 - a
        judgement of the worst possible, produced by a type error."""
        with pytest.raises(ValueError, match="boolean"):
            DimensionScore(score=True, reasoning_he="x")

    def test_a_quote_attributed_to_the_wrong_speaker_is_rejected(self) -> None:
        """The customer's complaint published as the banker's words reads as
        evidence, which makes it worse than an invented quote."""
        from callqa.judge.validation import verify_evidence
        from callqa.models import Evidence, JudgeResponse

        response = JudgeResponse(scores={"empathy": DimensionScore(
            score=5, reasoning_he="נימוק",
            evidence=[Evidence(quote="אני לא מסכים עם החיוב הזה בכלל.",
                               timestamp="00:05", speaker="banker")])})
        problems = verify_evidence(response, self._redacted())
        assert any("attributed to banker" in p for p in problems)

    def test_a_quote_spliced_across_two_speakers_is_rejected(self) -> None:
        from callqa.judge.validation import verify_evidence
        from callqa.models import Evidence, JudgeResponse

        spliced = "במה אפשר לעזור? אני לא מסכים"
        response = JudgeResponse(scores={"empathy": DimensionScore(
            score=5, reasoning_he="נימוק",
            evidence=[Evidence(quote=spliced, timestamp="00:05", speaker="banker")])})
        assert verify_evidence(response, self._redacted())

    def test_a_dimension_with_no_evidence_is_rejected(self) -> None:
        from callqa.judge.validation import verify_evidence
        from callqa.models import JudgeResponse

        response = JudgeResponse(scores={
            "empathy": DimensionScore(score=5, reasoning_he="נימוק", evidence=[])})
        assert any("no evidence" in p for p in verify_evidence(response, self._redacted()))

    def test_an_impossible_timestamp_is_rejected(self) -> None:
        from callqa.judge.validation import verify_evidence
        from callqa.models import Evidence, JudgeResponse

        response = JudgeResponse(scores={"empathy": DimensionScore(
            score=5, reasoning_he="נימוק",
            evidence=[Evidence(quote="במה אפשר לעזור?", timestamp="99:99",
                               speaker="banker")])})
        assert verify_evidence(response, self._redacted())

    def test_a_long_call_keeps_its_ending(self) -> None:
        """Cutting at a character budget kept only the opening, so the
        dimensions about how a call ENDS were scored on a transcript that
        stopped an hour earlier."""
        from callqa.judge.prompts import ELISION_MARKER, format_transcript

        turns = [RedactedTurn(speaker="banker" if i % 2 == 0 else "customer",
                              start=i * 5.0, end=i * 5.0 + 4.0,
                              text=f"משפט מספר {i} " + "מילה " * 8)
                 for i in range(400)]
        text = format_transcript(
            RedactedTranscript(call_id="L", engine="regex", turns=turns), max_chars=3000)
        assert "משפט מספר 0 " in text
        assert "משפט מספר 399 " in text
        assert ELISION_MARKER in text


class TestRubricIntegrity:
    def test_a_duplicate_dimension_id_is_refused(self, tmp_path: Path) -> None:
        """by_id silently kept the last one, so a duplicated id doubled one
        weight and deleted the other dimension - including a gate."""
        import yaml

        from callqa.rubric import load_rubric

        rubric = yaml.safe_load((Path("config/rubric.yaml")).read_text(encoding="utf-8"))
        rubric["dimensions"][1]["id"] = rubric["dimensions"][0]["id"]
        path = tmp_path / "rubric.yaml"
        path.write_text(yaml.safe_dump(rubric, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate dimension"):
            load_rubric(path)

    def test_a_rubric_with_no_gate_is_refused(self, tmp_path: Path) -> None:
        import yaml

        from callqa.rubric import load_rubric

        rubric = yaml.safe_load((Path("config/rubric.yaml")).read_text(encoding="utf-8"))
        for dim in rubric["dimensions"]:
            dim["gate"] = False
        path = tmp_path / "rubric.yaml"
        path.write_text(yaml.safe_dump(rubric, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="no gate dimension"):
            load_rubric(path)


# ------------------------------------------------------------- what gets out

class TestAggregatesOnlyPublishWhatHeld:
    def test_a_call_held_for_review_is_not_averaged_into_a_bankers_score(
        self, tmp_path: Path
    ) -> None:
        from callqa.aggregation import load_scorecards

        (tmp_path / "scores").mkdir()
        (tmp_path / "results").mkdir()
        for call_id, total, status in (("OK1", 90.0, "success"),
                                       ("BAD", 0.0, "needs_human_review")):
            (tmp_path / "scores" / f"{call_id}.json").write_text(json.dumps({
                "call_id": call_id, "banker_id": "B1", "scores": {},
                "weighted_total": total, "judge_engine": "mock", "model": "m",
                "prompt_sha256": "x", "prompt_version": "1",
                "timestamp": "2026-01-01T00:00:00Z",
            }), encoding="utf-8")
            (tmp_path / "results" / f"{call_id}.json").write_text(
                json.dumps({"call_id": call_id, "status": status}), encoding="utf-8")

        published = load_scorecards(tmp_path)
        assert [c.call_id for c in published] == ["OK1"]
        assert len(load_scorecards(tmp_path, include_unpublished=True)) == 2


class TestEndpointsAreConstrained:
    @pytest.mark.parametrize("url", [
        "http://evil.example/v1",
        "ftp://somewhere/v1",
        "file:///etc/passwd",
    ])
    def test_an_endpoint_that_would_export_call_data_is_refused(self, url: str) -> None:
        from callqa.config import JudgeConfig

        with pytest.raises(ValueError):
            JudgeConfig(model="m", base_url=url)

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
        "https://pod-8000.proxy.example.net/v1",
    ])
    def test_a_local_or_encrypted_endpoint_is_allowed(self, url: str) -> None:
        from callqa.config import JudgeConfig

        assert JudgeConfig(model="m", base_url=url).base_url == url

    def test_a_mistyped_config_key_is_not_silently_ignored(self) -> None:
        from callqa.config import AudioConfig

        with pytest.raises(ValueError):
            AudioConfig(target_sample_rat=16000)
