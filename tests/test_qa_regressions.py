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


# ------------------------------------------------- the diarizer's two APIs

class TestPyannoteOutputShapes:
    """pyannote 3.x and 4.x return different objects, and the project has to
    read both because which one a machine has depends on when it was built."""

    @staticmethod
    def _segment(start: float, end: float):  # noqa: ANN205
        from collections import namedtuple
        return namedtuple("Segment", "start end")(start, end)

    def test_the_three_x_annotation_is_read(self) -> None:
        from callqa.speakers.pyannote_engine import _segments_from_output

        segment = self._segment

        class Annotation:
            def itertracks(self, yield_label=False):  # noqa: ANN001, ANN202, FBT002
                yield segment(0.0, 5.0), None, "SPEAKER_00"
                yield segment(5.0, 9.0), None, "SPEAKER_01"

        result = _segments_from_output(Annotation())
        assert [(s.label, s.start, s.end) for s in result] == [
            ("SPEAKER_00", 0.0, 5.0), ("SPEAKER_01", 5.0, 9.0)]

    def test_the_four_x_exclusive_view_is_preferred(self) -> None:
        from callqa.speakers.pyannote_engine import _segments_from_output

        segment = self._segment

        class Output:
            speaker_diarization = [(segment(0.0, 5.0), "A"), (segment(4.5, 9.0), "B")]
            exclusive_speaker_diarization = [(segment(0.0, 5.0), "A"), (segment(5.0, 9.0), "B")]

        overlapping = _segments_from_output(Output(), prefer_exclusive=False)
        exclusive = _segments_from_output(Output(), prefer_exclusive=True)
        assert overlapping[1].start == 4.5
        assert exclusive[1].start == 5.0

    def test_a_bare_segment_does_not_crash(self) -> None:
        """pyannote's Segment IS a NamedTuple, so testing isinstance(item,
        tuple) unpacked its own floats and then asked a float for .start."""
        from callqa.speakers.pyannote_engine import _segments_from_output

        segment = self._segment

        class Output:
            exclusive_speaker_diarization = None
            speaker_diarization = [segment(0.0, 5.0), segment(5.0, 9.0)]

        result = _segments_from_output(Output())
        assert [(s.start, s.end) for s in result] == [(0.0, 5.0), (5.0, 9.0)]


# --------------------------------------------------------- the .env round trip

class TestDotenv:
    @pytest.fixture(autouse=True)
    def _restore_environment(self):  # noqa: ANN202
        """These tests apply values to the real process environment, exactly
        as the loader does for the CLI; nothing may outlive the test."""
        import os

        saved = dict(os.environ)
        yield
        os.environ.clear()
        os.environ.update(saved)

    def test_shell_values_win_over_the_file(self, tmp_path: Path, monkeypatch) -> None:
        from callqa.dotenv import load_dotenv

        env = tmp_path / ".env"
        env.write_text("RUNPOD_API_KEY=from-file\nHF_TOKEN=hf-file\n", encoding="utf-8")
        monkeypatch.setenv("RUNPOD_API_KEY", "from-shell")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        applied = load_dotenv(env)
        assert applied == ["HF_TOKEN"]
        import os
        assert os.environ["RUNPOD_API_KEY"] == "from-shell"

    def test_empty_placeholders_are_not_applied(self, tmp_path: Path, monkeypatch) -> None:
        """`.env.example` ships every key blank; a blank must not shadow a
        value the operator set elsewhere or read as 'configured'."""
        from callqa.dotenv import load_dotenv

        env = tmp_path / ".env"
        env.write_text("RUNPOD_API_KEY=\n", encoding="utf-8")
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        assert load_dotenv(env) == []

    def test_endpoints_are_written_without_losing_other_lines(self, tmp_path: Path,
                                                               monkeypatch) -> None:
        """runpod_cli hands the pod's endpoints to the pipeline through .env;
        the operator's own lines and comments must survive that write."""
        from callqa.dotenv import write_env_values

        env = tmp_path / ".env"
        env.write_text("# keep me\nRUNPOD_API_KEY=rp\nCALLQA_ASR__BASE_URL=\n", encoding="utf-8")
        write_env_values({"CALLQA_ASR__BASE_URL": "https://x/asr",
                          "CALLQA_JUDGE__BASE_URL": "https://x/v1"}, env)
        text = env.read_text(encoding="utf-8")
        assert "# keep me" in text and "RUNPOD_API_KEY=rp" in text
        assert "CALLQA_ASR__BASE_URL=https://x/asr" in text
        assert text.count("CALLQA_ASR__BASE_URL=") == 1
        assert oct(env.stat().st_mode & 0o777) == "0o600"

    def test_the_example_file_parses_and_names_every_key_the_code_reads(self) -> None:
        import re

        from callqa.dotenv import parse_env_file

        example = Path(__file__).resolve().parent.parent / ".env.example"
        keys = set(parse_env_file(example.read_text(encoding="utf-8")))
        assert {"RUNPOD_API_KEY", "HF_TOKEN", "CALLQA_JUDGE__API_KEY",
                "CALLQA_ASR__BASE_URL", "CALLQA_DASHBOARD_TOKEN"} <= keys
        # nothing in the example may carry a value: it is a template
        for line in example.read_text(encoding="utf-8").splitlines():
            if re.match(r"^[A-Z_]+=", line):
                assert line.endswith("="), f"template line has a value: {line}"


class TestPresidioConstructionFailure:
    def test_a_presidio_that_imports_but_cannot_construct_degrades_to_regex(self, monkeypatch) -> None:
        """PresidioRedactor() loads a spaCy model (en_core_web_lg) that is not
        on PyPI; on a machine with presidio installed but the model missing,
        construction raises OSError, not ImportError. build_redactor caught
        only ImportError, so a half-installed presidio killed the whole
        engine container instead of falling back to the built-in redactor."""
        from callqa import redaction
        from callqa.config import RedactionConfig

        def boom(self, config):  # noqa: ANN001, ARG001
            raise OSError("[E050] Can't find model 'en_core_web_lg'")

        monkeypatch.setattr(redaction.PresidioRedactor, "__init__", boom)
        redactor = redaction.build_redactor(RedactionConfig(), mock=False)
        assert redactor.name == "regex"


class TestGershayimInJudgePrompt:
    def test_ascii_quote_inside_hebrew_word_is_folded_to_gershayim(self) -> None:
        """Hebrew abbreviations carry an ASCII '"' (חו"ל, ש"ח, ת"ז). The judge
        must quote the transcript verbatim inside JSON strings, and a bare '"'
        ends the string mid-word - observed derailing vLLM's guided decoding
        into an unrecoverable whitespace loop. The prompt now carries U+05F4
        instead; evidence verification strips both, so quotes still match."""
        from callqa.judge.prompts import format_transcript
        from callqa.models import RedactedTranscript, RedactedTurn

        red = RedactedTranscript(call_id="C1", engine="regex", turns=[
            RedactedTurn(speaker="banker", start=0.0, end=2.0,
                         text='חיוב בחו"ל של 100 ש"ח'),
        ])
        out = format_transcript(red)
        assert 'חו״ל' in out and 'ש״ח' in out
        assert 'חו"ל' not in out and 'ש"ח' not in out


class TestEvidenceQuoteSnapping:
    def _redacted(self):
        from callqa.models import RedactedTranscript, RedactedTurn
        return RedactedTranscript(call_id="C1", engine="regex", turns=[
            RedactedTurn(speaker="banker", start=0.0, end=8.0,
                         text="אני מבטל את החסימה ומזמין לך כרטיס חדש עם מספר חדש ליתר ביטחון."),
            RedactedTurn(speaker="customer", start=8.0, end=12.0,
                         text="אוקיי תודה רבה על הטיפול המהיר והאדיב."),
        ])

    def test_near_verbatim_quote_snaps_to_the_real_span(self) -> None:
        """7-14B judges emit NEAR-quotes (a dropped conjunction: 'אני מזמין'
        for 'ומזמין'), and rejecting them made a valid scorecard unreachable.
        The snap replaces the quote with the real transcript span, so the
        stored evidence is still only text that was actually said."""
        from callqa.judge.validation import verify_evidence
        from callqa.models import DimensionScore, Evidence, JudgeResponse

        resp = JudgeResponse(scores={"listening": DimensionScore(
            score=4, reasoning_he="ok", evidence=[Evidence(
                quote="אני מזמין לך כרטיס חדש עם מספר חדש ליתר ביטחון.",
                timestamp="00:05", speaker="banker")])})
        problems = verify_evidence(resp, self._redacted())
        assert problems == []
        assert "ומזמין" in resp.scores["listening"].evidence[0].quote

    def test_invented_quote_still_rejected(self) -> None:
        """Snapping must not weaken the anti-hallucination gate: text far from
        anything in the transcript is still refused."""
        from callqa.judge.validation import verify_evidence
        from callqa.models import DimensionScore, Evidence, JudgeResponse

        resp = JudgeResponse(scores={"listening": DimensionScore(
            score=4, reasoning_he="ok", evidence=[Evidence(
                quote="הבטחתי לך ריבית של עשרים אחוז על הפיקדון הזה",
                timestamp="00:05", speaker="banker")])})
        problems = verify_evidence(resp, self._redacted())
        assert any("not found verbatim" in p for p in problems)


class TestASRSpacedHyphenDigitRuns:
    def test_id_dictated_with_space_hyphen_separators_is_masked(self) -> None:
        """The real ivrit.ai ASR writes dictated numbers as '314 -15992 -6265'
        (a space BEFORE each hyphen). DIGIT_RUN_RE allowed only one separator
        char between digits, so the run split into fragments below the masking
        threshold and a complete national ID left the redaction stage unmasked
        on the first real recording (REAL002)."""
        from callqa.config import RedactionConfig
        from callqa.models import DialogTranscript, DialogTurn
        from callqa.redaction import RegexRedactor

        dialog = DialogTranscript(call_id="C1", attribution_mode="stereo", turns=[
            DialogTurn(speaker="banker", start=0.0, end=4.0,
                       text="תודה, אני חוזר על זה לוודא 314 -15992 -6265. הכל נכון?"),
        ])
        red = RegexRedactor(RedactionConfig())._redact(dialog, [])
        assert "15992" not in red.turns[0].text
        assert "6265" not in red.turns[0].text
        assert "█" in red.turns[0].text

    def test_amounts_with_plain_spaces_are_still_not_joined_into_ids(self) -> None:
        """The guard the one-separator rule was protecting: two amounts
        separated by a space must not merge into one maskable run."""
        from callqa.config import RedactionConfig
        from callqa.models import DialogTranscript, DialogTurn
        from callqa.redaction import RegexRedactor

        dialog = DialogTranscript(call_id="C2", attribution_mode="stereo", turns=[
            DialogTurn(speaker="banker", start=0.0, end=4.0,
                       text="זה עולה 500 300 שקל בסך הכול."),
        ])
        red = RegexRedactor(RedactionConfig())._redact(dialog, [])
        assert red.turns[0].text == "זה עולה 500 300 שקל בסך הכול."


class TestHebrewNumberWordNormaliser:
    """Numbers dictated digit by digit come out of the ASR as Hebrew WORDS,
    and no digit regex can see them. On the first real recording the card's
    last four digits sat in the transcript as 'ארבע חמש שמונה אפס' three times
    over, unmasked. fold_number_words() folds dictation-word runs to digits
    between normalisation and detection, with an index map back so the mask
    lands on the original words."""

    @staticmethod
    def _redact(text: str) -> str:
        from callqa.config import RedactionConfig
        from callqa.models import DialogTranscript, DialogTurn
        from callqa.redaction import RegexRedactor

        dialog = DialogTranscript(call_id="C1", attribution_mode="stereo", turns=[
            DialogTurn(speaker="customer", start=0.0, end=4.0, text=text),
        ])
        return RegexRedactor(RedactionConfig())._redact(dialog, []).turns[0].text

    def test_fold_maps_every_char_back_to_its_source(self) -> None:
        from callqa.redaction import fold_number_words

        text = "הקוד הוא שלוש ארבע חמש שש ותודה"
        folded, spans = fold_number_words(text)
        assert "3456" in folded
        assert len(spans) == len(folded)
        # every non-folded char maps to itself
        for i, ch in enumerate(folded):
            s, e = spans[i]
            if not ch.isdigit():
                assert text[s:e] == ch
        # every folded digit maps to the whole word run
        run = folded.index("3456")
        s, e = spans[run]
        assert text[s:e] == "שלוש ארבע חמש שש"

    def test_card_last_four_dictated_as_words_is_masked(self) -> None:
        red = self._redact("בשביל האימות, מה ארבע הספרות האחרונות של הכרטיס? ארבע חמש שמונה אפס.")
        assert "ארבע חמש שמונה אפס" not in red
        assert "█" in red
        # the question about the digits is speech, not dictation - it stays
        assert "ארבע הספרות האחרונות" in red

    def test_id_dictated_as_words_with_context_is_masked_as_id(self) -> None:
        from callqa.redaction import find_pii

        # 123456782 passes the Israeli ID checksum
        text = 'תעודת זהות: אחת שתיים שלוש ארבע חמש שש שבע שמונה שתיים'
        matches = find_pii(text)
        assert [m.entity_type for m in matches] == ["ISRAELI_ID"]
        start, end = matches[0].start, matches[0].end
        assert text[start:end] == "אחת שתיים שלוש ארבע חמש שש שבע שמונה שתיים"

    def test_conjunction_prefix_is_part_of_the_run(self) -> None:
        red = self._redact("מספר הכרטיס הוא שמונה, אפס, ארבע, וחמש")
        assert "וחמש" not in red

    def test_short_runs_and_quantities_stay_untouched(self) -> None:
        for text in (
            "רגע אחד בבקשה",                       # one digit-word
            "שתי דקות ואני איתך",                   # quantity, not dictation
            "שלוש ארבע פעמים ניסיתי",               # below the 4-word threshold
            "זה עולה שלוש מאות שקל",                # quantity words never fold
        ):
            assert self._redact(text) == text

    def test_dictation_pause_spacing_with_id_context_is_masked(self) -> None:
        """A customer reading an identifier in groups gets bare-space gaps
        ('926 9265') that the structural-separator rule alone rejects; an
        ID/account context word before the run lifts that requirement."""
        red = self._redact("מה מספר תעודת הזהות? 415 926 9265")
        assert "9265" not in red

    def test_repeated_last_four_of_a_masked_number_is_masked(self) -> None:
        """The banker reads back the tail of a number the customer already
        dictated; leaving the fragment in the clear undoes the mask."""
        red = self._redact("מספר הכרטיס 4580-1234-5678 ,כן, המסתיים ב-5678 נכון?")
        assert "5678" not in red

    def test_plain_amounts_near_digit_words_are_not_swallowed(self) -> None:
        red = self._redact("החיוב הוא 30 שקלים ועוד 10 שקלים עמלה")
        assert red == "החיוב הוא 30 שקלים ועוד 10 שקלים עמלה"


class TestParallelDiarization:
    """ASR and diarization are independent on a mono call; overlapping them
    took the perf QA round from ~401 s sequential (172.7 s ASR + 228.5 s
    diarization) to ~347 s of combined wall on the 6-core dev machine. The
    overlap runs in a separate PROCESS because ctranslate2 and torch each
    bundle their own libiomp5 and one process holding both aborts at random
    on Intel macOS."""

    def test_worker_bad_config_fails_cleanly(self, tmp_path) -> None:
        import subprocess
        import sys

        out = tmp_path / "out.json"
        proc = subprocess.run(
            [sys.executable, "-m", "callqa.speakers.diar_worker",
             str(tmp_path / "missing.wav"), "C1", str(out)],
            input=b"{not json", capture_output=True, timeout=120)
        assert proc.returncode == 1
        assert not out.exists()
        assert b"diarization worker failed" in proc.stderr

    @staticmethod
    def _early(cmd: str):
        import subprocess
        import sys

        from callqa.pipeline import _EarlyDiarization
        proc = subprocess.Popen([sys.executable, "-c", cmd],
                                stdin=subprocess.PIPE)
        proc.stdin.close()
        return _EarlyDiarization, proc

    def test_collect_falls_back_when_the_worker_dies(self, tmp_path) -> None:
        cls, proc = self._early("import sys; sys.exit(3)")
        early = cls(proc, tmp_path / "out.json", tmp_path / "log")
        assert early.collect("C1", duration_sec=60.0) is None

    def test_collect_returns_the_worker_segments(self, tmp_path) -> None:
        out = tmp_path / "out.json"
        segs = [{"label": "SPEAKER_00", "start": 0.0, "end": 2.5},
                {"label": "SPEAKER_01", "start": 2.5, "end": 4.0}]
        cls, proc = self._early(
            f"import json,pathlib; pathlib.Path({str(out)!r})."
            f"write_text({json.dumps(json.dumps(segs))})")
        early = cls(proc, out, tmp_path / "log")
        segments = early.collect("C1", duration_sec=60.0)
        assert [s.label for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
        assert segments[1].end == 4.0
        assert not out.exists(), "the worker artifact is cleaned up after use"

    def test_mock_engines_never_spawn_a_worker(self, tmp_path) -> None:
        """Mock diarization is instant; a subprocess would only add noise."""
        from callqa.models import AudioArtifact
        from callqa.pipeline import _EarlyDiarization

        class FakeEngines:
            class config:  # noqa: N801
                class speakers:  # noqa: N801
                    parallel_diarization = True
            mono_diarizer = type("D", (), {"name": "mock"})()

        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFF")
        art = AudioArtifact(call_id="C1", is_stereo=False, mono_wav=str(wav),
                            sample_rate=16000, vad_engine="energy")
        assert _EarlyDiarization.start(FakeEngines(), art, "C1") is None
