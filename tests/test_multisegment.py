"""A call recorded in several files: assembly, metadata, the pipeline."""

from __future__ import annotations

import json
import wave
import zipfile

import numpy as np
import pytest

from callqa.journey import g711, nmf
from callqa.journey.assemble import AssembleError, assemble_call
from callqa.journey.importers.common import AudioSource
from callqa.journey.models import CallAudio, Segment
from callqa.models import TranscriptSegment
from tests.test_nmf import chunked, tone

RATE = 8000


def stereo_part(seconds: float, freq_a: int = 300, freq_b: int = 500) -> bytes:
    return nmf.write_nmf({0: chunked(tone(freq_a, seconds), 10.0),
                          1: chunked(tone(freq_b, seconds, 6000), 10.0)})


def mono_part(seconds: float) -> bytes:
    return nmf.write_nmf({0: chunked(tone(300, seconds), 10.0)})


def _source(tmp_path, parts: dict[str, bytes], as_zip: bool = True) -> AudioSource:
    if as_zip:
        path = tmp_path / "rec.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in parts.items():
                zf.writestr(f"x/{name}", data)
    else:
        path = tmp_path / "rec"
        path.mkdir()
        for name, data in parts.items():
            (path / name).write_bytes(data)
    return AudioSource.open(path)


def _call(names: list[str]) -> CallAudio:
    return CallAudio(call_key="abc", call_id="abc",
                     segments=[Segment(seq=i + 1, file_name=n) for i, n in enumerate(names)])


def test_three_stereo_parts_make_one_stereo_call(tmp_path):
    names = ["1_1_1.NMF", "1_1_2.NMF", "1_1_3.NMF"]
    src = _source(tmp_path, {n: stereo_part(s) for n, s in zip(names, (2.0, 1.0, 3.0), strict=True)})
    out = assemble_call(_call(names), src, tmp_path / "out", gap_sec=1.0)
    with wave.open(str(out.wav_path)) as wf:
        assert wf.getnchannels() == 2
        assert wf.getnframes() == int((2 + 1 + 3 + 2) * RATE)
    spans = out.segmap.segments
    assert [(s.start_in_call, s.end_in_call) for s in spans] == [(0.0, 2.0), (3.0, 4.0), (5.0, 8.0)]
    assert out.segmap.layout == "stereo"
    assert out.segmap.segment_at(3.5) == 2
    # the map never carries the recorder's digit-run file names
    assert "1_1_" not in out.map_path.read_text(encoding="utf-8")


def test_mixed_parts_fall_back_to_mono(tmp_path):
    names = ["a.nmf", "b.nmf"]
    src = _source(tmp_path, {"a.nmf": stereo_part(1.0), "b.nmf": mono_part(1.0)}, as_zip=False)
    out = assemble_call(_call(names), src, tmp_path / "out")
    with wave.open(str(out.wav_path)) as wf:
        assert wf.getnchannels() == 1
    assert out.segmap.layout == "mono"


def test_a_missing_part_is_never_skipped(tmp_path):
    src = _source(tmp_path, {"a.nmf": mono_part(1.0)})
    with pytest.raises(AssembleError):
        assemble_call(_call(["a.nmf", "b.nmf"]), src, tmp_path / "out")


def test_wav_parts_assemble_too(tmp_path):
    folder = tmp_path / "wav"
    folder.mkdir()
    for n in ("p1.wav", "p2.wav"):
        with wave.open(str(folder / n), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((np.ones(16000) * 1000).astype(np.int16).tobytes())
    out = assemble_call(_call(["p1.wav", "p2.wav"]), AudioSource.open(folder), tmp_path / "o",
                        gap_sec=0.5)
    with wave.open(str(out.wav_path)) as wf:
        assert wf.getnframes() == int(2.5 * RATE)


# ------------------------------------------------------------------ metadata


def _metadata(tmp_path, body: str):
    from callqa.ingestion import load_metadata
    path = tmp_path / "metadata.csv"
    path.write_text(body, encoding="utf-8")
    return load_metadata(path)


def test_declared_segments_are_one_call(tmp_path):
    v = _metadata(tmp_path, "call_id,banker_id,file_name,segment\n"
                            "C1,B1,c1b.nmf,2\nC1,B1,c1a.nmf,1\nC2,B1,c2.wav,\n")
    assert v.ok, v.problem_table()
    assert v.segments == {"C1": ["c1a.nmf", "c1b.nmf"]}
    assert set(v.rows) == {"C1", "C2"}


def test_an_undeclared_duplicate_is_still_an_error(tmp_path):
    v = _metadata(tmp_path, "call_id,banker_id,file_name\nC1,B1,a.wav\nC1,B1,b.wav\n")
    assert not v.ok
    assert "segment" in v.problem_table()


def test_a_repeated_segment_number_is_an_error(tmp_path):
    v = _metadata(tmp_path, "call_id,banker_id,file_name,segment\nC1,B1,a,1\nC1,B1,b,1\n")
    assert not v.ok


# ------------------------------------------------------------------ roles


def _seg(text: str, start: float) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=start + 2, text=text)


def test_part_conflict_is_flagged():
    from callqa.speakers.stereo import part_role_conflicts
    banker_lines = ["שלום הגעת לבנק, מדבר דני, במה אפשר לעזור?", "אשמח לעזור לך, תודה שפנית"]
    customer_lines = ["שלום, אני רוצה לברר על ההעברה", "תודה רבה"]
    labeled = [(0, _seg(banker_lines[0], 0)), (1, _seg(customer_lines[0], 3)),
               (0, _seg(banker_lines[1], 6)), (1, _seg(customer_lines[1], 9)),
               # part 2: speaker 1 now sounds like the banker - a flip
               (1, _seg(banker_lines[0], 20)), (0, _seg(customer_lines[0], 23)),
               (1, _seg(banker_lines[1], 26)), (0, _seg(customer_lines[1], 29))]
    assert part_role_conflicts(labeled, 0, [(1, 0, 15), (2, 15, 40)], "t") == [2]
    assert part_role_conflicts(labeled, 0, [(1, 0, 40)], "t") == []


def test_stereo_roles_are_swapped_when_the_channels_say_so():
    from callqa.models import DialogTranscript, DialogTurn
    from callqa.speakers.stereo import verify_stereo_roles
    turns = [DialogTurn(speaker="customer", start=0, end=2,
                        text="שלום הגעת לבנק, מדבר דני, במה אפשר לעזור?"),
             DialogTurn(speaker="banker", start=3, end=5, text="היי, אני רוצה לברר על ההלוואה"),
             DialogTurn(speaker="customer", start=6, end=8,
                        text="אשמח לעזור, אפשר מספר תעודת זהות לאימות?"),
             DialogTurn(speaker="banker", start=9, end=10, text="כן בטח")]
    fixed = verify_stereo_roles("t", DialogTranscript(call_id="t", attribution_mode="stereo",
                                                      turns=turns))
    assert fixed.roles_swapped and fixed.attribution_mode == "stereo_inferred"
    assert fixed.turns[0].speaker == "banker"


# ------------------------------------------------------------------ end to end


@pytest.fixture
def seg_workspace(tmp_path, monkeypatch):
    calls = tmp_path / "input" / "calls"
    calls.mkdir(parents=True)
    rng = np.random.default_rng(3)

    def speechy(seconds, freq):
        pcm = tone(freq, seconds, 9000).astype(np.int32)
        env = (np.sin(np.arange(len(pcm)) / RATE * 2 * np.pi * 0.7) > 0).astype(np.int32)
        return (pcm * env + rng.integers(-50, 50, len(pcm))).astype(np.int16)

    for n, sec in ((1, 25.0), (2, 20.0)):
        data = nmf.write_nmf({0: chunked(speechy(sec, 300), 5.0),
                              1: chunked(speechy(sec, 600)[::-1].copy(), 5.0)})
        (calls / f"1_77_{n}.nmf").write_bytes(data)
    (tmp_path / "input" / "metadata.csv").write_text(
        "call_id,banker_id,file_name,segment\nSEG1,B7,1_77_2.nmf,2\nSEG1,B7,1_77_1.nmf,1\n",
        encoding="utf-8")
    monkeypatch.setenv("CALLQA_PATHS__INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CALLQA_PATHS__STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("CALLQA_AUDIO__VAD", "energy")
    return tmp_path


def test_run_processes_a_split_call_as_one(seg_workspace):
    from callqa.cli import main
    code = main(["run", "--mock"])
    out = seg_workspace / "output"
    result = json.loads((out / "results" / "SEG1.json").read_text(encoding="utf-8"))
    assert code in (0, 1), result
    assert result["status"] in ("success", "needs_human_review")
    meta = json.loads((out / "ingestion" / "SEG1.json").read_text(encoding="utf-8"))
    assert meta["duration_sec"] == pytest.approx(46.0, abs=0.1)     # 25 + 1 + 20
    assert meta["channels"] == 2
    assert (out / "audio" / "assembled" / "SEG1.segmap.json").exists()


def test_stop_after_writes_a_partial_result_then_resumes(seg_workspace):
    from callqa.cli import _call_input, assemble_input
    from callqa.config import load_config
    from callqa.engines import build_engines
    from callqa.pipeline import process_call
    config = load_config(None, {"run": {"mock": True}})
    call = _call_input(seg_workspace / "input" / "calls" / "1_77_2.nmf", config)
    call = assemble_input(call, ["1_77_1.nmf", "1_77_2.nmf"],
                          seg_workspace / "input" / "calls", config)
    engines = build_engines(config)
    partial = process_call(call, engines, stop_after="features")
    out = seg_workspace / "output"
    assert partial.stages_completed[-1] == "features"
    assert (out / "results_partial" / "SEG1.json").exists()
    assert not (out / "results" / "SEG1.json").exists()
    full = process_call(call, engines)
    assert full.stages_completed[-1] == "report"
    assert (out / "results" / "SEG1.json").exists()


def test_unknown_stop_stage_is_refused(seg_workspace):
    from callqa.config import load_config
    from callqa.engines import build_engines
    from callqa.models import CallInput
    from callqa.pipeline import process_call
    config = load_config(None, {"run": {"mock": True}})
    res = process_call(CallInput(call_id="X1", audio_path=seg_workspace / "none.wav"),
                       build_engines(config), stop_after="nope")
    assert res.status == "failed" and "unknown stage" in res.error


def test_alaw_helpers_match_the_standard_tables():
    assert g711.decode_alaw(bytes([0xD5]))[0] == 8 and g711.decode_ulaw(bytes([0xFF]))[0] == 0
