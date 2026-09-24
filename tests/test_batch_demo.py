"""The synthetic batch behind the management-report demo.

Every artifact must be one the real pipeline could have written - valid against
its model, evidence copied verbatim from that call's redacted turns, text that
survives the redactor unchanged - or the demo proves nothing about the report.
And it must never land in the operator's real data/input or data/output, where
synthetic scores would be averaged into a real banker's report.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import shlex
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from callqa.aggregation import load_scorecards
from callqa.ingestion import load_metadata
from callqa.judge.prompts import PROMPT_VERSION, mmss
from callqa.judge.validation import MIN_QUOTE_CHARS, normalize_for_match, verify_evidence
from callqa.models import (
    CallMeta,
    CallResult,
    Features,
    JudgeResponse,
    RedactedTranscript,
    ScoreCard,
)
from callqa.ops.models import RunManifest
from callqa.pipeline import STAGES
from callqa.redaction import ENTITY_LABELS_HE, MASK, redact_text
from callqa.rubric import load_rubric, weighted_total

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_batch_demo", REPO_ROOT / "scripts" / "generate_batch_demo.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules while the class is built.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_script()

MARKUP_RE = re.compile(r"[\[\]{}$«»״]")
TOKEN_RE = re.compile(r"<([^<>:]{1,20}):" + MASK + ">")
FORBIDDEN_WORD = "אצלכם"


def _generate(workspace: Path, *extra: str) -> int:
    return demo.main(["--workspace", str(workspace), "--no-report", *extra])


def _ranks(values: Sequence[float]) -> np.ndarray:
    """1-based ranks, ties sharing their average rank (as Spearman requires).
    argsort(argsort(x)) breaks ties by position instead, which on 1-5 scores -
    nearly all ties - measures the file order as much as the relationship."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ordered = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def test_spearman_helper_against_hand_values() -> None:
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # x ranks (1, 2.5, 2.5, 4) against (1, 2, 3, 4): 4.5 / sqrt(4.5 * 5).
    assert _spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / (4.5 * 5) ** 0.5)


def _models(out: Path, folder: str, model: type[BaseModel]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for path in sorted((out / folder).glob("*.json")):
        item = model.model_validate_json(path.read_text(encoding="utf-8"))
        loaded[path.stem] = item
    return loaded


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> dict:
    ws = tmp_path_factory.mktemp("demo") / "batch"
    assert _generate(ws, "--calls", "120") == 0
    out = ws / "output"
    return {
        "ws": ws, "out": out,
        "results": _models(out, "results", CallResult),
        "meta": _models(out, "ingestion", CallMeta),
        "features": _models(out, "features", Features),
        "redacted": _models(out, "redacted", RedactedTranscript),
        "cards": _models(out, "scores", ScoreCard),
    }


@pytest.fixture(scope="module")
def big_batch(tmp_path_factory: pytest.TempPathFactory) -> dict:
    ws = tmp_path_factory.mktemp("demo-big") / "batch"
    started = time.perf_counter()
    assert _generate(ws, "--calls", "1000") == 0
    seconds = time.perf_counter() - started
    out = ws / "output"
    return {"seconds": seconds, "out": out,
            "results": _models(out, "results", CallResult),
            "meta": _models(out, "ingestion", CallMeta),
            "features": _models(out, "features", Features),
            "cards": _models(out, "scores", ScoreCard)}


# -- shape -------------------------------------------------------------------

def test_counts_and_statuses_follow_the_pipeline(batch: dict) -> None:
    results, cards = batch["results"], batch["cards"]
    assert len(results) == 120
    assert set(batch["meta"]) == set(results), "every call was ingested"
    by_status = Counter(r.status for r in results.values())
    assert by_status["failed"] == 1 and by_status["needs_human_review"] == 4
    assert by_status["success"] == 115

    held = [cid for cid, r in results.items() if r.status == "needs_human_review"]
    assert sum(cid in cards for cid in held) == 2, "half the held calls carry a scorecard"
    for cid, result in results.items():
        assert result.call_id == cid and result.report_path is None
        stages = result.stages_completed
        assert stages == STAGES[:len(stages)], "stages completed are a prefix of the pipeline"
        # An artifact exists exactly when the stage that writes it completed.
        assert (cid in batch["redacted"]) == ("redaction" in stages)
        assert (cid in batch["features"]) == ("features" in stages)
        if result.status == "success":
            assert stages == STAGES and cid in cards and result.error is None
        elif result.status == "failed":
            assert result.error == "RuntimeError: synthetic failure"
            assert len(stages) < len(STAGES) and cid not in cards
        else:
            assert result.error, "a held call says why"
            if cid in cards:
                assert stages == STAGES
                assert result.error.startswith("speaker roles inferred with low confidence")
                assert batch["meta"][cid].channels == 1
            else:
                assert "judge" not in stages
                assert result.error.startswith("judge output failed after")


def test_every_artifact_validates_and_agrees(batch: dict) -> None:
    rubric = load_rubric()
    for cid, meta in batch["meta"].items():
        assert meta.call_id == cid and meta.file_name == f"{cid}.wav"
        assert meta.call_type in {"service", "loans", "cards", "mortgage", "investments"}
        assert meta.channels in (1, 2) and meta.sample_rate == 16000
        assert meta.banker_channel == ("L" if meta.channels == 2 else None)
        assert 60 <= meta.duration_sec <= 1800
        assert re.fullmatch(r"DEMO\d{4}", cid) and re.fullmatch(r"B\d{3}", meta.banker_id)
    for cid, card in batch["cards"].items():
        meta = batch["meta"][cid]
        assert card.banker_id == meta.banker_id
        assert card.judge_engine == card.model == "synthetic-demo"
        assert card.prompt_version == PROMPT_VERSION
        assert card.rubric_sha256 == rubric.sha256
        assert re.fullmatch(r"[0-9a-f]{64}", card.prompt_sha256)
        assert list(card.scores) == [d.id for d in rubric.dimensions]
        total, gate_failed, failed = weighted_total(rubric, card.scores)
        assert (card.weighted_total, card.gate_failed, card.failed_gates) == (
            total, gate_failed, failed)
        # And by hand, from the spec's formula rather than the function under use:
        # sum(w * (s - 1) / 4 * 100), capped at 59 when a gate dimension is <= 2.
        raw = sum(d.weight * (card.scores[d.id].score - 1) / 4 * 100 for d in rubric.dimensions)
        gates = [d.id for d in rubric.dimensions if d.gate and card.scores[d.id].score <= 2]
        assert card.failed_gates == gates and card.gate_failed == bool(gates)
        assert card.weighted_total == pytest.approx(min(raw, 59.0) if gates else raw, abs=0.05)
        scored_at = datetime.fromisoformat(card.timestamp)
        assert scored_at.utcoffset() == timedelta(0)
        lag = scored_at.date() - date.fromisoformat(meta.call_date)
        assert timedelta(days=1) <= lag <= timedelta(days=6), "scored a few days after the call"
        assert 1 <= len(card.strengths_he) <= 2 and card.development_area_he and card.summary_he
    runs = _models(batch["out"], "runs", RunManifest)
    assert runs, "at least one run manifest"
    in_runs = [c.call_id for run in runs.values() for c in run.calls]
    assert sorted(in_runs) == sorted(batch["results"]), "every call is in exactly one run"
    for run_id, run in runs.items():
        assert run.run_id == run_id
        assert sum(run.counts.values()) == len(run.calls)


def test_metadata_csv_is_valid_and_names_nobody(batch: dict) -> None:
    path = batch["ws"] / "input" / "metadata.csv"
    assert path.read_bytes().startswith(b"\xef\xbb\xbf"), "utf-8-sig, so Excel reads it"
    validation = load_metadata(path)
    assert validation.ok, validation.problems
    assert set(validation.rows) == set(batch["results"])
    with path.open(encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert "banker_name" not in header
    for cid, row in validation.rows.items():
        meta = batch["meta"][cid]
        assert (row["banker_id"], row["call_date"], row["call_type"]) == (
            meta.banker_id, meta.call_date, meta.call_type)


def test_artifacts_are_lf_utf8_on_every_os(batch: dict) -> None:
    sample = next((batch["out"] / "redacted").glob("*.json")).read_bytes()
    assert b"\r\n" not in sample
    assert "בנק".encode() in sample or "ש".encode() in sample, "Hebrew stored as UTF-8"


# -- text --------------------------------------------------------------------

def test_evidence_is_verbatim_from_that_calls_turns(batch: dict) -> None:
    for cid, card in batch["cards"].items():
        redacted = batch["redacted"][cid]
        response = JudgeResponse(scores=card.model_copy(deep=True).scores)
        assert verify_evidence(response, redacted) == [], cid
        for dim, score in card.scores.items():
            assert len(score.evidence) == 1
            ev = score.evidence[0]
            holders = [t for t in redacted.turns if ev.quote in t.text]
            assert holders, f"{cid}/{dim}: quote not in any turn"
            turn = holders[0]
            assert (turn.speaker, mmss(turn.start)) == (ev.speaker, ev.timestamp)
            assert len(normalize_for_match(ev.quote)) >= MIN_QUOTE_CHARS
            # The real judge runner strips tags from quotes, so a quote holding a
            # mask token would not survive it; and the redactor must not change it.
            assert "<" not in ev.quote and redact_text(ev.quote)[0] == ev.quote


def test_transcripts_are_redacted_and_match_the_features(batch: dict) -> None:
    label_to_entities: dict[str, set[str]] = {}
    for entity, label in ENTITY_LABELS_HE.items():
        label_to_entities.setdefault(label, set()).add(entity)
    for cid, redacted in batch["redacted"].items():
        meta = batch["meta"][cid]
        assert redacted.enabled is True and redacted.engine == "synthetic-demo"
        turns = redacted.turns
        assert 10 <= len(turns) <= 24
        assert turns[0].speaker == "banker"
        assert all(a.speaker != b.speaker for a, b in zip(turns, turns[1:], strict=False))
        assert all(t.start < t.end for t in turns)
        assert all(a.end <= b.start for a, b in zip(turns, turns[1:], strict=False))
        assert turns[-1].end <= meta.duration_sec
        tokens: Counter[str] = Counter()
        for turn in turns:
            assert not MARKUP_RE.search(turn.text), f"{cid}: template markup left: {turn.text}"
            assert FORBIDDEN_WORD not in turn.text
            tokens.update(TOKEN_RE.findall(turn.text))
        assert set(tokens) <= set(label_to_entities), "only the redactor's own labels"
        counted: Counter[str] = Counter()
        for entity, n in redacted.redaction_counts.items():
            counted[ENTITY_LABELS_HE[entity]] += n
        assert counted == tokens, f"{cid}: redaction_counts disagree with the tokens"
    # Nothing the redactor would still mask: run it over a sample of whole turns.
    for redacted in list(batch["redacted"].values())[:25]:
        for turn in redacted.turns:
            assert redact_text(turn.text)[0] == turn.text, turn.text
    for cid, features in batch["features"].items():
        meta, turns = batch["meta"][cid], batch["redacted"][cid].turns
        assert features.call_duration_sec == meta.duration_sec
        banker = sum(t.end - t.start for t in turns if t.speaker == "banker")
        customer = sum(t.end - t.start for t in turns if t.speaker == "customer")
        assert features.talk_ratio == pytest.approx(banker / (banker + customer), abs=0.002)
        assert 0.35 <= features.talk_ratio <= 0.8
        gaps = [b.start - a.end for a, b in zip(turns, turns[1:], strict=False)]
        assert features.dead_air_total_sec == pytest.approx(
            sum(g for g in gaps if g > 3.0), abs=0.2)
        if meta.channels == 1:
            assert not features.overlap_metrics_available
            assert features.interruptions_by_banker == features.interruptions_by_customer == 0


def test_judge_prose_is_clean_hebrew(batch: dict) -> None:
    hebrew = re.compile(r"[א-ת]")
    for card in batch["cards"].values():
        prose = [s.reasoning_he for s in card.scores.values()]
        prose += [*card.strengths_he, card.development_area_he, card.summary_he]
        for text in prose:
            assert hebrew.search(text)
            assert not MARKUP_RE.search(text) and "<" not in text, text
            assert FORBIDDEN_WORD not in text
            # The report re-scrubs all of it at render; none may change.
            assert redact_text(text)[0] == text, text


def test_listening_prose_agrees_with_the_measured_interruptions(batch: dict,
                                                                  big_batch: dict) -> None:
    """The listening reasoning quotes the interruption count beside what the
    judge saw; "interrupted the customer ... with no interruptions" is a
    contradiction a manager reads in the drill-down."""
    checked = Counter()
    for data in (batch, big_batch):
        for cid, card in data["cards"].items():
            features = data["features"][cid]
            listening = card.scores["listening"]
            if not features.overlap_metrics_available:
                # One channel: the count is a structural zero, so no claim either way.
                assert "לא ניתן למדוד קטיעות" in listening.reasoning_he
                assert "ללא קטיעות" not in listening.reasoning_he
                continue
            n = features.interruptions_by_banker
            if listening.score <= 2:
                checked["low"] += 1
                assert n >= 3 - listening.score, (cid, listening.score, n)
                assert "ללא קטיעות" not in listening.reasoning_he, listening.reasoning_he
            elif listening.score == 5:
                checked["high"] += 1
                assert n <= 1, (cid, n)
    assert checked["low"] >= 20 and checked["high"] >= 20, checked


def test_reasoning_matches_the_level(batch: dict) -> None:
    """A score-1 identification shows none; a score-5 compliance discloses."""
    seen = set()
    for cid, card in batch["cards"].items():
        text = " ".join(t.text for t in batch["redacted"][cid].turns)
        ident = card.scores["identification"]
        if ident.score == 1:
            seen.add("id1")
            assert '<ת"ז:' not in text and "תאריך לידה" not in text
            assert "כשל בשער חובה" in ident.reasoning_he
        if ident.score == 5:
            assert '<ת"ז:' in text and "<תאריך לידה:" in text
        comp = card.scores["compliance"]
        if comp.score == 5:
            seen.add("c5")
            quote = comp.evidence[0].quote
            assert re.search(r"\d", quote) or "סיכון" in quote, quote
        if comp.score <= 2:
            assert "שער חובה" in comp.reasoning_he and card.gate_failed
    assert seen == {"id1", "c5"}


def test_phrase_library_renders_at_every_level() -> None:
    """Every scenario at every level, for every gender pairing: no template
    markup survives, each dimension has exactly one quotable span, and the
    call fits the turn limit even with the most fillers."""
    for scenario in demo.SCENARIOS:
        for level in range(1, 6):
            levels = dict.fromkeys(demo.KNOWN_DIMENSIONS, level)
            for banker_f in (False, True):
                for customer_f in (False, True):
                    for seed in range(3):
                        rng = np.random.default_rng([seed, level])
                        slots = demo.build_slots(rng, scenario, banker_female=banker_f,
                                                 customer_female=customer_f)
                        dialog, id_reasoning = demo.compose_dialog(
                            rng, scenario, levels, slots, banker_female=banker_f,
                            customer_female=customer_f, n_fillers=2)
                        assert 18 <= len(dialog.turns) <= 24
                        quoted = Counter(d for t in dialog.turns for d in t.quotes)
                        assert quoted == Counter(demo.KNOWN_DIMENSIONS)
                        for turn in dialog.turns:
                            assert not MARKUP_RE.search(turn.text), (scenario.key, turn.text)
                            assert FORBIDDEN_WORD not in turn.text
                            for quote in turn.quotes.values():
                                assert quote in turn.text and "<" not in quote
                                assert len(normalize_for_match(quote)) >= MIN_QUOTE_CHARS
                        rendered = demo._render(id_reasoning, {}, banker_f, customer_f)[0]
                        assert not MARKUP_RE.search(rendered)


def test_judge_prose_library_renders_for_every_level_and_gender() -> None:
    slots = {"disclosure": "העמלות", "probe": "הצורך"}
    templates = [t for levels in demo.REASONING.values() for ts in levels.values() for t in ts]
    templates += [t for by_level in demo.STRENGTH.values() for t in by_level.values()]
    templates += list(demo.DEVELOPMENT.values())
    for template in templates:
        for banker_f in (False, True):
            for customer_f in (False, True):
                text = demo._render(template, slots, banker_f, customer_f)[0]
                assert not MARKUP_RE.search(text), text
                assert FORBIDDEN_WORD not in text


# -- the cohort the report reads ---------------------------------------------

def test_load_scorecards_keeps_every_success_call_in_one_cohort(batch: dict) -> None:
    success = {cid for cid, r in batch["results"].items() if r.status == "success"}
    cohort = load_scorecards(batch["out"])
    assert {c.call_id for c in cohort} == success
    assert len({(c.rubric_sha256, c.prompt_version) for c in cohort}) == 1


def test_gate_failure_rate_is_sane(batch: dict) -> None:
    cohort = load_scorecards(batch["out"])
    rate = sum(c.gate_failed for c in cohort) / len(cohort)
    assert 0.02 <= rate <= 0.18


def test_same_seed_same_batch(tmp_path: Path) -> None:
    for name in ("a", "b"):
        assert _generate(tmp_path / name, "--calls", "60", "--seed", "11") == 0
    assert _generate(tmp_path / "c", "--calls", "60", "--seed", "12") == 0

    def snapshot(name: str) -> dict[str, bytes]:
        ws = (tmp_path / name).resolve()
        out = ws / "output"
        snap: dict[str, bytes] = {}
        for p in sorted(out.rglob("*.json")):
            data = p.read_bytes()
            if p.parent.name == "results":
                # The one field that names the folder: the scorecard's absolute
                # path. Compared relative to the workspace, not skipped - the
                # statuses, errors and stages must be deterministic too.
                result = json.loads(data)
                if result["scorecard_path"]:
                    result["scorecard_path"] = (
                        Path(result["scorecard_path"]).relative_to(ws).as_posix())
                data = json.dumps(result, sort_keys=True).encode()
            snap[p.relative_to(out).as_posix()] = data
        snap["input/metadata.csv"] = (ws / "input" / "metadata.csv").read_bytes()
        return snap

    a, b = snapshot("a"), snapshot("b")
    assert any(k.startswith("results/") for k in a) and "input/metadata.csv" in a
    assert a == b, "same seed, byte-identical artifacts"
    c = snapshot("c")
    assert {k: v for k, v in a.items() if k.startswith("scores/")} != {
        k: v for k, v in c.items() if k.startswith("scores/")}


# -- the workspace -----------------------------------------------------------

def _tree(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_never_touches_the_checkouts_real_data(tmp_path: Path) -> None:
    real = [REPO_ROOT / "data" / "input", REPO_ROOT / "data" / "output"]
    before = [_tree(p) for p in real]
    assert _generate(tmp_path / "ws", "--calls", "20") == 0
    # Pointed at the real data - directly, at its parent, or inside it - it refuses.
    for target in (REPO_ROOT / "data", REPO_ROOT / "data" / "output",
                   REPO_ROOT / "data" / "input" / "demo", REPO_ROOT):
        assert _generate(target, "--calls", "5") == 2, target
    assert not (REPO_ROOT / "data" / "input" / "demo").exists()
    assert [_tree(p) for p in real] == before


def test_default_workspace_is_its_own_folder() -> None:
    assert demo.DEFAULT_WORKSPACE == REPO_ROOT / "data" / "demo-batch"


def test_refuses_to_clear_a_folder_it_did_not_create(tmp_path: Path) -> None:
    foreign = tmp_path / "somebody-elses"
    (foreign / "output").mkdir(parents=True)
    keep = foreign / "output" / "important.json"
    keep.write_text("{}", encoding="utf-8")
    assert _generate(foreign, "--calls", "5") == 2
    assert keep.is_file()


def test_regeneration_clears_the_previous_batch(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    assert _generate(ws, "--calls", "40") == 0
    (ws / "output" / "reports").mkdir()
    (ws / "output" / "reports" / "executive.html").write_text("stale", encoding="utf-8")
    assert _generate(ws, "--calls", "10", "--weeks", "2") == 0
    out = ws / "output"
    assert len(list((out / "results").glob("*.json"))) == 10
    assert len(list((out / "scores").glob("*.json"))) <= 10
    assert not (out / "reports").exists(), "the stale report went with its batch"
    runs = _models(out, "runs", RunManifest)
    assert sum(len(r.calls) for r in runs.values()) == 10


def _batch_state(ws: Path) -> dict[str, bytes]:
    return {p.relative_to(ws).as_posix(): p.read_bytes()
            for p in sorted(ws.rglob("*")) if p.is_file()}


def test_bad_arguments_are_refused_before_the_previous_batch_is_cleared(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A negative seed used to raise from inside numpy - after the previous
    batch had been deleted - and calls=0 / bankers=0 / weeks=0 the same."""
    ws = tmp_path / "ws"
    assert _generate(ws, "--calls", "15") == 0
    before = _batch_state(ws)
    for bad in (["--seed", "-1"], ["--seed", "x"], ["--calls", "0"], ["--bankers", "0"],
                ["--weeks", "53"], ["--calls", "20001"]):
        with pytest.raises(SystemExit) as exc:
            _generate(ws, *bad)
        assert exc.value.code == 2, bad
        assert "error: argument" in capsys.readouterr().err
    for kwargs in ({"seed": -1}, {"calls": 0}, {"bankers": 0}, {"weeks": 0}, {"weeks": 53},
                   {"calls": 2.5}, {"calls": True}):
        with pytest.raises(ValueError):
            demo.generate_batch(ws, **kwargs)
    assert _batch_state(ws) == before, "the previous batch is untouched"
    # numpy integers are whole numbers too.
    assert demo.generate_batch(ws, calls=np.int64(12), seed=np.int64(0)).n_calls == 12


def test_a_locked_state_database_is_refused_cleanly(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows will not delete a database another process holds open; that must
    be a clean refusal (exit 2), not a traceback."""
    ws = tmp_path / "ws"
    assert _generate(ws, "--calls", "5") == 0
    (ws / "callqa_state.db").write_bytes(b"")
    real_unlink = Path.unlink

    def locked(self: Path, missing_ok: bool = False) -> None:
        if self.name == "callqa_state.db":
            raise PermissionError(32, "The process cannot access the file", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    before = _batch_state(ws)
    with pytest.raises(demo.WorkspaceError, match="callqa_state.db"):
        demo.prepare_workspace(ws)
    assert _generate(ws, "--calls", "5") == 2
    assert _batch_state(ws) == before, "refused before anything was deleted"


def _rosh_hashana(year: int) -> date:
    """1 Tishrei of Hebrew `year` (Reingold & Dershowitz, Calendrical Calculations)."""
    def elapsed(y: int) -> int:
        months = (235 * y - 234) // 19
        parts = 12084 + 13753 * months
        days = 29 * months + parts // 25920
        return days + 1 if (3 * (days + 1)) % 7 < 3 else days

    delay = (2 if elapsed(year + 1) - elapsed(year) == 356
             else 1 if elapsed(year) - elapsed(year - 1) == 382 else 0)
    return date.fromordinal(-1373427 + elapsed(year) + delay)


def _bank_holidays(year: int) -> set[date]:
    """The Tishrei holidays of `year` and the spring holidays that precede them."""
    rh = _rosh_hashana(year)
    pesach = _rosh_hashana(year + 1) - timedelta(days=163)          # 15 Nisan
    iyar5 = pesach + timedelta(days=20)
    # Independence Day moves off Friday/Saturday to Thursday, and off Monday to Tuesday.
    shift = {4: -1, 5: -2, 0: 1}.get(iyar5.weekday(), 0)
    return {rh, rh + timedelta(days=1), rh + timedelta(days=9), rh + timedelta(days=14),
            rh + timedelta(days=21), pesach, pesach + timedelta(days=6),
            iyar5 + timedelta(days=shift), pesach + timedelta(days=50)}


def test_business_days_skip_weekends_and_bank_holidays() -> None:
    assert _rosh_hashana(5787) == date(2026, 9, 12)          # a known anchor
    first, days = demo.business_days(demo.MAX_WEEKS)
    assert first.weekday() == 6 and days[-1] == demo.END_DATE
    assert all(d.weekday() in (6, 0, 1, 2, 3) for d in days), "Sunday-Thursday only"
    holidays = {d for d in _bank_holidays(5786) | _bank_holidays(5787)
                if first <= d <= demo.END_DATE and d.weekday() in (6, 0, 1, 2, 3)}
    assert date(2025, 10, 2) in holidays, "Yom Kippur 5786 falls inside the window"
    assert holidays == set(demo.HOLIDAYS)
    assert not holidays & set(days)
    week_days = [first + timedelta(days=7 * w + k) for w in range(demo.MAX_WEEKS)
                 for k in range(5)]
    assert days == [d for d in week_days if d not in holidays]


def test_report_command_points_the_cli_at_the_workspace(tmp_path: Path) -> None:
    cmd, env = demo.report_command(tmp_path)
    assert cmd[0] == sys.executable and cmd[1:] == ["-m", "callqa", "executive-report"]
    ws = tmp_path.resolve()
    assert Path(env["CALLQA_PATHS__INPUT_DIR"]) == ws / "input"
    assert Path(env["CALLQA_PATHS__OUTPUT_DIR"]) == ws / "output"
    assert Path(env["CALLQA_PATHS__STATE_DB"]).parent == ws
    assert env["PYTHONUTF8"] == "1"
    assert str(REPO_ROOT / "src") in env["PYTHONPATH"].split(os.pathsep)


def test_the_report_is_built_opened_and_its_dashboard_hint_runs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The default path end to end: the real `callqa executive-report` in a
    subprocess, --open, and a dashboard command that survives a space in the
    path (Windows user folders: C:\\Users\\Dana Levi\\...)."""
    opened: list[str] = []
    monkeypatch.setattr(demo.webbrowser, "open", lambda uri, *a, **k: opened.append(uri))
    ws = tmp_path / "demo batch"
    assert demo.main(["--workspace", str(ws), "--calls", "60", "--open"]) == 0
    reports = ws.resolve() / "output" / "reports"
    for name in ("executive.html", "executive.json", "executive_calls.csv"):
        assert (reports / name).is_file() and (reports / name).stat().st_size > 0, name
    assert opened == [(reports / "executive.html").as_uri()]
    lines = capsys.readouterr().out.splitlines()
    hints = [line for line in lines if line.startswith("In the dashboard: ")]
    if not (REPO_ROOT / "dashboard" / "server.py").is_file():
        # The dashboard is optional and deletable; with it gone there is no hint.
        assert hints == []
        return
    hint = hints[0]
    command = hint.removeprefix("In the dashboard: ")
    expected = [sys.executable, str(REPO_ROOT / "dashboard" / "server.py"), "--output-dir",
                str(ws.resolve() / "output")]
    if os.name == "nt":
        assert command == demo.subprocess.list2cmdline(expected)
    else:
        assert shlex.split(command) == expected


def test_a_failed_report_exits_1_and_keeps_the_batch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(workspace: Path) -> tuple[list[str], dict[str, str]]:
        return [sys.executable, "-c", "raise SystemExit(3)"], dict(os.environ)

    monkeypatch.setattr(demo, "report_command", failing)
    opened: list[str] = []
    monkeypatch.setattr(demo.webbrowser, "open", lambda uri, *a, **k: opened.append(uri))
    ws = tmp_path / "ws"
    assert demo.main(["--workspace", str(ws), "--calls", "10", "--open"]) == 1
    assert len(list((ws / "output" / "results").glob("*.json"))) == 10
    assert opened == [], "nothing to open"


def test_a_linked_output_folder_is_refused_before_anything_is_cleared(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    assert _generate(ws, "--calls", "5") == 0
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.json").write_text("{}", encoding="utf-8")
    real_output = ws / "output"
    real_output.rename(tmp_path / "old-output")
    try:
        real_output.symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("this system does not allow creating symbolic links")
    before = _batch_state(ws / "input")
    assert _generate(ws, "--calls", "5") == 2
    assert (elsewhere / "keep.json").is_file()
    assert _batch_state(ws / "input") == before, "input/ was not cleared either"


# -- the structure the report is meant to find (1000 calls) -----------------

def test_a_thousand_calls_are_quick(big_batch: dict) -> None:
    assert len(big_batch["results"]) == 1000
    # The contract is < 5 s on a laptop; CI machines get headroom.
    assert big_batch["seconds"] < 20, big_batch["seconds"]


def test_a_thousand_calls_have_the_intended_structure(big_batch: dict) -> None:
    meta, cards, features = big_batch["meta"], big_batch["cards"], big_batch["features"]
    cohort = load_scorecards(big_batch["out"])
    assert len(cohort) == sum(r.status == "success" for r in big_batch["results"].values())
    rate = sum(c.gate_failed for c in cohort) / len(cohort)
    assert 0.05 <= rate <= 0.10, rate

    shares = Counter(m.call_type for m in meta.values())
    expected = {"service": 0.45, "loans": 0.20, "cards": 0.15, "mortgage": 0.12,
                "investments": 0.08}
    for ctype, share in expected.items():
        assert abs(shares[ctype] / 1000 - share) < 0.04, (ctype, shares[ctype])
    by_type = {t: [c for c in cohort if meta[c.call_id].call_type == t] for t in expected}
    gate_rate = {t: sum(c.gate_failed for c in cs) / len(cs) for t, cs in by_type.items()}
    assert gate_rate["investments"] > gate_rate["service"]

    days = sorted({date.fromisoformat(m.call_date) for m in meta.values()})
    assert days[-1] <= date(2026, 9, 17)
    assert all(d.weekday() in (6, 0, 1, 2, 3) for d in days), "Sunday-Thursday only"
    durations = sorted(m.duration_sec for m in meta.values())
    assert 240 <= durations[500] <= 420, "median around five and a half minutes"
    assert 0.07 <= sum(m.channels == 1 for m in meta.values()) / 1000 <= 0.13

    per_banker = Counter(c.banker_id for c in cohort)
    assert len({m.banker_id for m in meta.values()}) == 40
    assert any(3 <= n <= 6 for n in per_banker.values()), "a few bankers with a handful of calls"
    means = {b: np.mean([c.weighted_total for c in cohort if c.banker_id == b])
             for b, n in per_banker.items() if n >= 8}
    assert max(means.values()) - min(means.values()) >= 25, "clear stars and strugglers"

    first_sunday = days[0] - timedelta(days=(days[0].weekday() + 1) % 7)

    def week(c: ScoreCard) -> int:
        return (date.fromisoformat(meta[c.call_id].call_date) - first_sunday).days // 7

    def training_gain(dim: str) -> float:
        before = [c.scores[dim].score for c in cohort if week(c) < 7]
        after = [c.scores[dim].score for c in cohort if week(c) >= 7]
        return float(np.mean(after) - np.mean(before))

    assert training_gain("empathy") >= 0.3, "training shows after week 7"
    assert training_gain("listening") >= 0.3, "training shows after week 7"
    assert abs(training_gain("clarity")) < 0.25, "mild otherwise"

    def mean_score(ctype: str, dim: str) -> float:
        return float(np.mean([c.scores[dim].score for c in by_type[ctype]]))

    for hard in ("investments", "mortgage"):
        for dim in ("clarity", "compliance"):
            assert mean_score(hard, dim) < mean_score("service", dim) - 0.2, (hard, dim)

    ids = [c.call_id for c in cohort]

    def rho(feature: Any, dim: str, only: list[str] = ids) -> float:
        return _spearman([feature(features[i]) for i in only],
                         [cards[i].scores[dim].score for i in only])

    stereo = [i for i in ids if features[i].overlap_metrics_available]
    assert rho(lambda f: f.talk_ratio, "listening") < -0.25, "more banker talk, less listening"
    assert rho(lambda f: f.talk_ratio, "empathy") < -0.15
    assert rho(lambda f: f.interruptions_by_banker / f.call_duration_sec, "listening",
               stereo) < -0.2, "interruptions go with poorer listening"
    assert rho(lambda f: f.banker_questions_per_minute, "suitability") > 0.15
    assert rho(lambda f: f.dead_air_total_sec / f.call_duration_sec, "clarity") < -0.25
    assert rho(lambda f: f.call_duration_sec, "closure") < -0.08, "long calls close worse"
    assert rho(lambda f: f.call_duration_sec, "clarity") < -0.08
