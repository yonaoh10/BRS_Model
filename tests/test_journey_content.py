"""The content stage: views, quote verification, task parsing, the runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from callqa.config import load_config
from callqa.journey.content import run_content
from callqa.journey.llm.client import judge_config_for, resolve_profile
from callqa.journey.llm.mock import MockContentEngine
from callqa.journey.llm.tasks import (
    Prompt,
    TaskError,
    parse_card,
    parse_returns,
    parse_story,
)
from callqa.journey.llm.verify import verify_evidence
from callqa.journey.store import dataset_dir, load_content
from callqa.journey.transcript_view import (
    ContentView,
    Line,
    call_view,
    compress,
    load_lexicon,
)
from callqa.journey.vocab import load_taxonomy
from callqa.models import RedactedTranscript, RedactedTurn

REPO_ROOT = Path(__file__).resolve().parent.parent
TAX = load_taxonomy()


def _view(*texts: str, uncertain: tuple[int, ...] = ()) -> ContentView:
    lines = [Line(no=i + 1, who="bank" if i % 2 == 0 else "customer", text=t,
                  uncertain=(i + 1) in uncertain) for i, t in enumerate(texts)]
    return ContentView("i1", "call", lines)


VIEW = _view("שלום, הגעת למרכז הבנקאות, במה אפשר לעזור?",
             "כבר דיברתי איתכם אתמול ואף אחד לא חזר אליי",
             "אני אבדוק ואחזור אלייך עד מחר בצהריים",
             "העברה של 8000 שקל נחסמה ולא אישרו לי אותה",
             uncertain=(1,))


# ---------------------------------------------------------------- verification


def test_exact_quote_is_accepted_with_the_lines_own_words():
    ev, why = verify_evidence(VIEW, 3, "אבדוק ואחזור אלייך")
    assert ev is not None and not why
    assert ev.line == 3 and ev.speaker == "bank" and "אבדוק ואחזור אלייך" in ev.quote


def test_quote_on_the_wrong_line_is_refused():
    ev, why = verify_evidence(VIEW, 2, "אבדוק ואחזור אלייך")
    assert ev is None and "L2" in why


def test_uncertain_line_is_never_quoted():
    ev, why = verify_evidence(VIEW, 1, "הגעת למרכז הבנקאות")
    assert ev is None and "⚠" in why


def test_a_changed_number_or_negation_is_refused():
    assert verify_evidence(VIEW, 4, "העברה של 9000 שקל נחסמה")[0] is None
    # a quote cut just after a negation keeps it: the stored words say what the line says
    ev, _ = verify_evidence(VIEW, 4, "אישרו לי אותה")
    assert ev is not None and ev.quote == "ולא אישרו לי אותה"
    ev, _ = verify_evidence(VIEW, 4, "ואישרו לי אותה")
    assert ev is None or ev.quote.startswith("ולא")


def test_missing_line_and_bad_types():
    assert verify_evidence(VIEW, 99, "משהו ארוך מספיק")[0] is None
    assert verify_evidence(VIEW, "3", "אבדוק ואחזור")[0] is None
    assert verify_evidence(VIEW, 3, None)[0] is None


# ---------------------------------------------------------------- task A


def _card_json(**over) -> str:
    base = {"topic": "transfers_payments", "issue_he": "העברה נחסמה",
            "customer_request_he": "לשחרר את ההעברה", "outcome": "not_resolved",
            "outcome_ev": [], "commitments": [
                {"kind": "callback", "by": "bank", "when_he": "מחר",
                 "ev": [{"line": 3, "quote": "אבדוק ואחזור אלייך"}]}],
            "prior_contact_mentioned": True,
            "prior_ev": [{"line": 2, "quote": "כבר דיברתי איתכם אתמול"}],
            "retold": "yes", "retold_ev": [{"line": 2, "quote": "כבר דיברתי איתכם"}],
            "banker_aware_of_history": "no", "redirect": "none", "frustration": 2,
            "confidence": "high"}
    base.update(over)
    return json.dumps(base, ensure_ascii=False)


def test_card_parses_and_verifies():
    res = parse_card(_card_json(), VIEW, TAX, first=False)
    card = res.card
    assert not res.failed_quotes and not card.problems
    assert card.commitments[0].kind == "callback" and card.commitments[0].evidence.line == 3
    assert card.retold == "yes" and card.prior_contact_mentioned


def test_a_promise_without_verified_evidence_is_dropped():
    raw = _card_json(commitments=[{"kind": "callback", "by": "bank", "when_he": "",
                                   "ev": [{"line": 3, "quote": "נשלח לך מכתב בדואר"}]}],
                     retold_ev=[{"line": 1, "quote": "הגעת למרכז הבנקאות"}])
    res = parse_card(raw, VIEW, TAX, first=False)
    assert res.card.commitments == []
    assert res.card.retold == "unknown"          # a claim with no verified evidence
    assert len(res.failed_quotes) == 2


def test_first_contact_is_never_retold():
    assert parse_card(_card_json(), VIEW, TAX, first=True).card.retold == "first_contact"


@pytest.mark.parametrize("over", [{"topic": "pizza"}, {"outcome": "great"},
                                  {"frustration": 7}, {"commitments": "x"}])
def test_card_rejects_values_outside_the_schema(over):
    with pytest.raises(TaskError):
        parse_card(_card_json(**over), VIEW, TAX, first=False)


def test_card_rejects_non_json():
    with pytest.raises(TaskError):
        parse_card("סליחה, אני לא יכול", VIEW, TAX, first=False)


def test_free_text_from_the_model_is_redacted():
    card = parse_card(_card_json(issue_he="הלקוח עם ת.ז. 123456782 ביקש"), VIEW, TAX,
                      first=False).card
    assert "123456782" not in card.issue_he


# ---------------------------------------------------------------- tasks B and C


def _returns(items) -> str:
    return json.dumps({"returns": items}, ensure_ascii=False)


def _ret(no, cat="unclosed_loop", brk=False, q=()):
    return {"contact": no, "category": cat, "reason_he": "סיבה", "is_break_point": brk,
            "quote_ids": list(q)}


def test_returns_parse_and_keep_one_break_point():
    out = parse_returns(_returns([_ret(2, brk=True, q=["Q1"]), _ret(3, brk=True)]),
                        to_classify=[2, 3], categories=list(TAX.categories), qids=["Q1"])
    assert [a.contact_no for a in out] == [2, 3]
    assert [a.is_break_point for a in out] == [True, False]


@pytest.mark.parametrize("items", [
    [_ret(2)],                                   # 3 missing
    [_ret(2), _ret(2)],                          # duplicate
    [_ret(2), _ret(4)],                          # not to classify
    [_ret(2, q=["Q9"]), _ret(3)],                # unknown quote id
    [_ret(2, cat="angry"), _ret(3)],             # unknown category
])
def test_returns_rejects(items):
    with pytest.raises(TaskError):
        parse_returns(_returns(items), to_classify=[2, 3], categories=list(TAX.categories),
                      qids=["Q1"])


def test_story_parse():
    raw = json.dumps({"headline_he": "הבטחה שלא קוימה", "narrative_he": "פסקה.",
                      "status": "open", "status_note_he": "", "break_contact": 2,
                      "quote_ids": ["Q1"]}, ensure_ascii=False)
    s = parse_story(raw, contacts=[1, 2, 3], qids=["Q1"])
    assert s.status == "open" and s.break_contact == 2
    with pytest.raises(TaskError):
        parse_story(raw, contacts=[1], qids=["Q1"])


# ---------------------------------------------------------------- views


def test_compress_keeps_the_ends_and_the_cues():
    lex = load_lexicon()
    texts = [f"שורת מילוי מספר {i} בלי שום דבר מיוחד בה בכלל" for i in range(60)]
    texts[30] = "אני אבדוק ואחזור אלייך מחר"
    view = _view(*texts)
    small = compress(view, lex, max_chars=1500)
    assert small.chars <= 1500 or len(small.shown) <= 12
    assert 1 in small.shown and 60 in small.shown and 31 in small.shown
    assert "הושמטו" in small.render()
    assert compress(view, lex, max_chars=10**6) is view


def test_call_view_marks_the_join_and_low_confidence(tmp_path):
    t = RedactedTranscript(call_id="c1", engine="x", turns=[
        RedactedTurn(speaker="banker", start=0.0, end=4.0, text="שלום"),
        RedactedTurn(speaker="customer", start=9.0, end=12.0, text="שלום לך"),
        RedactedTurn(speaker="banker", start=13.0, end=15.0, text="במה אפשר לעזור")])
    segmap = tmp_path / "c1.segmap.json"
    segmap.write_text(json.dumps({"segments": [{"start_in_call": 0, "end_in_call": 10},
                                               {"start_in_call": 11, "end_in_call": 20}]}))
    dialog = tmp_path / "c1.dialog.json"
    dialog.write_text(json.dumps({"turns": [
        {"words": [{"probability": 0.9}]}, {"words": [{"probability": 0.9}]},
        {"words": [{"probability": 0.1}, {"probability": 0.2}]}]}))
    view = call_view("i1", t, segmap_path=segmap, dialog_path=dialog)
    assert [ln.uncertain for ln in view.lines] == [False, True, True]
    assert [ln.part for ln in view.lines] == [1, 1, 2]
    assert "— קטע הקלטה 2 —" in view.render()
    weak = call_view("i1", t, role_confidence=0.1)
    assert all(ln.uncertain for ln in weak.lines)


# ---------------------------------------------------------------- the runner


def _demo():
    spec = importlib.util.spec_from_file_location(
        "generate_journey_demo", REPO_ROOT / "scripts" / "generate_journey_demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def demo_config(tmp_path, monkeypatch):
    ws = tmp_path / "jdemo"
    _demo().generate(ws, stories=20, seed=5)
    from callqa.cli import main
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(ws / "output"))
    assert main(["journey", "import", "--xlsx", str(ws / "input" / "handoff.xlsx"),
                 "--atlas", str(ws / "input" / "atlas")]) == 0
    return load_config(None, {"paths": {"output_dir": str(ws / "output")}})


def test_mock_run_end_to_end_and_the_cache(demo_config):
    layer, stats = run_content(demo_config, mock=True)
    assert stats.failed == 0 and stats.cards > 0 and stats.no_text == 0
    assert layer.judgements and layer.verdicts
    assert all(j.decided_by == "llm" and j.basis == "content" for j in layer.judgements.values())
    stored = load_content(demo_config, _ds(demo_config))
    assert stored is not None and len(stored.cards) == len(layer.cards)
    # a second run reads every answer from the cache
    _layer2, stats2 = run_content(demo_config, mock=True)
    assert stats2.engine_calls == 0 and stats2.cache_hits > 0


def _ds(config) -> str:
    from callqa.journey.store import resolve_dataset_id
    return resolve_dataset_id(config, None)


class _Flaky(MockContentEngine):
    """Answers badly first, then like the mock."""

    def __init__(self, bad: str) -> None:
        super().__init__()
        self.bad = bad
        self.seen: set[str] = set()

    def answer(self, prompt: Prompt, feedback: str | None = None) -> str:
        if prompt.task == "card" and prompt.user not in self.seen:
            self.seen.add(prompt.user)
            return self.bad
        return super().answer(prompt, feedback)


def test_a_bad_answer_is_retried_with_feedback(demo_config):
    layer, stats = run_content(demo_config, engine=_Flaky("not json at all"))
    assert stats.retries >= stats.cards > 0 and stats.failed == 0


class _Broken(MockContentEngine):
    def answer(self, prompt: Prompt, feedback: str | None = None) -> str:
        if prompt.task == "story":
            raise RuntimeError("server went away")
        return super().answer(prompt, feedback)


def test_a_failing_task_does_not_stop_the_batch(demo_config):
    layer, stats = run_content(demo_config, engine=_Broken())
    assert stats.failed == stats.stories and layer.cards and not layer.verdicts


def test_unredacted_transcripts_are_not_read(demo_config):
    red = demo_config.paths.output_dir / "redacted"
    for p in red.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        data["enabled"] = False
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    layer, stats = run_content(demo_config, mock=True)
    calls = [c for c in layer.cards if not c.startswith("um")]
    assert calls == [] and stats.no_text > 0


def test_content_layer_holds_no_account_numbers(demo_config):
    run_content(demo_config, mock=True)
    folder = dataset_dir(demo_config, _ds(demo_config))
    import csv
    accounts = [row["account"] for row in csv.DictReader(
        (folder / "private" / "accounts.csv").open(encoding="utf-8"))]
    text = (folder / "content.json").read_text(encoding="utf-8")
    assert accounts and not any(a in text for a in accounts)


def test_cli_content_then_report(demo_config, monkeypatch, capsys):
    from callqa.cli import main
    assert main(["journey", "content", "--mock", "--limit-stories", "5"]) == 0
    assert "5 stories" in capsys.readouterr().out
    assert main(["journey", "report"]) == 0
    html = (demo_config.paths.output_dir / "reports" / "journey.html").read_text(encoding="utf-8")
    assert "שכבת תוכן: mock" in html


# ---------------------------------------------------------------- profiles


def test_profiles_and_endpoint_overrides(tmp_path):
    cfg = load_config(None, {"journey": {"llm": {"profile": "gpu",
                                                 "gpu_base_url": "https://llm.bank.local/v1",
                                                 "model": "bank-model"}}})
    prof = resolve_profile(cfg)
    assert (prof.ctx_tokens, prof.concurrency, prof.transcript_mode) == (32768, 6, "full")
    jc = judge_config_for(cfg, prof)
    assert jc.base_url == "https://llm.bank.local/v1" and jc.model == "bank-model"
    cpu = resolve_profile(cfg, "cpu")
    assert cpu.concurrency == 1 and cpu.transcript_mode == "compressed"
    assert 5000 < cpu.transcript_chars < 20000
