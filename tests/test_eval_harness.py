"""The evaluation harness: metrics, whole-system eval, and the regression gate."""

from __future__ import annotations

from callqa.eval.harness import DEFAULT_GOLDEN, EvalReport, compare, evaluate
from callqa.eval.metrics import cer, redaction_prf, role_accuracy, wer


def test_wer_and_cer() -> None:
    assert wer("a b c d", "a b c d") == 0.0
    assert wer("a b c d", "a x c d") == 0.25          # one substitution of four
    assert wer("a b c", "a b c d") == round(1 / 3, 4)  # one insertion
    assert cer("abcd", "abcd") == 0.0
    assert cer("abcd", "abxd") == 0.25


def test_redaction_prf_scores_find_pii_against_gold() -> None:
    text = "תעודת הזהות שלי היא 123456782 והטלפון 052-1234567"
    prf = redaction_prf(text, ["123456782", "052-1234567"])
    assert prf["recall"] == 1.0 and prf["precision"] == 1.0 and prf["f1"] == 1.0

    # An identifier with no shape, recognised only because it answers a
    # verification question. This case used to score 0.0 and was the leak the
    # harness was built to expose.
    answered = redaction_prf("שם האם לאימות? רות", ["רות"])
    assert answered["recall"] == 1.0

    # Recall must still be ABLE to fall, or the metric proves nothing. A name
    # mentioned in passing, with no question to anchor it, is the class that
    # remains open: nothing in "דנה" says identifier and nobody asked for it.
    missed = redaction_prf("העברתי את הבקשה לטיפול. דנה תחזור אליך מחר", ["דנה"])
    assert missed["gold"] == 1                  # the label really was located
    assert missed["recall"] == 0.0


def test_role_accuracy() -> None:
    ref = [{"speaker": "banker"}, {"speaker": "customer"}, {"speaker": "banker"}]
    assert role_accuracy(ref, ref) == 1.0
    swapped = [{"speaker": "customer"}, {"speaker": "customer"}, {"speaker": "banker"}]
    assert role_accuracy(ref, swapped) == round(2 / 3, 4)


def test_evaluate_the_golden_set_is_known_good(workspace) -> None:  # noqa: ANN001
    report = evaluate(workspace)   # workspace is mock mode
    assert report.n_calls == 6
    assert report.wer_mean == 0.0 and report.cer_mean == 0.0
    assert report.role_accuracy_mean == 1.0
    assert report.redaction_recall_mean == 1.0
    assert report.redaction_precision_mean == 1.0
    assert report.judge_qwk == 1.0
    assert report.fingerprint["code_version"]


def test_the_committed_baseline_matches_the_golden_set(workspace) -> None:  # noqa: ANN001
    baseline = EvalReport.model_validate_json(
        (DEFAULT_GOLDEN.parent / "baseline.json").read_text(encoding="utf-8"))
    report = evaluate(workspace)
    assert compare(report, baseline) == [], "the committed baseline must not regress"


def _report(**over) -> EvalReport:  # noqa: ANN003
    base = dict(created_at="t", n_calls=6, seconds_total=1.0, wer_mean=0.0, cer_mean=0.0,
                role_accuracy_mean=1.0, redaction_recall_mean=1.0,
                redaction_precision_mean=1.0, redaction_f1_mean=1.0, judge_qwk=1.0)
    base.update(over)
    return EvalReport(**base)


def test_compare_flags_the_right_regressions() -> None:
    good = _report()
    # a redaction leak and a WER rise are regressions
    worse = _report(redaction_recall_mean=0.8, wer_mean=0.2)
    metrics = {r["metric"] for r in compare(worse, good)}
    assert metrics == {"redaction_recall_mean", "wer_mean"}
    # an improvement is not a regression
    better = _report(wer_mean=0.0, judge_qwk=1.0)
    assert compare(better, good) == []
    # a change within tolerance is not a regression
    tiny = _report(redaction_recall_mean=0.99)
    assert compare(tiny, good) == []


def test_evaluation_leaves_no_raw_transcripts_behind(tmp_path, monkeypatch) -> None:
    """The harness runs the whole pipeline into a temporary tree, and that tree
    holds RAW, unredacted transcripts. It was never deleted: harmless on the
    synthetic set, and customer transcripts left in /tmp on every run once
    pointed at the real golden set it exists for."""
    import tempfile

    from callqa.config import Config

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config = Config()
    config.run.mock = True
    config.audio.vad = "energy"
    config.asr.engine = "mock"
    config.judge.engine = "mock"
    evaluate(config)
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("callqa-eval-")]
    assert not leftovers, f"raw evaluation tree left behind: {leftovers}"


def test_the_gate_notices_redaction_that_was_never_applied(monkeypatch) -> None:
    """Recall scores the DETECTOR over the reference text, so with redaction
    switched off it still read 1.0 and the gate passed. Applied recall scores
    the transcript the pipeline actually wrote, and one surviving identifier
    is a regression with zero tolerance."""
    from callqa.config import Config

    config = Config()
    config.run.mock = True
    config.audio.vad = "energy"
    config.asr.engine = "mock"
    config.judge.engine = "mock"
    good = evaluate(config)
    assert good.redaction_applied_recall_mean == 1.0

    config.redaction.enabled = False
    leaking = evaluate(config)
    assert leaking.redaction_recall_mean == 1.0             # the detector is still fine
    assert leaking.redaction_applied_recall_mean == 0.0     # the artifact is not
    assert {r["metric"] for r in compare(leaking, good)} == {"redaction_applied_recall_mean"}
