"""Evaluation tooling: agreement statistics, the label form, the gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from callqa.config import load_config
from callqa.journey.evaluate import (
    Label,
    cohen_kappa,
    compare_baseline,
    evaluate,
    macro_f1,
    read_labels,
)
from callqa.journey.store import dataset_dir, load_content, load_dataset, resolve_dataset_id
from callqa.journey.vocab import load_taxonomy

REPO_ROOT = Path(__file__).resolve().parent.parent
TAX = load_taxonomy()


def test_kappa_and_f1_by_hand():
    pairs = [("a", "a"), ("a", "a"), ("a", "b"), ("b", "b")]
    # po = 3/4; pe = (3*2 + 1*2) / 16 = 0.5; kappa = 0.5
    assert cohen_kappa(pairs) == pytest.approx(0.5)
    # a: p=1, r=2/3 -> 0.8; b: p=0.5, r=1 -> 2/3
    assert macro_f1(pairs) == pytest.approx((0.8 + 2 / 3) / 2)
    assert cohen_kappa([]) is None and macro_f1([]) is None


def test_read_labels_checks_categories(tmp_path):
    ok = tmp_path / "ok.csv"
    ok.write_text("﻿story_no,contact_no,category,retold\n1,2,unclosed_loop,yes\n",
                  encoding="utf-8")
    assert read_labels(ok, TAX) == [Label(1, 2, "unclosed_loop", "yes", "")]
    bad = tmp_path / "bad.csv"
    bad.write_text("story_no,contact_no,category\n1,2,angry\nx,1,legit_return\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable rows"):
        read_labels(bad, TAX)


def test_baseline_gate():
    base = {"category": {"accuracy": 0.9, "kappa": 0.8}, "failure": {"accuracy": 0.9}}

    class R:
        def to_json(self):  # noqa: ANN202
            return {"category": {"accuracy": 0.85, "kappa": 0.8}, "failure": {"accuracy": 0.95},
                    "retold": None}
    problems = compare_baseline(R(), base)
    assert problems == ["category accuracy fell from 0.900 to 0.850"]


def _demo():
    spec = importlib.util.spec_from_file_location(
        "generate_journey_demo", REPO_ROOT / "scripts" / "generate_journey_demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def read_demo(tmp_path, monkeypatch):
    ws = tmp_path / "jd"
    _demo().generate(ws, stories=30, seed=4)
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(ws / "output"))
    from callqa.cli import main
    assert main(["journey", "import", "--xlsx", str(ws / "input" / "handoff.xlsx"),
                 "--atlas", str(ws / "input" / "atlas")]) == 0
    assert main(["journey", "content", "--mock"]) == 0
    return ws, load_config(None, {"paths": {"output_dir": str(ws / "output")}})


def test_eval_against_the_demo_truth(read_demo):
    ws, config = read_demo
    ds_id = resolve_dataset_id(config, None)
    labels = read_labels(ws / "input" / "truth_labels.csv", TAX)
    result = evaluate(load_dataset(config, ds_id), load_content(config, ds_id), labels, TAX)
    # every truth label found its return: the numbering matches the report's
    assert result.matched == len(labels) > 10 and result.pending == 0
    assert result.category.accuracy > 0.7 and result.category.low is not None
    assert result.lines()


def test_cli_eval_writes_and_gates(read_demo, tmp_path, capsys):
    ws, config = read_demo
    from callqa.cli import main
    labels = str(ws / "input" / "truth_labels.csv")
    base = tmp_path / "base.json"
    assert main(["journey", "eval", "--labels", labels, "--write-baseline", str(base)]) == 0
    assert main(["journey", "eval", "--labels", labels, "--baseline", str(base)]) == 0
    data = json.loads(base.read_text(encoding="utf-8"))
    data["category"]["accuracy"] = 1.5
    base.write_text(json.dumps(data), encoding="utf-8")
    assert main(["journey", "eval", "--labels", labels, "--baseline", str(base)]) != 0
    assert "REGRESSION" in capsys.readouterr().err
    assert (dataset_dir(config, resolve_dataset_id(config, None)) / "eval.json").exists()


def test_label_form_is_blind(read_demo, capsys):
    ws, config = read_demo
    from callqa.cli import main
    assert main(["journey", "label-sample", "--n", "12"]) == 0
    form = dataset_dir(config, resolve_dataset_id(config, None)) / "labels" / "label_form.html"
    html = form.read_text(encoding="utf-8")
    assert html.count('class="item"') == 12
    content = load_content(config, resolve_dataset_id(config, None))
    # none of the model's reasons or headlines appear in the form
    definitions = {str(v.get("definition", "")) for v in TAX.categories.values()}
    for j in content.judgements.values():
        if j.reason_he not in definitions:        # the form shows the definitions themselves
            assert j.reason_he not in html
    for v in content.verdicts.values():
        assert v.headline_he not in html
    assert "אצלכם" not in html
