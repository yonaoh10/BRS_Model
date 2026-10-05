"""A wrong invocation ends in one sentence, never a Python traceback, and
never leaves anything on disk.

These are the invocations the interface audit of 1.5.0 tried against an
empty workspace (A1 §1.2), the two that printed a stack trace then
(`journey reveal` with no dataset, `nmf-info` on a missing file) among
them. Every one must exit non-zero with a message on stderr that holds no
"Traceback", and the workspace must stay empty.
"""

from __future__ import annotations

import pytest

from callqa.cli import main

WRONG = [
    ["journey", "report", "--level", "sesion"],
    ["journey", "reprot"],
    ["journey", "import", "--dry-run"],
    ["journey", "report", "--contact", "banana"],
    ["journey", "report", "--contact", "7"],
    ["journey", "report", "--dataset", "nope"],
    ["journey", "report", "--dataset", "ds-20990101-deadbeef"],
    ["journey", "report"],
    ["journey", "content"],
    ["journey", "estimate"],
    ["journey", "atlas-check"],
    ["journey", "label-sample"],
    ["journey", "eval"],
    ["journey", "process"],
    ["journey", "import", "--xlsx", "/nonexistent/handoff.xlsx", "--dry-run"],
    ["journey", "import", "--contract", "/nonexistent/folder", "--dry-run"],
    ["journey", "import", "--audio", "/nonexistent/recordings.zip", "--dry-run"],
    ["journey", "reveal", "5"],
    ["nmf-info", "/nonexistent.nmf"],
    ["executive-report", "--from", "2026-13-01"],
]


def _run(args, monkeypatch, tmp_path, capsys, caplog):
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("CALLQA_PATHS__INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.chdir(tmp_path)
    try:
        code = main(args)
    except SystemExit as exc:            # argparse's own usage errors
        code = exc.code
    captured = capsys.readouterr()
    # commands that report through the logger print on the console in real
    # use (the handler writes to stderr); under pytest that text is caplog's
    return code, captured.out + captured.err + caplog.text, out_dir


@pytest.mark.parametrize("args", WRONG, ids=lambda a: " ".join(a))
def test_a_wrong_invocation_is_one_sentence_and_writes_nothing(args, monkeypatch, tmp_path,
                                                                capsys, caplog):
    code, text, out_dir = _run(args, monkeypatch, tmp_path, capsys, caplog)
    assert code not in (0, None)
    assert "Traceback" not in text and "File \"" not in text, text
    assert text.strip(), "a failure must say something"
    assert not out_dir.exists() or not any(out_dir.rglob("*")), \
        f"a failed command left files: {sorted(p.name for p in out_dir.rglob('*'))}"


def test_a_missing_dataset_id_says_what_an_id_looks_like(monkeypatch, tmp_path, capsys, caplog):
    _code, text, _ = _run(["journey", "report", "--dataset", "ds-20990101-deadbeef"],
                          monkeypatch, tmp_path, capsys, caplog)
    assert "ds-YYYYMMDD-xxxxxxxx" in text and "/" not in text.split("no dataset with id")[-1][:60]
