#!/usr/bin/env python3
"""Build eval/golden/references.json — the synthetic golden reference set.

Deterministic and PII-free-to-commit: derived from the canned dialogs in
callqa.asr.fixtures (known speaker turns and text) plus the KNOWN seeded
identifiers (hand-labelled, independent of find_pii). Scores are captured from a
mock run, so the synthetic set is a known-good snapshot the eval gate defends. A
bank points config/CLI at its own real, human-labelled golden set instead.

Usage: python scripts/build_golden_set.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from callqa.models import DialogTranscript, ScoreCard  # noqa: E402

# Structured identifiers the redactor is designed to catch, seeded into the
# fixture dialogs. Gold PII per call = those present in its transcript, located
# by string search (independent of find_pii, so recall is a real measurement).
# Deliberately NOT labelled: a mother's name given as a STANDALONE answer to
# "what is your mother's name?" — find_pii only catches it inline ("שם האם הוא
# X"), so it is a known redaction gap, tracked separately (docs/MLOPS.md), not
# folded into the recall baseline.
KNOWN_PII = ["123456782", "052-1234567", "765432", "4580"]


def _reference_for(out: Path, call_id: str) -> dict:
    # Reference = the captured known-good OUTPUT (the raw dialog the pipeline
    # produced), so WER/role measure DEVIATION from known-good on the synthetic
    # set. A real golden set instead carries human-authored ground truth.
    dialog = DialogTranscript.model_validate_json(
        (out / "transcripts" / f"{call_id}.dialog.json").read_text(encoding="utf-8"))
    transcript = " ".join(t.text for t in dialog.turns)
    return {
        "transcript": transcript,
        "speakers": [{"speaker": t.speaker, "text": t.text} for t in dialog.turns],
        "pii": [lit for lit in KNOWN_PII if lit in transcript],
    }


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="callqa-golden-"))
    inp, out = work / "input", work / "output"
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_sample_data.py"),
                    "--input-dir", str(inp)], check=True, capture_output=True)
    cfg = work / "cfg.yaml"
    cfg.write_text(
        f"paths: {{input_dir: {inp}, output_dir: {out}, state_db: {work}/s.db,"
        f" models_dir: {work}/models}}\n"
        "run: {mock: true}\naudio: {vad: energy}\nasr: {engine: mock}\n"
        "judge: {engine: mock, model: mock}\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "callqa", "run", "--config", str(cfg), "--mock"],
                   cwd=REPO_ROOT, check=True, capture_output=True)

    references: dict[str, dict] = {}
    for card_path in sorted((out / "scores").glob("*.json")):
        cid = card_path.stem
        ref = _reference_for(out, cid)
        card = ScoreCard.model_validate_json(card_path.read_text(encoding="utf-8"))
        ref["scores"] = {dim: ds.score for dim, ds in card.scores.items()}
        references[cid] = ref

    golden = REPO_ROOT / "eval" / "golden"
    golden.mkdir(parents=True, exist_ok=True)
    (golden / "references.json").write_text(
        json.dumps(references, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {golden / 'references.json'} — {len(references)} golden calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
