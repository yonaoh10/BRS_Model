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
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from callqa.eval.metrics import gold_spans  # noqa: E402 - same locator the metric uses
from callqa.models import DialogTranscript, ScoreCard  # noqa: E402

# Every identifier seeded into the fixture dialogs, hand-labelled. Gold PII per
# call = those present in its transcript, located by whole-word search
# (independent of find_pii, so recall is a REAL measurement, not a restatement
# of what the detector happens to find).
#
# Two kinds, deliberately:
#   SHAPED    - recognisable on their own: an ID passes a checksum, a phone has
#               a prefix, a long digit run is account-like.
#   SHAPELESS - nothing about the string says "identifier". A mother's name, a
#               date of birth, a street address. The only signal is that a
#               banker asked for them, so they are only maskable by reading the
#               question that precedes the answer.
#
# The shapeless ones are labelled HERE, in the answer key, even while the
# detector misses them. A recall metric that quietly omits the identifiers a
# system is known to leak reports 1.0 and means nothing; it must be allowed to
# go red. See docs/MLOPS.md.
KNOWN_PII = [
    # shaped
    "123456782",        # Israeli ID (checksum-valid, synthetic)
    "052-1234567",      # mobile
    "765432",           # account number
    "4580",             # card last four
    # shapeless
    # Labelled exactly as SPOKEN, prefix letter included: Hebrew glues its
    # one-letter prepositions onto the following word ("ב"+"חמישי" -> "בחמישי"),
    # so a literal written without the prefix has no word boundary to anchor to
    # and silently labels nothing. Scoring is overlap-based, so a detector that
    # masks the date without the preposition still counts as a hit.
    "רות",                                # mother's name, standalone answer
    "בחמישי למרץ שמונים ושתיים",           # date of birth, spoken in words
    "רחוב הרצל 15 בחיפה",                  # street address
]


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
        # whole-word, so "רות" is not "found" inside "שירות" - the same locator
        # the metric uses, so the label and the score agree by construction.
        "pii": [lit for lit in KNOWN_PII if gold_spans(transcript, [lit])],
    }


def main() -> int:
    # Printing a Hebrew path to a redirected stream is a UnicodeEncodeError
    # under the Windows ANSI code page (callqa.portable.configure_stdio, inlined
    # because this script may run before callqa is installed).
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except (AttributeError, ValueError, OSError):
                pass
    work = Path(tempfile.mkdtemp(prefix="callqa-golden-"))
    inp, out = work / "input", work / "output"
    utf8 = dict(os.environ, PYTHONUTF8="1")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_sample_data.py"),
                    "--input-dir", str(inp)], check=True, capture_output=True, env=utf8)
    cfg = work / "cfg.yaml"
    cfg.write_text(
        f"paths: {{input_dir: '{inp.as_posix()}', output_dir: '{out.as_posix()}', "
        f"state_db: '{work.as_posix()}/s.db', models_dir: '{work.as_posix()}/models'}}\n"
        "run: {mock: true}\naudio: {vad: energy}\nasr: {engine: mock}\n"
        "judge: {engine: mock, model: mock}\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "callqa", "run", "--config", str(cfg), "--mock"],
                   cwd=REPO_ROOT, check=True, capture_output=True, env=utf8)

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
