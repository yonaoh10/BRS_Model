# Changelog

All notable changes to callqa are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project follows
semantic versioning: the CLI commands, exit codes, and on-disk artifact schemas
are the public contract.

## [1.0.0] — 2026-09-17

First release intended to be operated in a bank for years: the pipeline is
wrapped in the operational discipline needed to run, measure, and maintain it
over its lifetime. The pipeline core (`process_call` and the stage artifact
contracts) is unchanged from 0.1.0 — everything here is additive.

### Added — MLOps discipline
- **Reproducibility.** A run manifest per batch (`data/output/runs/`) capturing
  an environment fingerprint (code version, git sha, effective-config hash,
  rubric hash, prompt version, model weight hashes), per-call and per-stage
  timing, counts, and estimated cost. `callqa verify <call_id>` reports whether
  a stored result is still reproducible and, if not, which input changed.
- **Model weight hashing.** `scripts/download_models.py` records a content hash
  of each model's weights in the manifest; `callqa preflight` verifies them.
- **Preflight.** `callqa preflight` checks config, model presence/hashes,
  endpoint reachability, free disk, and inputs before a batch starts.
- **Evaluation.** `callqa eval` scores the whole system against a golden set
  (WER/CER, speaker role accuracy, redaction recall/precision/F1, judge QWK,
  timing); `--baseline` gates a change against a blessed report. Golden set in
  `eval/golden/`; baseline in `eval/baseline.json`.
- **Drift.** `callqa drift` flags movement in the score distribution (PSI),
  review rate (Wilson interval), and quality signals (Shewhart control limits)
  against a known-good baseline; history in `data/output/drift/history.jsonl`.
- **Feedback loop.** `callqa review` / `callqa review-queue` capture a reviewer's
  verdict on a held call into the calibration set.
- **Retention.** `callqa retention` retires raw PII-bearing artifacts (default
  90 days), keeps derived non-PII, and logs every deletion.
- **Release.** This changelog, CI (`.github/workflows/ci.yml`) running lint,
  tests, the mock end-to-end, and the evaluation gate.
- Documentation: `docs/MLOPS.md`.

### Added — earlier in this cycle
- A redacted-audio player in the dashboard call drawer (the browser only ever
  reaches audio silenced wherever the transcript was masked).

### Fixed
- Redaction: comma-grouped and space-hyphen-space separated identifiers no
  longer leak.
- CLI: a path-traversal `--call-id` is refused before it can escape `output_dir`.
- State: `--max-workers > 1` no longer deadlocks on the WAL pragma.
- Scoring: scorecards from different rubric/prompt versions are never pooled;
  a constant-output gate dimension is flagged and blocks calibration; rater
  means round half-up.
- Reporting: banker index links resolve and distinct bankers no longer collide.
- Features: `longest_banker_monologue_sec` no longer shrinks on overlapping
  segments.

### Known gaps
- A mother's name given as a standalone answer (not inline "שם האם הוא X") is
  not masked — surfaced by the evaluation harness, tracked in `docs/MLOPS.md`.

## [0.1.0]

Initial pipeline: ingestion, audio prep, ASR, speaker attribution, redaction,
features, LLM-judge scoring, reports, and calibration; resumable with a SQLite
state ledger and a strict exit-code contract.
