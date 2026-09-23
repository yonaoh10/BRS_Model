# Changelog

All notable changes to callqa are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project follows
semantic versioning: the CLI commands, exit codes, and on-disk artifact schemas
are the public contract.

## [1.1.0] — 2026-09-23

The hand-off release: prepared for a bank's implementation engineer to download
the repository, extract it, and deploy it on infrastructure of his own choosing.

### Removed — every infrastructure assumption
- The dev-phase cloud option in full: the RunPod lifecycle CLI and pod scripts,
  the remote ASR engine and its config, and the `.env` cloud endpoint keys.
- The Docker image and compose file. The project ships **no** containerisation
  opinion; how it is packaged and scheduled is the deploying engineer's call.
- The QA scratch material committed under `qa/` (still in git history), which
  carried un-redacted transcripts of a test call, a macOS crash report, and
  scripts hardcoded to the original developer's home directory.
- `pydantic-settings` from `requirements.txt` — pinned, imported nowhere, and a
  dependency the bank would have had to review and keep patched for nothing.

### Added
- **`docs/DEPLOYMENT.md`** — the front door, written for an engineer with no
  contact with the people who built this. Prerequisites, proving the install
  before downloading 25 GB of models, the gated model licence, serving the
  judge on any OpenAI-compatible endpoint, and what a security review will ask.
- `scripts/license_inventory.py` — generates a licence inventory from what is
  installed on the deployed machine, flagging copyleft, restricted and
  undeclared. `download_models.py` records each model's declared licence into
  `MODELS_MANIFEST.json` at download time.
- `speakers.mode` now works (`auto` / `stereo` / `mono`); it was a documented
  setting that no code read.
- A documentation test: every repo file a document points at, every `callqa`
  subcommand it shows, and every `CALLQA_*` variable it names is checked.
- CI unpacks `git archive` and runs the suite and the pipeline from it, because
  what a reviewer downloads is not the working tree.

### Fixed — security
- **Identifiers with no shape are now masked.** A mother's name, a date of
  birth or an address given as a standalone answer to a verification question
  survived into the judge prompt, the reports, the dashboard and the
  browser-playable audio. Detection now reads the question and masks the
  answer, across turns and across speakers. Measured on the golden set:
  redaction recall **0.611 → 1.000**, precision unchanged at 1.000.
- **Presidio is opt-in (`redaction.presidio`, default off).** Constructing it
  loads a spaCy pipeline, and presidio downloads that model when missing — an
  outbound network call at runtime from a system that promises it makes none,
  falling back silently when it failed. The built-in pass already detects
  payment cards, e-mail and IBAN, so nothing is lost by default.
- `probe_channels` fails closed on every path. Its documented promise — an
  undecodable file is `dual_mono`, never `stereo` — held only where ffmpeg was
  installed; without it the stdlib reader raised `EOFError` straight out. This
  was the failing test that made CI red at 1.0.0.

### Fixed — correctness
- The offline wheels bundle downloaded nothing at all: one `--platform` tag
  cannot resolve the pin set, and the first error aborted the script. The whole
  air-gapped install path went through this step.
- A reviewer's verdict written into an existing-but-empty `human_ratings.csv`
  was silently swallowed: the row was written with no header, and the next read
  took it *as* the header.
- A non-integer cell in `human_ratings.csv` crashed calibration with a bare
  `ValueError`; a rubric change crashed report rendering with a bare `KeyError`.
  Both now name the row, the call and what to do.
- NER honoured a hardcoded `models/` instead of `paths.models_dir`.
- `git_sha()` returns `none` unless the repository root has its own `.git`, so
  an extract unpacked inside another checkout cannot stamp that repository's
  commit onto every run manifest.
- `ops/preflight.py` imported a module the cloud removal had deleted.
- The Makefile called `python`, which a stock Debian or RHEL host does not have.

### Changed — honesty of the evaluation
- Every eval report now states, in a field, that on the synthetic golden set
  WER/CER/role-accuracy/QWK are **regression sentinels, not accuracy** — the
  references are the mock pipeline's own output — and the CLI prints it under
  the numbers. Redaction recall and precision are the exception and say so.
- The golden set labels the shapeless identifiers it previously omitted. A
  safety metric that cannot go red is not a safety metric.
- `eval --set-baseline` refuses a modified working tree. The 1.0.0 baseline
  recorded `git_sha "...-dirty"`: a blessed reference produced by code that was
  never committed.

### Known limits
- A name mentioned in passing that nobody asked for is not masked. Nothing
  about the string marks it as an identifier and no question anchors it.
  `redaction.ner` helps, at a cost. Documented in `docs/DEPLOYMENT.md` §9.
- There is still no measured agreement between the judge and human QA
  reviewers: that needs ≥20 human-rated calls, which the project has never had.

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
  not masked — surfaced by the evaluation harness. **Fixed in 1.1.0.**

## [0.1.0]

Initial pipeline: ingestion, audio prep, ASR, speaker attribution, redaction,
features, LLM-judge scoring, reports, and calibration; resumable with a SQLite
state ledger and a strict exit-code contract.
