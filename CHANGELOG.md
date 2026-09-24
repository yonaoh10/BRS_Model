# Changelog

All notable changes to callqa are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project follows
semantic versioning: the CLI commands, exit codes, and on-disk artifact schemas
are the public contract.

## [1.5.0] — 2026-09-24

Two levels of analysis - the single contact and the banker session - built
exactly as the bank's own Atlas repeat-contact project defines and shows them
(ATL_R01, ATL_R02, ATL_R03).

### Added
- Banker sessions built from the Atlas log rows by the bank's rules: a new
  session when the banker changes, after a pause of more than 30 minutes, or
  on code 201; rows of one second keep the export's order. An export's own
  session table is checked against the sessions its rows make, and any
  difference is reported at import (`atlas_sessions_differ`).
- The session kind (execute > info > unclassified > screen opening; 990 alone
  = not about the customer) and the unit class (banking centre, back office,
  the account's branch, another branch or unit). The code lists ship as
  `journey_atlas_codes.yaml`, the lists of ATL_R02.
- Report, single-contact level: what stood behind each contact type in
  Atlas; a contact explorer; each contact's sessions in its story card; and
  `journey report --contact STORY:N`, a page for one contact in three
  layers (the contact table, what was said, Atlas) with the "executed since
  the previous contact" test of ATL_R04.
- Report, session level: the definitions, the page of fifteen numbers with
  what each one means, sessions by kind and unit, tables by status, topic and
  unit, three stories as a timeline table, a session explorer, and appendices
  with the code classification and the completeness checks.
- `journey report --level all|call|session`; `data/output/reports/journey_contacts.csv`,
  `data/output/reports/journey_sessions.csv`, `data/output/reports/journey_codes.csv`.
- `journey atlas-check [--expect eval/atlas_r02_expected.yaml]`: the checks
  and the page of numbers, compared with the figures ATL_R02 published on the
  96 stories.
- `journey.atlas` settings: `gap_min`, `start_op`, `windows`,
  `coverage_from`, `codes`, `top_units`, `cases`.

### Changed
- Sessions are tied to every message of a correspondence (each at its own
  time and direction), not to the correspondence as a whole; a contact of
  unknown direction is measured from its moment, not from its end; on equal
  distance, the earlier contact wins.
- Atlas coverage is by day: a story is covered when its first contact is on
  or after the first day the log holds (ATL_R02).
- The banker figures are measured on ATL_R02's base (covered stories with at
  least one session). "Crossed" = the banking centre and a branch both worked
  on the story; handoffs are every change of unit class; "sessions with no
  execute operation" replaces "view-only sessions".
- After an abandoned call, the customer "came back first" when any next
  contact came before a banker opened the account (ATL_R02).
- The synthetic demo exports log rows only; its sessions are built on import.

## [1.4.1] — 2026-09-24

The reports on the desktops they are really opened on.

### Fixed
- A report opened in Internet Explorer or in Edge's "IE mode" - which bank
  desktops often apply to local HTML files - came out white, unstyled and in
  the default font: that engine drops every CSS rule written with a CSS
  variable. Every such rule now carries a literal fallback in front of it
  (the light theme); modern browsers keep the variables, dark mode included.
  Both reports also ask the browser for its newest mode (`X-UA-Compatible`).
- When a mail gateway strips the report's styles, or the page is opened in
  such a mode, a notice at the top says so and what to do (open it in Edge,
  or use the PDF); wherever the page is shown as designed, it is hidden.

### Added
- `callqa journey report --pdf`: the report printed to PDF by the Edge or
  Chrome already on the machine (`CALLQA_BROWSER` overrides), for e-mail.
  The printed version holds a table of every story and the ten stories with
  the most returns in full.
- `docs/examples/journey-demo.pdf`; the sample reports regenerated.

## [1.4.0] — 2026-09-24

Customer journeys and repeat contacts. A tool that takes a batch of customers
- every contact of each account, the recorded calls, the written
correspondence and the Atlas banker log - and answers why customers come back,
what it costs, and what to do about it. It runs entirely on the bank's
machines, on a CPU with a small local model or on an internal GPU server.

### Added
- `callqa journey import`: a batch from the vendor-format handoff workbook
  (three sheets, found by their content), or from a documented CSV contract,
  with the recordings (a folder or a ZIP) and the Atlas exports. Accounts
  become keyed pseudonyms at import ("story NNN"); the map back lives only in
  the dataset's private folder, and `callqa journey reveal` logs every look.
  An import report of counts and problems (orphan files, a declared repeat
  count that does not match) to check before anything runs. `--dry-run`.
- Calls recorded in several files: the parts of one call id are joined into
  one recording with a segment map, transcribed once, and speaker roles are
  checked across the joins. NICE `.nmf` recordings are read directly
  (G.711 natively, other codecs through ffmpeg); `callqa nmf-info` shows a
  file's structure - never its audio - and `callqa nmf-convert` makes WAVs
  for a listening check. `metadata.csv` accepts a `segment` column.
- The deterministic engine: each story's timeline, Atlas banker sessions
  attached to the contact they served, the facts of every return
  (abandoned, bank-initiated, answered with an operation / a look / no
  trace), promises kept or broken from facts (an outbound contact or an Atlas
  operation before the customer came back, within 2 Sunday-Thursday days),
  and each story's status with its basis.
- `callqa journey content`: the language model reads the recorded calls and
  messages - a card per contact (topic, request, outcome, the bank's
  promises, "told it again"), the reason for each return in six categories,
  and a headline and paragraph per story. It reads numbered lines, and every
  quote is verified against its line and stored in the line's own words.
  Uncertain lines are never quoted, a claim with no verified evidence is
  dropped, and answers are cached by content. One setting,
  `journey.llm.profile`, switches between `cpu` and `gpu`. A mock engine
  runs the whole chain in CI.
- `callqa journey process`: transcribe, read and report a whole batch,
  stories with the most returns first, stopping cleanly at `--until` or
  `--max-hours` and continuing on the next run. One run per dataset at a
  time. `callqa journey estimate` measures on the machine and predicts.
- `callqa journey report`: `data/output/reports/journey.html`, with the
  stories and the returns as CSV and the numbers as JSON beside it.
  - Level 1: an opinion, key figures (each with a "?" giving its
    definition, base and what would make it wrong), findings and actions.
  - Level 2: returns, reasons (strict and with inference), what is known
    about returns with no content, topics, time between contacts,
    Kaplan-Meier time to close, promises, abandonment, banker effort and
    handoffs, call quality inside journeys, branches, and a table of every
    metric.
  - Level 3: a story explorer and a card per story with a two-lane
    timeline: the customer above, the bankers below, promise arcs.
  - `--no-quotes` for wide distribution. Listed in the index and the
    dashboard.
- `callqa journey label-sample` (a blind offline labelling form over a
  stratified sample) and `callqa journey eval`: accuracy, macro-F1 and
  Cohen's kappa with story-level intervals, a confusion matrix, and a
  baseline gate (`eval/journey_baseline.json`, run in CI against the
  synthetic demo's known truth).
- `scripts/generate_journey_demo.py`: 100 synthetic stories in the vendor
  format, with split NICE recordings and Atlas exports.
  `docs/examples/journey-demo.html` is the report it produces.
- `docs/journey_report_he.md`: running it, the report, the method, the
  accuracy check and privacy.

### Changed
- `VLLMJudge.chat_json`: the judge's structured-output request is shared
  with the journey tasks; scoring behaves exactly as before.
- `asr.beam_size` is configurable.

## [1.3.0] — 2026-09-24

The management report. One self-contained HTML file over a whole batch of
calls - made for a month of ~1,000 - written for management and readable at
three depths, with every figure traceable to the calls behind it.

### Added
- `docs/examples/executive-demo.html` and `.pdf`: a sample management report over
  1,000 synthetic calls, to see the result before there is real data.
- `callqa executive-report` and, automatically, every `callqa report`:
  `data/output/reports/executive.html`, plus a numbers-only CSV for Excel and
  JSON for BI tools beside it.
  - **Level 1, executive summary:** an overall verdict and a bottom line
    written from the numbers by fixed rules (no language model; the same batch
    always yields the same opinion), six key indicators with trend lines, the
    five main findings ranked by severity and size (always including the
    strongest good news), and three recommended actions, each with its impact
    in index points and gate failures removed, the bankers to focus on and the
    owner.
  - **Level 2, analysis:** score distribution per dimension and an exact
    decomposition of "where the points go" (per dimension plus the gate
    penalty, summing to 100 minus the index); the index distribution; trends
    by day, week or month with confidence bands and the gate-failure rate;
    breakdowns by call type, call length and recording layout; bankers as a
    dot plot with confidence intervals and a banker-by-dimension heat map;
    behavioural drivers (talk share, interruptions, questions, dead air,
    monologues, length) against the index; compliance risk per gate; and
    coverage, privacy and calibration status.
  - **Level 3, detail:** an explorer over every call - filters, sort, search,
    pagination, a drill-down with each dimension's score, reasoning and
    quote, links to the call and banker reports, a banker profile, and a CSV
    export of exactly what is filtered; a case library of excellent and weak
    moments per dimension; and a printable list of the calls needing attention.
  - Every finding links to its analysis and to its calls. Differences are
    called differences only when significant at 99% (Welch), at least 3 points
    and backed by at least 8 calls; small groups are shown, not ranked.
  - Scope with `--from/--to` (ISO or Israeli day-first dates), `--call-type`,
    `--banker`, `--run`; `--no-quotes` for a version with no call text at all;
    `--name`, `--title`.
  - Print/PDF layout (each level on a new page), dark mode, phone widths, no
    script errors with scripting on, and everything but the explorer readable
    with it off.
- `scripts/generate_batch_demo.py`: a realistic synthetic batch (1,000 calls by
  default, in its own `data/demo-batch` folder) and its management report,
  clearly labelled as a demonstration - to see the report before real data
  exists.
- The dashboard links and serves the management report.
- `docs/executive_report_he.md`: what each part shows and how each number is
  computed.
- CI builds the report over 1,000 synthetic calls on Linux and Windows and
  opens it in a real browser (Edge on Windows) to check its script ran.

### Privacy
- The report reads only redacted artifacts. Quotes and reasoning come only
  from calls whose redacted transcript parses, is theirs and says redaction
  was on (fail closed); judge prose is re-redacted at render time; error
  texts, file names, banker names and model paths never appear; embedded data
  is escaped so no text can end the script element; the explorer places all
  text with textContent.

### Changed
- `callqa report` also writes the management report and links it from the
  index; `first_run.py` and `make mock-e2e` run `calibrate` before `report`,
  so the report shows the calibration status.

### Fixed
- Redaction treats unusual spaces (thin, no-break, line separator, vertical
  tab...) as separators: "123 456 782" written with thin spaces was invisible
  to every identifier pattern.
- A scorecard is counted only as the call whose file it is: a copied or
  renamed card no longer reports under, and borrows the redaction clearance
  of, another call.

## [1.2.0] — 2026-09-24

Windows. The bank's desktops are Microsoft VDI - Windows, no administrator
rights, no WSL - and the software now installs and runs there natively. Every
change was driven by a read-only audit of the whole project for Windows and by
a real Windows machine: CI runs the suite, the README's steps and an offline
install on Windows, and a second workflow runs the REAL engines there (ASR,
VAD, NER, redaction, and an llama.cpp judge over the OpenAI protocol).

### Added
- `scripts/install.py`, `scripts/first_run.py`, `scripts/build_offline_bundle.py`
  - one Python command per job, the same on Windows and Linux, replacing the
  bash scripts. No activation, so PowerShell's execution policy never matters.
- `scripts/start_llama_server.py` - the judge on a machine without a GPU;
  `download_models.py --llm-gguf` fetches one quantized file for it.
- `callqa.portable` - the OS calls whose POSIX spelling was wrong on Windows.
- Preflight: data location (OneDrive, network share), folder privacy, the
  Visual C++ runtime, the audio decoder, and a Hugging Face cache copied
  without its symlinks.
- `--log-file` for scheduled runs; `asr.device`; `judge.allow_public_endpoint`.

### Fixed — would not run on Windows at all
- `os.uname()` does not exist there: every call failed taking its lock.
- `os.kill(pid, 0)` is CTRL_C_EVENT / TerminateProcess there, not a probe.
- `pip install -e .` broke in a folder whose path has Hebrew in it.
- Hebrew printed to a redirected stream (a Scheduled Task, CI) raised
  UnicodeEncodeError; subprocess output was decoded in the ANSI code page.
- ctranslate2 4.5.0 imports `pkg_resources`, which current setuptools no longer
  ships: pinned to 4.6.2.
- A Windows path written into config.yaml's double quotes broke the YAML.

### Fixed — privacy and security
- pyannote.audio 4 sends usage telemetry to otel.pyannote.ai by default: off,
  and the CLI enforces `HF_HUB_OFFLINE`.
- Data folders are owner-only on Windows (chmod does nothing there; folders
  under C:\\ inherit "Users: read" on multi-session hosts); the installer does
  the same for the project folder.
- On a shared host: the judge default is 127.0.0.1 (localhost is [::1] first
  on Windows); the judge client checks the server serves the configured model
  before sending anything, never goes through the system proxy for loopback,
  and refuses a public internet endpoint; the dashboard binds exclusively.
- Stale redacted audio fails closed, and the dashboard never serves a WAV
  older than its transcript.

### Fixed — correctness
- Distinct recording names could share a call_id (all-Hebrew names became
  "call") and silently reuse another recording's results.
- `first_run.py` works in `data/demo/`: rerun after real calls were in place,
  it marked them processed with mock output.
- `watch` processed files Windows was still copying (copies pre-size the file).
- Excel CSVs: empty rows, mixed UTF-8/cp1255 rows, TAB/';' separators, bidi
  marks, case-insensitive duplicate call_ids, device names (CON, NUL).
- Model hashes, the rubric hash and the config hash are identical on Windows
  and Linux.
- mp3/m4a and telephony (G.711) WAV decode through PyAV - installed with the
  engines - so no ffmpeg.exe is needed.

### Changed
- `download_models.py` writes the judge model into `.env`, not config.yaml.
- PyYAML pinned to 6.0.3 (pyannote needs >=6.0.2).
- README is written for Windows first; the judge options are stated as
  measured: a GPU server for real scoring, a CPU judge to see the chain work.

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

### Fixed — found by an independent adversarial audit
The release candidate was audited by 109 independent reviewers across eight
dimensions, each finding then adversarially re-verified; 89 were confirmed.
Every one that does not require an owner's decision is fixed. The ones that
mattered most:
- **Redacted audio leaked on overlapping speech.** A turn counted as silenced
  if any silenced word overlapped it in *time*, and stereo turns overlap as a
  matter of course, so one speaker's masked identifier "covered" the other
  speaker's audible one. Silencing is now accounted per turn against the masks
  in the redacted transcript itself, so the audio can never be less silenced
  than the text is masked, whatever rule or model masked it.
- **`redaction.ner: true` did nothing** once presidio became opt-in, because NER
  lived inside presidio. It now works on its own, and a missing NER model is a
  hard stop at start-up rather than a quiet fallback.
- **Account numbers next to a currency word leaked** ("החשבון שלי 481902
  שקלים"). "בחשבון" (in the account) still introduces a balance; "החשבון" /
  "לחשבון" now introduce an identifier.
- **A customer naming themselves** ("קוראים לי ...", "שמי ...") is masked.
- **The air-gapped install**: the offline installer used a bare `pip` into the
  system Python, which RHEL lacks and PEP 668 forbids; it now builds a virtual
  environment and checks the bundle's Python version. Verified with the
  network cut: 393 tests pass on exactly the pinned versions.
- **CI tested a dependency set nobody deploys** (pyproject floors resolved
  against live PyPI). It now installs the pinned requirements.
- **`callqa eval` left raw transcripts in /tmp**, and its redaction metric could
  not see redaction being switched off. It now cleans up, and a new
  zero-tolerance metric checks the transcript the pipeline actually wrote.
- **Retention** now retires the original recordings `watch` kept in
  `input/processed/` and `input/failed/`, and crash-orphaned raw temp files;
  its window now ships in `config.yaml` instead of only as a code default.
- **Drift**: two of six signals had never collected a sample; the control band
  collapsed to a point on ordinary data; a one-call baseline was accepted.
- **The model downloader** silently skipped the judge model when the gated
  diarization step lacked a token, and turned every error into "not installed".
- **Preflight** checks model size against the download record (a truncated
  copy used to pass), the diarization cache where pyannote actually reads it,
  and says which input problem it found.
- `run --max-workers N` loaded each model N times; silero VAD ignored the
  sample rate; a reviewer's score was dropped after a rubric change; resumed
  runs invented per-stage timings; the judge reported a refused API key as an
  unreachable server.

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
- A third party named in passing, with no question and no self-naming phrase,
  is not masked by the rules. `redaction.ner: true` covers it, at a cost.
  Documented in `docs/DEPLOYMENT.md` §9.
- Editing the rubric does not re-score calls already processed: resume reuses
  stored scorecards. `callqa verify` reports which calls are stale; reprocess
  them with `--force`. A report for a call missing a current dimension now
  refuses with that instruction instead of failing obscurely.
- LICENSE names no copyright holder or client. That is the owner's decision.
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
