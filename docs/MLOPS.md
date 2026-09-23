# Operating callqa for years — the MLOps discipline

This document is the operational contract for running callqa in a bank over its
lifetime: how a result is made reproducible, how quality is measured, how drift
is caught, how a reviewer's judgement is fed back, how a batch is made safe to
start, how raw customer data is retired, and how a release is cut.

It is written to two constraints that shape every decision here:

1. **The core is frozen.** `process_call` and the stage artifact contracts in
   `models.py` do not change. Every capability below wraps the core at the
   driver level or by reading the artifacts the core already writes — the same
   way `dashboard/` reads them. New machinery lives in `src/callqa/ops/` and
   `src/callqa/eval/`; it imports the core, never the reverse, and never touches
   `dashboard/`.
2. **Every dependency is justified against the air gap.** The bank runs
   disconnected. No capability here adds a runtime dependency: each is a plain
   file (JSON / JSONL / CSV) plus the SQLite database that already exists, read
   and written with the standard library and the packages already declared
   (`pydantic`, `numpy`, `scikit-learn`). See *Why not MLflow / DVC* below.

---

## 1. What already exists (and is reused, not rebuilt)

A good deal of the discipline is already latent in the pipeline. The layer below
promotes and connects these rather than replacing them.

| Concern | Already present | Where |
|---|---|---|
| Judge provenance | prompt hash (over the real sent text), prompt version, rubric hash, model name, engine, n_samples, retries, judge latency, timestamp | `ScoreCard`, `models.py`; stamped in `judge/runner.py` |
| Rubric identity | content SHA-256 of `rubric.yaml` | `rubric.py` |
| Redaction state | `enabled` flag records whether redaction ran | `RedactedTranscript` |
| Per-stage timing | `stages.completed_at` per (call, stage) | `state.py` SQLite ledger |
| Judge–human agreement | QWK, MAE, confusion, human-vs-human, pass thresholds, min-calls, single-rubric guard | `calibration.py` |
| Human ratings contract | `call_id,rater_id,<8 dimensions>` CSV | `load_human_ratings`, `data/input/human_ratings.csv` |
| Speaker-separation scoring | DER, best-pairing, role accuracy, NIST collar | `scripts/eval_diarization.py` |
| Model inventory | `MODELS_MANIFEST.json` (id / path / size / downloaded_at) | `scripts/download_models.py` |
| Config safety | strict (typo-proof) sections, egress validation | `config.py` |
| Drift signal sources | score, gate, review status, redaction counts, role confidence, diarization alignment, transcription confidence — all persisted per call | `models.py` artifacts |
| Deletability | core never imports `dashboard/` | verified; guarded by a test |

What is genuinely absent: a code/config fingerprint on a result; model *weight*
hashing and a runtime reader of the manifest; a machine-readable run record; a
`verify` capability; a single evaluation command and any WER or redaction
precision/recall metric; **any drift monitoring at all**; a path for a reviewer's
verdict back into calibration; a preflight check; a retention policy; and release
discipline (the version has been `0.1.0` since the first commit).

---

## 2. Reproducibility — can we re-run a year-old call and get the same score?

A result is reproducible if everything that determined it is captured and can be
checked. The pipeline already stamps the prompt, prompt version, rubric hash,
model name and engine onto every `ScoreCard`. What is missing is the code
version, the effective configuration, and the model *weights* (a name is not a
fingerprint — a model can be re-trained under the same name).

**Environment fingerprint** (`ops/provenance.py`). A small record combining:
package `__version__`; the git commit SHA when a checkout exists, or `git:
none` on a tarball deploy (the air-gapped case); the Python version; the effective
`Config` (via `Config.model_dump`) and its SHA-256; the rubric hash; the prompt
version; and, per model, the weight hash from the manifest.

**Weight hashing** (`scripts/download_models.py`). The manifest is extended at
download time with a `sha256` per model (content hash of the weight files, which
are immutable once downloaded). Preflight verifies presence with a fast
size+mtime check by default and a full re-hash under `--deep`. Hashing multi-GB
weights on every run would be wasteful; hashing once at download and verifying
cheaply thereafter is the right trade for an air-gapped box.

**Run manifest** (`ops/runrecord.py`, `data/output/runs/<run_id>.json`). Written
by the drivers around a batch: the fingerprint, the config snapshot, the model
hashes, prompt/rubric hashes, start/end, per-call status and timing (end-to-end
plus per-stage, derived from the `stages.completed_at` diffs the core already
records), counts and failure reasons, and an estimated cost. A `runs/by-call/
<id>.json` sidecar links each call to the run that produced it.

> **Cost is measured as GPU-hours from timing × a configurable rate, not tokens.**
> On an on-prem vLLM the real cost is machine time, which timing already
> captures; token accounting would require reading usage inside the judge stage,
> a core change we deliberately avoid.

**`callqa verify <call_id>`** answers the two questions directly: is this stored
result still reproducible, and if not, *which input changed*. It reads the call's
run link and stored `ScoreCard` hashes, re-derives the current fingerprint, and
reports `reproducible` or names the divergence — rubric, prompt, config, model
weights, or code version — with a non-zero exit when something moved.

> **A known limitation, surfaced rather than hidden.** The core's resume keys off
> stage-done flags, not content hashes, so re-running a call whose rubric changed
> serves the *old* score unless `--force` is given. Making resume content-addressed
> would change core semantics, so we do not; instead `verify` makes the staleness
> visible and the operator re-runs with `--force`. See *Flagged core changes*.

---

## 3. Evaluation as a capability

Quality today is measured by running things and reading reports. This replaces
that with a golden reference set and one command that scores the whole system.

**Golden set** (`eval/golden/`, built by `scripts/build_golden_set.py`).
Synthetic and in-repo — it contains no real customer data, so it can live in
version control and run in CI. It is derived deterministically from the canned
dialogs in `asr/fixtures.py`, which carry known speaker labels and text (⇒
transcription and speaker-separation references) and a seeded, checksum-valid ID
and phone (⇒ redaction span references), together with the seeded human ratings
(⇒ judge-agreement references). A bank that wants real-world fidelity points
`callqa eval --golden-dir` at its own labelled set, kept out of the repo because it holds
customer data.

**Metrics** (`eval/metrics.py`; the DER scorer stayed in
`scripts/eval_diarization.py`, which the harness shells out to rather than
importing — moving it into the package was considered and not done, because it
needs an RTTM reference the synthetic set does not have):
- **Transcription** — WER and CER via a small standard-library Levenshtein
  (≈30 lines; adding `jiwer` for that is not justified air-gapped).
- **Speaker separation** — DER, best-pairing separation, role accuracy, reused
  from the existing scorer (now importable; the script becomes a thin wrapper).
- **Redaction** — recall, precision, F1 by running the real `redaction.find_pii`
  over the reference and comparing detected spans to the gold PII spans. Recall
  is the one that matters for a leak; precision guards against over-masking.
- **Judge agreement** — QWK, reused from `calibration.calibrate` against the gold
  human scores.
- **Timing** — end-to-end and per-stage.

**`callqa eval`** runs the pipeline over the golden set and writes
`data/output/eval/<timestamp>.json` plus a readable summary.

**Before/after (the MLOps part).** `callqa eval --baseline eval/baseline.json`
diffs every metric against a blessed baseline and fails (non-zero exit) on a
regression beyond tolerance. This is what turns "did the rubric/model/threshold
change help or hurt?" from a guess into a measurement, and it is wired as a CI
gate so a regression cannot land silently.

> The harness earned its keep immediately: building the golden set surfaced a
> **real redaction gap** — a mother's name given as a *standalone* answer ("what
> is your mother's name?" → "רות") is not masked, because `find_pii` only catches
> it inline ("שם האם הוא X"). It is a genuine leak path for the mono/turn-split
> case. It is tracked here rather than fixed under this MLOps work (redaction is
> core); it is not folded into the recall baseline so the baseline stays clean,
> but it is a candidate fix for the redaction module.

---

## 4. Drift — catching slow decay before anyone complains

This is the part that did not exist at all. In two years the telephony, the
recording format, or the mix of call types will change and the system will get
quietly worse. The signals that would reveal it are already computed and stored
per call; what was missing is aggregation over time and a justified alarm.

`callqa drift --set-baseline` captures reference distributions from a known-good
window into `data/output/drift/baseline.json`. `callqa drift` aggregates the
current window from the on-disk artifacts and flags movement. Each threshold is a
standard statistical limit, not a number picked by feel:

| Signal | Source | Test & threshold — and why |
|---|---|---|
| Score distribution | `ScoreCard.weighted_total` | **PSI** — <0.10 stable, 0.10–0.25 moderate, >0.25 significant. The population-stability index is the standard drift measure in bank model risk; the 0.25 cut is its conventional "material shift" line. |
| Review rate | `CallResult.status` | **Wilson 99% interval** on the baseline proportion at the current sample size. A proportion moving outside its binomial confidence band is a real change, not noise — the interval scales correctly with how many calls we have. |
| Identifiers masked / call | `RedactedTranscript.redaction_counts` | **Shewhart 3σ** control limit on baseline median ± MAD. A sudden drop can mean the redactor stopped seeing a format (a leak risk); a spike, a transcription change. 3σ ≈ 0.3% false-alarm rate — the classic control-chart limit. |
| Role-decision confidence | `DialogTranscript.role_confidence` | 3σ control limit. Falling confidence means speaker attribution is degrading. |
| Alignment health | `DiarizationQualityRecord.words_by_nearest / words_attributed` | 3σ control limit. A rising share of words attributed by nearest-neighbour fallback means transcription and diarization are drifting apart. |
| Transcription confidence | `TranscriptQuality.mean_logprob`, `low_confidence_ratio` | 3σ control limit. The earliest signal that the audio the bank is feeding the system has changed. |

MAD (median absolute deviation) rather than standard deviation makes the control
limits robust to the occasional pathological call. Each run appends to
`data/output/drift/history.jsonl`, a plain-text time series that trends without a
time-series database. `callqa drift` exits non-zero when any signal is flagged,
so an operator can wire it to whatever alerting the bank already runs.

> Note: three of these signals live in the RAW `transcripts/` directory, which
> holds PII and is permission-restricted. The drift reader inherits that
> restriction — it runs with the same access the pipeline itself has.

---

## 5. The feedback loop — a reviewer's judgement must not be thrown away

A call held for human review is the most valuable thing the system produces: a
human's verdict on a hard case. Today that verdict goes nowhere — the reviewer
reads the report and nothing is captured.

`callqa review-queue` lists the calls marked `needs_human_review` that have no
recorded verdict yet. `callqa review <call_id> --rater <id> --scores ...` records
the reviewer's per-dimension scores by appending a row to
`data/input/human_ratings.csv` — the exact format `calibrate` already consumes.
Every review therefore enlarges the calibration set, and the next
`callqa calibrate` measures the judge against more, and harder, human ground
truth. The loop closes with the tools already in the box.

Capture is a CLI command, deliberately **not** a dashboard POST. The dashboard
stays read-only and deletable; the feedback loop must survive `rm -rf dashboard/`.

---

## 6. Operational readiness — fail before the work starts

`callqa preflight` verifies, before a batch touches a single call, that:
- the configuration is valid (the strict loader already rejects typos);
- the required models are present and their weights match the manifest hashes
  (the runtime manifest reader that was missing) — and, for the dev-phase remote
  engines, that the endpoint is reachable (reusing the existing connectivity
  checks);
- there is enough free disk for the batch (`shutil.disk_usage` against an
  estimate of average artifact size × call count);
- the input `metadata.csv` and `human_ratings.csv` are well-formed (reusing the
  existing validators).

It reports each check and exits non-zero if any critical one fails, so a missing
model or a full disk stops the run at second zero rather than at call 400.

The **run manifest** (§2) is the machine-readable record that answers the
operational questions — how long the batch took, how many calls failed and why,
what it cost — without anyone reading a log.

---

## 7. Retention — raw customer data must not live forever

Raw transcripts and raw audio contain customer identifiers. The bank will require
an answer to how long they live and how they are destroyed.

The policy ships as configuration — `retention.raw_days` (default **90**) — so
each deployment can set its own window. `callqa retention --status` lists the raw,
PII-bearing artifacts older than the window (`transcripts/*.json`,
`transcripts/*.dialog.json`, `audio/wav/*`, `redacted_audio/*`). `callqa retention
--apply` destroys them and **keeps** the derived, non-PII outputs — redacted
transcripts, scores, reports, calibration — writing every deletion to
`data/output/retention/log.jsonl` as the audit trail. (On an SSD an unlink does
not guarantee the bytes are overwritten; the policy documents this and the bank's
storage layer, e.g. full-disk encryption, is the backstop.)

---

## 8. Release discipline

The version becomes meaningful: **1.0.0** marks the first release whose CLI and
artifact contracts are stable and supported, and `__version__` now flows into the
run manifest so every result records the code that made it. A `CHANGELOG.md`
(Keep-a-Changelog) records what each release changed. Releases are git-tagged.

Continuous integration (`.github/workflows/ci.yml`) runs the linter, the test
suite, the mock end-to-end run, and the evaluation harness against its baseline.
CI runs in the *development* environment to validate a release before it is
handed to the bank; it does not run inside the air gap, so it violates no
constraint — it is the gate that protects the bank from a bad release.

---

## Why not MLflow / DVC

Every concern above is met by a plain file plus the SQLite database that already
exist. Run manifests, eval and drift reports, the drift history, and the ratings
CSV are artifacts an air-gapped operator can open in a text editor, diff with
`diff`, and back up with `cp`. MLflow assumes a tracking server, a backing
database and a web UI the bank will not host; DVC assumes a remote store or
git-LFS workflow that reshapes the repository and its operations. Both pull in
large dependency trees and infrastructure assumptions to provide capabilities we
already get from the existing artifact-plus-state-DB pattern. The only genuine
computational need — QWK, PSI, a Wilson interval, a control limit — is covered by
the `scikit-learn` already present and a few lines of `numpy`. The lightest thing
that works here is a JSON file and the database that is already on disk, so that
is what this builds.

---

## Flagged core changes (considered, and deliberately NOT made)

Two capabilities would be marginally cleaner with a change to the frozen core.
Both have additive equivalents that ship instead; both are recorded here in case
the bank ever wants them.

- **Content-addressed resume.** Resume decides "already done" from stage-done
  flags, not content hashes, so a cached scorecard is not automatically
  invalidated when the rubric, prompt, or config changes — only `--force`
  recomputes. Making `_ArtifactStore` compare content hashes would auto-rescore
  on any input change, but it changes resume semantics the whole pipeline
  depends on. *Not done; `callqa verify` surfaces the staleness instead.*
- **Provenance stamped into the ScoreCard.** Writing the code version, config
  hash and model hash into the `ScoreCard` itself would make each result
  self-contained. It changes the judge stage's output contract, so *not done;
  the run manifest and the per-call run link carry the same information beside
  the scorecard.*
