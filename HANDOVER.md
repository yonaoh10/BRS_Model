# HANDOVER — Hebrew Call-QA PoC (`callqa`)

> **למי שקורא בעברית:** המסמך הזה נכתב בשביל Claude Code CLI שירוץ על המחשב שלך
> וימשיך את העבודה מהנקודה שבה הסשן הקודם עצר. הסשן הקודם רץ בסביבת ענן מבודדת
> **בלי GPU, בלי דמון Docker, ועם חסימת רשת** ל‑`huggingface.co` ול‑`runpod.io`,
> ולכן חלק מהדברים נבנו ואומתו רק בסימולציה. כל מה שאומת, כל מה שלא, וכל מה שנשאר
> לעשות כתוב כאן במפורש. הגוף באנגלית כי זו שפת העבודה של הסוכן.
>
> **מצב נכון ל‑2026‑09‑16:** קוד על `main` ב‑GitHub (`yonaoh10/BRS_Model`),
> ראש `480c092` (‏23 קומיטים מעל `6625eab`), **251 בדיקות עוברות** (+1 מדולגת),
> lint נקי. מאז ה‑14 בספטמבר: הורצו מודלים אמיתיים על הקלטה אמיתית מקצה‑לקצה,
> נבנתה ורצה תשתית הענן על **network volume**, ורצו **שני סבבי QA אדוורסריים
> מלאים** (סבב 2 ב‑15/9, סבב 3 ב‑16/9) שכל הממצאים שלהם טופלו או תועדו במפורש.
> ראו §12 לרשימת מה שחדש. הדבר היחיד שעדיין תלוי בזמינות ענן: אימות מלא של
> השיפוט על מודל 27B בכל פעם שצריך (המנגנון עובד; הריצה עצמה תלויה בזמינות GPU).

---

## 0. Read this first (for the agent)

You are continuing a project that is essentially complete and has been through
**three** adversarial QA cycles. The four gaps the very first session could not
close are now ALL closed — this is history, kept so you understand why the code
looks the way it does:

1. ~~**Run Docker**~~ — done on a Mac with a daemon (16/9). See §2/§12 for the
   disk incident and the 32 GB VM cap that resulted.
2. ~~**Reach huggingface.co / runpod.io**~~ — done. Real ivrit.ai ASR, pyannote
   diarization and a gemma-3-27b judge have all run on real audio.
3. ~~**Use a GPU**~~ — done, on a rented RunPod 4090; timings are measured, not
   estimated (`docs/performance_he.md`).
4. ~~**Finish QA round 2**~~ — done (redaction/pipeline/security/judge briefs),
   and a **round 3** ran on 16/9 against everything new since `6625eab`. All
   findings are dispositioned in `qa/round3-2026-09-16/SUMMARY.md`.

So there is no "first job" gap-list any more. What remains is incremental (§9
"lower priority" items) plus re-running the 27B judge whenever cloud GPU
capacity allows. **Read §12 first — it is the list of what changed since this
document was originally written, and it supersedes anything below that conflicts
with it.**

Do not push to GitHub unless the user asks. The user's standing choice is
to work directly on `main` (no PRs) when they do ask.

---

## 1. What the system is

A single-call **Hebrew banker–customer call QA pipeline** for a bank PoC.
Input: one recording + a metadata row. Output: a redacted transcript, objective
conversational features, an 8-dimension LLM-judged scorecard (0–100 with a
gate rule), and a Hebrew RTL HTML report. Batch drivers aggregate per banker
and calibrate the judge against human ratings (QWK).

The original build spec is at
`/root/.claude/uploads/d4d362d9-b739-597f-9bba-87f2de21d657/a33c7b28-callqapocspec.md`
in the old sandbox — it will NOT exist on your machine. The spec's key rules
are restated in §3; the README (bilingual EN/HE, byte-for-byte parallel) covers
the operator view.

**Hard constraints from the spec that shape everything:**

- Runs fully **offline** on the bank's server (`HF_HUB_OFFLINE=1`). Nothing
  downloads at runtime. Only `scripts/download_models.py` fetches anything.
- **Raw PII must never leave the redaction stage.** Not into the judge prompt,
  reports, logs, results JSON, dashboard, or filenames.
- The bank deliverable is **CLI + static HTML only**. A web app is out of
  scope — so the dashboard (`dashboard/`) and the cloud option (`cloud/`) are
  isolated, removable add-ons with verified clean removal.
- Exit-code contract for `process`: `0` success, `1` needs_human_review,
  `2` failed.
- Mock mode (`--mock`) must run the entire chain with no models, no GPU, no
  network. It does, in ~4 s for 6 calls.

---

## 2. Environment facts you must verify on the new machine

Before anything else, establish what you actually have:

```bash
docker info                      # daemon running?
nvidia-smi                       # GPU?
curl -sI https://huggingface.co | head -1
curl -sI https://rest.runpod.io | head -1
python3 --version                # 3.11+ required (3.10+ for pyannote 4)
ffmpeg -version | head -1        # required
```

The previous sandbox: Linux VM, 4 CPU, 15 GB RAM, Python 3.11.15, ffmpeg
6.1.1, Docker CLI 29.3.1 **without a daemon**, no GPU, pypi reachable,
huggingface/runpod/arxiv blocked. If your machine matches that, stop and tell
the user — nothing in §9(b)–(d) can proceed.

---

## 3. Architecture (what every file is for)

```
src/callqa/
  cli.py            process | watch | run | report | calibrate | validate-inputs
  pipeline.py       process_call(call, engines) -> CallResult. THE core. 8 stages,
                    per-stage JSON artifacts, resume via state DB, per-call lock,
                    review_reasons -> needs_human_review, never raises out.
  models.py         all pydantic artifacts (CallInput, CallMeta, AudioArtifact,
                    Transcript*, DialogTranscript, RedactedTranscript, Features,
                    ScoreCard, CallResult, STATUS_EXIT_CODES)
  config.py         pydantic config; StrictModel (extra=forbid); env overrides
                    CALLQA_<SECTION>__<KEY>; validate_endpoint() for judge/asr URLs
  resources.py      find_config(name): $CALLQA_CONFIG_DIR -> cwd/config -> repo/config
  dotenv.py         stdlib .env loader (project root, then cwd); shell wins;
                    write_env_values() used by cloud CLI (0600)
  engines.py        Engines DI container built ONCE per process; lazy imports
  state.py          SQLite: stages table, locks table (atomic steal + 6h TTL),
                    atomic_write_* (tmp+rename, 0600); connections closed
  ingestion.py      metadata.csv validation (encodings, ragged rows, path-safe
                    file_name, CALL_ID_RE), probe_audio (ffprobe + wave fallback),
                    sanitize_call_id()
  audio.py          16 kHz convert, channel split, probe_channels() (correlation-
                    based dual-mono / dead-channel detection, sampled across the
                    file), energy_vad (absolute floor + flat-level guard),
                    silero_vad (reads WAV itself; any failure -> energy fallback)
  asr/              faster_whisper_engine.py (ivrit.ai CT2), mock_engine.py,
                    fixtures.py (3 canned Hebrew dialogs; dialog 0 seeds PII),
                    remote_engine.py (cloud only), base.py
  speakers/
    stereo.py       merge_stereo (channel path) / assign_mono_roles (mono path)
    diarization.py  DiarizedSegment; reduce_to_two_speakers (+_merge_adjacent);
                    _SpeakerLookup (max-overlap per WORD, nearest fallback);
                    _smooth_islands (non-cascading); attribute_segments (splits
                    Whisper segments at speaker changes); join_words (handles
                    both "leading-space token" and "stripped token" conventions)
    roles.py        infer_roles: 7 weighted signals -> banker index + confidence
                    = |margin| / TOTAL_SIGNAL_WEIGHT (11.5). Signals: opening 3.0,
                    identity_request 2.5 (must be a REQUEST), identity_supply 2.0
                    (inverted; first utterance of each number only), question_rate
                    1.5, service_language 1.0, closing 1.0, first_speaker 0.5
    pyannote_engine.py  LazyPyannoteDiarizer: supports pyannote 3.x and 4.x
                    APIs, prefers exclusive_speaker_diarization, num_speakers=2,
                    local-path or HF-id model, device auto
    mock_engine.py  alternating 8 s speakers
  redaction.py      normalize_for_detection (drops bidi/zero-width, folds
                    Arabic-Indic/full-width digits, index map back); DIGIT_RUN_RE
                    with separators; _classify_run by checksum (Israeli ID,
                    Luhn, phone shape) + context (ID/phone/account/birth words
                    before, currency words after = amount, NOT masked);
                    EMAIL, IBAN, *short-code, mother's name, DOB; whole-dialog
                    detection with per-turn write-back (_CONTINUATION);
                    redact_names word-boundary; sanitize_error(); RegexRedactor
                    (loud when disabled -> enabled=False on artifact),
                    PresidioRedactor layered on top
  features.py       talk_ratio, longest_banker_monologue_sec, interruptions_*
                    (+ overlap_metrics_available flag, False on mono),
                    patience_median_sec, banker_question_count (count_questions
                    shared with roles.py), speech_rate_wpm, dead_air_total_sec
  judge/
    runner.py       retries incl. transport errors -> NeedsHumanReviewError;
                    _scrub() re-redacts model prose; median_low for n_samples;
                    long-call rule (gates on first 300 s; rest elided-middle)
    prompts.py      Hebrew system/user prompt; format_transcript with
                    _elide_middle (head 2/3 + tail 1/3 + marker); prompt_sha256
    validation.py   strict JSON; verify_evidence: per-turn containment, speaker
                    match, mm:ss timestamp within call, MIN_QUOTE_CHARS=8,
                    no empty evidence
    vllm_judge.py   OpenAI-compatible /chat/completions, bearer api_key
    mock_judge.py
  rubric.py         load_rubric (rejects duplicate ids / no gate; stamps sha256),
                    weighted_total (GATE_CAP=59 if any gate dim <= 2)
  calibration.py    QWK per dim + pooled; _safe_qwk -> None on constant data;
                    PASS = pooled>=0.70 AND mean-of-dims>=0.70 AND
                    n>=MIN_CALLS_FOR_PASS(20) AND no flagged dim (<0.60)
  aggregation.py    load_scorecards excludes non-success calls (reads results/)
  reporting/        Jinja2 autoescape ON; call_report (attribution banner +
                    signal table on mono), banker_report (safe_filename),
                    calibration_report, index; templates/*.j2 shipped in wheel
config/
  config.yaml, rubric.yaml (8 dims, gates: identification, compliance),
  recommendations_he.yaml, config.cloud.yaml (dev overlay: asr.engine=remote)
scripts/
  generate_sample_data.py   6 synthetic stereo WAVs + metadata.csv + human_ratings.csv
  download_models.py        --asr/--diarization/--llm/--ner; builds pyannote
                            pipeline once to warm HF cache; writes LOCAL PATH of
                            LLM into config (comment-preserving line edit)
  install_offline.sh / build_offline_bundle.sh   air-gapped install
  start_vllm.sh             requires CALLQA_JUDGE_API_KEY, binds 127.0.0.1
  prepare_real_call.py      two mono tracks -> stereo WAV + metadata row
  benchmark_asr.py          measures ASR realtime factor on THIS machine
  eval_diarization.py       DER / DER-best-pairing / role accuracy vs CSV or RTTM
  remove_cloud_option.py    deletes cloud/ + unwires 3 code refs + Makefile +
                            .gitignore + README marked sections; self-verifies
  docker_entrypoint.sh      seeds sample data if empty; run|watch|dashboard|shell
  first_run.sh              local or --docker one-command bring-up
dashboard/  (REMOVABLE: rm -rf dashboard/ tests/test_dashboard_server.py)
  server.py     stdlib HTTP, 127.0.0.1 (or --bind for container), token auth
                (CALLQA_DASHBOARD_TOKEN or random), Origin/Host checks, CSP,
                nosniff, reports served with is_relative_to guard
  prototype.html  RTL Hebrew console, 6 rail views, esc() everywhere
  qa_dashboard.py Playwright visual QA (contrast vs painted bg, overflow,
                  tap targets, bidi, focus ring, meter sticky, JS errors)
  DESIGN.md, tokens.css
cloud/      (REMOVABLE via scripts/remove_cloud_option.py)
  runpod_cli.py   up/status/urls/down/destroy via rest.runpod.io; writes
                  endpoints into .env; state file opened 0600
  asr_server.py   stdlib HTTP ASR server for the pod; bearer auth; 256 MB cap;
                  rejects truncated multipart
  bootstrap_pod.sh (vllm pinned), README.md (EN/HE)
docs/
  diarization_he.md         research + design of single-file speaker separation
  simulated_call_script_he.md  two Hebrew screenplays seeded with PII + behaviours
  performance_he.md         measured vs estimated timings
tests/  210 tests. test_qa_regressions.py = one test per QA finding fixed.
```

**Data flow:** `ingestion → audio → asr → speakers → redaction → features →
judge → report`. Artifacts under `data/output/{ingestion,audio,transcripts,
redacted,features,scores,results,reports}/`. `transcripts/` holds RAW text and
carries a README warning; everything else is redacted.

---

## 4. Timeline of what was done (so you know why things look the way they do)

1. **Core build** from spec: all 8 stages, mock mode, drivers, reports,
   calibration, tests, offline runbook. Pushed to `main` (user's choice, no PR).
2. **Bilingual README** (EN then HE, identical structure).
3. **Cloud dev option** (RunPod REST, no SSH): removable, verified removal.
   Sandbox could not reach RunPod → never exercised live. User once pasted a
   live RunPod key in chat; they were told to revoke it and did. **Never ask for
   secrets in chat; use `.env`.**
4. **Operator dashboard** in bank orange, WCAG-verified, Playwright QA, then a
   security pass (path traversal, focus ring, HTML escaping).
5. **Simulated call screenplay** for testing with a friend + `prepare_real_call.py`.
6. **User's real recording** (`Call_with______.m4a`, 3:05, AAC 44.1 kHz) turned
   out to be **dual-mono** (both channels bit-identical). This drove: channel
   probing, m4a acceptance, silero VAD fix (torchaudio/torchcodec), attribution
   banner in reports.
7. **User insisted single-file diarization is a core requirement** (it is).
   Research → chose `pyannote/speaker-diarization-community-1` (pyannote.audio
   4.x, CC-BY-4.0, offline, exclusive output). Rebuilt speakers stage:
   word-level attribution, extra-speaker folding, 7-signal role inference,
   confidence gate, audit trail in artifact + report, eval script.
8. **Adversarial QA round 1**: 7 agents (pipeline, redaction, speakers,
   ingestion, judge, security, deployment). ~100 findings. All fixed in
   `a48a5fd` + `a27be5e`. Round 1 reports are in the OLD sandbox only at
   `/tmp/.../scratchpad/qa/*.md` — **they will not exist on your machine**; the
   commit messages of those two commits summarise every fix.
9. **Bring-up work** (`6625eab`): `.env.example`, dotenv loader, compose,
   Dockerfile (non-editable install), entrypoint, `first_run.sh`. Simulated
   in a clean venv with sources removed: mock run 3.9 s, dashboard boots with
   .env token. **Never built as an actual image.**
10. **QA round 2** launched twice, killed both times by spend limits. **Not done.**

---

## 5. Key decisions and the reasons (don't re-litigate without cause)

- **Diarization model**: community-1 over 3.1 (3.1 = 19.9% DER on 2-spk in the
  independent benchmark, roughly 2× community-1's), over Sortformer (4-spk cap,
  English-first), over DiariZen (fine, but no exclusive output). Engine supports
  3.x API too.
- **Word-level attribution**, not segment-level: Whisper segments straddle turn
  changes; segment-level put one party's words in the other's mouth.
- **Role confidence normalised by total possible signal weight**, not fired
  weight: otherwise a call decided only by "who spoke first" reads 100%.
- **Interruption metrics flagged unavailable on mono**: exclusive diarization
  makes them structural zeros; a zero presented as a measurement is worse than
  "not available".
- **Amounts are NOT redacted** (currency word within 14 chars after a 6+ digit
  run): masking prices blinds compliance/clarity scoring. Everything else 6+
  digits contiguous is masked (privacy bias).
- **Redaction disabled = held for review**, loud ERROR log, `enabled=False` on
  artifact. Never a silent pass-through.
- **Calibration PASS needs 20 calls** and per-dimension agreement; kappa on
  constant data is `None`, never 1.0.
- **Config is strict** (`extra="forbid"`); endpoints must be http(s), and
  plaintext http only to loopback.
- **call_id charset** `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` enforced in
  `CallInput` and metadata; filenames sanitised via `sanitize_call_id`.
- **Dashboard/cloud stay out of the `callqa` package** so removal is a delete.
- **Colour**: brand orange `#ea580c` is chrome only, never status (ΔE 2.9 vs
  warning under deuteranopia); primary button `#c2410c` (5.18:1).

---

## 6. What is VERIFIED vs UNVERIFIED (be honest with the user)

**Verified in the sandbox (real execution):**
- 210 tests, ruff clean, full mock E2E (6 calls → 11 HTML reports) ~4 s.
- Non-editable wheel install runs the pipeline from another directory
  (templates ship; config resolves via `resources.find_config`).
- Cloud removal and dashboard removal in throwaway copies: tests pass,
  mock E2E passes, zero code references (one prose line in
  `docs/performance_he.md` remains and is reported by the script).
- Dashboard: Playwright QA 0 findings across light/dark × desktop/phone;
  token/Origin/Host/traversal tests; CSP/nosniff headers; `--bind 0.0.0.0`
  simulated container path.
- Silero VAD runs offline (weights in wheel), 3.0 s for a 3-min file on 4 CPUs.
- Channel probing correctly classifies the user's dual-mono m4a.
- Every QA round-1 fix has a regression test.

**Unverified (never executed anywhere):**
- faster-whisper with the ivrit.ai model; pyannote community-1 diarization;
  vLLM judge; presidio path. All lazy-imported and unit-tested with fakes only.
- `docker build` / `docker compose up` for real.
- RunPod round trip (`runpod_cli.py up` → bootstrap → `run --config
  config/config.cloud.yaml`).
- `scripts/install_offline.sh` on a genuinely air-gapped box (was run in a
  scratch copy with network).
- Real-audio accuracy: DER, role accuracy, redaction recall on real ASR output
  (numbers-as-words is a known risk), judge quality.
- Timing on GPU/CPU for ASR and judge (only estimates in
  `docs/performance_he.md`).

---

## 7. Secrets and safety rules (standing, from the user)

- Secrets only via `.env` (template: `.env.example`) or environment. **Never in
  chat, never committed.** `.env` is gitignored; loader never logs values.
- Real customer recordings never go to a public cloud. The dev phase uses
  synthetic or consented recordings only (the user's own staged call is fine).
- The RunPod proxy URL is public; the pod must be stopped when idle
  (`python cloud/runpod_cli.py down`). `status --warn-hours` exists.
- Bank vLLM must run with `--api-key` (enforced by `start_vllm.sh`).

---

## 8. How to run (quick reference)

```bash
# one-command, local python
cp .env.example .env && ./scripts/first_run.sh
# one-command, docker
./scripts/first_run.sh --docker          # == docker compose up --build
# dashboard: http://127.0.0.1:8765/?t=<CALLQA_DASHBOARD_TOKEN from .env>

# manual
pip install -r requirements.txt && pip install -e .
python scripts/generate_sample_data.py
python -m callqa run --mock && python -m callqa report --mock && python -m callqa calibrate --mock
python -m callqa process --audio path/to/call.m4a --call-id REAL001 --banker-id B900   # real
python dashboard/server.py
python dashboard/qa_dashboard.py

# real models (needs HF_TOKEN in .env and accepted model conditions)
pip install -r requirements-server.txt      # torch 2.8, pyannote.audio 4.0.7, faster-whisper...
python scripts/download_models.py --asr --diarization
python scripts/download_models.py --llm --llm-model dicta-il/dictalm2.0-instruct
CALLQA_JUDGE_API_KEY=... ./scripts/start_vllm.sh <local model path>
python -m callqa run

# cloud GPU (RunPod)
python cloud/runpod_cli.py up      # writes CALLQA_* endpoints into .env
# then bootstrap on the pod per cloud/README.md step 5
python -m callqa run --config config/config.cloud.yaml
python cloud/runpod_cli.py down

# measure
python scripts/benchmark_asr.py --audio data/input/calls/REAL001.wav
python scripts/eval_diarization.py --dialog data/output/transcripts/REAL001.dialog.json --reference refs/REAL001.csv

make test | make lint | make mock-e2e | make up | make dashboard
```

---

## 9. What to do next, in order

### (a) Finish QA round 2 — verify the round-1 fixes and break the new code
Run four read-only adversarial reviews (or do them yourself) against
`git diff 6c50ac8..HEAD`. Briefs, verbatim intent:

1. **Redaction** (`src/callqa/redaction.py`, `judge/validation.py`,
   `judge/runner.py`): property-test the index map (invisible chars at
   edges/inside/only, emoji, combining marks, non-ASCII digits) — output must
   equal original minus masked spans; cross-turn spans over 3 turns / empty
   middle turn / exact-boundary; false negatives (digit→letter ASR errors,
   `+972 5x`, IBAN spacing, card split across turns, numbers interleaved with
   words); false positives (can the amount exception hide `החשבון 481902
   שקלים`? dates, times, %, years); performance on 2 h transcript and 100k
   alternating digit/separator chars; `_scrub` order vs verification;
   `sanitize_error` truncating mid-mask; END-TO-END grep of every artifact,
   log and `/api/state` after a run with a stubbed PII-laden ASR fixture.
2. **Pipeline/CLI** (`pipeline.py`, `state.py`, `cli.py`, `config.py`,
   `resources.py`, `ingestion.py`, `dotenv.py`): `_ArtifactStore.load`
   semantics (right call_id wrong contents; earlier stage recomputed while
   later still marked done → stale downstream?); lock steal race with barrier
   + dead pids + foreign hostname + future `acquired_at`; `release_lock`
   cannot release another's lock; `_call_input` matching (two rows same
   file_name; call_id ≠ stem; `--call-id` mismatch; `run` exit on invalid
   metadata); `watch` (file grows during 1 s re-stat, deleted during it,
   dir/symlink, `--once` idle, extension set vs
   `ingestion.FFMPEG_AUDIO_EXTENSIONS`); review-reason thresholds
   (`MIN_CALL_SECONDS=20`, `MIN_SPEECH_SECONDS=10`) holding legit short calls;
   strict config still loads both yaml files and every `CALLQA_*` override;
   `find_config`/`load_dotenv` from /tmp, subdir, bogus `CALLQA_CONFIG_DIR`,
   empty `config/`, installed wheel; hostile-cwd `.env` injection.
3. **Speakers/judge** (`speakers/*`, `features.py`, `judge/*`,
   `calibration.py`, `rubric.py`, `aggregation.py`): `join_words` round-trip
   against BOTH ASR engines (must equal raw text); `_smooth_islands` vs
   brute-force reference (randomised); nearest-fallback O(n) timing at 30k
   words/6k segments; `_merge_adjacent` across a different speaker; role
   inversion hunt (outbound sales call, customer who is a bank employee,
   agent-to-agent transfer, customer quoting script, near-silent banker);
   `overlap_metrics_available` honest in report AND judge prompt
   (`format_features`); `banker_index` tie cases + old artifact without
   field; `verify_evidence` false rejects (quote spanning `<ת"ז:████>`,
   niqqud, quote in both speakers' turns, >999-min call) and false accepts;
   fake OpenAI endpoint driving 500/429/401/empty/hang/`content:null`/10 MB →
   bounded attempts and wall time; `_elide_middle` edge cases; median note
   not leaking/duplicating; calibration arithmetic by hand; rubric sha256
   tolerated by all consumers.
4. **Security/handover** (`dashboard/`, `cloud/`, `scripts/`, Dockerfile,
   compose, `.env.example`, `dotenv.py`, `config.py`, `audio.py`): re-run all
   dashboard attacks incl. `--bind 0.0.0.0` + `ALLOWED_HOSTS` from a host
   browser, percent-decoding traversal variants, token leakage; `.env`
   precedence/symlink/unreadable/huge/`=`-in-value; every attacker-text→path/
   HTML sink; `validate_endpoint` bypasses (userinfo, IPv6, https to attacker
   — intended); shell review of all scripts incl. `"$@"` in entrypoint;
   compose semantics: `depends_on: service_completed_successfully` — **if the
   mock batch exits 1 (needs_human_review) the dashboard never starts; this
   is a likely real bug, check it first**; `./config` rw mount masking image
   copy; `env_file: required: false` needs compose ≥ 2.24; asr_server body at
   cap / chunked / boundary in data; permissions after a run; wheel in clean
   venv from another dir; both removals; README runbook walked literally.

Fix everything reproduced, add a regression test per fix to
`tests/test_qa_regressions.py`, keep `make test && make lint` green.

### (b) Docker for real
`docker compose config` → `docker compose build` → `docker compose up`. Expect
to hit: the `depends_on` issue above (calibration FAIL on 6 calls exits
non-zero from `calibrate`, but the pipeline service runs `run --mock` which
exits 0 on success — verify); the `./config` mount; proxy/CA if the machine
uses one. Then `docker compose run --rm pipeline process --audio ...` on the
user's real m4a.

### (c) Real models on real audio
1. User accepts conditions at
   `https://huggingface.co/pyannote/speaker-diarization-community-1`, puts
   `HF_TOKEN` in `.env`.
2. `pip install -r requirements-server.txt` (CPU torch is fine for a first run;
   pins: torch 2.8.0, torchaudio 2.8.0, torchcodec 0.7.0, pyannote.audio
   4.0.7, faster-whisper 1.1.1, ctranslate2 4.5.0). Watch for the
   torchaudio/torchcodec/ffmpeg-lib interaction; `audio.py` avoids torchaudio
   for VAD, but pyannote may still need torchcodec working.
3. `python scripts/download_models.py --asr --diarization`. Confirm the HF
   cache warm step actually loads the pipeline (that code is untested).
4. Run the user's staged call (they have `Call_with______.m4a`, dual-mono) →
   check: transcript is Hebrew and numbers come out as **digits** (if as
   words, redaction cannot catch them — this is the single biggest known
   risk; consider a Hebrew number-word normaliser before redaction);
   diarization produces 2 speakers; role confidence; report banner.
5. Judge: either rent GPU via RunPod path, or point `judge.base_url` at any
   OpenAI-compatible API over https (allowed by `validate_endpoint`; the
   redacted transcript is what leaves). `response_format: json_object` is sent
   — some providers reject it; `vllm_judge.py` may need that made optional.
6. Score the diarization: write `refs/REAL001.csv` (`start,end,speaker`) by
   listening, run `eval_diarization.py`. Targets: DER < 15%, role acc > 95%.

### (d) Measure and document
`benchmark_asr.py` on the user's machine; replace the estimates in
`docs/performance_he.md` with measurements. Update `docs/diarization_he.md`
with real DER numbers.

### Known open items not yet addressed (lower priority)
- `n_samples>1` at temperature 0 sends identical requests (no `seed`/`n`).
- No global per-call deadline on judging (worst case retries × timeout).
- `watch` has no dashboard surface; two processes on one state DB contend.
- Dashboard actions (`--allow-actions`) not implemented; Run/Cloud screens in
  `DESIGN.md` are design only.
- Validation messages are English; intake UI promises Hebrew (needs
  structured problem codes in `ingestion.py`).
- No re-run surface after rubric/prompt change (`prompt_sha256`,
  `rubric_sha256` exist on scorecards; nothing lists stale calls).
- Israeli ID whose first two digits look like a landline area code
  (02/03/04/08/09) is masked but labelled `<טלפון>` unless an ID context word
  precedes it.
- `_read_wav` loads whole files (3 h mono ≈ 1.7 GB RSS); streaming VAD would fix.
- Two open questions for the user: does `#ea580c` match the bank's real brand
  orange; does the dashboard Overview show the right three things first.

---

## 10. Test/quality gates you must keep

```bash
python -m pytest -q                       # 210 passing at handover
ruff check src tests scripts dashboard cloud
python dashboard/qa_dashboard.py          # 0 findings (needs playwright + chromium)
python scripts/remove_cloud_option.py     # dry run must print a clean plan
```

Every bug fix gets a test in `tests/test_qa_regressions.py`, written as the
reproduction it came from, with a docstring saying what it defends against.
Commit messages in this repo explain *why* at length; keep that style.

---

## 11. Files the previous session left uncommitted

At the original handover the working tree was clean except this file
(`HANDOVER.md`), deliberately not committed. It is still untracked by choice.

---

## 12. What changed since this document was first written (2026‑09‑14 → 09‑16)

This section supersedes any older text above that conflicts with it.

**Redaction — Hebrew number‑word normaliser (the biggest known hole, closed).**
`src/callqa/redaction.py`: `fold_number_words` rewrites runs of ≥4 Hebrew
digit‑words ("ארבע חמש שמונה אפס") to digits before detection, with an
index‑map back so masks land on the original words; `find_pii` composes
`normalize_for_detection → fold_number_words → _spans_in_normalized` and maps
back. On the real call this took masked identifiers from 4 to 8 (the four new
catches were previously invisible). Round‑3 hardening on top: the amount
exception no longer suppresses 9+ digit or explicitly‑labelled account numbers
near a currency word; `normalize_for_detection` strips Unicode Cf/Mn/Me (niqqud
inside a number no longer splits it); `_repeated_fragments` matches a suffix
read‑back, not any substring, and skips amounts.

**Judge — snap safety.** The 0.90 quote‑snap (`judge/validation.py`) now also
requires numbers and negations to be preserved verbatim: a snap that would flip
a fee or drop a "לא" is rejected and the call goes to human review. Prose is
tag‑stripped in `_scrub`; the request timeout is now
`JudgeConfig.request_timeout_sec`.

**Pipeline — ASR ∥ diarization.** On a mono call, diarization runs in a
SEPARATE PROCESS (`speakers/diar_worker.py`) alongside ASR — separate because
ctranslate2 and torch each bundle libiomp5 and cannot co‑reside on Intel macOS
(see the ct2‑before‑torch eager load in `asr/faster_whisper_engine.py`, and the
SIGSEGV crash report in `qa/rescued-2026-09-15/`). Measured ~401 s → ~347 s.
Toggle: `speakers.parallel_diarization`. The worker has a parent‑death watchdog.
`_ArtifactStore` now forces every later stage to recompute once any stage is
recomputed (no stale downstream). `state.release_lock` deletes only our own
lock. `Config` is a `StrictModel` (a typo'd section fails loudly); the env
parser ignores CALLQA_* vars without a `__` section.

**Reports / dashboard — the full transcript.** Both the static per‑call report
and the dashboard detail drawer now show the whole REDACTED transcript (RTL,
speaker labels, mm:ss, mask tokens visible, judge evidence highlighted in
place). The dashboard serves it from `/api/transcript/<call_id>` (NOT
`/api/state`, which stays pollable), token+Origin+Host gated, `redacted/` only,
size‑capped. `qa_dashboard.py` = 0 findings.

**Packaging.** The wheel now ships `callqa/config_defaults/*.yaml`, so an
installed (non‑editable) wheel is self‑contained; a repo‑root `config/` or
`CALLQA_CONFIG_DIR` still wins. A test guards the two copies against drift.

**Cloud — network volume (host‑independence).** `cloud/runpod_cli.py` no longer
uses a pod‑local volume (host‑pinned, the "not enough free GPUs on the host
machine" wall). `up` creates a DATA‑CENTRE network volume and a pod as a unit,
walking `CANDIDATE_DATA_CENTERS` until one has a free GPU, and `--gpu` accepts a
comma‑separated list of acceptable cards. The volume persists across `destroy`
(only the pod is disposable); `destroy --volume` removes it; `status`/`down`
print the standing storage cost ($0.07/GB‑mo, 80 GB = $5.60/mo). `up`/`down`/
`destroy`/`status` treat a 404'd pod as gone and recover. **Private‑repo note:**
the pod cannot `git clone` a private repo, so its bootstrap prints a message to
rsync the tree and serve by hand; that is the path used to serve the 27B judge.

**Perf doc** (`docs/performance_he.md`) carries measured local diarization
(pyannote‑3.1, 228.5 s, ~221 s in embeddings), the ASR variation matrix, the
concurrency result, and states plainly that this Intel Mac runs 3.1 (≈2× the
DER of community‑1) only because of the torch‑2.2.2 pin — the bank's Linux
server must use community‑1.

**REAL002** now stores real ASR + real pyannote‑3.1 diarization +
number‑word‑normalised redaction; only the judge stage is mock (56.2) until the
gemma re‑run lands (`cloud/rejudge_real002_gemma.{md,sh}`). The one fully‑valid
real scorecard (gemma‑3‑27b, 82.5) is preserved in
`qa/rescued-2026-09-15/real002_gemma2.log`.

**Rescued QA material** from the 09‑15 crash is under `qa/rescued-2026-09-15/`;
round‑3 findings and dispositions are under `qa/round3-2026-09-16/`.
