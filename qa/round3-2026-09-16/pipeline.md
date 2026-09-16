# Round 3 Adversarial QA — Pipeline (2026-09-16)

Reviewer: adversarial read-only pass. Repo: /Users/yonatanohayon/Desktop/BRS_Model-main
Commit reviewed (HEAD): 618f756 (see `git rev-parse HEAD`). runpod_cli.py may change concurrently (owner editing for network-volume migration).
Python: /Users/yonatanohayon/Desktop/BRS_Model-main/.venv/bin/python

Findings appended incrementally the moment each is confirmed. Most-severe first is imposed in the final SUMMARY reorder; raw findings are in discovery order below.

## SUMMARY (most-severe first)
Counts: 3 HIGH, 5 MEDIUM, 6 LOW, 1 INFO.

| ID | Sev | One-line | Where | Verified |
|----|-----|----------|-------|----------|
| F1 | HIGH | `release_lock` deletes any owner's lock → double-processing after a lock steal | state.py:216 | repro RAN |
| F2 | HIGH | RunPod `up` creates+bills pod before saving state / no-id → stranded invisible pod | runpod_cli.py:181 | code + repro sketch |
| F15 | HIGH | Installed wheel omits `config/*.yaml`; every command fails outside a dir with `config/` | pyproject.toml:33 | wheel built+installed, repro RAN |
| F3 | MED | `down` claims "billing has ended" without confirming the stop | runpod_cli.py:260 | code |
| F4 | MED | RunPod state deadlock when pod removed out-of-band (404 before state clear) | runpod_cli.py:148,278 | code |
| F6 | MED | Broken metadata.csv silently → default channel L (inverts stereo speakers), no redaction, in watch/process | cli.py:82,96 | code |
| F7 | MED | Deleting one stage artifact leaves later stages serving stale downstream data | pipeline.py:98 | code |
| F8 | MED | Parent SIGKILL orphans the diar worker (holds GPU) + leaks .diar.json/.log | pipeline.py:447 | code |
| F5 | LOW | No proactive billing guard; running-hours is local-clock only | runpod_cli.py:241 | code |
| F9 | LOW | Worker `[]`/exit-0 consumed as real diarization, skips in-process fallback | pipeline.py:196 | code |
| F10 | LOW | darwin eager CT2 load fires on resumed runs too (contradicts docstring) | faster_whisper_engine.py:54 | code |
| F11 | LOW | ct2-before-torch rests on a silent `"torch" not in sys.modules` guard; no assert | faster_whisper_engine.py:54 | code |
| F12 | LOW | Config: nested typo hard-fails but top-level section typo silently ignored | config.py:172 | code |
| F13 | LOW | Hostile-cwd `.env` can redirect egress to any https host (validation allows) | dotenv.py:30 | code |
| F14 | INFO | Duplicate `WATCHED_EXTENSIONS` definition (dead copy/paste) | cli.py:32 | code |

Resume correctness (task #4) verified sound (see Checked and sound). runpod reviewed at committed HEAD 618f756; working tree is mid network-volume migration (see Notes/Scope).

---

## FINDINGS (discovery order)

### F1 [HIGH] release_lock deletes ANY owner's lock — breaks per-call atomicity after a lock steal
- file:line: `src/callqa/state.py:216-218` (`release_lock`) vs `acquire_lock` steal at `state.py:195-211`.
- What: `release_lock` runs `DELETE FROM locks WHERE call_id=?` with NO pid/hostname predicate. It removes whatever lock exists for that call_id, even one another process now owns.
- Repro (logical, deterministic):
  1. Process A acquires lock for call X (row: pid=A). A stalls (e.g. slow model, paused) so its lock ages past `LOCK_TTL_SECONDS` (6h) — OR A dies and a same-host pid is reused, but the simplest path is age.
  2. Process B calls `acquire_lock(X)`, sees `stale_by_age`, and steals via the conditional UPDATE (row now pid=B). B starts real work on X.
  3. A is still alive (it was only stale-by-AGE, never confirmed dead) and eventually hits `finally: state.release_lock(X)` in `pipeline.process_call` (`pipeline.py:451`). That DELETE removes B's lock.
  4. Now X is unlocked while B is mid-processing. Process C `acquire_lock(X)` succeeds cleanly → X is processed twice concurrently. The module docstring's "per-call atomicity ... prevents double processing" (pipeline.py:9, state.py header) is violated.
- Also triggers on the stale-by-death branch when a same-host PID is recycled: B steals A's lock believing A dead; a *different* process reusing pid=A can never happen to release, but ANY holder of the DELETE path clobbers cross-owner. The unconditional DELETE is the core defect.
- Fix direction: `DELETE FROM locks WHERE call_id=? AND pid=? AND hostname=?` with the caller's own identity (mirror the guarded UPDATE in acquire).
- REPRO RAN (scratchpad/lock_test.py): seed an aged foreign-pid lock; our process steals it by age; a fresh foreign owner re-takes it; our `release_lock("X")` DELETES that fresh lock (`lock row = None`); a new process then `acquire_lock("X")` succeeds cleanly. Output: "new process acquired X cleanly -> DOUBLE PROCESSING possible".
- Why it matters: double-processing corrupts shared per-call artifacts/state, wastes GPU, and can publish two divergent scorecards for one call. This is the exact invariant the lock exists to hold.

### F2 [HIGH] RunPod `up`: pod is created (billing starts) BEFORE state is saved — a strand window
- file:line: `cloud/runpod_cli.py:181-190` (committed HEAD 618f756).
- What: `pod = _request("POST", "/pods", payload)` actually creates the pod and STARTS BILLING. `pod_id` is then read (line 182); if the response has neither `id` nor `podId`, line 184 raises `RunPodError` and `save_state` (line 190) never runs. Any exception (or SIGKILL) between the successful POST and `save_state` leaves a live, billing pod with NO local record.
- Repro: stub `_request` so `POST /pods` returns `{"ok": true}` (no id key) — the committed code raises "pod created but no id in response"; `.runpod_state.json` has no `pod_id`; `down`/`destroy` both print "No pod recorded; nothing to stop/destroy." If RunPod really created the pod, it bills indefinitely and is invisible to this CLI.
- Why it matters: this is the exact OVERBILL/STRAND class the file's COST SAFETY banner promises to prevent. RunPod's own key for the pod id can drift (`id` vs `podId` is already hedged, implying uncertainty). Should persist pod_id from the response immediately, and reconcile by name (GET /pods?name=) on the next `up`/`down`.

### F3 [MEDIUM] RunPod `down` prints "GPU billing has ended" without confirming the stop took effect
- file:line: `cloud/runpod_cli.py:260-264`.
- What: `_request("POST", f"/pods/{pod_id}/stop")` then unconditionally prints "Pod stopped. GPU billing has ended." It never re-GETs `desiredStatus` to confirm the pod left RUNNING. A 200 that queues but does not complete the stop, or a pod that fails to stop, still yields the reassuring message.
- Repro: stub stop to return `{}` while a follow-up GET still shows RUNNING — the operator is told billing ended when it has not. Fix: re-poll status after stop and only claim success on a non-RUNNING desiredStatus.
- Why it matters: the false confirmation is precisely what lets a pod keep billing after the operator believes they stopped it.

### F4 [MEDIUM] RunPod state deadlock when the pod was removed out-of-band (console / expiry)
- file:line: `cloud/runpod_cli.py:148-155` (up), `268-281` (destroy), `254-265` (down).
- What: all three read `state["pod_id"]` and immediately `_request` against `/pods/{id}`. If the pod no longer exists on RunPod (terminated in console, or expired), that request 404s -> `RunPodError` -> `main` returns 2 BEFORE `STATE_FILE.unlink()` (destroy, line 279) ever runs. State is never cleared, so `up` keeps GET-404ing the ghost pod instead of creating a new one, and `destroy` keeps 404ing instead of clearing state. Recovery requires hand-deleting `.runpod_state.json`.
- Repro: put a bogus `pod_id` in `.runpod_state.json`, run `up`/`down`/`destroy` -> each returns 2, state persists. Fix: treat 404 on the pod as "already gone", clear state, and (for up) fall through to create.
- Why it matters: locks the operator out of the normal lifecycle; the natural next move (`destroy`) cannot even clean up.

### F5 [LOW] RunPod: no proactive billing guard; running-hours estimate is local-clock only
- file:line: `cloud/runpod_cli.py:241-248`, `189`.
- What: the "running longer than --warn-hours" warning only fires when a human runs `status`. Nothing polls. A forgotten pod bills silently until someone chooses to check. Also `hours` is derived from `state["last_started_at"]` (local wall-clock set at `up`), so a console-side stop/start or RunPod auto-restart desyncs the estimate from real billed time.
- Why it matters: answers the brief's "can a pod be left running silently?" — yes; the only meter-stop (`down`) and the only warning (`status`) are both manual.

### F6 [MEDIUM] Broken metadata.csv silently degrades `process`/`watch` to default attribution (channel L can invert speakers)
- file:line: `src/callqa/cli.py:82-93` (`_metadata_lookup`) + `96-137` (`_call_input`).
- What: `_metadata_lookup` logs an error and returns `{}` when `metadata.csv` fails validation. In `cmd_process` and `watch_loop`, `_call_input` then finds no row and builds a `CallInput` with defaults — "banker unknown, channel L, banker name not redacted" (its own warning text, cli.py:124-127). For a STEREO call, defaulting `banker_channel` to L when the banker is actually on R INVERTS the two speakers, and stereo attribution reports high confidence, so the inverted call is published as `success` (no human-review flag). Only `cmd_run` hard-fails on invalid metadata (cli.py:175-177); the production `watch` driver does not.
- Repro: drop a call whose metadata row is present but the CSV has one malformed row (e.g. a duplicate call_id elsewhere) -> whole `validation.ok` is False -> every call in that watch cycle processed with default channel L and no banker-name redaction.
- Why it matters: silent role inversion on real bank calls plus un-redacted banker name — a correctness AND privacy regression that never reaches a human.

### F7 [MEDIUM] Cross-owner stale downstream: deleting/recomputing one stage's artifact leaves later stages serving stale artifacts
- file:line: `src/callqa/pipeline.py:98-121` (`_ArtifactStore.load`) + `state.py:123-135`.
- What: `load()` only revalidates a stage against its OWN artifact (parse + call_id match). It has no notion of upstream freshness. If the `asr` artifact is missing/removed (is_done->False) it recomputes ASR, but if `speakers` is still marked done with an existing `.dialog.json`, `load("speakers")` returns the OLD dialog built from the OLD ASR. Redaction/features/judge/report then all run on a dialog inconsistent with the freshly recomputed transcript.
- Repro: run to completion (mock ok), delete only `data/output/transcripts/<id>.json` (ASR bundle), keep `<id>.dialog.json`, rerun without --force -> ASR recomputed, speakers/dialog stale, downstream keyed off stale dialog. No warning.
- Why it matters: partial artifact loss (disk cleanup, rsync, a stage-specific bug) yields a silently inconsistent scorecard rather than a recompute or an error. --force is the only safe recovery and nothing tells the operator that.

### F8 [MEDIUM] Orphaned diarization worker + leaked temp files when the PARENT is SIGKILLed
- file:line: `src/callqa/pipeline.py:447-451` (finally/cancel) + `speakers/diar_worker.py:32-55`.
- What: the parent cleans up the worker only via `finally: early_diar.cancel()`. A `kill -9` of the parent between `_EarlyDiarization.start` and `collect` cannot run the finally, so: (a) the worker (pyannote/torch, ~228 s on the dev box) is orphaned and keeps holding CPU/GPU to completion; (b) `<wav>.diar.json` and `<wav>.diar.log` are leaked; (c) the worker has NO parent-death watchdog (it never reads stdin again and there is no PDEATHSIG — unavailable on macOS anyway). On the next run of the same call the SQLite lock is stolen fast (parent pid dead, same host), a NEW worker spawns writing the SAME `<wav>.diar.json`, so briefly two diarizers run against one card.
- Repro: `kill -9` the `process` PID during the ASR stage of a mono call; observe the `python -m callqa.speakers.diar_worker` child still running (`ps`) and the two temp files remaining next to the wav.
- Why it matters: silent GPU/CPU occupancy after an operator thinks they killed the job, plus temp-file accumulation. (Corruption is NOT possible: both writers use `atomic_write_text` tmp+rename with unique mkstemp names — see Checked and sound.)

### F9 [LOW] Empty worker output (`[]`, exit 0) is consumed as a real diarization, skipping the in-process fallback
- file:line: `src/callqa/pipeline.py:196-207` (`collect`) + `pipeline.py:331-337`.
- What: `collect` returns `segments` (possibly `[]`) whenever the worker exits 0 with valid JSON. Only `None` triggers the in-process fallback. A worker that exits 0 having written `[]` yields `diarized == []`, which is `not None`, so `assign_mono_roles(call_id, bundle.mono, [])` runs on zero speaker turns instead of retrying in-process. A legitimately-empty pyannote result is indistinguishable here from a silent worker degradation that produced no segments.
- Why it matters: low, because in-process diarization on the same audio would likely also yield empty; but the design intends the worker to be "best-effort with an in-process fallback", and empty-but-exit-0 defeats that fallback. Worth an explicit `if not segments: return None` guard.

### F10 [LOW] darwin eager CT2 load fires on EVERY fresh process, including resumed runs where ASR is already done
- file:line: `src/callqa/asr/faster_whisper_engine.py:54-55` vs the `model` property docstring `58-66`.
- What: `if sys.platform == "darwin" and "torch" not in sys.modules: _ = self.model` runs at construction unconditionally on a fresh darwin process (torch is never resident that early). So a resumed run whose `asr` stage is complete — which the `model` docstring says "never calls transcribe()" and must not hold the ~1 GB CT2 model — DOES eagerly load it anyway. The two comments contradict; the deferred-load memory saving the property advertises does not apply on darwin.
- Why it matters: darwin-only wasted ~1 GB on resumes (dev machines only; the Linux bank target never takes the branch). Not a correctness bug, but the code comment is misleading about what happens.

### F11 [LOW] ct2-before-torch ordering rests on a fragile `"torch" not in sys.modules` guard with no assertion
- file:line: `src/callqa/asr/faster_whisper_engine.py:54`, exercised by `engines.build_engines:57-71` and the in-process diar fallback at `pipeline.py:335-337`.
- What: survival on Intel macOS depends on CT2 claiming OpenMP before torch. The only enforcement is the construction-time guard. Today the order holds (build_engines constructs FasterWhisperEngine at engines.py:59 before anything imports torch; redactor/presidio and pyannote are built after). But ANY future import of torch before `build_engines` reaches line 59 (e.g. a redaction backend that imports torch, or reordering the builder) silently skips the eager load, and CT2 then loads AFTER torch on the first transcribe -> the segfault commit 35dee8b fixed. There is no `assert "ctranslate2" loaded before torch` or logged warning. The dangerous co-residence (in-process diar fallback + faster_whisper in one process) DOES occur and is only safe by this ordering.
- Why it matters: a latent, silent-until-segfault fragility; a one-line guard/log would make a future reordering fail loudly instead of crashing at random.

### F12 [LOW] Config: nested typo hard-fails but top-level section typo is silently ignored
- file:line: `src/callqa/config.py:51-58` (StrictModel), `172` (`class Config(BaseModel)`), `203-215` (`_env_overrides`).
- What: sub-sections are `StrictModel` (extra=forbid) but `Config` itself is a plain `BaseModel` (pydantic default extra=ignore). So `CALLQA_JUDGE__NONEXISTENT=1` raises a ValidationError (good), but `CALLQA_JUGDE__BASE_URL=...` (misspelled SECTION) is silently dropped — the operator believes they redirected the judge endpoint and nothing applied. Same for a mistyped top-level key in config.yaml.
- Why it matters: the StrictModel docstring's whole rationale ("a typo ... simply never applied") is defeated at the top level. Make `Config` a StrictModel too.

### F13 [LOW] Hostile-cwd `.env` can redirect egress to any https host (validation allows it)
- file:line: `src/callqa/dotenv.py:30-36` (candidate_files includes `Path.cwd()`), `61-83` (load), + `config.py:24-48` (validate_endpoint).
- What: when the project root has no `.env`, `load_dotenv` loads `./.env` from the current working directory. A `.env` in an attacker-controlled cwd can set `CALLQA_JUDGE__BASE_URL=https://evil.example/v1` (or ASR base_url); `validate_endpoint` permits ANY https host (only http is restricted to loopback), so redacted transcripts and the judge prompt egress to the attacker. Shell-env precedence protects values already exported, but not first-set ones.
- Repro: from `/tmp/x` containing a `.env` with `CALLQA_JUDGE__BASE_URL=https://evil...`, run `python -m callqa ...` with no repo-root `.env` present -> the override loads.
- Why it matters: documented-but-risky. For a PII pipeline, cwd-sourced egress endpoints deserve at least a loud log of which `.env` was loaded and which CALLQA_*__BASE_URL it set, or an allowlist.

### F15 [HIGH] Installed wheel does NOT ship config/*.yaml — every command fails from any directory without a local `config/`
- file:line: `pyproject.toml:33-35` (package-data ships only `reporting/templates/*.j2` + a `reporting/*.yaml` glob that matches nothing) vs `src/callqa/rubric.py:60-61`, `src/callqa/reporting/common.py:35`, `src/callqa/resources.py:18-53`.
- What: `config/` (config.yaml, rubric.yaml, recommendations_he.yaml) lives at the REPO ROOT, outside `src/callqa/`, and is not declared as package data. Templates ship (verified in the built wheel); the config yamls do NOT. In an installed (non-editable) wheel the module lives under `site-packages/callqa/`, whose parents contain no `config/` dir, so `find_config` can only fall back to `cwd/config`.
- Repro (RAN): built `callqa-0.1.0-py3-none-any.whl`, `unzip -l` shows the three `config/*.yaml` are absent (only 7 yaml/j2 entries, all templates). Installed into a fresh venv, ran from a scratch dir with no `config/`:
  `from callqa.rubric import load_rubric; load_rubric()` -> `FileNotFoundError: could not find rubric.yaml. Looked in: <cwd>/config`.
  Every engine build (`load_rubric` in engines.py:44), report (`common.py` recommendations), and `_load_config`'s `find_config("config.yaml")` hits the same wall.
- Why it matters: the bank's on-prem deployment is a wheel install. As shipped, `callqa` run from anywhere but a directory that happens to contain a hand-placed `config/` fails on the first call. The pyproject already ships templates for exactly this reason ("an installed wheel fails every call with TemplateNotFound") — the config yamls were the other half of that fix and were missed. Workaround exists (`CALLQA_CONFIG_DIR` or cwd/config) but nothing self-contained.

### F14 [INFO/LOW] Duplicate `WATCHED_EXTENSIONS` definition
- file:line: `src/callqa/cli.py:32-33` and `37-38` — the same frozenset is assigned twice (copy/paste). Harmless; the second wins. Dead code noise only.




---

## CHECKED AND SOUND (attacked, held up)

- **Resume correctness (task #4) — RAN (scratchpad/resume_test.py, mock).** Ran a 25s call to completion, then deleted the `speakers..report` stage rows (simulating a crash right after ASR). Re-run resumed from ASR (`asr` kept, `speakers..report` recomputed) and ended with all 8 stages present. The early-diar guard `bundle is None and not store.is_done("speakers")` (pipeline.py:295) is correct: a resumed run whose ASR is done has `bundle is not None`, so the worker is NOT spawned — its output would only be discarded. No stale/half state observed.
- **Corrupt/truncated/partial .diar.json (task #1).** `collect` parses under `try/except (OSError, ValueError, KeyError, TypeError)` and falls back to in-process on any parse failure; a nonzero exit is turned into an OSError with a log tail; a timeout kills + cancels + falls back. The parent only READS `out_path` after `proc.wait()` returns, so there is no read-while-writing race. Worker writes via `atomic_write_text` (tmp + `os.replace`), so a truncated file can only be a not-yet-renamed tmp, never the target. Sound (except the empty-`[]` case, F9).
- **No .diar.json filename corruption across overlapping runs (task #1).** The per-call SQLite lock serializes normal runs of one call. Even in the orphan-worker window (F8), both the orphan and the new worker use `atomic_write_text` with unique `mkstemp` tmp names, so concurrent writes cannot interleave into a corrupt file — last `os.replace` wins and both carry the same call's audio. Corruption angle sound; resource-waste angle is F8.
- **ct2-before-torch ordering on the normal path (task #2).** `build_engines` constructs `FasterWhisperEngine` (engines.py:59) before any torch import (redactor/presidio and `LazyPyannoteDiarizer` are built after and are lazy). On darwin the construction-time eager load claims OpenMP for ct2 first; the in-process diar fallback then loads torch second — the order the crash-fix commit validated. Holds today; fragility if reordered is filed as F11.
- **Two engines never co-reside in the WORKER process (task #2).** `diar_worker` imports only pyannote/torch (no faster_whisper), and the parent's ASR uses ct2; separate processes each hold one OpenMP runtime. Sound.
- **`atomic_write_*` (state.py:50-71).** tmp file in the same dir + `os.replace`, with `except BaseException` unlink of the tmp on failure. Correct atomic-write discipline; a SIGKILL mid-write leaves only a stray `.<name>.*.tmp`, never a partial target.
- **`is_stage_done` existence check (state.py:123-135).** A stage recorded done whose artifact file was removed correctly reports not-done and recomputes. `load()` additionally rejects a wrong-`call_id` artifact. Both sound in isolation (the cross-stage staleness gap is F7).
- **CALL_ID / file_name path-traversal (ingestion.py:151-171).** `CALL_ID_RE` restricts ids; absolute or separator-bearing `file_name` is rejected before it can escape `calls/`. Sound.
- **Secrets file perms (runpod_cli.py save_state, dotenv.write_env_values).** Both open with `O_CREAT,0o600` then chmod 0600 — no world-readable window. Sound.
- **POD_START_CMD secret hygiene (runpod_cli.py).** No `set -x`; `GIT_TERMINAL_PROMPT=0` on clone; all output tee'd to the volume. HF_TOKEN/PUBLIC_KEY not traced. Sound.
- **Templates ship in the wheel (task #5).** `reporting/templates/*.j2` are present in the built wheel (verified). Only the `config/*.yaml` half is missing (F15).
- **review-reason thresholds (pipeline.py:56-57, 275-285).** MIN_CALL_SECONDS=20 / MIN_SPEECH_SECONDS=10 flag (do not drop) short/near-silent calls into needs_human_review; a legit short call is reported, not discarded. Behaviour is a flag, not a hard fail — sound.

## NOTES / SCOPE
- runpod_cli.py was reviewed at COMMITTED HEAD 618f756 (pod-local volume version). The working tree is mid-migration to NETWORK volumes (owner editing concurrently); that in-progress version adds `destroy --volume` and a `DELETE /networkvolumes` path not present in the committed code. F2/F3/F4/F5 were verified against the committed version and should be re-checked against the landed network-volume commit — in particular the new create path (volume created before pod, and pod created before state save) likely widens the F2 strand window to volumes too.
- All repros used /Users/yonatanohayon/Desktop/BRS_Model-main/.venv/bin/python; scratch scripts live in the session scratchpad (lock_test.py, resume_test.py, wheel/, freshvenv/), not in the repo.
