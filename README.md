# Call-QA PoC — Hebrew Conversation Intelligence for Banker Calls

A production-grade, single-call pipeline that takes one banker↔customer
recording (Hebrew) and produces the complete per-call output:

1. **Transcription** — ivrit.ai Whisper (faster-whisper CT2), Hebrew forced.
2. **Speaker attribution** — stereo channel split (primary); pyannote
   diarization fallback for mono recordings.
3. **PII redaction** — Hebrew-aware (Israeli ID with checksum, phones,
   payment cards, aggressive account-like numbers, names).
4. **Objective features** — talk ratio, interruptions, patience, questions,
   monologue length, dead air, speech rate.
5. **LLM judge** — a weighted 8-dimension rubric scored by a locally served
   vLLM model, with verbatim evidence quotes (verified, never fabricated).
6. **Per-call HTML report** — Hebrew, RTL, fully offline (inline CSS).

Around that core sit three thin layers: drivers (`process` / `watch` / `run`),
per-banker aggregation (`report`), and judge calibration vs. human QA ratings
(`calibrate`, QWK).

**Everything runs end-to-end in mock mode on a machine with no GPU and no
models** — deterministic fake engines behind the same interfaces, switched to
real engines by config only. Emotion recognition is explicitly out of scope.

## Quick start (dev machine, mock mode)

```bash
pip install -r requirements.txt
pip install -e .

python scripts/generate_sample_data.py          # 6 synthetic stereo WAVs + CSVs

# The atomic unit of work - ONE call (primary acceptance test):
python -m callqa process --mock --audio data/input/calls/CALL001.wav

# PoC batch + aggregate reports + calibration:
python -m callqa run --mock
python -m callqa report --mock
python -m callqa calibrate --mock
# open data/output/reports/index.html

make test                                        # pytest suite
```

Exit codes for `process`: `0` success · `1` needs_human_review · `2` failed —
wire any external scheduler or recording-system hook straight into it.

## CLI

| Command | Purpose |
|---|---|
| `callqa process --audio F [--call-id --banker-id --banker-channel]` | one call, full pipeline |
| `callqa watch` | production ingestion: poll input dir, process each stable file once, move to `processed/`/`failed/` |
| `callqa run [--max-workers N]` | PoC driver over `data/input/metadata.csv` |
| `callqa report` | per-banker reports + `reports/index.html` |
| `callqa calibrate` | QWK vs `human_ratings.csv` → `reports/calibration.html` |
| `callqa validate-inputs` | strict metadata / ratings validation |

All commands accept `--config`, `--mock`, `--force`. Any config field is
env-overridable: `CALLQA_SECTION__FIELD` (e.g. `CALLQA_JUDGE__BASE_URL`).

## Input contract

- `data/input/calls/*.{wav,mp3}` — file stem (or metadata) = `call_id`.
- `data/input/metadata.csv` — required: `call_id, banker_id, file_name`;
  optional: `call_date, call_type, banker_channel (L/R), banker_name`.
- `data/input/human_ratings.csv` — `call_id, rater_id,` one column per rubric
  dimension id, scores 1–5 (1–2 raters per call).

## Design guarantees

- **Idempotent + resumable**: per call×stage state in SQLite; re-runs skip
  completed stages; `--force` reruns. Artifacts written atomically.
- **Per-call atomicity**: an SQLite lock per `call_id` prevents double
  processing; every call ends in an explicit status envelope; one bad call
  never stops a driver.
- **Privacy**: only redacted text reaches the judge, reports, and logs. Raw
  transcripts live only under `data/output/transcripts/` (with a warning
  README). Logs carry call_ids and stage names, never transcript content.
- **Determinism/audit**: judge temperature 0.0; every scorecard stores model
  id, prompt SHA-256, prompt version, retries, latency, timestamp.
- **Evidence verification**: every judge quote must appear verbatim in the
  redacted transcript or the response is rejected and retried; after final
  failure the call is marked `needs_human_review` — never fabricated.
- **Gate rule**: a gate dimension (identification, compliance) scored ≤2 caps
  the total at 59 and flags the call.

---

# Bank-server deployment runbook

The developer machine never downloads models. All model files are fetched on
the bank server by `scripts/download_models.py`.

1. **Transfer** the repository plus the wheels bundle. Build the bundle on an
   internet-connected machine first:
   `./scripts/build_offline_bundle.sh` (creates `wheels/`).
2. **Install offline** on the server:
   `./scripts/install_offline.sh --server`
   (installs `requirements.txt` + `requirements-server.txt` from `wheels/`,
   then the `callqa` package — no network).
3. **Install ffmpeg**: `apt/yum install ffmpeg`, or drop a static build of
   `ffmpeg`/`ffprobe` into `$PATH` on a fully offline host.
4. **Download models** per available VRAM (this step needs internet or
   pre-staged files):

   | VRAM | ASR | Judge LLM (`--llm-model`) | vLLM flags |
   |---|---|---|---|
   | 24 GB | `--asr` (ivrit CT2, ~1.6GB) | `dicta-il/dictalm2.0-instruct` (7B fp16), or a 12–27B AWQ/GPTQ build | `--quantization awq` for AWQ |
   | ≥48 GB | `--asr` | `meta-llama/Llama-3.3-70B-Instruct` AWQ (gated) | `--quantization awq --tensor-parallel-size 2` |

   ```bash
   python scripts/download_models.py --asr
   python scripts/download_models.py --llm --llm-model dicta-il/dictalm2.0-instruct
   # only if recordings turn out to be mono (gated; accept terms + set HF_TOKEN):
   python scripts/download_models.py --diarization
   # optional NER-based person redaction:
   python scripts/download_models.py --ner    # then set redaction.ner: true
   ```
   The script writes `models/MODELS_MANIFEST.json` and points `judge.model`
   in `config/config.yaml` at the downloaded LLM.
5. **Go offline**: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
6. **Serve the judge**: `./scripts/start_vllm.sh <model-id-or-path> 8000`
   (see the script header for quantization flags). The pipeline only checks
   connectivity to `judge.base_url`; it never launches vLLM.
7. **Place inputs**: recordings under `data/input/calls/`, plus
   `metadata.csv` and (for calibration) `human_ratings.csv`.
8. **Validate**: `python -m callqa validate-inputs` — fails with a problem
   table on any metadata issue.
9. **Process the PoC batch**: `python -m callqa run` (add
   `--max-workers 2` if the GPU has headroom). In ongoing production use
   `python -m callqa watch` instead — it polls the input directory and
   processes each new recording once its file size is stable.
10. **Calibrate**: `python -m callqa calibrate` →
    `data/output/reports/calibration.html` (PASS at overall QWK ≥ 0.70;
    dimensions with QWK < 0.60 are flagged "do not deploy without human
    review").
11. **Reports**: `python -m callqa report` →
    `data/output/reports/index.html` linking every per-call and per-banker
    report (all static HTML, no external assets).

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM (ASR) | set `asr.compute_type: int8` in config |
| CUDA OOM (vLLM) | use an AWQ/GPTQ model, lower `--max-model-len`, `--gpu-memory-utilization 0.9` |
| "mono recording but no diarizer" / mono files detected | install server deps, `download_models.py --diarization` with `HF_TOKEN` (accept pyannote terms on HF) |
| vLLM endpoint unreachable | start `scripts/start_vllm.sh`; check `judge.base_url` port; `curl localhost:8000/v1/models` |
| Low ASR confidence (`quality` block flags many segments) | check recording sample rate/noise; confirm `asr.language: he`; consider re-recording setup |
| ASR model directory missing/empty | run `download_models.py --asr` and check `paths.models_dir` |
| Call stuck as "already being processed" | previous process died mid-call: the lock is auto-stolen when the pid is dead; otherwise delete the row from the `locks` table in the state DB |

## Repository layout

See `src/callqa/` — `pipeline.py` (`process_call()`, the production core),
`engines.py` (DI container built once per process), stage modules
(`ingestion`, `audio`, `asr/`, `speakers/`, `redaction`, `features`,
`judge/`), `aggregation.py`, `calibration.py`, `reporting/`, and `cli.py`.
Config lives in `config/` (`config.yaml`, `rubric.yaml`,
`recommendations_he.yaml`). Tests in `tests/` (`make test`).
