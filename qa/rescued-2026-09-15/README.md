# Rescued QA material — night of 2026-09-14 → 2026-09-15

The overnight session crashed when the disk filled. Its four QA agents and the
real-judge runs left their working material in the macOS `/private/tmp` tree,
which does not survive a reboot. This directory is a verbatim rescue of that
material, minus three things: `raw_transcript.txt` (raw, un-redacted transcript
— must never enter git), the `repo*.tar.gz` snapshots (redundant with git
history), and Python `__pycache__`.

## The single most important file

`real002_gemma2.log` — holds the **only record of the first fully valid real
scorecard**: REAL002 judged by `RedHatAI/gemma-3-27b-it-quantized.w4a16`
(vLLM on the RunPod 4090, reached through a local tunnel at `127.0.0.1:18000`
after the runpod.net proxy 524'd on every attempt — Cloudflare kills responses
slower than ~100 s, and a 27B on a 4090 is slower than that):

    05:56:49 judge done: call_id=REAL002 total=82.5 gate_failed=False retries=0
    05:56:49 call failed: stage=judge error=[Errno 28] No space left on device

The judge succeeded; the scorecard write hit the full disk one second later.
The stored `data/output/scores/REAL002.json` at rescue time is still the mock
one (56.2). `real002_gemma.log` is the earlier proxy attempt (4 × HTTP 524).

## Directory map

- `qa_perf/` — run-performance agent. ASR benchmark matrix (beam × VAD ×
  threads: `asr_*.json`), cProfile (`asr_profile.*`), silero memory probe, and
  the **first local CPU diarization measurement** (`final.log`,
  `diar_*.json`): pyannote 3.1 on the 185 s call = **381.5 s wall, 370 s of it
  in the embedding step** (19 embedding calls). `par2_*.log` show ASR (3
  threads) and diarization (3 threads) running concurrently: combined wall
  394 s ≈ diarization alone, i.e. diarization fully shadows ASR.
  `REAL002.mono.wav` is the 16 kHz mono conversion of `Calls/Call_001.m4a`
  (derivable; kept on disk, not committed).
- `qa_judge/` — judge/rubric-calibration agent, died before writing
  conclusions. `exp1_arith.py` (weighted_total / gate-cap arithmetic by hand),
  `exp3_evidence.py` (verify_evidence / quote-snap edge cases on REAL002's
  redacted transcript), `exp4_supply.py` (quote-length thresholds).
- `qa_redaction/` — Hebrew number-word normaliser design agent.
  `trace_gap_b.py` traces why the cross-turn ID fragments `415 / 926 -9265`
  escape redaction on REAL002; `prototype.py` is a working "Pass B"
  implementation (digit-word table incl. ו־ prefix, folds runs of ≥4
  digit-words between normalize_for_detection and span detection, with an
  origin-span map), measured on REAL002.
- `real003*.log` — the dictalm2.0-7B judge failing evidence verification six
  times on REAL003 (cross-speaker quote splices) → needs_human_review, the
  designed floor for a judge below the quality bar.
- `lean_judge.sh` — pod-side script that downloaded and served gemma-3-27b
  (w4a16) with vLLM 0.11.2, max-model-len 8192, gpu-mem 0.92.
- `judge_schema.json` — the rubric-derived json_schema sent as
  `response_format` to pin the scorecard shape server-side.
