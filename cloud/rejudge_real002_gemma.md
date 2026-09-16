# Landing the gemma-3-27b scorecard for REAL002 (turnkey)

REAL002 is the one call with a *fully valid real scorecard* — gemma-3-27b
judged it at **total=82.5, gate passed, zero retries** on 2026-09-15 05:56 —
but the write died on a full disk one second later, so the stored scorecard
is still the mock one (56.2). The only record of the real result is
`qa/rescued-2026-09-15/real002_gemma2.log`. These steps re-run it and store it.

## Blocker as of 2026-09-16

The stopped pod `glvssw76jglgta` is pinned to a physical RunPod host that has
**no free RTX 4090**, so `runpod_cli.py up` returns HTTP 500 "not enough free
GPUs on the host machine". A pod-local volume can only restart on its own
host. Options, in preference order:

1. **Wait** for that host to free a GPU, then run the steps below. The gemma
   weights are still on the volume, so no re-download. Zero cost until it
   starts. (A poller may already be watching.)
2. **Destroy + recreate** (`runpod_cli.py destroy --yes`, then `up`): unblocks
   on any host but loses the volume, so the fresh pod re-downloads gemma
   (~15 GB, ~15 min) via `lean_judge.sh`. Costs a few $ + the download time.
   The user's standing instruction is to KEEP the volume, so this needs their
   OK first.
3. **Migrate to a network volume** (the real fix for this failure mode): a
   network volume + on-demand pod launches on any available host while keeping
   the weights. Requires the pod up once to copy the weights across.

## Steps once the pod is RUNNING

```bash
# 1. bring it up (writes CALLQA_* endpoints into .env)
python cloud/runpod_cli.py up

# 2. serve gemma (NOT the default dictalm bootstrap). Copy the rescued script
#    to the pod and run it; it downloads gemma-3-27b-w4a16 and serves vLLM on
#    :8000. Takes ~15 min on first boot.
#    (qa/rescued-2026-09-15/lean_judge.sh is the exact script that worked.)

# 3. The runpod.net proxy sits behind Cloudflare, which 524s any response
#    slower than ~100 s — and a 27B on a 4090 is slower than that. So tunnel
#    the vLLM port instead of using the proxy URL:
ssh -N -L 18000:127.0.0.1:8000 <pod-ssh-user@host> -p <pod-ssh-port> \
    -i ~/.ssh/callqa_runpod_ed25519 &

# 4. re-run ONLY the judge+report stages against the tunnel:
bash cloud/rejudge_real002_gemma.sh "$CALLQA_JUDGE__API_KEY"

# 5. STOP THE POD (the whole disaster started with a pod left running):
python cloud/runpod_cli.py down
```

After step 4 the stored scorecard, the report and the dashboard all show the
real 82.5. Everything else about REAL002 (real ASR, real pyannote-3.1
diarization, number-word-normalised redaction) is already stored — only the
judge stage is mock.
