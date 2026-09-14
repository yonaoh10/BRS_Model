#!/usr/bin/env bash
# Runs ON the rented GPU machine (DEV PHASE ONLY).
#
# Deliberately mirrors the bank-server runbook (README steps 2-6): install
# deps, download models with scripts/download_models.py, serve the judge with
# vLLM. The only extra is cloud/asr_server.py, which exposes the SAME
# FasterWhisperEngine over HTTP so a laptop can reach it.
#
# Normally this runs unattended as the pod's container start command
# (runpod_cli.py passes it via dockerStartCmd and sets the env vars below at
# pod creation), so `runpod_cli.py up` is the entire bring-up. To run it by
# hand in a web terminal instead:
#   export CALLQA_ASR_API_KEY=...      # printed by: runpod_cli.py urls
#   export VLLM_API_KEY=...            # printed by: runpod_cli.py urls
#   export CALLQA_LLM_MODEL=dicta-il/dictalm2.0-instruct
#   bash cloud/bootstrap_pod.sh
set -euo pipefail

: "${CALLQA_ASR_API_KEY:?set CALLQA_ASR_API_KEY (see: python cloud/runpod_cli.py urls)}"
: "${VLLM_API_KEY:?set VLLM_API_KEY (see: python cloud/runpod_cli.py urls)}"
LLM_MODEL="${CALLQA_LLM_MODEL:-dicta-il/dictalm2.0-instruct}"
WORKDIR="${CALLQA_WORKDIR:-/workspace/BRS_Model}"
LOGDIR="$WORKDIR/logs"

echo "=== 1/5 system packages ==="
apt-get update -qq && apt-get install -y -qq ffmpeg git >/dev/null

echo "=== 2/5 python dependencies ==="
cd "$WORKDIR"
pip install -q -r requirements.txt
pip install -q -r requirements-server.txt
pip install -q "vllm==${VLLM_VERSION:-0.11.2}"
pip install -q -e .

echo "=== 3/5 models (same script the bank server runs) ==="
mkdir -p "$LOGDIR" models
python scripts/download_models.py --asr --models-dir models
python scripts/download_models.py --llm --llm-model "$LLM_MODEL" --models-dir models

echo "=== 4/5 judge: vLLM on :8000 ==="
# Serve the copy download_models.py just fetched (it lives on the persistent
# volume), but keep the HF id as the served name so clients address the model
# by id. Serving the id directly made vLLM download the weights a second time
# into its own cache, which the container disk paid for on every boot.
LLM_LOCAL_DIR="models/${LLM_MODEL//\//--}"   # same mapping as download_models.py
[ -d "$LLM_LOCAL_DIR" ] || LLM_LOCAL_DIR="$LLM_MODEL"
nohup vllm serve "$LLM_LOCAL_DIR" \
    --served-model-name "$LLM_MODEL" \
    --port 8000 \
    --max-model-len 8192 \
    --api-key "$VLLM_API_KEY" \
    --gpu-memory-utilization 0.85 \
    > "$LOGDIR/vllm.log" 2>&1 &
# 0.85, not the 0.90 default: the ASR server on :8001 shares this GPU and
# holds ~2.5 GB; at 0.90 vLLM refuses to start when ASR wins the boot race.

echo "=== 5/5 ASR server on :8001 ==="
nohup env CALLQA_ASR_API_KEY="$CALLQA_ASR_API_KEY" \
    python cloud/asr_server.py \
        --port 8001 \
        --model-dir models/ivrit-whisper-large-v3-turbo-ct2 \
    > "$LOGDIR/asr.log" 2>&1 &

echo
echo "Both services are starting. vLLM needs a few minutes to load weights."
echo "Watch:  tail -f $LOGDIR/vllm.log   |   tail -f $LOGDIR/asr.log"
echo "Check:  curl -H \"Authorization: Bearer \$CALLQA_ASR_API_KEY\" localhost:8001/health"
echo "        curl -H \"Authorization: Bearer \$VLLM_API_KEY\" localhost:8000/v1/models"
