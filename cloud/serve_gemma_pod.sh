#!/usr/bin/env bash
# Serve the gemma-3-27b judge ON the RunPod pod, by hand.
#
# Why by hand: the pod's automatic bootstrap (runpod_cli.py dockerStartCmd)
# `git clone`s the repo, which fails for a PRIVATE GitHub repo - the pod has no
# credentials. So for a private repo the judge is served with this script
# instead. Copy it to the pod and run it (it reads the pod's secrets from
# /proc/1/environ, which an SSH session does not inherit):
#
#   scp -P <ssh-port> -i ~/.ssh/callqa_runpod_ed25519 \
#       cloud/serve_gemma_pod.sh root@<pod-ip>:/workspace/
#   ssh -p <ssh-port> -i ~/.ssh/callqa_runpod_ed25519 root@<pod-ip> \
#       'nohup bash /workspace/serve_gemma_pod.sh > /workspace/logs/serve.log 2>&1 &'
#
# The weights land on /workspace (the NETWORK VOLUME), so a later pod on the
# same volume skips the ~17 GB download - snapshot_download verifies and
# returns instantly when the files are already present. Then, from the laptop,
# tunnel past the Cloudflare 524 (the proxy kills >100s responses; a 27B is
# slower than that) and judge:
#
#   ssh -N -L 18000:127.0.0.1:8000 -p <ssh-port> -i ~/.ssh/callqa_runpod_ed25519 root@<pod-ip> &
#   bash cloud/rejudge_real002_gemma.sh "$CALLQA_JUDGE__API_KEY"
set -euo pipefail

export $(tr "\0" "\n" < /proc/1/environ | grep -E "^(VLLM_API_KEY|HF_TOKEN)=" | xargs -d "\n")
export HF_HUB_ENABLE_HF_TRANSFER=1
MODEL_ID="RedHatAI/gemma-3-27b-it-quantized.w4a16"
MODEL_DIR="/workspace/models/gemma-w4a16"
mkdir -p /workspace/logs "$MODEL_DIR"

echo "[serve] installing vllm + hf ..."
pip install -q --retries 5 --timeout 120 vllm==0.11.2 huggingface_hub hf_transfer

echo "[serve] fetching weights (skips if already on the volume) ..."
python - <<PY
from huggingface_hub import snapshot_download
print("[serve] at", snapshot_download("$MODEL_ID", local_dir="$MODEL_DIR"))
PY

echo "[serve] starting vllm on :8000 ..."
exec vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_ID" \
    --port 8000 --max-model-len 8192 --max-num-seqs 4 \
    --api-key "$VLLM_API_KEY" --gpu-memory-utilization 0.92
