#!/usr/bin/env bash
set -e
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^(VLLM_API_KEY|CALLQA_LLM_MODEL|HF_TOKEN)=")
cd /workspace/BRS_Model
pip install -q --retries 10 --timeout 120 vllm==0.11.2 huggingface_hub hf_transfer
echo DEPS-DONE
mkdir -p models logs
HF_HUB_ENABLE_HF_TRANSFER=1 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("RedHatAI/gemma-3-27b-it-quantized.w4a16",
                  local_dir="models/RedHatAI--gemma-3-27b-it-quantized.w4a16")
PY
echo MODEL-DONE
exec vllm serve models/RedHatAI--gemma-3-27b-it-quantized.w4a16 \
  --served-model-name RedHatAI/gemma-3-27b-it-quantized.w4a16 \
  --port 8000 --max-model-len 8192 --max-num-seqs 8 \
  --api-key "$VLLM_API_KEY" --gpu-memory-utilization 0.92
