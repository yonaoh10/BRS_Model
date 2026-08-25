#!/usr/bin/env bash
# Serve the judge LLM with vLLM on the bank server.
# The pipeline never launches vLLM itself - it only checks connectivity.
#
# Usage: ./scripts/start_vllm.sh <model-id-or-local-path> [port]
#
# Quantization flags per GPU budget:
#   24 GB (e.g. RTX 4090 / L4):
#     fp16 7B model            -> no extra flags
#     AWQ-quantized 12-27B     -> add: --quantization awq
#     GPTQ-quantized model     -> add: --quantization gptq
#   48 GB+ (e.g. A6000 / 2x24GB):
#     Llama-3.3-70B AWQ        -> add: --quantization awq --tensor-parallel-size 2
#   If you hit CUDA OOM: lower --max-model-len, add --gpu-memory-utilization 0.90,
#   or pick a smaller/quantized model.
set -euo pipefail

MODEL="${1:?usage: start_vllm.sh <model-id-or-local-path> [port]}"
PORT="${2:-8000}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 8192
