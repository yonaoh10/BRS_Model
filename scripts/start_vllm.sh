#!/usr/bin/env bash
# Serve the judge LLM with vLLM on the bank server.
# The pipeline never launches vLLM itself - it only checks connectivity.
#
# Usage: CALLQA_JUDGE_API_KEY=... ./scripts/start_vllm.sh <model-id-or-local-path> [port] [extra vllm flags...]
#
# The API key is required. An unauthenticated vLLM on a bank network will
# answer anyone who can reach the port, and what it answers with is built from
# call transcripts. Set the SAME value as judge.api_key (CALLQA_JUDGE__API_KEY).
# The server binds to 127.0.0.1 by default; set CALLQA_VLLM_HOST to change it
# deliberately rather than by omission.
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

MODEL="${1:?usage: start_vllm.sh <model-id-or-local-path> [port] [extra flags...]}"
PORT="${2:-8000}"
shift $(( $# > 2 ? 2 : $# ))
HOST="${CALLQA_VLLM_HOST:-127.0.0.1}"

if [[ -z "${CALLQA_JUDGE_API_KEY:-}" ]]; then
    echo "ERROR: CALLQA_JUDGE_API_KEY is not set." >&2
    echo "       Generate one with:  openssl rand -hex 32" >&2
    echo "       Then export the same value as CALLQA_JUDGE__API_KEY for the pipeline." >&2
    exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# "$@" carries any quantization flags from the table above.
exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$CALLQA_JUDGE_API_KEY" \
    --max-model-len 8192 "$@"
