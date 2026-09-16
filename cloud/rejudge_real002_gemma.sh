#!/usr/bin/env bash
# Re-run ONLY the judge + report stages of REAL002 against gemma-3-27b on the
# pod, reached through an SSH tunnel on 127.0.0.1:18000 (the runpod.net proxy
# 524s on responses slower than ~100 s, and a 27B on a 4090 is slower). All
# other REAL002 stages stay as stored. See cloud/rejudge_real002_gemma.md.
#
# Usage: cloud/rejudge_real002_gemma.sh <judge_api_key>
#   - the tunnel must already be up (ssh -N -L 18000:127.0.0.1:8000 ...)
#   - vLLM on the pod must be serving RedHatAI/gemma-3-27b-it-quantized.w4a16
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <judge_api_key>" >&2
  exit 2
fi
KEY="$1"
cd "$(dirname "$0")/.."

# Drop only the judge+report stage records and the mock scorecard, so the
# resumable ASR/diarization/redaction/features artifacts are reused as-is.
sqlite3 data/callqa_state.db \
  "DELETE FROM stages WHERE call_id='REAL002' AND stage IN ('judge','report');
   DELETE FROM locks  WHERE call_id='REAL002';"
rm -f data/output/scores/REAL002.json

env CALLQA_SPEAKERS__DIARIZATION_MODEL=pyannote/speaker-diarization-3.1 \
    CALLQA_ASR__COMPUTE_TYPE=int8 HF_HUB_OFFLINE=1 \
    CALLQA_JUDGE__ENGINE=vllm \
    CALLQA_JUDGE__BASE_URL=http://127.0.0.1:18000/v1 \
    CALLQA_JUDGE__API_KEY="$KEY" \
    CALLQA_JUDGE__MODEL=RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    .venv/bin/python -m callqa process \
        --audio Calls/Call_001.m4a --call-id REAL002 --banker-id B900

echo "--- stored scorecard ---"
python3 -c "import json;d=json.load(open('data/output/scores/REAL002.json'));\
print('total',d['weighted_total'],'engine',d['judge_engine'],'model',d['model'])"
