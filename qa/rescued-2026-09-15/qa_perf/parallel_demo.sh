#!/bin/zsh
# Run ASR (baseline config) and pyannote diarization concurrently as TWO processes.
SP=/private/tmp/claude-501/-Users-yonatanohayon-Desktop-BRS-Model-main/e125e1e3-6108-476d-ab24-71bc406394de/scratchpad/qa_perf
PY=/Users/yonatanohayon/Desktop/BRS_Model-main/.venv/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1
start=$(date +%s.%N)
$PY $SP/asr_bench.py par_asr 5 1 0 0 $SP/asr_par.json > $SP/par_asr.log 2>&1 &
PID_A=$!
$PY $SP/diar_bench.py par_diar 2 full $SP/diar_par.json > $SP/par_diar.log 2>&1 &
PID_D=$!
wait $PID_A; echo "asr done at $(echo "$(date +%s.%N) - $start" | bc)s"
wait $PID_D; echo "diar done at $(echo "$(date +%s.%N) - $start" | bc)s"
end=$(date +%s.%N)
echo "PARALLEL combined wall: $(echo "$end - $start" | bc) s"
tail -1 $SP/par_asr.log
tail -1 $SP/par_diar.log
