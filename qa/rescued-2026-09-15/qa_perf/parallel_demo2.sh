#!/bin/zsh
# Tuned parallel: cap each engine at 3 threads so they don't fight over 6 cores.
SP=/private/tmp/claude-501/-Users-yonatanohayon-Desktop-BRS-Model-main/e125e1e3-6108-476d-ab24-71bc406394de/scratchpad/qa_perf
PY=/Users/yonatanohayon/Desktop/BRS_Model-main/.venv/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1
start=$(date +%s.%N)
env OMP_NUM_THREADS=3 $PY $SP/asr_bench.py par2_asr 5 1 3 0 $SP/asr_par2.json > $SP/par2_asr.log 2>&1 &
PID_A=$!
env OMP_NUM_THREADS=3 $PY $SP/diar_bench.py par2_diar 2 full $SP/diar_par2.json > $SP/par2_diar.log 2>&1 &
PID_D=$!
wait $PID_A; echo "asr done at $(echo "$(date +%s.%N) - $start" | bc)s"
wait $PID_D; echo "diar done at $(echo "$(date +%s.%N) - $start" | bc)s"
end=$(date +%s.%N)
echo "PARALLEL2 (3+3 threads) combined wall: $(echo "$end - $start" | bc) s"
tail -1 $SP/par2_asr.log
tail -1 $SP/par2_diar.log
