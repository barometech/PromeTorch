#!/bin/bash
# Russian audit single-process для 7B+ моделей (TP-4 OOM).
set -uo pipefail
cd /home/paperclipdnb/promethorch
> /tmp/sp_audit.csv

PROMPT="Расскажи про Москву одним предложением."

run() {
    local M="$1"
    local MPATH="$HOME/gguf_models/$M"
    [ -f "$MPATH" ] || { echo "$M|NOT_FOUND" >> /tmp/sp_audit.csv; return; }
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 5
    local LOG="/tmp/sp_${M%.gguf}.log"
    echo "[$(date +%T)] $M ..."
    PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 PT_SPEC_K=1 PT_NO_THINK=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
        timeout 480 ./build_elbrus/examples/gguf/test_gguf_inference "$MPATH" \
            --max-tokens 200 --temp 0.5 --chat \
            "$PROMPT" > "$LOG" 2>&1
    local SPEED=$(grep -oP '[0-9.]+(?= tok/s)' "$LOG" | tail -1)
    SPEED="${SPEED:-FAIL}"
    local OUT=$(awk '/^--- Full Response/{f=1;next} f' "$LOG" | head -10 | tr '\n' ' ' | tr -s ' ' | head -c 300)
    echo "[$(date +%T)] $M: $SPEED tok/s | $OUT"
    echo "${M}|${SPEED}|${OUT}" >> /tmp/sp_audit.csv
    sleep 5
}

run mistral-7b-Q4_K_M.gguf
run qwen2.5-7b-Q4_K_M.gguf
run llama3-8b-Q4_K_M.gguf
run qwen3-14b-Q4_K_M.gguf

echo
echo "=== summary ==="
cat /tmp/sp_audit.csv
echo DONE > /tmp/sp_audit.done
