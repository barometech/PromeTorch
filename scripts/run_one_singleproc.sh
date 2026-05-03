#!/bin/bash
# Single-process fallback для 7B+ моделей (TP-4 OOM на 125GB / 4 NUMA).
cd /home/paperclipdnb/promethorch
MODEL="${1:-}"
[ -z "$MODEL" ] && exit 1
MPATH="$HOME/gguf_models/$MODEL"
[ -f "$MPATH" ] || { echo "$MODEL: NOT FOUND"; exit 2; }

pkill -9 -f test_gguf_inference 2>/dev/null
sleep 5
rm -f /dev/shm/prometorch_ddp_*
sleep 2

LOG="/tmp/${MODEL%.gguf}_sp.log"

# nprocs=1 = single process = full 32 threads, без SHM AllReduce overhead
PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_SPEC_K=1 PT_NO_THINK=1 \
PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    timeout 360 ./build_elbrus/examples/gguf/test_gguf_inference "$MPATH" \
        --max-tokens 200 --temp 0.5 --chat \
        "Расскажи про Москву одним предложением." > "$LOG" 2>&1

SPEED=$(grep -oP '[0-9.]+(?= tok/s)' "$LOG" | tail -1)
SPEED="${SPEED:-FAIL}"
OUT=$(awk '/^--- Full Response/{flag=1; next} flag' "$LOG" | head -10 | tr '\n' ' ' | tr -s ' ' | head -c 300)
echo "${MODEL}|${SPEED}|${OUT}" >> /tmp/audit_summary_direct.csv
echo DONE > /tmp/${MODEL%.gguf}.done
