#!/bin/bash
# Verify qwen3-14B TP-4 после rebuild18 (был OOM с прошлой сборки)
set -u
cd ~/promethorch
loginctl enable-linger "$USER" 2>/dev/null || true

MODEL="$HOME/gguf_models/qwen3-14b-Q4_K_M.gguf"
BIN="./build_elbrus/examples/gguf/test_gguf_inference"

[ -x "$BIN" ] || { echo "ERR: $BIN not built"; exit 1; }
[ -f "$MODEL" ] || { echo "ERR: $MODEL not found"; exit 1; }

mkdir -p run_logs
TS=$(date +%H%M%S)

echo "=== qwen3-14B TP-4 verify (rebuild18) ==="
date +"Start: %F %T"
free -h | head -2

pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

for rank in 0 1 2 3; do
    PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 \
    PT_NUMA_REPLICATE=0 PT_DDP_SHM=1 PT_Q8_SOA=1 \
    PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    numactl --cpunodebind=$rank --membind=$rank \
    "$BIN" "$MODEL" \
        --nprocs 4 --rank $rank \
        --master-addr 127.0.0.1 --master-port 29500 \
        --max-tokens 50 --greedy --chat \
        "Что такое космос? Расскажи коротко." \
        > run_logs/qwen3_14b_tp4_${TS}_rank${rank}.log 2>&1 &
done

wait
date +"End: %F %T"
echo ""
echo "=== Rank 0 output ==="
cat run_logs/qwen3_14b_tp4_${TS}_rank0.log | tail -30
echo QWEN14B_TP4_DONE
