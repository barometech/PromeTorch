#!/bin/bash
# Verify mistral-7B TP-4 после d9dce9e (sanity — baseline 5.1 tok/s, не сломать)
set -u
cd ~/promethorch
loginctl enable-linger "$USER" 2>/dev/null || true

MODEL="$HOME/gguf_models/mistral-7b-Q4_K_M.gguf"
BIN="./build_elbrus/examples/gguf/test_gguf_inference"

[ -x "$BIN" ] || { echo "ERR: $BIN not built"; exit 1; }
[ -f "$MODEL" ] || { echo "ERR: $MODEL not found"; exit 1; }

mkdir -p run_logs
TS=$(date +%H%M%S)

echo "=== mistral-7B TP-4 verify (sanity post-d9dce9e) ==="
date +"Start: %F %T"

pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

for rank in 0 1 2 3; do
    PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 \
    PT_NUMA_REPLICATE=0 PT_DDP_SHM=1 PT_Q8_SOA=1 \
    numactl --cpunodebind=$rank --membind=$rank \
    "$BIN" "$MODEL" \
        --nprocs 4 --rank $rank \
        --master-addr 127.0.0.1 --master-port 29500 \
        --max-tokens 50 --greedy --chat \
        "Что такое космос? Расскажи коротко." \
        > run_logs/mistral_tp4_${TS}_rank${rank}.log 2>&1 &
done

wait
date +"End: %F %T"
echo ""
echo "=== Rank 0 output ==="
cat run_logs/mistral_tp4_${TS}_rank0.log | tail -25
echo MISTRAL_TP4_DONE
