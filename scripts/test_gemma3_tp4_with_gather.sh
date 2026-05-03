#!/bin/bash
# Verify gemma3-4B TP-4 после ce41588 (post_attn_norm + post_ffw_norm wired)
# Требует PT_TP_GATHER=1 (use_gather path для full vector RMSNorm).
set -u
cd ~/promethorch
loginctl enable-linger "$USER" 2>/dev/null || true

MODEL="$HOME/gguf_models/gemma3-4b-Q4_K_M.gguf"
BIN="./build_elbrus/examples/gguf/test_gguf_inference"
[ -x "$BIN" ] || { echo "ERR: $BIN not built"; exit 1; }
[ -f "$MODEL" ] || { echo "ERR: $MODEL not found"; exit 1; }

mkdir -p run_logs
TS=$(date +%H%M%S)

echo "=== gemma3-4B TP-4 verify (commit ce41588 — post-norm wired) ==="
date +"Start: %F %T"

pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

for rank in 0 1 2 3; do
    PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 \
    PT_NUMA_REPLICATE=0 PT_DDP_SHM=1 PT_Q8_SOA=1 \
    PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_TP_GATHER=1 \
    numactl --cpunodebind=$rank --membind=$rank \
    "$BIN" "$MODEL" \
        --nprocs 4 --rank $rank \
        --master-addr 127.0.0.1 --master-port 29500 \
        --max-tokens 50 --greedy --chat \
        "Что такое космос? Расскажи коротко." \
        > run_logs/gemma3_tp4_${TS}_rank${rank}.log 2>&1 &
done

wait
date +"End: %F %T"
echo ""
echo "=== Rank 0 output ==="
cat run_logs/gemma3_tp4_${TS}_rank0.log | tail -30
echo GEMMA3_TP4_DONE
