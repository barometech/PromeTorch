#!/bin/bash
# Run qwen3:4b Q4_K_M inference on Эльбрус 8C2 4-NUMA server via PromeTorch
# Tensor Parallel (4 processes, 1 per NUMA node, 8 cores each).
#
# Usage: ./run_tp_elbrus.sh [--greedy|--sample] [prompt]

set -u
cd ~/promethorch

# Safety: systemd must not kill processes on SSH disconnect.
loginctl enable-linger user 2>/dev/null || true

MODEL="$HOME/gguf_models/qwen3-4b-Q4_K_M.gguf"
PROMPT="${2:-Write a short haiku about artificial intelligence}"
MODE="${1:---greedy}"
MAX_TOK=100

BIN="./build_elbrus/examples/gguf/test_gguf_inference"
if [ ! -x "$BIN" ]; then
    echo "ERR: $BIN not built. Run: cmake --build build_elbrus --target test_gguf_inference -j 16"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERR: $MODEL not found"
    exit 1
fi

mkdir -p run_logs

echo "=== PromeTorch TP-4 inference (qwen3:4b Q4_K_M, Эльбрус 8C2 4-NUMA) ==="
date +"Start: %F %T"

# Launch rank 0..3, one per NUMA node. Rank 0 prints, others silent.
for rank in 0 1 2 3; do
    # --membind pins this rank's malloc'd memory (sliced Q4_K weights, scratch
    # buffers, SHM sum slot) to the local NUMA node's DDR. Combined with
    # --cpunodebind this gives per-rank local HBM bandwidth = 4× aggregate
    # vs 1 NUMA bottleneck.
    # NOTE: --membind and --preferred conflict on Elbrus numactl ("Conflicting
    # policies"). Use --membind alone for strict local allocation.
    # OMP_NUM_THREADS=7 (not 8): same sweet-spot reason as 1-proc 24-vs-32 —
    # leave 1 core per NUMA node free for OS daemons / IRQ handlers. Sweep:
    # 4t=2.1, 6t=3.0, 7t=3.4, 8t=2.9.
    PT_NO_NUMA_POOL=1 \
    OMP_NUM_THREADS=7 \
    OMP_PLACES=cores OMP_PROC_BIND=close \
    PT_DDP_SHM=1 \
    numactl --cpunodebind=$rank --membind=$rank \
        "$BIN" "$MODEL" \
        --nprocs 4 --rank $rank \
        --master-addr 127.0.0.1 --master-port 29500 \
        --max-tokens $MAX_TOK $MODE \
        "$PROMPT" \
        > run_logs/tp4_rank${rank}.log 2>&1 &
    echo "Rank $rank launched PID=$!"
done

wait
date +"End: %F %T"
echo
echo "=== Rank 0 output ==="
cat run_logs/tp4_rank0.log
