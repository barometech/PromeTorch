#!/bin/bash
# Run qwen3:4b Q4_K_M inference on Эльбрус 8C2 4-NUMA server via PromeTorch
# Tensor Parallel (4 processes, 1 per NUMA node, 8 cores each).
#
# Usage: ./run_tp_elbrus.sh [--greedy|--sample] [prompt]

set -u
cd ~/promethorch

# Safety: systemd must not kill processes on SSH disconnect.
loginctl enable-linger "$USER" 2>/dev/null || true

# Guard: PT_PIN_THREADS=1 в TP-режиме катастрофически ломает производительность.
# ThreadPool маппит worker_id на абсолютные CPU ID (0..31), а ранки 1-3 уже
# numactl --cpunodebind'ы на CPU 8-15 / 16-23 / 24-31 — pin на 0..7 либо
# отбрасывается kernel'ом, либо клампит всех на одно allowed CPU. Падает до
# 1.4 tok/s вместо 9.4. Bisect: 2026-04-30.
if [ "${PT_PIN_THREADS:-}" = "1" ]; then
    echo "ERROR: PT_PIN_THREADS=1 в TP-режиме ломает NUMA-binding и режет tok/s в ~7 раз." >&2
    echo "       Сними этот env и запусти снова. См. BENCH_ELBRUS.md." >&2
    exit 1
fi

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
    # OMP_NUM_THREADS=8: после Round 4 Step 1 (persistent ThreadPool, commit
    # a338ae6) sweet-spot сместился с 7t (mutex+CV pool, 8t крошился sync'ом)
    # на 8t (full NUMA node utilization). Sweep 2026-05-01:
    # 7t = 9.9 tok/s, 8t = 10.5 tok/s (+6%). Старая запись (mutex+CV):
    # 4t=2.1, 6t=3.0, 7t=3.4, 8t=2.9.
    # PT_PIN_THREADS is deliberately NOT set here. In TP mode each rank is
    # already numactl --cpunodebind'd to 8 contiguous cores, but the ThreadPool
    # pin logic maps worker_id to absolute CPU IDs (0..31). Ranks 1–3 would
    # therefore try to pin workers to CPUs outside their cpuset, which the
    # kernel either rejects (setaffinity error) or forces onto the single
    # first allowed CPU — serialising the rank and dropping tok/s to ~1.8.
    # Confirmed 2026-04-22 bisect (vliw_mission/bisect_phase0-3_results.md).
    #
    # PT_NUMA_REPLICATE=1 gives a small, in-noise +0.2 tok/s on TP (5.3→5.5),
    # but it's "free" in the sense that memory is already membind'd and the
    # copy happens once at load. Keep it on.
    PT_NO_NUMA_POOL=1 \
    OMP_NUM_THREADS=8 \
    PT_NUMA_REPLICATE=0 \
    PT_DDP_SHM=1 \
    PT_Q8_SOA="${PT_Q8_SOA:-1}" \
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
