#!/bin/bash
# Run qwen3:4b Q4_K_M inference on Эльбрус 8C2 4-NUMA server via PromeTorch,
# single-process mode.
#
# Measured on 2026-04-22 bisect (50-token greedy gen, qwen3:4b):
#   R=0 P=0 (default, plain interleave):     4.6 tok/s
#   R=0 P=1 (pinned threads):                 4.5 tok/s
#   R=1 P=0 (NUMA replicated weights):        4.3 tok/s   ← regresses
#   R=1 P=1 (replicate + pin):                4.4 tok/s
#
# Thread count sweep (T=8..32 at R=0 interleave, 50-tok greedy, 2026-04-22):
#   T=8  → 2.2      T=16 → 3.8      T=24 → 4.7
#   T=28 → 5.0      T=30 → 5.0      T=32 → 4.0
# → optimal T=30 (5.0 tok/s, +6.4% vs old T=24 baseline).
#
# Why PT_NUMA_REPLICATE=1 HURTS in 1-proc mode: `numactl --interleave=all`
# already scatters every page across the 4 DDR controllers; adding per-node
# replicas ~quadruples working-set memory (9.5 GB extra, vs 2.4 GB base)
# and blows the TLB, while the bandwidth picture was already balanced.
# Replication only pays in TP mode where each rank is --membind'd to ONE node.
# OMP_PLACES / OMP_PROC_BIND are NOT set — the hot GEMV path uses
# c10::ThreadPool, not OpenMP, so those vars never reached the real workers.
#
# Usage: ./run_1proc_elbrus.sh [--greedy|--sample] [prompt]

set -u
cd ~/promethorch

loginctl enable-linger "$USER" 2>/dev/null || true

MODEL="$HOME/gguf_models/qwen3-4b-Q4_K_M.gguf"
MODE="${1:---greedy}"
PROMPT="${2:-Write a short haiku about artificial intelligence}"
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

echo "=== PromeTorch 1-proc inference (qwen3:4b Q4_K_M, Эльбрус 8C2, 24t + interleave=all) ==="
date +"Start: %F %T"

OMP_NUM_THREADS=30 \
numactl --interleave=all \
    "$BIN" "$MODEL" \
    --max-tokens $MAX_TOK $MODE \
    "$PROMPT" \
    2>&1 | tee run_logs/1proc_best.log

date +"End: %F %T"
