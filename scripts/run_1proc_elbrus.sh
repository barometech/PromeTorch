#!/bin/bash
# Run qwen3:4b Q4_K_M inference on Эльбрус 8C2 4-NUMA server via PromeTorch,
# single-process mode. Best-known peak config as of 2026-04-20:
#   - 24 threads (leaves headroom for IO/kernel daemons, 32 threads causes
#     context-switching overhead that eats ~10%)
#   - numactl --interleave=all (round-robins Q4_K weight pages across all 4
#     NUMA DDR controllers, eliminating cross-NUMA bandwidth contention that
#     plagues first-touch placement)
#
# Measured: 3.8 tok/s on qwen3:4b Q4_K_M (vs 2.8 default 32t plain, +36%).
# Details in BENCH_ELBRUS.md.
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

OMP_NUM_THREADS=24 \
OMP_PLACES=cores OMP_PROC_BIND=close \
numactl --interleave=all \
    "$BIN" "$MODEL" \
    --max-tokens $MAX_TOK $MODE \
    "$PROMPT" \
    2>&1 | tee run_logs/1proc_best.log

date +"End: %F %T"
