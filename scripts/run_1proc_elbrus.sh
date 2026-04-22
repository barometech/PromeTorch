#!/bin/bash
# Run qwen3:4b Q4_K_M inference on Эльбрус 8C2 4-NUMA server via PromeTorch,
# single-process mode. Config as of 2026-04-22:
#   - 24 threads (leaves headroom for IO/kernel daemons)
#   - PT_PIN_THREADS=1: worker i pinned to a specific core; kernel may not
#     migrate workers between NUMA nodes, which would otherwise invalidate
#     the thread_local numa_node cache and cause remote DRAM fetches.
#   - PT_NUMA_REPLICATE=1: hot weights (attn_output, ffn_gate/up/down) are
#     replicated once per NUMA node at load; each worker reads its local copy.
#     Pre-2026-04-22 script never set this; baseline 4.7 tok/s ran with
#     replication silently disabled (see vliw_mission/agent_3_numa_audit.md).
#   - numactl --interleave=all: round-robins non-replicated pages across all
#     4 DDR controllers (embedding, KV cache, biases).
#   - OMP_PLACES / OMP_PROC_BIND intentionally NOT set — the hot GEMV path
#     uses c10::ThreadPool, not OpenMP. Those vars are cargo for our code.
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
PT_PIN_THREADS=1 \
PT_NUMA_REPLICATE=1 \
numactl --interleave=all \
    "$BIN" "$MODEL" \
    --max-tokens $MAX_TOK $MODE \
    "$PROMPT" \
    2>&1 | tee run_logs/1proc_best.log

date +"End: %F %T"
