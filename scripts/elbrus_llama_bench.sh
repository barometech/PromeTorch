#!/bin/bash
# llama.cpp bench на 32 ядрах Эльбрус 8C2 — fair compare с PromeTorch TP-4.
set -uo pipefail
RES="/tmp/llama_bench.log"
> "$RES"

declare -a MODELS=(
    "qwen3-4b-Q4_K_M.gguf"
    "qwen3-1.7b-Q4_K_M.gguf"
    "mistral-7b-Q4_K_M.gguf"
    "qwen2.5-7b-Q4_K_M.gguf"
    "llama3-8b-Q4_K_M.gguf"
    "qwen3-14b-Q4_K_M.gguf"
)

for M in "${MODELS[@]}"; do
    P="$HOME/gguf_models/$M"
    if [ ! -f "$P" ]; then echo "[skip] $M not found" | tee -a "$RES"; continue; fi
    echo "=== $M ===" | tee -a "$RES"
    # 32-thread, NUMA interleave для равномерного DDR access
    numactl --interleave=all ~/llama.cpp/build/bin/llama-bench \
        -m "$P" -t 32 -p 32 -n 64 -r 2 2>&1 | tail -25 | tee -a "$RES"
    echo | tee -a "$RES"
done

echo DONE_BENCH > /tmp/llama_bench.done
