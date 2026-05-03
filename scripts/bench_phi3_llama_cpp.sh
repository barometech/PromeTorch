#!/bin/bash
# llama.cpp 32-thread baseline для phi3.5-mini
set -u
cd ~/promethorch

echo "=== llama.cpp 32t baseline для phi3.5-mini Q4_K_M ==="
date +"Start: %F %T"

LLAMA_BENCH=$HOME/llama.cpp/build/bin/llama-bench
[ -x "$LLAMA_BENCH" ] || { echo "ERR: $LLAMA_BENCH not found"; exit 1; }

# Decode benchmark на phi3.5-mini, fair config: numactl --interleave=all + 32 threads
numactl --interleave=all "$LLAMA_BENCH" \
    -m "$HOME/gguf_models/phi35-mini-Q4_K_M.gguf" \
    -t 32 -ngl 0 -p 0 -n 64 -r 2 2>&1 | tail -10

date +"End: %F %T"
echo PHI3_LLAMA_BENCH_DONE
