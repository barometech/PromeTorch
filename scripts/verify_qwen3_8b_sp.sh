#!/bin/bash
set -u
cd ~/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

echo "=== qwen3-8B SP verify ==="
date +"Start: %F %T"

env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    timeout 240 ./build_elbrus/examples/gguf/test_gguf_inference \
    "$HOME/gguf_models/qwen3-8b-Q4_K_M.gguf" \
    --max-tokens 40 --greedy --chat \
    "Что такое космос? Расскажи коротко." 2>&1 | tail -15

date +"End: %F %T"
echo QWEN3_8B_DONE
