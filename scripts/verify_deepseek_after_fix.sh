#!/bin/bash
# deepseek-coder-7B retest после commit 81a79bd (legacy rope.scale_linear)
set -u
cd ~/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

echo "=== deepseek-coder-7B SP retest (после rope.scale_linear fix) ==="
date +"Start: %F %T"

# Базовый prompt без --chat (deepseek-coder = base model)
echo ""
echo "--- TEST 1: code completion (без chat template) ---"
env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    timeout 240 ./build_elbrus/examples/gguf/test_gguf_inference \
    "$HOME/gguf_models/deepseek-coder-7b-Q4_K_M.gguf" \
    --max-tokens 50 --greedy \
    "def fibonacci(n):" 2>&1 | tail -10

echo ""
echo "--- TEST 2: chat instruct ---"
env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    timeout 240 ./build_elbrus/examples/gguf/test_gguf_inference \
    "$HOME/gguf_models/deepseek-coder-7b-Q4_K_M.gguf" \
    --max-tokens 40 --greedy --chat \
    "Write a Python function to compute Fibonacci numbers." 2>&1 | tail -10

date +"End: %F %T"
echo DEEPSEEK_RETEST_DONE
