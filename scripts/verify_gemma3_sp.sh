#!/bin/bash
# Verify gemma3-4B SP after d9dce9e (no regression check)
set -u
cd ~/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
    "$HOME/gguf_models/gemma3-4b-Q4_K_M.gguf" \
    --max-tokens 50 --greedy --chat \
    "Что такое космос? Расскажи коротко." 2>&1 | tail -25

echo GEMMA3_SP_DONE
