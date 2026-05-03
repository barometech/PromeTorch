#!/bin/bash
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 5

# qwen3-4B with hidden state dump
PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 PT_DUMP_HIDDEN=1 PT_NO_THINK=1 \
PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
        --max-tokens 5 --greedy --chat \
        "Привет" \
    > /tmp/q34b_hidden.log 2>&1
echo DONE_Q3
