#!/bin/bash
# Test qwen2.5-7B Russian via single-process (TP-4 OOM fallback).
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 5

# Single-proc, no TP — mmap not 4× replicated
PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 PT_SPEC_K=1 PT_NO_THINK=1 \
PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/qwen2.5-7b-Q4_K_M.gguf \
        --max-tokens 200 --temp 0.5 --chat \
        "Расскажи про Москву одним предложением." \
    > /tmp/q25_7b_ru.log 2>&1
echo DONE > /tmp/q25_7b_ru.done
