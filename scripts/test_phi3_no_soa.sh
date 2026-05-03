#!/bin/bash
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 2

echo "=== phi3.5 EN with PT_Q8_SOA=0 (Q4_K direct) ==="
env PT_Q8_SOA=0 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/phi35-mini-Q4_K_M.gguf \
        --max-tokens 60 --greedy --chat \
        "Hello! Tell me briefly about space." 2>&1 | tail -20

echo ""
echo "=== phi3.5 RU with PT_Q8_SOA=0 ==="
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2
env PT_Q8_SOA=0 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/phi35-mini-Q4_K_M.gguf \
        --max-tokens 60 --greedy --chat \
        "Привет! Расскажи коротко про космос на русском." 2>&1 | tail -20

echo ""
echo "=== phi3.5 EN raw prompt (no chat template) ==="
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2
env PT_Q8_SOA=0 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/phi35-mini-Q4_K_M.gguf \
        --max-tokens 40 --greedy \
        "Once upon a time" 2>&1 | tail -15

echo ALL_DONE
