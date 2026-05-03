#!/bin/bash
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 5

# Mistral with hidden state dump (working baseline)
PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 PT_DUMP_HIDDEN=1 \
PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf \
        --max-tokens 5 --greedy --chat \
        "Привет" \
    > /tmp/m7b_hidden.log 2>&1
echo DONE_M7
