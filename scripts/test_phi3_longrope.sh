#!/bin/bash
# Verify phi3.5-mini после LongRoPE + всех других gemma3/phi3 fixes.
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

run() {
    local label="$1"; shift
    echo "==== $label ===="
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 2
    env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
        PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 "$@" \
        ./build_elbrus/examples/gguf/test_gguf_inference \
            /home/paperclipdnb/gguf_models/phi35-mini-Q4_K_M.gguf \
            --max-tokens 60 --greedy --chat \
            "Hello! Tell me briefly about Moscow." 2>&1 | tail -20
    echo ""
}

run "phi3.5-mini EN — LongRoPE applied"
run "phi3.5-mini RU — LongRoPE applied"

# Также verify gemma3 + qwen3 regression
echo "==== gemma3-4B RU (after GeGLU + post_norm + per-layer SWA) ===="
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2
env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/gemma3-4b-Q4_K_M.gguf \
        --max-tokens 60 --greedy --chat \
        "Привет! Расскажи коротко про космос на русском." 2>&1 | tail -18
echo ""

echo "==== qwen3-4B RU regression ===="
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2
env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 PT_NO_THINK=1 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
        --max-tokens 60 --greedy --chat \
        "Привет! Расскажи про космос." 2>&1 | tail -18
echo ALL_DONE
