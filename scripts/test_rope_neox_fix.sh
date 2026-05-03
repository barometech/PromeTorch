#!/bin/bash
# Test RoPE NeoX vs NORM fix for qwen3 family + verify regression-free on mistral
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 3

run_one() {
    local label="$1"
    local model="$2"
    local prompt="$3"
    local extra_env="$4"
    echo "=== $label ==="
    pkill -9 -f test_gguf_inference 2>/dev/null
    rm -f /dev/shm/prometorch_ddp_*
    sleep 2
    PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 $extra_env \
        ./build_elbrus/examples/gguf/test_gguf_inference \
            "$model" \
            --max-tokens 50 --greedy --chat \
            "$prompt" 2>&1 | tail -25
    echo ""
}

# 1. qwen3-4B (broken before fix) — Russian
run_one "qwen3-4B Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке." \
    "PT_NO_THINK=1"

# 2. qwen3-1.7B Russian
run_one "qwen3-1.7B Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/qwen3-1.7b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке." \
    "PT_NO_THINK=1"

# 3. qwen2.5-7B Russian
run_one "qwen2.5-7B Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/qwen2.5-7b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке."

# 4. mistral-7B Russian — ДОЛЖЕН РАБОТАТЬ КАК РАНЬШЕ (NORM RoPE)
run_one "mistral-7B Russian (NORM RoPE — regression check)" \
    /home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке."

# 5. gemma3-4B Russian
run_one "gemma3-4B Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/gemma3-4b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке."

# 6. phi3.5-mini Russian
run_one "phi3.5-mini Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/phi35-mini-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке."

echo ALL_DONE
