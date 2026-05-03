#!/bin/bash
# Test RoPE NeoX vs NORM fix for qwen3 family + verify regression-free on mistral
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 3

# Pass extra env as `env` prefix so it expands as separate K=V tokens, not a single command word.
run_one() {
    local label="$1"
    local model="$2"
    local prompt="$3"
    shift 3
    echo "=== $label ==="
    pkill -9 -f test_gguf_inference 2>/dev/null
    rm -f /dev/shm/prometorch_ddp_*
    sleep 2
    env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
        PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 "$@" \
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
    PT_NO_THINK=1

# 2. qwen3-1.7B Russian
run_one "qwen3-1.7B Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/qwen3-1.7b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке." \
    PT_NO_THINK=1

# 3. qwen3-0.6B Russian
run_one "qwen3-0.6B Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/qwen3-0.6b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке." \
    PT_NO_THINK=1

# 4. qwen3-4B English (regression check — must work too)
run_one "qwen3-4B English (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Hello! Tell me briefly about space." \
    PT_NO_THINK=1

# 5. gemma3-4B Russian
run_one "gemma3-4B Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/gemma3-4b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке."

# 6. phi3.5-mini Russian
run_one "phi3.5-mini Russian (NeoX RoPE)" \
    /home/paperclipdnb/gguf_models/phi35-mini-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском языке."

echo ALL_DONE
