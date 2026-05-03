#!/bin/bash
# Полный тест gemma3-4B + регрессия проверка остальных моделей.
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 3

run_one() {
    local label="$1"; local model="$2"; local prompt="$3"; shift 3
    echo "=== $label ==="
    pkill -9 -f test_gguf_inference 2>/dev/null
    rm -f /dev/shm/prometorch_ddp_*
    sleep 2
    env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
        PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 "$@" \
        ./build_elbrus/examples/gguf/test_gguf_inference \
            "$model" \
            --max-tokens 80 --greedy --chat \
            "$prompt" 2>&1 | tail -25
    echo ""
}

# CRITICAL: gemma3-4B Russian + English
run_one "gemma3-4B RU (per-layer SWA)" \
    /home/paperclipdnb/gguf_models/gemma3-4b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про космос на русском."

run_one "gemma3-4B EN" \
    /home/paperclipdnb/gguf_models/gemma3-4b-Q4_K_M.gguf \
    "Hello! Tell me briefly about space."

# REGRESSION: qwen3-4B + mistral-7B
run_one "qwen3-4B RU regression" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Привет! Расскажи про космос на русском." \
    PT_NO_THINK=1

run_one "mistral-7B RU regression" \
    /home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf \
    "Привет! Расскажи про космос на русском."

echo ALL_DONE
