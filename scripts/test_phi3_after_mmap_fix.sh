#!/bin/bash
# Полный тест после d9dce9e: phi3 + остальные 6 моделей не сломались
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

run() {
    local label="$1"; shift
    local model="$1"; shift
    local prompt="$1"; shift
    echo "==== $label ===="
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 2
    env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
        PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 "$@" \
        ./build_elbrus/examples/gguf/test_gguf_inference \
            "/home/paperclipdnb/gguf_models/$model" \
            --max-tokens 40 --greedy --chat \
            "$prompt" 2>&1 | tail -10
    echo ""
}

# Phi3 — главная цель
run "phi3.5-mini RU" "phi35-mini-Q4_K_M.gguf" "Что такое космос? Расскажи коротко."
run "phi3.5-mini EN" "phi35-mini-Q4_K_M.gguf" "Hello! Tell me briefly about Moscow."

# Mistral — sanity что не сломали
run "mistral-7B RU"  "mistral-7b-Q4_K_M.gguf"  "Что такое космос? Расскажи коротко."

# Qwen3-4B — sanity
run "qwen3-4B RU"    "qwen3-4b-Q4_K_M.gguf"    "Что такое космос? Расскажи коротко."

echo ALL_DONE
