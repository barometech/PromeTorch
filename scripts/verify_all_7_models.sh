#!/bin/bash
# Полная регрессионная проверка всех 7 GGUF моделей на Эльбрусе.
# Используется после major load-path / forward-path изменений.
# 2026-05-03: создан после d9dce9e (phi3 mmap split fix).
set -u
cd ~/promethorch
loginctl enable-linger "$USER" 2>/dev/null || true

BIN="./build_elbrus/examples/gguf/test_gguf_inference"
[ -x "$BIN" ] || { echo "ERR: $BIN not built"; exit 1; }

mkdir -p run_logs
TS=$(date +%Y%m%d_%H%M%S)

PROMPT_RU="Что такое космос? Расскажи коротко."

run_sp() {
    local label="$1"; shift
    local model="$1"; shift
    echo "==== $label SP ===="
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 2
    env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
        PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
        "$BIN" "$HOME/gguf_models/$model" \
        --max-tokens 40 --greedy --chat \
        "$PROMPT_RU" 2>&1 | tail -5
    echo ""
}

run_tp4() {
    local label="$1"; shift
    local model="$1"; shift
    echo "==== $label TP-4 ===="
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 2
    for rank in 0 1 2 3; do
        PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 \
        PT_NUMA_REPLICATE=0 PT_DDP_SHM=1 PT_Q8_SOA=1 \
        PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
        numactl --cpunodebind=$rank --membind=$rank \
        "$BIN" "$HOME/gguf_models/$model" \
            --nprocs 4 --rank $rank \
            --master-addr 127.0.0.1 --master-port 29500 \
            --max-tokens 40 --greedy --chat \
            "$PROMPT_RU" \
            > run_logs/all7_${TS}_${label}_rank${rank}.log 2>&1 &
    done
    wait
    tail -5 run_logs/all7_${TS}_${label}_rank0.log
    echo ""
}

# 1. SP проверки (быстрее на Эльбрусе чем TP для маленьких моделей при наличии 32 cores)
run_sp "phi3.5-mini-SP" "phi35-mini-Q4_K_M.gguf"
run_sp "qwen3-1.7B-SP"  "qwen3-1.7b-Q4_K_M.gguf"
run_sp "qwen3-4B-SP"    "qwen3-4b-Q4_K_M.gguf"
run_sp "qwen2.5-7B-SP"  "qwen2.5-7b-Q4_K_M.gguf"
run_sp "mistral-7B-SP"  "mistral-7b-Q4_K_M.gguf"
run_sp "gemma3-4B-SP"   "gemma3-4b-Q4_K_M.gguf"

# 2. TP-4 проверки (на моделях которые в TP-4 работают)
run_tp4 "phi3.5-mini"   "phi35-mini-Q4_K_M.gguf"
run_tp4 "qwen3-1.7B"    "qwen3-1.7b-Q4_K_M.gguf"
run_tp4 "qwen3-4B"      "qwen3-4b-Q4_K_M.gguf"
run_tp4 "mistral-7B"    "mistral-7b-Q4_K_M.gguf"

echo ALL_7_MODELS_DONE
