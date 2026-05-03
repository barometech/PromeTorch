#!/bin/bash
# TP-4 RoPE NeoX bench: tok/s + Russian quality на всех затронутых моделях.
# Это hot path 11.4 tok/s. Если регрессия — РОЛБЭК.
cd /home/paperclipdnb/promethorch
mkdir -p run_logs

run_tp4() {
    local label="$1"
    local model="$2"
    local prompt="$3"
    local extra="$4"
    echo "==== $label ===="
    pkill -9 -f test_gguf_inference 2>/dev/null
    rm -f /dev/shm/prometorch* run_logs/rope_${label}_*.log
    sleep 3
    for r in 0 1 2 3; do
      PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
      PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      PT_DDP_SHM=1 $extra \
      numactl --cpunodebind=$r --membind=$r \
      timeout 240 ./build_elbrus/examples/gguf/test_gguf_inference \
        "$model" \
        --nprocs 4 --rank $r --master-addr 127.0.0.1 --master-port 29610 \
        --max-tokens 80 --greedy --chat \
        "$prompt" > run_logs/rope_${label}_rank$r.log 2>&1 &
    done
    wait
    echo "---- rank0 output ----"
    grep -E "rope=|tok/s|^Generated|tok/s|^[A-ZА-Я]" run_logs/rope_${label}_rank0.log | head -30
    grep -E "tok/s" run_logs/rope_${label}_rank0.log | tail -3
    echo ""
}

run_tp4 "mistral7b_ru" \
    /home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про Москву на русском."

run_tp4 "qwen25_7b_ru" \
    /home/paperclipdnb/gguf_models/qwen2.5-7b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про Москву на русском."

run_tp4 "qwen3_4b_ru" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про Москву на русском." \
    "PT_NO_THINK=1"

run_tp4 "qwen3_4b_en" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Hi! Tell me briefly about Moscow." \
    "PT_NO_THINK=1"

run_tp4 "qwen3_17b_ru" \
    /home/paperclipdnb/gguf_models/qwen3-1.7b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про Москву на русском." \
    "PT_NO_THINK=1"

echo ALL_DONE
