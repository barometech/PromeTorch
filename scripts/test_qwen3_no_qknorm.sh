#!/bin/bash
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 2

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
            --max-tokens 60 --greedy --chat \
            "$prompt" 2>&1 | tail -20
    echo ""
}

# Bisect: QK-norm vs no QK-norm with NEOX RoPE active
run_one "qwen3-4B EN — NEOX RoPE + QK-norm ON" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Hello! Tell me briefly about Moscow." \
    PT_NO_THINK=1

run_one "qwen3-4B EN — NEOX RoPE + QK-norm OFF" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Hello! Tell me briefly about Moscow." \
    PT_NO_THINK=1 PT_NO_QK_NORM=1

run_one "qwen3-4B RU — NEOX RoPE + QK-norm OFF" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про Москву." \
    PT_NO_THINK=1 PT_NO_QK_NORM=1

run_one "qwen3-4B EN — NEOX + QK-after-RoPE" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Hello! Tell me briefly about Moscow." \
    PT_NO_THINK=1 PT_QK_AFTER_ROPE=1

run_one "qwen3-4B RU — NEOX + QK-after-RoPE" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про Москву." \
    PT_NO_THINK=1 PT_QK_AFTER_ROPE=1

echo ALL_DONE
