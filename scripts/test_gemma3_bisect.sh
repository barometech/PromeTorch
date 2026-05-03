#!/bin/bash
# Bisect: per-layer SWA vs PT_FORCE_ALL_GLOBAL для gemma3-4B.
# Если ALL_GLOBAL даёт тот же результат что и per-layer — bug не в SWA.
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
            /home/paperclipdnb/gguf_models/gemma3-4b-Q4_K_M.gguf \
            --max-tokens 50 --greedy --chat \
            "Hello! Tell me briefly about Moscow." 2>&1 | tail -15
    echo ""
}

run "gemma3-4B EN — per-layer SWA (default)"
run "gemma3-4B EN — PT_FORCE_ALL_GLOBAL=1 (bisect)" PT_FORCE_ALL_GLOBAL=1
run "gemma3-4B RU — per-layer SWA"
echo "  RU prompt..."

pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2
env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/gemma3-4b-Q4_K_M.gguf \
        --max-tokens 50 --greedy --chat \
        "Привет! Расскажи про Москву." 2>&1 | tail -15

echo ALL_DONE
