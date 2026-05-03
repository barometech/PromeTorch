#!/bin/bash
# Bisect: phi3.5 quant path vs FP fallback. Если FP работает а quant нет —
# bug в Q5_K kernel/split. Если оба broken — глубже.
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2

run() {
    local label="$1"; shift
    echo "==== $label ===="
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 2
    env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 "$@" \
        ./build_elbrus/examples/gguf/test_gguf_inference \
            /home/paperclipdnb/gguf_models/phi35-mini-Q4_K_M.gguf \
            --max-tokens 40 --greedy --chat \
            "Hello! Tell me briefly about Moscow." 2>&1 | tail -15
    echo ""
}

run "phi3.5 — DEFAULT quant_gemv" PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1
run "phi3.5 — PT_NO_QUANT_GEMV=1 (FP fallback)" PT_NO_QUANT_GEMV=1
echo ALL_DONE
