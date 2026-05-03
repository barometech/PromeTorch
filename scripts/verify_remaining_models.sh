#!/bin/bash
# Verify оставшиеся модели после d9dce9e
# qwen3-1.7B SP, qwen3-8B SP, qwen2.5-7B SP
set -u
cd ~/promethorch
loginctl enable-linger "$USER" 2>/dev/null || true

BIN="./build_elbrus/examples/gguf/test_gguf_inference"
[ -x "$BIN" ] || { echo "ERR: $BIN not built"; exit 1; }

PROMPT="Что такое космос? Расскажи коротко."

run_sp() {
    local label="$1"; shift
    local model="$1"; shift
    echo "==== $label ===="
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 2
    timeout 90 env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
        PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
        "$BIN" "$HOME/gguf_models/$model" \
        --max-tokens 40 --greedy --chat \
        "$PROMPT" 2>&1 | tail -8
    echo ""
}

run_sp "qwen3-1.7B SP" "qwen3-1.7b-Q4_K_M.gguf"
run_sp "qwen2.5-7B SP" "qwen2.5-7b-Q4_K_M.gguf"
run_sp "qwen3-8B SP"   "qwen3-8b-Q4_K_M.gguf"

echo REMAINING_DONE
