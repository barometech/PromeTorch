#!/bin/bash
# Минимальный TP-4 bench только для qwen3-4B и qwen3-1.7B на чистом state.
# Цель — подтвердить 11.4 tok/s baseline + русский.
cd /home/paperclipdnb/promethorch
mkdir -p run_logs
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_* /dev/shm/prometorch*
sleep 5

run_tp4() {
    local label="$1"
    local model="$2"
    local prompt="$3"
    shift 3
    echo "==== $label ===="
    pkill -9 -f test_gguf_inference 2>/dev/null
    rm -f /dev/shm/prometorch* run_logs/${label}_*.log
    sleep 5
    for r in 0 1 2 3; do
      env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
          PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
          PT_DDP_SHM=1 "$@" \
          numactl --cpunodebind=$r --membind=$r \
          timeout 240 ./build_elbrus/examples/gguf/test_gguf_inference \
            "$model" \
            --nprocs 4 --rank $r --master-addr 127.0.0.1 --master-port 29611 \
            --max-tokens 60 --greedy --chat \
            "$prompt" > run_logs/${label}_rank$r.log 2>&1 &
    done
    wait
    echo "---- rank0 ----"
    grep -E "rope=|tok/s" run_logs/${label}_rank0.log | head -5
    echo "---- output ----"
    tail -8 run_logs/${label}_rank0.log
    echo ""
}

# qwen3-4B Russian (приоритет — 11.4 baseline)
run_tp4 "q34b_ru" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про Москву на русском." \
    PT_NO_THINK=1

# qwen3-1.7B Russian (помельче, всегда влезет)
run_tp4 "q317b_ru" \
    /home/paperclipdnb/gguf_models/qwen3-1.7b-Q4_K_M.gguf \
    "Привет! Расскажи коротко про Москву на русском." \
    PT_NO_THINK=1

# qwen3-4B English
run_tp4 "q34b_en" \
    /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    "Hello! Tell me briefly about Moscow." \
    PT_NO_THINK=1

echo ALL_DONE
