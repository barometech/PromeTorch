#!/bin/bash
# Identical conditions to legacy "11.4 tok/s" baseline test.
# Single-process, 32 threads, --max-tokens 200, --temp 0.5
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 5

echo "==== qwen3-4B 1-proc OMP=32 max-tok=200 (legacy 11.4 conditions) ===="
env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_SPEC_K=1 PT_NO_THINK=1 \
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
    ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
        --max-tokens 200 --temp 0.5 --chat \
        "Расскажи про Москву одним предложением." \
    2>&1 | tail -10
echo ""

echo "==== qwen3-4B TP-4 max-tok=200 ===="
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch*
sleep 5
for r in 0 1 2 3; do
  env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
      PT_SPEC_K=1 PT_NO_THINK=1 \
      PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      PT_DDP_SHM=1 \
      numactl --cpunodebind=$r --membind=$r \
      timeout 360 ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
        --nprocs 4 --rank $r --master-addr 127.0.0.1 --master-port 29613 \
        --max-tokens 200 --temp 0.5 --chat \
        "Расскажи про Москву одним предложением." \
        > run_logs/q34b_baseline_rank$r.log 2>&1 &
done
wait
echo "---- rank0 tok/s ----"
grep -E "tok/s" run_logs/q34b_baseline_rank0.log | tail -3
echo "---- rank0 tail ----"
tail -10 run_logs/q34b_baseline_rank0.log
echo ALL_DONE
