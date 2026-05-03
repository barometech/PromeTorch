#!/bin/bash
# qwen3-4B TP-4 with speculative decode (PT_SPEC_K=1) — recover 11.4 baseline.
cd /home/paperclipdnb/promethorch
mkdir -p run_logs
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch* run_logs/q34b_spec_*.log
sleep 5

for r in 0 1 2 3; do
  env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
      PT_SPEC_K=1 PT_NO_THINK=1 \
      PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      PT_DDP_SHM=1 \
      numactl --cpunodebind=$r --membind=$r \
      timeout 240 ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
        --nprocs 4 --rank $r --master-addr 127.0.0.1 --master-port 29612 \
        --max-tokens 60 --greedy --chat \
        "Привет! Расскажи коротко про Москву на русском." \
        > run_logs/q34b_spec_rank$r.log 2>&1 &
done
wait
echo DONE > /tmp/q3_spec.done
echo "---- rank0 ----"
grep -E "rope=|tok/s" run_logs/q34b_spec_rank0.log | head -5
tail -10 run_logs/q34b_spec_rank0.log
