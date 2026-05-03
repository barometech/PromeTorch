#!/bin/bash
# Verify gemma3-4B на TP-4 path после всех fix (per-layer SWA + post_norm + GeGLU).
cd /home/paperclipdnb/promethorch
mkdir -p run_logs
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch* run_logs/gemma3_tp4_*.log
sleep 5

# Note: TP-4 path may throw для gemma3 (post_attention_norm unsupported в TP).
# В этом случае сообщит: "TP: post_attention_norm unsupported (Gemma3 not yet wired)"

for r in 0 1 2 3; do
  env PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
      PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      PT_DDP_SHM=1 \
      numactl --cpunodebind=$r --membind=$r \
      timeout 240 ./build_elbrus/examples/gguf/test_gguf_inference \
        /home/paperclipdnb/gguf_models/gemma3-4b-Q4_K_M.gguf \
        --nprocs 4 --rank $r --master-addr 127.0.0.1 --master-port 29614 \
        --max-tokens 60 --greedy --chat \
        "Привет! Расскажи коротко про космос на русском." \
        > run_logs/gemma3_tp4_rank$r.log 2>&1 &
done
wait
echo "---- rank0 ----"
tail -25 run_logs/gemma3_tp4_rank0.log
echo ALL_DONE
