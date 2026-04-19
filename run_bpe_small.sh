#!/bin/bash
cd ~/promethorch
rm -f /dev/shm/pir_w_* /dev/shm/pir_sync_init
rm -f ~/promethorch/logs/pir_bpe_*.log
mkdir -p ~/promethorch/logs ~/promethorch/checkpoints_bpe
for node in 0 1 2 3; do
  PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
  numactl --cpunodebind=$node --preferred=$node \
  ./build_mt/examples/pir/train_pir_elbrus \
    --fused --full --batch_size 2 --rank $node --nprocs 4 \
    --max_steps 1000 --log_interval 50 --gen_interval 200 --gen_tokens 100 \
    --save_interval 200 --save_dir checkpoints_bpe \
    --grad_accum 20 --seed $((42+node)) \
    --lr 0.0006 \
    --data data/russian_mega.tokens > ~/promethorch/logs/pir_bpe_$node.log 2>&1 &
  sleep 15
done
wait
echo ALL_DONE > ~/promethorch/logs/pir_bpe_done
