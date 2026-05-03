#!/bin/bash
# Master night runner: audit + llama bench sequentially.
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
pkill -9 -f llama-bench 2>/dev/null
sleep 3
rm -f /tmp/audit2.done /tmp/llama_bench.done /dev/shm/prometorch_ddp_*

echo "[$(date +%T)] starting audit" > /tmp/night.log
bash scripts/full_russian_audit.sh >> /tmp/night.log 2>&1
echo "[$(date +%T)] audit DONE" >> /tmp/night.log
echo DONE > /tmp/audit2.done

echo "[$(date +%T)] starting llama bench" >> /tmp/night.log
bash scripts/elbrus_llama_bench.sh >> /tmp/night.log 2>&1
echo "[$(date +%T)] llama bench DONE" >> /tmp/night.log
echo DONE > /tmp/llama_bench.done

echo "[$(date +%T)] ALL DONE" >> /tmp/night.log
echo DONE > /tmp/night.done
