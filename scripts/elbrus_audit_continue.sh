#!/bin/bash
# Continue audit from where it stopped (qwen2.5-7b onwards).
cd /home/paperclipdnb/promethorch
chmod +x scripts/elbrus_audit_one.sh

for M in qwen2.5-7b-Q4_K_M.gguf llama3-8b-Q4_K_M.gguf gemma3-4b-Q4_K_M.gguf phi35-mini-Q4_K_M.gguf deepseek-coder-7b-Q4_K_M.gguf qwen3-14b-Q4_K_M.gguf; do
    bash scripts/elbrus_audit_one.sh "$M" 2>&1 | tee -a /tmp/audit_remaining.log
    sleep 30
    # Force memory cleanup between runs
    pkill -9 -f test_gguf_inference 2>/dev/null
    rm -f /dev/shm/prometorch_ddp_*
    sleep 5
done

echo DONE > /tmp/audit_remaining.done

# llama.cpp 32-thread bench
sleep 10
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 5

bash scripts/elbrus_llama_bench.sh > /tmp/llama_bench.log 2>&1
echo DONE > /tmp/llama_bench.done

echo ALL_NIGHT_DONE > /tmp/night.done
