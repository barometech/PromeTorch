#!/bin/bash
# Run remaining models one by one with cleanup. qwen3-0.6b/1.7b/4b уже сделаны.
cd /home/paperclipdnb/promethorch
chmod +x scripts/elbrus_audit_one.sh
> /tmp/audit_summary.csv

# Already done in earlier run:
echo "qwen3-0.6b-Q4_K_M.gguf|25.6| (mojibake, technically OK)" >> /tmp/audit_summary.csv
echo "qwen3-1.7b-Q4_K_M.gguf|17.3|: </think> ТПоск ХПоже" >> /tmp/audit_summary.csv
echo "qwen3-4b-Q4_K_M.gguf|9.8|<think> </think> </think> Кон К" >> /tmp/audit_summary.csv

for M in mistral-7b-Q4_K_M.gguf qwen2.5-7b-Q4_K_M.gguf llama3-8b-Q4_K_M.gguf gemma3-4b-Q4_K_M.gguf phi35-mini-Q4_K_M.gguf deepseek-coder-7b-Q4_K_M.gguf qwen3-14b-Q4_K_M.gguf; do
    bash scripts/elbrus_audit_one.sh "$M" 2>&1 | tee -a /tmp/audit_remaining.log
    sleep 30  # extra cleanup time
done

echo DONE > /tmp/audit_remaining.done

# llama.cpp bench AFTER all PromeTorch tests
sleep 10
pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 5

bash scripts/elbrus_llama_bench.sh > /tmp/llama_bench.log 2>&1
echo DONE > /tmp/llama_bench.done

echo ALL_NIGHT_DONE > /tmp/night.done
