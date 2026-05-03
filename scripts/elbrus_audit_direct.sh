#!/bin/bash
# Direct audit без вложенных bash subshells (избегает fork OOM на 7B+).
cd /home/paperclipdnb/promethorch
mkdir -p audit_results

PROMPT="Расскажи про Москву одним предложением."
> /tmp/audit_summary_direct.csv

for M in qwen2.5-7b-Q4_K_M.gguf llama3-8b-Q4_K_M.gguf gemma3-4b-Q4_K_M.gguf qwen3-14b-Q4_K_M.gguf; do
    MPATH="$HOME/gguf_models/$M"
    [ -f "$MPATH" ] || continue

    # cleanup
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 5
    rm -f /dev/shm/prometorch_ddp_*
    sync
    sleep 2

    LOG="audit_results/${M%.gguf}_direct_log.txt"
    echo "[$(date +%T)] running $M"

    # Direct invocation: run_tp_elbrus.sh, не ещё один wrapper
    PT_MODEL="$MPATH" PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_SPEC_K=1 \
        PT_NO_THINK=1 PT_MAX_TOK=200 PT_CHAT=1 \
        timeout 360 bash scripts/run_tp_elbrus.sh "--temp 0.5" "$PROMPT" \
        > "$LOG" 2>&1

    SPEED=$(grep -oP '\(\K[0-9.]+(?= tok/s)' "$LOG" | tail -1)
    SPEED="${SPEED:-FAIL}"
    OUT=$(awk '/^--- Full Response/{flag=1; next} flag' run_logs/tp4_rank0.log 2>/dev/null | head -8 | tr '\n' ' ' | tr -s ' ' | head -c 250)
    [ -z "$OUT" ] && OUT=$(awk '/^--- Full Response/{flag=1; next} flag' "$LOG" | head -8 | tr '\n' ' ' | tr -s ' ' | head -c 250)
    echo "[$(date +%T)] $M: $SPEED tok/s | $OUT"
    echo "${M}|${SPEED}|${OUT}" >> /tmp/audit_summary_direct.csv

    sleep 10
done

echo DONE > /tmp/audit_direct.done
