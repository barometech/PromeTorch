#!/bin/bash
# Single-model wrapper, runs in foreground (caller wraps with nohup setsid).
# Args: <model.gguf>
cd /home/paperclipdnb/promethorch
MODEL="${1:-}"
[ -z "$MODEL" ] && exit 1
MPATH="$HOME/gguf_models/$MODEL"
[ -f "$MPATH" ] || { echo "$MODEL: NOT FOUND"; exit 2; }

pkill -9 -f test_gguf_inference 2>/dev/null
sleep 5
rm -f /dev/shm/prometorch_ddp_*
sleep 2

PT_MODEL="$MPATH" PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_SPEC_K=1 \
    PT_NO_THINK=1 PT_MAX_TOK=200 PT_CHAT=1 \
    timeout 360 bash scripts/run_tp_elbrus.sh "--temp 0.5" \
    "Расскажи про Москву одним предложением." > /tmp/${MODEL%.gguf}_run.log 2>&1

# Save final answer
SPEED=$(grep -oP '\(\K[0-9.]+(?= tok/s)' "/tmp/${MODEL%.gguf}_run.log" | tail -1)
SPEED="${SPEED:-FAIL}"
OUT=$(awk '/^--- Full Response/{flag=1; next} flag' run_logs/tp4_rank0.log 2>/dev/null | head -10 | tr '\n' ' ' | tr -s ' ' | head -c 300)
echo "${MODEL}|${SPEED}|${OUT}" >> /tmp/audit_summary_direct.csv
echo DONE > /tmp/${MODEL%.gguf}.done
