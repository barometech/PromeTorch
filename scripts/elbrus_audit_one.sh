#!/bin/bash
# Run audit for ONE model with full cleanup before/after.
# Usage: bash elbrus_audit_one.sh <model_filename>
set -uo pipefail
cd /home/paperclipdnb/promethorch
MODEL="${1:-qwen3-4b-Q4_K_M.gguf}"
MPATH="$HOME/gguf_models/$MODEL"
PROMPT="Расскажи про Москву одним предложением."

# Resource limits: fix fork OOM на 7B+ моделях (Elbrus signals/fd quota)
ulimit -u 8192 2>/dev/null || true
ulimit -n 8192 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true

# Cleanup
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 5
rm -f /dev/shm/prometorch_ddp_*
sync
# drop kernel page cache (mmap'd weights from prev models). Best effort.
echo 1 | sudo -n tee /proc/sys/vm/drop_caches 2>/dev/null || true
sleep 2

LOG="audit_results/${MODEL%.gguf}_perblock_log.txt"
mkdir -p audit_results

if [ ! -f "$MPATH" ]; then
    echo "[$(date +%T)] $MODEL: SKIP (not found)"
    exit 1
fi

echo "[$(date +%T)] running $MODEL"
PT_MODEL="$MPATH" PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_SPEC_K=1 \
    PT_NO_THINK=1 PT_MAX_TOK=200 PT_CHAT=1 \
    timeout 600 bash scripts/run_tp_elbrus.sh "--temp 0.5" "$PROMPT" \
    > "$LOG" 2>&1

SPEED=$(grep -oP '\(\K[0-9.]+(?= tok/s)' "$LOG" | tail -1)
SPEED="${SPEED:-FAIL}"
OUT=$(awk '/^--- Full Response/{flag=1; next} flag' "$LOG" | head -8 | tr '\n' ' ' | tr -s ' ' | head -c 250)
echo "[$(date +%T)] $MODEL: $SPEED tok/s | $OUT"

echo "${MODEL}|${SPEED}|${OUT}" >> /tmp/audit_summary.csv
