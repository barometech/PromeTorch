#!/bin/bash
# ============================================================================
# full_russian_audit.sh — прогон ВСЕХ моделей TP-4 на русском.
# Собирает: model | tok/s | output | время загрузки.
# Запуск на Эльбрус 8C2: ssh paperclipdnb@... bash full_russian_audit.sh
# ============================================================================

set -uo pipefail
cd ~/promethorch
mkdir -p audit_results

PROMPT="Расскажи про Москву одним предложением."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT="audit_results/audit_${TIMESTAMP}.md"

declare -a MODELS=(
    "qwen3-0.6b-Q4_K_M.gguf"
    "qwen3-1.7b-Q4_K_M.gguf"
    "qwen3-4b-Q4_K_M.gguf"
    "qwen2.5-7b-Q4_K_M.gguf"
    "mistral-7b-Q4_K_M.gguf"
    "llama3-8b-Q4_K_M.gguf"
    "gemma3-4b-Q4_K_M.gguf"
    "phi35-mini-Q4_K_M.gguf"
    "deepseek-coder-7b-Q4_K_M.gguf"
    "qwen3-14b-Q4_K_M.gguf"
)

cat > "$REPORT" <<EOF
# Эльбрус 8C2 — Russian Audit ($TIMESTAMP)

Промпт: \`$PROMPT\`
Конфиг: TP-4, PT_Q8_SOA=1, PT_LM_HEAD_FP=1, --temp 0.5, max_tok=120
Hardware: Эльбрус 8C2 (32-core e2k v5, 4 NUMA), 125GB DDR4

| Модель | Скорость | Выход (первые 200 символов) | Статус |
|--------|---------:|-----------------------------|--------|
EOF

for MODEL in "${MODELS[@]}"; do
    MPATH="$HOME/gguf_models/$MODEL"
    if [ ! -f "$MPATH" ]; then
        echo "| $MODEL | — | not found | SKIP |" >> "$REPORT"
        continue
    fi

    echo
    echo "=========================================="
    echo "MODEL: $MODEL"
    echo "=========================================="

    LOG="audit_results/${MODEL%.gguf}_log.txt"
    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 2

    # Cleanup stale shm AllReduce segment (предыдущий запуск мог оставить)
    rm -f /dev/shm/prometorch_ddp_29500 2>/dev/null
    PT_MODEL="$MPATH" PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_SPEC_K=1 \
        PT_NO_THINK=1 PT_MAX_TOK=200 PT_CHAT=1 \
        timeout 480 bash scripts/run_tp_elbrus.sh "--temp 0.5" "$PROMPT" \
        > "$LOG" 2>&1

    # Extract speed
    SPEED=$(grep -oP '\(\K[0-9.]+(?= tok/s)' "$LOG" | tail -1)
    SPEED="${SPEED:-FAIL}"

    # Extract output (Full Response section)
    OUTPUT=$(awk '/^--- Full Response/{flag=1; next} flag' "$LOG" | head -8 | tr '\n' ' ' | tr -s ' ')
    OUTPUT=$(echo "${OUTPUT:0:200}" | sed 's/|/\\|/g')

    # Status: did Russian render?
    if [ -z "$OUTPUT" ] || [ "$SPEED" = "FAIL" ]; then
        STATUS="FAIL"
    elif echo "$OUTPUT" | grep -qE '[А-Яа-яЁё]{3,}'; then
        STATUS="OK"
    else
        STATUS="POOR"
    fi

    echo "| ${MODEL%.gguf} | ${SPEED} tok/s | ${OUTPUT} | ${STATUS} |" >> "$REPORT"
    echo "[$MODEL] speed=$SPEED status=$STATUS"
done

echo "=== REPORT ==="
cat "$REPORT"
echo
echo "Saved: $REPORT"
