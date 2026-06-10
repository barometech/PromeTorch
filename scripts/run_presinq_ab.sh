#!/bin/bash
# ============================================================================
# run_presinq_ab.sh — A/B качество: обычный GGUF vs pre-SINQ GGUF на Эльбрусе
# ============================================================================
# pre-SINQ (Huawei, Apache-2.0, arXiv:2509.22944) — Sinkhorn-нормализация
# весов ДО GGUF-квантизации. Выход — стандартный GGUF, наш загрузчик читает
# без изменений. Цель A/B: подтвердить (1) загружается, (2) тот же tok/s
# (тот же Q4_K), (3) лучше качество (меньше ошибка квантования).
#
# Запуск на 8C2 (lemur-1): bash scripts/run_presinq_ab.sh "<prompt>"
#
# Модели (~/gguf_models/):
#   baseline : qwen3-4b-Q4_K_M.gguf            (наш стандартный)
#   presinq  : qwen3-4b-presinq-Q4_K_S.gguf    (huawei-csl, скачать заранее)
#
# КАВEAT: baseline = Q4_K_M, presinq = Q4_K_S (репо huawei-csl даёт только S).
# S vs M отличается тем что M использует Q6_K для части attn_v/ffn_down.
# Для чистой изоляции SINQ-эффекта в идеале нужен и обычный Q4_K_S — но
# даже M-vs-S сравнение показывает направление (SINQ должен сгладить
# деградацию русского/математики, которую мы ловили на этой модели).
# ============================================================================
set -u
cd ~/prometorch 2>/dev/null || cd ~/promethorch

PROMPT="${1:-Объясни кратко что такое тензор в машинном обучении.}"
MODELS_DIR="$HOME/gguf_models"
BASELINE="$MODELS_DIR/qwen3-4b-Q4_K_M.gguf"
PRESINQ="$MODELS_DIR/qwen3-4b-presinq-Q4_K_S.gguf"
MAX_TOK="${PT_MAX_TOK:-120}"
OUT_DIR="$HOME/presinq_ab"
mkdir -p "$OUT_DIR"

run_one() {
    local tag="$1" model="$2"
    echo "======================================================"
    echo "  [$tag] $(basename "$model")"
    echo "======================================================"
    if [ ! -f "$model" ]; then
        echo "  SKIP: $model не найден"
        return 1
    fi
    # Тот же TP-4 путь что и baseline 10.9 tok/s. PT_MODEL переопределяет
    # дефолтную модель в run_tp_elbrus.sh.
    PT_MODEL="$model" PT_MAX_TOK="$MAX_TOK" \
        timeout 240 bash scripts/run_tp_elbrus.sh --greedy "$PROMPT" \
        2>&1 | tee "$OUT_DIR/${tag}.log" | tail -25
    echo
}

echo "PROMPT: $PROMPT"
echo
run_one baseline "$BASELINE"
run_one presinq  "$PRESINQ"

echo "======================================================"
echo "  СВОДКА tok/s"
echo "======================================================"
for tag in baseline presinq; do
    SPD=$(grep -oE '[0-9.]+ tok/s' "$OUT_DIR/${tag}.log" 2>/dev/null | tail -1)
    echo "  $tag: ${SPD:-N/A}"
done
echo
echo "Полные логи: $OUT_DIR/{baseline,presinq}.log"
echo "Сравни связность/русский/математику вручную — это quality A/B, не speed."
