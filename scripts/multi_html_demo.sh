#!/bin/bash
# Multi-step tool-call demo через mistral-7B TP-4 — 3 dense HTML проекта
# в одной session, каждый через write_file tool с loop'ом до 5 итераций.
set -uo pipefail
cd ~/promethorch

mkdir -p /tmp/promeserve_demo
rm -f /tmp/promeserve_demo/*.html

# Use mistral-7B TP-4 (8.5 tok/s, 7B size, proven tool-call format)
export PT_MODEL="$HOME/gguf_models/mistral-7b-Q4_K_M.gguf"
export PROMESERVE_TOOL_ROOT=/tmp/promeserve_demo
export PT_MAX_TOK=800

PROMPTS=(
"Создай плотную HTML страницу про Москву. Минимум 5 разделов: история, география, население, культура, экономика. С тегами h1, h2, p, ul, li. Сохрани через write_file как moscow.html"
"Создай плотную HTML страницу про космос. Минимум 5 разделов: звёзды, планеты, галактики, чёрные дыры, космические исследования. Сохрани через write_file как cosmos.html"
"Создай плотную HTML страницу про искусственный интеллект. Минимум 5 разделов: история ИИ, машинное обучение, нейросети, применения, будущее. Сохрани через write_file как ai.html"
)
NAMES=(moscow cosmos ai)

OVERALL_LOG=run_logs/multi_html_demo_$(date +%H%M%S).log
mkdir -p run_logs

for i in 0 1 2; do
    echo "============================================================"
    echo "===== Project $((i+1))/3: ${NAMES[$i]} ====="
    echo "============================================================"
    echo
    bash scripts/tp4_tool_orchestrator.sh "${PROMPTS[$i]}"
    echo
done | tee "$OVERALL_LOG"

echo
echo "=========================================="
echo "=== ВСЕ СГЕНЕРИРОВАННЫЕ HTML ФАЙЛЫ ==="
echo "=========================================="
ls -la /tmp/promeserve_demo/*.html 2>/dev/null
echo
for f in /tmp/promeserve_demo/*.html; do
    [ -f "$f" ] || continue
    echo "--- $f ($(stat -c %s "$f") bytes) ---"
    head -3 "$f"
    echo "..."
done

echo
echo MULTI_HTML_DEMO_DONE
