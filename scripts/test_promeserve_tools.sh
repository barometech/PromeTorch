#!/bin/bash
# ============================================================================
# test_promeserve_tools.sh — multi-step tool-call demo через PromeServe
#
# Юзер ставит 1 task → PromeServe сам делает loop:
#   1. модель эмитит <tool_call>{"name":"write_file",...}</tool_call>
#   2. server выполняет write_file (saves в /tmp/promeserve/)
#   3. server append'ит <tool_response> в prompt
#   4. модель продолжает; max 5 iterations
#   5. финальный response отдаётся клиенту одним JSON
#
# Тестируется на mistral-7b (нормально работает на русском у нас) +
# фикс LM_HEAD_FP=1 для cyrillic precision.
#
# Required: PromeServe запущен на 127.0.0.1:11434 на эльбрусе.
# ============================================================================

set -euo pipefail
PORT="${PROMESERVE_PORT:-11434}"
HOST="${PROMESERVE_HOST:-127.0.0.1}"
MODEL="${PROMESERVE_MODEL:-mistral-7b-Q4_K_M.gguf}"
TOOL_ROOT="${PROMESERVE_TOOL_ROOT:-/tmp/promeserve}"

mkdir -p "$TOOL_ROOT"

# Helper: HTTP POST + extract message.content
chat() {
    local task="$1"
    curl -s -X POST "http://$HOST:$PORT/api/chat" \
        -H "Content-Type: application/json" \
        --data @- <<JSON
{
  "model": "$MODEL",
  "stream": false,
  "tools": [],
  "messages": [
    {"role": "user", "content": "$task"}
  ],
  "options": {
    "num_predict": 600,
    "temperature": 0.7,
    "repeat_penalty": 1.1
  }
}
JSON
}

# ----------------------------------------------------------------------------
# Tasks: 4 разных HTML страницы с вертикальной ориентацией.
# Каждая модель должна emit <tool_call>{"name":"write_file"...}</tool_call>
# с HTML контентом в arguments.content.
# ----------------------------------------------------------------------------

TASK1='Создай простую HTML страницу про Москву. Заголовок "Москва - столица России", \
один абзац с описанием города (3-4 предложения), вертикальная ориентация \
(meta viewport portrait), цвет фона светло-голубой. Сохрани файл как \
moscow.html через инструмент write_file.'

TASK2='Создай HTML страницу-меню для кафе "Прометей". 5 блюд с ценами в рублях, \
вертикальная компоновка, тёмная тема, моноширинный шрифт. Сохрани как menu.html \
через write_file.'

TASK3='Создай HTML визитку для разработчика на Эльбрусе. Имя, должность, контакты, \
3 навыка (C++, VLIW e2k, LLM inference), вертикальный layout, нейтральная палитра. \
Сохрани как card.html через write_file.'

TASK4='Создай HTML простую todo-list страницу с 5 задачами по оптимизации LLM \
(квантизация, KV cache, kernel fusion, NUMA-aware, SIMD intrinsics). \
Чекбоксы, заголовок, вертикальная ориентация. Сохрани как todo.html через write_file.'

echo "=== PromeServe Tool-Call HTML Demo ==="
echo "Server: http://$HOST:$PORT"
echo "Model:  $MODEL"
echo "Output: $TOOL_ROOT/"
echo

for i in 1 2 3 4; do
    var="TASK$i"
    task="${!var}"
    echo "--- Task $i ---"
    echo "Prompt: ${task:0:80}..."
    out=$(chat "$task")
    echo "Response (first 300 chars):"
    echo "$out" | python3 -c "import sys,json
try:
    d = json.load(sys.stdin)
    msg = d.get('message',{}).get('content','')
    print(msg[:300])
    log = d.get('tool_log','')
    if log:
        print('--- tool_log ---')
        print(log[:500])
except Exception as e:
    print('parse error:', e)
    print(sys.stdin.read()[:500])"
    echo
done

echo "=== Files generated in $TOOL_ROOT ==="
ls -la "$TOOL_ROOT"/*.html 2>/dev/null || echo "(no HTML files)"
