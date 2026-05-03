#!/bin/bash
# Tool-call HTML demo через PromeServe API. Запускается на эльбрусе.
set -uo pipefail
HOST="${1:-127.0.0.1}"
PORT="${2:-11500}"
BASE="http://$HOST:$PORT/api/chat"
mkdir -p /tmp/promeserve

run_one() {
    local title="$1"
    local fname="$2"
    local task="$3"
    local out="/tmp/promeserve/${fname}_response.json"
    echo
    echo "=== $title → $fname ==="
    curl -s -m 600 -X POST "$BASE" -H "Content-Type: application/json" -d @- > "$out" <<JSON
{
  "model": "/home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf",
  "stream": false,
  "tools": [],
  "messages": [
    {"role":"user","content":"$task"}
  ],
  "options":{"num_predict":700,"temperature":0.5,"repeat_penalty":1.1}
}
JSON
    echo "[response saved] $out"
    head -c 500 "$out"
    echo
    echo "[tool_log:]"
    grep -oP '"tool_log":"[^"]*"' "$out" | head -c 800
    echo
}

run_one "Москва — 1 страница" "moscow" \
    "Создай простую HTML страницу про Москву. Заголовок «Москва - столица России», параграф из 4 предложений, вертикальная (portrait) ориентация (meta viewport), цвет фона светло-голубой #cfe9f5. Сохрани через write_file как moscow.html. После tool_response — короткий итог."

run_one "Меню кафе" "cafe_menu" \
    "Создай HTML страницу-меню для кафе «Прометей». 5 блюд с ценами в рублях, вертикальная компоновка, тёмная тема (фон #1a1a2e белый текст), моноширинный шрифт. Сохрани как menu.html через write_file."

run_one "Визитка разработчика" "card" \
    "Создай HTML визитку разработчика: имя «Промет Эльбрусов», должность «LLM Engineer», 3 контакта (email/github/telegram), 3 навыка (C++, VLIW e2k, LLM inference), вертикальный layout 720×1280, нейтральная серо-синяя палитра. Сохрани как card.html."

run_one "Todo list" "todo" \
    "Создай HTML todo-list страницу с заголовком «Эльбрус LLM TODO» и 5 задачами по оптимизации (1. квантизация, 2. KV cache, 3. kernel fusion, 4. NUMA-aware, 5. SIMD intrinsics). Каждая задача с чекбоксом. Вертикальная ориентация. Сохрани как todo.html."

echo
echo "=== Generated files ==="
ls -la /tmp/promeserve/*.html 2>/dev/null
