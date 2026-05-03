#!/bin/bash
# All-in-one: start PromeServe, wait ready, run 4 HTML tasks, stop server.
set -uo pipefail
cd /home/paperclipdnb/promethorch

pkill -9 -f promeserve 2>/dev/null
pkill -9 -f test_gguf 2>/dev/null
sleep 5
rm -f /dev/shm/prometorch_ddp_*
mkdir -p /tmp/promeserve_demo

# Start PromeServe
nohup setsid ./build_elbrus/promeserve/promeserve \
    --port 11500 --device cpu \
    --model /home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf \
    </dev/null >/tmp/pserve.log 2>&1 & disown

PSERVE_PID=$!
echo "[$(date +%T)] PromeServe started, PID=$PSERVE_PID"

# Wait ready (max 5 min)
for i in {1..60}; do
    sleep 5
    if curl -s -m 2 http://127.0.0.1:11500/api/version 2>/dev/null | grep -q version; then
        echo "[$(date +%T)] PromeServe ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "[$(date +%T)] PromeServe failed to start"
        tail -30 /tmp/pserve.log
        exit 1
    fi
done

# Run 4 tasks
declare -a TASKS=(
    "moscow|Создай простую HTML страницу про Москву. Заголовок «Москва - столица России», абзац из 4 предложений, вертикальная (portrait) ориентация, фон светло-голубой #cfe9f5. Используй tool write_file чтобы сохранить как moscow.html."
    "menu|Создай HTML меню кафе «Прометей». 5 блюд с ценами в рублях, вертикальная компоновка, темный фон #1a1a2e. Используй tool write_file чтобы сохранить как menu.html."
    "card|Создай HTML визитку «Промет Эльбрусов» (LLM Engineer): 3 контакта, 3 навыка, вертикальная 720x1280, серо-синяя палитра. Tool write_file → card.html."
    "todo|Создай HTML todo с заголовком «Эльбрус LLM TODO», 5 задач с чекбоксами (квантизация, KV cache, kernel fusion, NUMA, SIMD). Tool write_file → todo.html."
)

for entry in "${TASKS[@]}"; do
    name="${entry%%|*}"
    task="${entry#*|}"
    echo
    echo "[$(date +%T)] === $name ==="
    out="/tmp/promeserve_demo/${name}_resp.json"
    curl -s -m 600 -X POST http://127.0.0.1:11500/api/chat \
        -H "Content-Type: application/json" \
        -d "$(cat <<JSON
{
  "model": "/home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf",
  "stream": false,
  "tools": [],
  "messages": [{"role":"user","content":"$task"}],
  "options":{"num_predict":700,"temperature":0.5,"repeat_penalty":1.1}
}
JSON
)" > "$out" 2>&1
    echo "[response saved: $out, $(stat -c %s "$out") bytes]"
    head -c 600 "$out"
    echo
done

echo
echo "=== Generated HTML files ==="
ls -la /tmp/promeserve/ 2>/dev/null
echo "=== Demo done ==="
echo DONE > /tmp/demo.done
