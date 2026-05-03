#!/bin/bash
# Single ssh-window demo. Запускает PromeServe + 1 task → HTML.
set -uo pipefail
cd /home/paperclipdnb/promethorch

pkill -9 -f promeserve 2>/dev/null
pkill -9 -f test_gguf 2>/dev/null
sleep 5
rm -f /dev/shm/prometorch_ddp_*
mkdir -p /tmp/promeserve

# Start PromeServe в foreground process group
(./build_elbrus/promeserve/promeserve --port 11500 --device cpu \
    --model /home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf \
    > /tmp/pserve.log 2>&1) &
PSERVE_PID=$!
echo "[$(date +%T)] PromeServe PID=$PSERVE_PID"

# Wait for ready
for i in {1..60}; do
    sleep 5
    if curl -s -m 2 http://127.0.0.1:11500/api/version 2>/dev/null | grep -q version; then
        echo "[$(date +%T)] READY"
        break
    fi
done

if ! curl -s -m 2 http://127.0.0.1:11500/api/version 2>/dev/null | grep -q version; then
    echo "[$(date +%T)] FAILED to start"
    tail -50 /tmp/pserve.log
    kill $PSERVE_PID 2>/dev/null
    exit 1
fi

# Send 4 HTML tasks
for entry in \
    "moscow|Создай простую HTML страницу про Москву. Заголовок «Москва - столица России», абзац из 4 предложений. Используй tool write_file чтобы сохранить как moscow.html. Затем кратко опиши что сделал." \
    "menu|Создай HTML меню кафе «Прометей»: 5 блюд с ценами в рублях, тёмный фон. Используй tool write_file чтобы сохранить как menu.html." \
    "card|Создай HTML визитку «Промет Эльбрусов» (LLM Engineer): 3 контакта, 3 навыка, серо-синяя палитра. Tool write_file → card.html." \
    "todo|Создай HTML todo list с заголовком «Эльбрус LLM TODO» и 5 чекбоксами (квантизация, KV cache, kernel fusion, NUMA, SIMD). Tool write_file → todo.html." \
; do
    name="${entry%%|*}"
    task="${entry#*|}"
    echo
    echo "[$(date +%T)] === $name ==="
    out="/tmp/promeserve/${name}_resp.json"
    curl -s -m 600 -X POST http://127.0.0.1:11500/api/chat \
        -H "Content-Type: application/json" \
        --data-raw "{\"model\":\"/home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf\",\"stream\":false,\"tools\":[],\"messages\":[{\"role\":\"user\",\"content\":\"$task\"}],\"options\":{\"num_predict\":700,\"temperature\":0.5,\"repeat_penalty\":1.1}}" \
        > "$out" 2>&1
    sz=$(stat -c %s "$out" 2>/dev/null || echo 0)
    echo "[$(date +%T)] $name: $sz bytes"
    head -c 600 "$out"
    echo
done

echo
echo "=== HTML files ==="
ls -la /tmp/promeserve/*.html 2>&1

echo
echo "=== STOPPING ==="
kill $PSERVE_PID 2>/dev/null
sleep 2
echo DONE
