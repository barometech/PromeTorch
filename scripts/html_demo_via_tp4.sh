#!/bin/bash
# Tool-call HTML demo через test_gguf_inference TP-4 (без PromeServe HTTP).
# 1 запрос пользователя → bash оркестрирует tool-call loop вокруг binary.
set -uo pipefail
cd /home/paperclipdnb/promethorch
mkdir -p /tmp/promeserve

MODEL="$HOME/gguf_models/mistral-7b-Q4_K_M.gguf"
SYS_PROMPT='You are a helpful assistant with one tool: write_file(path, content). To use it emit EXACTLY: <tool_call>{"name":"write_file","arguments":{"path":"NAME.html","content":"<html>...</html>"}}</tool_call>. After receiving <tool_response> from system, give a 1-sentence final answer. Do not call multiple tools per turn.'

run_one() {
    local name="$1"
    local task="$2"
    local prompt="<s>[INST] ${SYS_PROMPT}

User task: ${task} [/INST]"

    echo
    echo "=== $name ==="
    pkill -9 -f test_gguf_inference 2>/dev/null
    rm -f /dev/shm/prometorch_ddp_*
    sleep 3

    # Iter 1: model emits tool_call
    PT_MODEL="$MODEL" PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
    PT_SPEC_K=1 PT_NO_THINK=1 PT_MAX_TOK=600 PT_CHAT=0 \
        timeout 600 bash scripts/run_tp_elbrus.sh "--temp 0.5" "$prompt" \
        > "/tmp/promeserve/${name}_iter1.log" 2>&1

    local resp1=$(awk '/^--- Full Response/{flag=1; next} flag' run_logs/tp4_rank0.log 2>/dev/null)
    echo "[iter1 response, first 600 chars]:"
    echo "${resp1:0:600}"

    # Parse <tool_call>
    local tc_payload=$(echo "$resp1" | grep -oP '(?s)<tool_call>\K.*?(?=</tool_call>)' | head -1)
    if [ -z "$tc_payload" ]; then
        echo "[!] no tool_call detected, taking response as final answer"
        return
    fi
    echo "[tool_call payload]:"
    echo "$tc_payload" | head -c 400
    echo

    # Extract path + content via simple python
    local tool_path=$(/usr/bin/python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('arguments', {}).get('path', ''))
except Exception as e:
    print('PARSE_ERR', file=sys.stderr)
" "$tc_payload" 2>/dev/null)
    local tool_content=$(/usr/bin/python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('arguments', {}).get('content', ''), end='')
except Exception as e:
    pass
" "$tc_payload" 2>/dev/null)

    if [ -z "$tool_path" ] || [ -z "$tool_content" ]; then
        echo "[!] failed to parse tool_call args"
        return
    fi

    # Sandbox: only basename, only /tmp/promeserve/
    tool_path=$(basename "$tool_path")
    local fp="/tmp/promeserve/$tool_path"
    printf '%s' "$tool_content" > "$fp"
    local sz=$(stat -c %s "$fp")
    echo "[tool exec] wrote $fp ($sz bytes)"
    echo "${name}|${tool_path}|${sz}|first_response_chars=${#resp1}" >> /tmp/demo_summary.csv
}

> /tmp/demo_summary.csv

run_one moscow "Создай HTML страницу про Москву. Заголовок «Москва - столица России», абзац из 4 предложений. Сохрани как moscow.html через write_file."
run_one menu   "Создай HTML меню кафе «Прометей»: 5 блюд с ценами в рублях. Сохрани как menu.html через write_file."
run_one card   "Создай HTML визитку «Промет Эльбрусов» (LLM Engineer): имя, должность, 3 контакта, 3 навыка. Сохрани как card.html через write_file."
run_one todo   "Создай HTML todo-list страницу с заголовком «Эльбрус LLM TODO» и 5 задачами с чекбоксами. Сохрани как todo.html через write_file."

echo
echo "=== HTML files ==="
ls -la /tmp/promeserve/*.html 2>/dev/null
echo
echo "=== summary ==="
cat /tmp/demo_summary.csv
echo DONE > /tmp/demo.done
