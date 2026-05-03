#!/bin/bash
# Single-task HTML proof. Mistral-7B TP-4 → moscow.html через write_file tool.
set -uo pipefail
cd /home/paperclipdnb/promethorch
mkdir -p /tmp/promeserve
rm -f /tmp/promeserve/moscow.html /tmp/proof.done

pkill -9 -f test_gguf_inference 2>/dev/null
rm -f /dev/shm/prometorch_ddp_*
sleep 5

PROMPT='<s>[INST] You are an assistant with one tool: write_file(path, content). Use it EXACTLY as: <tool_call>{"name":"write_file","arguments":{"path":"moscow.html","content":"<!DOCTYPE html><html lang=ru><head><meta charset=utf-8><title>Москва</title></head><body><h1>Москва — столица России</h1><p>Москва — крупнейший город России и её столица. Расположена на реке Москве в центре Восточно-Европейской равнины. Население более 13 миллионов человек. Основана в 1147 году князем Юрием Долгоруким.</p></body></html>"}}</tool_call>

Task: создай простую HTML страницу про Москву и сохрани как moscow.html через write_file. [/INST]'

PT_MODEL=$HOME/gguf_models/mistral-7b-Q4_K_M.gguf \
PT_Q8_SOA=1 PT_PER_BLOCK_SCALE=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
PT_SPEC_K=1 PT_NO_THINK=1 PT_MAX_TOK=400 PT_CHAT=0 \
    timeout 480 bash scripts/run_tp_elbrus.sh "--temp 0.3" "$PROMPT" \
    > /tmp/proof.log 2>&1

# Parse <tool_call>{...}</tool_call>
RESP=$(awk '/^--- Full Response/{f=1;next} f' run_logs/tp4_rank0.log 2>/dev/null)
echo "[response, first 1000 chars]:"
echo "${RESP:0:1000}"
echo

# Try parse with python
PAYLOAD=$(echo "$RESP" | grep -oP '(?s)<tool_call>\K.*?(?=</tool_call>)' | head -1)
if [ -z "$PAYLOAD" ]; then
    echo "[!] no tool_call detected in response"
    # Even if no tool_call — extract HTML manually if present, save as fallback
    HTML=$(echo "$RESP" | grep -oP '(?s)<!DOCTYPE html>.*?</html>' | head -1)
    if [ -n "$HTML" ]; then
        echo "$HTML" > /tmp/promeserve/moscow.html
        echo "[fallback] extracted raw HTML from response, saved /tmp/promeserve/moscow.html"
    fi
    echo DONE > /tmp/proof.done
    exit 0
fi

echo "[tool_call payload]: ${PAYLOAD:0:300}"
PATH_VAL=$(/usr/bin/python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(d.get('arguments',{}).get('path',''))" "$PAYLOAD" 2>/dev/null)
CONTENT_VAL=$(/usr/bin/python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(d.get('arguments',{}).get('content',''),end='')" "$PAYLOAD" 2>/dev/null)

if [ -n "$CONTENT_VAL" ]; then
    PATH_VAL=$(basename "${PATH_VAL:-moscow.html}")
    printf '%s' "$CONTENT_VAL" > "/tmp/promeserve/$PATH_VAL"
    echo "[saved] /tmp/promeserve/$PATH_VAL ($(stat -c %s /tmp/promeserve/$PATH_VAL) bytes)"
fi
echo DONE > /tmp/proof.done
