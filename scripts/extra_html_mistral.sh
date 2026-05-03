#!/bin/bash
# Generate one extra dense HTML via mistral-7B TP-4 (no CoT, cleaner output).
# Demonstrate that mistral instruction-following also works with same tool-call format.
set -uo pipefail
cd ~/promethorch
loginctl enable-linger "$USER" 2>/dev/null || true

mkdir -p /tmp/promeserve_demo

MODEL="$HOME/gguf_models/mistral-7b-Q4_K_M.gguf"
BIN=./build_elbrus/examples/gguf/test_gguf_inference
mkdir -p run_logs

# Single project: "Программирование на Python" — testing a different topic
NAME="python"
TASK="Создай HTML страницу про программирование на Python на русском. 5 разделов: история, синтаксис, библиотеки, применение, будущее. Каждый раздел 2-3 предложения. Теги h1, h2, p, ul, li. Сохрани через write_file как python.html."

PROMPT="Тебе доступен tool: write_file(path, content) — сохраняет файл в /tmp/promeserve_demo/. Вызови его в формате:
<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"<filename>\",\"content\":\"<полный HTML>\"}}</tool_call>
$TASK
Эмитируй ОДИН <tool_call>, без других слов до или после."

echo "=== mistral-7B TP-4 single-project demo ==="
date +"Start: %F %T"

pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2
rm -f /dev/shm/prometorch_ddp_*

for rank in 0 1 2 3; do
    PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 \
    PT_NUMA_REPLICATE=0 PT_DDP_SHM=1 PT_Q8_SOA=1 \
    numactl --cpunodebind=$rank --membind=$rank \
    timeout 360 "$BIN" "$MODEL" \
        --nprocs 4 --rank $rank \
        --master-addr 127.0.0.1 --master-port 29500 \
        --max-tokens 1500 --greedy --chat \
        "$PROMPT" \
        > run_logs/extra_${NAME}_rank${rank}.log 2>&1 &
done
wait

# Extract and write file via lenient parser (same as v5)
python3 <<PYEOF
import re, os
with open("run_logs/extra_${NAME}_rank0.log", "r") as f:
    text = f.read()
m_resp = re.search(r'\[Generate-TP\] Prompt tokens: \d+\n(.*?)\[Generate-TP\] \d+ tokens', text, re.S)
if not m_resp:
    print("NO_GENERATION_BLOCK"); raise SystemExit(1)
resp = m_resp.group(1)

m_path = re.search(r'"path"\s*:\s*"([^"]+)"', resp)
if not m_path:
    print("NO_PATH"); print(resp[:600]); raise SystemExit(1)
path = os.path.basename(m_path.group(1))

m_content_start = re.search(r'"content"\s*:\s*"', resp)
content_raw = resp[m_content_start.end():]
m_end = re.search(r'"\s*\}\s*\}\s*(?:\}|<|\Z)', content_raw)
if m_end:
    content_str = content_raw[:m_end.start()]
else:
    m_html = re.search(r'</html>', content_raw)
    content_str = content_raw[:m_html.end()] if m_html else content_raw

import json as _json
try:
    content = _json.loads('"' + content_str + '"')
except Exception:
    s = content_str
    s = s.replace(chr(92)*2, chr(0)).replace(chr(92)+'n', '\n').replace(chr(92)+'t', '\t').replace(chr(92)+'"', '"').replace(chr(92)+'/', '/').replace(chr(0), chr(92))
    content = s

full = "/tmp/promeserve_demo/" + path
with open(full, "w") as f:
    f.write(content)
print(f"WROTE {full} ({len(content)} chars)")
PYEOF

echo
ls -la /tmp/promeserve_demo/python.html 2>&1
echo EXTRA_DONE
