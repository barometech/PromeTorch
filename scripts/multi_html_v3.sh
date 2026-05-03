#!/bin/bash
# Multi-HTML demo v3 — qwen3-4B TP-4 (10.9 tok/s) + /no_think hint
# для bypass CoT + max_tokens 1500 + timeout 360s.
set -uo pipefail
cd ~/promethorch
loginctl enable-linger "$USER" 2>/dev/null || true

mkdir -p /tmp/promeserve_demo
rm -f /tmp/promeserve_demo/*.html

MODEL="$HOME/gguf_models/qwen3-4b-Q4_K_M.gguf"
BIN=./build_elbrus/examples/gguf/test_gguf_inference
TOOL_ROOT=/tmp/promeserve_demo
mkdir -p run_logs

run_one_project() {
    local name="$1"
    local task="$2"

    echo "============================================================"
    echo "===== Project: $name ====="
    echo "============================================================"

    pkill -9 -f test_gguf_inference 2>/dev/null
    sleep 2

    # /no_think префикс для qwen3 — bypass CoT thinking, прямой ответ.
    local PROMPT="/no_think Тебе доступен tool: write_file(path, content) — сохраняет файл в /tmp/promeserve_demo/. Вызови его ТОЧНО так:
<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"<filename>\",\"content\":\"<полный HTML>\"}}</tool_call>
$task
Эмитируй ОДИН <tool_call>, без других слов до или после."

    local TS=$(date +%H%M%S)
    # Launch TP-4
    for rank in 0 1 2 3; do
        PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 \
        PT_NUMA_REPLICATE=0 PT_DDP_SHM=1 PT_Q8_SOA=1 \
        PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
        numactl --cpunodebind=$rank --membind=$rank \
        timeout 360 "$BIN" "$MODEL" \
            --nprocs 4 --rank $rank \
            --master-addr 127.0.0.1 --master-port 29500 \
            --max-tokens 1500 --greedy --chat \
            "$PROMPT" \
            > run_logs/multiv3_${name}_rank${rank}.log 2>&1 &
    done
    wait

    # Extract response from rank 0
    local OUT=run_logs/multiv3_${name}_rank0.log
    # Get content after "--- Generation" until next [Generate-TP] line
    local RESP=$(awk '/^--- Generation/{flag=1; next} /^\[Generate-TP\]/{flag=0} flag' "$OUT")

    # Detect and execute tool call
    python3 <<PYEOF
import sys, re, json, os
text = """$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read())[1:-1])')"""
m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.S)
if not m:
    print("NO_TOOL_CALL", file=sys.stderr)
    print("--- response (first 1500 chars) ---", file=sys.stderr)
    print(text[:1500], file=sys.stderr)
    sys.exit(1)
try:
    d = json.loads(m.group(1))
    args = d.get("arguments", {})
    path = os.path.basename(args.get("path", "out.html"))
    content = args.get("content", "")
    full = "/tmp/promeserve_demo/" + path
    with open(full, "w") as f:
        f.write(content)
    print(f"WROTE {full} ({len(content)} chars)")
except Exception as e:
    print(f"PARSE_ERR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    echo
}

run_one_project "moscow" "Создай HTML страницу про Москву на русском. 5 разделов: история, география, население, культура, экономика. Каждый раздел 2-3 предложения. Теги h1, h2, p, ul, li. Сохрани как moscow.html."
run_one_project "cosmos" "Создай HTML страницу про космос на русском. 5 разделов: звёзды, планеты, галактики, чёрные дыры, исследования. Каждый раздел 2-3 предложения. Теги h1, h2, p, ul, li. Сохрани как cosmos.html."
run_one_project "ai" "Создай HTML страницу про искусственный интеллект на русском. 5 разделов: история, машинное обучение, нейросети, применения, будущее. Каждый раздел 2-3 предложения. Теги h1, h2, p, ul, li. Сохрани как ai.html."

echo
echo "=========================================="
echo "=== ВСЕ HTML ФАЙЛЫ ==="
echo "=========================================="
ls -la /tmp/promeserve_demo/*.html 2>/dev/null
echo
for f in /tmp/promeserve_demo/*.html; do
    [ -f "$f" ] || continue
    echo "--- $f ($(stat -c %s "$f") bytes) ---"
    head -3 "$f"
    echo "..."
done

echo MULTI_HTML_V3_DONE
