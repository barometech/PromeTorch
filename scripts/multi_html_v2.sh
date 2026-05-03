#!/bin/bash
# Multi-step tool-call demo v2 — qwen3-4B SP с native chat template (--chat).
# Один процесс, один model load, 3 разных prompts через python orchestrator.
# Каждый prompt → model emits <tool_call> → bash executes write_file →
# Прокачивает ответ обратно → final answer. Loop max 3 iter.
set -uo pipefail
cd ~/promethorch

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

    # Single full prompt: tool descr + task. --chat let model wrap chat template.
    # Tool format универсальный: <tool_call>{"name":..., "arguments":{...}}</tool_call>
    local PROMPT="Тебе доступен ОДИН tool: write_file(path, content) — сохраняет файл в sandbox /tmp/promeserve_demo/. Вызови tool ТОЧНО в формате:
<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"<filename.html>\",\"content\":\"<полный HTML с тегами>\"}}</tool_call>
$task
Ответ: только один <tool_call>, без других слов до или после."

    local OUT=run_logs/multiv2_${name}.log
    env PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=32 \
        PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1 \
        timeout 240 "$BIN" "$MODEL" \
        --max-tokens 2000 --greedy --chat --temp 0 \
        "$PROMPT" > "$OUT" 2>&1

    # Extract response between markers
    local RESP=$(awk '/^--- Generation/{flag=1; next} /^\[Generate\]/{flag=0} flag' "$OUT")

    # Detect and execute tool call
    python3 - <<PY
import sys, re, json, os
text = """$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read())[1:-1])')"""
m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.S)
if not m:
    print("NO_TOOL_CALL", file=sys.stderr)
    print("--- response (first 500) ---", file=sys.stderr)
    print(text[:500], file=sys.stderr)
    sys.exit(1)
try:
    d = json.loads(m.group(1))
    if d.get("name") != "write_file":
        print(f"WRONG_TOOL: {d.get('name')}", file=sys.stderr); sys.exit(1)
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
PY

    echo
}

run_one_project "moscow" "Создай ПЛОТНУЮ HTML страницу про Москву на русском. Минимум 7 разделов: история, география, климат, население, культура, экономика, достопримечательности. Каждый раздел минимум 3 предложения. Используй h1, h2, h3, p, ul, li теги. Сохрани как moscow.html."
run_one_project "cosmos" "Создай ПЛОТНУЮ HTML страницу про космос на русском. Минимум 7 разделов: звёзды, планеты, галактики, чёрные дыры, тёмная материя, экзопланеты, исследования. Каждый раздел минимум 3 предложения. Используй h1, h2, h3, p, ul, li теги. Сохрани как cosmos.html."
run_one_project "ai" "Создай ПЛОТНУЮ HTML страницу про искусственный интеллект на русском. Минимум 7 разделов: история ИИ, машинное обучение, глубокое обучение, нейросети, NLP, компьютерное зрение, будущее. Каждый раздел минимум 3 предложения. Используй h1, h2, h3, p, ul, li теги. Сохрани как ai.html."

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

echo MULTI_HTML_V2_DONE
