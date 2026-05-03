#!/bin/bash
# ============================================================================
# tp4_tool_orchestrator.sh — tool-call loop поверх test_gguf_inference TP-4
#
# Юзер ставит task → orchestrator делает loop:
#   1. собирает prompt с system+tools description
#   2. запускает test_gguf_inference --nprocs 4 (TP-4, 11.4 tok/s)
#   3. парсит ответ на <tool_call>{"name":"write_file"...}</tool_call>
#   4. выполняет write_file (sandbox /tmp/promeserve/)
#   5. дописывает <tool_response>...</tool_response> в prompt, goto 1
#   6. max 5 итераций, потом эмитит финальный response
#
# Скорость: 11.4 tok/s lossless TP-4 (qwen3-4b Q4_K_M на Эльбрус 8C2).
# Качество русского: limited на qwen3-4b (см BUG-12). Используем mistral-7b
# или qwen3-8b для production-quality русского.
# ============================================================================

set -uo pipefail
cd ~/promethorch

MODEL="${PT_MODEL:-$HOME/gguf_models/mistral-7b-Q4_K_M.gguf}"
TOOL_ROOT="${PROMESERVE_TOOL_ROOT:-/tmp/promeserve}"
MAX_ITERS=5
MAX_TOK="${PT_MAX_TOK:-600}"

mkdir -p "$TOOL_ROOT" run_logs

USER_TASK="${1:-Создай простую HTML страницу про Москву и сохрани как moscow.html}"

# ---------------------------------------------------------------------------
# Tool registry: write_file (write file under TOOL_ROOT)
# ---------------------------------------------------------------------------
exec_tool() {
    local name="$1"
    local args="$2"
    case "$name" in
    write_file)
        # Parse JSON args → path, content
        local path content
        path=$(echo "$args" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('path',''))")
        content=$(echo "$args" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('content',''), end='')")
        if [ -z "$path" ]; then
            echo "{\"error\":\"missing path\"}"
            return
        fi
        # Sanitize: no abs paths, no ..
        path=$(basename "$path")
        local fp="$TOOL_ROOT/$path"
        printf '%s' "$content" > "$fp"
        local sz=$(stat -c %s "$fp")
        echo "{\"ok\":true,\"path\":\"$fp\",\"bytes\":$sz}"
        ;;
    *)
        echo "{\"error\":\"unknown tool: $name\"}"
        ;;
    esac
}

# ---------------------------------------------------------------------------
# Detect <tool_call>{...}</tool_call> в response. Echo "name||args_json" or empty.
# ---------------------------------------------------------------------------
parse_tool_call() {
    local text="$1"
    python3 - "$text" <<'PY'
import sys, re, json
t = sys.argv[1]
m = re.search(r'<tool_call>(.*?)</tool_call>', t, re.S)
if not m:
    sys.exit(0)
payload = m.group(1).strip()
# Strip ``` wrappers
payload = re.sub(r'^```(?:json)?\s*', '', payload)
payload = re.sub(r'\s*```$', '', payload)
try:
    d = json.loads(payload)
    name = d.get('name','')
    args = json.dumps(d.get('arguments', {}))
    print(f"{name}||{args}")
except Exception as e:
    print(f"PARSE_ERR||{e}", file=sys.stderr)
PY
}

# ---------------------------------------------------------------------------
# System prompt: tools description (write_file)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT='You are a helpful assistant with one tool available.

Tool: write_file
- Description: write a file under sandbox /tmp/promeserve/. Returns {"ok":true,"path":...,"bytes":N}.
- Parameters: {"path":"name.ext","content":"<file body>"}

To use the tool, emit EXACTLY this format:
<tool_call>{"name":"write_file","arguments":{"path":"...","content":"..."}}</tool_call>

After getting <tool_response> from the system, give a 1-sentence final answer.
Do NOT make multiple tool_calls in one turn. Do NOT continue text after the closing </tool_call> tag in the same turn.'

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "=== TP-4 Tool Orchestrator ==="
echo "Model:  $MODEL"
echo "Sandbox: $TOOL_ROOT"
echo "Task:   $USER_TASK"
echo

# Build initial prompt (qwen-style; mistral works similarly through chat template)
PROMPT="<|im_start|>system
$SYSTEM_PROMPT<|im_end|>
<|im_start|>user
$USER_TASK<|im_end|>
<|im_start|>assistant
"

for it in $(seq 1 $MAX_ITERS); do
    echo ">>> Iter $it / $MAX_ITERS"
    # Run test_gguf_inference TP-4 in raw mode (we already supplied template)
    OUT_FILE="run_logs/orch_iter${it}.log"
    PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_SPEC_K=1 PT_NO_THINK=0 \
    PT_MAX_TOK=$MAX_TOK PT_CHAT=0 \
        timeout 240 bash scripts/run_tp_elbrus.sh "--temp 0.5" "$PROMPT" \
        > "$OUT_FILE" 2>&1
    # Extract response (start after "--- Generation" header)
    RESP=$(awk '/^--- Generation/{flag=1; next} /^\[Generate-TP\]/{flag=0} flag' "$OUT_FILE" | head -200)
    echo "--- Response (truncated):"
    echo "${RESP:0:400}"
    echo "..."

    # Detect tool call
    TC=$(parse_tool_call "$RESP")
    if [ -z "$TC" ]; then
        echo ">>> No tool_call detected — final answer reached"
        echo
        echo "=== FINAL ==="
        echo "$RESP"
        break
    fi

    # Execute tool
    NAME="${TC%%||*}"
    ARGS="${TC#*||}"
    echo ">>> Tool call: name=$NAME"
    RESULT=$(exec_tool "$NAME" "$ARGS")
    echo ">>> Tool result: $RESULT"

    # Append to prompt: assistant turn ended with </tool_call>, then tool response
    TRIMMED=$(echo "$RESP" | sed -n 's|.*\(<tool_call>.*</tool_call>\).*|\1|p')
    PROMPT="$PROMPT$TRIMMED<|im_end|>
<|im_start|>tool
<tool_response>$RESULT</tool_response><|im_end|>
<|im_start|>assistant
"
    echo
done

echo
echo "=== Files в $TOOL_ROOT ==="
ls -la "$TOOL_ROOT"/ | head -20
