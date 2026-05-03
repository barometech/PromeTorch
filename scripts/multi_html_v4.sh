#!/bin/bash
# Multi-HTML demo v4 — lenient parser (handles missing </tool_call> closing tag).
# Use existing run_logs/ от v3b — модель УЖЕ сгенерировала tool_calls, нужно только распарсить.
set -uo pipefail
cd ~/promethorch

mkdir -p /tmp/promeserve_demo
rm -f /tmp/promeserve_demo/*.html

# Lenient parser — matches <tool_call> + JSON (with or without closing tag)
parse_and_save() {
    local name="$1"
    local rank0_log="run_logs/multiv3_${name}_rank0.log"
    [ -f "$rank0_log" ] || { echo "Missing $rank0_log"; return 1; }

    python3 <<PYEOF
import re, json, os
with open("$rank0_log", "r") as f:
    text = f.read()

# Extract response — start after "Prompt tokens: NNN" line, end at "[Generate-TP] NNN tokens in"
m_resp = re.search(r'\[Generate-TP\] Prompt tokens: \d+\n(.*?)\[Generate-TP\] \d+ tokens', text, re.S)
if not m_resp:
    print("NO_GENERATION_BLOCK")
    raise SystemExit(1)
resp = m_resp.group(1)

# Lenient: <tool_call>{...} (closing optional)
m = re.search(r'<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|\$|\Z)', resp, re.S)
if not m:
    # Even more lenient: find { "name": "write_file", "arguments": {...} }
    m = re.search(r'(\{\s*"name"\s*:\s*"write_file"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})\s*\}\s*\}', resp, re.S)
    if not m:
        m = re.search(r'(\{\s*"name"\s*:\s*"write_file".*?"content"\s*:\s*"(?:[^"\\\\]|\\\\.)*?"\s*\}\s*\})', resp, re.S)
    if not m:
        print("NO_TOOL_CALL_FOUND")
        print("--- response head ---")
        print(resp[:800])
        raise SystemExit(1)

raw = m.group(1)
# Try parse — if JSON ends without closing braces, append them
try:
    d = json.loads(raw)
except json.JSONDecodeError:
    # Try adding closing braces progressively
    for suffix in ["}", "}}", "}}}", "\"}}"]:
        try:
            d = json.loads(raw + suffix)
            break
        except json.JSONDecodeError:
            continue
    else:
        print(f"PARSE_ERR: {raw[:300]}")
        raise SystemExit(1)

if d.get("name") != "write_file":
    print(f"WRONG_TOOL: {d.get('name')}")
    raise SystemExit(1)

args = d.get("arguments", {})
path = os.path.basename(args.get("path", "$name.html"))
content = args.get("content", "")
full = "/tmp/promeserve_demo/" + path
with open(full, "w") as f:
    f.write(content)
print(f"WROTE {full} ({len(content)} chars)")
PYEOF
}

echo "============================================"
echo "Lenient parser — re-process v3b run_logs"
echo "============================================"

for project in moscow cosmos ai; do
    echo
    echo "===== $project ====="
    parse_and_save "$project" || true
done

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

echo MULTI_HTML_V4_DONE
