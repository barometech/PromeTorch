#!/bin/bash
# v5 — bulletproof extractor: regex-based content extraction даже при truncated JSON.
# Извлекает path + content прямо из текста, не пытаясь parse JSON.
set -uo pipefail
cd ~/promethorch

mkdir -p /tmp/promeserve_demo
rm -f /tmp/promeserve_demo/*.html

extract_one() {
    local name="$1"
    local rank0_log="run_logs/multiv3_${name}_rank0.log"
    [ -f "$rank0_log" ] || { echo "Missing $rank0_log"; return 1; }

    python3 <<PYEOF
import re, os
with open("$rank0_log", "r") as f:
    text = f.read()

# Extract response between Prompt tokens line and final tokens-result line
m_resp = re.search(r'\[Generate-TP\] Prompt tokens: \d+\n(.*?)\[Generate-TP\] \d+ tokens', text, re.S)
if not m_resp:
    # Fallback — full body after "Generation"
    m_resp = re.search(r'--- Generation.*?\n(.*?)(?:\[Generate-TP\] \d+ tokens|\Z)', text, re.S)
if not m_resp:
    print("NO_GENERATION_BLOCK")
    raise SystemExit(1)
resp = m_resp.group(1)

# Extract path (greedy stop at next quote)
m_path = re.search(r'"path"\s*:\s*"([^"]+)"', resp)
if not m_path:
    print("NO_PATH")
    print("--- response head 600 ---")
    print(resp[:600])
    raise SystemExit(1)
path = os.path.basename(m_path.group(1))

# Extract content — greedy from "content":" up to either:
#   - last closing quote followed by closing braces (clean case)
#   - or last </html> tag (truncated JSON case)
#   - or end of response (very truncated)
m_content_start = re.search(r'"content"\s*:\s*"', resp)
if not m_content_start:
    print("NO_CONTENT_KEY")
    raise SystemExit(1)
content_raw = resp[m_content_start.end():]

# Try to find clean ending: " followed by } or }} or }}}
m_end = re.search(r'"\s*\}\s*\}\s*(?:\}|<|\Z)', content_raw)
if m_end:
    content_str = content_raw[:m_end.start()]
else:
    # Truncated — find last </html> or </body> or use whole rest
    m_html = re.search(r'</html>', content_raw)
    if m_html:
        content_str = content_raw[:m_html.end()]
    else:
        content_str = content_raw

# Decode JSON-escaped chars via json module trick: wrap into a string and parse
import json as _json
try:
    content = _json.loads('"' + content_str + '"')
except Exception:
    # Fallback manual decode for truncated escape sequences
    s = content_str
    s = s.replace(chr(92)*2, chr(0))  # \\\\ -> \0 placeholder
    s = s.replace(chr(92)+'n', '\n')
    s = s.replace(chr(92)+'t', '\t')
    s = s.replace(chr(92)+'"', '"')
    s = s.replace(chr(92)+'/', '/')
    s = s.replace(chr(0), chr(92))
    content = s

full = "/tmp/promeserve_demo/" + path
with open(full, "w") as f:
    f.write(content)
print(f"WROTE {full} ({len(content)} chars)")
PYEOF
}

echo "============================================"
echo "v5 extractor — robust regex content capture"
echo "============================================"

for project in moscow cosmos ai; do
    echo
    echo "===== $project ====="
    extract_one "$project" || true
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
    head -5 "$f"
    echo "..."
done

echo MULTI_HTML_V5_DONE
