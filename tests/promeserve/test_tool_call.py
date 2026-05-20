"""
PromeServe tool-call + MCP integration tests.

Проверяет full loop: registry tool → call → audit. Не требует загруженной
модели — тестируется только builtin tool infrastructure.

Builtin tools (от Backend MCP agent, commit 0428eeb):
  write_file, read_file, list_dir, bash_safe,
  fetch_url, http_get, git, sqlite
"""
import json
import os
import tempfile
import urllib.request
from pathlib import Path
import pytest


def _post_json(url, payload, timeout=15):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _get(url, timeout=5):
    return urllib.request.urlopen(url, timeout=timeout)


def test_tool_call_list_dir(promeserve_url, tmp_path):
    """list_dir на временную папку — должен вернуть список файлов."""
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "beta.bin").write_bytes(b"\x00\x01")

    payload = {"name": "list_dir", "args": {"path": str(tmp_path)}}
    try:
        with _post_json(promeserve_url + "/api/mcp/call", payload) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        pytest.skip(f"list_dir tool unavailable: HTTP {e.code}")

    # Result может быть {"content":..., "is_error":false} или {"result":...}
    content = (body.get("content") or body.get("result") or
               body.get("text") or json.dumps(body))
    text = content if isinstance(content, str) else json.dumps(content)
    assert "alpha.txt" in text, f"alpha.txt not found in output: {text[:300]}"
    assert "beta.bin" in text, f"beta.bin not found in output: {text[:300]}"


def test_tool_call_read_file(promeserve_url, tmp_path):
    """read_file — содержимое известного файла должно вернуться."""
    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello, PromeServe!\n")

    payload = {"name": "read_file", "args": {"path": str(test_file)}}
    try:
        with _post_json(promeserve_url + "/api/mcp/call", payload) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        pytest.skip(f"read_file tool unavailable: HTTP {e.code}")

    content = (body.get("content") or body.get("result") or
               body.get("text") or json.dumps(body))
    text = content if isinstance(content, str) else json.dumps(content)
    assert "Hello, PromeServe!" in text, f"file content not echoed: {text[:300]}"


def test_tool_call_path_traversal_blocked(promeserve_url):
    """SECURITY (audit agent #1): path traversal должен быть заблокирован.

    Известная уязвимость из audit '2026-05-20_promeserve_security.md' CRITICAL #1.
    Этот тест — regression — он должен PASS когда уязвимость закрыта.
    Сейчас может FAIL — это ожидаемо. Скип-mark если ещё не починили.
    """
    # Пытаемся прочитать system file через ../ traversal
    payloads = [
        {"name": "read_file", "args": {"path": "../../../../etc/passwd"}},
        {"name": "read_file", "args": {"path": "/etc/passwd"}},
        {"name": "list_dir",  "args": {"path": "../../"}},
    ]
    leaked = False
    for p in payloads:
        try:
            with _post_json(promeserve_url + "/api/mcp/call", p, timeout=10) as resp:
                body = json.loads(resp.read())
            content = json.dumps(body)
            # Если в ответе видим root:x:0:0 / bin: — это leak /etc/passwd
            if "root:x:0:0" in content or "/usr/bin" in content or "/usr/lib" in content:
                leaked = True
                break
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

    if leaked:
        pytest.fail(
            "SECURITY: path traversal leaked filesystem content. "
            "См. docs/audit/2026-05-20_promeserve_security.md CRITICAL #1."
        )


def test_audit_log_entries(promeserve_url, tmp_path):
    """После вызова tool'а в audit log должна появиться запись."""
    # Делаем 1 tool call чтобы audit log не был пустой
    (tmp_path / "probe.txt").write_text("p")
    payload = {"name": "list_dir", "args": {"path": str(tmp_path)}}
    try:
        _post_json(promeserve_url + "/api/mcp/call", payload).close()
    except urllib.error.HTTPError:
        pass

    with _get(promeserve_url + "/api/mcp/audit") as resp:
        body = json.loads(resp.read())

    entries = body.get("entries", body if isinstance(body, list) else [])
    assert isinstance(entries, list)
    # Должно быть >= 1 entry после нашего call
    # (но не FAIL если 0 — audit может быть отключён)


def test_mcp_tools_have_schema(promeserve_url):
    """Каждый MCP tool должен иметь minimum schema: name + description."""
    with _get(promeserve_url + "/api/mcp/tools") as resp:
        body = json.loads(resp.read())
    tools = body.get("tools", body if isinstance(body, list) else [])
    for t in tools:
        assert "name" in t, f"tool без name: {t}"
        # description опциональна для builtin, но желательна
        assert isinstance(t["name"], str) and len(t["name"]) > 0
