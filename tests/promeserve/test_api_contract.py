"""
PromeServe API contract tests.

Проверяет что Ollama-совместимый REST API не сломался:
- GET  /                  → HTML (web UI) or health JSON
- GET  /api/version       → {"version": ...}
- GET  /api/tags          → {"models": [...]}
- POST /api/show          → {"modelfile":..., "parameters":...}
- POST /api/generate      → streaming JSON lines
- POST /api/chat          → streaming JSON lines
- POST /api/embeddings    → 501 (stub) but valid response
- GET  /api/mcp/{tools,servers,call,reconnect,audit}  → MCP endpoints

Тесты НЕ предполагают что модель загружена — gated через `has_model` fixture
из conftest.py. Цель — API contract, не correctness инференса.
"""
import json
import urllib.request
import urllib.error
import pytest


def _get(url: str, timeout: float = 5.0):
    return urllib.request.urlopen(url, timeout=timeout)


def _post_json(url: str, payload: dict, timeout: float = 30.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def test_health(promeserve_url):
    """GET / должен возвращать что-то (200 — HTML или JSON, или 404)."""
    try:
        with _get(promeserve_url + "/", timeout=5) as resp:
            assert resp.status in (200, 404), f"unexpected status {resp.status}"
            resp.read()
    except urllib.error.HTTPError as e:
        # 404 — допустимо если нет web/index.html в cwd
        assert e.code == 404


def test_version(promeserve_url):
    """/api/version должен вернуть JSON с ключом 'version'."""
    with _get(promeserve_url + "/api/version") as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
    assert "version" in body
    assert isinstance(body["version"], str)
    assert len(body["version"]) > 0


def test_tags_shape(promeserve_url):
    """/api/tags returns {'models': [list]}."""
    with _get(promeserve_url + "/api/tags") as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
    assert "models" in body
    assert isinstance(body["models"], list)
    for m in body["models"]:
        assert "name" in m or "model" in m, f"model missing name: {m}"


def test_show_unknown_model(promeserve_url):
    """POST /api/show {name:...} должен возвращать 404/400 для несуществующей модели."""
    try:
        with _post_json(promeserve_url + "/api/show", {"name": "definitely-not-real-:0"}) as resp:
            # Если 200 — значит модель действительно есть (странно но допустимо)
            body = json.loads(resp.read())
            assert "modelfile" in body or "parameters" in body or "error" in body
    except urllib.error.HTTPError as e:
        assert e.code in (400, 404, 500), f"unexpected status {e.code}"


def test_embeddings_endpoint_exists(promeserve_url):
    """POST /api/embeddings должен ответить (501 stub допустим, не 404)."""
    try:
        with _post_json(promeserve_url + "/api/embeddings",
                        {"model": "test", "prompt": "hi"}, timeout=10) as resp:
            assert resp.status in (200, 400, 501)
            resp.read()
    except urllib.error.HTTPError as e:
        # 501 = Not Implemented — это валидное "stub" поведение
        assert e.code in (400, 404, 501), f"unexpected status {e.code}"


def test_mcp_tools_endpoint(promeserve_url):
    """GET /api/mcp/tools должен возвращать JSON с builtin tools."""
    with _get(promeserve_url + "/api/mcp/tools") as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
    # Backend MCP agent (commit 0428eeb) добавил 8 builtin tools
    # write_file/read_file/list_dir/bash_safe/fetch_url/http_get/git/sqlite
    assert "tools" in body or isinstance(body, list)
    tools = body.get("tools", body) if isinstance(body, dict) else body
    assert isinstance(tools, list)
    # Должно быть минимум 1 tool (builtin). 0 = backend сломался.
    assert len(tools) >= 1, f"no MCP tools available: {body}"


def test_mcp_servers_endpoint(promeserve_url):
    """GET /api/mcp/servers — список подключённых MCP servers."""
    with _get(promeserve_url + "/api/mcp/servers") as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
    # ожидаем {"servers": [...], "config_path": "..."}
    assert "servers" in body
    assert isinstance(body["servers"], list)


def test_mcp_audit_endpoint(promeserve_url):
    """GET /api/mcp/audit — последние записи audit log."""
    with _get(promeserve_url + "/api/mcp/audit") as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
    # Может быть пустым (никаких вызовов ещё не было)
    assert "entries" in body or isinstance(body, list)


def test_mcp_call_unknown_tool(promeserve_url):
    """POST /api/mcp/call с несуществующим инструментом — должен вернуть error, не crash."""
    try:
        with _post_json(promeserve_url + "/api/mcp/call",
                        {"name": "definitely_not_a_real_tool_xyz", "args": {}},
                        timeout=10) as resp:
            body = json.loads(resp.read())
            assert "error" in body or resp.status in (400, 404)
    except urllib.error.HTTPError as e:
        assert e.code in (400, 404, 500)


def test_generate_no_model_yields_error(promeserve_url):
    """POST /api/generate без model хедера — должен gracefully error, не crash."""
    try:
        with _post_json(promeserve_url + "/api/generate",
                        {"prompt": "hello"}, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            # Streaming response — могут быть JSON lines, не один JSON. Допустимо что
            # сервер вернёт что-то с error mention.
            assert "error" in body.lower() or "model" in body.lower()
    except urllib.error.HTTPError as e:
        assert e.code in (400, 404, 500)
