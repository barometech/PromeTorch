# PromeServe Integration Tests

Pytest battery которая запускает реальный `promeserve` бинарник и говорит с ним по HTTP.

## Запуск

```bash
# Authoritative path (если установлен)
export PROMESERVE_BIN=/path/to/promeserve
pytest tests/promeserve/

# Авто-поиск (build_cpu_gguf/promeserve.exe, build_promeserve/, build_elbrus/)
pytest tests/promeserve/

# Свой порт (default 18434)
PROMESERVE_TEST_PORT=19999 pytest tests/promeserve/

# Verbose
pytest -v tests/promeserve/
```

## Структура

- `conftest.py` — session-scoped fixture: стартует promeserve один раз
- `test_api_contract.py` — Ollama-compatible REST API contract
- `test_tool_call.py` — MCP builtin tools + audit + security regressions

## Что покрыто

| Endpoint | Тест |
|----------|------|
| `GET /` | health |
| `GET /api/version` | shape |
| `GET /api/tags` | shape + список моделей |
| `POST /api/show` | unknown model → 404 |
| `POST /api/generate` | no-model → graceful error |
| `POST /api/embeddings` | 501 stub допустим |
| `GET /api/mcp/tools` | минимум 1 builtin tool |
| `GET /api/mcp/servers` | shape |
| `GET /api/mcp/audit` | shape |
| `POST /api/mcp/call` | unknown tool → error |
| `POST /api/mcp/call` builtin | list_dir, read_file работают |
| **SECURITY** | path traversal → blocked (regression test) |

## Покрытый бэклог багов

- **Test gap audit (2026-05-20):** PromeServe был 6354 LoC и 0 tests → этот suite закрывает базовый API contract.
- **Security audit (2026-05-20):** CRITICAL #1 (path traversal в `sandbox_path()`) — regression test `test_tool_call_path_traversal_blocked` в `test_tool_call.py`. Сейчас может FAIL — fix в roadmap.
