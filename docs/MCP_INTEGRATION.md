# PromeServe — MCP Integration Guide

PromeServe — это HTTP-сервер на C++, который понимает Ollama-compatible API
для LLM inference на PromeTorch. С релиза 0.2 поддерживается:

1. **Tool-call loop (Этап 1)** — Hermes/Qwen3 формат `<tool_call>{...}</tool_call>`,
   parallel calls, audit log, max-iter guard, расширенный builtin набор
   (file ops, bash, fetch, web_search, python, git, sqlite).
2. **MCP client (Этап 2)** — подключение к external MCP server'ам через
   JSON-RPC 2.0 over stdio. Tools этих server'ов автоматически становятся
   доступны модели наравне с built-in.

Эта инструкция объясняет как сконфигурировать MCP-сервера, какие
endpoints доступны для UI, и какие пять рекомендованных server'ов мы
советуем поднять "из коробки".

---

## 1. Конфиг — `~/.promeserve/mcp.json`

Формат **полностью совместим** с Cline / Cursor / Claude Desktop, поэтому
один и тот же файл подходит для всех клиентов:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/promeserve"]
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/home/user/repo"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "$GITHUB_TOKEN"
      }
    }
  }
}
```

Переменные вида `$NAME` в `env` блоке резолвятся из родительского окружения
сервера PromeServe — не клади токены в файл напрямую.

Альтернативно, путь к конфигу можно переопределить через
`PROMESERVE_MCP_CONFIG=/path/to/mcp.json`.

После старта PromeServe spawn'ит для каждой записи subprocess, выполняет
JSON-RPC handshake (`initialize`, `notifications/initialized`), читает
`tools/list`, и регистрирует каждый удалённый tool в общий ToolRegistry
под именем `mcp__<server>__<remote_tool>`. Когда модель эмитирует
`<tool_call>{"name":"mcp__filesystem__read_file","arguments":{...}}</tool_call>`,
наш ToolRegistry форвардит вызов в нужный subprocess.

---

## 2. Рекомендованные 5 server'ов

| Server | Назначение | Установка |
|--------|------------|-----------|
| **filesystem** | sandboxed FS read/write/list, configurable roots | `npx -y @modelcontextprotocol/server-filesystem <path>` |
| **fetch** | URL → markdown (для read web content) | `uvx mcp-server-fetch` |
| **memory** | knowledge-graph long-term memory | `npx -y @modelcontextprotocol/server-memory` |
| **git** | read-only git log/diff/blame/status | `uvx mcp-server-git --repository <repo>` |
| **github** | issues, PRs, releases, code search (official) | `npx -y @modelcontextprotocol/server-github` |

Полный каталог: https://github.com/modelcontextprotocol/servers
+ https://glama.ai/mcp/servers

---

## 3. Built-in tools (без MCP, всегда доступны)

| Tool | Описание | Env-флаг (если есть) |
|------|----------|----------------------|
| `write_file(path, content)` | Запись в `PROMESERVE_TOOL_ROOT` (default `/tmp/promeserve`) | — |
| `read_file(path)` | Чтение первых 4KB файла | — |
| `list_dir(path)` | Листинг директории | — |
| `bash_safe(command)` | whitelist: ls/cat/head/tail/grep/wc/file/date/echo/pwd | — |
| `fetch_url(url)` / `http_get(url)` | HTTP GET, timeout 30s, body ≤ 64KB | — |
| `web_search(query, top_n)` | DuckDuckGo HTML (no API key) | `PROMESERVE_ENABLE_WEB_SEARCH=1` |
| `execute_python(code)` | `python3 -c`, timeout 15s, stdout ≤ 8KB | `PROMESERVE_ENABLE_PYTHON=1` |
| `git(command)` | Read-only git: log/diff/status/show/branch/blame/ls-files | — |
| `sqlite(db_path, query)` | SELECT/PRAGMA/WITH only | — |

Sandbox: всё ограничено `PROMESERVE_TOOL_ROOT`. Path traversal (`..`) запрещён.

---

## 4. HTTP endpoints для UI

| Метод | URL | Что |
|-------|-----|-----|
| GET | `/api/mcp/servers` | список всех MCP-серверов + статус + tool count |
| GET | `/api/mcp/tools` | полный каталог tools (builtin + MCP) с префиксами |
| POST | `/api/mcp/reconnect` | body: `{"server":"name"}` — force reconnect |
| POST | `/api/mcp/call` | body: `{"name":"...","arguments":{...}}` — debug |
| GET | `/api/mcp/audit?n=50` | последние N audit-записей |

Эти endpoints предназначены для frontend-агентов (F2): они позволяют
рисовать MCP-UI и видеть, какие tools модель использовала.

---

## 5. Audit log

Каждый tool call (имя + args + result + duration + ok) логируется в
`~/.promeserve/audit.jsonl` в формате JSONL. Можно tail'ить или читать
через endpoint `/api/mcp/audit`.

```json
{"ts":"2026-05-20T09:13:57.960Z","tool":"list_dir","args":"{\"path\":\"\"}","result":"{\"files\":[]}","ok":true,"duration_ms":0}
```

---

## 6. Max-iter guard

По умолчанию tool-loop ограничен **10 итерациями** (защита от infinite
recursion). Переопределяется через `PROMESERVE_MAX_TOOL_ITER`.

---

## 7. Parallel tool calls

Если модель эмитит несколько `<tool_call>` блоков в одном assistant turn,
они выполняются **параллельно** (до 4 одновременно) через `std::async`.
Результаты возвращаются как несколько `<tool_response>` блоков в одной
tool-роле сразу. Это особенно полезно когда модель хочет проверить
несколько файлов одновременно или прочитать несколько URL.

---

## 8. Подключить Claude.ai / Cline / Cursor к PromeServe

PromeServe умеет выглядеть как Ollama для любого клиента (OpenAI-compat
API). Для использования как MCP-сервер (Этап 3, в работе) Claude Desktop
конфиг должен будет содержать:

```json
{
  "mcpServers": {
    "prometorch": {
      "command": "/usr/local/bin/promeserve",
      "args": ["--mcp-stdio"]
    }
  }
}
```

Этап 3 пока в разработке — на текущем релизе PromeServe выступает только
**клиентом** MCP, не сервером.

---

## 9. Troubleshooting

**Сервер MCP не стартует.**
- Проверь `~/.promeserve/audit.jsonl` — там есть `tool not found: mcp__X__Y`
  если sub-server упал на initialize.
- `GET /api/mcp/servers` покажет `last_error` для каждого упавшего сервера.
- `npx`/`uvx` должен быть в `$PATH` PromeServe. На Windows установи Node.js
  + Python с `pipx install uv`.

**Tool registry не видит MCP tools.**
- MCP startup происходит в фоне (detached thread в `ApiHandlers` ctor) —
  подожди ~5 секунд после старта сервера прежде чем дёргать `/api/mcp/tools`.
- Если конфиг пустой/не существует — `/api/mcp/servers` вернёт `[]`,
  это норма.

**Reconnect не работает.**
- `POST /api/mcp/reconnect` с `{"server":"name"}`. Если server не был в
  конфиге, force-reconnect не воссоздаст его — нужно править `mcp.json`.

**Audit log пустой.**
- Должен создаться автоматически при первом вызове любого tool. Если
  директория `~/.promeserve/` не создаётся — `HOME` или `USERPROFILE`
  не выставлены.

**MCP server не отвечает (timeout).**
- Default call timeout — 30 секунд. Если внешний tool требует больше,
  это сейчас захардкожено в `mcp_client.h` (`call(method, params, timeout_ms)`).
  В версии 0.3 будет настраиваемо через config.

---

## 10. Что НЕ поддерживается (yet)

- **Streamable HTTP transport** — только stdio пока. Это значит remote
  MCP servers (через сеть) не доступны напрямую — придётся ставить
  локальный proxy server.
- **Resources** (`resources/list`, `resources/read`) — спека MCP знает их,
  но PromeServe пока не маппит их в свой API. Они есть в плане Этапа 3.
- **Prompts** (`prompts/list`, `prompts/get`) — то же самое.
- **OAuth 2.1** — для remote serv'ов; пока не нужен, потому что stdio-only.
- **Real-time tool list changes** (`notifications/tools/listChanged`) —
  игнорируется; чтобы подцепить новые tools server'а — `POST /api/mcp/reconnect`.

---

## Reference

- Spec hub: https://modelcontextprotocol.io/specification/2025-11-25
- Official servers: https://github.com/modelcontextprotocol/servers
- Cline config docs: https://docs.cline.bot/getting-started/mcp-servers
- PromeServe research: `docs/research/AGENT_STACK_R2.md`
