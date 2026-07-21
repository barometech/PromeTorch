# Agent Stack + MCP Research — отчёт R2 для PromeServe

**Дата:** 2026-05-20
**Контекст:** Item #89 (`Tool-call loop + MCP в PromeServe`) уже in_progress.
В `promeserve/tool_call.h` лежит работающий MVP Qwen-стиля `<tool_call>`-тегов
+ 4 built-in tools (write_file/read_file/list_dir/bash_safe). Цель отчёта —
дать команде **исчерпывающую карту экосистемы**, чтобы дальше двигаться
не наощупь, а по проверенным паттернам, и спроектировать MCP-слой,
plugin-систему и режимы Code/Deep Research/Computer Use поверх существующего
`<tool_call>`-MVP.

---

## TL;DR

**Топ-5 pickup'ов на ближайшую неделю-две:**

1. **MCP — стандарт де-факто.** Anthropic выкатил спеку, OpenAI, Google
   и Microsoft её приняли. У нас тегированный `<tool_call>` уже работает —
   следующий шаг это **JSON-RPC 2.0 поверх stdio** к external MCP серверам,
   а потом — Streamable HTTP (replace SSE, который deprecated с 2025-03).
   Это даёт ~30 готовых ref-серверов (filesystem, git, postgres, slack,
   github, fetch, memory, sequential-thinking, time) и ~24 000 community
   (Glama registry) бесплатно.
2. **Hermes-формат уже наш формат.** Qwen3 и Qwen2.5 на которых мы крутимся
   уже идут с Hermes-style chat template (`<tool_call>{...}</tool_call>`).
   Это значит наш парсер из `tool_call.h` — корректный, и нужно лишь
   расширить его на streaming-stop и parallel tool calls.
3. **C++ MCP SDK нет.** Официальных: Python, TypeScript, Rust (`rmcp`, 4.7M
   downloads), C#, Java, Kotlin, Swift. Для PromeServe реализуем **MCP client
   на C++ с нуля** — это ~600 LoC JSON-RPC 2.0 (либо взять `nlohmann/json`
   и `subprocess.h`). MCP server — позже на ~800 LoC.
4. **Plugin model выбрать.** В Open WebUI берут Python (subprocess + impl
   Functions/Tools/Pipelines), в LibreChat — OpenAPI-манифесты с Bearer
   auth, в Cline — MCP-marketplace. **Рекомендация: на PromeServe пойти
   MCP-first** (плагин = MCP server), без своего третьего стандарта.
   Wasm/Extism — опционально для untrusted code execution в Code Mode.
5. **Computer Use реалистичен**, но требует sandbox VM/Docker и vision
   модели (у нас её нет). **Browser-use + Playwright MCP** — лучший путь:
   accessibility-tree вместо скриншотов, нашего qwen3-4B хватит.

**План на месяц (1 параграф).** Этап 1 (1–2 дня): расширить `tool_call.h` —
parallel tool_calls, streaming-stop на `</tool_call>`, audit log,
client-supplied tools (external=true уже есть, нужен test). Этап 2 (3–5
дней): `promeserve/mcp_client.h` — JSON-RPC 2.0 over stdio, `~/.config/promeserve/mcp.json`,
поднять filesystem + memory + fetch + sequential-thinking, прогнать через
qwen3-4B. Этап 3 (5–7 дней): `promeserve/mcp_server.h` — exposeить PromeTorch
как MCP server (resources = loaded models, tools = generate/embed/tokenize),
протестить с Claude Desktop и Cline. Этап 4 (3 дня): plugin manifest =
`mcp.json` (т.е. plugin = MCP server). Этап 5 (1–2 недели): режимы Code
(встроенный Cline-style диалог + Docker exec) / Deep Research (STORM-style
outline → sub-queries → synthesis) / Computer Use (delegate в Playwright MCP).

---

## 1. MCP — спецификация и SDK

### 1.1 Что такое MCP в одном абзаце

Model Context Protocol — открытый стандарт от Anthropic (ноябрь 2024,
spec версионируется по дате), JSON-RPC 2.0 поверх двух транспортов
(stdio, Streamable HTTP). Описывает **три типа примитивов**: **tools**
(функции, которые модель может вызвать), **resources** (read-only
данные с URI вроде `file://`, `git://`, `postgres://table/...`),
**prompts** (готовые шаблоны системных промптов). Сервер декларирует
capabilities, клиент discoверит и вызывает. Принят как стандарт OpenAI,
Google, Microsoft в 2025.

Канонические доки:
- Spec hub: https://modelcontextprotocol.io
- Last stable spec: `2025-11-25` (transports rev)
- GitHub org: https://github.com/modelcontextprotocol
- Official registry: https://registry.modelcontextprotocol.io

### 1.2 Транспорты — где какой

| Транспорт | Когда использовать | Плюсы | Минусы |
|-----------|-------------------|-------|--------|
| **stdio** | Local subprocess (CLI tools, расширения IDE) | Простой, без сети, без auth, мгновенный | 1 client per process, не для облака |
| **Streamable HTTP** (с 2025-03) | Remote MCP servers, multi-tenant | OAuth 2.1 + DCR, server-sent events для streaming, multi-client | Сложнее, требует TLS + auth |
| **HTTP+SSE** (deprecated) | Legacy | — | Заменён Streamable HTTP, выпиливают |

В Streamable HTTP сервер обязан экспортить **один endpoint** (обычно `/mcp`),
который умеет POST (JSON-RPC) и GET (SSE-стрим уведомлений). Если ответ
streaming-блочный — Content-Type `text/event-stream`, если разовый
JSON — `application/json`. Это позволяет одному endpoint'у работать
и в режиме request-response, и в режиме long-lived subscription.

Источники:
- Transports spec: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
- Cloudflare guide: https://developers.cloudflare.com/agents/model-context-protocol/transport/
- Кратко в нашем случае: **stdio для Этапа 2** (поднимем 4–5 ref-серверов
  как subprocess), **Streamable HTTP для Этапа 3** (PromeServe станет
  MCP server для удалённых клиентов).

### 1.3 Protocol messages

Все сообщения — стандартные JSON-RPC 2.0 (id, method, params, result/error).

**Handshake** (первое, что делает client после spawn'а server):
```json
// → client
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
  "protocolVersion":"2025-11-25",
  "capabilities":{"tools":{},"resources":{"subscribe":true}},
  "clientInfo":{"name":"PromeServe","version":"0.3"}
}}
// ← server
{"jsonrpc":"2.0","id":1,"result":{
  "protocolVersion":"2025-11-25",
  "capabilities":{"tools":{"listChanged":true},"resources":{}},
  "serverInfo":{"name":"filesystem","version":"1.2"}
}}
// → client (notification, без id)
{"jsonrpc":"2.0","method":"notifications/initialized"}
```

**tools/list** + **tools/call**:
```json
// → tools/list
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
// ← результат
{"jsonrpc":"2.0","id":2,"result":{
  "tools":[
    {
      "name":"read_file",
      "title":"Read file",
      "description":"Read contents of a file by absolute path",
      "inputSchema":{
        "type":"object",
        "properties":{"path":{"type":"string"}},
        "required":["path"]
      }
    }
  ]
}}
// → tools/call
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
  "name":"read_file",
  "arguments":{"path":"/tmp/x.txt"}
}}
// ← результат
{"jsonrpc":"2.0","id":3,"result":{
  "content":[{"type":"text","text":"hello\n"}],
  "isError":false
}}
```

**resources/list** + **resources/read** + **resources/subscribe**:
```json
// → resources/list
{"jsonrpc":"2.0","id":4,"method":"resources/list","params":{"cursor":""}}
// ← с пагинацией
{"jsonrpc":"2.0","id":4,"result":{
  "resources":[
    {"uri":"file:///tmp/log.txt","name":"log","mimeType":"text/plain"}
  ],
  "nextCursor":""
}}
// → resources/read
{"jsonrpc":"2.0","id":5,"method":"resources/read","params":{
  "uri":"file:///tmp/log.txt"
}}
// ← contents array
{"jsonrpc":"2.0","id":5,"result":{"contents":[
  {"uri":"file:///tmp/log.txt","mimeType":"text/plain","text":"..."}
]}}
// → подписка на изменения
{"jsonrpc":"2.0","id":6,"method":"resources/subscribe","params":{"uri":"file:///tmp/log.txt"}}
// ← уведомление server → client (notification)
{"jsonrpc":"2.0","method":"notifications/resources/updated","params":{
  "uri":"file:///tmp/log.txt"
}}
```

**prompts/list** + **prompts/get** (готовые шаблоны вроде "summarize commit
diff"):
```json
{"jsonrpc":"2.0","id":7,"method":"prompts/list","params":{}}
{"jsonrpc":"2.0","id":7,"result":{"prompts":[
  {"name":"git/commit-summary","arguments":[{"name":"sha","required":true}]}
]}}
{"jsonrpc":"2.0","id":8,"method":"prompts/get","params":{
  "name":"git/commit-summary","arguments":{"sha":"abc123"}
}}
{"jsonrpc":"2.0","id":8,"result":{
  "description":"Summarize commit abc123",
  "messages":[{"role":"user","content":{"type":"text","text":"<diff>...</diff>"}}]
}}
```

Источники:
- Protocol handbook: https://portkey.ai/blog/mcp-message-types-complete-mcp-json-rpc-reference-guide/
- Tools spec: https://modelcontextprotocol.io/specification/2025-11-25/server/tools
- Resources spec: https://modelcontextprotocol.io/specification/2025-03-26/server/resources

### 1.4 Authentication

| Pattern | Где | Что нужно |
|---------|-----|-----------|
| **None** (stdio) | Local subprocess | Просто spawn |
| **API key in env** | Legacy / quick servers | `env:{GITHUB_TOKEN:...}` в конфиге, токен в process env |
| **OAuth 2.1 + PKCE + DCR** | Remote Streamable HTTP | Обязательная схема по спеке |
| **mTLS** | Внутри VPC | Опционально, OAuth тоже подходит |

OAuth 2.1 в MCP отличается от обычного web-OAuth тем что:
- **Dynamic Client Registration (RFC 7591) обязателен** — клиент не знает заранее
  список серверов и должен уметь регистрироваться runtime'но.
- **PKCE обязателен** (RFC 7636), implicit grant запрещён.
- **Protected Resource Metadata** — сервер публикует `.well-known/oauth-protected-resource`
  с указателем на authorization server.

Для PromeServe Этапа 2 (stdio только) auth не нужен. На Этапе 3 (HTTP)
минимум — Bearer token в `Authorization`-header (наш собственный API key),
полноценный OAuth 2.1 можно отложить.

Источник: https://www.permit.io/blog/oauth-on-mcp

### 1.5 Streaming responses, errors, retry

- Server может вернуть `content: [{"type":"text", "text":...}]` сразу,
  или поток через SSE (если transport — Streamable HTTP).
- Ошибки: либо JSON-RPC error object (`error: {code, message}`), либо
  result со `isError: true` и content с описанием (различение: RPC-уровневые
  vs domain-уровневые).
- Retry: spec не предписывает. На практике — exponential backoff
  на сетевые сбои, no-retry на `isError:true` (это logical error модели).
- Timeouts: обычно client задаёт. Default по нашим оценкам — 30 s на tool_call,
  600 s на initialize.

### 1.6 SDK реализации

| SDK | Maintainer | URL | Качество |
|-----|-----------|-----|----------|
| **TypeScript** | официальный | https://github.com/modelcontextprotocol/typescript-sdk | reference, всегда первый |
| **Python** | официальный | https://github.com/modelcontextprotocol/python-sdk | + `FastMCP` decorator API, pypi `mcp` |
| **Rust** | официальный (`rmcp` crate) | https://github.com/modelcontextprotocol/rust-sdk | 4.7M downloads, async |
| **C#** | официальный | https://github.com/modelcontextprotocol/csharp-sdk | .NET 8+ |
| **Java** + **Kotlin** | официальные | github.com/modelcontextprotocol/{java,kotlin}-sdk | |
| **Swift** | официальный | github.com/modelcontextprotocol/swift-sdk | |
| `mcpkit` (Rust) | community | https://docs.rs/mcpkit | `#[mcp_server]` macro |
| **C++** | — | **отсутствует** | реализовать самим |

Для нас (C++ HTTP сервер) — пишем минимальный JSON-RPC client из ~600
LoC поверх `nlohmann/json` + `popen()`/`fork+pipe()`. Хороший reference —
[Rust `rmcp` source](https://github.com/modelcontextprotocol/rust-sdk),
самая чистая официальная реализация. Также полезно глянуть DeepWiki
обзор: https://deepwiki.com/modelcontextprotocol/rust-sdk

### 1.7 Топ ref/popular MCP серверов

Все из официального `modelcontextprotocol/servers` или с registry:

| Server | URL | Назначение |
|--------|-----|------------|
| **filesystem** | github.com/modelcontextprotocol/servers/tree/main/src/filesystem | sandboxed FS R/W, configurable roots |
| **git** | servers/tree/main/src/git | git log/blame/diff/status |
| **github** (official) | https://github.com/github/github-mcp-server | issues, PR, releases, code search |
| **gitlab** | servers/tree/main/src/gitlab | аналог github |
| **postgres** | servers/tree/main/src/postgres | read-only SQL + schema |
| **sqlite** | servers/tree/main/src/sqlite | local DB inspect |
| **slack** | servers/tree/main/src/slack (Zencoder fork) | channels + messages |
| **brave-search** | servers/tree/main/src/brave-search | web + local search |
| **google-drive** | servers/tree/main/src/gdrive | drive list + read |
| **google-maps** | servers/tree/main/src/google-maps | geocode + directions |
| **fetch** | servers/tree/main/src/fetch | URL → markdown |
| **memory** | servers/tree/main/src/memory | knowledge-graph long-term memory |
| **sequential-thinking** | servers/tree/main/src/sequentialthinking | iterative reflection scratchpad |
| **puppeteer** | servers/tree/main/src/puppeteer | browser via Puppeteer |
| **playwright** (Microsoft) | https://github.com/microsoft/playwright-mcp | accessibility-tree browser |
| **time** | servers/tree/main/src/time | timezone math |
| **everything** | servers/tree/main/src/everything | test server со всеми primitives |
| **redis** | servers/tree/main/src/redis | KV |
| **sentry** | servers/tree/main/src/sentry | error tracking lookup |
| **aws** (Microsoft mcp catalog) | https://github.com/microsoft/mcp | Azure-ориентированный, по аналогии |
| **azure-mcp** | github.com/microsoft/mcp | resource group + identity |
| **kubernetes** (community) | пример в abordage/awesome-mcp | kubectl wrapper |
| **docker** (community) | пример в awesome-mcp | docker ps/run |
| **jira** (Atlassian) | atlassian/jira-mcp-server | tickets |
| **linear** | community | issue tracking |
| **notion** | community | page CRUD |
| **stripe** | официальный | payments queries |
| **cloudflare** | официальный | workers + DNS |
| **playwright-universal** | xkiranj/playwright-universal-mcp | контейнеризованный browser |
| **executeautomation/mcp-playwright** | github.com/executeautomation/mcp-playwright | альтернатива MS |
| **mcpfinder** | github.com/mcpfinder/mcpfinder | MCP сервер, который ищет другие MCP серверы (мета) |

Полные списки:
- Official: https://github.com/modelcontextprotocol/servers
- Curated community: https://github.com/wong2/awesome-mcp-servers
- Auto-updated: https://github.com/abordage/awesome-mcp

### 1.8 Каталоги / marketplaces

| Платформа | URL | Что есть | Особенности |
|-----------|-----|----------|-------------|
| **Official Registry** | https://registry.modelcontextprotocol.io | Канонический, минимальный | Без UI, JSON API только. Поддерживается Anthropic + Steering Group |
| **Glama** | https://glama.ai/mcp | ~24 000 серверов | Proxy gateway, OAuth-managed, per-tool ACL, inspector, security scanning |
| **Smithery** | https://smithery.ai | ~2 000 | Index, discovery, install |
| **mcp.so** | https://mcp.so | ~20 000+ | Marketplace UX |
| **mcpservers.org** | https://mcpservers.org | ~4 000 | Catalog |
| **Cline marketplace** | в VS Code расширении | downloadable `.mcp.json` | Один клик install в Cline |
| **Continue marketplace** | в Continue.dev | ~hundreds | для AI-coding |
| **Mastra registry** | mastra.ai | meta-registry | для JS экосистемы |

Источник: https://www.gentoro.com/blog/what-is-anthropics-new-mcp-registry/ +
https://apigene.ai/blog/mcp-marketplace

---

## 2. Tool-Calling форматы — сравнение

Все форматы решают одну и ту же задачу — "модель выбирает функцию
+ JSON-аргументы". Различаются обёрткой в промпте и форматом ответа.

### 2.1 OpenAI (`tools` array)

Старое поле `functions:` deprecated с 2024-06. Новое:

```json
// Request
{
  "model":"gpt-4o",
  "messages":[{"role":"user","content":"Какая погода в Москве?"}],
  "tools":[{
    "type":"function",
    "function":{
      "name":"get_weather",
      "description":"Get current weather",
      "parameters":{
        "type":"object",
        "properties":{
          "location":{"type":"string"}
        },
        "required":["location"]
      }
    }
  }],
  "tool_choice":"auto"
}

// Response (assistant message)
{
  "role":"assistant",
  "content":null,
  "tool_calls":[{
    "id":"call_abc",
    "type":"function",
    "function":{
      "name":"get_weather",
      "arguments":"{\"location\":\"Moscow\"}"
    }
  }]
}

// Continuation — клиент шлёт обратно
{"role":"tool","tool_call_id":"call_abc","content":"{\"temp\":\"-3°C\"}"}
```

Особенности:
- `tool_choice` принимает `"auto"`, `"none"`, `"required"`, или объект для force.
- `parallel_tool_calls: true` по умолчанию — модель может вернуть несколько
  tool_calls в одном ответе.
- Strict mode (`"strict": true` на function): сервер OpenAI делает structured
  output schema enforcement, гарантия валидного JSON.

### 2.2 Anthropic (`tools` через content block)

```json
// Request
{
  "model":"claude-opus-4-7",
  "messages":[{"role":"user","content":"Какая погода в Москве?"}],
  "tools":[{
    "name":"get_weather",
    "description":"Get current weather",
    "input_schema":{
      "type":"object",
      "properties":{"location":{"type":"string"}},
      "required":["location"]
    }
  }]
}

// Response — content array со смесью text и tool_use
{
  "role":"assistant",
  "content":[
    {"type":"text","text":"Сейчас проверю..."},
    {"type":"tool_use","id":"toolu_01","name":"get_weather",
     "input":{"location":"Moscow"}}
  ],
  "stop_reason":"tool_use"
}

// Continuation — user message с tool_result content block
{
  "role":"user",
  "content":[{"type":"tool_result","tool_use_id":"toolu_01",
              "content":"{\"temp\":\"-3°C\"}"}]
}
```

Ключевые отличия от OpenAI:
- Schema поле — `input_schema` (не `parameters`), но тот же JSON-Schema.
- Tool use — это content block внутри assistant message, **не** отдельный
  `tool_calls` поле. Допускает interleave с text.
- `stop_reason: "tool_use"` — сигнал клиенту что нужно выполнить.
- Multiple tool_use в одном response = parallel calls.

### 2.3 Mistral

OpenAI-compatible: ровно тот же `tools[]`/`tool_calls[]`. Codestral, Mistral
Small/Large/Medium — все через тот же endpoint. Один ноtable bug: Mistral
консервативнее в выборе вызова tool — часто отвечает из памяти даже когда
надо вызвать. Workaround: `tool_choice: "any"` или явная инструкция
в system prompt.

### 2.4 Gemini

```json
// Request
{
  "contents":[{"role":"user","parts":[{"text":"Какая погода?"}]}],
  "tools":[{"functionDeclarations":[{
    "name":"get_weather",
    "description":"...",
    "parameters":{"type":"OBJECT","properties":{"location":{"type":"STRING"}}}
  }]}]
}

// Response
{"candidates":[{"content":{"parts":[{
  "functionCall":{"name":"get_weather","args":{"location":"Moscow"}}
}]}}]}

// Continuation
{"role":"function","parts":[{"functionResponse":{
  "name":"get_weather",
  "response":{"temp":"-3°C"}
}}]}
```

Особенности: типы в верхнем регистре (`OBJECT`/`STRING`), отдельный role
`function` (не `tool`). Иначе изоморфно.

### 2.5 MCP tools/call

Уже описан в §1.3 — это transport-уровневый формат, **универсальный**.
Все остальные форматы можно нормализовать под него:

```json
// Каноническая форма
{"name":"get_weather","arguments":{"location":"Moscow"}}
// + result
{"content":[{"type":"text","text":"..."}], "isError":false}
```

### 2.6 Llama 3.1 / Llama 3.2 — `<|python_tag|>`

Meta пошла своим путём. Built-in tools (`brave_search`, `wolfram_alpha`,
`code_interpreter`) и custom tools.

```
<|start_header_id|>system<|end_header_id|>
Environment: ipython
Tools: brave_search, wolfram_alpha

You have access to functions:
{"name":"get_weather","description":...}
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Какая погода в Москве?<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
<|python_tag|>get_weather.call(location="Moscow")<|eom_id|>
```

- `<|python_tag|>` начинает Python-style вызов.
- `<|eom_id|>` ("end of message, expect continuation") — ждёт результата
  tool'а.
- `<|eot_id|>` — конец final ответа.

Парсинг сложнее (Python syntax, не JSON), но прямолинейный. xLAM
(Salesforce) использует этот же формат.

### 2.7 Qwen3 / Qwen2.5 — Hermes-style (**наш формат**)

```
<|im_start|>system
You are a helpful assistant.
# Tools
You have access to the following functions:
<tools>
[{"type":"function","function":{"name":"get_weather", ...}}]
</tools>
For each function call, return a json object with name and arguments
within <tool_call></tool_call> tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json>}
</tool_call>
<|im_end|>
<|im_start|>user
Какая погода в Москве?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "Moscow"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp":"-3°C"}
</tool_response><|im_end|>
<|im_start|>assistant
В Москве -3°C.<|im_end|>
```

Особенности:
- `<tool_call>` и `</tool_call>` — added tokens в BPE, лёгкий streaming-parse.
- Multiple `<tool_call>` блоков в одном assistant message → parallel calls.
- `<tool_response>` оборачивает результат, отдаётся как user-роль continuation.
- Qwen-Agent имеет встроенный parser, не нужно дополнительных vLLM флагов
  (`--tool-call-parser hermes` — НЕ нужен для Qwen3).

Это ровно то, что у нас в `tool_call.h` — мы выбрали правильный формат.

### 2.8 Hermes / NousResearch fine-tunes

Hermes-2-Pro и Hermes-4-14B (на Llama/Mistral базах) используют **тот же
синтаксис** что Qwen3: `<tool_call>...</tool_call>` теги. Это де-факто
открытый стандарт для open-weight моделей с tool calling. Phi-3 fine-tunes
часто следуют той же конвенции.

### 2.9 Universal converter — псевдокод

Цель: на входе любой формат запроса → внутри представляем как **MCP-нормализованный**,
на выходе генерируем prompt в нужном формате модели.

```cpp
// Канонический tool descriptor (изоморфен MCP tools/list result)
struct CanonicalTool {
    string name;
    string description;
    json input_schema;        // JSON Schema (OpenAPI-compatible)
};

struct CanonicalCall { string id; string name; json arguments; };
struct CanonicalResult { string call_id; string content; bool is_error; };

// Adapters in
CanonicalTool from_openai(const json& tool);     // tool.function.{name,description,parameters}
CanonicalTool from_anthropic(const json& tool);  // tool.{name,description,input_schema}
CanonicalTool from_gemini(const json& tool);     // tool.functionDeclarations[i]
CanonicalTool from_mcp(const json& tool);        // прямой mapping

// Adapters out — генерируют формат для конкретной модели
string format_for_qwen3(const vector<CanonicalTool>& tools);
string format_for_llama31(const vector<CanonicalTool>& tools);
string format_for_anthropic_api(...);  // не нужен — мы локальные модели

// Parser model output -> CanonicalCall(s)
vector<CanonicalCall> parse_qwen3(const string& text);    // ищем <tool_call>...</tool_call>
vector<CanonicalCall> parse_llama31(const string& text);  // <|python_tag|>name(args)<|eom_id|>
vector<CanonicalCall> parse_openai(const string& text);   // <function=name>{json}</function>
                                                          // (старые fine-tunes)

// Dispatch
CanonicalResult execute(const CanonicalCall& c) {
    if (registry.is_external(c.name)) return forward_to_client(c);
    if (registry.is_mcp(c.name))      return mcp_client.tools_call(c.name, c.arguments);
    return registry.execute(c.name, c.arguments);  // built-in
}

// Format result back to model
string format_result_for_qwen3(const CanonicalResult& r);
// → <tool_response>{...}</tool_response>
```

Это даёт **single endpoint** `/api/chat` с поддержкой любых клиентов
(Ollama-style, OpenAI-style, Anthropic-style — auto-detect по форме тела),
и внутри унифицированный canonical loop.

Сравнения форматов:
- https://propelius.ai/blogs/function-calling-vs-tool-use-ai-agents/
- https://www.glukhov.org/llm-performance/benchmarks/structured-output-comparison-popular-llm-providers
- https://qwen.readthedocs.io/en/latest/framework/function_call.html
- https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

---

## 3. Agent loops (паттерны)

### 3.1 ReAct (Yao et al. 2022, arXiv:2210.03629)

**Идея.** Interleave **Thought** (reasoning trace) + **Action** (tool call)
+ **Observation** (result). Модель сама выбирает когда думать, когда
действовать, когда отвечать.

```
Thought 1: Чтобы ответить, нужна температура в Москве. Вызову get_weather.
Action 1:  get_weather[location="Moscow"]
Observation 1: -3°C, cloudy
Thought 2: Температура получена. Можно отвечать.
Answer:   В Москве сейчас -3°C, облачно.
```

Псевдокод main loop:
```python
def react_loop(question, tools, max_steps=10):
    trajectory = f"Question: {question}\n"
    for step in range(max_steps):
        out = llm(trajectory + f"Thought {step+1}:")
        thought = parse_thought(out)
        action = parse_action(out)
        if action.is_final():
            return action.answer
        obs = execute(action.tool, action.args)
        trajectory += f"Thought {step+1}: {thought}\n"
        trajectory += f"Action {step+1}: {action}\n"
        trajectory += f"Observation {step+1}: {obs}\n"
    raise StepLimit
```

Это **базовый паттерн** для всех современных агентов. Plain ReAct в чистом
виде уже редок — обычно его смешивают с tool-calling JSON (наш `<tool_call>` =
структурированный Action), и иногда без явных `Thought:` маркеров (модель
сама встраивает рассуждения в `content`).

Источники:
- Paper: https://arxiv.org/abs/2210.03629
- Prompting guide: https://www.promptingguide.ai/techniques/react

### 3.2 Plan-and-Execute (LangChain)

Два LLM:
1. **Planner** — большой/качественный, генерирует пошаговый план целиком.
2. **Executor** — меньший/быстрый, выполняет шаги по одному (внутри
   re-prompts'итя как ReAct).
3. **Re-planner** — после каждого шага дать planner'у пересмотреть план.

```python
def plan_execute(task):
    plan = planner_llm.generate_plan(task)          # ["step1", "step2", ...]
    results = []
    for step in plan:
        r = react_executor(step, tools)
        results.append(r)
        plan = planner_llm.replan(task, plan, results)  # may shorten/extend
        if planner_llm.is_done(task, plan, results):
            return synthesize(results)
```

Плюс: дешевле, чем full ReAct на large модели каждый шаг. Минус: rigid plan
плохо адаптируется. LangGraph даёт хорошую реализацию.

Источник: https://blog.langchain.com/planning-agents/

### 3.3 Tree of Thoughts — Yao 2023 (arXiv:2305.10601)

Вместо линейной траектории — **дерево** "thoughts". На каждом узле модель
генерирует k продолжений, **сама оценивает их** (`value/vote`), и search
(BFS/DFS/beam) выбирает лучшие ветки. Подходит для задач с обратной связью
типа Game of 24, mini-Crosswords, creative writing.

```python
def tot_dfs(node, depth, max_depth):
    if depth == max_depth or is_solution(node):
        return node
    candidates = llm.expand(node, k=5)            # 5 next thoughts
    scored = [(llm.value(c), c) for c in candidates]
    for _, c in sorted(scored, reverse=True)[:3]: # top-3 branches
        result = tot_dfs(c, depth+1, max_depth)
        if is_solution(result):
            return result
    return failure
```

Слишком тяжёл для production tool-calling — каждое expand'а это N вызовов
LLM. Используется точечно для отдельных reasoning-задач.

### 3.4 Reflexion (Shinn 2023, arXiv:2303.11366)

Three modules: **Actor** (genericая politika, ReAct-like), **Evaluator**
(оценивает trajectory), **Self-Reflection** (генерирует natural language
"урок" из неудачной попытки и кладёт в memory). Memory size Ω = 1–3
(чтобы помещался в контекст). Trial → eval → reflect → retry.

```python
def reflexion(task, max_trials=5):
    memory = deque(maxlen=3)
    for trial in range(max_trials):
        traj = actor.solve(task, memory=memory)
        eval_score = evaluator(traj)
        if eval_score == pass:
            return traj.answer
        reflection = reflector.reflect(traj, eval_score)
        memory.append(reflection)
    return None
```

Эффективен на HotpotQA, ALFWorld. Для нас — как опция режима Deep Research
("если ответ не сходится, отрефлексируй и попробуй другие запросы").

### 3.5 Voyager (Wang 2023, arXiv:2305.16291)

Long-horizon агент для Minecraft. Три компонента:
1. **Automatic curriculum** — LLM сам предлагает что попробовать дальше.
2. **Skill library** — растущая БД исполняемого кода (JavaScript-функций).
   Каждый освоенный навык кладётся как embeddings + код.
3. **Iterative prompting** — добавляет в prompt error trace + self-verification,
   итерирует пока код не пройдёт.

Идея skill library легко переносится в наш PromeServe: agent написал
полезную shell/python функцию — сохраняем в `~/.local/share/promeserve/skills/`
с описанием, на retrieval'е достаём.

### 3.6 AutoGPT / BabyAGI / AgentGPT — первая волна (2023)

**BabyAGI** — 105 строк Python, три "агента" (одна GPT-4 под капотом):
- task_execution_agent
- task_creation_agent  
- task_prioritization_agent

Loop: pop задачу из приоритетной очереди → выполнить → сохранить в Pinecone →
сгенерить новые задачи → переранжировать.

**AutoGPT** — расширение с большим набором tools (file I/O, web search,
shell, GPT memory). Архитектурно — тот же loop, но с persistence и
"goal-oriented" целеполаганием. Ушёл в коммерческий продукт.

**Урок для нас:** базовый паттерн (task queue + execute + reflect) не требует
LLM-стороннего стандарта — это просто структура данных. Можно прикрутить
к нашему `<tool_call>` loop'у в Deep Research mode.

Источник: https://babyagi.wiki

### 3.7 OpenHands / SWE-agent — coding agents

**SWE-agent** (Princeton/Stanford, NeurIPS 2024). Берёт GitHub issue,
автоматически фиксит. Ключевая innovation — **Agent-Computer Interface
(ACI)**: специально спроектированные tool wrappers для file editing
(вместо raw shell), которые сильно повышают success rate. mini-SWE-agent
в 100 строк уже даёт 74% на SWE-bench Verified.
URL: https://github.com/SWE-agent/SWE-agent

**OpenHands** (ex-OpenDevin, All-Hands-AI). MIT, 70k★, 490+ contributors.
Sandbox в Docker, поддерживает любые LLM, имеет SDK + CLI + GUI. Решает
50%+ реальных GitHub issues.
URL: https://github.com/All-Hands-AI/OpenHands

**Sakana ShinkaEvolve** (ICLR 2026). Evolutionary code search через LLM
mutations — adaptive parent sampling, novelty rejection, bandit-based
LLM selector. Apache 2.0. URL: https://github.com/SakanaAI/ShinkaEvolve.
Подходит для AutoML/scientific discovery, не general agent stack.

### 3.8 Safeguards для production agent loops

| Safeguard | Что | Дефолт |
|-----------|-----|--------|
| **Max iterations** | Limit count of tool calls | 10 (у нас в TOOL_CALL_PLAN.md уже) |
| **Cost limit** | $ или tokens budget | $1 / 100k tokens (если remote) |
| **Action whitelist** | Только разрешённые tools | да, см. registry |
| **Tool timeout** | Per-call timeout | 30 s |
| **Sandbox** | FS chroot / Docker / VM | `/tmp/promeserve/` + Docker для Code Mode |
| **Approval mode** | human-in-loop confirm на риск-ных tools | configurable (`PROMESERVE_AUTO_APPROVE=write_file,bash_safe`) |
| **Audit log** | append-only лог всех tool_call'ов | `~/.local/share/promeserve/audit.log` JSONL |
| **Prompt-injection defense** | sanitize tool outputs, strip system-impersonation | tag `<external_content>` обёртка |
| **Memory leakage** | очищать secrets из контекста перед save | regex для AWS/GitHub patterns |

---

## 4. Production agent frameworks

### 4.1 CrewAI

- GitHub: https://github.com/crewAIInc/crewAI
- License: MIT
- Stars: ~30k+ (Q1 2026)
- Идея: **role-based collaboration**. Декларативно описываешь Agents
  (role, goal, backstory, tools) и Tasks, фреймворк оркестрирует через
  Crews/Flows.
- Когда брать: быстрый прототип multi-agent (2–3 дня), маркетинг/контент/
  research пайплайны.
- Как встроить в PromeServe: **не встраивать.** Использовать как external
  client поверх нашего OpenAI-compat endpoint (`OPENAI_API_BASE=http://promeserve:11434/v1`).

### 4.2 AutoGen / AG2 (Microsoft)

- GitHub: https://github.com/microsoft/autogen
- License: MIT
- 2026 статус: **maintenance mode** — Microsoft переключился на свой
  "Agent Framework", разработка фич замедлилась.
- Идея: agents общаются как chat, поддерживается group chat + code executor +
  human-in-loop. Хороший research-фреймворк (исходный paper из MSR).
- Когда брать: исследование multi-agent диалогов.
- Встройка: как external client.

### 4.3 LangGraph (LangChain)

- GitHub: https://github.com/langchain-ai/langgraph
- License: MIT
- 2026 статус: де-факто **дефолт для production deployments** (v0.4+).
- Идея: **state machine**. Граф nodes + edges, state persistance,
  checkpointing, human-in-loop interrupts, streaming. Используется
  под капотом многих продуктов.
- Когда брать: complex workflows со state и retry, long-running с pauses.
- Встройка: как external client поверх нашего endpoint.

### 4.4 Anthropic Claude Agent SDK

- npm: https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk
- pypi: `claude-agent-sdk`
- Docs: https://platform.claude.com/docs/en/agent-sdk/overview
- Что это: **тот же engine, на котором построен Claude Code** (CLI агент).
  Tools, context management, sub-agents, MCP integration. 1M+ npm/week,
  600+ dependents.
- Биллинг: с июня 2026 — встроен в Pro/Max/Team подписку ($20–$200/mo
  Agent SDK credit).
- Когда брать: production агенты, которым нужны Claude tools (computer-use,
  file edit, web search, MCP server pool). **Не имеет смысла как
  layer внутри PromeServe** — он завязан на Anthropic API. Может
  использоваться извне как клиент.

### 4.5 OpenAI Agents SDK

- GitHub: https://github.com/openai/openai-agents-python (+ -js)
- Docs: https://openai.github.io/openai-agents-python/
- License: MIT
- Идея: provider-agnostic (поддерживает Chat Completions + Responses
  API + 100+ LLMs через LiteLLM). Core абстракции: **Agent**, **Tool**,
  **Handoff** (передача между агентами), **Guardrails**, **Tracing**,
  **Sandbox**.
- Когда брать: multi-agent workflows c handoffs, voice agents.
- Встройка: как external client.

### 4.6 LlamaIndex Agents

- GitHub: https://github.com/run-llama/llama_index
- Docs: https://docs.llamaindex.ai
- Идея: **AgentWorkflow** (новый, 2026) — graph-based, multi-agent с handoffs.
  Три типа: FunctionCalling, ReAct, Custom. Глубокая интеграция с RAG
  (всё-таки LlamaIndex про индексацию).
- Когда брать: RAG + agent в одном пакете.
- Встройка: external.

### 4.7 Pydantic AI

- Docs: https://pydantic.dev/docs/ai/overview
- License: MIT
- Идея: **type-safe agents**. Pydantic schemas для tool args, structured
  outputs с auto-correction если LLM возвращает invalid JSON.
- Когда брать: production где важна корректность типов и валидируемые
  outputs.

### 4.8 Smolagents (Hugging Face)

- GitHub: https://github.com/huggingface/smolagents
- License: Apache 2.0
- Идея: **CodeAgent** — модель пишет **Python код** как action (не JSON
  tool calls). Composable (loops, conditionals), expressive. Library
  всего ~1000 LoC. Альтернатива — `ToolCallingAgent` (классический JSON).
- Sandbox опции: Blaxel, E2B, Modal, Docker, Pyodide+Deno WASM.
- Когда брать: agents, которым нужна вычислительная экспрессия (плотная
  работа с числами, файлами, последовательными вычислениями).

### 4.9 Agno (ex-Phidata)

- GitHub: https://github.com/agno-agi/agno
- License: MPL 2.0 / commercial для AgentOS
- 39k+★, 424+ contributors.
- Идея: full-stack — Python SDK + **AgentOS** runtime (stateless FastAPI) +
  control plane UI. Self-claim "10 000× faster agent creation" (микросекунды vs
  миллисекунды).
- Когда брать: production где нужна готовая control plane и stateless scale.

### 4.10 Сводная матрица

| Framework | Best for | Production readiness | Speed to first demo | Лицензия |
|-----------|----------|---------------------|---------------------|----------|
| **LangGraph** | state-rich workflows, HITL, checkpointing | ⭐⭐⭐⭐⭐ | 10–14 d | MIT |
| **CrewAI** | role-based crews, content/research | ⭐⭐⭐⭐ | 2–3 d | MIT |
| **AutoGen/AG2** | research, conversational multi-agent | ⭐⭐⭐ (maintenance) | 5–7 d | MIT |
| **Claude Agent SDK** | Claude-powered prod agents | ⭐⭐⭐⭐⭐ | 1–2 d | Anthropic SDK License |
| **OpenAI Agents SDK** | multi-agent + handoffs + voice | ⭐⭐⭐⭐ | 2–4 d | MIT |
| **LlamaIndex Agents** | RAG + agent в одном | ⭐⭐⭐⭐ | 3–5 d | MIT |
| **Pydantic AI** | type-safe, structured outputs | ⭐⭐⭐⭐ | 3–4 d | MIT |
| **Smolagents** | code-action agents | ⭐⭐⭐ | 1 d | Apache 2.0 |
| **Agno** | full-stack prod, AgentOS UI | ⭐⭐⭐⭐ | 2–3 d | MPL 2.0 |

Для PromeServe **рекомендация**: **никакой из этих NOT в C++ ядро не
встраиваем.** Они все Python/JS. PromeServe экспонирует OpenAI-compatible
+ MCP endpoint, и любой из них может работать поверх. Внутри PromeServe —
свой минимальный loop на C++ (то что уже есть в `tool_call.h`) + MCP client.

Источники:
- Сравнения 2026: https://medium.com/data-science-collective/langgraph-vs-crewai-vs-autogen-which-agent-framework-should-you-actually-use-in-2026-b8b2c84f1229
- Pratik Pathak: https://pratikpathak.com/langgraph-vs-crewai-vs-autogen-2026/
- OpenAgents: https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared

---

## 5. Plugin marketplaces — архитектуры

### 5.1 Open WebUI Tools/Functions/Pipelines

Самая зрелая plugin-система среди open-source LLM UI.

- **Tools.** Python скрипты, исполняются **в самом Open WebUI процессе**.
  Любые библиотеки, упакованные в Open WebUI. LLM может их вызвать через
  function calling.
- **Functions.** Три подвида:
  - **Pipe** — регистрируется как новая "модель" в селекторе. Перехватывает
    request, может вообще не использовать LLM.
  - **Filter** — pre-process/post-process сообщений.
  - **Action** — кнопка в UI рядом с сообщением, при клике вызывает Python.
- **Pipelines.** Тоже Python, но **на отдельном сервере** (FastAPI),
  OpenAI-API совместимом. Можно ставить любые pip dependencies, в отличие
  от Functions/Tools (там — только встроенный в Open WebUI набор).

Сильные стороны:
- Богатый набор примитивов (4 типа точек расширения).
- Marketplace формат — просто Python код, легко публиковать.
- Полная мощь Python (можно вызывать любые pkgs в Pipeline'е).

Слабые:
- **Безопасность.** Любой код может всё. Документация прямо предупреждает:
  "Only install from trusted sources." Нет sandboxing.
- Single-process для Tools/Functions → плагин может сломать сервер.

URL: https://docs.openwebui.com/features/extensibility/plugin/

### 5.2 LibreChat plugins

- Структура: **OpenAPI spec** (YAML/JSON) + **manifest JSON** + API endpoint
  снаружи.
- Manifest требует `name_for_model`, `description_for_model`, ссылку на
  OpenAPI spec в `api.url`.
- Auth: только **Bearer**.
- Где: `api/app/clients/tools/.well-known/openapi/*.yaml`.
- ChatGPT-compat (старый формат ChatGPT plugins из 2023).

Сильные: stateless, легко проверить через swagger, multi-tenant-friendly.
Слабые: нет инкапсулированных prompts/resources как в MCP. Auth только Bearer.

URL: https://docs.librechat.ai/features/plugins/chatgpt_plugins_openapi.html

### 5.3 Cline / Cursor MCP marketplace

- Plugin = **MCP server config** (один объект JSON с `command`+`args`+`env`).
- Marketplace внутри Cline (sidebar): one-click install.
- Каталог MCP-серверов из MCP.so / Glama / Smithery.
- Auth: каждый server разруливает сам (env vars, OAuth, токены).

Это **самая правильная модель** для нас, потому что:
- MCP — стандарт, не self-rolled.
- Один формат конфига работает для Cline, Cursor, Claude Desktop, нас.
- Sandboxing через subprocess + filesystem permissions.

URL: https://docs.cline.bot

### 5.4 VS Code Extension API — для аналогии

- Manifest `package.json` с `contributes.commands`, `contributes.menus`, …
- Activation events (`onCommand:foo`, `onLanguage:python`).
- Sandboxing: extension host — отдельный Node process, isolated from
  renderer.
- Marketplace: Microsoft hosted, signed packages, ratings, telemetry.

Слишком тяжёлая модель для нашего use case (нам не нужны kommandy в UI).
Но **principle of separate process** перенимаем.

### 5.5 Hugging Face Spaces / dynamic_space MCP

MCP-сервер `hf-mcp-server` экспонирует `dynamic_space` — можно spawn'ить
Gradio/Streamlit space по требованию из агента. Хороший pattern для
"спецоборудование" в Web UI (ML demos, viewers).

### 5.6 Архитектурные варианты для PromeServe

| Архитектура | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **Subprocess + MCP (JSON-RPC)** | стандарт, sandboxed by OS, любой язык | overhead spawn, IPC latency | **✅ выбор #1** |
| **WASM modules (Extism/wasmtime)** | strong sandbox, memory-safe, multi-language | сложнее писать плагины, нужна WASI | опционально для Code Mode (untrusted code exec) |
| **Embedded Python (`Python.h`)** | мощно, любые pkgs | GIL, нет sandbox, утечки память в C++ | ❌ не делаем |
| **Lua scripts** | lightweight (~200KB), легко embed | мало готовых ML lib | ❌ слабее WASM |
| **HTTP webhook** | language-agnostic, network-friendly | latency, auth complexity | для Этапа 3 как **дополнение** к stdio MCP |

**Решение для PromeServe:** **plugin = MCP server**, конфиг в
`~/.config/promeserve/mcp.json` (формат Cline). Sandboxed code execution
(в Code Mode) — через WASM/Extism как отдельный capability built-in tool
`run_wasm`. Embedded Python и Lua не делаем.

Источники:
- Open WebUI plugin docs: https://docs.openwebui.com/features/extensibility/plugin/
- LibreChat plugin docs: https://docs.librechat.ai/features/plugins/chatgpt_plugins_openapi.html
- WASM sandboxing for AI: https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/
- Extism LLM sandbox: https://extism.org/blog/sandboxing-llm-generated-code/

---

## 6. Computer-use / Browser automation

### 6.1 Claude Computer Use (Anthropic)

- Доступен с октября 2024 (Claude 3.5 Sonnet beta), production в Claude
  Opus 4.6 и Opus 4.7 (2026).
- Built-in tool `computer_20250124` (или новее) — модель эмитирует
  `tool_use` с `action: "screenshot"/"left_click"/"type"/"key"/"mouse_move"`
  и пр. Координаты в пикселях.
- Контракт: client делает screenshot → шлёт base64 → Claude reasoning →
  Claude отвечает action → client исполняет в VM/Docker → screenshot →
  loop.
- **Vision-первый подход** — модель видит pixels.
- Requires: VM/Docker desktop, virtual mouse driver, screenshot tool.

Для PromeServe реалистично:
- ❌ Наши локальные модели (qwen3-4B) **не видят images** — нет vision
  backbone. Computer Use в этом стиле требует multimodal VL модель.
- ✅ Можно делегировать в **Playwright MCP** (см. ниже) — там accessibility
  tree вместо screenshots, текстовая модель справится.

URL: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool

### 6.2 OpenAI Operator

- Доступ через subscription (ChatGPT Pro $200/mo), не публичный SDK на
  май 2026.
- Architecture: ~похожа на Computer Use (screenshots + actions), плюс
  Browser only (не desktop).
- Нет open SDK — нельзя самостоятельно интегрировать.

### 6.3 Playwright MCP (Microsoft)

- GitHub: https://github.com/microsoft/playwright-mcp
- npm: `@playwright/mcp` (`npx @playwright/mcp@latest`)
- Docker: `mcr.microsoft.com/playwright/mcp`
- **Ключевое отличие от Claude/Operator** — использует **accessibility tree**
  (ARIA roles, names, structured DOM dump), а не screenshots. **Текстовая
  модель умеет с этим работать.**
- Tools: `browser_navigate`, `browser_click`, `browser_type`, `browser_snapshot`
  (a11y dump), `browser_screenshot`, `browser_press`, `browser_evaluate` (JS).
- Поддерживает Chromium, Firefox, WebKit, MS Edge.
- **Это наш выбор для Computer Use.**

### 6.4 browser-use (browser-use/browser-use)

- GitHub: https://github.com/browser-use/browser-use
- Python, 79k+★
- Chromium через CDP. HTML → LLM решает действие → выполнить → repeat.
- Любая LLM поддерживается, можно self-hosted.
- Менее структурированный чем Playwright MCP (raw HTML), но проще
  написать новые actions.

### 6.5 Open-Interpreter

- GitHub: https://github.com/openinterpreter/open-interpreter
- 57k+★, terminal-based.
- Не browser, а **shell/Python/JS exec в локальном окружении**.
- Не sandboxed (даёт root доступ к локальной машине).
- Полезен как reference для Code Mode (как LLM пишет код, exec'ит,
  читает stderr, итерирует).

### 6.6 Что реалистично интегрировать

**Этап 5 для PromeServe (Computer Use Mode):**
1. **Сначала: Playwright MCP в качестве external server.** Поднимаем
   `npx @playwright/mcp` через subprocess, регистрируем tools в наш
   registry. qwen3-4B/7B справляется с accessibility tree (тестировали
   на похожих структурах). Это даёт browser automation **бесплатно**.
2. **Опционально: browser-use** как Python-плагин (если хочется raw HTML
   подход) — но Playwright MCP лучше.
3. **Computer Use** в стиле Anthropic — **только когда у нас будет
   multimodal модель** (qwen-VL 7B в roadmap'е после tok/s).
4. **Shell agent (Open Interpreter style)** — уже есть `bash_safe`. Расширим
   до Docker-sandbox `bash_docker` для Code Mode.

Источники:
- Playwright MCP repo: https://github.com/microsoft/playwright-mcp
- Anthropic CU docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
- browser-use docs: https://docs.browser-use.com/open-source/introduction
- Computer Use agents comparison 2026: https://www.digitalapplied.com/blog/computer-use-agents-2026-claude-openai-gemini-matrix

---

## 7. Deep Research системы

### 7.1 OpenAI Deep Research

- Доступ: ChatGPT Pro/Team.
- Архитектура: reasoning model (o-series) с RL обучением на multi-step
  research. Этапы: clarifying questions → plan → search/browse → synthesize.
- Уникально: **clarification phase** — модель сама задаёт уточняющие
  вопросы перед началом длинной (5–30 мин) работы.
- Closed source.

### 7.2 Gemini Deep Research / Deep Research Max

- Gemini 2.5 / 3 Pro с режимом "Deep Research".
- Архитектура (по public talks):
  - **Async task manager** для long-running inference (минуты, не секунды).
  - **1M token context window** + RAG fallback для overflow.
  - Interleaved search → read → search loop.
  - В выводе — research report со ссылками.
- Deep Research Max (апрель 2026) — улучшенный, доступен через Gemini API.

### 7.3 Perplexity Pro Search

- Stateless query → search → synthesize.
- Менее multi-step чем OpenAI/Gemini Deep Research, но самый быстрый
  отклик (~30 s).
- Хороший reference для "lite" research режима.

### 7.4 GPT-Researcher (assafelovic/gpt-researcher)

- GitHub: https://github.com/assafelovic/gpt-researcher
- License: MIT
- **Multi-agent open-source clone** Deep Research.
- Agents: **planner** (декомпозиция query → sub-questions), **researchers**
  (web search per sub-q), **publisher** (synthesis с цитированиями).
- Output: markdown report с inline references.

### 7.5 STORM (Stanford)

- GitHub: https://github.com/stanford-oval/storm
- License: MIT, paper NAACL 2024
- Идея: writes Wikipedia-style articles from scratch.
- Two phases:
  1. **Pre-writing** — Perspective-Guided Question Asking (LLM surveys
     existing Wiki articles по теме, генерирует разные perspectives) +
     Simulated Conversation (writer ↔ topic expert grounded in web search).
  2. **Writing** — outline → expand sections → cross-reference + citations.
- Использует DSPy для end-to-end оптимизации.
- **Лучший reference для open-source deep research** — обкатан на тысячах
  тем, есть UI (Co-STORM для interactive).

### 7.6 Open Deep Research forks

- HuggingFace open-deep-research (smolagents-based): https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research
- LangChain open_deep_research: https://github.com/langchain-ai/open_deep_research
- assafelovic/gpt-researcher (см. 7.4)
- ai-engineer-foundation/anthropic-deep-research (Anthropic-style clone)

### 7.7 Архитектурные паттерны Deep Research

Общий **outline-first** паттерн:

```
1. CLARIFY (опционально)
   LLM → list of ambiguities → user answers → updated query

2. PLAN
   LLM → outline (3–8 sections) + sub-questions per section

3. RESEARCH (parallel)
   for each sub-question:
     web_search(q) → top-N results → fetch + extract → summarize
     reflect: достаточно ли? если нет → refine query → search again

4. SYNTHESIZE
   per section: combine summaries + cite → draft section
   global: cross-reference, dedupe, format

5. POLISH
   final pass: improve prose, fix citations, add executive summary

6. RETURN
   markdown/PDF with citations
```

Под капотом — ReAct/Plan-and-Execute с очень аккуратным **memory management**:
вместо запихивания всех scraped страниц в контекст, каждая страница
суммаризуется отдельным LLM call, и в финальный синтез идут summaries +
citations.

Для PromeServe Deep Research Mode (Этап 5):
- Используем нашу TP-4 модель (qwen3-4B, ~11 tok/s) для plan + synthesis.
- Web search через **fetch MCP server** + **brave-search MCP server**
  (key из env).
- Storage промежуточных результатов в `memory MCP server` (knowledge graph).
- Output — markdown с inline links + опционально PDF (мы это уже умеем).

Источники:
- STORM repo: https://github.com/stanford-oval/storm
- GPT-Researcher: https://github.com/assafelovic/gpt-researcher
- Comparison: https://leehanchung.github.io/blogs/2025/02/26/deep-research/
- Gemini DR: https://blog.google/products/gemini/google-gemini-deep-research/
- ByteByteGo deep research: https://blog.bytebytego.com/p/how-openai-gemini-and-claude-use

---

## 8. Конкретный план для PromeServe

PromeServe сейчас — простой HTTP сервер на C++ (`promeserve/main.cpp`,
`promeserve/api_handlers.h`). Endpoints: `/api/chat`, `/api/generate`,
`/api/models`. Уже есть **tool_call.h** с Qwen-стиль `<tool_call>` парсером
и 4 built-in tools (write_file, read_file, list_dir, bash_safe + есть `external=true` слот
для client-supplied tools). Item #89 (`Tool-call loop + MCP в PromeServe`) уже in_progress.

### Этап 1 (MVP+) — расширить `tool_call.h`

**Цель.** Доделать tool-call loop до production: streaming-stop,
parallel tool calls, audit log, http_get, http_post built-in, max-iter
guard.

**Что писать:**
- `promeserve/tool_call.h` — добавить:
  - `detect_all_tool_calls(text)` → vector<ToolCall> (parallel parse).
  - Streaming-stop callback в SamplingLoop: при появлении `</tool_call>`
    в выводе — остановить generation, выполнить, продолжить с
    `<tool_response>...</tool_response>` в prompt'е.
  - `http_get(url)` (libcurl или ручной socket), `http_post(url, body)`.
  - `bash_docker(cmd)` — wrap в `docker run --rm -v $PWD:/work:ro alpine`
    для Code Mode.
  - Audit log: append JSONL в `$XDG_STATE_HOME/promeserve/audit.jsonl`.
  - Iteration limit env var: `PROMESERVE_MAX_TOOL_ITER=10`.
- `promeserve/api_handlers.h` — на `/api/chat`:
  - Parse `tools[]` из request body (client-supplied tools).
  - Call `registry.register_external_tools(...)`.
  - Запустить agent loop:
    ```
    while iter < max_iter:
      out = generate(messages + format_with_tools)
      calls = detect_all_tool_calls(out)
      if not calls: return final
      for call in calls:
        if registry.is_external(call.name):
          return out + tool_calls echoed to client    # client executes
        result = registry.execute(call.name, call.args)
        append <tool_response>{result}</tool_response>
      iter++
    return error("max iter")
    ```

**Объём:** ~300 LoC C++.
**Deps:** только то что уже есть (libcurl можно опционально).
**Test:** `tests/promeserve/test_tool_call.cpp` (mock generate + 3
сценария: single call, parallel, max-iter).

### Этап 2 — MCP client

**Цель.** PromeServe умеет подключаться к external MCP серверам и
использовать их tools как свои.

**Что писать:**
- `promeserve/mcp_client.h` (~600 LoC):
  - `class MCPSubprocess` — fork+pipe, stdin/stdout JSON-RPC.
  - `class MCPClient` — connect/initialize/list_tools/call_tool/list_resources/
    read_resource/list_prompts/get_prompt/close. Async ID matching через
    map<int, promise<json>>.
  - JSON parsing — взять `nlohmann/json` (header-only, в нашем дереве уже
    был для GGUF), но если хотим без deps — расширить `extract_string_field`
    нашу примитивную парсилку.
- `promeserve/mcp_config.h` — парсить `~/.config/promeserve/mcp.json`
  формата Cline:
  ```json
  {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/promeserve"]
      },
      "memory": {"command":"npx","args":["-y","@modelcontextprotocol/server-memory"]},
      "fetch":  {"command":"uvx","args":["mcp-server-fetch"]},
      "github": {
        "command":"npx","args":["-y","@modelcontextprotocol/server-github"],
        "env":{"GITHUB_PERSONAL_ACCESS_TOKEN":"$GITHUB_TOKEN"}
      }
    }
  }
  ```
- На startup: для каждого entry — spawn'ить subprocess, initialize,
  cache tools/list. Register each tool в наш `ToolRegistry` как
  `mcp.<server>.<tool_name>`, executor → forward в MCP client.
- Lifecycle: process supervisor — restart на crash, log to audit.

**Объём:** ~700 LoC (client + config + tests).
**Deps:** subprocess (fork+pipe POSIX, CreateProcess Win), nlohmann/json
рекомендуется.
**Test:** spawn'нуть Everything ref server (`npx @modelcontextprotocol/server-everything`)
и прогнать tools/resources/prompts.

### Этап 3 — MCP server (PromeServe expose'ит себя)

**Цель.** Claude Desktop, Cline, Cursor, Continue подключаются к нашему
PromeServe и видят его как MCP сервер с tools + resources + prompts.

**Что експонируем:**
- **Tools:**
  - `prometorch.generate(model, prompt, max_tokens, temperature)` — generate text.
  - `prometorch.embed(model, text)` — embeddings (если у нас будут).
  - `prometorch.tokenize(model, text)` — encode/decode (у нас уже есть Qwen3 BPE bit-exact).
  - `prometorch.list_models()` — perепаковка `/api/models`.
- **Resources:**
  - `model://qwen3-4b/info` — JSON c metadata (vocab, params, quant).
  - `model://qwen3-4b/tokenizer` — raw tokenizer.json (для debug).
  - `log://session/<id>` — recent generations history.
  - `bench://qwen3-4b/latest` — последний tok/s бенчмарк.
- **Prompts:**
  - `prometorch/elbrus-bench` — готовый промпт для воспроизведения наших
    бенчей.
  - `prometorch/translate-ru-en` — translation с системным промптом.

**Что писать:**
- `promeserve/mcp_server.h` (~800 LoC):
  - Streamable HTTP transport: один endpoint `POST /mcp` для JSON-RPC,
    `GET /mcp` для SSE notifications.
  - Также **stdio mode**: если запущены `promeserve --mcp-stdio`, читаем
    JSON-RPC из stdin, пишем в stdout. Это режим для Claude Desktop
    (он spawn'ит как subprocess).
  - Реализовать handler'ы для всех методов из спеки 2025-11-25.
  - Capabilities declaration: `tools`, `resources` (subscribe=true для
    bench/log), `prompts`.
- `~/.config/Claude/claude_desktop_config.json` пример:
  ```json
  {
    "mcpServers": {
      "prometorch": {
        "command": "/usr/local/bin/promeserve",
        "args": ["--mcp-stdio"],
        "env": {"PROMESERVE_MODEL_DIR":"/opt/models"}
      }
    }
  }
  ```

**Объём:** ~800 LoC.
**Deps:** ничего нового.
**Test:** запустить с Claude Desktop, Cline, Cursor — проверить что
видны tools.

### Этап 4 — Plugin system (== MCP servers)

**Цель.** Расширяемость PromeServe = добавление MCP server в `mcp.json`.
Никакого собственного plugin API не делаем.

**Что писать:**
- `promeserve/web/plugins.html` — UI для просмотра подключённых MCP
  серверов, статуса, инструментов, ресурсов. Кнопки enable/disable,
  reload, view logs.
- `promeserve/api_handlers.h`: новые endpoints
  - `GET /api/plugins` — список MCP-серверов + статус.
  - `POST /api/plugins` — добавить server.
  - `DELETE /api/plugins/:name` — удалить.
  - `POST /api/plugins/:name/reload` — restart subprocess.
- **Опционально**: marketplace browser в Web UI — pull списка с Glama
  (REST API `https://glama.ai/api/mcp/v1/servers`) с пагинацией, поиском,
  one-click install (записать в `mcp.json` + reload).
- **WASM plugins (опция для Code Mode):** `promeserve/wasm_plugin.h` — embed
  `wasmtime` C-API. Built-in tool `run_wasm(module, fn, args)`. Конфиг
  в `~/.config/promeserve/wasm/*.wasm`. Pyodide+Deno для Python кода
  в sandbox (для Smolagents-style code agents).

**Объём:** ~400 LoC (без marketplace), +500 с marketplace, +600 с WASM.
**Deps:** опционально wasmtime C-API.

### Этап 5 — Modes (Code / Deep Research / Computer Use)

**Цель.** Готовые presets агента под три use case.

**5a. Code Mode** (~1 неделя)
- Базируется на SWE-agent / OpenHands паттернах.
- Tools: `read_file`, `write_file`, `edit_file` (diff-based, как у Cline),
  `bash_docker` (sandboxed), `git_status`, `git_diff`, `run_tests`,
  `search_code` (ripgrep wrapper).
- Plugin: filesystem MCP + git MCP + sequential-thinking MCP.
- System prompt: "You are a coding agent. Plan → edit → test → iterate."
- UI: split screen с file tree + chat + diff viewer.

**5b. Deep Research Mode** (~2 недели)
- Архитектура STORM-style: outline → sub-questions → search → synthesize.
- Tools: `web_search` (через brave-search MCP), `fetch` (через fetch MCP),
  `memory_add`/`memory_query` (через memory MCP), `write_file` для драфтов.
- Многошаговый loop с reflection (Reflexion-style для retry на bad answers).
- Output: markdown report с inline `[ref-N]` ссылками + bibliography.
- Опционально PDF (у нас уже есть pipeline).

**5c. Computer Use Mode** (~3 дня после готовности Playwright MCP)
- Plugin: Playwright MCP (`npx @playwright/mcp`).
- Tools: автоматически exposed `browser_navigate`, `browser_click`,
  `browser_type`, `browser_snapshot`, `browser_screenshot`.
- System prompt: "You control a browser. Use accessibility snapshot
  to plan actions."
- Полезно для: сайт-парсинг, login flows, form filling, end-to-end
  tests.

**5d. Бонус: Voyager-style skill library**
- Tool `save_skill(name, code, description)` — сохраняет полезный
  shell/python snippet в `~/.local/share/promeserve/skills/`.
- Tool `recall_skill(query)` — embeddings retrieval по описаниям.
- Skills видны в Code Mode и Deep Research Mode.

**Объём этапа 5 (mode UI + presets):** ~2000 LoC C++ + ~3000 LoC
HTML/JS/CSS.

### Сводная таблица этапов

| Этап | Что | LoC | Внешние deps | Срок |
|------|-----|-----|--------------|------|
| 1 | tool_call.h MVP+ | ~300 | libcurl (опц.) | 1–2 дня |
| 2 | MCP client | ~700 | nlohmann/json (рек.) | 3–5 дней |
| 3 | MCP server | ~800 | — | 5–7 дней |
| 4 | Plugin system + UI | ~400–1500 | wasmtime (опц.) | 3–5 дней |
| 5 | Modes (Code/DR/CU) | ~5000 | + node для MCP servers | 2–3 недели |

**Итого:** ~7–10k LoC за месяц при steady темпе. Возможно меньше, если
сократить marketplace UI и WASM до Этапа 6.

### Что НЕ делать

- ❌ Не встраивать LangChain/CrewAI/AutoGen внутрь PromeServe. Они работают
  поверх как клиенты — мы остаёмся OpenAI-compat + MCP сервер.
- ❌ Не делать свой plugin API. MCP — стандарт, всё уже есть.
- ❌ Не embed'ить Python через `Python.h`. GIL + утечки.
- ❌ Не делать Computer Use в стиле Anthropic пока нет VL модели. Идти
  через Playwright MCP (accessibility tree, текстовая модель справляется).
- ❌ Не self-roll'ить OAuth — отложить до момента, когда реально нужен
  remote multi-tenant.

### Что обязательно сделать как первый шаг

1. **Доделать Этап 1** (tool_call.h — parallel calls, streaming stop,
   http_get, audit log). После этого `<tool_call>` стабилен.
2. **Подключить первый MCP server** (Этап 2 на минимуме: только filesystem
   server, без полного config'а). Подтвердить что наш JSON-RPC работает.
3. **Smoke test** с qwen3-4B TP-4 на Эльбрусе через `curl /api/chat` —
   модель должна использовать MCP filesystem tools (mkdir/list/read).
4. **Commit + JOURNAL.md** — каждый шаг.

---

## Топ-5 ссылок для команды

1. **MCP spec hub** — основной транспорт + JSON-RPC + примитивы:
   https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
2. **Official MCP servers repo** — все ref-серверы (filesystem, git, fetch,
   memory, sequential-thinking и т.д.): https://github.com/modelcontextprotocol/servers
3. **Rust SDK (`rmcp`)** — лучший reference для C++ port'а:
   https://github.com/modelcontextprotocol/rust-sdk
4. **Qwen function calling docs** — формальное описание Hermes-style
   `<tool_call>` (наш формат): https://qwen.readthedocs.io/en/latest/framework/function_call.html
5. **Playwright MCP** — Computer Use для текстовых моделей через
   accessibility tree: https://github.com/microsoft/playwright-mcp

---

## Полный список источников

### MCP
- Spec hub: https://modelcontextprotocol.io
- Transports: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
- Tools: https://modelcontextprotocol.io/specification/2025-11-25/server/tools
- Resources: https://modelcontextprotocol.io/specification/2025-03-26/server/resources
- Authorization: https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization
- Official registry: https://registry.modelcontextprotocol.io
- Reference servers: https://github.com/modelcontextprotocol/servers
- Python SDK: https://github.com/modelcontextprotocol/python-sdk
- TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk
- Rust SDK: https://github.com/modelcontextprotocol/rust-sdk
- DeepWiki Rust: https://deepwiki.com/modelcontextprotocol/rust-sdk
- MCP cheatsheet 2026: https://www.webfuse.com/mcp-cheat-sheet
- OpenTelemetry semconv: https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/
- DEV: complete guide 2026: https://dev.to/x4nent/complete-guide-to-mcp-model-context-protocol-in-2026-architecture-implementation-and-4a11
- Message types ref: https://portkey.ai/blog/mcp-message-types-complete-mcp-json-rpc-reference-guide/
- OAuth on MCP: https://www.permit.io/blog/oauth-on-mcp
- OAuth + DCR: https://stytch.com/blog/mcp-oauth-dynamic-client-registration/
- Stack Overflow blog: https://stackoverflow.blog/2026/01/21/is-that-allowed-authentication-and-authorization-in-model-context-protocol/
- Gentoro on registry: https://www.gentoro.com/blog/what-is-anthropics-new-mcp-registry/
- Glama registry: https://glama.ai/mcp/servers
- Smithery: https://smithery.ai
- mcp.so: https://mcp.so
- mcpfinder: https://github.com/mcpfinder/mcpfinder
- Curated lists: https://github.com/wong2/awesome-mcp-servers, https://github.com/abordage/awesome-mcp, https://github.com/appcypher/awesome-mcp-servers
- Cursor MCP servers: https://github.com/cursor/mcp-servers
- GitHub MCP server: https://github.com/github/github-mcp-server
- Microsoft MCP catalog: https://github.com/microsoft/mcp
- Apigene marketplace guide: https://apigene.ai/blog/mcp-marketplace

### Tool calling formats
- Qwen docs: https://qwen.readthedocs.io/en/latest/framework/function_call.html
- Qwen3 repo: https://github.com/QwenLM/Qwen3
- vLLM tool calling: https://docs.vllm.ai/en/latest/features/tool_calling/
- Hermes-4-14B card: https://huggingface.co/NousResearch/Hermes-4-14B
- Llama 3.1 prompt format: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
- Llama 3.3 prompt format: https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/prompt_format.md
- Mistral JSON mode: https://docs.mistral.ai/capabilities/structured_output/json_mode
- Function calling vs tool use: https://propelius.ai/blogs/function-calling-vs-tool-use-ai-agents/
- Structured output comparison: https://www.glukhov.org/llm-performance/benchmarks/structured-output-comparison-popular-llm-providers
- Instructor modes comparison: https://python.useinstructor.com/modes-comparison/
- LLM tool calling guide: https://tonyseah.medium.com/llm-tool-calling-complete-guide-from-server-configuration-to-client-implementation-9e8a4552af12
- tlcl (llama.cpp tool caller): https://github.com/fairydreaming/tlcl

### Agent loops
- ReAct paper: https://arxiv.org/abs/2210.03629
- ReAct prompting guide: https://www.promptingguide.ai/techniques/react
- ToT paper: https://arxiv.org/abs/2305.10601
- ToT repo: https://github.com/princeton-nlp/tree-of-thought-llm
- ToT prompting guide: https://www.promptingguide.ai/techniques/tot
- Reflexion paper: https://arxiv.org/abs/2303.11366
- Reflexion repo: https://github.com/noahshinn/reflexion
- Voyager paper: https://arxiv.org/abs/2305.16291
- Voyager repo: https://github.com/MineDojo/Voyager
- Plan-and-execute: https://blog.langchain.com/planning-agents/
- SWE-agent: https://github.com/SWE-agent/SWE-agent
- mini-SWE-agent: https://github.com/SWE-agent/mini-swe-agent
- OpenHands: https://github.com/All-Hands-AI/OpenHands
- ShinkaEvolve (Sakana): https://github.com/SakanaAI/ShinkaEvolve
- BabyAGI history: https://babyagi.wiki

### Frameworks
- CrewAI: https://github.com/crewAIInc/crewAI
- AutoGen: https://github.com/microsoft/autogen
- LangGraph: https://github.com/langchain-ai/langgraph
- Claude Agent SDK: https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk + https://platform.claude.com/docs/en/agent-sdk/overview
- OpenAI Agents SDK Python: https://github.com/openai/openai-agents-python (+ js: https://github.com/openai/openai-agents-js)
- OpenAI Agents docs: https://openai.github.io/openai-agents-python/
- LlamaIndex agents: https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/
- LlamaIndex AgentWorkflow blog: https://www.llamaindex.ai/blog/introducing-agentworkflow-a-powerful-system-for-building-ai-agent-systems
- Pydantic AI: https://pydantic.dev/docs/ai/overview/
- Smolagents: https://github.com/huggingface/smolagents
- Smolagents docs: https://huggingface.co/docs/smolagents/en/index
- Agno: https://github.com/agno-agi/agno
- 2026 framework comparisons: https://pratikpathak.com/langgraph-vs-crewai-vs-autogen-2026/, https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared, https://medium.com/data-science-collective/langgraph-vs-crewai-vs-autogen-which-agent-framework-should-you-actually-use-in-2026-b8b2c84f1229

### Plugins / Marketplaces
- Open WebUI plugin docs: https://docs.openwebui.com/features/extensibility/plugin/
- Open WebUI Tools: https://docs.openwebui.com/features/extensibility/plugin/tools/
- Open WebUI Functions: https://docs.openwebui.com/features/extensibility/plugin/functions/
- Open WebUI Pipelines: https://docs.openwebui.com/features/extensibility/pipelines/
- Pipelines repo: https://github.com/open-webui/pipelines
- LibreChat plugin docs: https://docs.librechat.ai/features/plugins/chatgpt_plugins_openapi.html
- LibreChat manifest example: https://github.com/danny-avila/LibreChat/blob/main/api/app/clients/tools/manifest.json
- Cline: https://github.com/cline/cline
- Cline docs: https://docs.cline.bot
- Cline MCP marketplace: https://news.ycombinator.com/item?id=43105538
- Extism: https://extism.org/blog/sandboxing-llm-generated-code/
- NVIDIA WASM agents: https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/

### Computer Use / Browser
- Claude Computer Use docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
- Anthropic introducing computer use: https://www.anthropic.com/news/3-5-models-and-computer-use
- claude-quickstarts: https://github.com/anthropics/claude-quickstarts
- Playwright MCP (Microsoft): https://github.com/microsoft/playwright-mcp
- executeautomation/mcp-playwright: https://github.com/executeautomation/mcp-playwright
- browser-use: https://github.com/browser-use/browser-use
- browser-use docs: https://docs.browser-use.com/open-source/introduction
- open-interpreter: https://github.com/openinterpreter/open-interpreter
- Computer Use 2026 comparison: https://www.digitalapplied.com/blog/computer-use-agents-2026-claude-openai-gemini-matrix

### Deep Research
- STORM repo: https://github.com/stanford-oval/storm
- STORM project: https://storm-project.stanford.edu/research/storm/
- GPT Researcher: https://github.com/assafelovic/gpt-researcher
- Gemini Deep Research: https://gemini.google/overview/deep-research/
- Gemini Deep Research API: https://ai.google.dev/gemini-api/docs/deep-research
- ZenML on Gemini DR: https://www.zenml.io/llmops-database/building-gemini-deep-research-an-agentic-research-assistant-with-custom-tuned-models
- Lee Han Chung comparison: https://leehanchung.github.io/blogs/2025/02/26/deep-research/
- ByteByteGo blog: https://blog.bytebytego.com/p/how-openai-gemini-and-claude-use
- Step-DeepResearch arxiv: https://arxiv.org/pdf/2512.20491
- HF smolagents open_deep_research: https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research
- LangChain open_deep_research: https://github.com/langchain-ai/open_deep_research

---

**Конец отчёта R2. Следующий шаг — Этап 1 в коде.**
