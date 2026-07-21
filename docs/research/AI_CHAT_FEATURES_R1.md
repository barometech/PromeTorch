# Современные фичи AI-чатов — отчёт R1 для PromeServe

**Дата:** 2026-05-20
**Автор:** исследовательский агент (Claude Opus 4.7)
**Цель:** каталог фичей лучших open-source AI-чатов и локальных LLM-интерфейсов, чтобы команда PromeServe (C++ HTTP сервер для inference моделей через PromeTorch) могла приоритезировать UI и архитектурные решения.

**Контекст PromeServe (snapshot):**
- `promeserve/http_server.h` (17 KB) — HTTP listener.
- `promeserve/api_handlers.h` (59 KB) — `/api/chat`, `/api/generate`, OpenAI-совместимые роуты.
- `promeserve/model_manager.h` + `model_loader.h` — загрузка GGUF, TP-4, NMCard диспатч.
- `promeserve/tool_call.h` (17 KB) — текущий tool-call loop (по `TOOL_CALL_PLAN.md`).
- `promeserve/web/index.html` (39 KB) — single-file vanilla web UI (нужно эволюционировать).

---

## TL;DR — 15 must-have фич для PromeServe

1. **Code Artifacts с inline preview** (LibreChat / Claude / Open WebUI): рендерим HTML, React-JSX, Mermaid, SVG прямо в чате внутри sandboxed `<iframe srcdoc>`. Лучший копируемый референс — `LibreChat/client/src/components/Artifacts`.
2. **MCP клиент (Streamable HTTP + stdio)** в PromeServe: подключаемся к любым публичным MCP-серверам (filesystem, github, brave-search, postgres). JSON-RPC 2.0 по spec `2025-06-18`.
3. **MCP сервер для PromeServe** (Streamable HTTP): открыть наш inference + tool registry как MCP endpoint, чтобы Claude Desktop / Cursor / VS Code могли подключаться к PromeServe как к источнику моделей и тулов.
4. **Streaming SSE** для `/api/chat` (token-by-token + tool-call deltas) — обязательно для UX. Ollama-совместимый формат уже у нас.
5. **Multi-format tool-call parser**: Hermes `<tool_call>{json}</tool_call>` (Qwen3), Llama3 `<|python_tag|>`, Mistral `[TOOL_CALLS]`, native OpenAI JSON. Унифицированный детектор у нас уже в `tool_call.h` — расширить.
6. **Code blocks с подсветкой + copy + run** — Shiki/Highlight.js + кнопка "Run in sandbox" для Python/JS.
7. **File attachments + RAG drag-drop**: PDF/DOCX/TXT chunking, vector store (любой из 9 в Open WebUI). Минимум — txt/md inline + image preview.
8. **Multimodal images**: base64 `<img>` отправка, рендеринг ответов с изображениями (генерация уже подключаемая через server tool).
9. **Voice input/output**: Whisper STT + KittenTTS/Piper TTS, hot-key push-to-talk.
10. **Plugin/Function system а-ля Open WebUI**: Python-файл с YAML frontmatter, embedded Python (pybind11) или subprocess JSON-RPC.
11. **Conversation history search + анкорные ссылки** на сообщения (`#msg-<uuid>`), оглавление (TOC) для длинных чатов справа.
12. **Deep Research mode**: supervisor + параллельные researcher агенты, итеративный web search, отчёт ≥2000 слов с цитатами (GPT-Researcher / Open Deep Research).
13. **Split chat / side-by-side comparison** (Msty, LibreChat): тот же prompt → две модели одновременно, сравнение outputs.
14. **Markdown export + share-link** (PDF через headless Chromium, готовый `wkhtmltopdf`).
15. **Workspaces / Knowledge Stacks**: изолированные RAG-стеки на проект (AnythingLLM / Msty knowledge stacks).

---

## 1. Frontend проектов — детальный каталог

### 1.1 Open WebUI

- **GitHub:** https://github.com/open-webui/open-webui
- **Stars:** 138k
- **License:** Open WebUI License (требует сохранять бренд "Open WebUI" в форках; не чистый MIT/Apache)
- **Тулчейн:** Svelte 4 + SvelteKit + Vite + TypeScript + Tailwind. Backend — Python (FastAPI). Frontend ≈ 32.8% Svelte, 23.9% JS, 5.2% TS.
- **Build:** `npm run build` → static bundle; backend подаёт через uvicorn.
- **Где смотреть код фронта:**
  - `src/lib/components/chat/` — главный chat UI
  - `src/lib/components/chat/Messages.svelte` — рендеринг сообщений
  - `src/lib/components/chat/MessageInput.svelte` — input bar
  - `src/lib/components/admin/Settings/` — admin panel
  - `static/` — статика
- **Фичи UI:**
  - **Artifacts** — "Persistent Artifact Storage" с KV API для журналов, трекеров, leaderboard-ов
  - **Code blocks** — встроенный Python редактор для function-calling tools
  - **Multimodal**: Whisper / OpenAI / Deepgram / Azure STT; Azure / ElevenLabs / OpenAI TTS; voice/video calls
  - **File handling**: локальный RAG, 9 vector DB (Chroma, PGVector, Qdrant, Milvus, Elasticsearch, OpenSearch, Pinecone, S3Vector, Oracle 23ai), `#` команда для документов
  - **Web search** — 15+ провайдеров (Brave, DuckDuckGo, Searxng, Tavily, Bing, Google PSE и т.д.)
  - **Model Builder** — GUI для создания Ollama-моделей с system prompts и character cards
  - **Pipelines** — внешний Python framework для filter/pipe/action функций с rate limiting, monitoring, translation, toxicity filter
- **Что копировать к нам в `promeserve/web/`:**
  - Архитектуру разделения `Tools` (LLM-вызываемые функции) vs `Functions` (Pipe/Filter/Action) vs `Pipelines` (out-of-process)
  - UI для управления tool calls (collapsible tool-use blocks)
  - Admin panel pattern (settings, users, models)
  - Voice call UX (hands-free push-to-talk)

### 1.2 LibreChat

- **GitHub:** https://github.com/danny-avila/LibreChat
- **Stars:** 37.2k
- **License:** MIT
- **Тулчейн:** React + Vite + TypeScript + Tailwind, монорепо через Turbo. ≈ 76.7% TS, 22.5% JS.
- **Build:** `npm run build:client && npm run build:api`
- **Где смотреть код фронта:**
  - `client/src/` — React приложение
  - `client/src/components/Chat/` — chat UI
  - `client/src/components/Artifacts/` — Code Artifacts (React, HTML, Mermaid)
  - `client/src/components/SidePanel/` — боковая панель с агентами/пресетами
  - `packages/` — shared types и data-provider
- **Фичи UI:**
  - UI вдохновлён ChatGPT, но с расширениями
  - **Multi-model**: Anthropic, OpenAI, Azure, Google, AWS Bedrock, OpenRouter и любой OpenAI-совместимый endpoint
  - **Code Interpreter** sandbox через E2B/Riza с поддержкой Python, JS, Ruby и др.
  - **Agent framework** с MCP support и Skills System
  - **Code Artifacts** для React/HTML/Mermaid — рендерятся в правом сплите как Claude
  - **Web Search** с reranking
  - **Image generation/editing tools**
  - **Multi-user**: OAuth2, LDAP, email + 2FA
  - **Import/export**: conversation в JSON, поиск по истории
  - **Agent Marketplace** — community-built agents
- **Plugin/extension:**
  - Native MCP integration
  - Custom Endpoints (любой OpenAI-совместимый API)
  - Subagents (делегация задач)
  - Skills System (переиспользуемые инструкции)
- **Что копировать:**
  - **React Artifacts pattern** — самый чистый референс для inline-preview (iframe + sandbox + hot reload)
  - Modular endpoint config (JSON-based)
  - Agent + Subagent UI

### 1.3 Chatbox

- **GitHub:** https://github.com/Bin-Huang/chatbox
- **Stars:** 40k
- **License:** GPL-3.0
- **Тулчейн:** Electron + React + Vite + TypeScript (95.6% TS). Cross-platform: Windows, macOS, Linux, web, mobile.
- **Где смотреть код фронта:**
  - `src/main/` — Electron main process
  - `src/renderer/` — React UI
  - `src/preload/` — preload скрипты
  - `src/shared/` — общие утилиты
- **Фичи UI:**
  - Multiple LLM providers (OpenAI, Claude, Gemini, Ollama)
  - Local data storage (privacy-first)
  - Markdown + LaTeX + code highlighting
  - Team collaboration
  - 8+ языков, dark theme
- **Что копировать:**
  - Electron-style packaging для desktop версии (если PromeServe нужен tray app)
  - Local-first sync logic (CRDT для conversations)

### 1.4 NextChat (ChatGPT-Next-Web)

- **GitHub:** https://github.com/ChatGPTNextWeb/NextChat
- **Stars:** 88.1k
- **License:** MIT
- **Тулчейн:** Next.js + React + TypeScript (91.7%) + SCSS + Tauri для desktop. Bundle ~100 KB.
- **Где смотреть код фронта:**
  - `app/` — Next.js app router
  - `app/components/chat.tsx` — главный chat
  - `app/store/` — Zustand stores
  - `src-tauri/` — Tauri desktop
- **Фичи UI:**
  - 1-click deploy на Vercel
  - Cross-platform (Web/iOS/macOS/Android/Linux/Windows)
  - Privacy-first (всё в localStorage)
  - Markdown + LaTeX + Mermaid + code highlight
  - Dark mode + PWA
  - Streaming, fast init
  - **Prompt templates** — sharing tool
  - **Chat history compression** (auto-summarization)
  - **Plugins system** (с v2.15.0) — отдельный репо `NextChat-Awesome-Plugins`
  - 18+ языков
- **Что копировать:**
  - **PWA setup** — наш UI станет installable
  - Compression стратегия для длинных контекстов
  - Zustand-style state management (без Redux boilerplate)

### 1.5 Hugging Face Chat-UI

- **GitHub:** https://github.com/huggingface/chat-ui
- **Stars:** 10.7k
- **License:** Apache-2.0
- **Тулчейн:** SvelteKit + TypeScript (62.2%) + Svelte (35.8%) + Vite. Mongo для persistence.
- **Где смотреть код фронта:**
  - `src/lib/components/chat/` — chat compoenents
  - `src/lib/components/InferenceProvider.svelte` — провайдер выбора модели
  - `models/` — model configs (JSON)
- **Фичи UI:**
  - Powers HuggingChat
  - **MCP Tools** — calls tools from MCP servers
  - **LLM Router (Omni)** — server-side intelligent model selection через `Arch-Router-1.5B`
  - Multimodal (images)
  - Per-model tool calling toggle
- **Что копировать:**
  - **LLM Router pattern** — автоматический выбор модели по типу запроса (для PromeServe с разными моделями qwen3-1.7B fast / qwen3-14B slow это критично)
  - Mongo schema для conversations + assistants
  - Per-model capability flags

### 1.6 AnythingLLM

- **GitHub:** https://github.com/Mintplex-Labs/anything-llm
- **Stars:** 60.3k
- **License:** MIT
- **Тулчейн:** Vite + React (98.4% JS) + Express backend + Node collector service.
- **Где смотреть:**
  - `frontend/` — Vite + React
  - `server/` — Express
  - `collector/` — document processing (PDF, DOCX, etc.)
  - `embed/` — web widget (submodule)
  - `browser-extension/` — Chrome extension
- **Фичи UI:**
  - **Workspaces** — изолированные RAG-пространства с per-user permissions (Docker)
  - **No-code AI Agent builder** — drag-and-drop
  - **Document ingestion** PDF/TXT/DOCX/EPUB + vector DB
  - **MCP-compatibility**
  - 40+ LLM provider integrations
  - Browser extension + embeddable widget
- **Что копировать:**
  - **Workspaces** как primitive — каждый чат принадлежит workspace со своим набором документов, моделей, инструментов
  - **Embed widget** pattern — для интеграции PromeServe в чужие сайты
  - Document collector как отдельный сервис

### 1.7 Continue.dev

- **GitHub:** https://github.com/continuedev/continue
- **Stars:** 33.3k
- **License:** Apache 2.0
- **Тулчейн:** TypeScript (84.4%), Kotlin (3.8% — JetBrains plugin), Python, Rust.
- **Где смотреть:**
  - `core/` — общая логика
  - `extensions/vscode/` — VS Code extension
  - `extensions/intellij/` — JetBrains
  - `gui/` — React UI (renderer внутри VS Code webview)
  - `.continue/` — конфигурация (`config.json`, agents, rules)
  - `skills/` — skills modules
- **Фичи:**
  - **Source-controlled AI checks** — markdown checks в `.continue/checks/`
  - **Custom slash commands** через config
  - **Context providers** (`@file`, `@codebase`, `@docs`, `@url`, `@diff`)
  - **CLI tool** `cn`
  - **MCP servers** интеграция
  - **Autocomplete** (Tab inline) + **Chat** + **Agent mode**
  - VS Code + JetBrains + CLI
- **Что копировать:**
  - **Config schema** — `config.json` с моделями, контекст-провайдерами, slash-командами
  - **Context provider abstraction** — мощный паттерн для PromeServe (@file, @url, @repo)
  - **Skills** (markdown инструкций) как persistent промпты

### 1.8 Cherry Studio

- **GitHub:** https://github.com/CherryHQ/cherry-studio
- **Stars:** 46k
- **License:** AGPL-3.0 (коммерческая лицензия по запросу)
- **Тулчейн:** Electron + React + TypeScript (96.6%) + pnpm + vitest + playwright.
- **Где смотреть:**
  - `src/` — main код
  - `packages/` — модули
- **Фичи:**
  - Diverse LLM providers (OpenAI, Gemini, Anthropic, Ollama)
  - **300+ pre-configured AI assistants** (character cards)
  - Document processing
  - Global search
  - MCP Server support
  - **MCP Marketplace** (на roadmap)
- **Что копировать:**
  - **Assistant gallery** (предустановленные character cards)
  - Search across all conversations

### 1.9 Jan

- **GitHub:** https://github.com/janhq/jan
- **Stars:** 42.6k
- **License:** Apache 2.0
- **Тулчейн:** Tauri + TypeScript (75.4%) + Rust (19.7%) + Swift (1.5%) + Python.
- **Где смотреть:**
  - `core/` — ядро
  - `web-app/` — frontend (TS)
  - `src-tauri/` — Rust desktop wrapper
  - `extensions/` — система расширений
  - `mlx-server/` — Apple MLX backend
- **Фичи:**
  - Local AI (Llama, Gemma, Qwen, GPT-OSS) через llama.cpp
  - Cloud (OpenAI, Claude, Mistral, Groq, MiniMax)
  - **Custom Assistants**
  - **OpenAI-compatible API** на `localhost:1337`
  - Privacy-focused (всё локально по умолчанию)
  - **MCP integration**
- **Что копировать:**
  - **Tauri вместо Electron** (10× меньше bundle) — если решим делать desktop версию
  - **Extension system** в `/extensions` (TypeScript plugins)
  - OpenAI-compatible local API на фиксированном порту

### 1.10 Msty

- **Website:** https://msty.ai
- **License:** Proprietary (closed-source freemium)
- **Платформа:** Windows, macOS, Linux desktop (Electron-like)
- **Главные фичи (по сайту):**
  - **Side-by-side chat** — две модели одновременно
  - **Knowledge Stacks** — изолированные RAG-стеки на тему/проект
  - **Delve Mode** — deep research-style ответы с разворачиваемыми ветками рассуждений
  - **Prompts library** — пресеты
  - **Personas / Workflows / Automations**
  - **Private workspaces**
  - **Claw** — multi-step task agent с tool access
- **Что копировать (без копирования кода — он закрыт):**
  - **Split chat** UX (две модели рядом) — реально полезно для нашего qwen3-1.7B vs qwen3-14B сравнения
  - **Knowledge Stacks** как имя для workspaces
  - **Delve Mode** UX с разворачиваемыми "цепочками мыслей"

### 1.11 Сводная таблица

| Проект | Stars | License | Stack | MCP | Artifacts | Voice | Plugins |
|---|---|---|---|---|---|---|---|
| Open WebUI | 138k | OWUI License | Svelte+Py | через Pipelines | ✓ | ✓ | Tools/Functions/Pipelines |
| LibreChat | 37.2k | MIT | React+Node | native | ✓ React/HTML/Mermaid | через TTS plug | Agents + MCP |
| Chatbox | 40k | GPL-3.0 | Electron+React | нет | базовое markdown | нет | минимум |
| NextChat | 88.1k | MIT | Next.js+Tauri | через плагины | базовое | нет | Awesome-Plugins repo |
| HF Chat-UI | 10.7k | Apache 2.0 | SvelteKit | native | базовое | нет | через MCP |
| AnythingLLM | 60.3k | MIT | Vite+React+Express | compatible | базовое | нет | Agent skills |
| Continue | 33.3k | Apache 2.0 | TS+VSCode | native | inline | нет | Skills+MCP |
| Cherry Studio | 46k | AGPL-3.0 | Electron+React | native | базовое | нет | MCP Marketplace (план) |
| Jan | 42.6k | Apache 2.0 | Tauri+TS+Rust | native | базовое | нет | TS extensions |
| Msty | n/a | Proprietary | Electron-like | через Claw | ✓ | нет | внутренние workflows |

---

## 2. MCP + Tool-call + Agent loops

### 2.1 Model Context Protocol (MCP) — спецификация

- **Сайт:** https://modelcontextprotocol.io/introduction
- **Spec latest:** https://modelcontextprotocol.io/specification/latest
- **Spec freeze 2025-06-18** (текущая стабильная версия)
- **Архитектура:** клиент-серверная, JSON-RPC 2.0
- **Транспорты:**
  - **stdio** — для локальных серверов (Claude Desktop запускает subprocess)
  - **Streamable HTTP** — HTTP POST для request, optional SSE для streaming. Auth: OAuth, bearer, API key
  - (устаревший SSE-only тоже встречается)
- **Слои:**
  - **Data layer**: JSON-RPC сообщения, lifecycle, primitives
  - **Transport layer**: framing, auth, connection
- **Серверные примитивы:**
  - **Tools** — executable functions (file ops, API calls)
  - **Resources** — data sources (file contents, DB records). Методы `*/list`, `*/get`
  - **Prompts** — переиспользуемые templates
- **Клиентские примитивы (что сервер просит у клиента):**
  - **Sampling** (`sampling/createMessage`) — сервер просит клиента вызвать LLM
  - **Elicitation** (`elicitation/create`) — сервер просит пользователя ввести что-то
  - **Logging** — отправка логов клиенту
- **Cross-cutting:**
  - **Notifications** (без id) — `notifications/tools/list_changed` и т.д.
  - **Tasks (experimental)** — durable execution wrappers
- **Lifecycle:**
  1. Клиент `initialize` с `protocolVersion`, `capabilities`, `clientInfo`
  2. Сервер отвечает `protocolVersion`, `capabilities`, `serverInfo`
  3. Клиент `notifications/initialized`
  4. `tools/list` → массив `{name, title, description, inputSchema}`
  5. `tools/call` с `{name, arguments}` → `{content: [{type, text/image/resource}]}`
  6. Сервер может слать `notifications/tools/list_changed`

### 2.2 Реализация MCP в PromeServe — план

**MCP-сервер (PromeServe → внешний клиент):**

PromeServe должен слушать MCP запросы по Streamable HTTP на отдельном роуте `/mcp/v1/`:
- `POST /mcp/v1/` — JSON-RPC requests, ответ может быть SSE stream если запрос — sampling
- `GET /mcp/v1/` — establish SSE для server-initiated notifications

Экспонируем:
- **tools**: наш текущий tool registry (`write_file`, `read_file`, `http_get`, `bash`) + всё что добавим
- **resources**: загруженные модели (`promeserve://models/qwen3-4b/info`), конфиг, метрики
- **prompts**: шаблоны типа "summarize", "translate-to-ru"

Pseudocode header (`promeserve/mcp_server.h`):
```cpp
namespace promeserve::mcp {
class MCPServer {
  void handle_initialize(const json& req, json& resp);
  void handle_tools_list(const json& req, json& resp);
  void handle_tools_call(const json& req, json& resp);
  void handle_resources_list(...);
  void handle_resources_read(...);
  void send_notification(const std::string& method, const json& params);
  // SSE channel для notifications
  std::vector<SSEChannel*> sse_clients;
};
}
```

**MCP-клиент (PromeServe → внешние MCP-серверы):**

Когда PromeServe запускает агентский loop, он должен подгружать инструменты из external MCP-серверов:
- subprocess для stdio серверов (npm/uvx-style: `npx @modelcontextprotocol/server-filesystem /path`)
- HTTP клиент для Streamable HTTP

Конфиг (`promeserve/mcp_servers.json`):
```json
{
  "filesystem": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/promeserve"]
  },
  "brave-search": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
    "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}
  },
  "github": {
    "transport": "stdio",
    "command": "docker",
    "args": ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"]
  }
}
```

Pseudocode (`promeserve/mcp_client.h`):
```cpp
namespace promeserve::mcp {
class MCPClient {
  bool initialize(const ServerConfig& cfg);
  std::vector<ToolSchema> list_tools();
  json call_tool(const std::string& name, const json& args);
  // stdio через popen + fdopen, HTTP через curl_multi
};

class MCPManager {
  std::map<std::string, std::unique_ptr<MCPClient>> servers;
  std::vector<ToolSchema> all_tools(); // префикс tool с server_name
};
}
```

### 2.3 Топ-30 публичных MCP-серверов

Источники: https://github.com/modelcontextprotocol/servers, https://github.com/punkpeye/awesome-mcp-servers (87.2k stars), https://github.com/wong2/awesome-mcp-servers, https://github.com/tolkonepiu/best-of-mcp-servers (400 серверов, суммарно ~1M stars).

**Официальные reference серверы** (https://github.com/modelcontextprotocol/servers/tree/main/src):

| # | Имя | Путь | Назначение |
|---|---|---|---|
| 1 | Everything | `src/everything` | Тестовый сервер со всеми примитивами |
| 2 | Fetch | `src/fetch` | Загрузка web-страниц + конверсия в markdown |
| 3 | Filesystem | `src/filesystem` | Безопасные file ops с контролем доступа |
| 4 | Git | `src/git` | Read/search/manipulate Git репо |
| 5 | Memory | `src/memory` | Knowledge graph persistent memory |
| 6 | Sequential Thinking | `src/sequentialthinking` | Динамическое reflective problem-solving |
| 7 | Time | `src/time` | Time/timezone конверсия |

**Архивные официальные** (servers-archived, многие переехали в community):

| # | Имя | Назначение |
|---|---|---|
| 8 | GitHub (now in github/github-mcp-server) | Repos/issues/PRs/CI — 28.3k stars, официальный |
| 9 | Slack | Каналы/сообщения |
| 10 | PostgreSQL | Read-only DB access + schema inspection |
| 11 | SQLite | DB interaction + BI |
| 12 | Puppeteer | Browser automation, scraping |
| 13 | Google Drive | File access/search |
| 14 | Google Maps | Locations/directions/places |
| 15 | GitLab | GitLab API |
| 16 | Sentry | Issue retrieval/analysis |
| 17 | Redis | Key-value ops |
| 18 | Brave Search | Web/local search via Brave API (v2.x — 7 tools) |

**Топ community/коммерческие:**

| # | Имя | URL | Stars | Назначение |
|---|---|---|---|---|
| 19 | **Context7** | https://github.com/upstash/context7 | 55.7k | Up-to-date docs+examples для библиотек. Tools: `resolve-library-id`, `query-docs` |
| 20 | **Playwright MCP** | https://github.com/microsoft/playwright-mcp | 32.8k | Microsoft official browser automation через accessibility tree (детерминированно, без скриншотов) |
| 21 | **GitHub MCP** | https://github.com/github/github-mcp-server | 28.3k | 51 tool: repos/issues/PRs/workflows |
| 22 | **Exa Search** | https://github.com/exa-labs/exa-mcp-server | n/a | `web_search_exa`, `web_fetch_exa`, advanced filters |
| 23 | **Tavily** | https://github.com/tavily-ai/tavily-mcp | n/a | Search optimized для агентов |
| 24 | **Notion** | https://github.com/makenotion/notion-mcp-server | 1k+ | Pages/databases CRUD |
| 25 | **Linear** | https://github.com/linear/linear-mcp-server | n/a | Issues/projects |
| 26 | **Slack** community | https://github.com/korotovsky/slack-mcp-server | 1.6k | Channels/threads/users |
| 27 | **Cloudflare** | https://github.com/cloudflare/mcp-server-cloudflare | n/a | Workers/KV/D1/R2 управление |
| 28 | **AWS** | https://github.com/awslabs/mcp | n/a | EC2/S3/Lambda |
| 29 | **Kubernetes** | https://github.com/Flux159/mcp-server-kubernetes | n/a | kubectl-like ops |
| 30 | **Browser Use** | https://github.com/co-browser/browser-use-mcp-server | n/a | AI-driven browser автоматизация |
| 31 | **Stripe** | https://github.com/stripe/agent-toolkit | n/a | Payments через MCP |
| 32 | **Figma** | https://www.figma.com/developers/mcp | n/a | Designs/components |
| 33 | **Obsidian** | https://github.com/MarkusPfundstein/mcp-obsidian | n/a | Notes vault |
| 34 | **Spotify** | https://github.com/marcelmarais/spotify-mcp-server | n/a | Playback/playlists |
| 35 | **Calendar** (Apple) | https://github.com/Omnisia/apple-mcp | n/a | Mail/calendar/contacts |

### 2.4 Tool-call форматы — как унифицировать

**OpenAI function calling** (https://platform.openai.com/docs/guides/function-calling):
- `tools: [{type: "function", function: {name, description, parameters: JSONSchema}}]`
- Response: `message.tool_calls: [{id, type: "function", function: {name, arguments: jsonstring}}]`
- Tool result: `{role: "tool", tool_call_id, content}`
- Parallel calls: модель возвращает несколько `tool_calls` за раз
- `tool_choice`: `"auto"` / `"none"` / `"required"` / `{type: "function", function: {name}}`

**Anthropic tool use** (https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/overview):
- `tools: [{name, description, input_schema: JSONSchema}]`
- Server tools (Anthropic-side): `{type: "web_search_20260209"}`, `code_execution`, `web_fetch`
- Client tools (user-defined + bash/text_editor) — модель возвращает `stop_reason: "tool_use"` и content block `{type: "tool_use", id, name, input}`
- Result: user-message с content `{type: "tool_result", tool_use_id, content}`
- `tool_choice: {type: "auto"|"any"|"tool"|"none"}`
- `strict: true` гарантирует совпадение со schema
- Parallel — несколько tool_use в одном responseе

**MCP** — наш формат уже выше (tools/list → tools/call).

**Унификация в PromeServe** (расширение `tool_call.h`):
```cpp
struct ToolCall {
  std::string id;          // OpenAI id или Anthropic tool_use_id
  std::string name;
  json arguments;
  enum Format { OPENAI, ANTHROPIC, HERMES_QWEN, LLAMA3, MISTRAL };
};

struct ToolResult {
  std::string call_id;
  json content;  // string | array of content blocks
  bool is_error;
};

class ToolDispatcher {
  // model-specific форматтер промпта
  std::string render_tools(Format fmt, const std::vector<ToolSchema>&);
  // парсер из generation stream
  std::vector<ToolCall> parse(Format fmt, const std::string& output);
  // вызов: local registry | MCP server | sandbox
  ToolResult execute(const ToolCall&);
};
```

**Model-specific промпт-форматы** (уже в TOOL_CALL_PLAN.md):
- **Qwen3 / Hermes**: `<tool_call>{"name":"...","arguments":{...}}</tool_call>` внутри tag-блока
- **Llama 3.1+**: `<|python_tag|>` + JSON или `[{"name":...,"parameters":...}]`
- **Mistral**: `[AVAILABLE_TOOLS] [...] [/AVAILABLE_TOOLS]` + `[TOOL_CALLS][...]`
- **Gemma / Phi**: обычно plain JSON в text без special tokens
- **DeepSeek**: `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>`

Стандартный парсер: regex детекция нескольких форматов, fallback на JSON balanced object с `name` ключом.

### 2.5 Agent loops — реализации

**ReAct (Reason + Act):**
```
loop:
  thought = LLM(prompt + history)  // "Thought: ..."
  if thought.has_action:
    action, args = parse_action(thought)
    obs = execute(action, args)
    history += [thought, "Observation: " + obs]
  else:
    return thought.answer
```
Простой, baseline. Большинство Qwen3/Llama3 моделей нативно умеют ReAct.

**Plan-and-Execute:**
1. Planner LLM генерирует full plan (список шагов)
2. Executor LLM выполняет шаг за шагом, может перепланировать

**Tree of Thoughts (ToT):**
- Несколько кандидатов на каждом шаге → BFS/DFS по дереву мыслей → backtracking

**Reflexion:**
- После каждого attempt — самокритика → коррекция → retry

**Открытые фреймворки:**

| Фреймворк | URL | Stars | Особенности |
|---|---|---|---|
| **CrewAI** | https://github.com/joaomdmoura/crewAI | 51.8k | Role-based teams, sequential/hierarchical processes, Flows (event-driven) |
| **AutoGen** | https://github.com/microsoft/autogen | 58.2k (maintenance) | AgentChat, group chat, distributed runtime |
| **LangGraph** | https://github.com/langchain-ai/langgraph | 32.5k | State graphs (nodes/edges/checkpoints), human-in-the-loop, durable execution |
| **OpenAI Agents SDK** | https://github.com/openai/openai-agents-python | 26.5k | Agents, Handoffs, Tools, Guardrails, Sessions, Tracing, Realtime (voice) |
| **Claude Agent SDK** | https://github.com/anthropics/claude-agent-sdk-python | 7k | `query()`, `ClaudeSDKClient`, in-process SDK MCP servers, Hooks |
| **Microsoft Agent Framework** | https://github.com/microsoft/agent-framework | n/a | Преемник AutoGen, рекомендован MS |
| **Semantic Kernel** | https://github.com/microsoft/semantic-kernel | n/a | Planner + Skills + Connectors |

**Ключевые идеи для PromeServe:**
- **Handoffs** (OpenAI Agents SDK) — агент A передаёт control агенту B. Реализуется как special tool `handoff(agent_name)`.
- **Guardrails** (OpenAI) — pre/post-validators, можно блокировать unsafe outputs.
- **Hooks** (Claude SDK) — pre/post-tool-call interceptors. Идеально для PromeServe аудита.
- **State graphs** (LangGraph) — checkpoints для long-running агентов.
- **In-process tools** (Claude SDK) — наши tools на C++ работают в-процессе, без IPC. Это уже плюс.

---

## 3. Системы плагинов

### 3.1 Open WebUI: Tools / Functions / Pipelines

Источник: https://docs.openwebui.com/features/extensibility/

**Три уровня расширений:**

**A) Tools** — функции которые LLM может вызвать (классический tool calling).
- Python файл с YAML frontmatter в docstring:
```python
"""
title: Weather Tool
author: someone
author_url: https://github.com/...
version: 1.0.0
icon_url: https://example.com/icon.svg
required_open_webui_version: 0.4.0
requirements: requests, beautifulsoup4
"""
from pydantic import BaseModel, Field

class Tools:
    class Valves(BaseModel):
        API_KEY: str = Field(default="", description="OpenWeather API key")

    def __init__(self):
        self.valves = self.Valves()

    def get_weather(self, location: str) -> str:
        """
        Получить погоду для города.
        :param location: Название города
        :return: JSON строка с погодой
        """
        # ...
        return "..."
```
- Type hints парсятся автоматически в JSON Schema → отправляются в LLM
- **Valves** — admin-настраиваемые параметры (API ключи)
- **UserValves** — per-user параметры

**B) Functions** — три подтипа:
- **Pipe** — кастомная "модель". Появляется в селекторе моделей. Pipe(body) обрабатывает запрос целиком (можно делать роутинг, ансамбли)
- **Filter** — пред/пост-процессор (`inlet`, `outlet`, `stream` методы). Модифицирует input/output. Пример: токсичность, переводы
- **Action** — UI кнопка под сообщением. При клике вызывает `action(body)` → может что-то сделать (translate, summarize, regenerate)

**C) Pipelines** — внешний Python сервер (`open-webui/pipelines` репо). Out-of-process plugin host. Решает: тяжёлые depencies (torch, transformers) которые не нужны в основном процессе.

**Установка для пользователя:** admin загружает Python файл через UI / paste код. Сервер парсит frontmatter, `pip install` requirements в venv, импортирует через `importlib`.

### 3.2 LibreChat plugins / Agents

- **Custom Endpoints** — JSON-конфиг для любого OpenAI-compat API
- **Agents** — определяются в UI: tools (web search, code interpreter, image gen), files (RAG), instructions, model
- **MCP** — `librechat.yaml` секция `mcpServers` (stdio + SSE)
- **Skills** — markdown инструкции, прикреплённые к агенту
- **Subagents** — агент может вызвать другого агента как tool

### 3.3 Continue.dev custom commands

- `config.json` в `~/.continue/`:
```json
{
  "models": [{...}],
  "contextProviders": [
    {"name": "file"}, {"name": "codebase"}, {"name": "url"}
  ],
  "slashCommands": [
    {"name": "test", "description": "Generate tests", "prompt": "..."}
  ],
  "mcpServers": {...}
}
```
- **Context providers** — программируемые `@`-источники контекста
- **Skills** в `.continue/skills/*.md` — markdown с YAML frontmatter (как Claude Code skills)

### 3.4 Архитектурный выбор для PromeServe

Рекомендация: **multi-tier plugin system**, разные виды плагинов под разные задачи.

**Tier 1: Built-in C++ tools** (уже частично есть в `tool_call.h`)
- Скомпилированы внутрь PromeServe
- Zero overhead, type-safe
- Примеры: `write_file`, `read_file`, `http_get`, наши custom inference helpers

**Tier 2: MCP клиенты** (для всего community-экосистемы)
- subprocess/HTTP MCP-серверы
- Конфиг через JSON
- Получаем filesystem, github, slack, brave-search, playwright бесплатно
- Изоляция (subprocess) — security plus

**Tier 3: WASM-плагины** (для performance-критичных custom tools)
- `wasmtime` C API или `wasmer-c-api`
- Plugin file: `.wasm`
- Capability-based security через WASI
- Zero subprocess overhead, but C++ ABI стабильность
- Используется в Envoy, Spin, Wasmtime — production-grade

**Tier 4: Python-плагины через embedded Python**
- pybind11 уже есть в PromeTorch
- Plugin file: `.py` с frontmatter а-ля Open WebUI
- Тяжёлый: 50 MB+ Python runtime, GIL
- Только если нужны Python ML библиотеки
- Изоляция слабая (один интерпретатор на процесс)

**Tier 5: subprocess JSON-RPC** (для legacy tools)
- Любой язык, любой бинарник
- Slowest, but most flexible
- Это уже MCP по сути (tier 2 покрывает)

**Финальная рекомендация:** Tier 1 (для core) + Tier 2 (MCP клиент — для экосистемы) — это 90% value. Tier 3 (WASM) добавить через 6-12 месяцев когда появится спрос на user plugins. Python embed — **избегать** для PromeServe (наш сервер должен оставаться lean C++).

---

## 4. Modes — Code / Deep Research / Computer Use / Multi-agent

### 4.1 Code mode

Что делают топ-чаты:
- **Auto-detect** language по тройным backticks и/или AST sniff
- **Syntax highlight** — Shiki (Highlight.js, Prism — устаревают, Shiki выигрывает за счёт VS Code grammar и themes)
- **Copy button** — clipboard API
- **Run sandbox**:
  - **Python** — Pyodide в браузере (WASM, ~6 MB), или E2B/Riza/Daytona в облаке
  - **JavaScript** — `eval` в iframe (опасно) или Sandpack (Codesandbox)
  - **HTML** — `<iframe srcdoc="..." sandbox="allow-scripts">` (LibreChat artifacts)
  - **Mermaid** — `mermaid.js` рендерит SVG
  - **React/Vue** — Sandpack или WebContainers
- **Code Interpreter** (OpenAI ChatGPT, LibreChat) — Docker-sandboxed Python + библиотеки (pandas, matplotlib)

Open-source референсы:
- **Sandpack** — https://github.com/codesandbox/sandpack — best React/Vue/Vanilla preview
- **E2B SDK** — https://github.com/e2b-dev/E2B — secure cloud sandbox API
- **Riza** — https://riza.io — alternatives для cloud code execution
- **Pyodide** — https://pyodide.org — Python in WASM, offline

Что брать в PromeServe:
- Mermaid + Shiki сразу
- HTML/JS artifacts через sandboxed iframe — copy-paste из LibreChat
- Code Interpreter — отложить, пока модели не сильные в Python execution loops

### 4.2 Deep Research mode

**Что это:** агент делает множество итераций web search → анализ → новые поисковые запросы → синтез → отчёт ≥2000 слов с цитатами.

**Коммерческие:** OpenAI Deep Research (в ChatGPT Pro), Gemini Deep Research, Perplexity Deep Research, Anthropic Research (в Claude).

**Open-source:**

| Проект | URL | Stars | Архитектура |
|---|---|---|---|
| **GPT-Researcher** | https://github.com/assafelovic/gpt-researcher | 27.2k | Planner agents + Execution agents + Publisher. Tree-like recursive exploration. ~5 мин/запрос, ~$0.40. Reports 2000+ слов с цитатами. Apache 2.0 |
| **Open Deep Research** (LangChain) | https://github.com/langchain-ai/open_deep_research | 11.4k | Supervisor-Researcher pattern. Parallel sub-агенты с isolated context windows. MCP-compatible search tools. MIT |
| **STORM** (Stanford) | https://github.com/stanford-oval/storm | n/a | Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking. Вдохновение для GPT-Researcher |
| **Microsoft Magentic-One** | https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one | n/a | Multi-agent research, web browsing |

**Архитектурный паттерн (single best summary):**
```
Coordinator/Supervisor LLM:
  - получает query
  - planner: split на 3-5 sub-topics
  - spawn researcher агентов (parallel) с isolated context

Researcher агент (loop, 5-15 итераций):
  - web_search(sub_topic) → top N результатов
  - для каждого: fetch + extract relevant
  - reflect: enough? missing aspects?
  - if not enough: gen новые запросы

Writer/Publisher LLM:
  - получает все findings
  - generates outline
  - пишет sections с citations
  - publishes (PDF/MD/Word)
```

**Реализация в PromeServe:** новый endpoint `POST /api/research`:
- параметры: `query`, `depth` (1-5), `breadth` (3-10), `max_minutes`
- использует наш MCP-клиент к `brave-search` или `tavily` или `exa`
- использует нашу же модель для reasoning
- streaming progress events через SSE (текущая фаза + найденные источники)

### 4.3 Computer Use / Operator

**Claude Computer Use** (https://www.anthropic.com/news/3-5-models-and-computer-use):
- Скриншот → LLM → mouse_move/click/type/scroll команды → новый скриншот → loop
- Tools: `computer` (mouse + keyboard + screenshot), `bash`, `text_editor`
- Доступно через API: `tools: [{type: "computer_20250124", name: "computer", display_width_px, display_height_px, display_number}]`
- OSWorld benchmark: 22% (Claude 3.5 Sonnet, лучше всех)
- Демо: https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo

**OpenAI Operator** (теперь интегрирован в ChatGPT):
- Browser-based control, не desktop
- VLM смотрит на скриншот, выдаёт coordinates

**Open-source аналоги:**
- **OpenAdapt** — https://github.com/OpenAdaptAI/OpenAdapt — record + replay GUI actions
- **Anthropic Quickstarts Computer Use Demo** — Docker контейнер с VNC + reference loop
- **Self-Operating Computer** — https://github.com/OthersideAI/self-operating-computer — multi-modal GUI agent (~10k stars)
- **Browser Use** — https://github.com/browser-use/browser-use — playwright + DOM (не скриншот). 50k+ stars.

Что брать в PromeServe:
- **Browser Use паттерн** (DOM а не скриншоты) — намного дешевле и работает с маленькими моделями
- Можно через Playwright MCP server (см. раздел 2.3)
- Полноценный Computer Use — отложить, требует сильную VLM

### 4.4 Multi-agent — делегация

Уже разобрали в 2.5. Основные паттерны:

- **Handoffs / Subagents** — модель A передаёт control модели B через специальный tool
- **Group chat** (AutoGen) — несколько агентов общаются в общем чате
- **Hierarchical** (CrewAI process="hierarchical") — manager агент назначает задачи workers
- **Parallel research** (Open Deep Research) — fan-out сабтопиков

Для PromeServe:
- Реализуем handoff как built-in tool: `handoff(target_model: str, prompt: str)` → переключение модели в `model_manager.h`
- Поддержка multiple моделей одновременно — у нас уже есть в `model_manager.h`, нужно расширить для concurrent inference (TP-4 + другой модели на других ядрах одновременно — это требует careful resource control)

---

## 5. Anchor links / навигация в чате

### 5.1 Что делают топ-чаты

| Feature | ChatGPT | Claude.ai | Open WebUI | LibreChat | NextChat |
|---|---|---|---|---|---|
| Ctrl+F поиск по чату | да (browser) | да (in-app) | да | да | да |
| Поиск по всем чатам | да | да | да | да | базовое |
| Якорные ссылки на сообщения | (`#msg-...`) Claude — да | да | через ID | да | нет |
| Table of contents (sidebar) | ChatGPT — нет | да (по headings) | да (для длинных) | да | нет |
| Bookmark/Pin | да | да | да | да | да |
| Export markdown | да | да | да | да | да |
| Export PDF | через печать | да | да | да | да |
| Share via link | да | да (artifacts only) | да | да | да |
| Branching (alternate replies) | да | да (edit + retry) | да | да | да |
| Edit prior message + regenerate | да | да | да | да | да |

### 5.2 Конкретные реализации

**Anchor links на сообщения:**
- Каждое сообщение — `<div id="msg-{uuid}">`
- URL — `/chat/{conv_id}#msg-{uuid}` (LibreChat именно так)
- Click на timestamp копирует ссылку (Claude.ai pattern)
- `scrollIntoView({behavior: "smooth"})` при загрузке

**Table of contents:**
- Парсим markdown ответы → собираем `h1`/`h2`/`h3`
- Sidebar справа, sticky scroll
- Click → scroll to heading
- Open WebUI делает auto-TOC для responses >500 слов

**Bookmark/Pin:**
- Кнопка на сообщении (icon star)
- Сохраняем в `conversation.bookmarks: [{msg_id, note}]`
- Sidebar — "Pinned messages"

**Search:**
- In-conversation — JS `MatchHighlight` + scroll
- Cross-conversation — backend full-text search (SQLite FTS5 / Postgres trigram / MeiliSearch)
- LibreChat использует MeiliSearch

**Export:**
- Markdown — конкатенация сообщений с `## User` / `## Assistant`
- PDF — на бэке: markdown → HTML → headless Chromium (Puppeteer) или wkhtmltopdf или WeasyPrint
- Можно на фронте: `window.print()` или `jsPDF` + `html2canvas` (хуже)

**Share via link:**
- Создаём read-only snapshot conversation в БД
- Публичный URL `/share/{snapshot_id}` (короткий slug)
- Optional password / expiry
- ChatGPT pattern: clone-to-own-account button

**Branching:**
- Сообщения хранятся как дерево, не список
- `message.parent_id` + `message.children: []`
- UI: стрелки `< 2/3 >` для переключения веток
- Claude.ai/ChatGPT pattern

### 5.3 Что брать в PromeServe (приоритет)

P1 (must-have, лёгкие):
1. `id` атрибут на сообщениях + URL hash navigation
2. Markdown export (просто JSON → MD конкатенация)
3. Ctrl+F фокус на встроенный search input (а не browser Ctrl+F)
4. Edit + regenerate

P2 (важно, средние):
5. TOC sidebar для длинных ответов
6. PDF export через headless Chromium (есть в нашем `docs/elbrus_report/` уже)
7. Bookmark/Pin
8. Branching

P3 (опционально):
9. Share-via-link (требует public auth/snapshot)
10. Cross-conversation search (требует индекс)

---

## 6. Дополнительные фичи которые встречаются у всех топ-чатов

- **System prompt editor** per-conversation (Cherry Studio, Msty, NextChat)
- **Temperature / top_p / top_k sliders** в UI
- **Character cards / Personas** (Cherry Studio 300+, Msty, AnythingLLM)
- **Conversation folders / tags** (Chatbox, Cherry Studio)
- **Auto-title generation** (всеми) — после 1-го сообщения LLM генерирует title
- **Auto-history compression / summarization** (NextChat) — для long contexts
- **Keyboard shortcuts** (Claude.ai, ChatGPT): `Cmd+K` (search), `Cmd+Shift+O` (new chat), `Esc` (stop)
- **Theming**: light/dark/system + custom CSS (Open WebUI)
- **Markdown extensions**:
  - GFM tables, task lists
  - LaTeX через KaTeX (быстрее MathJax)
  - Mermaid SVG
  - Footnotes `[^1]`
  - Citation pills `[1]` → клик показывает источник
- **Streaming с pause/resume** (Claude.ai pause button)
- **Citation footnotes inline** (Perplexity-style) — для web search results
- **Cost/token counter** в footer
- **Multi-account / SSO** (LibreChat OAuth/LDAP, Open WebUI users)
- **Rate limiting visible** (количество запросов left)
- **Model badge** — какая модель отвечала на каждое сообщение (для multi-model conversations)
- **Voice cloning preview** (Cherry Studio готовит)
- **Drag-drop reorder** prompts/files

---

## 7. Финальные рекомендации для PromeServe

### 7.1 Топ-3 must-have на ближайший спринт (1-2 недели)

1. **MCP-клиент + 5 базовых MCP-серверов** (`filesystem`, `fetch`, `brave-search`, `git`, `github`). Это даёт PromeServe сразу армию готовых тулов без написания собственного кода. Хедер `promeserve/mcp_client.h` + конфиг `mcp_servers.json` + stdio subprocess wrapper. ~800 LoC C++.
2. **Streaming SSE для `/api/chat`** с tool-call deltas в формате Ollama-совместимом + OpenAI-совместимом параллельно. Текущий `tool_call.h` блокирующий — нужен event-driven парсер на streaming output. ~400 LoC.
3. **Code Artifacts MVP** (HTML+Mermaid+SVG) в `promeserve/web/index.html`. Sandbox iframe, post-message API для обновлений, save/share через хост. ~600 LoC JS, копируется из LibreChat по сути.

### 7.2 Топ-5 nice-to-have (следующие 2-4 недели)

4. **MCP-сервер** (PromeServe экспонирует наш inference + tools через Streamable HTTP) — Claude Desktop / Cursor / VS Code смогут подключаться к нашему PromeServe как к источнику моделей.
5. **Workspaces / Knowledge Stacks** — изолированные RAG + tool sets per project. Реюз нашего GGUF loader для embeddings.
6. **Deep Research mode** — supervisor+researcher паттерн через наш MCP (используем brave-search/exa). `POST /api/research`.
7. **Split chat** — две модели одновременно (qwen3-1.7B fast vs qwen3-14B quality). Реюз нашего multi-model `model_manager.h`.
8. **TOC + anchor links + markdown export + PDF**. Anchor + TOC — 1 день фронта; PDF — headless Chrome subprocess.

### 7.3 Архитектурные решения

**Frontend stack:**
- **Текущая ситуация:** `promeserve/web/index.html` — single-file vanilla HTML 39 KB.
- **Рекомендация:** не уходить в полный React/Svelte SPA. Промежуточный шаг — **Preact + HTM + Vite** или **Alpine.js**, чтобы остаться "single binary + static files" deploy. ~30 KB framework, всё что нужно: components, reactive state, no build complexity.
- **Альтернатива (если команда хочет TS):** **Svelte 4 + Vite** — как Open WebUI и HF Chat-UI. Минимум boilerplate, статическая сборка, отличный output bundle size.
- **Не брать:** Next.js (overkill для embedded UI), Electron (мы — server, не desktop), Tauri (та же причина).

**Plugin system:**
- **Tier 1 (core)**: C++ built-in tools (текущий `tool_call.h`)
- **Tier 2 (ecosystem)**: MCP клиент к stdio + Streamable HTTP серверам — **главное решение**
- **Tier 3 (future)**: WASM плагины через `wasmtime-c-api` — когда появится спрос
- **Не делать**: embedded Python, subprocess-Python plugin host — overhead + complexity

**Tool-call формат:**
- Внутренний format — нормализованный (`ToolCall { id, name, arguments }`)
- Адаптеры на input (parse) и output (render prompt) для OpenAI, Anthropic, Hermes/Qwen, Llama3, Mistral
- Streaming парсер с state machine (детекция тегов в token stream без буферизации всего output)

**MCP роль PromeServe:**
- Одновременно **клиент** (подключается к community MCP-серверам) и **сервер** (экспонирует наши модели/tools для Claude Desktop, Cursor, VS Code).
- Это превращает PromeServe в bridge: внешние агенты вызывают наши локальные модели на Эльбрусе/NMCard через MCP, и наши агенты получают доступ к community ecosystem.

**Conversation storage:**
- SQLite FTS5 для cross-conversation search
- Сообщения как дерево (parent_id) для branching
- Snapshot для share-links (immutable copy)

**Streaming:**
- SSE (Server-Sent Events) — обязательно, без него UX неприемлемый
- WebSocket — только если будем делать collaborative editing или voice (для bidi audio chunks)
- HTTP/2 + chunked — fallback

**Security:**
- MCP клиенты — каждый в subprocess, кап-based filesystem доступ через `filesystem` MCP с whitelist путей
- Tool sandbox — `bash` tool с whitelist команд (уже есть), для full shell — Docker контейнер через DAP-like protocol
- WASM tier — capability-based (WASI)
- HTML artifacts — `sandbox="allow-scripts"` без `allow-same-origin` (рекомендация LibreChat)

### 7.4 Roadmap

| Sprint | Цель | Главные deliverables |
|---|---|---|
| S1 (1-2 нед) | MCP client + streaming + artifacts MVP | Подключение к 5 серверам, SSE, HTML preview |
| S2 (3-4 нед) | MCP server + Workspaces | Claude Desktop может подключиться к PromeServe |
| S3 (5-6 нед) | Deep Research + Split chat | Multi-step research через наши модели |
| S4 (7-8 нед) | Frontend rewrite (Svelte/Preact) | Полноценный UI вместо single-file |
| S5 (9-10 нед) | Plugin system Tier 3 (WASM) | User-installable plugins |
| S6 (11-12 нед) | Voice (Whisper + Piper) | Voice mode end-to-end |

---

## Источники (selected)

- [Open WebUI repo](https://github.com/open-webui/open-webui)
- [LibreChat repo](https://github.com/danny-avila/LibreChat)
- [Chatbox repo](https://github.com/Bin-Huang/chatbox)
- [NextChat repo](https://github.com/ChatGPTNextWeb/NextChat)
- [HuggingFace Chat-UI](https://github.com/huggingface/chat-ui)
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)
- [Continue.dev](https://github.com/continuedev/continue)
- [Cherry Studio](https://github.com/CherryHQ/cherry-studio)
- [Jan](https://github.com/janhq/jan)
- [Msty](https://msty.ai/)
- [MCP introduction](https://modelcontextprotocol.io/introduction)
- [MCP architecture](https://modelcontextprotocol.io/docs/concepts/architecture)
- [MCP official servers](https://github.com/modelcontextprotocol/servers)
- [punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)
- [wong2/awesome-mcp-servers](https://github.com/wong2/awesome-mcp-servers)
- [best-of-mcp-servers (ранкинг)](https://github.com/tolkonepiu/best-of-mcp-servers)
- [Context7 (docs MCP)](https://github.com/upstash/context7)
- [Playwright MCP](https://github.com/microsoft/playwright-mcp)
- [GitHub MCP server](https://github.com/github/github-mcp-server)
- [Exa MCP](https://github.com/exa-labs/exa-mcp-server)
- [Anthropic Tool Use docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AutoGen](https://github.com/microsoft/autogen)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- [GPT-Researcher](https://github.com/assafelovic/gpt-researcher)
- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
- [Browser Use](https://github.com/browser-use/browser-use)
- [Sandpack](https://github.com/codesandbox/sandpack)
- [E2B](https://github.com/e2b-dev/E2B)
- [Pyodide](https://pyodide.org)
- [Claude Computer Use announcement](https://www.anthropic.com/news/3-5-models-and-computer-use)
- [Anthropic Computer Use demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)
- [Self-Operating Computer](https://github.com/OthersideAI/self-operating-computer)
- [OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt)
- [STORM (Stanford)](https://github.com/stanford-oval/storm)
- [Microsoft Magentic-One](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
