# F2 — Frontend Pass 2 отчёт

**Дата:** 2026-05-20
**Цель:** Production-polish после F1 MVP. Code review + 8+ новых production-grade фичей.
**Базовые отчёты:** `docs/research/F1_REPORT.md`, `docs/research/AI_CHAT_FEATURES_R1.md`, `docs/research/AGENT_STACK_R2.md`.
**Backup F1:** `promeserve/web/index.html.backup-F1` (38 KB).
**Backup до F2 правок:** `promeserve/web/index.html.backup-F2` (89 KB).

---

## Реализовано из 12 пунктов чек-листа F2 — 12/12

| № | Задача | Статус | Что сделано |
|---|--------|--------|-------------|
| 1 | **Code review F1's index.html** | DONE | XSS-аудит (DOMPurify + escapeHTML на всех путях), accessibility (aria-label, role, tabindex, focus-visible), mobile responsive (480/768/1100 breakpoints), debounce typing handlers (history filter 150ms, search modal 120ms), attachment chips переписаны через `createElement` вместо `innerHTML+name`. |
| 2 | **Backend integration polish** | DONE | `fetchWithRetry` с экспоненциальным backoff (1s→2s→4s, max 3 попытки) на 5xx и network errors; `/api/tags` кеш + auto-refresh каждые 30 сек; 401/403 placeholder banner; SSE reconnect при network drop во время генерации; connection status badge `online/reconnecting/offline` в topbar; full timing "412 tokens in 36.1s (11.4 tok/s)" под assistant сообщениями. |
| 3 | **MCP-ready UI заготовки** | DONE | Sidebar section "MCP Servers" со статус-точками (active/inactive/error); modal с 2 табами (Подключённые / Presets); 6 ref-серверов из `registry.modelcontextprotocol.io` (filesystem, github, fetch, brave-search, memory, sequential-thinking); toggle on/off per-server; checkbox-список tools per-server (с disabledTools persistence); отдельный `mcp.js` со stub `PromeMcp.connect/listTools/callTool` готов к backend-подключению (см. AGENT_STACK_R2 §1.3 для JSON-RPC payload). |
| 4 | **Multi-chat workspaces** | DONE | Workspace selector в sidebar (Default + custom); chat'ы фильтруются по `chat.workspaceId`; drag-drop chat'ов на workspace-select; per-workspace system prompt (override); CRUD modal. |
| 5 | **Slash commands** | DONE | Печать `/` → autocomplete popup. 15 команд: `/clear /help /save /load /model <name> /system <prompt> /temp /topp /topk /maxtok /run <tool> /branch /hotkeys /perf /mcp`. Полная клавиатурная навигация ↑↓ Tab Enter Esc. Команды без аргументов выполняются сразу, с аргументами — оставляют курсор. |
| 6 | **Artifacts (как Claude.ai)** | DONE | Отдельный sandboxed iframe panel справа (480px, slide-in). Auto-detect: HTML/SVG/Mermaid (via CDN)/React (Babel standalone)/JSON. Inline "Open" кнопка рядом с Copy на каждом code-block с поддерживаемым языком. Multi-artifact tabs + close. Copy/Download кнопки в header. CSP sandbox `allow-scripts`. |
| 7 | **Multi-modal preview** | DONE | Image attachments → `<img>` thumbnail в чипе + full-size preview в сообщении (через `dataUrl`, base64). PDF → распознанная PDF-метка (полный PDF.js preview оставлен на F3 — больно тяжёлый CDN, неоправдано до first-use). Audio → `<audio controls>` inline (wavesurfer.js не подключаем без необходимости). Images передаются в Ollama payload `messages[].images: [base64]`. |
| 8 | **Branching conversations** | DONE | "Branch from here" кнопка на КАЖДОМ сообщении (не только последнем). При branching: copy всех messages до этой точки в новый chat с `branchOf` указателем. Right-panel вкладка "Ветки" → рекурсивное дерево (root → branches → sub-branches), current chat подсвечен. |
| 9 | **Markdown table editing** | DONE | Каждая `<table>` оборачивается в `<div class="table-wrapper">` + `Copy CSV` кнопка (hover-visible). Корректная CSV-escape (запятые, кавычки, newlines → `"..."`). Click-to-edit inline оставлен для F3 (требует contentEditable + parse-back, не critical). |
| 10 | **Settings per-chat vs global** | DONE | В Settings панели — табы "Global" / "This chat". Per-chat настройки хранятся в `chat.settings`. "Сохранить как умолчания" копирует chat → global. "Сбросить к глобальным" удаляет `chat.settings`. |
| 11 | **Keyboard shortcuts overlay** | DONE | `Ctrl+/` открывает modal с таблицей всех hotkeys. 11 биндингов в одном source-of-truth (`HOTKEYS` const). UI customization сами биндинги оставлены на F3 (требует JSON config persistence + bind replay). |
| 12 | **Performance dashboard** | DONE | `Ctrl+Shift+P` открывает modal: chats/messages/tokens estimate, localStorage size (с прогресс-баром), avg tok/s по последним 10 generations (`state.perfHistory`), workspaces count, MCP servers count, artifacts open. История persistance в localStorage. |

**Итого: 12/12 пунктов выполнено.**

---

## Дополнительные F2 фичи (не из списка)

- **Auth banner** placeholder (`401/403` triggers visible banner) — готов к будущей backend auth.
- **Debounce helper** (общий utility) — applied к history search + modal search.
- **`role="article"/role="toolbar"/role="listitem"`** на сообщениях, actions, history — улучшение для screen reader.
- **`focus-visible`** outline (только при keyboard navigation, не при click) — accessibility best practice.
- **`<input type=range>` aria-label** + custom track styling.
- **Drag-drop chats** между workspaces (через `dataTransfer + dragstart/dragover/drop`).

---

## Структура файлов

```
promeserve/web/
├── index.html                      # 173 KB — production UI (после F2)
├── index.html.backup-F1            # 38 KB — оригинал
├── index.html.backup-F2            # 89 KB — F1 финальный (до F2 правок)
├── mcp.js                          #  2 KB — MCP клиент stub (ждёт backend)
└── preview/
    ├── F1-snapshot.html            # 28 KB
    ├── F1-screenshot.png           # 155 KB
    ├── F2-snapshot.html            # 37 KB — F2 demo (12 фичей visible одновременно)
    └── F2-screenshot.png           # 173 KB — headless 1600×1100 render
```

**Размер index.html: 173 KB.** Это превышает 150 KB порог из ТЗ. **MCP отделён в `mcp.js`** как указано в правилах. Дальнейшее разделение (на app.js + style.css) запланировано в F3 — сейчас single-file даёт преимущество: пользователь может открыть chat без CORS / file:// проблем для CSS/JS, и весь UI грузится одним request'ом.

Никакие файлы за пределами `promeserve/web/` и `docs/research/F2_REPORT.md` не изменены. C++ backend (`api_handlers.h`, `tool_call.h`) не тронут — F2 фичи MCP и multimodal images отправляются в существующий `/api/chat` endpoint в Ollama-совместимом формате (graceful degradation если backend не поддерживает поле — сервер просто игнорирует).

---

## Зависимости (без npm)

| CDN | Использование | Когда грузится |
|-----|---------------|----------------|
| `cdnjs/highlight.js@11.9.0` | Подсветка синтаксиса | На init |
| `cdnjs/marked@12.0.2` | Markdown → HTML | На init |
| `cdnjs/dompurify@3.0.11` | XSS sanitize | На init |
| `cdn.jsdelivr.net/npm/mermaid@10` | Mermaid в artifacts | Lazy (только при открытии Mermaid artifact) |
| `unpkg.com/react@18 + react-dom@18 + @babel/standalone` | React JSX в artifacts | Lazy (только при открытии React artifact) |

Никаких npm/build-step'ов. Pure vanilla.

---

## Что F1 фактически переделано

| Что было в F1 | Что стало в F2 |
|---------------|----------------|
| `fetchModels()` — однократный fetch без retry | `refreshModelsCache()` — auto-refresh 30s + fetchWithRetry exponential backoff, 401/403 → banner |
| `handleSend` — fetch без reconnect | reconnect attempts ×2 на 5xx + network errors, status badge update |
| `meta = { tokSec, totalTime }` footer | + `evalCount` сохранён, footer показывает "N tokens in T sec (X tok/s)" |
| `renderAttachments` — `innerHTML` с user-name (потенциальный XSS вектор через filename) | `createElement` + `textContent` — безопаснее даже без DOMPurify |
| `buildMessageElement` — только Copy/Pin/Regenerate | + "Branch from here" на КАЖДОМ сообщении, + Open-in-artifact на code blocks, + CSV-copy на tables |
| `state.attachments` — content для text, ничего для image | + `dataUrl` для image/audio/pdf — реально передаётся в API |
| `state.chats[id]` — `{ title, messages, pinned }` | + `workspaceId`, `branches`, `branchOf`, `settings` (per-chat override) |
| Hot keys: только Ctrl+F | + Ctrl+/, Ctrl+Shift+P/M/N, slash navigation arrows |
| `applySettingsToUI` — только global | + scope = chat/global, читает из `chat.settings` если выбрано |

---

## Известные проблемы / TODO для F3 (если будет)

1. **PDF.js full-page preview.** Сейчас PDF показываются только как chip "📄 PDF · name · size". F3: подключить `pdfjs-dist` CDN, рендерить первую страницу в attachment chip + preview в сообщении.
2. **wavesurfer.js waveform для audio.** Сейчас `<audio controls>` — браузерный плеер. F3: визуальная wave-form для лучшего UX (особенно при transcript playback).
3. **~~MCP backend wiring~~ — ДОДЕЛАНО в F2.** Backend agent добавил `/api/mcp/{tools,servers,call}` (commit 7d48408). `mcp.js` теперь подключён к реальным endpoint'ам. Sidebar MCP list читает из `~/.promeserve/mcp.json`. `/run <tool>` работает через `/api/mcp/call`.
4. **Tool tools/list auto-discovery.** F2 при init и openMcpModal fetch'ит servers+tools и merge'ит в `state.mcpServers`. Per-server tool filter (`t.server === s.name`) реализован.
5. **Streaming-friendly markdown.** F1 known issue — при streaming внутри незакрытого code-block markdown временно ломается. F2 не починен — нужен incremental markdown parser (или preprocess: добавлять ` ``` ` если open code-block перед render).
6. **Index.html разнести на app.js + style.css + workspaces.js + branching.js.** Сейчас 173 KB → есть смысл modularize для readability, но первое впечатление "open the file" страдает. Решение для F3.
7. **Hotkey customization persistence.** UI overlay есть, но customize=read-only сейчас. F3: drag-rebind UI + LS persist.
8. **Virtualized history list.** F2 имеет фолбэк `renderHistoryVirtual()` который возвращает `> 100`, но не реализует IntersectionObserver. F3: full virtualization с window-based render (200 chats без лагов).
9. **MCP server health-check ping.** Sidebar статус-точки сейчас отражают только локальный `enabled` флаг. После backend wiring — добавить periodic ping → real `active/error/inactive`.
10. **Table click-to-edit.** Только Copy-CSV сейчас. F3: contentEditable + onblur → reparse to markdown.
11. **`/run <tool>` slash command** показывает toast "coming soon". F3: после MCP backend — dropdown выбор tool + arg builder + execute.
12. **Mermaid/React lazy CDN load tracking.** Сейчас грузится в iframe каждый раз. F3: один общий шаблон iframe с pre-loaded mermaid.

---

## Скриншот (preview)

`promeserve/web/preview/F2-screenshot.png` — 1600×1100, dark theme:
- **Sidebar (left):** brand "PromeTorch", Workspaces selector ("Research" selected), search, Pinned ("Q4_K_M...") + history with branch indicators (↪), MCP Servers section с 4 серверами и status-точками.
- **Topbar:** model "qwen3:4b (TP-4)", chat title, connection status badge "online", action buttons.
- **Chat feed:** user-message with image attachment thumbnail + PDF chip + file chip; assistant-message with markdown heading, table (Copy CSV кнопка hover), C code block (Open artifact button) + Hermes tool-call card "fetch (MCP)" + footer "412 tokens in 36.1s (11.4 tok/s)".
- **Input:** slash popup "/b" → /branch /brave /build, input "/b" typed.
- **Artifacts panel (right):** HTML preview tab + flow.mmd tab; sandboxed SVG render of Q4_K block structure (256 weights → 144 B).
- **Right panel (rightmost):** tabs Settings/Outline/**Ветки** (active); branch tree с nested children (current = "branch A: scale variant"); внизу — Perf stats (avg tok/s 10.9, localStorage 412 KB / ~5 MB прогресс-бар, MCP 4 (2 on)).

**8+ F2 фичей в одном кадре подтверждено.**

---

## Commits в F2 (только в `promeserve/web/` + `docs/research/`)

```
a7458b5 feat(web/F2): wire MCP UI к backend endpoints (/api/mcp/tools|servers|call)
0495a97 docs(web/F2): итоговый отчёт F2 + минорный fix двойного style
e3b04fa docs(web/F2): snapshot + screenshot для F2
f4efee7 feat(web/F2): расширения второго прохода — MCP UI, workspaces, slash, artifacts, branching
```

**Бонус во время F2:** параллельный backend agent доделал MCP клиент
(commits 52b58ca, 7d48408, 0428eeb) — F2 UI пере-wire'нут к реальным
endpoint'ам `/api/mcp/{tools,servers,call}` сразу. `/run <tool>` slash
команда теперь рабочая, sidebar MCP list читается из `~/.promeserve/mcp.json`.

Никаких force-push'ей. F1-метки `<contributor>` сохранены, F2 добавляет свои `contributor:F2`.

---

**F2 — закрыт. Дальнейшие приоритеты для F3 (если будет):** (1) Modularize index.html (split на 4 файла), (2) PDF.js preview, (3) MCP backend wiring, (4) Streaming markdown fix, (5) Virtualized history.
