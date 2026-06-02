# Аудит #20 — PromeServe frontend state

- **Дата:** 2026-06-02
- **HEAD:** `85c0fb5`
- **Объект:** `promeserve/web/` (`index.html` 3752 строки / 180 KB, `mcp.js` 4.8 KB, `preview/F1-F2 snapshots`)
- **Цель:** проверить синхронизацию F2-фронта с backend (commits `c78c9ef` / `0428eeb` / `a7458b5` / `7d48408`)

---

## 1. Структурная проверка index.html

| Проверка | Результат |
|---|---|
| DOCTYPE/html/head/body/script wrappers | OK (строки 1, 2, 3, 1288, 3749-3752) |
| Завершение IIFE `App = (() => { ... })()` | OK (строка 3749) |
| CDN deps (marked, highlight.js, DOMPurify) | подключены, но **DOMPurify ни разу не вызывается** (grep `DOMPurify`/`purify` → 0 matches в JS) |
| Inline script syntax | parsing OK (нет `<\/script>` ошибок, все шаблон-строки escape'нуты корректно) |
| Двойной IIFE / duplicate `App` | один (грамотный wrapper, F2 расширил в том же scope) |

---

## 2. localStorage — ключи и коллизии

11 ключей, единый префикс `pt_*`: `pt_chats`, `pt_active_chat`, `pt_model`, `pt_settings`, `pt_theme`, `pt_lang`, `pt_workspaces`, `pt_active_workspace`, `pt_mcp_servers`, `pt_perf_history`, `pt_auth_token`. Коллизий нет.

**Замечание:** `LS_KEY_AUTH` объявлен (1319) — но в коде **никогда не читается и не пишется** (grep → 0 usage). Auth banner показывается, но токена нет в payload.

---

## 3. Таблица features

| # | feature | claimed_in_F2_report | implemented_in_index.html | wired_to_backend | works_end_to_end | rec_action |
|---|---|---|---|---|---|---|
| 1 | `/api/chat` streaming NDJSON | yes | yes (line 2601, NDJSON+SSE fallback parsing 2644-2682) | **yes** (backend `handle_chat` строит NDJSON, формат `obj.message.content`/`obj.done`/`obj.eval_count` совпал) | yes | — |
| 2 | `/api/generate` | yes (slash + общий API) | **нет — НЕ используется фронтом** (grep `/api/generate` → 0 hits) | n/a | n/a | удалить из F2 claim либо подключить для one-shot `/run`-ответов |
| 3 | `/api/tags` model picker | yes | yes (line 2887, авто-refresh каждые 30 s, line 3691) | yes | yes | — |
| 4 | `/api/show` model info | в плане | **нет** (grep → 0) | n/a | n/a | не критично |
| 5 | `/api/mcp/tools` | yes (a7458b5) | yes via `PromeMcp.listTools()` в `mcp.js`, читается в `openMcpModal` 3073, init 3699 | yes | partial — tools возвращаются только при открытии modal | OK для MVP |
| 6 | `/api/mcp/servers` | yes | yes (3074, 3696) | yes | **частично** — мерж backend↔localStorage идёт по `s.name`, дубликаты возможны после rename | дедупликация |
| 7 | `/api/mcp/call` (slash `/run`) | yes | yes (3324) — но **не передаёт `serverId`**, только tool name + args | yes | yes для single MCP server; для двух серверов с одноимённым tool — undefined behavior | quick-fix: передавать `server` префикс |
| 8 | `/api/mcp/reconnect` / `/api/mcp/audit` | backend endpoint есть | **нет UI кнопки/вызова** ни в одном месте | n/a | no | добавить "Reconnect"+"Audit log" в MCP modal |
| 9 | Per-server tool checkboxes filtering | yes — "фильтруют какие tools доступны" | UI рендерится (3134-3141), `disabledTools` персистится | **НЕТ** — `disabledTools` нигде не читается при формировании payload | **false claim**: чекбокс это UI-only state | wire в send_payload как `payload.tools = [...]` фильтр |
| 10 | Status dots на серверах | yes — "реальный state" | dot класс берётся из `s.enabled` (3056, 3122) — это **локальный toggle** | **НЕТ** — backend поле `available`/`status` мапится один раз в `openMcpModal` (3086), но dot класс по-прежнему завязан на `s.enabled`, а не на `s.available`/`status` | **hardcoded-like (UI bool)** | dot класс от `s.status==='ready'` |
| 11 | MCP presets (6 шт) | yes | yes (1356-1407) — но **никогда не записывает `~/.promeserve/mcp.json`**, "Добавить" только в state | partial — backend читает свой файл сам | UI-only | добавить POST `/api/mcp/install_preset` |
| 12 | Image multimodal в payload (Ollama base64) | yes | yes (2549-2553) — strip prefix корректно | **yes** (Ollama-format), но **backend `handle_chat` НЕ парсит `images[]`** в messages (grep `images` в api_handlers.h → 0 hits) | **no** — отправляется, игнорируется сервером | задокументировать ограничение |
| 13 | Audio `<audio controls>` | yes | yes (1907) FileReader+dataURL (2358) | n/a (только локальный preview) | preview-only | OK |
| 14 | PDF chips в payload | yes | chip рендерится (1903, 2378-2383), `dataUrl` есть | **НЕТ** — в payload (2549) PDF не добавляется (только `image/`) | UI preview only | quick-fix: либо отправлять как text-extract, либо ясно показать badge "preview only" |
| 15 | Branching tree "Branch from here" | yes | yes — кнопка на каждом message (1945-1948), `branchFromMessage` 3491-3514, `renderBranchTree` 3522-3543 | n/a (чисто клиентский feature) | yes | — |
| 16 | Workspaces (multi) | yes | yes (loadWorkspaces 2917, renderWorkspaceSelector, system prompt per WS 2540) | n/a | yes | — |
| 17 | Slash commands (15 шт) | yes | yes (1322-1338, `checkSlash`, `executeSlashCommand` 3273-3349) | `/run` → backend; `/model` → tags cache; остальные локальные | yes | — |
| 18 | Hotkeys overlay + 11 биндингов | yes | overlay рендерится (3581-3589); listeners (2791-2828) | **только 5 биндингов из 11 реально работают**: Ctrl+F, Ctrl+/, Ctrl+Shift+P, Ctrl+Shift+M, Ctrl+Shift+N, Esc, Enter (handleEnter), Tab/↑/↓ (slash). Биндинги "/" и "Shift+Enter" не нужны как listeners (это input-level) — claim не ложный, но overlay вводит в заблуждение, нумеруя их как 11 hotkeys | partial cosmetic | в overlay пометить "input-level" |
| 19 | Artifacts panel (HTML/SVG/Mermaid/React/JSON sandbox) | yes | yes (3372-3488) — iframe sandbox, Mermaid+React через CDN | n/a | yes | — |
| 20 | Tables CSV copy | yes | yes (1989-2014, table-wrapper + кнопка) | n/a | yes | — |
| 21 | Performance Dashboard | yes | yes (3592-3647), perf history persists в LS, шкала localStorage usage | n/a | yes | — |
| 22 | SSE/NDJSON reconnect (exponential backoff) | yes | yes — retry 5xx + network (2596-2632), `fetchWithRetry` для GET (2843-2867) | yes | yes — но **внутри открытого stream** при потере соединения retry НЕ происходит (только при initial fetch); реконнектится только при следующей отправке | claim "SSE reconnect" — частично |
| 23 | Auth banner (401/403) | yes | yes (`showAuthBanner` 2869, `#authBanner` ID) | **partial** — banner показывается, но **никакой UI для ввода токена нет**, `LS_KEY_AUTH` мёртв, нет `Authorization` header в fetch'ах | broken end-to-end | либо удалить banner, либо добавить input + header |
| 24 | Light/dark themes | yes | dark в `:root` (20-50), light в `html[data-theme="light"]` (52-66) | n/a | yes — но light **не переопределяет** `--accent`, `--accent-hover`, `--success`, `--danger`, `--warning`, `--info`; накладываются с dark, на ярком фоне останутся приемлемыми (контрастные), визуально OK | minor: cosmetics |
| 25 | F2-snapshot.html / F2-screenshot.png в repo | yes | **есть**: `web/preview/F2-snapshot.html` (37 KB, 564 строки) и `F2-screenshot.png` (176 KB) | n/a | snapshot НЕ соответствует актуальному index.html: 564 vs 3752 строк (~6.6×). Скорее всего snapshot — это «частичный DOM serialize» а не источник кода. Скриншот — статичный | задокументировать что это иллюстрации, а не source-of-truth |
| 26 | DOMPurify вызывается на model output | implied (CDN подключён) | **НЕТ** — `marked()` рендерит markdown без санитайзера | partial XSS-risk (markdown allows HTML by default в marked) | wire `DOMPurify.sanitize(marked.parse(text))` |

---

## 4. Краткое резюме (≤300 слов)

Frontend архитектурно цел: 3752-строчный single-file без syntax-ошибок, IIFE-wrapper закрыт, единый префикс `pt_*` в localStorage (11 ключей, коллизий нет), CDN deps подключены. Backend wiring (`/api/chat`, `/api/tags`, `/api/mcp/*`) реально работает через `fetch`+NDJSON-парсер с fallback на SSE-формат — формат ответа backend (`obj.message.content`/`obj.eval_count`/`obj.done`) совпадает с тем, что фронт парсит.

**Найдено 7 расхождений между F2-claim и реальностью:**

1. **Per-server tool checkboxes — UI-only.** `disabledTools` персистится, но при формировании `payload` (line 2557) `tools` array вообще не строится. Backend `tools_mode` (handle_chat_with_tools) никогда не активируется из этого UI.
2. **Status dots — псевдо-realtime.** Класс зависит от `s.enabled` (локальный toggle), а не от backend-поля `available`/`status`, которое мапится один раз при open.
3. **Image multimodal — отправляется, игнорируется.** `payload.images[]` корректно базируется по Ollama-формату, но `api_handlers.h` не парсит это поле (grep `images` → 0 hits).
4. **PDF chips — НЕ в payload** (фильтр в 2549 берёт только `image/`).
5. **Auth banner — мёртвый код.** `LS_KEY_AUTH` объявлен, не читается. Нет input для токена, нет `Authorization` header в fetch.
6. **DOMPurify подключён, но не вызывается** — XSS-риск через markdown.
7. **`/api/mcp/reconnect` и `/api/mcp/audit`** — backend есть, UI кнопок нет. `/api/generate` фронтом вообще не используется.

**Snapshot F2-snapshot.html** существует (37 KB / 564 строки) и **не соответствует** актуальному index.html (3752 строки) — это DOM-сериализация, не source.

**Production-readiness: средняя.** Error handling и retry есть, но в открытом stream нет SSE-reconnect — только при initial fetch. Light theme не переопределяет accent-палитру (косметика).

