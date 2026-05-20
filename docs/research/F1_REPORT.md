# F1 — Frontend Pass 1 отчёт

**Дата:** 2026-05-20
**Цель:** модернизация `promeserve/web/index.html` до уровня Open WebUI / LibreChat в single-file vanilla JS.
**Базовые отчёты:** `docs/research/AI_CHAT_FEATURES_R1.md`, `docs/research/AGENT_STACK_R2.md`.

---

## Реализовано (10/10 из чек-листа F1)

| № | Фича | Статус | Комментарий |
|---|------|--------|-------------|
| 1 | **Modern chat UI + markdown + code blocks** | DONE | `marked.js` (CDN) + `highlight.js` + DOMPurify санитайз. Copy-кнопка на каждый блок. Tables, blockquotes, headings, lists. |
| 2 | **SSE / NDJSON streaming + cursor + Stop** | DONE | Поддерживается оба формата: NDJSON (Ollama) и SSE (`data: ...\n`). Курсор-блик, AbortController на Stop. Throttle highlight каждые 200 символов. |
| 3 | **Settings panel (правая, collapsible)** | DONE | System prompt textarea, temperature/top_p/top_k/max_tokens slider'ы, model selector в топбаре, reset button. Persistence в localStorage. |
| 4 | **Chat history sidebar** | DONE | localStorage. New / rename / delete / export JSON. Search-фильтр по titles + по содержимому сообщений. |
| 5 | **TOC + якоря на сообщения** | DONE | Правая панель → вкладка "Оглавление". Клик → smooth-scroll + highlight 1.5s. Каждое сообщение имеет `id="msg-<uid>"`. |
| 6 | **Tool-call rendering** | DONE | Парсер `<tool_call>{...}</tool_call>` и `<tool_response>{...}</tool_response>`. Expandable card с argument JSON pretty-print + result секцией. Совместимо с `promeserve/tool_call.h` форматом Hermes. |
| 7 | **File attachments + drag-and-drop** | DONE | Кнопка "+", drag-drop в input box (визуальный `.dragover` highlight). Text-файлы инлайнятся в сообщение как fenced code block (limit 12K). Бинарники — chip с размером и mime. |
| 8 | **Voice input (Web Speech API)** | DONE | Кнопка микрофона видна только при наличии `webkitSpeechRecognition`. Pulse-анимация при записи. Auto-язык по `state.lang`. |
| 9 | **Export current chat** | DONE | Toolbar: "Copy as Markdown" + "Download .md". Per-chat экспорт JSON в sidebar. |
| 10 | **Bookmarks / pins** | DONE | Звёздочка на каждом сообщении → пинит в `m.pinned`. Pinned messages показываются в отдельной секции sidebar. Чаты тоже можно закреплять. |

**Бонусы вне чек-листа:**

- Светлая / тёмная тема (default тёмная) с переключателем в footer'е sidebar.
- Переключатель языка RU/EN с полной I18N таблицей.
- Welcome screen с 4 предложенными промптами.
- Regenerate-кнопка на user-сообщениях — обрезает историю и переотправляет.
- Toast-уведомления (copy/export/reset).
- Hotkey `Ctrl+F` (поиск), `Esc` (закрыть модалки).
- Search modal с подсветкой найденного через `<mark>`.
- Rename modal с `<input>` фокусом.

---

## Размеры и зависимости

- **Was:** `promeserve/web/index.html` = **38 972 байт** (~39 KB, 770 lines).
- **Became:** `promeserve/web/index.html` = **91 417 байт** (~89 KB, 2034 lines) — single file.
- **Backup:** `promeserve/web/index.html.backup-F1` (оригинал сохранён).
- **Snapshot:** `promeserve/web/preview/F1-snapshot.html` (~28 KB) — статический mock с 4 сообщениями, code block, 2× tool_call card, attachments, settings panel. Открывается напрямую в браузере без сервера.
- **Screenshot:** `promeserve/web/preview/F1-screenshot.png` (155 KB, 1600×1200) — рендер через headless Edge.

### CDN зависимости (без npm)

| Библиотека | Использование |
|-----------|----------------|
| `cdnjs/highlight.js@11.9.0` + atom-one-dark / atom-one-light styles | Подсветка синтаксиса в `<pre><code>` |
| `cdnjs/marked@12.0.2` | Markdown → HTML |
| `cdnjs/dompurify@3.0.11` | Sanitize HTML before injecting в `innerHTML` (XSS guard) |

Никаких npm/build-step'ов, никаких frameworks (React/Vue/Svelte/Preact). Pure vanilla.

---

## Известные проблемы / TODO для F2

1. **Multimodal images.** Сейчас drag-drop изображений сохраняет ATT, но base64 в API не отправляется. F2: добавить `images: []` в payload Ollama, рендерить `<img>` в ответе.
2. **Code Artifacts (sandboxed iframe).** Inline-preview HTML/SVG/Mermaid из ответа модели — пока показывается только как code-block. F2: добавить `<iframe srcdoc>` с CSP sandbox.
3. **MCP клиент.** Все из R2 — не подключен. F2: подключиться к MCP-серверам через JSON-RPC (Streamable HTTP).
4. **Split chat / side-by-side.** R1 рекомендует — пока не сделано.
5. **Web search / RAG.** R1 рекомендует — backend не готов; UI hook'и можно сделать в F3.
6. **PDF / DOCX attachments.** Сейчас читается только text-like. F2: pdf.js для PDF preview.
7. **TTS output.** Только STT через Web Speech. F2: добавить speech-synthesis или server-side Piper/Kitten.
8. **Markdown сообщений: streaming-edge cases.** При генерации внутри code-block markdown может временно ломаться (например незакрытый ```). На финал-рендере это починено, но в процессе видны артефакты. F2: использовать incremental markdown parser или streaming-friendly preprocess.
9. **Mobile right-panel UX.** На <1100px пока работает как drawer, но не интуитивно — F2 редизайн.
10. **Авто-title.** Сейчас просто substring 40 — F2: использовать модель для генерации заголовка.
11. **Voice TTS playback.** R1 → Piper / KittenTTS. Только трогает back-end.
12. **Hermes vs Llama3 vs Mistral tool-call формат разделение.** Сейчас рендерим только `<tool_call>` (Hermes/Qwen3). F2: добавить `<|python_tag|>` и `[TOOL_CALLS]`.

---

## Скриншот (preview)

`promeserve/web/preview/F1-screenshot.png` — 1600×1200, dark theme:
- Left sidebar с pinned section + 5 chats + footer (theme/lang).
- Topbar: model selector "qwen3:4b (TP-4)", chat title, action buttons (search/export/copy/settings).
- Chat feed: user attachment chip, markdown headings, ordered/unordered lists, syntax-highlighted C code, tool_call card (arguments), tool_response card (output, success-styled).
- Right panel: settings tab с System Prompt, sliders.

---

## Структура файлов после F1

```
promeserve/web/
├── index.html                      # 91 KB — основной UI (production)
├── index.html.backup-F1            # 39 KB — оригинал до F1 (для сравнения)
└── preview/
    ├── F1-snapshot.html            # 28 KB — статический demo без backend
    └── F1-screenshot.png           # 155 KB — headless render
```

Никакие файлы за пределами `promeserve/web/` не тронуты. C++ backend (`api_handlers.h`, `tool_call.h` и т.д.) не изменён.

---

**F1 — закрыт. F2 ждёт приоритизации (multimodal? MCP клиент? artifacts?).**
