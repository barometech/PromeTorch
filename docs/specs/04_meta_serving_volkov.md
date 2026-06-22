# Спецификация: serving-архитектура PromeServe уровня продакшн

**Автор:** Sergei Volkov (Staff Engineer, инфраструктура инференса)
**Дата:** 2026-06-21
**Скоуп:** serving-слой `promeserve/` + точки входа в `torch/io/gguf_model.h`. Корректность ядра инференса (kernels, dequant, RoPE) — вне скоупа, считаю их рабочими.

---

## Executive summary

PromeServe сегодня — корректный, но строго **однопоточный по вычислению** Ollama-совместимый сервер: HTTP-слой многопоточный (thread-pool + bounded queue), а вся генерация сериализована одним `generate_mutex_` поверх единственной загруженной модели с единственным KV-cache, единственным scratch-pool и единственным CUDA-graph. Для N одновременных пользователей это означает: запросы 2..N **полностью простаивают в очереди**, пока первый не закончит все свои токены. Это и есть главный разрыв с Ollama (где есть параллельные слоты `num_parallel`) и тем более с vLLM (continuous batching + paged KV). Throughput-разрыв single-stream (88 vs 188 tok/s) — отдельная история про ядро; serving-слой добавляет к нему **отсутствие масштабирования по пользователям**: при 4 клиентах эффективный throughput на пользователя падает почти в 4×, а p99 TTFT (time-to-first-token) деградирует катастрофически из-за head-of-line blocking.

**Главные узкие места serving-слоя (по убыванию важности):**
1. `generate_mutex_` сериализует ВСЕ запросы — нет ни батчинга, ни параллельных слотов.
2. Единый глобальный KV-cache на модель (`model->kv_cache`) — `reset()` на каждый запрос; два запроса физически не могут сосуществовать.
3. CUDA-graph + scratch-pool + device-указатели (`d_past_len_`, `d_token_id_`) забейканы на один контекст — захардкоженный single-stream.
4. Нет prefill/decode-планировщика: длинный prefill одного запроса блокирует декод всех.
5. Нет отмены по разрыву соединения, нет backpressure по реальной нагрузке (только queue-depth по числу сокетов), нет метрик.
6. Одна модель в памяти: смена модели = полная перезагрузка, выбивает всех.

**Дорожная карта (реалистично, пошагово):**
- **Фаза 0** (дни): метрики + per-request cancellation + честный `num_parallel`-семафор вместо глухого мьютекса.
- **Фаза 1** (1–2 недели): per-request KV-cache (вынести KV из модели в request-context) → N независимых последовательных слотов без батчинга. Это уже даёт Ollama-паритет по многопользовательности.
- **Фаза 2** (3–4 недели): continuous batching декода — единый scheduler-поток, batched GEMV→GEMM по активным слотам, paged KV. Это путь к vLLM-классу throughput.
- **Фаза 3** (опц.): prefill/decode disaggregation, chunked prefill, приоритеты.

Ниже — детальный разбор.

---

## 1. Карта текущей архитектуры (что реально в коде)

### 1.1 HTTP-слой — нормальный
`promeserve/http_server.h`: thread-per-connection через worker-pool. `accept_loop()` кладёт сокеты в bounded `std::queue<socket_t>` (глубина `max_queue_depth=128`), воркеры (`hardware_concurrency()` по умолчанию) разбирают. Очередь полна → быстрый `503 + Retry-After: 1`. `TCP_NODELAY` выставлен, чанкованный streaming через `StreamWriter`. Bind по умолчанию на `127.0.0.1` (правильно). **Этот слой готов к нагрузке** — проблема ниже по стеку.

### 1.2 Слой генерации — bottleneck
`promeserve/api_handlers.h`:
- `handle_generate` / `handle_chat` после загрузки модели берут **`std::lock_guard<std::mutex> lock(generate_mutex_)`** (строки ~776 и ~881). С этого момента весь декод (десятки–сотни forward-проходов) держит глобальный мьютекс. Все остальные воркеры, дойдя до этой строки, блокируются. Фактически: **N HTTP-воркеров, 1 «GPU-воркер»**.
- KV-cache берётся из `model->kv_cache` и `reset()`-ится в начале каждого запроса (`generate_streaming`, строка ~968). Глобальное состояние модели → даже без мьютекса два запроса затёрли бы KV друг друга.
- CUDA-graph: при изменении размера KV вызывается `model->invalidate_graph()`. Граф захватывает scratch-указатели один раз и реплеится — это однопоточный, single-context механизм by design.

### 1.3 Ядро инференса — single-request by construction
`torch/io/gguf_model.h`:
- `struct KVCache` — один экземпляр в `GGUFModel` (`model->kv_cache`). `allocate()` резервирует `[max_seq, kv_dim]` непрерывно на слой; `append()` пишет по смещению `seq_len`. Это **contiguous KV**, не paged.
- `InferenceScratchPool scratch_` / `CPUScratchPool cpu_scratch_` — по одному на модель. Буферы рассчитаны на batch=1 (`buf_x[1,H]`, `buf_logits[1,vocab]`).
- `forward_decode()` (CUDA) и `forward_decode_cpu()` — строго single-token, single-sequence. Читают/пишут `kv_cache.seq_len` как глобальный курсор.
- CUDA-graph бейкает `d_past_len_`, `d_token_id_`, `decode_stream_`, scratch-указатели — один контекст на процесс.
- `generate()` (строка ~4792) — блокирующий цикл «forward → sample → forward», возвращает целиком строку (используется в tool-loop). Тоже single-sequence.
- Есть `forward_decode_cpu_batched(tokens, K, logits_out)` — но это **спекулятивный verify** (K дублей одной последовательности, K-serial внутри, «no speedup» по комментарию на ~4294). Это НЕ кросс-запросный батчинг и переиспользовать его как таковой нельзя.

**Вывод по разделу:** многопользовательский throughput сегодня = throughput одного запроса, делённый на число пользователей, плюс штраф на head-of-line blocking. Узкое место — не HTTP, а монопольное владение единственным вычислительным контекстом.

---

## 2. Как это делают другие (референс)

| | PromeServe (now) | Ollama / llama.cpp-server | vLLM |
|---|---|---|---|
| Параллелизм | 1 (глоб. мьютекс) | N слотов (`--parallel`, `n_seq`), общий контекст с разделением на seq | continuous batching, десятки–сотни слотов |
| KV-cache | 1 contiguous на модель | контекст разбит на `n_seq` слотов, contiguous-per-slot | **PagedAttention** — KV блоками 16 ток., виртуализация |
| Планировщик | нет (FIFO сокетов) | server slot scheduler, in-flight batching по токенам | iteration-level scheduler (prefill+decode микшируются) |
| Prefill vs decode | сериализованы | chunked prefill (новее), один проход на батч | chunked prefill + decode в одном step |
| Streaming | NDJSON chunked, per-token | SSE/NDJSON, per-token | SSE, per-token |
| Отмена | нет (досчитывает) | по disconnect/abort | по disconnect |
| Метрики | нет | базовые (timings в ответе) | Prometheus (TTFT, TPOT, queue, KV util) |

Ключевая идея, которую надо перенять у llama.cpp-server **в первую очередь** (она дешевле vLLM и даёт 80% эффекта): **«слоты»** — фиксированный пул из `S` независимых последовательностей, каждая со своим KV-срезом и своим декод-курсором, обслуживаемые одним вычислительным потоком, который на каждом шаге прогоняет батч из активных слотов. vLLM-уровень (paged KV + динамические слоты) — следующий шаг.

---

## 3. Целевая serving-архитектура

### 3.1 Модель исполнения
Один **GPU/compute-поток** («engine loop»), которому HTTP-воркеры передают `Request` через очередь. HTTP-воркер НЕ считает — он только парсит, кладёт запрос в scheduler и стримит токены, которые engine-loop ему отдаёт через per-request lock-free очередь/condvar. Это убирает контеншн мьютекса целиком: вычисление всегда в одном потоке, синхронизация — только на границе передачи токенов.

```
HTTP worker ──parse──> Scheduler queue ──> Engine loop (1 thread)
   ▲                                            │ step(): batch decode активных слотов
   └────────── per-request token channel ◄──────┘ → push токены обратно
```

### 3.2 Per-request состояние (`SequenceState`)
Вынести из `GGUFModel` всё, что сейчас глобально:
```cpp
struct SequenceState {
    int slot_id;
    std::vector<int32_t> tokens;        // prompt + generated
    int64_t kv_offset;                  // позиция в paged/slot KV
    int n_past;                         // = декод-курсор (было kv_cache.seq_len)
    SamplingParams sp;                  // temp/top_k/top_p/repeat_penalty
    TokenChannel out;                   // куда engine пишет токены для HTTP-воркера
    std::atomic<bool> cancelled;        // выставляется при disconnect
    enum { WAITING_PREFILL, DECODING, DONE } phase;
    Timings t;                          // TTFT, TPOT, eval_count
};
```
`GGUFModel` становится **stateless по запросу**: веса (read-only, шарятся свободно) + методы, принимающие `SequenceState&` и пишущие в KV по `slot`/`block`-таблице, а не в глобальный `seq_len`.

### 3.3 KV-cache: per-request → paged
**Шаг A (slots, contiguous):** аллоцировать KV как `[S_slots][num_layers][max_seq_per_slot, kv_dim]`. Каждый слот — независимый contiguous-блок. `append()` принимает `slot_id`. Минимальная переделка существующего `KVCache`: добавить измерение слота, заменить `seq_len` на `n_past` в `SequenceState`. Этого достаточно для Ollama-паритета.

**Шаг B (paged, как vLLM):** KV — пул блоков по `B=16` токенов; на слот — block-table `int32[ceil(max_seq/16)]`. Attention-kernel читает K/V по block-table (gather). Даёт: нет фрагментации, динамический рост контекста, prefix-sharing (общий системный промпт → общие блоки). Дороже по kernel-работе — это Фаза 2/3.

VRAM-бюджет (qwen3:4b, 36 слоёв, kv_dim≈1024, fp16 KV): ~`36×2×kv_dim×2B` = ~147 КБ/токен/слот. 8 слотов × 4096 ток. = ~4.6 ГБ только KV — это надо явно бюджетировать и ограничивать `S` и `max_seq` исходя из свободной VRAM (Ollama делает то же через `num_ctx × num_parallel`).

### 3.4 Continuous batching (Фаза 2 — главный throughput-рычаг)
Engine-loop, итерация = один «step»:
1. **Admission:** взять из очереди новые запросы, пока есть свободные слоты и KV-блоки.
2. **Prefill:** для новых слотов прогнать промпт. Чтобы длинный prefill не убивал decode-латентность остальных — **chunked prefill**: резать промпт на куски (например 256 ток.) и мешать с decode-шагами.
3. **Batched decode:** собрать по 1 токену с каждого активного слота → `[A, H]` (A=число активных), один проход по слоям с **GEMM `[A,H]×[H,out]`** вместо A независимых GEMV. Это и есть источник throughput-выигрыша: на decode мы memory-bound, веса читаются один раз на батч из A запросов вместо A раз.
4. **Sample** per-slot (у каждого свои temp/top_k/penalty — sampling остаётся независимым).
5. **Emit:** разослать токены в `TokenChannel` каждого слота; завершённые (EOS/max/cancel) — освободить слот и KV-блоки.

**Критично для текущей кодовой базы:** decode сейчас — GEMV (`forward_decode_cpu`, `launch_cublas_hgemv`). Для батча нужен GEMM-путь `[A,H]`. На CPU это `cblas_sgemm`/PromeBLAS уже есть. На CUDA — `cublasHGEMM`/`cublasSgemmStridedBatched`, тоже доступно. **Но CUDA-graph под батч переменного A не подходит** — граф фиксирует форму. Решения: (а) реплей графа только при `A==1` (fallback на текущий быстрый путь для одиночного юзера), (б) набор пред-захваченных графов на дискретные A∈{1,2,4,8} с паддингом, (в) eager-режим при A>1. Рекомендую (а)+(в) на старте: одиночный пользователь не теряет ни tok/s, многопользовательский режим работает без графа.

### 3.5 Prefill/decode scheduler
Политика на step: приоритет decode (низкая латентность для тех, кто уже генерит) с бюджетом на prefill-токены за step (chunked). Это предотвращает: новый запрос с 8k промптом замораживает всех. llama.cpp пришёл к этому же (chunked prefill в 2024). Disaggregation prefill/decode на разные устройства — Фаза 3, не для одной GPU.

---

## 4. Streaming, отмена, грейсфул-деградация

### 4.1 Streaming — починить cancellation
Сейчас `StreamWriter::write` возвращает `false` при разрыве сокета, и `generate_streaming` корректно выходит (строки ~1095). Но в новой модели HTTP-воркер и engine разделены — нужен явный сигнал отмены: при `write`-фейле воркер выставляет `SequenceState.cancelled=true`, engine на следующем step освобождает слот. **Сейчас отмены по disconnect фактически нет в tool-loop и в blocking `generate()`** — там клиент может отвалиться, а сервер досчитывает 128 токенов впустую, держа мьютекс. Это прямой DoS-вектор и трата ресурса. Cancellation — обязательна и дешева.

### 4.2 Грейсфул-деградация (приоритеты по важности)
1. **Backpressure по слотам, не по сокетам.** Сейчас 503 только при переполнении сокет-очереди (128). Нужен второй уровень: если все слоты заняты и admission-очередь длиннее порога → 503 с `Retry-After`, оцениваемым из текущего TPOT × длины очереди. Честный сигнал клиенту лучше, чем 60-секундный таймаут.
2. **Per-request timeout уже есть** (`server_timeout_ms`, проверка раз в decode-шаг) — сохранить, но привязать к `SequenceState`, а не к локальной переменной.
3. **Деградация при нехватке KV:** если новый запрос не помещается по блокам — не падать, а оставлять в WAITING, либо (vLLM-style) **preemption**: вытеснить хвост самой длинной/низкоприоритетной последовательности (recompute или swap KV в host RAM). На старте достаточно «не допускать сверх лимита слотов».
4. **Изоляция падений:** исключение в одном слоте не должно ронять engine-loop — try/catch вокруг per-slot работы, слот → DONE с ошибкой, остальные живут. Сейчас try/catch есть вокруг всего запроса, чего хватает для модели «1 запрос», но не для батча.
5. **Смена модели:** сейчас `load_model` выбивает всех (одна модель в памяти). Минимум — дренаж: не принимать новые запросы под новую модель, пока активные на старой не закончат. Мульти-модель в памяти — отдельный бюджет VRAM, опционально.

### 4.3 Корректность API под батчем
Ollama-поля ответа (`prompt_eval_count`, `eval_count`, `*_duration`) сейчас считаются per-request — это сохраняется, т.к. тайминги в `SequenceState`. Важно: `created_at`, порядок чанков и финальный `done:true` остаются строго per-stream. Батчинг — деталь реализации, наружу инвариантен. Проверить, что `tool_log`/tool-loop путь тоже переведён на cancellation.

---

## 5. Метрики (нет совсем — добавить в Фазе 0)
Минимальный набор (экспонировать на `/metrics`, Prometheus-формат — клиенты уже умеют):
- `ttft_seconds` (histogram) — time-to-first-token, p50/p95/p99.
- `tpot_seconds` (histogram) — time-per-output-token (inter-token).
- `tokens_per_second` (gauge) — суммарный decode-throughput по батчу.
- `active_slots`, `waiting_requests`, `kv_blocks_used/total` (gauges).
- `requests_total`, `requests_rejected_503_total`, `requests_cancelled_total` (counters).
- `prefill_tokens_total`, `decode_tokens_total` (counters).

Без этих метрик невозможно ни доказать паритет с Ollama, ни ловить регрессии. Это первое, что я бы добавил — оно дешёвое и сразу окупается.

---

## 6. Дорожная карта (честно, по фазам)

### Фаза 0 — «не ломая ядро» (2–4 дня)
- `/metrics` + базовые счётчики/гистограммы (TTFT, TPOT, 503, cancelled).
- Cancellation по disconnect в `generate_streaming`/`generate_streaming_chat` (есть) **и** в tool-loop/blocking `generate()` (нет — добавить флаг отмены, прокидываемый в цикл).
- Заменить `generate_mutex_` на счётный семафор `S=1` (поведение то же), но с явным «admission» и 503 при превышении — подготовка к слотам. tok/s одиночного запроса не меняется.
**Результат:** наблюдаемость + нет впустую сожжённого compute на отвалившихся клиентах.

### Фаза 1 — слоты / per-request KV (1–2 недели) — Ollama-паритет
- Вынести KV-курсор и KV-буферы в `SequenceState`; `KVCache` → `[S_slots]`-мерный, contiguous-per-slot.
- `forward_decode*` принимают `slot_id`/`n_past` вместо глобального `seq_len`.
- Engine-loop с round-robin по слотам (ещё БЕЗ batched-GEMM — каждый слот считается по очереди в рамках step, но запросы больше не ждут друг друга целиком, чередуются по токенам).
- CUDA-graph: оставить только для `S_active==1`; при >1 — eager.
**Результат:** N пользователей реально обслуживаются параллельно (interleaved decode), p99 TTFT перестаёт зависеть от длины чужого ответа. Throughput на пользователя ещё не растёт (нет батч-GEMM), но multi-tenant latency — как у Ollama.

### Фаза 2 — continuous batching (3–4 недели) — путь к vLLM-классу
- Batched decode-GEMM `[A,H]` (CPU: PromeBLAS sgemm; CUDA: cublas batched/HGEMM).
- Chunked prefill + prefill/decode scheduler с бюджетом.
- (Опц.) paged KV + block-table + prefix-sharing общего system-prompt.
**Результат:** агрегированный throughput растёт с числом активных запросов (memory-bound веса читаются раз на батч), а не делится.

### Фаза 3 — опционально
- Preemption/swap KV при нехватке, приоритеты, дисагрегация prefill/decode, пред-захваченные графы на дискретные A.

---

## 7. Риски и честные оговорки
- **CUDA-graph vs батч** — фундаментальный конфликт форм. Я сознательно сохраняю граф только для single-user; многопользовательский режим стартует без графа. Это корректный, не «полумерный» компромисс: пользователь-одиночка не теряет святые tok/s, multi-tenant получает функциональность. Закрытие — пред-захваченные графы на фиксированные A (Фаза 3).
- **Batched-GEMM пути уже есть** (cblas/PromeBLAS, cublas HGEMM/strided) — это снижает риск Фазы 2; основная работа — рефактор state, не новые kernels.
- **paged attention** требует нового attention-kernel (gather по block-table) — это самый дорогой пункт; до него Ollama-паритет уже достигнут на contiguous-слотах, так что его можно отложить без потери multi-tenant.
- **Sampling остаётся per-slot** — никакой экономии тут нет и не нужно, это дешёвая часть.
- Разрыв single-stream 88 vs 188 tok/s — это **не** serving-слой; его закрывают в ядре (kernels/quant). Serving-дизайн выше его не чинит, но и не мешает: он умножает любой single-stream tok/s на эффективный параллелизм.

---

## Приложение: точки изменения в коде
| Что | Файл | Сейчас |
|---|---|---|
| Глобальная сериализация | `promeserve/api_handlers.h` ~776, ~881 | `std::lock_guard(generate_mutex_)` вокруг всего декода |
| Глобальный KV reset | `promeserve/api_handlers.h` ~968, ~1188 | `model->kv_cache.reset()` per request |
| KV-cache (1 на модель) | `torch/io/gguf_model.h` ~446 `struct KVCache` | contiguous `[max_seq,kv_dim]`, курсор `seq_len` |
| Scratch (batch=1) | `torch/io/gguf_model.h` ~554 `InferenceScratchPool` | `buf_*[1,...]` single-sequence |
| CUDA-graph single-context | `torch/io/gguf_model.h` ~2575 `forward_decode` | `d_past_len_`/`d_token_id_`/graph забейканы |
| Decode = GEMV | `torch/io/gguf_model.h` `forward_decode_cpu`, `launch_cublas_hgemv` | нет батч-GEMM по запросам |
| HTTP/streaming (готов) | `promeserve/http_server.h` | thread-pool + queue + chunked — переиспользуется как есть |
| Cancellation (частично) | `promeserve/api_handlers.h` ~1095 (есть), tool-loop (нет) | по `writer.write()==false` только в streaming-пути |
