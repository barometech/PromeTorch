# Спецификация: оптимальный decode-loop для PromeServe

**Автор:** Priya Nair (Principal Engineer, NVIDIA TensorRT-LLM)
**Зона ответственности:** decode-loop архитектура — launch-overhead, синхронизации, фьюзинг, KV-cache layout, sampling on-device.
**Объект:** `torch/io/gguf_model.h` (`forward_decode`, `generate`), `aten/src/ATen/cuda/FlashDecoding.cu`.
**Замер:** qwen3:4b A100 — 88 tok/s (11.4 мс/ток) у нас vs 188 tok/s (5.3 мс/ток) Ollama. Дефицит ~6.1 мс/ток.

---

## EXECUTIVE SUMMARY (куда утекает бюджет на токен)

1. **Launch-overhead УЖЕ решён.** Полный decode (≈363 ядра: 10/слой × 36 + emb + final_norm + lm_head) запечён в CUDA Graph, replay = 1 launch. Здесь терять нечего — дефицит НЕ в launch'ах.
2. **🔴 FP32 KV-cache на горячем пути = 2× bandwidth в attention.** Graph-путь (`launch_flash_decode_graph`) читает `key_cache`/`value_cache` в FP32. FP16-KV (`launch_flash_decode_fp16`) висит ТОЛЬКО на не-graph ветке — отключён, т.к. «baked offsets break CUDA Graph» (комм. строка 2818). Сам код оценивает потерю: ~112 МБ/шаг лишнего чтения (FlashDecoding.cu:463-465). На длинном контексте это главный сток. **Оценка: 1.5–2.5 мс/ток.**
3. **🔴 Sampling вне графа + блокирующий D2H sync каждый токен.** `generate()` (4905-4914): после graph-replay делается `cudaStreamSynchronize` (2637), ЗАТЕМ argmax-ядро launch'ится на `nullptr` (default stream, НЕ `decode_stream_`!), затем **блокирующий `cudaMemcpy` D2H** одного int64 (4912). Это: лишний launch + cross-stream сериализация + второй full-pipeline-stall на токен. **Оценка: 0.8–1.5 мс/ток** (два sync вместо одного + bubble на default↔decode stream).
4. **🟡 lm_head FP32-logits D2H в general/penalty-пути.** В greedy-GPU всё ок (argmax на девайсе), но как только `repetition_penalty>1` или `temperature>0` — `get_row(logits)` тянет весь вектор 151936×4B≈600 КБ на хост и сэмплит на CPU. Для PromeServe (всегда с штрафами/семплингом) это убивает скорость. **Оценка: 1–2 мс/ток в реальном serving-режиме.**
5. **🟡 copy+accumulate = 2 ядра там, где нужно 1.** Output-residual и down-residual делают `launch_copy` (x→buf) + `q4km_persistent_gemv_accumulate`. Это 2×36=72 лишних мелких memcpy-ядра/токен. В графе launch бесплатен, но лишний read/write H-вектора и точка десинхронизации остаются. **Оценка: 0.3–0.6 мс/ток.**
6. **🟡 H2D двух указателей каждый replay.** `d_past_len_`+`d_token_id_` шлются двумя `cudaMemcpyAsync` (2634-2635) перед каждым replay. Мелочь, но сериализует старт графа.

**Итог дефицита ~6 мс:** ≈2 мс FP16-KV + ≈1.2 мс argmax/sync-bubble + ≈1.5 мс sampling-D2H (serving) + ≈0.5 мс copy-split + ≈0.8 мс прочее (final_norm вне фьюза, raw-FP32 lm_head bandwidth).

**ТОП-3 действия (ROI):** (A) FP16-KV ВНУТРИ графа через device-pointer offset (как уже сделано для `d_past_len_`); (B) argmax/sampling ПОЛНОСТЬЮ on-device и ВНУТРИ графа — убрать D2H каждый токен; (C) перенести второй `cudaStreamSynchronize` в один на токен и убрать default-stream launch.

---

## 1. Текущая анатомия токена (что реально происходит)

### 1.1 Fast-path replay (`forward_decode`, 2630-2640)
```
*h_past_len_pinned_ = past_len;  *h_token_id_pinned_ = token;
cudaMemcpyAsync(d_past_len_, ..., s)   // H2D #1
cudaMemcpyAsync(d_token_id_, ..., s)   // H2D #2
cudaGraphLaunch(decode_graph_exec_, s) // 1 launch, ~363 ядра внутри
cudaStreamSynchronize(s)               // STALL #1 — ждём logits
return sp.buf_logits                   // FP32 [1, vocab]
```

### 1.2 Граф (запечён один раз, на 2-м токене)
Per-layer (Q4_K fused, 36 слоёв):
| # | Ядро | Фьюз |
|---|------|------|
| 1 | `q4km_fused_rmsnorm_qkv_gemv` | attn_norm+Q+K+V (4→1) ✅ |
| 2 | `fused_qknorm_rope_kvwrite_graph` | qk-norm+RoPE+KV-write ✅ |
| 3-4 | `flash_decode` partial + reduce | 2 ядра, **FP32 KV** 🔴 |
| 5-6 | `copy` + `q4km_persistent_gemv_accumulate` | o-proj+residual (split 🟡) |
| 7 | `q4km_fused_rmsnorm_gate_up_gemv` | ffn_norm+gate+up (3→1) ✅ |
| 8 | `silu_mul` | silu(gate)*up ✅ |
| 9-10 | `copy` + `q4km_persistent_gemv_accumulate` | down+residual (split 🟡) |

≈10/слой ×36 = 360 + embedding(1) + final_norm(1) + lm_head(1) = **≈363 ядра/replay**.

### 1.3 Sampling (`generate`, 4905-4914) — ВНЕ графа
```
launch_argmax(logit_row, d_argmax_idx, V, nullptr)  // default stream!
cudaMemcpy(&h_idx, d_argmax_idx, ... D2H)            // STALL #2, блокирующий
```
→ **2 полных stall'а на токен** + argmax на не том стриме + D2H каждый токен.

---

## 2. Целевой дизайн (TRT-LLM-tight)

### 2.1 Принцип: один replay, один sync, ноль D2H на горячем пути
```
обновить d_past_len_/d_token_id_ (1 H2D, можно слить в 1 буфер из 2 полей)
cudaGraphLaunch(full_graph)        // включает attn+lm_head+argmax+sample
cudaMemcpyAsync(&next_token, d_next_token_, D2H, s)  // 4 байта, async
cudaStreamSynchronize(s)           // ЕДИНСТВЕННЫЙ stall на токен
```
Ключ: sampling **внутри** графа пишет финальный `d_next_token_` (int32). Хост забирает 4 байта одним async-копированием, совмещённым с sync. Никаких 600 КБ logits на хост, никакого второго stall, никакого default-stream ядра.

### 2.2 FP16-KV внутри графа (ТОП-1)
- Сделать `flash_decode_fp16` **graph-совместимым** ровно как FP32-версию: читать длину из `d_past_len_` (device pointer), а не из baked `total_seq`. Проблема «baked offsets» решается тем же приёмом, что уже применён в `launch_flash_decode_graph` и `launch_fused_qknorm_rope_kvwrite_graph`.
- KV-write в FP16-кэш вернуть в граф (kernel `fp16_kv_cache_write_kernel` уже есть, FlashDecoding.cu:468) — слить в `fused_qknorm_rope_kvwrite_graph`, чтобы писать сразу FP16, без отдельного FP32-кэша на decode.
- Эффект: −112 МБ/шаг чтения, −2× attention bandwidth. Это самый крупный единичный выигрыш на длинном контексте.

### 2.3 Sampling on-device, в графе (ТОП-2)
- Один persistent-kernel: `argmax`/`top-k+top-p+temperature+repetition_penalty` → `d_next_token_` (int32). Greedy = block-reduce argmax по 151936 (уже есть `launch_argmax`, надо лишь запускать на `decode_stream_` и захватить в граф).
- Repetition penalty on-device: держать `d_generated_` (кольцевой буфер token-id на девайсе) и применять штраф к logits ядром перед argmax — убирает CPU-цикл по `generated` и D2H всего вектора. Это разблокирует serving-режим (penalty/sampling) без потери скорости.
- top-k/top-p: device-side через partial-sort в shared mem на vocab (или two-pass threshold). Стандарт TRT-LLM — оставить весь sampling на GPU.

### 2.4 Слить copy+accumulate → один GEMV-with-residual (ТОП-3, дешёво)
- `q4km_persistent_gemv_accumulate` уже умеет `+=`. Добавить вариант, который инициализирует выход residual'ом в том же ядре (читает `x_ptr`, пишет `out = x + W@a`), убирая `launch_copy`. −72 ядра/токен и −1 проход по H-вектору ×2×36.

### 2.5 Мелочи
- Слить `final_norm` + lm_head-GEMV (RMSNorm+HGEMV fused) — как уже сделано для attn/ffn norm.
- `d_past_len_`+`d_token_id_` → одна 16-байтная структура, один H2D.
- lm_head: argmax не требует FP32-вывода всего вектора в глобальную память, если sampling-kernel читает logits из того же стрима — но это упирается в дизайн 2.3; держать logits в FP32 ок, главное не копировать на хост.

---

## 3. Приоритеты и ожидаемый выигрыш

| # | Действие | Сложность | Выигрыш (мс/ток) | Кому критично |
|---|----------|-----------|------------------|---------------|
| A | FP16-KV в графе (device-ptr длина) | средняя | 1.5–2.5 | длинный контекст |
| B | Sampling on-device + в графе, D2H 4 байта async | средняя | 1.0–1.5 | greedy И serving |
| C | Убрать 2-й sync + default-stream argmax | низкая | 0.5–1.0 | все |
| D | Repetition-penalty/top-k/top-p on-device | высокая | 1.0–2.0 | PromeServe (всегда с семплингом) |
| E | copy+accumulate → fused residual-GEMV | низкая | 0.3–0.6 | все |
| F | final_norm+lm_head fuse, single H2D | низкая | 0.2–0.4 | все |

**Прогноз:** A+B+C+E (без переписывания сэмплера) → ~3–4 мс возврата → **11.4 → 7–8 мс/ток (~130–145 tok/s)**. Полный комплект A–F с device-side сэмплером → **5.5–6.5 мс/ток (~150–180 tok/s)**, паритет с Ollama.

---

## 4. Контроль регрессий (правило 11.4-святая)
- Каждый шаг мерить tok/s ДО/ПОСЛЕ. Падение >0.2 → откат (правило `feedback_speed_first`).
- FP16-KV проверить bit-similarity на 110+ токенах (история мусора на длинной генерации, `a42d2ac`/`d25b95b`) — FP16 KV не должен деградировать numerics.
- `PT_NO_GRAPH=1` оставить как диагностический bypass для изоляции graph-багов от kernel-багов.
- Профилировать через уже встроенные `PROF_BEGIN/PROF_END` метки + Nsight Systems: смотреть на gaps между ядрами (bubbles) и на D2H-маркеры на таймлайне.

---

## 5. Где НЕ терять время
- Launch-overhead закрыт графом — не оптимизировать повторно (антициклы, `AVOIDRECURSION.md`).
- FP16-веса для GEMV decode НЕ дают ускорения (N=1 bandwidth-bound, Q4_K 2.5ГБ < FP16 5.9ГБ — `feedback_fp16_wont_speedup_decode`). Фокус на KV-bandwidth и sync/D2H, НЕ на пересборке весов в FP16.
- Кастомный custom-GEMM-kernel — dead code, не трогать.
