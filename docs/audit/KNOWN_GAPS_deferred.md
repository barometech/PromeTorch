# Отложенные gaps (нужен свободный GPU для валидации)

Найдено аудитом сессии 2026-07 (`docs/audit/2026_session_*.md`). Требуют
прогона на GPU — отложены, НЕ забыты. Все code-фиксы, не требовавшие GPU, уже
закоммичены (GeGLU×3, gemv_scratch guard, promeserve race, license, tok/s).

## 1. YaRN / LongRoPE attn_factor на GPU decode/prefill (MED) — ✅ ЗАКРЫТО 2026-07-21 (build_cudnn_q40, `39e795c`)
- **Что:** `config.rope_attn_factor` (YaRN mscale) применяется в 3 CPU-спотах
  (gguf_model.h:3480/4440/6001), но GPU rope-кернелы его НЕ получают (в
  `aten/src/ATen/cuda/*.cu` нет attn_factor/mscale).
- **Семантика (подтверждена):** attn_factor умножает cos/sin ⇒ масштабирует
  Q и K после поворота на af ⇒ attention score ×af².
- **Кого бьёт:** только модели с attn_factor≠1 на GPU + длинный контекст
  (Phi-3 LongRoPE, deepseek2/GigaChat3 YaRN). Для qwen3/gemma3/mistral/llama
  attn_factor=1 → НЕ бьёт (их основной GPU-набор корректен).
- **Фикс (минимальный, без смены сигнатур):** на GPU decode/prefill умножить
  attention softmax-scale на `af²` в call-site flash-decode (эквивалент CPU).
  Компилируется без GPU; но валидацию (Phi-3 длинный контекст, bit-close к CPU)
  сделать ТОЛЬКО на свободном GPU — иначе риск тихо ухудшить Phi-3.
- **Проверено 2026-07-21 (GPU был свободен):** attn_factor у phi3:3.8b = **1.19**
  (из GGUF: `phi3.rope.scaling.attn_factor`), т.е. баг РЕАЛЕН. НО валидировать
  фикс НЕ НА ЧЕМ: единственная доступная LongRoPE-модель (phi3) — Q4_0, а
  GPU-путь её не грузит (см. gap #4 ниже, зависает). GigaChat3-модели локально
  нет. qwen3/gemma3/deepseek — все attn_factor=1, фикс на них no-op.
- **Сделано (`39e795c`):** attn_factor (af²) применён на GPU decode+prefill
  attention-scale. Для attn_factor=1 (qwen/gemma/llama/mistral) — no-op (проверено
  регрессом). Валидировано на phi3 (af=1.19) — выход связный, совпадает с Ollama.
- **Статус:** ЗАКРЫТО в отдельной сборке build_cudnn_q40 вместе с gap #4.

## 4. phi3 (Q4_0) на GPU — ✅ ЗАКРЫТО 2026-07-21 (build_cudnn_q40, `39e795c`)
- **Было:** phi3:3.8b = Q4_0 + merged (attn_qkv, ffn_up=[gate;up]). GPU не имел
  (а) Q4_0-загрузчика/kernel, (б) merged-split на GPU (был только CPU) → веса не
  попадали в VRAM → зависание/мусор.
- **Сделано:** q4_0_gemv_kernel (numeric-тест vs CPU diff 1e-7) + Q4_0 в 4 путях
  загрузки/GEMV + `split_quant_rows_gpu` (device-to-device разрез merged в VRAM) +
  attn_factor на GPU (gap #1). Всё — чистое добавление, Q4_K/Q6_K/Q5_K не тронуты.
- **Результат:** phi3 на GPU — hang → **связный вывод, ~86 tok/s** (≈ Ollama).
  Регресс qwen3:4b/gemma3:4b — связны, без деградации.
- **Осталось (LOW, не блокер):** CPU-путь phi3 всё ещё вырождается (rope_factors
  не грузятся в quant-load; GPU работает т.к. использует прямой Q4_0 GEMV). Фиксить
  при необходимости CPU-инференса phi3.
- **Сборка:** изменения в общих исходниках, но собраны в отдельную build_cudnn_q40;
  рабочий build_cudnn бинарь не пересобирался. Мерж в основную сборку — по решению.

## 2. Формальный re-замер RESULTS.md (LOW) — ✅ ЗАКРЫТО 2026-07-21
- **Что:** канон RESULTS.md = qwen3:4b 82.6 tok/s (2026-04-20).
- **Сделано:** формальный прогон 200 ток, 5-run median, A100, Q4_K_M, greedy +
  Ollama baseline того же дня. Результаты: qwen3:4b **100.6** (+22%), gemma3:4b
  **87.7** (+8%), deepseek-r1:8b **57.4** (+12%). Все выходы связные. RESULTS.md
  обновлён. Прирост — от per-token фиксов F16/NeoX/GeGLU, не от железа.
- **Статус:** ЗАКРЫТО. RESULTS.md — единый источник, «pending ~90» снято.

## 3. GEMV 2× refactor (см. PLAN_gemv_2x_refactor.md) — ЧАСТИЧНО (2026-07-21)
- Phase 0 baseline снят (gate+up FP32-fused: dram 22.4%, occ 47%, 80 мкс).
- **Сделано (build_cudnn_q40, `75867c5`+`a7976cf`):** ILP split-accumulators в
  горячих Q4_K/Q6_K decode-GEMV (gate_up, attn_output/ffn_down, q6k). qkv уже был
  split. Эффект: qwen3:4b 100→102.5 (+2.5%), deepseek-r1:8b 57.4→58.5 (+1.9%),
  gemma3/phi3 ~0% (иной hot-path). Bit-exact top-1, без регресса.
- **Потолок без ncu подтверждён эмпирически:** grid `sm_count*4→*8` (удвоение
  резидентных warps) дал **регресс −10%** (103→93) — kernel НЕ occupancy-bound,
  а тонко оттюнен; слепые правки occupancy/register вредят (совпадает с историей
  плана: NROWS-рерайт регрессил 75/60 vs 89).
- **Оставшийся 2× (до Ollama 189):** требует Phase 1 (quantize-x-once-per-layer)
  + Phase 2 (dp4a-примитив), которые в этом окружении НЕ взять вслепую: ncu
  (Nsight Compute) недоступен, а план требует ncu как единственного судью
  (dram%/occ/лимитер) — иначе повтор прошлых регрессов. Ждёт окружения с ncu.
- Phase 4 (on-device sampler) — argmax уже на GPU (`launch_argmax`, D2H 4 байта).
