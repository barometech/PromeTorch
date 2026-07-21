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

## 3. GEMV 2× refactor (см. PLAN_gemv_2x_refactor.md) — БОЛЬШЕЙ ЧАСТЬЮ СДЕЛАНО (2026-07-21, ncu)
- **ncu найден в системе** (`Nsight Compute 2024.1.0`) — не пришлось ставить. ncu =
  dev-инструмент замера, НЕ зависимость PromeTorch.
- **Phase 0 (`7e8c967`):** микробенч `examples/gguf/bench_gemv.cu` (CUDA-events +
  эфф. полоса) — гоняет ОДНО ядро изолированно. Профиль под ncu.
- **Ключевая находка ncu:** dp4a vs FP32 зависит от N. dp4a быстрее на малых/средних
  N (N=1024: 4.1×, 2560: 2.7×, 4096: 2.1×, 9728: 1.36×), FP32 быстрее только на
  N=19456 (gate_up fused). Прошлые наскоки били dp4a по gate_up → −10%. Правильно —
  dp4a для малых N, FP32 для gate_up.
- **Phase 1/2 (`7e8c967`):** quantize-x-once (q8_buf) + `gemv_scratch` маршрутизирует
  Q4_K с N≤12288 через dp4a (`launch_q4km_q8_gemv`). grid `*4→*6` (ncu: 0.67→1.0 wave).
  **qwen3:4b 100→109 (+9%), gemma3:4b 87.7→102.9 (+17%), deepseek 57.4→68.3 (+19%)**,
  top-1 bit-exact. gemma3 = 88% Ollama. `PT_NO_DP4A=1` — откат.
- **Отвергнуто по ncu (задокументировано):** grid*8 −10% (tail), launch_bounds(256,8)
  −5% (reg spill), loads-ahead −5% (reg 40→48). FP32-ядро register-bound на 23% dram;
  occupancy-трюки вредят — dp4a (меньше регистров) обходит это.
- **Осталось (доп. выигрыш):** gate_up (N=19456, 22% времени) на FP32 — dp4a там
  проигрывает; ускорить его = отдельная задача (возможно split-K или tensor-core
  prefill). flash_decode softmax сериализован (tid==0) — Phase 3.
- Phase 4 (on-device sampler) — argmax уже на GPU (`launch_argmax`, D2H 4 байта).
