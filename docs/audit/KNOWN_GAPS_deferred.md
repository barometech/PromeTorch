# Отложенные gaps (нужен свободный GPU для валидации)

Найдено аудитом сессии 2026-07 (`docs/audit/2026_session_*.md`). Требуют
прогона на GPU — отложены, НЕ забыты. Все code-фиксы, не требовавшие GPU, уже
закоммичены (GeGLU×3, gemv_scratch guard, promeserve race, license, tok/s).

## 1. YaRN / LongRoPE attn_factor на GPU decode/prefill (MED, узкий)
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
- **Статус:** отложено до свободного GPU. Не патчить вслепую.

## 2. Формальный re-замер RESULTS.md (LOW)
- **Что:** канон RESULTS.md = qwen3:4b 82.6 tok/s (2026-04-20). Re-замер
  2026-07 (после F16/NeoX/GeGLU) дал ~90 tok/s (87.5–92.3, A100, Q4_K_M,
  greedy 150 ток). Расхождение — build/дата.
- **Фикс:** формальный прогон всех моделей RESULTS.md (qwen3:4b/gemma3:4b/
  deepseek-r1:8b + Ollama baseline) на свободном GPU, обновить таблицу единожды.
- **Статус:** отложено до свободного GPU. Промежуточно все доки ссылаются на
  канон 82.6, ~90 помечено как pending.

## 3. GEMV 2× refactor (см. PLAN_gemv_2x_refactor.md)
- Phase 0 baseline снят (gate+up FP32-fused: dram 22.4%, occ 47%, 80 мкс).
- Phase 1+ требуют GPU (ncu-гейты). Старт по команде на свободном GPU.
