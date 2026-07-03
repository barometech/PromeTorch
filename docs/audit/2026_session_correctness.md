# Аудит корректности сессии (F16 attn_v / NeoX RoPE / GeGLU) — статический анализ

Дата: 2026-07-03. Только Read/Grep/Glob, без запуска.
Коммиты в фокусе: `a42d2ac` (F16 attn_v → qf16_gemv), `d25b95b` (NeoX RoPE GPU), `85eccc0` (gemma3 GeGLU).

Легенда severity: CRITICAL (мусорный вывод на реальной модели), HIGH (мусор на конкретной
конфигурации/пути), MED (тихий неверный результат в редком пути / perf), LOW (perf/косметика).

---

## CRITICAL

### C1. GeGLU (ffn_gelu) НЕ применён в `forward_decode_cpu_speculative`
- Файл: `torch/io/gguf_model.h:4132-4136`
- Что не так: FFN-активация хардкодит SiLU `(g/(1+exp(-g)))*up`, нет ветки `config.ffn_gelu`.
  Для gemma3 (ffn_gelu=true) этот путь даст неверную нелинейность → мусорный KV → зацикленный текст,
  тот же баг, что чинили в `85eccc0` для decode/prefill.
- Триггер: gemma3 + спекулятивный декод (`forward_decode_cpu_speculative`).
- Как чинить: продублировать ветку `if (config.ffn_gelu) { GELU tanh } else { SiLU }` как в
  `forward_decode_cpu` (3731-3765). Константа c=0.7978845608028654f, 0.044715f.

### C2. GeGLU НЕ применён в `forward_decode_cpu_batched`
- Файл: `torch/io/gguf_model.h:4580-4599`
- Что не так: SiLU-Mul захардкожен (AVX2 + scalar tail), нет `config.ffn_gelu`.
  gemma3 через batched-decode (спек-верификация K токенов) даст мусор.
- Триггер: gemma3 + batched/speculative verify путь.
- Как чинить: добавить GELU-ветку перед AVX2 SiLU, аналог CPU decode.

---

## HIGH

### H1. GeGLU НЕ применён в TP-пути (`forward_decode_cpu_tp`, Эльбрус) — все 3 подветки
- Файлы: `torch/io/gguf_model.h:6360-6363` (use_gather), `:6477`, `:6492` (иные TP-режимы),
  а также q8_soa4 fused путь `:6464-6467` (`q8_soa4_silu_quant_activation_fused` — SiLU-only).
- Что не так: везде хардкод SiLU `(g/(1+exp(-g)))*up`. Флаг `config.ffn_gelu` не читается.
  Если gemma3 запустить на Эльбрусе TP-4 → мусор (та же природа, что C1/C2).
- Смягчение: gemma3 TP-4 реально гоняли (CLAUDE.md: gemma3-4B TP-4 6.7) — если он давал
  корректный русский, значит либо ffn_gelu не выставлялся при том прогоне, либо тест был
  до GeGLU-осознания. В любом случае сейчас код TP хардкодит SiLU → для gemma3 неверно.
- Как чинить: провести `config.ffn_gelu` во все 3 SiLU-места TP + добавить GELU-вариант в
  `q8_soa4_silu_quant_activation_fused` (или отключать SoA-fused при ffn_gelu).

---

## MED

### M1. `cpu_quant_gemv_supported()` НЕ включает F16 (case 1) — по замыслу, но хрупко
- Файл: `torch/io/cpu_quant_gemv.h:1754-1757`
- Что не так: F16 (type 1) отсутствует в supported. Это НАМЕРЕННО (коммит a42d2ac): чтобы
  fused-guard'ы (`can_fuse`, `can_fuse_qkv`) видели F16-V как «неподдержанный» и откатывались
  на раздельные `cpu_quant_gemv`, где F16 case 1 уже есть. Проверено: guard'ы 3346-3354 (decode),
  3942-3947 (spec), 5875-5880 (TP) требуют q==k==v type — F16-V ломает равенство → fallback OK.
  q8_soa путь защищён `q8_soa.valid` (repack только Q4_K, `repack_q4k_to_q8soa4`) → F16-V даёт
  invalid soa → use_soa_qkv=false → fallback OK.
- Риск: если кто-то позже расширит fused-guard, чтобы допускать разные типы, F16-V молча
  вернётся в fused-путь без F16-обработки → V=0. Хрупкая инвариант, держится на неявном совпадении.
- Как чинить (опц.): оставить как есть, но добавить комментарий-предупреждение в guard'ы, либо
  сделать F16 supported и добавить F16 case во ВСЕ fused-кернелы.

### M2. Perf-регрессия TP QKV при F16-V (не корректность)
- Файл: `torch/io/gguf_model.h:5872-5917`
- Что не так: q в Q4_K репакается в q8_soa, но v в F16 → `q8_soa.valid`=false → `use_soa_qkv`=false
  для ВСЕГО QKV → откат на медленный раздельный путь (теряется тройной SoA-fused). Для qwen3:4b
  Q4_K_M на TP-4 это медленнее, но корректно. Замерить tok/s (святое правило 11.4).

---

## LOW

### L1. Диагностический тоггл PT_NO_GRAPH остался в коде — норм (гейтед)
- Файл: `torch/io/gguf_model.h:2652-2655`. Env-гейт, дефолт off, не мусор. Оставить.
- Прочих тогглов сессии (PT_DP4A_FFN/PT_FP16_LMHEAD/PT_NO_LLAMA_GEMV/PT_FORCE_GREEDY) в .h нет.

---

## Проверено ОК (несоответствий нет)

- **NeoX RoPE**: применён во ВСЕХ путях. GPU: `rope_kernel`, `fused_qknorm_rope_kvwrite[_graph]`
  (все 3 берут `config.rope_neox` из call-sites 2800/2813/7531). CPU decode/spec/batched/TP —
  `rope_apply_fused_neox` под `if (config.rope_neox)` (3468, 4422, 5977 и др.).
- **CPU↔GPU бит-в-бит соответствие NeoX**: пары (d, d+half_dim), формула
  `x0*c−x1*s / x0*s+x1*c` идентичны (GPU CUDAInference.cu/FlashDecoding.cu vs
  hot_loops.cpp:2035-2061). NORM = пары (2d,2d+1) — тоже совпадает.
- **GELU-константа CPU↔GPU**: `0.7978845608028654` и `0.044715` идентичны
  (gguf_model.h:3738 vs CUDAInference.cu silu_mul_kernel).
- **F16 qf16_gemv**: scalar + AVX2/F16C, layout row-major [N,K], row_stride_bytes — совпадает
  с остальными gemv-кернелами. case 1 подключён в `cpu_quant_gemv` (1740).
- **Сигнатуры/вызовы**: все 4 GPU call-site обновлены — launch_silu_mul (2990 decode, 7590 prefill),
  launch_rope (7531), launch_fused_qknorm_rope_kvwrite[_graph] (2789/2802). Дефолты в CUDAOps.h
  (gelu=false, neox=false) безопасны для llama/qwen SiLU/… но call-site ВСЕГДA передают явный флаг.
- **`aten_cuda_exports.def`**: mangled-имена соответствуют новым сигнатурам:
  `launch_rope ...HHHHM_N...` (+bool), `launch_silu_mul ...M_N...` (+bool),
  `launch_fused_qknorm_rope_kvwrite ..._N_J2...` (+bool в конце), `_graph ..._N3...`. Осиротевших нет.
- **deepseek2 (GigaChat3)**: MLA-путь, не llama-family FFN/RoPE — GeGLU/NeoX-фиксы неприменимы.

---

## Открытое (задокументировано автором, вне scope этих 3 фиксов)
- SWA-маска (sliding window) на prefill/GPU для gemma — нужна для контекста >1024 (коммит 85eccc0).
- logit soft-cap / query-scalar для gemma2 — отдельным шагом.
