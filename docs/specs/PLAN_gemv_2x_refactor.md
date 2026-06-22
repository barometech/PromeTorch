# PLAN: GEMV 2× refactor (quantize-x-once + tuned dp4a kernels)

**Цель:** qwen3:4b decode 89 → 150-200+ tok/s (обгон Ollama 188), без регресса корректности.
**Статус:** план. GPU использовать ТОЛЬКО по команде юзера. Сейчас команды нет.

## Почему наскок провалился (эмпирика этой сессии — учитывать!)
- Включение dp4a-v2 флагом: +1% (флаг не трогает горячие fused-ядра).
- Расфьюз gate+up → dp4a: −10% (unfuse-overhead: лишний RMSNorm + 2× реквант x).
- Рерайт v2 на NROWS=4/8 «bytes-in-flight»: **регресс 75/60 vs 89**. ncu: dram застряла на **15.6%**, occupancy 33.8% (NROWS=4, 64 рег) — latency-bound, NROWS его не лечит.
- **Истинный корень (ncu)**: prologue **квантует x в Q8_1 заново в КАЖДОМ вызове GEMV**; на мелких N (qkv k=1024) это 2% dram — квант не амортизируется. Плюс register pressure режет occupancy.
- **Вывод:** 2× не берётся правкой одного ядра. Нужен рефактор пайплайна квантования + per-kernel ncu-тюнинг. Существующий FP32-fused уже прилично оттюнен — обгонять его надо по-настоящему.

## Дисциплина валидации (железно на каждом шаге)
1. **ncu — единственный судья**: `dram__throughput.pct_of_peak`, `sm__warps_active.pct`, лимитер. НЕ «на глаз» по wall-clock.
2. **Микробенч-харнесс**: гонять ОДНО ядро изолированно (без unfuse-confound, что сбил меня). Phase 0.
3. **Bit-exact top-1 @ T=0** на 110+ токенах vs baseline (история мусора `a42d2ac`/`d25b95b`).
4. **Святое правило 11.4**: любой end-to-end регресс >0.2 tok/s → ОТКАТ шага.
5. Каждая фаза — отдельный коммит; не зашло — revert, не тащить дальше.

## Фазы

### Phase 0 — Харнесс + ncu-гейты (риск 0, GPU)
- Standalone microbench: один Q4_K GEMV [K,N] заданного размера, прогон под ncu, печать dram%/occ/time. Изолирует ядро от пайплайна.
- Зафиксировать целевые dram% по ядрам (gate/up: ≥55%, qkv, down, lm_head).

### Phase 1 — quantize-x-once-per-layer (ГЛАВНЫЙ рычаг; Meta + мой ncu-вывод)
- Новое ядро `quantize_q8_1_layer`: normed-x → Q8_1 в device-буфер ОДИН раз/слой (L2-резидентный).
- Все Q4_K GEMV (qkv/gate/up/down) читают пред-квантованный x; убрать in-kernel prologue.
- Снимает: 2% dram на мелких GEMV + 2× реквант gate/up.
- Гейт: dram↑ на мелких N, tok/s↑, bit-exact.

### Phase 2 — dp4a Q4_K GEMV-примитив, occupancy-tuned (NVIDIA + Intel)
- Один `__device__ __forceinline__` примитив: dp4a, читает пред-квантованный x, scale/min распаковка 1×/блок в регистры.
- **register budget**: `__launch_bounds__` / maxrregcount, держать occupancy ≥50%. NROWS-свип 2/4/8 — выбрать по ncu (НЕ по теории).
- loads-ahead + **split-accumulators 2-4** (Intel, ILP бесплатно).
- **split-K для мелких N** (k-proj/v-proj grid-starved: ncu 2-12% occ) — несколько блоков на строку + atomic/reduce.
- Завести примитив на FUSED gate+up и qkv (горячий путь), а не только gemv_scratch.
- Гейт: ncu dram% ≥55-70%, bit-exact, tok/s↑.

### Phase 3 — attention + Q6_K (Intel + Meta)
- flash-decode: warp-параллельный softmax max/sum через `__shfl_xor` (сейчас только tid==0 — 256-шаговая сериализация).
- Q6_K dp4a `vec_dot_q6_K_q8_1` (сейчас скалярный, ffn_down на Q4_K_M/14B горячий).

### Phase 4 — структурные (Groq; банкуем независимо, низкий риск)
- **on-device sampler**: argmax+rep-penalty на девайсе, кольцевой `d_generated_`, 4-байтный async D2H. Убрать 608 КБ D2H/токен + CPU-цикл (~−1.5-2 мс CPU, видно уже на коротком). ТОП-1 ROI по Groq.
- **FP16-KV в графе**: graph-совместимый flash-decode, читающий длину из `d_past_len_`. Растёт с контекстом.
- fused residual-GEMV (−72 ядра/токен).

### Phase 5 — prefill на Tensor Cores (Meta/Wei) — TTFT
- M>1: dequant→FP16 + `cublasGemmEx` HGEMM вместо M отдельных GEMV. TTFT ×5-10.

## Прогноз (честно, с риском)
- Phase 1+2 → главный decode-выигрыш, цель **150-200 tok/s** (обгон Ollama). Тут весь риск 2×.
- Phase 3 → стабильность на длинном контексте + 14B/Q4_K_M.
- Phase 4 → +структурные мс (надёжно даже если 1-2 буксуют).
- Phase 5 → TTFT (prompt processing).

Источники: `docs/specs/01..05` (спеки) + `verdict_{nvidia,meta,groq,intel}.md`.
Порядок: 0 → 1 → 2 (ядро рычага) → 4 (надёжные) → 3 → 5. Каждая под ncu-гейтом.
