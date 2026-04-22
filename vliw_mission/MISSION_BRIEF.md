# VLIW MISSION — Elbrus E8C2 правильная оптимизация Q4_K_M инференса

## Цель
Добраться до **10–15 tok/s** на qwen3:4b Q4_K_M на E8C2 через настоящую эксплуатацию VLIW-параллелизма LCC 1.29 (не обёртка над AVX2 translation, а native упаковка 6 компутных каналов на ядро × 32 ядра × 4 NUMA узла).

## Текущее состояние (на начало миссии)
- 1-proc best: **4.7 tok/s** (24-thread, numactl --interleave=all, Q4_K AVX2 prefetch)
- 4-proc TP best: **5.5 tok/s** (split output_proj + SHM AllReduce + NUMA-pinned)
- Peak E8C2: ~2300 GFLOPS/chip × 4 = ~9200 GFLOPS, nodelocal EML_MT достигает 1840 GFLOPS (92% одного chip'а)
- Bottleneck: неизвестен точно. Подозреваем memory bandwidth (Q4_K 2.5 GB/pass × 5.5 tok/s = 13.75 GB/s против ~20 GB/s/chip DRAM).
- Ранее доказано: native E2K intrinsics (qpmaddubsh) **-23% slower** чем LCC auto-translation из AVX2 (probe в `examples/benchmarks/q4k_e2k_kernel_probe.cpp`).

## Гипотеза
Главный резерв — **не** в SIMD-слое (там LCC уже хорошо), а в:
1. **Memory bandwidth efficiency** — нужен измеренный profile, а не догадки
2. **Нужна ли Q-дискуенция вообще?** Pre-dequant Q4_K → fp16 или int8 даёт память × 2–4 но может упереть вычисление вместо полосы
3. **Speculative decoding** — уже пробовали, serial проиграл. Перепроверить с batched verify на TP
4. **Cross-layer prefetching + weight pinning** — что не сделано

## Правила для агентов
**КРИТИЧНО: ВСЕ агенты пишут свои отчёты СЮДА в `vliw_mission/`**. Формат:
`vliw_mission/agent_<N>_<role>.md`

Пример: `vliw_mission/agent_1_e2k_architecture.md`, `vliw_mission/agent_2_memory_profile.md`.

Каждый отчёт должен содержать:
1. **Что найдено** (конкретные числа, файлы, pmccntr counters если есть)
2. **Предлагаемое изменение** (file:line, код)
3. **Ожидаемый speedup** (честная оценка, не "возможно поможет")
4. **Риск** (что может сломаться)
5. **Blocker** если есть (что не смогли проверить без live Elbrus)

**Запрещено:** агенту пушить в git, изменять код. Только read + write в `vliw_mission/`.

## Статус консультаций
- **Gemini 3.1 Pro**: API key был отозван Google'ом (leak в прошлой сессии). Ждём новый ключ от пользователя.
- **Gemini Deep Research**: то же.
- **Opus 4.7 агенты**: активны, работают по списку ниже.
