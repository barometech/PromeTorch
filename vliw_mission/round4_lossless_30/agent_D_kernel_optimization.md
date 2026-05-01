# Agent D — Inference kernel оптимизация под новый формат, цель 30 tok/s

## Роль
Performance-инженер VLIW e2k v5. Берёт уже работающий 9.4 tok/s SoA путь
и выводит к **30 tok/s** через kernel-уровневую работу.

## Зависимости
Может стартовать **независимо** — оптимизирует существующий q8_soa4_gemv.
После Agent A (новый формат) — портирует findings на новый layout если он
лучше.

## Что прочитать
1. `vliw_mission/round4_lossless_30/MISSION.md` (особенно §3, §4)
2. `torch/io/q8_soa_repack.h:255-340` — production q8_soa4_gemv kernel
3. `torch/io/cpu_quant_gemv.h:1-200` — Q4_K scalar baseline
4. `vliw_mission/e2k_vnni/q8_soa4_microbench.c` — standalone microbench
   (1.21 ms/GEMV K=2560 N=2432 single-core = 0.85× EML)
5. `docs/elbrus/` — APB, SWP, LCC docs (распакованные MCST PDF'ы)
6. `vliw_mission/round3/agent_4_lcc_audit.md` — что LCC compiler уже даёт
7. `c10/util/ThreadPool.h` — fork/join (старый mutex+CV, оптимизировать)
8. `torch/distributed/ddp.cpp:540-620` — SHM AllReduce + futex
9. `JOURNAL.md` (последние 200 строк) — вся история измерений

## Анализ текущего bottleneck

При 9.4 tok/s на TP-4 (28 used cores):
- 100 / 9.4 = **106 ms / token**
- Bandwidth = ~3.5 GB / 106 ms = **33 GB/s aggregate** = 33% от 100 GB/s

То есть мы НЕ упираемся в DDR bandwidth. Время уходит на:
- AllReduce между rank'ами
- ThreadPool fork/join overhead
- Quant_activation per-call (~0.1 ms × 5 = 0.5 ms/token)
- CPU compute (Q8 SoA dot × 200 ops)

Чтобы выйти к 30 tok/s = **33 ms / token**, нужно срезать ~73 ms = 70%.

## Approach (по убыванию ROI)

### A1. Persistent ThreadPool (~+30% throughput)
Round 3 Agent 1 design: replace mutex+queue+CV с broadcast descriptor +
per-worker ack slots + futex gen. Target: **5 µs** per call vs текущий
~100 µs. С ~200 calls/token = **19 ms saved per token**.

В прошлой сессии я попробовал реализовать — deadlock. Нужно правильно:
1. Сначала **microbench harness** в `examples/benchmarks/threadpool_overhead_bench.cpp`
   (он уже есть, но не подключён к CMakeLists). Подключить + запустить
   на Эльбрусе. Цель: <8 µs per call с broadcast pool.
2. Только после прохождения microbench — interлять production gguf_model.
3. Race conditions проверять через `tsan` на x86 build.

### A2. AllReduce optimization (~+15% throughput)
SHM AllReduce per layer. Сейчас sync-based. Альтернативы:
- Fused Compute+AllReduce: пока ranks AllReduce'ят attn output, master
  начинает RMSNorm на partial result (overlap)
- Hierarchical reduce: 4 → 2 → 1 stages (2 hops vs 4 для всех всех)

### A3. Speculative decoding (multi-token verify per forward)
PLD уже реализован, gated `PT_PLD=1`. Регрессировал из-за batched-decode
K-serial path. Если делать **batched verify** через q8_soa4_gemv (которая
batches по N=output_dim) — можно amortize forward cost между accepted
tokens.

При accept rate 60% и K=3 draft tokens: эффективно 2 tokens / forward
вместо 1 → +×2 throughput → 9.4 × 2 = **18.8 tok/s**.
При accept 75% и K=4: 3 tokens / forward → +×3 → **28 tok/s**.

Главное: PLD-style draft (n-gram lookup) работает только на repetitive
prompts. Для real chat — нужен tiny trained draft model (qwen3:0.6B).
Это путь Agent E.

### A4. Kernel fusion поверх Q8 SoA4
- Fused QKV: 3 SoA4 GEMV → 1 fused (общий quant_activation)
- Fused gate+up+SiLU+down: 4 ops → 1 fused. Память для gate/up/down
  читается consequetively, в общем кэше дольше.

### A5. APB tuning для Q8 SoA4 layout
LCC APB читает 4-32 cache lines в окно 32 line. Q8 SoA4 block = 176 байт
= 2.75 cache lines. APB не оптимально настроен. Возможно стоит увеличить
super-row до 128 элементов (= 8 K-groups × 16 байт = 128 байт = 2 cache
lines), либо align на 64 байт.

`#pragma loop count(8) ivdep` per loop directives — некоторые установлены,
не все.

## Output

1. `vliw_mission/round4_lossless_30/agent_D_results.md` — числа после каждой оптимизации (A1, A2, A3...) с regression check на quality
2. Patches:
   - `c10/util/ThreadPool.h` — persistent pool (если microbench OK)
   - `torch/io/q8_soa_repack.h` — APB tuning, fused kernel
   - `torch/distributed/ddp.cpp` — fused compute+AR (если выгодно)
3. Microbench results: `tools/bench_threadpool/output.log`
4. End-to-end TP-4 result после каждой оптимизации (3 runs each)

## Constraints
- **Numerics**: каждое изменение должно сохранять output bit-equivalent
  с baseline (max abs diff < 1e-5). Если падает > этого — revert.
- **Microbench first**: не интегрировать в production пока micro<8µs.
- **3 runs avg** для каждой final cifry — single-run noise может быть 0.5 tok/s.
- Никакого dead code: revert'ы откатывать чисто.
