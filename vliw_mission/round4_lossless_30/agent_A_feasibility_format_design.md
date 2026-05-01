# Agent A — Feasibility analysis + Format spec для нового PromeTorch формата

## Роль
Старший архитектор. Финальная цель — **спецификация формата** `.pt8`
(или другое имя) который:
1. Хранит веса qwen3:4b lossless относительно исходного GGUF Q4_K_M
2. Оптимизирован под inference на Эльбрусе E8C2
3. Загружается быстро (<30s на 4 ядрах)
4. Рассчитан на достижимость 30 tok/s после соответствующей kernel'-работы

## Что прочитать (в этом порядке)
1. `vliw_mission/round4_lossless_30/MISSION.md` — общий контекст
2. `torch/io/gguf_reader.h` (~500 строк) — текущий GGUF format parsing
3. `torch/io/gguf_dequant.h` — Q4_K/Q6_K/Q8_0 формулы decode
4. `torch/io/q8_soa_repack.h` — наш текущий best layout (Q8 SoA4)
5. `torch/io/cpu_quant_gemv.h:1-200` — Q4_K scalar kernel inner loop
6. `vliw_mission/round3/agent_5_soa_repack.md` — design Q8 SoA4
7. `JOURNAL.md` (последние 100 строк) — где мы сейчас

## Вопросы которые нужно ответить

### Q1. Bandwidth ceiling по форматам
Подтверди (или опровергни) расчёты §3 MISSION.md:
- На текущем измеренном aggregate DDR bandwidth Эльбруса (что именно
  измерено? см. JOURNAL.md "23% bandwidth utilization root cause")
- Какой реалистичный потолок utilization для tuned VLIW kernel? 60%? 80%?
- Какой формат имеет максимальный "потолок × utilization" product?

### Q2. Какая компрессия даёт максимум скорости БЕЗ потерь vs Q4_K_M
Q4_K_M уже компактен (5 бит/param). Можно ли через структурную
переупаковку (без изменения precision) сделать формат:
- C меньшим overhead на dequant (текущий Q4_K decode требует
  fp16→fp32 conversion of d/dmin + scales unpack + nibble extract)?
- C большей SIMD-пользой (например 8-row interleave вместо 4 для лучшей
  утилизации `qpmaddubsh` × 8 lanes)?
- C *block alignment* под 64-байт cache line / VLIW APB
  (Array Prefetch Buffer — окно 32, длина 4-32 cache lines)?

### Q3. Storage savings без потери
Существуют ли *структурные* экономии в Q4_K_M которые мы можем убрать
без потерь?
- 12 байт packed scales/mins (6-bit) per super-block — можно ли упаковать
  плотнее или развернуть более плоско для скорости decode?
- Группировка sub-block'ов (32 элем) — можно ли увеличить до 64 для
  меньше header overhead на data?

### Q4. Спецификация формата
Финальный output этого агента — **формальная спецификация `.pt8`**:
```
Header (256 байт):
  magic: "PT8\0"
  version: u32
  param_count: u64
  num_tensors: u32
  ...

Tensor table (per tensor):
  name: variable string
  shape: [u64; 4]
  dtype: u8 (Q8_SoA4 / Q4_SoA8 / FP16 ...)
  offset: u64
  size: u64
  metadata: ...

Data section:
  per tensor — packed по spec для dtype
```
(детали — твой output)

## Output формат

Файл `vliw_mission/round4_lossless_30/format_spec_v1.md` (твой deliverable):
1. Полная спека формата (.pt8 layout, на bytes/bits)
2. Pseudo-code для encoder (GGUF → .pt8)
3. Pseudo-code для decoder (.pt8 → in-memory tensors готовые к inference)
4. Сравнительная таблица с Q4_K / Q8_0 / Q8 SoA4 (размер, decode cost, max bandwidth utilization theoretical)
5. Honest assessment: "достижимы ли 30 tok/s с этим форматом" (yes/no/conditional)
6. List of follow-up questions для других agent'ов (что им нужно решить, чего ты не можешь)

## Constraints
- НЕ выдумывай sub-bit precision (1.58-bit, etc) — должно быть
  деplyable lossless из Q4_K_M
- Учитывай что Эльбрус 8C2 v5 имеет:
  - `qpmaddubsh` (16×u8×s8 → 8×i16) ✓
  - **НЕТ** `qpidotsbwss` (full INT8 dot) — это v7+
  - **НЕТ** FP16 hardware (есть только packing/unpacking utilities)
  - APB, SWP optimizations доступны
- Spec должна быть реализуема **в одну сессию работы** для converter
  (Agent B), без heroic engineering
