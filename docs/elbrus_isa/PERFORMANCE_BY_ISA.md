# PromeTorch — Performance by Elbrus ISA

Систематическая таблица: какие intrinsics доступны на какой ISA, какие
паттерны кода PromeTorch ими пользуется, какие library/runtime
требования нужны для воспроизведения заявленной скорости.

**Источник:** эмпирические замеры на конкретных машинах (lemur-1 8C2,
e16c.ru E16C, [коллега]ов 8C/4C). MCST публичный manual описывает API,
но runtime requirements (libeml_mt vs libeml, какие intrinsics) — нет.

---

## Таблица ISA → intrinsics → bandwidth → real-world tok/s

| ISA | Модель CPU | LCC `__iset__` | Integer SIMD mul-add | FP SIMD | EML BLAS | qwen3-4b TP-4 tok/s |
|-----|-----------|---------------:|----------------------|---------|----------|---------------------|
| **v3** | E2S, E4C | 3 | ❌ нет integer SIMD | `pfmuls`/`pfadds` (2-lane FP) | `libeml.so` (single-thread) | ~0.5 (scalar Q8 SoA) |
| **v4** | E8C | 4 | `pmaddubsh` (8 bytes, 2 i16 pairs) + `pmaddh` + `paddw` | `pfmuls`/`pfadds` (2-lane FP) | `libeml.so` или `libeml_mt.so` | ~4-6 (Q8 SoA v4 SIMD) |
| **v5** | E8C2, E8СВ | 5 | `qpmaddubsh` (16 bytes, 4 i16 pairs) | `qpfmuls`/`qpfadds` (4-lane FP) + `qpfnmas`/`qpfmas` (FMA) | `libeml_mt.so` (multi-thread) | **10.9** (Q8 SoA v5 VNNI) |
| **v6** | E16C | 6 | `qpmaddubsh` + дополнительные fused формы | `qpfmuls`/`qpfadds` + FMA | `libeml_mt.so` | ~12-14 (эстимат, не валидирован полностью) |

---

## Конкретные intrinsics (наш hot path)

### Q4_K → Q8 SoA4 GEMV (главный inference bottleneck)

**Файл:** `torch/io/q8_soa_repack.h`

| ISA | Used intrinsics | Lane width | Throughput (per kg-iter) |
|-----|-----------------|-----------:|-------------------------|
| v5+ | `qpmaddubsh` + `qpmaddh` + `qpaddw` + `qpfmuls` + `qpfadds` | 4 i16 / 4 fp32 | 16 i8-pairs / 1 instruction |
| v4 | `pmaddubsh` + `pmaddh` + `paddw` | 2 i16 / 2 i32 | 16 i8-pairs / 2 instructions (2 pmaddubsh per kg) |
| v3 | (none) | scalar | 16 i8-pairs / 16 instructions |

**Compile-time guards в коде** (актуально 2026-05-20):
```cpp
#if defined(__e2k__) && defined(__iset__) && __iset__ >= 5
#define PT_E2K_VNNI 1
#define PT_E2K_VNNI_HALF 0
#elif defined(__e2k__) && defined(__iset__) && __iset__ >= 4
#define PT_E2K_VNNI 0
#define PT_E2K_VNNI_HALF 1     // <- ВКЛЮЧАЕТ v4 SIMD path
#else
#define PT_E2K_VNNI 0
#define PT_E2K_VNNI_HALF 0     // <- v3 scalar fallback
#endif
```

### Attention dot product (Q @ K^T)

**Файл:** `torch/io/gguf_model.h:5988-6021`

| ISA | Path | Lanes |
|-----|------|------:|
| v5+ | `qpfmuls` + `qpfadds` | 4 fp32 |
| v4 | `pfmuls` + `pfadds` | 2 fp32 |
| v3 | scalar | 1 |

### Element-wise GEMM (для Linear, Conv, AttentionOut)

| ISA | EML provider | MT/ST |
|-----|--------------|-------|
| v3 | `libeml.so` | single-thread |
| v4 | `libeml_mt.so` (если установлен) или `libeml.so` | MT preferred |
| v5+ | `libeml_mt.so` | multi-thread per NUMA-node |

**Auto-detect:** `CMakeLists.txt` ищет header `<eml/cblas.h>` + library
`libeml_mt.so`/`libeml.so`. Если нет — `PT_USE_EML_BLAS=OFF` авто и
PromeTorch собирает TUDA 6×6 микроядро (медленнее ~30× в GEMM, но
работает на любом E2K без зависимости).

---

## Runtime requirements per ISA

### v3 (E2S, E4C) — Эльбрус-2С / Эльбрус-4С

**Минимальные требования:**
- LCC ≥ 1.26 (без `__has_include`)
- `libeml.so` (single-thread)
- `numactl` опционально (одинокий чип может не иметь NUMA)

**Известные ограничения:**
- LCC 1.26 не понимает `__has_include` → cmake check_include на v3
  возвращает negative → numa.h автомат disable
- Q4_K integer SIMD недоступен → SoA scalar fallback ~5× медленнее
  legacy `q4k_gemv_scalar` (autovectorized по pragma)
- Рекомендуемый mode: **single-process** (`run_1proc_elbrus.sh`),
  TP не оправдан bandwidth-wise

### v4 (E8C) — Эльбрус-8С первой версии

**Минимальные требования:**
- LCC ≥ 1.28 (рекомендуется 1.29+)
- `libeml.so` или `libeml_mt.so`
- numactl + libnuma-dev

**Особенности:**
- `pmaddubsh` доступен — Q8 SoA path работает (с 2026-05-20)
- `pmaddh`, `paddw` — для int32 accumulator extension
- НЕТ `qp*` 4-lane intrinsics — попытка их использовать даёт
  `built-in function not supported for current cpu mode`
- Рекомендуемый mode: TP-2 или TP-4 (8C имеет 1 NUMA-node × 8 cores
  или 2 ноды по 4 ядра в зависимости от revision)

### v5 (E8C2, E8СВ) — Эльбрус-8С2 / Эльбрус-8СВ

**Минимальные требования:**
- LCC ≥ 1.29
- `libeml_mt.so` ОБЯЗАТЕЛЬНО (single-thread libeml даст ~50% потерь)
- numactl + libnuma-dev

**Особенности:**
- `qpmaddubsh` (4 i16 pairs per instruction) — основной integer SIMD
- `qpfmuls`/`qpfadds`/`qpfnmas`/`qpfmas` — FP SIMD + FMA fold
- LCC SWP scheduler агрессивно pipelin'ит inner loops с
  `_Pragma("loop count(N)")` + `_Pragma("ivdep")` подсказками
- Рекомендуемый mode: TP-4 (4 NUMA × 8 cores = 32 cores) +
  `PT_Q8_SOA=1` + `numactl --cpunodebind=R --membind=R` per rank

### v6 (E16C) — Эльбрус-16С

**Минимальные требования:**
- LCC ≥ 1.29
- `libeml_mt.so` (но смотри **bug #10075** — на E16C `libeml_mt.so`
  помечен `elbrus-2c3` в dynamic loader → нужен `--no-eml` workaround
  в `scripts/build-elbrus.sh`)
- numactl + libnuma-dev

**Особенности:**
- Полный qpmaddubsh + FMA набор
- Larger registers, новые fused формы (см. extracted/E2K_opcodes.xlsx)
- TP-8 (или TP-16 на full E16C, нам [коллега] выделил 8/16 ядер)

---

## CRITICAL: что НЕ работает «из коробки»

### EML bugzilla #10075 на E16C

`libeml_mt.so` на E16C поставляется с тегом `elbrus-2c3` — dynamic
loader блокирует загрузку. Workaround: `--no-eml` флаг в
`build-elbrus.sh` → TUDA 6×6 микроядро GEMM (медленнее, но работает).

### MT_CBLAS_DEFS обязателен ДО `#include <eml/cblas.h>`

Без define EML декларирует `__cblas_sgemm` (internal name) вместо
публичного `cblas_sgemm` → линковщик не находит symbol. Наш fix:
```cpp
#if defined(TUDA_E2K) && defined(PT_USE_EML_BLAS)
#ifndef MT_CBLAS_DEFS
#define MT_CBLAS_DEFS
#endif
#include <eml/cblas.h>
#endif
```

### `-mtune=elbrus-vN` ломает LCC

LCC принимает `-mtune` только с **именами моделей** (elbrus-8c, 8c2,
16c), не с ISA-версиями. Default mtune мы выпилили полностью —
`-march=elbrus-vN` уже задаёт ISA, LCC сам подбирает scheduling.

### `loginctl enable-linger` обязателен

Без `loginctl enable-linger $USER` после SSH disconnect systemd-logind
убивает процессы. Все наши scripts начинаются с этой команды.

### `PT_PIN_THREADS=1` несовместим с TP

ThreadPool маппит worker_id на абсолютные CPU ID. Ранки 1+ через
numactl --cpunodebind на разных NUMA — pin их thread'ов на ноду 0
рушит производительность с 11 до 1.4 tok/s.

---

## Заявленные числа vs воспроизводимость

| Число в README | Конфигурация | Воспроизводимо |
|----------------|--------------|----------------|
| qwen3-4B **10.9 tok/s** TP-4 | 8C2 lemur-1, 4 NUMA × 8 cores, libeml_mt, `-march=elbrus-v5`, `PT_Q8_SOA=1` | ✅ |
| qwen3-1.7B **17.1** TP-4 | ↑ | (тот же конфиг) |
| qwen3-8B **2.6** SP | 8C2, single-proc fallback | ✅ |
| qwen3-14B **1.5** SP | 8C2, large model → fits в memory | ✅ |
| ×6.3 vs llama.cpp 32t | 8C2, fair compare numactl interleave | ✅ |
| ~4-6 tok/s на 8C | E8C v4, `PT_Q8_SOA=1` через pmaddubsh path | 🟡 эстимат, валидируется 2026-05-20 |

---

## История fixes по ISA detection

| Дата | Commit | Fix |
|------|--------|-----|
| 2026-05-03 | (legacy) | `-mtune=elbrus-8c2` хардкод → guarded |
| 2026-05-19 | `8cbbf84` | T2: убрать -mtune default ПОЛНОСТЬЮ |
| 2026-05-19 | `7428592` | T7: build-elbrus.sh detect lcc++/l++ |
| 2026-05-20 | `d00f13f` | T-NEW: detect_e2k_march матч моделей `e2k-{16c,8c2,8cb,8c,4c,2c}` (раньше парсил только `e2k-vN`, default v4 на 8C2 → 11.4 → 5.0 tok/s регрессия) |
| 2026-05-20 | `b3db5ce` | (откат) v4 auto-disable Q8 SoA — заменено на real v4 SIMD path |
| 2026-05-20 | (current) | v4 SIMD path в q8_soa4_gemv через pmaddubsh/pmaddh/paddw |

---

## TODO — что задокументировать дальше

- [ ] Стек памяти на каждой ISA: TLB size, L1/L2 sizes, кэш-линии
- [ ] DDR bandwidth per-channel × количество каналов
- [ ] Минимальная версия LCC для каждого intrinsic (важно — [коллега] на
      v4 имеет 1.28, на v6 — 1.29.15)
- [ ] Производительность v3 после real-machine smoke (сейчас только
      теоретически)
- [ ] EML version compatibility matrix (1.x vs 2.x)
