# Аудит 10: NM Card Mini + NM Quad backend status

**Дата:** 2026-05-20  
**Скоуп:** `c10/nmcard/`, `aten/src/ATen/nmcard/`, `nm_quad_qwen/`, `examples/nmcard/`, `test/cpp/test_nmcard.cpp`.

## 1. Резюме (TL;DR)

Два разных бэкенда NMC под одним брендом «NMCard», но архитектурно несовместимых:

| Бэкенд | Железо | Уровень | Состояние |
|---|---|---|---|
| **NM Card Mini** | NM6408 (1 чип, 16 NMC4 cores, 5 GB DDR3L, PCIe 2.0 x4) | интегрирован в ATen/dispatch (PrivateUse1) | DONE на эмуляторе, hardware path есть но minimal |
| **NM Quad** | 4 × NM6408 (64 NMC4 cores, 20 GB DDR, PCIe 3.0 x16) | **standalone** скрипты `nm_quad_qwen/`, в ATen НЕ интегрирован | infrastructure 12/12 BIT-EXACT, real speed 0.0072 tok/s |

## 2. Таблица: ops coverage (NM Card Mini)

| Категория | Hardware OP | Emulator | Tensor wrapper (NMCardDispatch) | Реальное железо callable? |
|---|---|---|---|---|
| matmul | OP_MATMUL=1 | да (Q16.16) | `mm/mv/bmm` | да (через DLL) |
| rmsnorm | OP_RMSNORM=2 | да | через emulator | да |
| softmax | OP_SOFTMAX=3 | да | через emulator | да |
| silu | OP_SILU=4 | да | `silu` wrapper | да |
| rope | OP_ROPE=5 | да | через emulator | да |
| attention | OP_ATTENTION=6 | да | через emulator | да |
| elem add/mul/sub | OP_ELEM_*=10/11/12 | да | `add/sub/mul` | да |
| gate_mul | OP_GATE_MUL=13 | да | через emulator | да |
| mul_scalar | OP_MUL_SCALAR=14 | да | `mul_scalar` | да |
| gelu | OP_GELU=15 | да | `gelu` | **только emulator** (комм. в NMCardOps.h:44) |
| layernorm | OP_LAYERNORM=16 | да | нет | нет wrapper |
| matmul_partial (multi-core) | OP_MATMUL_PARTIAL=22 | нет (single thread emu) | через NMCardMultiCore | да |
| backward ops (30-35) | declared в enum | да | через CPU fallback в NMCardOps.h | **нет hardware kernel** |
| sgd/adam (50/51) | declared | да | нет wrapper | **нет hardware kernel** |
| abs/sqrt/exp/log/sin/cos/tanh/sigmoid/round/log2/log10/tan | нет OP | нет (Q16.16 не реализован) | **scalar CPU loop в NMCardOps.h:49-119** | нет |
| reductions sum_dim/max/min/argmax/argmin/mean/dot | нет OP | sum/max только | scalar CPU loop в Dispatch | нет |
| comparisons eq/ne/lt/le/gt/ge | нет OP | да | через emulator | нет |
| addcmul/addcdiv | нет OP | нет | scalar CPU loop в Dispatch:641 | нет |

**Найдено:** ~20 «hardware» опкодов на стороне dispatcher, но ~25+ Tensor-level ops в `nmc_ops::` сваливаются в scalar `for`-loop на хосте даже когда тензор на `nmcard`. Это эмуляция-эмуляции, не device compute.

## 3. NM Quad path completeness (Qwen3-4B)

| Шаг pipeline | NMC kernel | Host driver | Bit-exact vs CPU? | Performance |
|---|---|---|---|---|
| Q4_K GEMV K=2560 | `nmc_q4k_gemv_fast.c` | `host_q4k_gemv.cpp` | да | 9.88 MMACs/sec |
| Q4_K GEMV 4-core M=32 | `nmc_q4k_gemv_4core.c` | `host_q4k_gemv_4core.cpp` | да (max_diff 4e-7) | 14.7× vs 1-core |
| Q4_K GEMV tile M=4096 | `nmc_q4k_gemv_tile.c` | `host_q4k_gemv_tile.cpp` | да (max_diff 0.17 fp32 noise) | 32 chunks |
| Q6_K GEMV M=128 | `nmc_q6k_gemv.c` / `_orig.c` | `host_q6k_gemv.cpp` | да (max_diff 0) | 253ms/1 core |
| Q6_K 16-core | `nmc_q6k_16core.c` / `_big.c` | `host_q6k_16core.cpp` / `_big.cpp` | да | 2840 rows/sec |
| RMSNorm K=2560 | `nmc_rmsnorm.c` | `host_rmsnorm.cpp` | да (3e-8) | 3.3ms |
| NEOX RoPE HEAD=128 | `nmc_rope.c` | `host_rope.cpp` | да (4e-7) | 3.3ms |
| Attention single-head | `nmc_attn.c` | `host_attn.cpp` | да (1.5e-8) | 55ms |
| SiLU + Softmax | `nmc_silu_softmax.c` | `host_silu_softmax.cpp` | да | 14.8ms |
| Step1 RMSNorm+Q | `nmc_qwen_step1.c` | `host_qwen_step1.cpp` | да (3e-8) | — |
| FFN substep | `nmc_qwen_ffn.c` | `host_qwen_ffn.cpp` | да (7e-8) | — |
| attn_full chain | `nmc_qwen_attn_full.c` | `host_qwen_attn_full.cpp` | да (5.96e-08) | 182ms |
| full layer (18 ops) | `nmc_qwen_full_layer.c` | `host_qwen_full_layer.cpp` | да (1.19e-07) | 5486ms (subset) |
| 36-layer chain | через host orchestrator | host loop | да (3.81e-05 noise) | ~200s subset |
| lm_head subset | `nmc_lm_head.c` | `host_lm_head.cpp` | да (M=128 max_diff=0) | ~500ms / 1 chunk из 1187 |
| argmax VOCAB=151936 | `nmc_argmax.c` | `host_argmax.cpp` | да | ~500ms |
| inline asm vfpu dot4 | `nmc_asm_dot4.c` (proof) | — | proof только | — |
| **End-to-end token** | composed | `host_qwen_full_layer.cpp` + host orchestrator | да | **139.5 sec/token (subset)** |

**Конфигурация:** N_HEADS_SUB=2 (vs real 32), M_FFN=1024 (vs real 9728), HEAD_DIM=128. **subset, не production Qwen.**

## 4. Sync / IPC механизм (NM Quad)

- DMA write race fix: **delay loop в каждом kernel'е** — `for (volatile int w = 0; w < 100000; ++w);` в начале `main()`. Подтверждено в 28 из 36 NMC kernel'ов. Workaround, не настоящая синхронизация.
- Inter-core sync: **EMI polling** — `volatile` глобалы в shared EMI на одном кластере. Все 4 core читают/пишут одни и те же linker-placed symbols.
- Inter-cluster (4-chip): **через host** — PL_WriteMemBlock на каждый chip, host AllGather, host broadcast в следующий ops.
- Нет PL_Sync барьеров используется (см. TODO в host_q4k_gemv_4core.cpp:171–174): «sequential IO_ServiceStart races against cluster EMI visibility».
- host_sync_ping.cpp — единственный файл с явным test sync ping механизма.

## 5. Floating-point bottleneck (документирован в NMC4_FLOAT_BOTTLENECK.md)

`nmc-gcc -mnmc4-float` НЕ инлайнит float ops — каждое `a*b` → `delayed call FMul` (~10 cycle wrapper). При ~3.3M float ops на Q4_K GEMV × 36 layers ≈ ~1 billion function calls/token. inline asm vfpu (`fpu rep 32 .float vreg = vreg*vreg+vreg`) даёт потенциал 5–10×, proof в `nmc_asm_dot4.c`, **не интегрирован в q*k_block_dot**.

## 6. Hardware ceiling vs target 3 tok/s

| Component | Numbers |
|---|---|
| EMI bandwidth (PL_WriteMemBlock практический) | ~5 GB/s |
| Qwen3-4B Q4_K_M размер | ~4 GB |
| Bandwidth-bound min/token | 0.8 sec → **1.25 tok/s теоретический max** |
| Target lossless | 3 tok/s — **физически невозможно** на NM Quad без weight reduction (см. NMC4_FLOAT_BOTTLENECK.md:64) |

## 7. NMCard backward / training (15 пунктов покрытия)

1. NMCardEmulator реализует `MATMUL_BACKWARD=30`, `SILU_BACKWARD=31`, `GELU_BACKWARD=32`, `SOFTMAX_BACKWARD=33`, `RMSNORM_BACKWARD=34`, `ROPE_BACKWARD=35`, `CROSS_ENTROPY=40`/`_BACKWARD=41`, `SGD_STEP=50`, `ADAM_STEP=51`.
2. На реальной карте есть только `*.abs` файлы (matmul_backward.abs, rmsnorm_backward.abs, silu_backward.abs, softmax_backward.abs, rope_backward.abs, gelu_backward.abs, cross_entropy.abs, sgd_update.abs, adam_update.abs) — НЕ интегрированы в NMCardHardware::dispatch_op.
3. `NMCardOps.h` exposes только `launch_silu` и `launch_relu` через emulator, **backward launchers'ов нет**.
4. Все scalar unary/binary/reduce ops (abs/sqrt/log/exp/sin/cos/tanh/sigmoid/comparisons/sum_dim/argmax) — **scalar host loop**, не идут на DSP даже на эмуляторе.
5. `NMCardDispatch.h` nmc_ops::mm() реально вызывает hardware path **только при is_available()** в `NMCardOps`-уровне, в LinearAlgebra default ATen.
6. `NMCardMultiCore` использует только `OP_MATMUL_PARTIAL`, остальные ops не имеют parallel варианта (gate/rms/silu single-core всегда).
7. `train_mnist_nmcard.cpp` есть, экзешник `nmcard_tests.exe` собран в `build_nmcard/`, но **нет artifacts** `train_mnist_nmcard.exe` в этой build dir — examples не собраны.
8. test_nmcard.cpp содержит 7 test functions (`test_device`, `test_q16_math`, `test_allocator`, `test_tensor_nmcard`, `test_emulator_ops`, `test_fixed_point_mode`, `test_tensor_dispatch`), CLAUDE.md заявляет 32-34 (несоответствие).
9. `NMCardHardware.h` определяет `DDR_END = 0x1FF00000` (≈500 MB) — **в 10× меньше** заявленных 5 GB DDR3L (под-документирован/неиспользован).
10. Q16.16 fixed-point режим (`NMCardMath.h`) реализован полностью (mul/div/sqrt/exp/log/sin/cos), но `use_fixed_point_` дефолт — TBD (нужно проверить `.cpp`). На карте формально работает, но real Qwen weights — fp16/Q4_K/Q6_K, не Q16.16.
11. **Mismatch архитектуры:** NM Card Mini path (ATen-integrated, dispatcher.abs, OP-based) ≠ NM Quad path (`nm_quad_qwen/` standalone, per-op `.abs` файлы, prebuilt monolithic kernels, `libnm_quad_load`).
12. Qwen3-4B port в `nm_quad_qwen/` **не использует** `aten/src/ATen/nmcard/` infrastructure. Стоит отдельно, runs на NM Quad host x86 (Linux, ssh paperclipdnb@93.182.6.134).
13. EMI message protocol: документация только в `ARCHITECTURE_ANALYSIS.md` (DDR polling vs ncl_hostSyncArray). Нет описания new NM Quad CommBridge / inter-chip protocol.
14. **Race condition** в 4-core launch (host_q4k_gemv_4core.cpp:171–174): sequential IO_ServiceStart внутри single thread → cores 1..3 occasionally видят stale data. SDK пример использует pthread-per-core — не портирован.
15. **PL_WriteMemBlock byte-address overlap** (QWEN_NMC4_PROGRESS.md:83): Wup (1.47M PL_Words) overwrote Wv в EMI. Workaround: re-upload Wv последним. Указывает на отсутствие proper DDR allocator на NM Quad (NMCardHardware::DDRAllocator существует только для NM Card Mini path).

## 8. Gaps to 3 tok/s

| Гэп | Effort | Gain |
|---|---|---|
| Q4_K/Q6_K block_dot через inline asm vfpu (32-wide FMA) | 2-3 дня | ×4-8 |
| 4-core cluster fused full_layer (внутри-cluster barrier через volatile EMI counter) | 1 день | ×3-4 |
| 4-chip tensor-parallel (host AllGather через PL_WriteMemBlock) | 2 дня | ×3-4 |
| Fused multi-layer kernel (eliminate 36× PL_LoadProgramFile = 9s overhead) | 1 день | ×1.5-2 |
| On-chip KV cache (skip per-layer host roundtrip) | 0.5 дня | ×1.4 |
| Speculative decoding (draft model, lossless) | 2 дня + draft model port | ×2 |
| **Кумулятив projection (PATH_TO_3TOKS.md)** | — | **до ~0.5 tok/s** |
| **Hardware ceiling (bandwidth-bound)** | — | **1.25 tok/s max lossless** |

**3 tok/s lossless физически невозможно** на NM Quad без либо (a) дополнительного hardware accelerator pathway, (b) Q3 quantization (lossy), (c) batch >1.

## 9. Next steps (приоритет)

1. **Implement vfpu inline asm в q4k_block_dot/q6k_block_dot** (5-10× immediate speedup, sole blocking item).
2. **4-core cluster fused full_layer** с in-EMI counter-barrier между ops (избавиться от race в 4-core launch).
3. **Достроить proper DDR allocator для NM Quad** (порт `DDRAllocator` из `NMCardHardware.h`) — fix PL_WriteMemBlock overlap once and forever.
4. **Интегрировать `nm_quad_qwen/`-pipeline в ATen** (новый `aten/src/ATen/nmquad/`) или явно зафиксировать, что это standalone tool.
5. **Поднять `train_mnist_nmcard.exe`** в `build_nmcard/` и прогнать e2e (сейчас не собран).
6. **Согласовать test counts** (CLAUDE.md: «32 tests», файл: 7 test functions / 34 TEST() блоков). Привести в соответствие.
7. **Декларировать честно** в CLAUDE.md: 3 tok/s **lossless невозможно** на текущем HW; реальный potential ~0.5 tok/s; для production либо Q3 (lossy), либо больший accelerator.
8. Удалить или пометить deprecated ~12 дубль-кёрнелов в `nm_quad_qwen/` (q4k_gemv vs q4k_gemv2 vs _fast vs _full vs _im vs _tile vs _4core — слишком много слабо отличающихся вариантов, мешает навигации).
