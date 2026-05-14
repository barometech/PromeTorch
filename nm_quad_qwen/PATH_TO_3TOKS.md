# Путь к 3 tok/s Qwen3-4B на NM Quad

## Текущее состояние (2026-05-14, 33 commits в сессию)

- ✅ 12/12 ops BIT-EXACT
- ✅ E2E token generation: 139.5 sec/token = **0.0072 tok/s** (1 core subset)
- ✅ 16-core (4 chips × 4 cores) BIT-EXACT, throughput 2840 Q6_K rows/sec
- Projection real Qwen3-4B на 16 cores: ~0.0022 tok/s

**Gap до 3 tok/s: ×420 speedup**

## Векторы оптимизации (приоритет по impact × сложность)

### 1. ASM SIMD intrinsics для Q4_K/Q6_K block_dot (×4-8)
Файл: `/usr/local/rc_module_nmc4_toolchain/doc/rg_asm_ru.pdf` — NMC4 ASM reference.
NMC4 имеет vfpu (vector float), qpfmas (quad pack FMA). 
- Заменить scalar loop в q4k_block_dot/q6k_block_dot на vfpu
- Q-decode (4-bit unpack + dequant): использовать SIMD shuffles
- **Effort: 2-3 дня read docs + implement + verify**

### 2. Cluster-internal 4-core full_layer kernel (×3-4)
Текущий full_layer однопоточный. Нужно:
- Каждая GEMV: split rows между 4 cores cluster
- Spin-barrier через volatile EMI counter между ops
- Host launches single ABS на 4 cores cluster_id=0
- **Effort: 1 день**

### 3. 4-chip parallel — tensor-parallel split (×3-4)
Split каждая GEMV's M между 4 chips. Inter-chip через PL_WriteMemBlock.
EMI per-chip independent; нужен AllGather на каждом ops через host.
- Replicate K-side data (x) на 4 chips
- Each chip computes M/4 rows
- Host gathers, broadcasts to next op
- **Effort: 2 дня**

### 4. Fused multi-layer kernel (×1.5-2)
Сейчас 36 × PL_LoadProgramFile = 9 sec overhead. Fused kernel = 1 load.
Kernel cycles через 36 layer weights в EMI, returns final x.
- **Effort: 1 день**

### 5. On-chip KV cache (×1.5)
Сейчас KV cache в host или EMI per layer. Если живёт на chip между token iterations, экономит upload overhead.
- **Effort: 0.5 дня**

### 6. Speculative decoding (lossless, ×2)
Draft model (smaller Qwen) предлагает 3-5 tokens, target verifies в batch.
Если все 5 ok — 5 tokens за 1 forward.
- **Effort: 2 дня (need draft model port too)**

## Композиция

| Opt | Speedup | Cumulative |
|-----|---------|-----------|
| Baseline (1 core, projection real) | 1× | 0.0022 tok/s |
| + 4-chip TP | 3× | 0.0066 |
| + 4-core cluster | 3× | 0.020 |
| + fused multi-layer | 1.7× | 0.034 |
| + on-chip KV | 1.4× | 0.048 |
| + asm SIMD vfpu | 5× | 0.24 |
| + speculative decode | 2× | 0.48 |

Hypothetical max **~0.5 tok/s** с full software stack. До 3 tok/s нужны:
- Дополнительный hardware accelerator (vfpu peak performance throughput)
- Possibly Q4 → Q3 quantization (lossy! не lossless)
- Larger batch size (multi-token prefill)

## Path NEXT SESSION

1. Read `rg_asm_ru.pdf` для NMC4 SIMD ISA
2. Implement vfpu Q4_K block_dot (write asm or use intrinsics)
3. Verify bit-exact, measure speedup
4. Integrate в full_layer kernel
5. Add 4-core cluster split
6. Run E2E benchmark
