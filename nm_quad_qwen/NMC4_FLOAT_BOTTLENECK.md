# NMC4 GCC Float Bottleneck — Discovered 2026-05-14

## Finding

`nmc-gcc -mnmc4-float -O3 -funroll-loops -ffast-math` НЕ inline'ит float arithmetic.

Каждое `a * b` компилируется в `delayed call FMul` (8 bytes, ~10 cycle wrapper).
Каждое `a + b` → `delayed call FAdd`.

## Evidence

```
$ nmc-objdump -d nmc_part.abs | grep "delayed call" | wc -l
Тысячи calls в gemv_q4k_gen() inner loop
```

`FMul` body (5c28-5c30):
```asm
sir = gr7
fp_pack_exp = sir
fpu rep 1 .packer = [--ar7] with .double <= .fixed_32
delayed return
fpu rep 1 [ar7++] = .packer
vnul
vnul
```

Hardware FPU исполняется но обёрнуто в function call → ~10× overhead per op.

## Impact на Qwen3-4B per layer

Q4_K block_dot: 256 weights × ~5 ops (decode + mul + add + scale + accumulate) = ~1280 ops/block.
10 blocks/row × 256 rows = 3.3M float ops per single Q4_K GEMV.
× 36 layers × multiple GEMVs = ~1 billion function calls per token!

## Speedup potential

Inline VFPU via asm:
- `fpu rep 32 .float vreg7 = vreg0 * vreg1 + vreg7` — does 32 mul+add в одной инструкции
- **5-10× speedup** for math-heavy inner loops
- Combined с loop unrolling и memory pipeline: potential ×10-20

## Why GCC doesn't inline

NMC4 backend `-mnmc4-float` mode emits library calls для всех float ops:
- Maintains compatibility с older NMC3 что не имел hardware FPU
- Wrapper conversion `.double <= .fixed_32` нужно для some operations
- GCC inliner не intelligent enough для vfpu pattern matching

## Path to fix

1. Manual inline asm для hot loops в q4k_block_dot, q6k_block_dot
2. Custom CFLAGS: `-mnmc4-float -funsafe-math-optimizations -mfloat-libcall=no` (если exists)
3. Custom builtin wrapper: `static inline float fmul_inline(float a, float b) { ... asm ... }`

## Estimate after fix

- Current: 4.4 sec per layer subset
- With vfpu inline: **0.5-1 sec per layer** (×5-10)
- E2E per token: 97 sec → **15-25 sec** (×4-6)
- × 16 cores parallel: **2-4 sec/token = 0.25-0.5 tok/s**
- Hardware bandwidth ceiling: 1.25 tok/s

## Required для 3 tok/s lossless

Hardware ceiling **физически 1.25 tok/s**. 3 tok/s невозможен на NM Quad lossless даже с perfect SIMD utilization.
