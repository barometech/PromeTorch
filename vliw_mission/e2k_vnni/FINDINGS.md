# E2K 8C2 INT8 VNNI — reverse-engineering findings (2026-04-25)

## Summary

MCST Elbrus 8C2 (ISA v5, LCC 1.29) **has `qpmaddubsh`** — a packed
(uint8 × int8) → int16 multiply-add-pairwise instruction. It is the only
INT8-oriented vector MAD available on 8C2; full VNNI (`qpidotsbwss`,
`qpbfdots`, `qpfmad` packed FMA) requires ISA v7+ (12C/16C).

## Micro-benchmark (1 core, K=2560 dot product, 2M iterations)

| Kernel | Time | GOPS | vs FP32 scalar |
|---|---|---|---|
| FP32 scalar   | 13.75 s | 0.74 | 1.0× |
| INT8 scalar   |  1.15 s | 8.94 | 12× |
| **INT8 `qpmaddubsh`** | **0.28 s** | **36.33** | **49×** |

Per-core INT8 MAD throughput on 8C2 ≈ **36 GOPS** (1.5 GHz, single
channel — VLIW could go higher if more packed MAD pipes are free).
Extrapolated: ~1.1 TOPS aggregate on 4-NUMA × 8-core Эльбрус 8СВ.

## Intrinsic probe results (v5 / 8C2)

**Available:**
- `qpmaddubsh(a, b)` — 16×(u8 × s8) → 8×i16 pairwise-add
- `qpmullw`, `qpsadbw`, `qpshufb`
- `pmaddh`, `pmaddubsh`, `psadbw`, `pshufb` (64-bit)
- Scalar `fmad`

**NOT available (error: "not supported for current cpu mode"):**
- `qpidotsbwss` / `qpidotsbwus` / `qpidotsbwuu` — INT8 dot product (VPDPBSSD analog)
- `qpidotshwss` / `qpidotshdss` — INT16 dot
- `qpbfdots` — BF16 dot product
- `qpfmad` — packed FP FMA

All of the above require `-march=elbrus-v7` (which doesn't link against
v5 objects), confirming these are post-8C2 additions.

## Implication for LLM inference

Current Q4_K_M decode on 8C2 = dequant → FP32 → scalar FMA. Profiler
says 82% of time is compute, 18% bandwidth.

If we rewrite Q4_K / Q8_0 GEMV to use `qpmaddubsh`:
- Compute portion × 49 (per microbench) → effectively zero
- Remaining bottleneck = pure memory bandwidth (18% today)
- **Projected ceiling: 20-30 tok/s TP-4** (vs 4.8 measured)

## Kernel design sketch

```c
// Q8_0 GEMV row: y[n] = sum_i(w[n,i] * a[i]) where w is int8, a is int8 (quantized)
// Per-row: read K int8 weights + K int8 activations, emit 1 float32.

static int q8_dot_vnni(const int8_t* w, const int8_t* a, int K) {
    v2di acc = {0, 0};                        // 4× int32 lanes
    // Fix-up for signed×signed via qpmaddubsh: shift activations to uint8
    // by adding 128. dot_unsigned_signed = dot_signed_signed + 128*sum(w).
    // (pre-computed sum_w per row, one int32 per row).
    for (int i = 0; i < K; i += 16) {
        v2di w_vec = *(const v2di*)(w + i);
        v2di a_vec = *(const v2di*)(a + i);      // after shift
        v2di p16 = __builtin_e2k_qpmaddubsh(a_vec, w_vec);  // 8× int16
        v2di p32 = __builtin_e2k_qpmaddh(p16, ones16);       // 4× int32
        acc = __builtin_e2k_qpaddw(acc, p32);
    }
    return reduce_i32(acc) - 128 * sum_w;
}
```

## Next steps

1. Q8_0 weight format probe (simplest: one int8 per weight + per-block fp16 scale)
2. Full GEMV kernel via `qpmaddubsh` with activation quantization on-the-fly
3. Integrate as env-gated `PT_Q4K_VNNI=1` path in `cpu_quant_gemv.h`
4. Benchmark on qwen3:4b Q4_K_M (requires either on-the-fly Q4→Q8 unpack,
   or offline repack of the model to Q8_0)

Probe source: `probe_qpmaddubsh.c` in this directory.
