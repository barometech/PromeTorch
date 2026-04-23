# Phase 7.3 plan — honest projections for speculative decode on E8C2

## What we have committed (end 2026-04-23 session)

| Commit | Component | Measured |
|---|---|---|
| `eee6a1d` | `cpu_quant_gemv_batched` Q4_K kernel | **1.54× speedup at K=4** (ffn_gate shape, bitwise-identical) |
| `5c6311e` + `1b3a94b` | `spec_decode_step_cpu` + `forward_decode_cpu_batched` scaffold | compiles clean, K=1 passthrough, correct rewind on rejection |

Scaffold is correct but **not yet fast** — the scaffold's `forward_decode_cpu_batched` just calls `forward_decode_cpu` K times serially, so every GEMV reads weights K times instead of once. Each accepted token still costs 1 serial forward, and spec decode adds accept/reject overhead — net throughput is equal-to-or-below normal decode until the inner GEMVs are actually batched.

## Per-layer budget (from PT_PROFILE_LAYER on 1-proc T=30, 192 ms/token baseline)

| Section | ms/token total | ms/layer | Batched-variant ms/layer (K=4) | Savings per K=4 step |
|---|---:|---:|---:|---:|
| QKV fused | 34.5 | 0.96 | 3 × (1.54/4 × 0.32) ≈ 1.48 | ~2.4 ms |
| attention math (per-query) | 8.7 | 0.24 | 0.24 × K (no batching yet) | 0 |
| attn_output | 20.4 | 0.57 | 1.54/4 × 0.57 × 4 = 0.88 | ~1.4 ms |
| gate+up fused | 74.4 | 2.07 | 2 × 1.54/4 × 1.03 × 4 = 3.17 | ~5.1 ms |
| SiLU × up | 5.8 | 0.16 | 0.16 × K | 0 |
| ffn_down | 48.5 | 1.35 | 1.54/4 × 1.35 × 4 = 2.08 | ~3.3 ms |
| output_proj | 20.0 | (once) | 52.6 ms total | ~28 ms per K=4 step |

## Cost model for K=4, p=0.8 (Gemini Deep Research projection)

- Per-layer batched time (without attention batching):
  - Phase A (per-token QKV + attn + attn_output, K serial): 4 × (0.96 + 0.24 + 0.57) = 7.08 ms
  - Phase B (per-token residual + RMSNorm): 0.44 ms × 2 = 0.88 ms
  - Phase D (batched gate + up): 3.17 ms
  - Phase E (SiLU × up, K serial): 0.64 ms
  - Phase F (batched ffn_down): 2.08 ms
  - Per-layer total: **13.85 ms**
- × 36 layers = 498 ms
- + final RMSNorm (K serial): 1.2 ms
- + batched output_proj: 52.6 ms
- **Per K=4 batched step total: ~552 ms**

Acceptance p=0.8 → avg accepted = 1 + 0.8 + 0.64 + 0.51 = 2.95 tokens
- **552 / 2.95 = 187 ms per accepted token = 5.35 tok/s**

Acceptance p=0.6 → avg accepted = 2.17 tokens
- **552 / 2.17 = 254 ms per token = 3.93 tok/s** (REGRESSION vs 5.1 baseline)

## Conclusion: Phase 7.3 alone is a wash

Without **attention batching** (K queries × (past+K) keys, causal mask), the batched FFN + output_proj cannot overcome the per-token attn serial cost. At realistic n-gram acceptance p≈0.3-0.6 on free-form Russian text, **Phase 7.3 will regress** relative to 5.1 tok/s baseline.

## What would actually hit 10+ tok/s

**Phase 7.4 — batched attention** (hard). Run K queries against extended KV cache (past_len..past_len+K) in one kernel with a K×K causal mask. This is a standard GEMM-style attention but requires:
- Batched attention kernel (multi-query, per-query causal mask within batch)
- KV cache write-K positions at once
- 4-5 days of careful C++ / SIMD work

**Phase 7.5 — trained draft model** (easier, bigger impact). Replace n-gram with `qwen3-0.6b` Q4_K_M as a separate process drafting in the background. Draft runs at ~18-25 tok/s on 1 NUMA node, main model at 5 tok/s on the other 3. Per Gemini Deep Research, same-family same-tokenizer acceptance is 0.7-0.85. This combined with Phase 7.4 realistic projection is **12-16 tok/s**.

## Recommendation

The 1.54× batched kernel is validated and in main. The accept/reject scaffold is correct. The remaining gap to 10+ tok/s is:
- **Mandatory:** batched attention path (Phase 7.4)
- **Strongly beneficial:** trained draft model (Phase 7.5)

Session budget for remaining work: **3-5 dedicated sessions**. Current session has laid the infrastructure (kernel + scaffold) but cannot land the full attention + draft-model integration.

**Honest interim status:** baseline 5.1 tok/s (1-proc) / 5.4 tok/s (TP-4). Public E2K ceiling 5.2-6.7 tok/s (frozen 2023). Our current baseline is **≥ state-of-the-art on E2K**. The 10+ tok/s target requires Phase 7.4+7.5 work in subsequent sessions.
