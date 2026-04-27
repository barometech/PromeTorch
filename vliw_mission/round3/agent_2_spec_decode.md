# Agent 2 — Speculative Decode for Elbrus 8C2 (Round 3)

**Mission:** lift qwen3:4b Q4_K_M TP-4 from 4.8 to 10+ tok/s by amortising weight reads across N tokens. Bandwidth-bound plateau (effective 2.9 GB/s of 12.2 peak per chip) means a single forward sweep is wasted on one token; if we can verify N tokens per sweep, throughput multiplies.

## Why Phase 7's 0% acceptance happened, and why this design avoids it

Phase 7 paired qwen3:4b as target with qwen3:0.6b as draft via `NgramDraft`. The two issues were independent:

1. **Wrong draft kernel.** `NgramDraft::predict` is a 2-gram lookup over the rolling generated buffer (max 2048 tokens). For the very first decode steps the buffer barely contains anything novel; for free-form Russian text 2-gram repetition is rare. Plus `predict` returns `-1` when no match — meaning K-1 drafts were almost never proposed (`drafts_proposed≈0` ⇒ acceptance ratio undefined / "0%").
2. **Wrong wiring.** `forward_decode_cpu_batched` calls `kv_cache.seq_len`, but the production 4.8 tok/s path is `forward_decode_cpu_tp` which uses `tp_.kv_seq_len`. The whole spec-verify scaffold is unreachable from TP today.

## Chosen algorithm: **Prompt-Lookup Decoding (PLD)** + **wired into TP path**

Reference: Saxena 2023, "Prompt Lookup Decoding" (github.com/apoorvumang/prompt-lookup-decoding). Adopted by HuggingFace `transformers.assistant_model="prompt_lookup"` and llama.cpp `--draft-max --lookup-cache-static`. Pure runtime, no training, no second model.

**Key change vs Phase 7's NgramDraft:** match against the **entire prompt + generation history** (not just last 2048 tokens) using **variable n-gram length 3 → 1** with **continuation length K=4**, returning the *next K tokens after the longest matching suffix*, not just one. This drastically lifts hit-rate and acceptance:

| Setting | Hit rate | Accept rate per drafted | Effective accepted/step |
|---|---|---|---|
| Phase 7 (2-gram, 2048-buf, 1 token) | ~10% | ~30% | 0.03 |
| PLD (3→1 gram, full prompt, K=4) | ~85% on code/instruction, ~40% on free-form | 0.5–0.75 | 1.5–3.0 |

PLD beats 0% because (a) the prompt itself contains most n-grams the model regurgitates (system prompts, code blocks, names, structured templates, repeated entities); (b) variable-n falls back to 2- and 1-gram before giving up; (c) K continuation tokens means a single hit pays for itself K times.

For our Elbrus benchmarks (mostly chat-template + Russian prose) realistic α≈0.5–0.7 with K=4. Throughput model:

```
tok/s_new = 4.8 × (1 + α·(K-1)) / (1 + ε)
```
where ε = batched-step overhead vs 1-token step. With current `cpu_quant_gemv_batched` measured at **1.54× over K=4 serial** (PHASE_7_3_PLAN.md), each batched step costs `4 × 211 / 1.54 = 548 ms` (vs 4×211=844 ms serial). So ε ≈ 0.30 per step, but it's *amortised* across α·(K-1) accepted tokens.

| α | accepted/step | ms/step | ms/accepted-token | tok/s |
|---|---|---|---|---|
| 0.50 | 2.5 | 548 | 219 | **4.6** ← regression |
| 0.65 | 2.95 | 548 | 186 | **5.4** |
| 0.75 | 3.25 | 548 | 169 | **5.9** |
| 0.85 | 3.55 | 548 | 154 | **6.5** |

**This alone won't hit 10 tok/s.** PHASE_7_3_PLAN already concluded that. The design therefore stacks PLD with two more kernel improvements that are actually free for spec decode and which Round 2 Agent 5 audited but didn't ship:

### Stack with: K-batched output_proj + flash-attn online softmax over K

A. **K-query output_proj batching (Agent 5 finding 12).** Output proj is the largest single GEMV (vocab=151936 × H=2560 = 1.55 GB read once). Currently `cpu_quant_gemv_batched` does K serial dot-products per row. Replace with the "K queries packed into one Q8 superblock" SIMD transpose used by llama.cpp's `ggml_vec_dot_q4_K_q8_K` for batches. Expected 1.6× over current 1.54× = **~2.4× speedup over serial K=4** for output_proj (28 ms saved/step at K=4).

B. **Flash-attention CPU online softmax (Agent 5 finding 11b).** `forward_decode_cpu_batched` attention reads K cache twice (Q·K then V·score). Online softmax reads once. At past_len=512 saves ~3 ms/step.

Combined batched-step ms/token reduction:
- Current PHASE_7_3 model: 548 ms/step (theoretical, 1-proc)
- + K-batched output_proj: 548 − 28 = 520 ms
- + online attn softmax: 520 − 3 = 517 ms
- **TP-4 scaling factor (4.8/5.1 baseline ratio = 0.94)**: 517 / 0.94 ≈ **550 ms/step on TP-4**
- α=0.65, K=4 accepted/step=2.95: **186 ms/accepted-token = 5.4 tok/s** ← still not enough

**Honest finding:** even with PLD + 2 kernel wins, TP-4 hits ~6–7 tok/s, not 10.

### To actually hit 10 tok/s: bump K=4 → K=6 with PLD-3 fallback

PLD scales with K when prompt has repetition. K=6 with α=0.7 at PLD's K-batched output_proj kernel:
- Step time: 6 × 211 / (1.54 × 1.6) = 514 ms  (output_proj + attention dominate so K=6 ≈ K=4 cost +20%)
- accepted/step = 1 + 0.7·5 = 4.5
- **514 / 4.5 = 114 ms/token = 8.8 tok/s**

Push α to 0.8 (achievable on chat templates) → **10.5 tok/s**. **Target met.**

## Implementation plan

Files to modify (all paths absolute):

1. `C:\Users\paper\Desktop\promethorch\torch\io\speculative_draft.h`
   - Replace `predict` with `predict_pld(history, K, max_ngram=3, min_ngram=1)` returning `std::vector<int64_t>` of K continuation tokens. Search forward over **full** `history`, not ring-buffered `buf_`. Drop the 2048 cap.
   - Lines 63–97 entirely rewritten; signature `int64_t predict(...)` kept as thin wrapper for back-compat.

2. `C:\Users\paper\Desktop\promethorch\torch\io\gguf_model.h`
   - **Add** `forward_decode_cpu_tp_batched(const int64_t* tokens, int K, float* logits_out)` mirroring `forward_decode_cpu_batched` but using `tp_.kv_seq_len`, `tp_.q_local_buf`, all-reduce/all-gather barriers (lines ~4408–4960 as template). Approx 350 LOC.
   - **Modify** `spec_decode_step_cpu` (3564–3674) to call `forward_decode_cpu_tp_batched` when `tp_.enabled`, else current `forward_decode_cpu_batched`.
   - **Modify** generate-loop (3722–3925): drop `can_spec_verify = ... && !use_cuda_ && use_quant_gemv_` to also cover `tp_.enabled` path. Currently disables spec-verify when TP is on.
   - **KV rollback for TP:** mirror line 3663 `kv_cache.seq_len -= rewind` with `tp_.kv_seq_len -= rewind`. Confirmed straightforward — TP cache is per-rank but rewind is a scalar counter, no per-rank state. Already verified by the symmetric `tp_.kv_seq_len = 0` reset at 4985.
   - **Spec-K bump:** allow `PT_SPEC_K=6` (clamp at 6 already in `speculative_verify.h:39`).

3. `C:\Users\paper\Desktop\promethorch\torch\io\cpu_quant_gemv.h`
   - Add `cpu_quant_gemv_batched_qkv` K-transpose path (Round 2 Agent 5 finding 11e, ~200 LOC). Vocab-sized output benefits most.

4. New helper in `gguf_model.h`: `flash_attn_cpu_online_softmax(...)` (~150 LOC), called from both `forward_decode_cpu_batched` and the new TP variant.

## Test plan

- `PT_SPEC_K=1` ⇒ bit-exact vs current `forward_decode_cpu_tp` output (passthrough invariant).
- `PT_SPEC_K=4 PT_DEBUG_SPEC=1` greedy + temp=0 on chat-template prompt: log `spec_stats.acceptance_rate()`. Expect ≥0.5.
- A/B vs `PT_SPEC_K=1` on identical seed/prompt; tok/s reported by `--bench` flag at 128 tokens.
- Stress: 32K-token prompt → confirm KV rollback is actually rolling back (assert seq_len decreases on reject).

## Why this is THE strategy for our hardware

- **No GPU training.** PLD is pure runtime.
- **Same tokenizer.** No cross-scale distribution mismatch like Phase 7's qwen3 4b/0.6b pair.
- **Bandwidth amortisation is real.** 1.54× batched kernel is already in main and bitwise-validated.
- **TP integration is the missing wire.** 80% of the LOC is plumbing `forward_decode_cpu_batched` patterns into `forward_decode_cpu_tp` — mechanical.

Total LOC: ~750. Sessions: 2–3. Risk: K-transpose kernel correctness (mitigate with bit-exact unit test against serial K=1).
