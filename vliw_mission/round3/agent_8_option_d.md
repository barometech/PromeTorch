# Agent 8 (Round 3) — Option D resurrected: ship the K-slice that we already wrote and forgot to use

**Date:** 2026-04-25
**Brief:** Round 2 Agent 9 §4-5 proposed converting the three replicated weights
(`attn_output`, `ffn_down`, `output_weight`) to K-sliced row-parallel partial-sum +
AllReduce. **It was 90% implemented and 100% disabled by default.** This agent
reverse-engineers what is missing, exactly which lines bypass the slices, and
the smallest diff to ship +50% (5.3→9 tok/s) the way Round 2 promised.

**TL;DR:** the K-slice buffers `tl.q_attn_output` / `tl.q_ffn_down` ARE built
in `init_tensor_parallel` (gguf_model.h:4291-4298) and ARE used on the legacy
AllReduce path (4708-4736). But (a) the **gather path** (default in current
benches, `PT_TP_GATHER=1`) reads `layer.q_attn_output.cpu_data` (the mmap'd
full parent), and (b) **`q_output_weight` is not K-sliced at all** — every rank
reads the full 290 MB output head and only N-row-slices the COMPUTE. Wire D
properly = 3 surgical edits, ~150 lines.

---

## 1. What "Option D" actually proposes

From `vliw_mission/round2/agent_9_numa_aggregate.md` §4.2 table at line 287:

| Weight | Today | Option D | Bytes saved/tok/chip | tps delta |
|---|---|---|---|---|
| `attn_output` (11% of weight bytes) | replicated + AR | K-sliced + AR | 264 MB | +13% |
| `ffn_down` (19%) | replicated + AR | K-sliced + AR | 460 MB | +23% |
| `output_weight` (12%) | replicated (N-row split, but each rank still streams full K=H of every owned vocab row) | K-sliced + AR (1 extra AR/tok, not per layer) | 290 MB | +14% |

§4.1 establishes the unavoidable floor of 2 AR/layer (RMSNorm needs full hidden);
trying to eliminate AR is wrong. The right move is just **slice the weight K-dim
so each rank reads 1/N of the bytes per call, AR-sum the partial output**.

§5.3 puts the projected ceiling at 17.5-21.7 tok/s after also wiring async-AR
overlap (B) and tighter SHM barrier (C). D alone is **+50% to 9 tok/s**, the
single biggest cheap win and a prerequisite for F.

**Why it didn't ship in Round 2:** the K-slice infrastructure (allocations,
`cpu_quant_gemv_k_slice` kernel for Q4_K/Q6_K/Q8_0) was added — see
`gguf_model.h:4291-4298` and `cpu_quant_gemv.h:1507`. But the **gather variant**
(PT_TP_GATHER=1, the Option-F prototype that subsequent rounds defaulted to)
took a different path: it N-row-splits the **compute** while reading the **full
replicated weight**, mmap parent. That path never gained K-slicing for any of
the three weights. Net: in current TP-4 benches with gather mode on, all three
weights are read in full from mmap by every rank → exactly the 1.64×-instead-of-
4× per-chip bandwidth wall §1.3 documented.

---

## 2. Verification — where the gather path bypasses K-slices

Key file: `torch/io/gguf_model.h`. Three sites to fix:

### 2.1 attn_output (gather path, line 4630-4677)
- 4635: `w_ao_full = layer.q_attn_output.numa_replica.get(_node);`
- 4636: `if (!w_ao_full) w_ao_full = layer.q_attn_output.cpu_data;` ← **mmap parent**
- 4638-4640: pointer-offsets by `h_row_start * w_ao_stride` (N-row slice ONLY).
- 4661-4667: `cpu_quant_gemv` reads K=full q_dim of the row slice.

Each rank reads `q_dim × H_local` blocks in full (1/N rows × full K). Aggregate
bytes/token = N × (1/N rows × full K) = full weight, replicated read pattern,
even if the tl K-slice exists in DDR.

### 2.2 ffn_down (gather path, line 4791-4834)
- 4796: `w_fd_full = layer.q_ffn_down.numa_replica.get(_node_fd);`
- 4797: `if (!w_fd_full) w_fd_full = layer.q_ffn_down.cpu_data;` ← **mmap parent**
- 4799-4800: pointer-offsets by `h_row_start * w_fd_stride` (N-row slice).
- 4818-4824: `cpu_quant_gemv` reads K=full inter for the row slice.

Same pattern as attn_output. tl.q_ffn_down K-slice (built at 4295) sits unused
on this path.

### 2.3 output_weight (line 4915-4942) — never K-sliced
- 4933-4934: `w_slice = q_output_weight.cpu_data + row_start * row_stride` —
  N-row split only. K-dim is full H per rank.
- 4942: `all_reduce_inplace(tp_.logits_buf.data(), V)` over 600 KB.

`q_output_weight` is NEVER passed through `slice_k_blocks` in
`init_tensor_parallel`. There is no `tp_.q_output_weight_k_slice` field at all.

### 2.4 Numbers (qwen3:4b Q4_K_M, N=4)
Per token, gather mode per rank reads:
- attn_output: full 36 layers × 4096 × 2048 × Q4_K(4.5 bits/elem) = ~155 MB
- ffn_down: 36 × 2048 × 9728 × Q4_K = ~370 MB
- output_weight: 1 × 152064 × 2048 × Q4_K = ~84 MB
**Sum: 609 MB/rank/token** of "replicated" reads.

After D, each rank reads 1/4 of these = **152 MB/rank/token** saved on the
3 weights. At 12 GB/s local DDR that's 38 ms saved per token. Current 164 ms/tok
@ 6.1 tok/s → 126 ms/tok = **7.9 tok/s** (+30%). Adding the bandwidth headroom
freed for QKV/gate/up to land contention-free pushes that toward the agent_9
projection of **+50% / 9 tok/s**.

---

## 3. Concrete implementation plan (~150 LoC)

### 3.1 K-slice `output_weight` in init_tensor_parallel
**File:** `torch/io/gguf_model.h`
**Add:** new `TPSlicedWeight q_output_weight_k_slice` field on `GGUFTPConfig`
(near line 605 where existing TP fields live). Plus three int64s
`output_weight_k_start/k_end/k_local`.

**Change at 4291-4298:** after `slice_k_blocks(layer.q_ffn_down, ...)`, also
slice the model-level output head ONCE (outside the layer loop). Insert at
line 4299, before scratch buffer allocation at 4302:
```cpp
if (!gather_mode_setup && q_output_weight.valid && q_output_weight.cpu_data) {
    slice_k_blocks(q_output_weight, tp_.q_output_weight_k_slice,
                   tp_.output_weight_k_start, tp_.output_weight_k_end,
                   tp_.output_weight_k_local, "output_weight");
}
```
Cost: +1 alloc per rank of `V × (H/N × bytes_per_block_per_block_size)` =
~21 MB/rank. ~10 LoC.

### 3.2 Use `tl.q_attn_output` on the gather path
**File:** `torch/io/gguf_model.h:4619-4677`

The gather path's logical flow is: produce row-sliced attention output → all_gather
to build full attn_full_buf → row-slice GEMV on FULL replicated W_ao → all_gather
hidden. Replace this with K-slice-AR pattern (already proven on legacy path 4707-4738):

1. Drop the first `all_gather(attn_full_buf)` at 4642+4657. Each rank holds its
   q_dim_local-slice; attn_full_buf is no longer needed. Instead each rank's
   `attn_full_buf[q_off..q_off+q_dim_local]` IS the input to its K-slice GEMV.
2. Replace the 4661-4667 `cpu_quant_gemv` call with `cpu_quant_gemv_k_slice`
   on `tl.q_attn_output.cpu_data` — exactly the legacy path 4711-4718 body.
3. Replace the second `all_gather(h_buf, H_local)` at 4684+4704 with
   `all_reduce_inplace(h_buf, H)` — same as legacy path 4736.

Net effect on this section: 2 all_gather calls (H_local = 512 fp32 = 2 KB) → 1
all_reduce (H = 8 KB). Bytes-per-call up by 4×, but call count halves and we
shed 155 MB/tok of mmap-parent reads. **~25 LoC** (mostly deletion + reuse of
existing K-slice block 4708-4736).

### 3.3 Use `tl.q_ffn_down` on the gather path
**File:** `torch/io/gguf_model.h:4779-4865`

Same surgery at the second-half of the gather block:
1. The pre-existing local silu compute at 4783-4786 already produces silu_local
   in tp_.silu_full_buf at offset `inter_offset`. Drop the `all_gather` at 4803+4815.
   tl.q_ffn_down K-slice expects a compact silu_local of length tp_.inter_local —
   reuse the legacy path's `silu_local` stack vector at 4869-4873 OR write into
   a member buffer.
2. Replace the 4819-4824 `cpu_quant_gemv` with `cpu_quant_gemv_k_slice` on
   `tl.q_ffn_down.cpu_data` (same as legacy 4875-4882).
3. Replace the second `all_gather(h_buf, H_local)` at 4844+4864 with
   `all_reduce_inplace(h_buf, H)` (same as legacy 4900).

**~30 LoC.**

### 3.4 K-slice the output projection at line 4915-4942
1. Replace N-row split with K-slice for the output head:
   ```cpp
   if (use_quant_gemv_ && tp_.q_output_weight_k_slice.valid) {
       int64_t local_blocks = tp_.output_weight_k_end - tp_.output_weight_k_start;
       // Compact rank-local slice of x_buf[cur] for the K-range.
       // x_buf is full H — need x_buf[k_start*256 .. k_start*256 + k_local].
       const float* x_local = tp_.x_buf[cur].data() + tp_.output_weight_k_start * 256;
       std::memset(tp_.logits_buf.data(), 0, V * sizeof(float));
       cpu_quant::cpu_quant_gemv_k_slice(
           tp_.q_output_weight_k_slice.quant_type,
           tp_.q_output_weight_k_slice.cpu_data,
           x_local,
           tp_.logits_buf.data(),
           local_blocks,
           tp_.q_output_weight_k_slice.rows,         // = V
           tp_.q_output_weight_k_slice.row_stride_bytes);
       torch::distributed::all_reduce_inplace(tp_.logits_buf.data(), V);
   } else { /* existing fallback paths */ }
   ```
   Note Q4_K elems_per_block=256, so `k_start * 256` is the float-offset into
   x_buf; q_output_weight is keyed on H=2048 (qwen3:4b) which is multiple of
   `nprocs * 256 = 1024` — divisibility checked by `slice_k_blocks` already.

2. Drop the row_start/row_end logic entirely (and the per-rank N-zero-padding).
**~30 LoC.**

### 3.5 NumaReplica retirement (Round 3 Agent 3 §1 says it's unused in TP)
Agent 3 confirmed `NumaReplica` is consulted only in `forward_decode_cpu()`.
The TP gather path's `numa_replica.get(_node)` calls at 4635, 4796 currently
return nullptr and fall back to mmap parent (because `replicate_weights_for_numa`
is gated on `PT_NUMA_REPLICATE=1` and is meant for the 1-process path — see
gguf_model.h:1548). **Don't drop NumaReplica** — it remains useful for
`forward_decode_cpu()`. Just stop calling `numa_replica.get()` in the TP gather
path because Option D makes those reads disappear entirely. ~5 LoC of cleanup
(removing the prefetch hints at 4647-4655 and 4805-4813 — they were warming
the WRONG buffer anyway).

### 3.6 Memory delta
- **+** per-rank: `q_output_weight_k_slice` ~21 MB, `attn_output_k_slice` and
  `ffn_down_k_slice` already there.
- **−** per-rank: gather path no longer faults the mmap parent attn_output and
  ffn_down. Those parents stay mmap'd, but with N-rank cold reads gone the
  pages don't get pulled into N copies of L3.
- Net: trivially +21 MB/rank for output, free 0-30 MB/rank of mmap parent
  resident-set. **No measurable RAM cost.**

---

## 4. Test plan + correctness

1. **Build the existing legacy (non-gather) bench first.** Currently
   `PT_TP_GATHER=0` already exercises the K-slice path for attn_output and
   ffn_down (lines 4708-4736 + 4868-4901). Confirm it produces identical logits
   to 1-proc on a 5-token prompt. If it diverges, the bug is in
   `cpu_quant_gemv_k_slice` or the AllReduce, not in this plan — fix there first.
2. **Add output_weight K-slice (3.1 + 3.4) on legacy path.** Compare argmax of
   logits between 1-proc and TP-4 over 32 prompt tokens. Within fp32 tolerance.
3. **Port to gather path (3.2 + 3.3).** Run tok/s on the qwen3:4b benchmark on
   Elbrus 8C2. Expected jump: **6.1 → 8-9 tok/s**.
4. **Diff guard:** keep `PT_TP_GATHER=1` AND introduce `PT_TP_OPTION_D=1` to
   gate the new gather-K-slice variant for one bench cycle. If the new path is
   strictly better in both correctness and speed, default it on, retire the
   gather-AllGather variant.
5. **Profile breakdown:** with `PT_PROFILE_LAYER=1` (gguf_model.h:4382-4404)
   confirm `ao_ms` and `fdown_ms` drop ~25-50% (less weight bandwidth) and
   `allreduce_ao_ms`/`allreduce_fdown_ms` stay ~constant (same H-sized AR).

---

## 5. Files to touch — exact lines

| File | Lines | Action |
|---|---|---|
| `torch/io/gguf_model.h` | ~605 | Add `TPSlicedWeight q_output_weight_k_slice` + 3 metadata int64s on `GGUFTPConfig` |
| `torch/io/gguf_model.h` | 4299 (insert) | Call `slice_k_blocks` for `q_output_weight` outside layer loop |
| `torch/io/gguf_model.h` | 4619-4677 | Replace gather attn_output block: drop 1st all_gather, use K-slice GEMV on `tl.q_attn_output`, replace 2nd all_gather with all_reduce |
| `torch/io/gguf_model.h` | 4779-4865 | Same surgery for ffn_down on gather path |
| `torch/io/gguf_model.h` | 4915-4942 | Replace N-row output_weight split with K-slice GEMV + AllReduce |
| `torch/io/gguf_model.h` | 4647-4655, 4805-4813 | Delete now-defunct prefetch hints into mmap parent |

**Total: ~150 LoC.** No kernel changes (`cpu_quant_gemv_k_slice` already
supports Q4_K/Q6_K/Q8_0 at `cpu_quant_gemv.h:1507`). No new collectives
(`all_reduce_inplace` already in `torch/distributed/ddp.cpp:517`). No
NumaReplica reshuffling.

---

## 6. Why this is THE highest-EV change in Round 3 (vs other agents)

| Round 3 agent | Projected gain | LoC | EV (gain × certainty) |
|---|---|---|---|
| Agent 1 ThreadPool | +0.4 tok/s | ~400 | low |
| Agent 2 spec-decode | +1-2 tok/s, α-dependent | ~500 | medium-low |
| Agent 3 SHM weights | 0 (audit says skip) | 0 | n/a |
| Agent 5 SoA repack | +1-2 tok/s | ~600 | medium |
| **Agent 8 (this) — Option D** | **+3-4 tok/s** | **~150** | **high** |

This is the MOST math-supported, smallest-diff, highest-confidence win on the
table. The infrastructure was already written in Round 2. We just have to
actually USE it on the path that the bench actually runs.

**Ship D first. Then B (async AR overlap). Then C (tighter barrier).** Combined
projected per Round 2 agent_9 §7.1 roadmap: 5.3 → 17-22 tok/s on existing
hardware, no new code paths.

---

**Files referenced (absolute paths):**
- `C:\Users\paper\Desktop\promethorch\torch\io\gguf_model.h` (514-577, 4060-4344, 4408-4966)
- `C:\Users\paper\Desktop\promethorch\torch\io\cpu_quant_gemv.h` (1500-1562)
- `C:\Users\paper\Desktop\promethorch\torch\distributed\ddp.cpp` (all_reduce_inplace at 517)
- `C:\Users\paper\Desktop\promethorch\vliw_mission\round2\agent_9_numa_aggregate.md` (§4-5)
- `C:\Users\paper\Desktop\promethorch\vliw_mission\round3\agent_3_shm_weights.md` (§1, §6 confirmation)
