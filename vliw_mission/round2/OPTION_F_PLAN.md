# Option F — Full replicate + row-parallel every GEMV (design doc)

**Target: 17-22 tok/s** per Agent 9 analysis. Next-session implementation.

## Current TP state (Round 3)

All 8 weight groups replicated via `NumaReplica` (`gguf_model.h:1560-1582`):
`ffn_gate/up/down`, `attn_output`, `attn_q/k/v`, `output_weight`.

Current TP sharding in `gguf_model.h:499-502`:
- **Col-parallel (no AR within layer):** Q/K/V, gate, up — each rank gets 1/4 of output rows
- **Row-parallel (needs AR):** attn_output K-slice, ffn_down K-slice
- **Replicated (each rank full):** output_weight

Result: 2 AllReduces per layer + 1 AR for final logits = 73 AR/token.

## Agent 9 measured breakdown of 6.1 → 17-22 ceiling

Current TP-4 gives only 1.15× over 1-proc because:
- 42% of weight bytes are REPLICATED-read (attn_output 11%, ffn_down 19%, output 12%)
- Per-chip bandwidth drops only to 0.61× of total
- AllReduce 27 ms/token = 95% straggler-barrier (compute is only 5 μs;
  median wait 375 μs per call)

Option F structural change:
1. **Replicate everything** ✓ (done)
2. **Row-parallel ALL GEMVs** — every rank computes 1/N of output rows
   for Q/K/V, gate, up, attn_output, ffn_down, output_proj
3. **Replace AllReduce (sum) with AllGather (concat)** where possible
4. **Async AllReduce overlap with compute** of next op
5. **Tighter SHM barrier** — replace spin+yield with eventfd/futex

## Concrete implementation plan

### Step 1 — `all_gather_inplace` helper in `torch/distributed/ddp.cpp`

```cpp
// All ranks deposit their slice into their own SHM slot, then each
// rank memcpy's all N slices into its output buffer. Replaces
// AllReduce(sum) pattern when slices are DISJOINT (no summation needed).
void all_gather_inplace(float* data, int64_t per_rank_count);
```

~80 LoC. Uses existing SHM infra.

### Step 2 — Extend `cpu_quant_gemv` to accept slice offsets

```cpp
void cpu_quant_gemv_row_slice(
    ..., int64_t n_start, int64_t n_end, ...);
```

Each rank computes `[n_start, n_end)` rows. Already done for ffn_down/
attn_output K-slice; extend to N-slice for every GEMV.

~60 LoC refactor of existing dispatch.

### Step 3 — Rewrite `forward_decode_cpu_tp`

Current structure:
```
for each layer:
  QKV col-parallel (1/4 rows each) → local slice, no AR
  attention (compute local Q attending to local K/V)
  attn_output row-parallel → AllReduce(sum)
  gate, up col-parallel → local slice, no AR
  SiLU
  ffn_down row-parallel → AllReduce(sum)
```

New structure:
```
for each layer:
  RMSNorm (replicated input)
  QKV row-parallel (each rank computes 1/4 of output rows) → AllGather
  attention (local Q over full K/V via gather)
  attn_output row-parallel → AllGather
  residual + RMSNorm (replicated)
  gate + up row-parallel → AllGather
  SiLU (elementwise, no comm)
  ffn_down row-parallel → AllGather
  residual
```

This changes the COMMUNICATION pattern from sum-reduce to concat-gather.
Each op's output is 1/4 of its rows per rank, gathered to full on all.

Key insight: AllGather volume = (N-1)/N × output_size, same as AllReduce.
BUT gather doesn't need the reduction phase (compute + sync variance),
so we save the 95% straggler overhead identified by Agent 9.

~200 LoC rewrite of `forward_decode_cpu_tp`.

### Step 4 — Async AllGather via double-buffer

After issuing AllGather on layer L output, proceed to RMSNorm of layer
L+1 input using the ALREADY-GATHERED residual stream. The AllGather runs
in background. Wait only when the next GEMV actually needs it.

Requires one extra SHM buffer per layer + completion flag per rank.

~150 LoC.

### Step 5 — Replace bounded-spin with futex

Current SHM AllReduce uses `spin_until(... __sync_synchronize())` with
1024 spin iters then `sched_yield()`. Under sustained straggler load
this still burns cycles.

Replace counters with `int32_t` and use `syscall(SYS_futex, ..., FUTEX_WAIT,
...)` — kernel puts waiter to sleep, peer wakes with `FUTEX_WAKE`. Zero
spin, zero wasted cycles.

~80 LoC in `ddp.cpp`. Tricky because glibc doesn't expose SYS_futex as a
nice wrapper; have to call `syscall(__NR_futex, ...)` directly.

## Total scope
~570 LoC across 3 files (`ddp.cpp`, `gguf_model.h`, `cpu_quant_gemv.h`),
1 full rebuild, 1 TP-4 bench for validation.

## Risks
- Correctness: AllGather replacing AllReduce changes the output IFF the
  current implementation somehow relied on sum-semantics. For our design
  (each rank produces disjoint row slices), AllGather should be strictly
  equivalent. Validate with argmax match on first 50 tokens vs baseline.
- Futex portability: glibc version on Elbrus may or may not have
  `<linux/futex.h>`; check first.
- Memory: replication already at 4×, doesn't grow further.
- Perf: Agent 9 estimated +190-260% to hit 17-22 tok/s. If async AllGather
  and tighter barrier together only give +80-120%, we land at 11-13 tok/s.
  Still a big win but short of target.

## Success criteria
- TP-4 ≥ 14 tok/s on qwen3:4b Q4_K_M = 2× over current 6.5.
- Bit-exact argmax to 1-proc baseline on first 50 tokens (correctness).
- No regression at 1-proc (the code changes only touch TP path).
