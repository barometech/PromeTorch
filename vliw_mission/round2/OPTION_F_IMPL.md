# Option F — implementation status (2026-04-22)

Structural skeleton of Option F landed. Steps 1, 3, 5 done; Step 2 obviated by
existing GEMV; Step 4 deferred pending measurement.

## What's in this commit

### Step 1 — `all_gather_inplace` helper (`torch/distributed/ddp.cpp`, `ddp.h`)
- New `all_gather_inplace(float* data, int64_t per_rank_count)`.
- SHM backend: every rank deposits its slice at `shm_rank_slot(rank)`, barrier,
  all ranks pull all other slots in parallel (no reducer bottleneck). Same
  generation/arrived/departed protocol as `all_reduce_shm`, but symmetric.
- TCP fallback: star-topology gather + broadcast through rank 0.
- `kShmSlotSize` = 1 MB already accommodates qwen3:4b payloads
  (largest = inter_local = 2432 floats = 9.5 KB).

### Step 2 — N-slice GEMV (obviated)
Existing `cpu_quant_gemv(K, N, row_stride, weight_ptr)` already accepts an
arbitrary row range. Gather path passes `weight_ptr + row_start * row_stride`
and `N = H_local`. No new kernel needed.

### Step 3 — `PT_TP_GATHER=1` path in `forward_decode_cpu_tp`
- Gated via env at process startup (`use_gather` captured once).
- Branches at 4 sites:
  1. Skip `std::fill(attn_full_buf, 0)` — no zero-pad needed.
  2. After attention: `all_gather_inplace(attn_full_buf, q_dim_local)`.
  3. `attn_output`: N-slice GEMV from **replicated** weight via
     `numa_replica.get(node) + h_row_start * row_stride`; writes to
     `h_buf + h_row_start`, then `all_gather_inplace(h_buf, H_local)`.
  4. `ffn_down`: write silu local at `silu_full_buf + inter_offset`,
     `all_gather_inplace(silu_full_buf, inter_local)`, then N-slice GEMV
     on replicated ffn_down → `h_buf + h_row_start`, then
     `all_gather_inplace(h_buf, H_local)`.
- `init_tensor_parallel` reads same env — under gather mode uses **uniform**
  `inter_local = inter / nprocs` (legacy path stays non-uniform super-block
  aligned for K-slice correctness).
- K-slice allocations for `tl.q_attn_output` / `tl.q_ffn_down` are **skipped**
  under gather mode (saves ~115 MB on qwen3:4b Q4_K_M × 4 ranks).
- Replaces 2 AllReduce/layer → 4 AllGather/layer. Net-win claim depends on
  per-call latency reduction (Step 5) + absence of serial reducer.

### Step 5 — Futex-based barriers (`ddp.cpp`)
- Gated via `PT_DDP_FUTEX=1` env. Logged at SHM init: `[futex]` vs
  `[spin+yield]`.
- `shm_wait_counter_ge(&counter, threshold, use_futex)`: 1024-iter spin,
  then `FUTEX_WAIT | FUTEX_PRIVATE_FLAG` with the observed value. Peer
  increments + `FUTEX_WAKE` when threshold hit.
- `shm_wait_gen_advance(&gen, gen_val, use_futex)`: same shape, for the
  generation counter (plain `uint32_t`, uses `__atomic_load_n`).
- Replaces both `all_reduce_shm` and `all_gather_shm` wait loops.
- Targets the 375 μs median straggler wait (Agent 9) — expected reduction
  to ~30 μs per wakeup (futex wake path vs scheduler tick).

### Step 4 — Async double-buffer AllGather (deferred)
Not implemented. Requires layer-to-layer restructure (issue AllGather on
layer L's output; do layer L+1's RMSNorm + QKV concurrently; wait only
when layer L+1's attn_output needs the gathered attn_full). ~150 LoC
touching the layer loop structure. Defer until Step 3+5 measured on
Elbrus — if ≥14 tok/s, we're done; if still short, async is the next lever.

## Elbrus session plan

```bash
# Build new ddp.cpp + gguf_model.h on Elbrus
cd /home/$USER/promethorch/build_mt  # or wherever TP build lives
cmake --build . --target gguf_tp_bench -j

# Baseline (legacy AllReduce, spin+yield) — should match 6.5 tok/s from last session
PT_DDP_SHM=1 ./run_tp4_bench.sh --prompt "The sky is" --max_tokens 30

# Futex only (same path, faster barriers)
PT_DDP_SHM=1 PT_DDP_FUTEX=1 ./run_tp4_bench.sh --prompt "The sky is" --max_tokens 30

# Full Option F (gather + futex)
PT_DDP_SHM=1 PT_DDP_FUTEX=1 PT_TP_GATHER=1 ./run_tp4_bench.sh --prompt "The sky is" --max_tokens 30

# Correctness check: first 30 tokens must match legacy path bit-exactly
# (argmax decode, same seed). Gather semantics are equivalent to AllReduce-SUM
# because every rank's slice is disjoint — concat == sum-with-zero-pad.
```

## Risks / things to watch

- **Correctness**: per-layer numerics change (different FP32 accumulation
  order — AllReduce sums 4 partial H-vectors on rank 0; AllGather concats
  disjoint N/4 slices). Because current K-slice + AllReduce path also
  computes partial sums on each rank that then get summed, the arithmetic
  paths differ. **Compare argmax** across first 50 tokens vs legacy.
- **Build**: linux/futex.h not included — we declare `FUTEX_WAIT=0`,
  `FUTEX_WAKE=1`, `FUTEX_PRIVATE_FLAG=128` ourselves. `SYS_futex` taken
  from `sys/syscall.h`; falls back to `__NR_futex` if needed. If LCC
  lacks both, futex path no-ops back to `sched_yield`.
- **Memory**: gather mode skips K-slice allocation — net ~115 MB saved
  on qwen3:4b TP-4. Replication already accounted for in earlier agents.
- **Straggler**: if futex doesn't collapse the 375 μs median wait to
  ~30 μs (e.g., Elbrus scheduler behaves differently), Option F's 4
  AllGather/layer could regress vs 2 AllReduce/layer. Fallback: unset
  `PT_TP_GATHER` but keep `PT_DDP_FUTEX` — get futex wins without
  doubling barrier count.

## LoC accounting vs plan (570 LoC estimate)

| Step | Planned | Actual | Notes |
|------|---------|--------|-------|
| 1    | 80      | ~110   | + TCP fallback path |
| 2    | 60      | 0      | Existing GEMV sufficient |
| 3    | 200     | ~140   | Tight branching; reused buffers |
| 4    | 150     | 0      | Deferred |
| 5    | 80      | ~80    | Futex helpers + signal sites |
| **Total** | **570** | **~330** | Step 4 pending; Step 2 free |
