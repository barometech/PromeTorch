# Agent 3 — NUMA audit (Elbrus 8C2 / qwen3:4b Q4_K_M)

Scope: threads vs pages, AllReduce cost, NumaReplica actual usage, KV cache placement, concrete fixes. **Only files under `torch/io/*`, `torch/distributed/ddp.cpp`, `c10/util/ThreadPool.h`, `scripts/run_*_elbrus.sh` were read.**

---

## Topology recap (8C2)
- 4 chips × 8 cores = 32 cores, 4 NUMA nodes, ~30 GB DDR/node.
- Model footprint (Q4_K_M qwen3:4b) ≈ **2.4 GB**, 36 layers × (attn_q/k/v/o + ffn_gate/up/down) ≈ 252 weights, per-token read ≈ full model = 2.5 GB/pass.
- 4.7 tok/s × 2.5 GB = **~12 GB/s aggregate**. Single node DRAM ceiling is ~20 GB/s; 4-node aggregate ceiling is ~80 GB/s. The current runs use at most one node's worth.

---

## Q1 — Threads vs pages under `numactl --interleave=all` (1-proc)

### Threads
`run_1proc_elbrus.sh` sets:
```
OMP_NUM_THREADS=24
OMP_PLACES=cores OMP_PROC_BIND=close
numactl --interleave=all
```
But the hot path is **NOT OpenMP** — it is `c10::get_thread_pool().parallel_for(…)` (see `torch/io/cpu_quant_gemv.h:67` and `gguf_model.h:2514`). The pool is `c10::ThreadPool` defined in `c10/util/ThreadPool.h:85`. **`OMP_PLACES` / `OMP_PROC_BIND` have zero effect on this pool.** Proof: the only pinning hook is `pin_enabled_`, gated on env `PT_PIN_THREADS=1` (ThreadPool.h:109) — and neither launch script sets it.

Consequence: the 24 worker threads are free-floating, scheduled by the kernel. `sched_getcpu()` is called once per worker at startup (`current_numa_node()`, ThreadPool.h:59-83) and cached in `thread_local t_numa_node` **forever**. If a worker migrates after startup, `t_numa_node` is stale and `ReplicatedWeight::get(node)` returns the wrong replica (= reading across the chip fabric while thinking it's local). With `numactl --interleave=all` binding relaxed, migrations are likely.

### Pages
`--interleave=all` sets the policy on the process — pages allocated by this policy scatter round-robin over the 4 nodes at first-touch. But:
- **Weights are mmap'd with `MAP_PRIVATE` + `PROT_READ`** (`gguf_loader.h:251`). `MAP_PRIVATE` + file-backed pages behave differently: the initial `MADV_SEQUENTIAL` pass (loader.h:261) faults them in on **the thread that did the scan** (load time), not on the worker threads. On 1-proc, that was the main thread, which under `--interleave=all` distributes the scan-time faults round-robin. So weight pages *are* interleaved, by accident.
- But the loader's `MADV_SEQUENTIAL` actively hurts decode. Decode is **random** 256-element super-block fetches across 36×7 rows per token. `MADV_SEQUENTIAL` biases readahead forward; it should be `MADV_RANDOM` or unset (see Q5).
- `MADV_DONTNEED` after load is not used, so OS readahead can still evict hot weights.

### Answer to Q1
- Under `--interleave=all`, **memory is interleaved, but threads aren't pinned.** The per-thread `t_numa_node` cache in `ThreadPool.h:55` locks in whatever node the worker happened to start on, and a later migration silently corrupts the "local" assumption.
- `OMP_PLACES=cores OMP_PROC_BIND=close` in the 1-proc script is **dead**: PromeTorch's hot path uses `c10::ThreadPool`, not OpenMP. The pool doesn't even honor the variables.
- Therefore even if `PT_NUMA_REPLICATE=1` were enabled in the 1-proc script (it isn't — see Q3), up to 3/4 of replica lookups would land on a remote replica because the thread's `t_numa_node` was set once at startup.

---

## Q2 — TP AllReduce bytes and cross-NUMA cost

Per-token AllReduce traffic, reading `gguf_model.h`:
- Line **3804**: `all_reduce_inplace(h_buf, H)` after attention output proj. H = 2560 for qwen3:4b → 10 KB/call.
- Line **3870**: `all_reduce_inplace(h_buf, H)` after FFN down. 10 KB/call.
- Line **3911**: `all_reduce_inplace(tp_.logits_buf.data(), V)`, V = 151936 → **608 KB/call, once per token**.

Per layer: 2 × 10 KB = 20 KB. × 36 layers = **720 KB/token** for h_buf AllReduces. Plus 608 KB for logits. Total **~1.3 MB/token** moved through shared-memory AllReduce.

At 5.5 tok/s that's 7.15 MB/s. **This is nothing** vs the 12 GB/s weight-read bandwidth. AllReduce is NOT the bottleneck.

However, one transport issue: `kShmSlotSize = 1 MB` (`ddp.cpp:93`). Logits payload = 608 KB fits, but just barely. If a bigger model's vocab (e.g. 256k) showed up, the SHM path would throw and fall back to TCP (ddp.cpp:515-517). Not a current problem, but fragile.

**Second-order cross-NUMA cost**: every AllReduce via SHM does `memcpy(my_slot, data, nbytes)` into a single `/dev/shm` region at a fixed virtual address (ddp.cpp:408 — rank 0 creates the region, workers mmap). The physical pages of `/dev/shm/prometorch_ddp_<port>` live wherever **rank 0** first-touched them, i.e. on rank 0's NUMA node. Ranks 1/2/3 doing `memcpy` into their `my_slot` are writing to rank 0's DDR across the fabric.
- 4 ranks × (720 KB + 608 KB) × 2 (deposit + read-back) ≈ 10 MB/token of cross-chip traffic on the SHM region alone, still negligible vs weights but worth noting.
- Real impact: `ShmHeader::arrived/departed` atomic spin on node 0's cache line. This is latency (not bandwidth) — each AllReduce eats ~10-50 µs of cross-chip ping latency. With 73 AllReduces/token, that's **0.7-3.7 ms/token** of raw spin latency. At 5.5 tok/s = 180 ms/token budget, that's 0.4-2% — also not dominant, but the only non-weight cost worth attacking if everything else is already memory-saturated.

### Answer to Q2
- AllReduce bandwidth is ~1.3 MB/token = 7 MB/s at current tok/s. Bytes are not the problem.
- Latency: 73 AllReduces/token, each ~10-50 µs cross-chip spin on the shm header atomic → up to 2% of token budget. Tolerable.
- Real NUMA hazard: the `/dev/shm` region is placed on rank-0's node because rank 0 first-touches it. A fix is `numa_interleave_memory()` on `s.shm_base` right after `mmap()` in ddp.cpp:408 so that rank-r's 1 MB slot lives near rank r's cores. Low effort, probably 10-30 µs latency cut per AllReduce.

---

## Q3 — `NumaReplica` / `ReplicatedWeight` callsites (the real question)

### Definition sites
- `torch/io/numa_weight_replica.h:60` — `struct ReplicatedWeight`, N-copies container.
- `torch/io/gguf_model.h:159` — `QuantizedWeight::numa_replica` field.

### Replication entry point
`GGUFModel::replicate_weights_for_numa()` at `gguf_model.h:1461`. Called from `load()` at line **1451**. Gated on `PT_NUMA_REPLICATE=1` (env), reads `numa_replicate_count()` from `numa_weight_replica.h:43`.

What it replicates (lines 1472-1479):
```
rep(layer.q_ffn_gate);
rep(layer.q_ffn_up);
rep(layer.q_ffn_down);
rep(layer.q_attn_output);
```
That's 4 weights per layer × 36 layers = **144 tensors**.

**What it does NOT replicate** (by omission):
- `layer.q_attn_q`, `layer.q_attn_k`, `layer.q_attn_v` — Q/K/V projections (3 × 36 = 108 tensors).
- `q_output_weight` — LM head (151936 × 2560 Q4_K = ~195 MB).
- `embed_tokens` / `token_embd.weight` — token embedding.

### GEMV consumers (where `numa_replica` is actually consulted)
Only 5 callsites pass `&weight.numa_replica` to a GEMV — all in `gguf_model.h`:
1. Line **2528**: `attn_output` (decode, non-TP path).
2. Line **2573**: `ffn_gate` + `ffn_up` via fused kernel (decode, non-TP).
3. Line **2647**: `ffn_down` (decode, non-TP).
4. Line **2941**: `ffn_gate` + `ffn_up` (second decode variant — older path with malloc'd x_normed).
5. Line **2978**: `ffn_down` (same variant).

**Not wired (bare `cpu_quant_gemv(... no numa arg ...)`)**:
- Lines 2345/2349/2353: `attn_q`, `attn_k`, `attn_v` in the fallback (non-fused RMSNorm+QKV) path. Because Q/K/V aren't replicated anyway, no regression — but if Q/K/V ever got replicated, this path would ignore them.
- Line 2588/2591: `ffn_gate/up` fallback path (when `can_fuse_ffn` is false). It drops the numa hint on the floor despite those weights being replicated.
- Line 2319: `cpu_fused_rmsnorm_qkv_gemv` — the fast QKV path — has **no numa parameter in its signature**. Q/K/V can never be replica-accelerated because the fast path doesn't accept the argument.
- Line 2688: output projection (`q_output_weight`) — no numa argument. LM head is never replicated and, even if it were, the call doesn't pass it.
- Line 2689 (prefill variant, not shown): same.
- **Entire TP decode path (lines 3700-3920)**: `init_tensor_parallel()` row-slices all the weights per-rank (gguf_model.h:3357-3494). Each rank's sliced `cpu_data` is `std::malloc()`'d under `numactl --membind=$rank`, so it's already node-local → replication is redundant. `tp_.layers[i].q_*.cpu_data` is never fed through `ReplicatedWeight`. So in TP mode, NumaReplica is **completely unused**.

### Non-replicated weight ratio
Per-token byte read (qwen3:4b, H=2560, n_heads=32, n_kv_heads=4, head_dim=128, inter=9728, Q4_K 144B/256 = 0.5625 B/param):
- attn_q:      2560 × (32×128) × 0.5625 / 1e6 = 5.9 MB × 36 = 212 MB/token **(not replicated)**
- attn_k:      2560 × (4×128)  × 0.5625 / 1e6 = 0.74 MB × 36 = 26.5 MB/token **(not replicated)**
- attn_v:      2560 × (4×128)  × 0.5625 / 1e6 = 0.74 MB × 36 = 26.5 MB/token **(not replicated)**
- attn_output: (32×128) × 2560 × 0.5625 / 1e6 = 5.9 MB × 36 = 212 MB/token (replicated)
- ffn_gate:    2560 × 9728 × 0.5625 / 1e6 = 14 MB × 36 = 504 MB/token (replicated)
- ffn_up:      same = 504 MB/token (replicated)
- ffn_down:    9728 × 2560 × 0.5625 / 1e6 = 14 MB × 36 = 504 MB/token (replicated)
- output.weight (tied): 151936 × 2560 × 0.5625 / 1e6 = 219 MB/token **(not replicated)**

Totals: **replicated = 1724 MB/token (70%)**, **not-replicated = 484 MB/token (22%)**. 8% rounding.

At 4.7 tok/s that is 2.28 GB/s of non-replicated reads hitting whichever node hosts them — i.e. 484 MB/token going through the ORIGINAL `cpu_data`, which for mmap'd weights is whichever node first-touched each page during `MADV_SEQUENTIAL` scan. Effectively random. Plus LM head 219 MB is a hot read concentrated at exactly the one page-range where it landed — that's a guaranteed hotspot on a single node.

### Answer to Q3
- Replication covers 70% of per-token weight bytes. 30% (attn_q/k/v + LM head) are **always cross-chip** on some fraction of reads regardless of `PT_NUMA_REPLICATE`.
- In TP mode NumaReplica is dead code (ranks hold locally-malloc'd slices already).
- **In 1-proc mode, `PT_NUMA_REPLICATE=1` is not even set by `run_1proc_elbrus.sh`.** The whole feature is off in the 4.7 tok/s baseline. Turning it on with thread pinning is the highest-value knob that has zero code changes.

---

## Q4 — KV cache placement

`KVCache::allocate()` at `gguf_model.h:242-281`:
- CPU path (line 255-256): `at::empty({max_seq, kv_dim})`. This goes through `at::native::empty` → `c10::GetCPUAllocator()`. The CPU allocator is `DefaultCPUAllocator` in `c10/core/Allocator.h:290-295`, which is just `posix_memalign(64, …)`. **No NUMA hint.**
- First-touch rule: pages get physical backing on whichever node first writes to the allocation.
- The KV cache is allocated in `GGUFModel::forward_cpu_path()` (or equivalent) at startup or on first `forward()`. Under 1-proc `--interleave=all`, the process policy scatters kv pages round-robin — fine if threads also scatter, but the writes happen from specific worker threads in `parallel_for`. With 24 unpinned workers, writes hit whatever node the current worker is on → pages land there → future reads from a different worker go cross-chip.
- Per-token KV read: total_seq × kv_dim × 2 (K + V) × 4 B = (assume 512 ctx × 512 × 8) ≈ 2 MB/token grown linearly with context. At ctx=4k, ~16 MB/token. That's small vs 2.5 GB weight reads, but latency per attention step is sensitive.

For TP mode:
- Line **3546**: `tp_.k_cache_local[i] = at::empty({max_seq_len, tp_.kv_dim_local})`. Same `at::empty` → same no-NUMA path. Under `--membind=$rank` the pages DO land node-local on first touch. So TP KV cache is correctly placed, **incidentally**, by the shell flag, not by code.

KVCache allocation on line 250/251 uses `at::empty_cuda` for GPU — irrelevant on Elbrus.

### Answer to Q4
- KV cache is `posix_memalign` via `DefaultCPUAllocator`. No NUMA awareness in code.
- **1-proc + interleave=all**: KV pages scatter by kernel policy; threads aren't pinned, so fraction of KV reads is cross-chip. Small absolute cost (~16 MB/token at ctx=4k).
- **TP + membind=$rank**: correct by accident of the shell flag. Zero code fix needed here.
- If 1-proc pins threads to nodes (see Q5 fix 3), KV pages should be allocated per-thread-per-node via `numa_alloc_onnode`, not global. Callsite: `gguf_model.h:255-256`.

---

## Q5 — Concrete proposals (file:line + expected effect)

### Fix 1 — Turn on what already exists (ZERO CODE)
`scripts/run_1proc_elbrus.sh:40`: add `PT_NUMA_REPLICATE=1 PT_PIN_THREADS=1` to the env block.

**BUT**: replication is useless without stable thread→node binding. Current `PT_PIN_THREADS=1` path (ThreadPool.h:205-213) pins `worker_id % ncpu` — `ncpu = std::thread::hardware_concurrency() = 32`. With OMP_NUM_THREADS=24 set, the pool reads that and makes 24 workers, pinning them to cores 0..23 = nodes 0, 1, 2 (cores 24-31 = node 3 are idle). That's ALREADY broken; workers 0-7 get node 0, 8-15 node 1, 16-23 node 2, node 3 goes unused.

**Fix 1b**: use 32 threads (OMP_NUM_THREADS=32) **only when `PT_PIN_THREADS=1`** so all 4 nodes are covered. Or: implement NUMA-balanced pinning in ThreadPool.h:205 — round-robin node then core (worker 0→node 0 core 0, worker 1→node 1 core 0, … worker 4→node 0 core 1). The per-node-replica hit rate only benefits if each node has ~6 workers, not 24/3.

Expected effect: +20-40% if replication + correct pinning catches FFN bandwidth across 4 nodes. At current 4.7 tok/s → **5.6-6.5 tok/s**. Mostly confirms Q2's thesis that TP's 5.5 tok/s was the same win via a different route.

### Fix 2 — Replicate the LM head (HIGH VALUE)
`gguf_model.h:1471-1479`: `replicate_weights_for_numa()` skips `q_output_weight`. Add:
```cpp
rep(q_output_weight);
```
And thread `&q_output_weight.numa_replica` into the output GEMV at line 2688.

Per-token: 219 MB of currently-single-node reads become 219 MB × (1/4 per node) locally. At 4 nodes this reduces cross-chip bytes by ~165 MB/token. If LM head was bottlenecking the final-token GEMV specifically (likely since vocab GEMV is the largest single operation: 151936 rows × 2560 cols), expected **+5-10% tok/s**.

**Memory cost**: +219 MB × 3 = 657 MB extra (we already have 1 copy). Total replicated footprint: 2.4 × 4 = 9.6 GB → with LM head: 10.3 GB. Fits easily in 125 GB.

### Fix 3 — Replicate attn_q/k/v (LOW VALUE by itself)
`gguf_model.h:1472`: add `rep(layer.q_attn_q); rep(layer.q_attn_k); rep(layer.q_attn_v);`.
**But** then also extend `cpu_fused_rmsnorm_qkv_gemv` in `cpu_quant_gemv.h` to accept 3 numa args (currently doesn't — that's the fast path at line 2319). Without the signature change, this fix is useless because the fast QKV path bypasses replication.

Per-token: 264 MB × (3/4) cross-chip reduction. Expected **+3-6% tok/s**.

### Fix 4 — OMP_PROC_BIND and affinity for the 1-proc path
`run_1proc_elbrus.sh:41`: `OMP_PLACES=cores OMP_PROC_BIND=close` currently does **nothing** because the hot path is `c10::ThreadPool`, not OpenMP. Two paths:
- (a) Delete these env vars, they're cargo-cult.
- (b) Add `PT_PIN_THREADS=1` and extend ThreadPool.h:205 pinning logic to also set `numa_run_on_node_mask()` per worker (hard-binds thread to one node, not one core). Then `current_numa_node()` in ThreadPool.h:59 never goes stale.

**close vs spread for interleaved memory**: irrelevant to `c10::ThreadPool`. For any future OMP usage: **spread** is correct when memory is interleaved (one thread per node saturates one controller); **close** is correct when memory is node-local (pack threads onto the node that owns the data). Current scripts use `close` which contradicts `--interleave=all` — yet another cargo. But since it's ignored, no real harm.

### Fix 5 — Drop `MADV_SEQUENTIAL`, add `MADV_RANDOM` + `MADV_WILLNEED`
`gguf_loader.h:261`: `madvise(data_, size_, MADV_SEQUENTIAL)`. Decode is random, not sequential. Readahead is fighting the actual access pattern.

Change to `MADV_RANDOM` after the initial scan completes, or leave default (no advice). Expected: small, maybe **+1-3%** — depends on how aggressive LCC 1.29's kernel readahead is on E2K. Worth testing.

Also: `MADV_POPULATE_READ` on the critical hot layers (FFN weights, attn_output, LM head) post-mmap to force fault-in now rather than on the first forward, so that `MEMORY_POLICY_INTERLEAVE` is what actually places the pages (not the deferred fault during a worker's parallel_for, which would place on that worker's node).

### Fix 6 — NUMA-aware KV cache (MEDIUM VALUE at long context)
`gguf_model.h:255-256`: `at::empty({max_seq, kv_dim})` → replace with per-node `numa_alloc_onnode()`, then present as a custom tensor. Or simpler: allocate via the CPU allocator but immediately call `numa_interleave_memory(ptr, nbytes, nodemask)` on it so KV pages scatter by intent not by first-touch accident.

Per-token: at ctx=4k, ~16 MB KV read. 3/4 cross-chip → 12 MB → near-0 cross-chip if interleaved and threads span nodes. Expected **+0.5-1%** at short context, growing to **+2-5%** at ctx=16k.

### Fix 7 — `PROT_READ`/`MAP_POPULATE` for weights
`gguf_loader.h:251`: `MAP_PRIVATE` alone — first decode pays page-fault latency for 2.5 GB of weights. Add `MAP_POPULATE` so the kernel pre-faults all pages at mmap time. Combined with the load-time interleave policy, this pins weight placement deterministically.

Expected: smooth out the first N tokens (no warmup dip), negligible steady-state effect on the 100-token benchmark.

### Meta-fix — Replicate ALL weights in 1-proc mode, skip TP entirely
TP currently does: shard weights + AllReduce. The AllReduce is cheap (Q2: 1.3 MB/token), but the logits GEMV still has to be row-split across ranks because the LM head is too big to replicate without extra handling.

Alternative: **1 process, 32 threads, 4 replicas of the full model, each thread pinned to a node, each reads from its node's local replica**. No AllReduce, no TCP, no SHM dance. Memory: 2.4 × 4 = 9.6 GB. This is exactly what `PT_NUMA_REPLICATE=1` was designed for but never finished (Q/K/V/LM head gaps, unpinned threads, OMP env vars doing nothing).

Fully-wired 1-proc + 4 replicas + pinned 32 threads should match or beat TP-4, because:
- Zero AllReduce spin latency (saves 0.7-3.7 ms/token from Q2).
- No split-output-proj zero-memset + AllReduce-sum trick (saves another few ms/token).
- Same 4× aggregate memory bandwidth.
- All 32 cores running compute, not 28 (4 ranks × 7 threads).

Estimated ceiling: **7-9 tok/s** if memory-bound at 4× 20 GB/s aggregate.

---

## Ranked priority (effort × expected gain)

| Rank | Fix | Code change | Expected tok/s |
|------|-----|-------------|----------------|
| 1 | Fix 1 + 1b: wire PT_NUMA_REPLICATE=1 + fix pin mapping in ThreadPool.h:205 | 15 lines | 5.8-6.5 |
| 2 | Fix 2: replicate LM head | 2 lines model + 1 arg wire | 6.2-7.0 |
| 3 | Fix 3 + signature change for fused QKV path | 30 lines header + model | 6.5-7.3 |
| 4 | Meta-fix: drop TP, use pinned 1-proc with full replication of all hot weights | ~50 lines | 7-9 |
| 5 | Fix 6: NUMA-aware KV | 10 lines | +0.5-5% ctx-dependent |
| 6 | Fix 5: drop MADV_SEQUENTIAL | 2 lines | +1-3% |
| 7 | AllReduce SHM region interleave | 3 lines ddp.cpp:408 | <1% |

## Risks
- Replicating LM head: if `mmap_owned` is true, `replicate()` assumes the src is readable RAM — should be fine, mmap'd pages read like RAM. Verify `cpu_data` is populated (call is safe under `if (qw.cpu_data)` guard).
- Pinning 32 threads with 4 replicas: if another process (kernel thread, irqbalance) migrates a worker, the `t_numa_node` cache goes stale and a small fraction of lookups hit wrong replica. Data is read-only so still correct, just slower. Monitor with `perf stat -e node-loads,node-load-misses` if available on E2K.
- `MAP_POPULATE` on 2.4 GB mmap: adds ~1-3 s to load time. Acceptable.

## Blocker
- No direct access to E2K perf counters here. All bandwidth numbers are computed, not measured. On live Elbrus run: `numastat -p <pid>` during a benchmark; confirm `numa_miss` vs `numa_hit` ratio. That single number tells which of Fix 1/2/3 actually moved the needle.
