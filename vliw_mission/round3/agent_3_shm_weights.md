# Agent 3 (Round 3) — Shared SHM weight pages across TP processes

**Date:** 2026-04-25
**Brief:** Replace per-rank weight replication with `/dev/shm` mmap-shared
single physical copy across TP processes. Goal: free DDR / better NUMA layout.
**TL;DR:** **Don't do it. Current TP-4 design already wins.** Keep `NumaReplica`
for the 1-process path; for TP-4, the right cheap win is something else (see §6).

---

## 1. The premise is wrong for TP-4

The mission brief assumes ranks have 4 replicated copies of every weight
(9.6 GB total) and that "each rank only reads ITS slice anyway, so overlap is
small." The code disagrees.

In multi-process TP-4 launched via 4 separate `numactl --cpunodebind=$rank
--membind=$rank` invocations:

- `init_tensor_parallel()` (`torch/io/gguf_model.h:4064-4344`) builds **per-rank
  malloc'd K-slices and row-slices** for every weight: `slice_rows()` (line 4156)
  and `slice_k_blocks()` (line 4222) `std::malloc` exactly the rank's bytes
  (`row_count_elems * stride` and `full.rows * local_row_stride`). These mallocs
  happen under `--membind=$rank`, so they land in the rank's local DDR.
- After slicing, `init_tensor_parallel` **frees** the full row-split source
  weights (line 4327-4341, only for non-mmap'd parents). The full
  `attn_output`/`ffn_down` parents are typically mmap-owned, so they stay
  mmap-resident on whichever node first-touched the GGUF pages.
- `NumaReplica` is consulted ONLY in the 1-process decode path (5 callsites:
  `gguf_model.h:2528, 2573, 2647, 2941, 2978` — all in `forward_decode_cpu()`,
  none in `forward_decode_cpu_tp()`). Confirmed in `vliw_mission/agent_3_numa_audit.md:101`:
  > In TP mode, NumaReplica is completely unused.

So TP-4 in production today has roughly:

| Buffer | Source | Bytes/rank | Locality |
|---|---|---|---|
| Q/K/V/gate/up row-slices (per-rank malloc) | `tp_.layers[i].q_*` | 1/4 of full | local to rank |
| attn_output/ffn_down K-slices (per-rank malloc) | `tp_.layers[i].q_attn_output / q_ffn_down` | 1/4 of full | local to rank |
| Full mmap'd parent weights (used only under PT_TP_GATHER for w_ao_full at 4635) | `layers[i].*.cpu_data` | 1× full | mmap (first-touch) |
| `NumaReplica` allocations | unused in TP path | 0 | n/a |

Per-rank weight bytes ≈ ~600 MB (1/4 of 2.4 GB) plus mmap parents that *might*
also be paged in from K-slice copy time. Total RSS per rank ≈ 0.6-1.5 GB.
**Aggregate ≈ 2.4-6 GB**, not 9.6 GB. The "75% of memory wasted" framing is the
1-proc + `PT_NUMA_REPLICATE=1` case, which is a different deployment.

## 2. Bandwidth arithmetic for the four options

Per-token weight read: 2.4 GB (qwen3:4b Q4_K_M). E8C2: per-chip local DDR
~12 GB/s sustained, cross-chip ~3-4 GB/s. 4 chips, 4 ranks.

Each rank reads only its slice (1/4 = 600 MB) plus zero-padded full vectors for
gather paths. Take 600 MB/rank/token as the dominant traffic.

**(a) Current per-rank malloc'd slice (status quo).** All reads local at 12 GB/s.
600 MB / 12 GB/s = 50 ms compute-bound floor per rank. Aggregate effective BW
= 4 × 12 = 48 GB/s. **Memory: ~2.4 GB total + mmap parents.**

**(b) Single SHM copy on creator node (node 0).** 2.4 GB total. Rank 0 reads
its 600 MB slice locally @ 12 GB/s = 50 ms. Ranks 1-3 read their 600 MB slice
through cross-chip @ ~3 GB/s = 200 ms. Token latency = max(50, 200) = **200 ms,
i.e. 5 tok/s** (worse than today's 4.8 tok/s by negligible amount, but no
headroom). Aggregate effective BW = 12 + 3+3+3 = 21 GB/s. **Loses 56% of BW.**

**(c) Per-NUMA SHM pool (one SHM region per node, populated once at startup,
each rank reads from its node's region).** Same memory as today (4 × 2.4 GB =
9.6 GB if every rank's region is the full model; or 4 × 0.6 GB = 2.4 GB if
each region holds only that node's slice). All reads local @ 12 GB/s = 50 ms.
Identical bandwidth to (a). **The only win is RAM if we choose the 4×0.6 GB
variant — but that's exactly today's `tp_.layers[i].q_*` malloc'd buffers
already, just with `MAP_SHARED` instead of `MAP_PRIVATE`.** No measurable BW
gain.

**(d) Single SHM copy + `numa_interleave` across all 4 nodes at page
granularity.** 2.4 GB total. Each rank reads 25% local (3 GB/s if cross-chip,
12 GB/s if local) and 75% remote. Mean per-byte BW ≈ 0.25×12 + 0.75×3 =
**5.25 GB/s per rank**, aggregate 21 GB/s. Same as (b). **Loses 56% of BW.**

| Option | Total RAM | Per-rank effective BW | Aggregate BW | tok/s estimate |
|---|---|---|---|---|
| (a) status quo | 2.4 GB + mmap | 12 GB/s | 48 GB/s | 4.8-9 |
| (b) single SHM on node 0 | 2.4 GB | 5.25 GB/s avg | 21 GB/s | ~5 |
| (c) per-NUMA SHM (4×600 MB) | 2.4 GB | 12 GB/s | 48 GB/s | 4.8-9 |
| (d) interleaved single SHM | 2.4 GB | 5.25 GB/s | 21 GB/s | ~5 |

**(c) ties (a) on bandwidth and matches it on RAM.** No win.
**(b) and (d) lose ~56% of bandwidth — direct regression.**

## 3. Why "shared pages = better cache" doesn't apply here

The classic argument for SHM-shared read-only weights — "OS dedups physical
pages across processes; one less copy in the cache" — assumes you have ONE
DRAM controller and the cache is the bottleneck. On E8C2 you have **four
independent DRAM controllers**, one per chip. The whole point of TP is
parallelizing across them. Sharing a physical page on one controller means
three controllers are *idle* while one is overworked. This is exactly what
agent_3_cache_mem.md §1.1 and agent_9_numa_aggregate.md §1.3 already proved
for `attn_output`/`ffn_down`/`output_weight`: replicated-but-read-by-all =
1.15× speedup, K-sliced = ~9 tok/s projected.

L3 isn't the bottleneck either. Working set 2.4 GB / chip-L3 16 MB = **150×
overflow** — caches are pure pass-through for weight streams (agent_3_cache_mem
§1). Whether the page is "logically deduped" or "physically replicated" in
DDR makes no observable cache difference — every line is a cold L3 miss
either way.

## 4. The real problem mission #3 should solve

The genuinely shared-data thing in TP-4 today is not weights but the **mmap'd
parent of `attn_output`/`ffn_down`**. Under `PT_TP_GATHER`, every rank reads
`layer.q_attn_output.numa_replica.get(_node)` (line 4635), falling back to
`layer.q_attn_output.cpu_data` — which is the mmap'd file region first-touched
on whichever node loaded the GGUF. Three out of four ranks pay cross-chip
latency for this. **NumaReplica is unused in this path** because no caller
runs `replicate_weights_for_numa()` between mmap and `init_tensor_parallel()`.

Round 2 already identified the fix (agent_9 §4-5, "Option D + F"): K-slice
the replicated weights and stop reading the mmap parent at all. That's a
~150-line patch (mostly making `slice_k_blocks` work for `output_weight`),
not an mmap/SHM redesign.

## 5. Correctness/concurrency risk if we did try (b) or (d)

If someone insists on (b) or (d) for memory savings:

- **First-touch reproducibility:** `MAP_SHARED` over a `/dev/shm` file places
  pages on whichever process FIRST touches each page. To pin to node 0 (option b),
  the creator must do `mbind(MPOL_BIND, node 0)` BEFORE any rank reads, and
  `numa_set_strict(1)` to make the policy hard-fail on memory pressure.
  `madvise(MADV_HUGEPAGE)` is fine — `MAP_SHARED` over tmpfs supports THP since
  Linux 4.8 — but `MADV_RANDOM` on the same region disables readahead which is
  what we want.
- **MAP_HUGETLB constraint:** `mmap(MAP_SHARED|MAP_HUGETLB)` over `/dev/shm`
  works only if `/dev/shm` is mounted with `huge=advise` (Elbrus default: no).
  Fall back to a separate `/mnt/hugetlbfs` if that route is wanted.
- **Init race:** all ranks must mmap+barrier before any starts decode, else
  a late-joiner triggers first-touch on node-not-zero and pulls pages off
  node 0. Add `pthread_barrier_wait` (which TP already has via SHM AllReduce
  ring) right after the load_ollama call, before `init_tensor_parallel`.
- **mlock budget:** 2.4 GB of `mlock`'d shared pages × 4 processes counts
  4× against `RLIMIT_MEMLOCK` even though physical RSS is 1×. Tune
  `/etc/security/limits.conf` or skip mlock.

None of these are exotic, but they're all extra failure surface for **zero
measured benefit**.

## 6. Recommendation: don't ship (a→b/c/d). Ship Round 2 Option D instead.

**Verdict: keep current per-rank malloc'd K-slices.** SHM-shared weights are
a regression on every realistic Elbrus 8СВ workload because cross-chip BW
(3-4 GB/s) is 3-4× worse than local (12 GB/s), and we have plenty of RAM
(125 GB) to keep replicated-via-malloc.

If memory pressure ever does become a concern (e.g. 14B model on this box):
**(c) per-NUMA SHM pool with each rank's region holding only that rank's K-slice**
— but at that point it's just `mmap(/dev/shm, MAP_SHARED|MAP_POPULATE)`
backing the same `tp_.layers[i].q_*.cpu_data` we already have, with the only
practical change being `MAP_SHARED` instead of `MAP_PRIVATE` on
`std::malloc`'s arena. Net win: zero (the malloc was already node-local).
**Skip.**

The actually-useful round-3 work, in priority order:
1. **K-slice `output_weight` and `attn_output`/`ffn_down` for the gather path**
   (Round 2 Option D, ~150 lines, +50% projected per agent_9 §4.2). This
   eliminates the cross-chip mmap-parent reads that *are* the SHM-style
   problem mission #3 was groping at.
2. Async AllReduce overlap (agent_9 §6.2, +6%, ~230 lines).
3. Tighter SHM barrier (agent_9 §6.4, +6%, ~100 lines).

These compound to the projected 17-22 tok/s ceiling. Mission #3's premise
(shared SHM pages) does not.

---

**Files referenced (absolute paths, all read-only this round):**
- `C:\Users\paper\Desktop\promethorch\torch\io\numa_weight_replica.h`
- `C:\Users\paper\Desktop\promethorch\torch\io\gguf_loader.h` (lines 240-323)
- `C:\Users\paper\Desktop\promethorch\torch\io\gguf_model.h` (lines 1535-1588,
  4064-4344, 4620-4660)
- `C:\Users\paper\Desktop\promethorch\vliw_mission\agent_3_numa_audit.md` (line 101 confirmation)
- `C:\Users\paper\Desktop\promethorch\vliw_mission\round2\agent_3_cache_mem.md`
- `C:\Users\paper\Desktop\promethorch\vliw_mission\round2\agent_9_numa_aggregate.md`
