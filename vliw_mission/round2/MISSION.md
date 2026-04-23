# ROUND 2 — TARGET 20 tok/s on Elbrus 8C2

## Current state (2026-04-24)

**Baseline measured:** 1-proc T=30 = **5.3 tok/s** / TP-4 T=7 = **6.1-6.2 tok/s** on qwen3:4b Q4_K_M.

**Public E2K ceiling:** 5.2-6.7 tok/s (Alex Mikhaliuk llama.cpp-e2k, 2023, frozen).

**Physical ceilings:**
- Per-chip DDR effective: ~20 GB/s × 0.4 KB/token-weights = ~50 tok/s absolute single-chip
- Aggregate 4-chip: ~80 GB/s × 0.4 KB = ~200 tok/s theoretical IF weights replicated
- Current utilization: ~30% of per-chip in 1-proc, ~15% of aggregate in TP-4

**Target: 20 tok/s** — 3.3× above current baseline. Requires unlocking cross-chip parallelism without AllReduce overhead OR fundamentally changing compute pattern.

## What's been tried

Already in main (commits `eee6a1d` → `179fce8`):
1. Batched Q4_K GEMV kernel (1.54× K=4 standalone, integrated)
2. Full `forward_decode_cpu_batched` with batched attention
3. Speculative decode scaffold (spec_decode_step_cpu)
4. NgramDraft + real draft-model (qwen3:0.6b) integration — 0% acceptance on free-form text
5. APB (Array Prefetch Buffer) enabled: `int64_t` + `#pragma loop count(N)` + `#pragma ivdep` on all 15 inner j-loops
6. LCC build flags: `-ffast -faligned -fprefetch -fcache-opt -mtune=elbrus-8c2 -frestrict-all -fswp-maxopers=800`
7. NUMA weight replication infrastructure (helps TP, regresses 1-proc with interleave)
8. Bounded-spin SHM AllReduce

All applied. Net gain ~+30% over cold start. Ceiling hit.

## Mission for Round 2 agents

**Everyone writes their findings to `vliw_mission/round2/agent_N_<role>.md`**. MCST official PDFs extracted to `vliw_mission/round2/_inputs/*.txt`:
- `mcst_main_guide.txt` (178 pages, 300 KB) — full MCST programming guide
- `loop_vectorization.txt` — MCST loop vectorization techniques
- `cache_optimization.txt` — cache behavior details
- `memory_controller.txt` — DDR controller patterns
- `lcc_auto_parallel.txt` — compiler auto-parallelization
- `hpc_architecture.txt` — multi-chip HPC tricks
- `transcendental_vec.txt` — FP function vectorization
- `eml_acceleration.txt` — EML library acceleration

**Rules:**
- Read the assigned documents carefully — don't paraphrase without reading
- Identify SPECIFIC concrete optimization we have NOT applied yet
- Quote exact source text (file + line number) for each claim
- Propose exact code change (file:line, before/after)
- Estimate expected speedup with honest uncertainty bounds
- Flag anything that would require hardware changes (not achievable)

Current code:
- `torch/io/cpu_quant_gemv.h` — Q4_K GEMV (3 kernel variants + batched)
- `torch/io/gguf_model.h` — forward_decode_cpu + forward_decode_cpu_batched + spec_decode_step_cpu
- `torch/distributed/ddp.cpp` — SHM AllReduce
- `c10/util/ThreadPool.h` — persistent thread pool
- `CMakeLists.txt:130-170` — Elbrus build flags
