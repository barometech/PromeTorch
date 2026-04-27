# Agent 4 — LCC compiler flag + asm audit (Round 3)

## Hot loops (header-only, included in `test_gguf_inference.cpp`)

- `q4k_gemv_scalar` — `torch/io/cpu_quant_gemv.h:688-766`, j-loop **:747**.
- `q4k_gemv_batched_scalar` — same file `:1903-1970`, j-loop **:1954**.
- `q4k_gemv_avx2_float` (SSE4.1 v2 — runs under LCC since `__AVX2__`
  unset) — `:381-488`, outer `bi` **:381**, j **:417**.
- Helpers `gguf::fp16_to_fp32` (`gguf_dequant.h:73`) and
  `get_scale_min_k4` (`:289`) are plain `inline` — no
  `__attribute__((always_inline))`.

## Current Elbrus flags (`CMakeLists.txt:130-189`)

`-O3 -ffast -faligned -fprefetch -fcache-opt -mtune=elbrus-8c2
-frestrict-all -fswp-maxopers=800 -fopenmp`. NOT present: `-O4`,
`-march=elbrus-v5`, `-fwhole-shared`, `-flist-prefetch`, `-fipo-invup`,
`-finline-{level,scale}=`, `-fforce-swp`, `-funroll-loops`,
`-fopenmp-simd`, `-fprofile-generate/-use`. Comment at **:160-166**:
plain `-fwhole` failed — GNU ld rejected EIR `.o`
(`'.pack_pure_eir' is illegal during non-relocatable linkage`).

## A. Flag / pragma changes — ranked

| # | Change | Where (file:line) | Mechanism | Uplift | Risk |
|---|---|---|---|---|---|
| **1** | `-fwhole-shared` (NOT plain `-fwhole`) on inference target only — both `target_compile_options` AND `target_link_options`. | `examples/gguf/CMakeLists.txt:4-19` | `-fwhole-shared` keeps PIC layout so GNU ld accepts the `.o` (the `:160` failure was non-PIC `-fwhole`). Cross-TU inlines `cpu_quant_gemv` into the ~3.4 kLoC `gguf_model.h` TU. | **+5–15 %** | LOW |
| **2** | Two-phase PGO: Phase 1 `-fprofile-generate=/tmp/gguf.prof` + run `./test_gguf_inference qwen3-4b.gguf "The sky is" 50`. Phase 2 `-fprofile-use=/tmp/gguf.prof`. | `scripts/build_pgo.sh`; CMake guard via `-DPT_PGO=generate\|use` | LCC ch. 6.4.5:3464 «профиль … помогает весьма сильно». Decode is fully deterministic (36 layers × 7 GEMV shapes/token) — ideal PGO. Improves SWP II via branch-merge. | **+5–20 %** | MED |
| **3** | `__attribute__((always_inline))` on `fp16_to_fp32`, `get_scale_min_k4` via `#if defined(__LCC__) #define GGUF_FORCE_INLINE inline __attribute__((always_inline)) #else #define GGUF_FORCE_INLINE inline #endif`. | `gguf_dequant.h:73, :289` | Both called 4×/j-iter of v2 path. `fp16_to_fp32` is ~25 stmts with denormal/Inf branches; LCC may decline plain `inline` (`OPTIMIZATION_FINDINGS.md`§6) — call inside SWP body breaks the pipeline. | **+3–10 %** | LOW |
| **4** | `-finline-level=2.0 -finline-scale=2.0` (only after #1). | `CMakeLists.txt:158` | `LCC_OPTIMIZATION.md:40-41`. No-op without `-fwhole-shared`. | +1–3 % | LOW |
| **5** | Replace `-mtune=elbrus-8c2` with `-march=elbrus-v5 -mtune=elbrus-8c2`. | `CMakeLists.txt:154` | `-mtune` only adjusts scheduler weights; `-march=v5` *enables* v5 vector ops (qpmaddubsh, qppermb, fapb, full predicate). Without it LCC may emit v3-compat code (`e2k_vnni/FINDINGS.md:166`). | +0–8 % | LOW |
| **6** | Add `-flist-prefetch -fipo-invup`. | `CMakeLists.txt:153` | `-flist-prefetch` extends auto-prefetch to non-strided pointer chains (KV-cache row-ptr). `-fipo-invup` hoists invariant loads — needs #1 first. | +0–3 % | LOW |
| **7** | (a) `_Pragma("loop count(20)") _Pragma("ivdep")` on **outer `bi`-loop** (currently no pragma). (b) Append `_Pragma("unroll(4)")` to j-loop pragma. | `cpu_quant_gemv.h:381` and `:417` | `bi` ~20 iters with no hint → APB conservative (round 2 agent_2 §2). On j (4 iters) `unroll(4)` lets LCC straight-line + SLP-pack instead of paying SWP prologue/epilogue at II=4. | **+5–10 %** | LOW |

## B. Asm-level proposals

Hot chain `qpmaddubsh → qpmaddh → qpaddw → fmuls → fadd`
(FINDINGS.md:170) yields II≈3 single-accumulator. Two structural fixes:

### B-1. Force-emitted L2 prefetch via inline asm

JOURNAL.md: `__builtin_prefetch` in `q4k_gemv_scalar` measured **0 %
uplift** — LCC silently elides hints when `-fprefetch`'s cost-model
thinks they're redundant. On `q4k_gemv_avx2_float` they DO fire
(+10 % per `OPTIMIZATION_FINDINGS.md` table). Bypass the gate with
mas=0x20 ("skip L1, into L2", guide line 3863):

```c
#if defined(__LCC__)
#define E2K_PLD_L2(addr) \
    __asm__ volatile("ldb,sm 0x0, [ %0 + 0 ], %%empty, mas=0x20" \
                     : : "r"(addr) : "memory")
#else
#define E2K_PLD_L2(addr) __builtin_prefetch((addr), 0, 2)
#endif
```

At `cpu_quant_gemv.h:729`, replace the three `__builtin_prefetch` calls
with `E2K_PLD_L2` at `bi+8` blocks ahead (~1.1 KB, covers ~100-tact DDR
latency at observed 2.7 GB/s/rank). `volatile` + `"memory"` clobber
prevents elision. **+3–8 %**.

### B-2. Forced register split for the FP accumulator

`cpu_quant_gemv.h:756-757` shares `dot` between both nibble lanes —
visible RAW per line forces SWP II ≥ 4 (fadd latency, agent_1 §8.4).
Split into 4 independent accumulators with a 0-instruction asm fence
so LCC can't re-fuse:

```cpp
float dot0=0, dot1=0, dot2=0, dot3=0;
// inside l-loop:
float w_lo = d1 * (qs[l]   & 0xF) - m1;
float w_hi = d2 * (qs[l]   >> 4)  - m2;
__asm__("" : "+f"(dot0), "+f"(dot1));   // 0-inst live-range fence
dot0 += w_lo * x[base_k + j + l];
dot1 += w_hi * x[base_k + j + 32 + l];
// split l in halves to feed dot2/dot3
dot = (dot0 + dot1) + (dot2 + dot3);
```

Canonical LCC idiom (cf. `e2k-ports` add_pd); textbook example
`LCC_OPTIMIZATION.md` §4.3. Difference between II=4 (fadd recurrence)
and II=1 (fadd pipelined). **+10–20 %** on scalar path — first verify
with `dprof -m TICKS,EXEC,BUB_E2` that the scalar path is actually hit.

## Next actions (no edits made)

1. Stage #1 + #3 + #5 + #7 in one patch — additive, all LOW-risk.
2. Bench; if gain < 10 %, attempt #2 PGO.
3. Defer #4, #6 until #1 measured.
4. B-1 + B-2 in one separate patch (`cpu_quant_gemv.h:729, 755-758`);
   bench independently.

Lower-bound additive headroom ~+13 % of 211 ms = **5.3 → ~6.0 tok/s**.
Stacks with agent_1 ThreadPool +8 % and agent_2/3 sharding — the LCC
axis alone will NOT reach 10 tok/s; it is a fractional contributor.
