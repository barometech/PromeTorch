# E2K / LCC optimization findings — distilled from downloaded sources

**Sources (all in this folder):**
- `mcst_official/elbrus_prog_2020-05-30.pdf` — MCST official programming guide
- `ports/` — ilyakurdyukov's performance patches for 30+ libraries (ffmpeg, openblas, x264, zstd, fftw, etc.)
- `ports/README.md` — compact porting cheat sheet
- `littlecc/` — minimal LCC backend for E2K (reference)
- `elbrus-docs/` — basic E2K ELF loader docs
- `habr_smartengines_optimization.html` — Smart Engines optimization article

---

## Critical findings (applied in commit 2273a9f)

### 1. APB (Array Prefetch Buffer) — hardware prefetch

The Elbrus E2K has a **hardware prefetch buffer** that streams memory into L1 before the CPU asks for it. It stays OFF unless you meet THREE conditions:

```cpp
// All three needed to enable APB:
#pragma loop count(4)     // Hint compiler about trip count
#pragma ivdep              // Assert no write-after-read aliasing
for (int64_t j = 0; j < N; j += step) {   // int64_t counter (NOT int!)
    // ... vector intrinsic body
}
```

LCC silently disables APB for:
- `int` loop counters (must be `int64_t` or `size_t`)
- Loops without `#pragma loop count`
- Loops where compiler can't prove aliasing is safe (need `#pragma ivdep`)

### 2. `-fprefetch -fcache-opt` compiler flags

Not on by default at -O3 on LCC 1.29. Enable explicitly:
```cmake
-fprefetch        # software prefetch hints in loops
-fcache-opt       # cache-blocked loop pipelining (SWP)
-fswp-maxopers=800  # raise SWP operation limit from default 300
-mtune=elbrus-8c2   # E8C2-specific scheduler weights
-frestrict-params  # implicit __restrict on function pointer params
```

### 3. `restrict` IGNORED with vector intrinsics

```
"Using the `restrict` keyword is good for performance, but note that it
is ignored by the LCC if you're using vector load/store intrinsics such
as `_mm_load_si128()`. For code with vector intrinsics use `#pragma ivdep`."
  — ilyakurdyukov/e2k-ports README
```

So in our Q4_K GEMV (which uses `_mm256_loadu_si256` everywhere), adding
`__restrict` to pointer params does NOTHING. The `#pragma ivdep` is what
actually unlocks reordering.

### 4. Intrinsic family preference (ilyakurdyukov/e2k-ports README)

```
MMX, SSE2, SSSE3, SSE4.1    — native support (1:1 QP ops)
AVX, AVX2                   — supported but NOT recommended
                              ("uses too much CPU registers")
SSE4.2, _mm_dp_ps            — EMULATED, slow, do not use
```

Our Q4_K kernel uses `_mm256_*` AVX2 intrinsics. ilyakurdyukov's verdict:
"supported but not recommended". Actual measurement: Phase 6 rewrote
Q4_K kernel in SSE4.1 128-bit vertical-accumulate form → measured IDENTICAL
performance to AVX2 on this workload (we're memory-bound; the compiler
handles both ABIs equivalently in practice when all the APB/ivdep hints
are in place).

### 5. Native E2K byte-permute: `__builtin_e2k_qppermb`

128-bit byte shuffle/permute — **NATIVE single-cycle** on E2K. Used across
ports as `_mm_shuffle2_epi8(a, b, ctrl)` which maps to `qppermb`. On x86
this has no direct analog, hence the port-level macro.

**Potential future Q4_K win:** replace `_mm_and_si128(x, mask_lo4)` +
`_mm_srli_epi16(x, 4)` nibble-extract sequence (2 ops) with a single
`qppermb`-based unpack when running on E2K. Not yet applied.

### 6. Force-inline for hot paths

```c
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
```

LCC may decline `inline` for "complicated" functions. For our
`q4k_q8_dot_avx2`, `hsum_avx`, `hsum_m128` helpers this matters — they're
used millions of times per token.

### 7. Compile flags summary

From the MCST official guide + our analysis:
```bash
lcc -O3 -ffast-math \
    -mtune=elbrus-8c2 \
    -fprefetch -fcache-opt \
    -frestrict-params \
    -fswp-maxopers=800 \
    -fopenmp \
    -lpthread -lnuma
```

Our CMakeLists now passes all of these in the `if(CMAKE_SYSTEM_PROCESSOR
MATCHES "e2k|elbrus")` block.

---

## Patterns we HAVEN'T yet applied (next iterations)

### A. `qppermb` native unpack for Q4_K nibbles

Current AVX2 path:
```cpp
__m256i raw = _mm256_loadu_si256(...);
__m256i lo = _mm256_and_si256(raw, mask_lo4);
__m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_lo4);
```

Potentially-native E2K version (128-bit half):
```cpp
__m128i raw = _mm_loadu_si128(...);
// Interleave-unpack lo/hi nibbles into a single __m128i with qppermb
__m128i unpacked = _mm_shuffle2_epi8(raw, raw, nibble_ctrl);
```

Requires benchmarking — may or may not beat LCC's auto-translation. Low priority
given we're memory-bound.

### B. `qpmaddubsh` explicit fast-path

Already probed in `examples/benchmarks/q4k_e2k_kernel_probe.cpp` — measured
-23% slower than LCC's AVX2 auto-translation when APB was OFF. With APB now
ON, worth re-probing.

### C. `-fwhole-shared` link-time optimisation

Enable cross-TU LCC optimization. Requires LCC ar wrapper — not enabled by
default. Potential 5-10% on link-heavy hot paths.

### D. PGO (Profile-Guided Optimization)

```bash
lcc -O3 -fprofile-generate=./prof myprog.c -o myprog_prof
./myprog_prof   # run typical workload
lcc -O3 -fprofile-use=./prof/myprog.gcda myprog.c -o myprog_fast
```

LCC docs claim "significant" speedups on branch-heavy code. Our kernels are
straight-line so the benefit is likely small, but worth measuring.

---

## Measured results (commit `2273a9f`)

| Config | Before APB | After APB | Δ |
|--------|-----------|-----------|---|
| 1-proc T=16 | 3.8 | 4.2 | +10% |
| 1-proc T=24 | 4.7 | 5.2 | +11% |
| 1-proc T=28 | 5.0 | 5.3 | +6% |
| 1-proc T=30 | 5.0 | 5.3 | +6% |
| 1-proc T=32 | 4.0 | 4.5 | +12% |
| TP-4 T=7    | 5.4 | 5.7–6.2 (avg 5.7) | +5–15% |

APB is the first reliable CPU-side perf win of the mission that isn't
dependent on spec-decode machinery. Infrastructure-level optimization
that should have been on from day one (2020 MCST guide clearly documents
it); just wasn't applied.
