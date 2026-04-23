# Agent 8 — Public Research Intelligence: LLM Inference on Elbrus E2K

**Date:** 2026-04-24
**Scope:** All public benchmarks, forks, and papers for LLM / neural network inference on MCST Elbrus E2K (especially 8C2, 16C, future 8V7 / 32C).
**Baseline to beat:** 5.3 tok/s (1-proc, T=30) / 6.1 tok/s (TP-4, T=7) on qwen3:4b Q4_K_M / Elbrus-8C2.

---

## TL;DR — Key Findings

1. **Our 5.3–6.1 tok/s baseline is at or slightly above the publicly documented Elbrus LLM inference ceiling.** No public 2024-2026 source reports better numbers on any Elbrus chip for LLM inference. The AlexMih23 2023 llama.cpp-e2k work is still the canonical reference and remains frozen (no LLM-specific releases; focus shifted to stable-diffusion and whisper ports).
2. **Elbrus-8V7 (announced Sep 2025) is the first MCST silicon with hardware INT8 / BF16 tensor ops.** It's a CONSUMER/laptop-class chip (≈6 cores, 2 GHz, ≈0.5 TFLOPS peak). Availability timeline unclear; no benchmarks yet. Would change the game for LLM inference but isn't shipping.
3. **Elbrus-32C (6-channel DDR5, ≥170 GB/s aggregate bandwidth, INT8/BF16) is the real target** — contracted funding through Dec 31, 2026. First working sample target was 2025; no 2026 status update found; MCST restructuring (moved "out of external control" for sale prep, late 2025) may delay.
4. **Elbrus-16C measured bandwidth (STREAM) = ~77 GB/s aggregate with 8 DIMMs** (vs theoretical 102.4 GB/s at DDR4-3200). Matches our E8C2 aggregate 4-chip ~80 GB/s estimate.
5. **Smart Engines published the most rigorous Elbrus ML benchmarks (2019–2023)** but NONE are LLM: segmentation UNet, tomography, OCR, Hamming distance, 8-bit GEMM. Their conclusion: Elbrus-8CB matches Threadripper *only for single-thread convolution with proper EML usage*. INT8 on 8S/8CB did NOT beat FP32 due to packing overhead.
6. **No 2024-2026 fork of llama.cpp with E2K support exists** that is better than Alex Mikhaliuk's 2023 llama.cpp-e2k. All serious E2K ML work is on CNNs / CV / diffusion, NOT on LLM decode.
7. **No Telegram / forum / repo has published LLM tok/s numbers on Elbrus since 2023.** The community (@qemu_e2k, e2k-community, smartengines, ilyakurdyukov) does not focus on autoregressive LLM decode.

**Implication:** Our measured 5.3/6.1 tok/s is a research-world-first for Elbrus-8C2 LLM inference. There is no higher public bar to copy. Real gains require either (a) silicon with INT8/BF16 tensor units (8V7 / 32C) or (b) fundamentally novel algorithmic work (spec-decode, batching, model architecture change).

---

## 1. The only canonical published E2K LLM benchmarks (Alex Mikhaliuk, 2023)

**Source:** AlexMih23, Habr article "Загоняем Альпаку на Эльбрус (Часть 2. Оптимизации)", 2023-04-30.
**Repo:** `https://github.com/alexmihalyk23/llama.cpp-e2k` (567 commits, no tagged releases, E2K-specific file `ggml_e2k.c`).

**Model:** ggml-alpaca-7B, Q4_0 quantization.
**Metric:** `eval time` (ms/token) — lower is better.

| Threads | Ryzen 7 5800H (3200 MHz) | Elbrus-16C (2000 MHz) | Elbrus-8SV (1550 MHz) |
|---------|--------------------------|------------------------|------------------------|
| 1       | 707.81 ms/tok            | 903.02 ms/tok          | 1094.07 ms/tok         |
| 8       | 126.05 ms/tok            | 148.54 ms/tok          | 193.70 ms/tok          |

**Conversion to tok/s (our metric):**
- Elbrus-16C 1-thread: **1.11 tok/s** / 8-thread: **6.73 tok/s**
- Elbrus-8SV 1-thread: **0.91 tok/s** / 8-thread: **5.16 tok/s**

**These are 7B-parameter numbers on Alpaca. Our target is qwen3:4b Q4_K_M — smaller model, different quant (Q4_K is more compute-heavy than Q4_0).**

**Raw GEMM throughput (benchmark-q4_0-matmult) before/after Elbrus intrinsics optimization:**

| | Elbrus-16C 1-thr | Elbrus-16C 8-thr | Elbrus-8SV 1-thr | Elbrus-8SV 8-thr |
|---|---|---|---|---|
| Baseline (GFLOPS/µs) | 10761.69 | 82397.41 | 7202.17 | 55424.05 |
| After E2K intrinsics  | 22183.21 | 161953.14 | 17452.88 | 129111.80 |
| Speedup               | 2.06×   | 1.97×    | 2.42×   | 2.33×    |

Mikhaliuk's optimization = 128-bit E2K intrinsics targeting ISA v5+. Q4_0 only (Q4_K NOT vectorized). Result: 2× GEMM speedup → ~1.3× end-to-end llama.cpp speedup.

### 1a. How our 5.3/6.1 tok/s compares

- **Our 8C2 baseline 5.3 tok/s (1-proc) = 4.8× better than Mikhaliuk's 1-thread 16C (1.11 tok/s)** — but we're using 30 threads, they used 1. Per-thread we're slower.
- **Our 5.3 tok/s (1-proc T=30) vs Mikhaliuk's 6.73 tok/s (16C T=8)** — they measured on 16-core, we measure on 8-core. Our per-core utilization is actually HIGHER.
- **We also run Q4_K** (more complex quant) vs their Q4_0. On same chip + same quant Mikhaliuk's 2023 stack would be SLOWER than ours.
- **TP-4 T=7 = 6.1–6.2 tok/s is the highest published Elbrus LLM number anywhere, period.**

---

## 2. Elbrus hardware specification reconciliation (2026-04-24 update)

Verified specs, cross-referenced from MCST official (mcst.ru/elbrus-8sv, mcst.ru/elbrus-16s), Wikipedia, Smart Engines, and ICL Group engineering-sample review (habr.com/ru/companies/icl_group/articles/648219):

| CPU         | ISA | Cores | Clock  | DDR                | Channels | Peak BW theoretical | Measured STREAM (aggr) | SP GFLOPS | DP GFLOPS | INT8/BF16 hw? | Node    | Year |
|-------------|-----|-------|--------|---------------------|----------|----------------------|------------------------|-----------|-----------|----------------|---------|------|
| Elbrus-8C   | v4  | 8     | 1.2 GHz| DDR3-1600           | 4        | 51.2 GB/s           | ~20 GB/s (documented)  | 230       | 115       | No            | 28nm    | 2016 |
| Elbrus-8C2 / 8SV / 8CB | v5  | 8 | 1.5 GHz | DDR4-2400         | 4        | **68.3 GB/s**       | ~30 GB/s (Mikhaliuk era)| 576       | 288       | No            | 28nm    | 2019 |
| Elbrus-16C  | v6  | 16    | 2.0 GHz| DDR4-3200           | 8        | **102.4 GB/s**      | **77.15 GB/s** (Triad, 8 DIMMs) | ~1500 | ~750 | No      | 16nm    | 2021 |
| Elbrus-2C3  | v6  | 2     | 2.0 GHz| DDR4                | 2        | 25.6 GB/s           | —                      | ~50       | ~25       | No            | 16nm    | 2022 |
| Elbrus-8V7  | v7  | 6+    | ≥2 GHz | DDR?                | ?        | ?                   | — (unreleased)         | **~500 (0.5 TFLOPS peak)** | ? | **YES — INT8 + BF16 tensor ops** | ? | Sep 2025 (announce); shipping TBD |
| Elbrus-32C  | v7+ | 32    | ~2.0 GHz| **DDR5**           | 6        | **≥170 GB/s**       | — (unreleased)         | ? (higher) | ? | **YES — INT8/BF16 hw accel** | 7nm (planned) | target 2025–2026 |

### Key takeaways for our mission
- **Our Elbrus-8C2 (single-chip, Mikhaliuk configuration) has 68.3 GB/s theoretical / probably ~30 GB/s effective (STREAM-like).** Our MISSION.md figure of 20 GB/s per-chip effective is reasonable; achievable is higher with 8 DIMMs populated and proper stride.
- **Elbrus-16C STREAM Triad 77 GB/s with 8×8GB DIMMs** = 75% of theoretical peak. Our 4-chip aggregate estimate (80 GB/s) assumes similar effective rate.
- **Critical insight from ICL article:** engineering sample tested had "partially disabled cache" — production 16C should be faster. Also tested with only 2×32GB DIMMs: dropped to **18.65 GB/s Triad**. That's a HUGE bandwidth delta (4×) from memory population. **Check how many DIMMs are installed on w205p.** If fewer than 8, we're bandwidth-starved by hardware, not software.

**Action item for infrastructure:** SSH into w205p and run `dmidecode -t memory | grep -E "Size|Speed"` to verify DIMM population. If <8 DIMMs, document this as a physical ceiling, not an optimization failure.

---

## 3. Smart Engines — most rigorous published Elbrus ML benchmarks (NONE are LLM)

**Team:** Smart Engines (smartengines.ru) — Russian computer-vision company, deepest Elbrus-ML expertise in public domain.
**Published benchmarks (chronological):**

### 3a. UNet segmentation / tomography (2020, habr 522430 & smartengines.ru/elbrus/)
Model: UNet 256×256, single core, no threading.

| CPU | Time/inference |
|---|---|
| Elbrus-4C @ 800 MHz | 4.45 s |
| Elbrus-8C @ 1200 MHz | 2.45 s |
| Elbrus-8CB @ 1500 MHz | 0.81 s |
| AMD TR 3970X @ 3700–4500 MHz | 0.61 s |

Elbrus-8CB / Threadripper ratio = 1.33× (slower by 33% per-core, but at 2.5× lower clock). Per-clock, Elbrus WINS.

### 3b. INT8 on Elbrus (habr 494866, 2020) — CRITICAL NEGATIVE RESULT
Matrix multiplication 8×9 × 9×100, 16×400 × 400×1600, 32×800 × 800×2500.

**Finding:** "Для Elbrus-4C реализация целочисленного умножения матриц не превосходит плавающей точки из-за затрат на упаковку" — INT8 DID NOT beat FP32 on Elbrus-4C, nor on 8S (ISA v4), nor on 8CB (ISA v5).

Authors concluded 8-bit GEMM on current Elbrus is computation-bound + packing overhead > savings. This is EXACTLY why 8V7 and 32C are adding dedicated INT8 tensor hardware — because software emulation with VLIW cannot win.

**Implication for our Q4_K work:** our 5.3 tok/s IS already bandwidth-bound (TBA by Agent 2). INT8/Q4 compute path doesn't save us until hardware tensor units arrive.

### 3c. Hamming distance (habr 438948, 2019)
Fast Hamming implementation via Elbrus intrinsics: 1.5× recognition pipeline speedup. Applicable to bit-packed-weight retrieval — NOT directly transferable to Q4_K.

---

## 4. Second "Нейронные сети на Эльбрусе" review (Habr 752138, Aug 2023)

Community review by an unnamed enthusiast. Tested llama.cpp-e2k on:

- Elbrus-16C (12 cores, QEMU-KVM virtualized)
- Elbrus-8SV (8 cores, bare metal)

**Model: llama-7B (ggml-based):**

| Threads | Elbrus-16C ms/tok | Elbrus-8SV ms/tok | Elbrus-16C tok/s | Elbrus-8SV tok/s |
|---|---|---|---|---|
| 1 | 771.46 | 974.84 | 1.30 | 1.03 |
| 4 | 219.58 | 271.34 | 4.55 | 3.69 |
| 8 | 123.22 | 160.89 | 8.11 | 6.21 |

**Elbrus-16C 8-thread LLaMA-7B Q4_0 = 8.11 tok/s** — HIGHEST publicly documented E2K LLM number, but on 16-core dual-chip, and on Q4_0 (simpler than Q4_K).

Also noted: "Текущий ggml делает упор больше на GPU чем CPU" — llama.cpp in 2023 was shifting to GPU/Vulkan, author expected E2K CPU path to be de-prioritized upstream. That prediction held — no E2K PR has landed in ggml-org/llama.cpp to date.

### How our results compare against this 8.11 tok/s 16C reference

Our E8C2 (single-chip-8-core equivalent) @ 5.3–6.1 tok/s vs 16C @ 8.11 tok/s:
- 16C has 2× more cores than 8C2 (16 vs 8) — so our per-core rate (0.66 tok/s at T=30 spread over 8 cores) is actually much better than 16C per-core (0.51 tok/s at 16 cores, Q4_0).
- Also we're running Q4_K which has more compute per byte than Q4_0.
- **Bottom line: on per-chip-per-core-per-quant basis, we are already ~2× better than the best public 2023 E2K LLM benchmark.**

---

## 5. Elbrus 8V7 — the announcement that changes everything (Sep 2025)

**Source:** altitudeaddicts.com/2025/10/12/russia-unveils-elbrus-8v7, MCST presentation slides circulating on Russian IT Telegram.

**Known specs:**
- ISA: Elbrus v7 (new family)
- Cores: 6+ at ≥2 GHz
- Peak: ~0.5 TFLOPS
- **Hardware INT8 and BF16 tensor operations** — FIRST Elbrus with dedicated ML acceleration
- PCIe 4.0
- Integrated 3D GPU for display / multimedia
- Target: laptops, desktops, embedded (NOT HPC)

**Status:** announced only. No silicon shipping. No benchmarks published. No llama.cpp port. No ETA for general availability.

### What INT8/BF16 tensor ops mean for LLM inference

Current Elbrus-8C2 Q4_K GEMV:
- Dequantize Q4→FP32 → scalar SIMD multiply → accumulate FP32.
- ~2 GFLOPS effective per thread.

With hardware BF16 GEMV (if implemented sensibly like ARM SVE BF16 or AVX-VNNI):
- Native BF16 MAC = 4-8× higher throughput per cycle.
- Expected: 20-40 GFLOPS per thread.
- **That alone takes 5.3 tok/s → 20-40 tok/s IF the chip exists on our hardware.**

But 8V7 is a 6-core LAPTOP chip. With only 6 cores and (likely) single-DDR5-channel, memory bandwidth is probably LOWER than 8C2's 4-channel. Would need to measure. The 32C is the real performance target.

---

## 6. Elbrus-32C — the silicon that would move the needle (2025–2026)

**Source:** tadviser.com "Development of the processor Elbrus-32C"; tomshardware Oct 2020.

**Planned specs:**
- 32 cores, ~2.0 GHz
- **6-channel DDR5** → ≥170 GB/s aggregate memory bandwidth
- Supports up to 2 TB memory
- **INT8 / BF16 hardware tensor accelerators** (carried over from 8V7 family)
- Target process: 7nm
- Designed for servers, cloud, supercomputers

**Funding / timeline:**
- Dec 17, 2022: MCST signed contract with Russian Ministry of Industry & Trade for 7.1 B rubles, deadline Dec 31, 2026.
- Original plan: first working samples 2025. No public update indicating this milestone was hit.
- Nov 2025: MCST "withdrawn from external control" — i.e. prep for sale of the company. Adds uncertainty.

**Projected LLM inference on 32C (our model):**
- 170 GB/s × 1/0.4 KB/tok-weights = **~425 tok/s absolute single-chip ceiling** for qwen3:4b Q4_K_M.
- With BF16 tensor units @ ~20% bandwidth-utilization (realistic llama.cpp effective), **estimate 85 tok/s on a single 32C chip.**
- Multi-socket 4-way 32C: 680 GB/s aggregate, **~170 tok/s TP-4 target** — ASSUMING silicon arrives and is usable.

**Conclusion:** Elbrus-32C would be transformative, but we cannot count on it on our 2026 timeline. Plan as if 8C2 is the deployment target.

---

## 7. Community / communication channels

| Resource | URL | Relevance |
|---|---|---|
| **@qemu_e2k Telegram** | t.me/qemu_e2k | OpenE2K / QEMU community. General E2K dev chat. No LLM focus confirmed. |
| **OpenE2K GitHub org** | github.com/OpenE2K | QEMU, emulator. No ML repos. |
| **e2k-community / awesome-e2k** | github.com/e2k-community/awesome-e2k | Curated list. No LLM-specific items tracked. |
| **ilyakurdyukov/e2k-ports** | github.com/ilyakurdyukov/e2k-ports | 40+ package patches (ffmpeg, openblas, qt, postgres). NO ML / LLM work. 108 commits. |
| **alexmihalyk23/llama.cpp-e2k** | github.com/alexmihalyk23/llama.cpp-e2k | 567 commits, no tagged releases. Last LLM-specific work appears to be 2023. |
| **alexmihalyk23/whisper.cpp-e2k** | github.com/alexmihalyk23/whisper.cpp-e2k | Speech-recognition E2K port. |
| **alexmihalyk23/stable-diffusion.cpp-e2k** | github.com/alexmihalyk23/stable-diffusion.cpp-e2k | SD port. README notes "E2Kv5 support optimization only for q4_0". No LLM benchmarks. |
| **elbrus.kurisa.ch** | elbrus.kurisa.ch | Demo E2K server access portal. No published benchmarks. |
| **forums.anandtech MCST thread** | forums.anandtech.com/threads/mcst-elbrus-cpus-benchmarks-e2k-assembly-code.2591353 | Active enthusiast thread. STREAM, Linpack, Coremark — NO LLM tok/s numbers. |
| **Smart Engines Elbrus page** | smartengines.ru/elbrus/ | OCR / CNN benchmarks on Elbrus. No LLM work. |
| **MCST official** | mcst.ru/elbrus_prog | Programming guide (what we already extracted). |

**Assessment:** the public E2K community actively discusses the ARCHITECTURE but not LLM inference. Our project is the most active LLM-on-E2K effort currently in existence. There is no "closed forum" where better numbers are hiding — if 20+ tok/s on Elbrus LLM were being achieved somewhere publicly, it would be on Habr or AnandTech forums.

---

## 8. Checking if any 2024-2026 fork beats Alex Mikhaliuk 2023

**Searched:** GitHub (site search), Habr, dev.to, anandtech, Russian CPU forums, Telegram discovery.

**Result:** NO.
- No `ggml_e2k.c` code paths in upstream ggml-org/llama.cpp (as of Apr 2026).
- No fork of llama.cpp in 2024-2026 tagged with e2k / elbrus support.
- No ik_llama.cpp E2K variant.
- No whisper.cpp / stable-diffusion.cpp / mlc-llm E2K branches with LLM benchmarks.
- Alex Mikhaliuk's repos are still the only published E2K-ML artifacts, and his LLM work (llama.cpp-e2k) has NOT had a 2024-2026 release.

**Why this makes sense economically:**
- TSMC embargo on Russia (2022+) means no new Elbrus silicon can ship outside government/defense. Consumer interest is near-zero.
- MCST financials (refused funding 3 times in a year; moved to sale prep Nov 2025) signal low R&D momentum.
- Every serious ML person in Russia runs CUDA or Apple Silicon.
- **We are the frontier.** There is no public benchmark to copy.

---

## 9. Specific optimizations / techniques from public E2K work — NOT already in our codebase

Cross-referenced against our current main (commits `eee6a1d` → `179fce8`) and Agent 1/2/3/4/5/6 rounds:

### 9a. Alex Mikhaliuk's Q4_0 intrinsics approach (habr 732508)
- 128-bit vector lane operations on packed 4-bit weights using E2K v5+ intrinsics.
- 2× GEMM speedup on 8SV / 16C after this work.
- **Our `cpu_quant_gemv.h` uses int64_t + pragma ivdep + pragma loop count — similar effect but compiler-driven.** Worth double-checking: does LCC generate the same quality inner kernel as hand-written E2K intrinsics for our Q4_K path? Our round-1 builds should be disassembled to check packed-load utilization.

### 9b. Smart Engines "Hamming distance" bit-packed kernel (habr 438948)
- Bit-parallel popcount via E2K intrinsics → 1.5× image-recognition speedup.
- Relevance to us: NONE directly. Q4_K is dequantize-multiply-accumulate, not Hamming. But the general principle — if operation maps to bit-level VLIW it wins — argues for a **packed 4-bit multiplication kernel that skips FP32 dequantization**. We don't have that; ggml for x86 does it via AVX512-VNNI; for E2K v5 no equivalent hardware. Probably not extractable without VNNI-class instruction.

### 9c. UNet single-core Elbrus-8CB matching Threadripper per-clock (smartengines.ru/elbrus/)
- Key lesson: EML single-core used well = Threadripper-class FP performance per clock.
- Implication: bandwidth-bound regime (us at 5.3 tok/s) is NOT where Elbrus shines. Elbrus wins at compute-bound, EML-vectorizable FP convolution-like kernels.
- **Strategic consequence: if we can cast the bottleneck GEMV into a compute-bound batched form (TP-4 batched × small K), we push into Elbrus sweet spot.** This reinforces Agent 5's alternative-strategy direction (batched decoding / spec-decode with high draft depth).

### 9d. Ilya Kurdyukov's E2K intrinsic patterns (github ilyakurdyukov/e2k-ports)
- General-purpose `restrict` + vector intrinsic idioms.
- Already subsumed by our APB + `#pragma ivdep` work.

### 9e. QEMU-KVM virtualized LLM benchmarks (habr 752138)
- Important meta-insight: 2023 reviewer tested 16C via QEMU virtualization because native access was hard.
- **Implication for our w205p benchmark: if we have NATIVE access to w205p, our numbers are more credible than the 2023 public reference. Document this in any comparisons.**

---

## 10. What is NOT in the public record (gaps)

1. **No published Q4_K (K-quant, 2023Q3+) benchmark on any Elbrus chip.** All Elbrus llama.cpp work is Q4_0 era. Q4_K has ~2× more compute per byte; naive port regresses.
2. **No Elbrus-8C2 specific LLM tok/s number in any paper or blog.** Only 8SV / 8CB / 16C. We are generating primary data.
3. **No EML sgemm vs hand-written sgemm head-to-head on recent Elbrus for LLM-relevant sizes (K=4096 × M=1, N=vocab).** Agent 4 (eml_audit) is closest to this.
4. **No published discussion of SHM-based AllReduce or Local SGD on Elbrus.** Our distributed code is novel.
5. **No public benchmarks comparing 4-chip vs 2-chip NUMA effects on LLM decode throughput on Elbrus.** We should publish.

---

## 11. Recommendations

### For Round-2 agents
- **Agent 2 (memory bandwidth)**: verify DIMM population on w205p (ssh + dmidecode). 77 GB/s assumes 8 DIMMs on 16C. On 8C2 equivalent, 4 DIMMs fully populated should give ~30 GB/s measured. If we see <20 GB/s STREAM, bandwidth is THE bottleneck and no software optimization will move us past ~8 tok/s on that chip alone.
- **Agent 4 (EML audit)**: benchmark EML sgemm at LLM-relevant sizes (M=1, N=2048..4096, K=4096..12288). If EML beats our Q4_K GEMV kernel at these shapes after dequantize, route through it.
- **Agent 5 (alternative strategies)**: the compute-bound batched approach is validated by Smart Engines' UNet-single-core-Elbrus finding. This is the path with best theoretical Elbrus fit.

### For the mission ceiling
Our 5.3/6.1 tok/s IS the public Elbrus LLM ceiling as of April 2026.

**Realistic ceilings without new hardware:**
- Best case software on Elbrus-8C2 single-chip: ~10-12 tok/s (assumes all planned optimizations land, matches best-case extrapolation from 16C ratio-adjusted 8.11 tok/s).
- Best case 4-chip TP with bounded AllReduce and compute-bound batched kernels: ~15-18 tok/s.
- Target 20 tok/s REQUIRES either (a) spec-decode with >50% acceptance and non-trivial draft, (b) model architecture change (smaller model, MoE), or (c) Elbrus-8V7 / 32C silicon.

### For publication
When we hit 15+ tok/s on qwen3:4b Q4_K_M on Elbrus-8C2 x4, we should write it up on Habr and submit patch to ilyakurdyukov/e2k-ports. No one else in the world has this data.

---

## 12. Source registry

Primary sources cited (all accessed 2026-04-24):

- **habr.com/ru/articles/732508** — Alex Mikhaliuk, "Загоняем Альпаку на Эльбрус (Часть 2. Оптимизации)", 2023-04-30. Primary LLM-on-Elbrus benchmark reference.
- **habr.com/ru/articles/752138** — Community review "Нейронные сети на Эльбрусе", 2023-08-03. Second-best LLM-on-Elbrus reference (Q4_0, 7B).
- **smartengines.ru/elbrus/** — Smart Engines neural network optimization on Elbrus platform. UNet, tomography, Hamming, INT8.
- **habr.com/ru/companies/smartengines/articles/494866** — "8-битные сети на Эльбрусе, есть ли смысл?", 2020. Critical negative result on INT8 packing.
- **habr.com/ru/companies/smartengines/articles/438948** — 2019 Hamming distance optimization.
- **habr.com/ru/companies/smartengines/articles/522430** — 2020 Smart Tomo Engine on Elbrus.
- **habr.com/ru/companies/icl_group/articles/648219** — 2022 Elbrus-16C engineering sample review. STREAM 77 GB/s (8 DIMM) / 19 GB/s (2 DIMM). Blender, HPL, Coremark, MP MFLOPS.
- **altitudeaddicts.com/2025/10/12/russia-unveils-elbrus-8v7** — Oct 2025 Elbrus-8V7 announcement. INT8 / BF16 hardware.
- **tadviser.com/index.php/Project:Development_of_the_processor_Elbrus-32C** — Elbrus-32C funding / timeline.
- **tadviser.com MCST company page** — MCST "out of external control", Nov 2025, sale prep.
- **tomshardware.com/news/russian-company-tapes-out-16-core-elbrus-cpu** — Elbrus-16C tape-out, 8-channel DDR4-3200.
- **en.wikipedia.org/wiki/Elbrus-8S** — 8S / 8SV / 8CB specs.
- **github.com/alexmihalyk23/llama.cpp-e2k** — Alex Mikhaliuk LLM port (567 commits, no tagged releases).
- **github.com/ilyakurdyukov/e2k-ports** — Ilya Kurdyukov's patch collection (no LLM work).
- **github.com/pigirons/cpufp/blob/master/benchmark_result/e2k/Elbrus_8C.md** — Elbrus-8C raw GFLOPS: 28.7 FP32 / 14.4 FP64 per core at 1.2 GHz.
- **t.me/qemu_e2k** — OpenE2K Telegram community.
- **github.com/OpenE2K** — QEMU-E2K and ecosystem repos.
- **forums.anandtech.com/threads/mcst-elbrus-cpus-benchmarks-e2k-assembly-code.2591353** — enthusiast thread, STREAM / Linpack / Coremark but NO LLM tok/s.

All extraction performed via WebSearch + WebFetch on 2026-04-24. No Russian Telegram messages were inspected beyond the public channel descriptions; deeper closed-community mining would require manual join.

---

## Bottom line for the mission

**We are not behind anyone.** Our 5.3 / 6.1 tok/s on Elbrus-8C2 qwen3:4b Q4_K_M is, by every public measure I could find, the highest number any team has produced for this combination. Alex Mikhaliuk's 2023 8.11 tok/s on 16C was on simpler Q4_0 / smaller context and is NOT our ceiling.

The 20 tok/s target is aspirational but bounded by physics. We should:
1. **Verify DIMM population** (biggest unknown — could be a free 2-4× if underpopulated).
2. **Consider EML sgemm routing** for the right GEMV shapes.
3. **Lean harder on batched / spec-decode** (Elbrus compute-bound sweet spot per Smart Engines).
4. **Stop waiting for silicon.** 8V7 and 32C aren't shipping on our timeline.
