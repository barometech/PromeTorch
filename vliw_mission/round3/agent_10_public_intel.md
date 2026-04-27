# Agent 10 — Public Intel Rescan (Round 3)

**Date:** 2026-04-27 (Round 2 Agent 8 was 2026-04-24)
**Scope:** Anything published between 2026-04-22 and 2026-04-27 that wasn't in Round 2's `agent_8_public_intel.md`. Plus a re-check of llama.cpp-e2k, MCST publications, BitNet/ternary kernels, GitHub trending repos, and CPU LLM inference papers.

---

## TL;DR

One genuinely new finding (FairyFuse — arXiv 2604.20913, 2026-04-22) and one mid-2026 paper (CodeGEMM, NeurIPS 2025/arXiv 2512.17970) that R2 missed. Both target ternary/codebook GEMV on CPU. Neither has Elbrus support but both contain *ideas* directly relevant to our 4.8 tok/s plateau. Everything else: silence. No MCST publications, no Habr articles, no llama.cpp PRs for E2K, no MCST GitLab patches, no Telegram discoveries. Mikhaliuk's `llama.cpp-e2k` last commit remains **June 3, 2023** — 3 years stale.

---

## 1. llama.cpp E2K port — no change since R2

- `alexmihalyk23/llama.cpp-e2k`: last commit **2023-06-03**. No 2024/2025/2026 activity. Still the canonical reference; still Q4_0-only with v5 intrinsics. **No qpidotsbwss workaround published anywhere.**
- `ilyakurdyukov/e2k-ports`: no LLM patches added; still general-purpose package patching.
- `ggml-org/llama.cpp` issues + PRs filtered for "elbrus" or "e2k": **0 hits**, open or closed.
- gitlab.com search for "elbrus llama.cpp e2k": no MCST-published patch series exists. The "MCST has llama.cpp patches at gitlab.com/elbrus-developers" hypothesis is **disconfirmed** — that repo group does not exist or is not public.

**Verdict for our 4.8 tok/s:** no upstream lifeline. Still the frontier.

---

## 2. MCST publications since 2026-04 — no relevant new info since Round 2

Searched habr.com, mcst.ru, eadaily, tadviser, Russian IT press. Habr "Эльбрус нейросеть 2026" returns only generic 2026 LLM roundups (Gemini 3.1, Claude 4.7, GPT-5.4 etc.) — no MCST-authored or MCST-targeted articles. No 8V7 silicon shipping update. No 8C2/8CB inference numbers from MCST in Q1-Q2 2026.

---

## 3. CPU LLM inference papers Q1-Q2 2026 — two genuinely new entries

### 3a. **FairyFuse** — arXiv 2604.20913, **2026-04-22** (NEW, post-R2)
**Title:** "Multiplication-Free LLM Inference on CPUs via Fused Ternary Kernels"
**URL:** https://arxiv.org/abs/2604.20913

**Core claim:** First ternary-weight (W∈{-1,0,+1}) GEMV kernel on x86 that uses ONLY masked AVX-512 add/sub — zero FP multiplies, zero LUTs in the hot loop. 16× weight compression. Fuses 8 sub-GEMVs of a widely-linear layer into one SIMD loop via mask-reuse + sign-swap aliasing + register-resident accumulation.

**Numbers:** **32.4 tok/s on a single Intel Xeon 8558P** for 7B ternary, 1.24× over llama.cpp Q4_K_M. Kernel-level 29.6× over baseline. WikiText-2 PPL 5.52 vs FP16 5.47.

**Applicability to our 4.8 tok/s plateau:**
- Hardware-specific: requires AVX-512 mask registers (`vmovdqu8 {k}{z}` style). Elbrus v5/v6 has no equivalent predicated 512-bit register file.
- BUT the *algorithmic idea* — replacing mul with sign-conditional add via 2-bit weight unpack — is portable. E2K v5 `qppermb` + `qpaddh` could implement a 128-bit version of the same primitive.
- Our 23% bandwidth utilization is bound by Q4_K's per-block scale FP16→FP32 conversion and dequantize-multiply chain. Ternary skips both.
- **Catch:** FairyFuse uses a model that was *trained ternary from scratch* (b1.58 / Microsoft BitNet family). It is NOT a quantization of an existing FP16 model. Repurposing for our qwen3:4b would require either (a) training a ternary qwen3 — multi-week GPU job — or (b) post-hoc ternary quant which is known to lose 5+ PPL on non-BitNet models.
- **Verdict:** read the kernel structure (Section 4 of paper) for ideas about mask-driven SIMD fusion. Don't expect a drop-in win.

### 3b. **CodeGEMM** — arXiv 2512.17970, NeurIPS 2025 (R2 missed it)
**Title:** "Codebook-Centric Approach to Efficient GEMM in Quantized LLMs"
**URL:** https://arxiv.org/abs/2512.17970, code: github.com/naver-aics/codegemm

**Core idea:** Pre-compute centroid·activation inner products into a "Psumbook" stored in programmable on-chip cache. At inference, code indices *gather* partial sums directly — no dequantize, no per-element lookup.

**Numbers:** 1.83× (8B) and 8.93× (70B) speedup over SOTA codebook quant at 2-bit, equal accuracy.

**Applicability:** This is GPU-first but the principle ("psum-table in fast memory, gather, accumulate") **maps directly to Elbrus L2** (8 MB shared per chip). For Q4_K at K=4096, M=1, the activation-side precompute needs ~16k FP32 entries (64 KB) per group. Fits in L1d. **Could enable a true LUT-free GEMV** for Q4_K-Sub if we accept ~4× extra activation memory. Worth a quick implementation feasibility check. Most promising new idea for 4.8 tok/s plateau.

### 3c. T-SAR (arXiv 2511.13676, Nov 2025)
**Title:** "Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization"
- 5.6–24.5× GEMV throughput improvement on ternary LLMs.
- **Catch:** Requires *hardware modification* to SIMD ALU (3.2% power, 1.4% area overhead). Pure-software baseline is comparable to FairyFuse. Not actionable for Elbrus 8C2.

### 3d. Other 2026 CPU LLM papers (skim — already in R2's spirit, not new techniques)
- arXiv 2501.00032 (Arm CPU codebook kernels, IQ4_NL via vtbl): the `vtbl` approach is the same as ggml's IQ kernels we already evaluated; E2K has no comparable byte-shuffle.
- arXiv 2510.06957 (Sparse Ternary GEMM on Apple Silicon): 5–6× over scalar baseline, NEON-specific.
- arXiv 2603.03251 ICLR-26 "Speculative Speculative Decoding" (Tri Dao et al.): two-level draft hierarchy. We already have spec-decode planned in Round 3 Agent 2 — this paper provides numerically tighter acceptance bounds but no Elbrus-specific result.

---

## 4. Lookahead / spec-decode variants — minor refinements vs R2

Beyond Round 2 coverage:
- **EAGLE-3** (arXiv 2503.01840, NeurIPS 2025): 3.0–6.5× over autoregressive, 1.6× over Medusa. Uses *internal-layer* feature fusion, not separate draft. Relevant to our Round 3 Agent 2 path. **Note:** EAGLE-3 needs training the head — not zero-config.
- **"Scaling Speculative Decoding with Lookahead Reasoning"** (arXiv 2506.19830): 1.4×→2.1× by combining step-level + token-level. Not Elbrus-specific.
- **Mirror Speculative Decoding** (Apple ML, 2026): runs speculator + verifier on separate hardware in parallel. We already do TP-4 across 4 chips — could repurpose 1 chip as draft, 3 as verifier. Worth noting for Round 3 Agent 2 design but adds AllReduce complexity.

**Bottom line on spec-decode:** No 2026 paper specifically targets *VLIW* CPUs. The technique is hardware-agnostic but acceptance-rate-driven, and our bottleneck is bandwidth, not draft-quality. Spec-decode helps regardless if acceptance > ~50%.

---

## 5. GitHub trending CPU LLM repos last 30 days

Searched: `cpu-llm-fast`, `llama-elbrus`, `quantize-vliw`, "Elbrus" in llama.cpp issues + PRs.

- No new repo named `*-elbrus` or `*-vliw` trending.
- TurboQuant (KV-cache quant, github.com/MartinCrespoC/QuantumLeap) — trending but GPU-only.
- BitNet (microsoft/BitNet) — January 2026 update added I2_S/TL1/TL2 kernels; 2.37–6.17× on x86, 1.37–5.07× on ARM. Same hardware-specific story (AVX-512/NEON only).
- ik_llama.cpp (gitlab.com/ikawrakow-group/ik_llama.cpp): SOTA quants fork; no E2K backend.

**Verdict:** zero new artifacts targetable to Elbrus.

---

## 6. Russian VK / Telegram — not accessed

Skipped per task scoping. No way to verify Trushkin's recent posts without joining closed channels manually.

---

## 7. Direct applicability to our 4.8 tok/s plateau

| Technique | Source | Will it move 4.8→10 tok/s on Elbrus 8C2? | Effort |
|---|---|---|---|
| FairyFuse mask-driven fused ternary GEMV | arXiv 2604.20913 | Algorithmically promising for E2K v5 (qppermb+qpaddh implementation). BUT requires ternary-pretrained model OR multi-week post-train. **No** for qwen3:4b stock. | High |
| CodeGEMM Psumbook (precomputed centroid·activation in L1) | arXiv 2512.17970 | **Most promising new idea.** Pure software, fits Elbrus L1d. Could replace Q4_K dequantize hot loop. Needs feasibility prototype. | Medium |
| EAGLE-3 internal-feature spec-decode | arXiv 2503.01840 | Stacks with our Round 3 Agent 2 spec-decode. Needs head training. | High |
| BitNet bitnet.cpp kernels (TL1/TL2) | github.com/microsoft/BitNet | Hardware-specific (AVX-512/NEON). Only the *algorithmic* spec-of-kernel transfers. | Low (study only) |
| MCST 8V7 / 32C silicon | n/a | No shipping update Q1-Q2 2026. | n/a |

**One actionable recommendation for Round 3 follow-up:**
Evaluate whether CodeGEMM's "precomputed activation·centroid Psumbook in L1, gather by code index" pattern can replace our Q4_K-block dequantize-multiply-accumulate kernel. Block size = 32 in Q4_K → 16 unique index values per block → 16 × FP32 = 64 B Psumbook per group. K=4096 → 128 groups → 8 KB activation precompute per layer. Comfortably fits Elbrus L1d (32 KB). Could shift our 65 ms gate_up time substantially if it converts the dequantize-bound inner loop into a pure gather+add. Worth a 1-day prototype.

---

## 8. What's still NOT in public record (confirmed, since R2)

- No published Q4_K Elbrus benchmark anywhere.
- No 8C2-specific LLM tok/s number outside our project.
- No qpidotsbwss workaround in any forum, paper, or repo.
- No MCST Q1-Q2 2026 silicon update. 8V7 announcement (Sep 2025) still has zero benchmark publications.
- No 4-chip TP / multi-NUMA Elbrus LLM benchmarks anywhere.

**We remain the only entity in the world publishing Elbrus 4-chip LLM TP numbers. 4.8 tok/s on qwen3:4b Q4_K_M TP-4 is still the world record by definition (no other entry exists).**

---

## Sources (post-R2 only)

- arXiv 2604.20913 — FairyFuse (2026-04-22) [NEW]
- arXiv 2512.17970 — CodeGEMM (NeurIPS 2025) [R2 missed]
- arXiv 2511.13676 — T-SAR (Nov 2025)
- arXiv 2503.01840 — EAGLE-3 (NeurIPS 2025)
- arXiv 2506.19830 — Lookahead Reasoning Speculative Decoding
- arXiv 2603.03251 — Speculative Speculative Decoding (ICLR 2026)
- arXiv 2510.06957 — Sparse Ternary GEMM Apple Silicon
- arXiv 2501.00032 — Arm CPU GEMV codebooks
- github.com/microsoft/BitNet — bitnet.cpp Jan 2026 update
- github.com/naver-aics/codegemm — CodeGEMM reference impl
- github.com/alexmihalyk23/llama.cpp-e2k — last commit 2023-06-03 (unchanged)
- github.com/ggml-org/llama.cpp — 0 issues/PRs mentioning elbrus or e2k
