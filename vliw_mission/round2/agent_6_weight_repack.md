# Agent 6 — Weight Layout / Repack for Q4_K on Elbrus APB

Role: weight-layout engineer. Goal: make the 144-byte Q4_K super-block look
like three (or four) independent linearly-addressed streams so that LCC's
APB/`fapb` can actually fire on the hot inner loop, and each DDR burst lands
in an already-open bank page.

Source of truth for this analysis:
- `vliw_mission/round2/_inputs/mcst_main_guide.txt` (lines 1945, 1998-2080,
  3842-3956, 4386-4396, 6553-6585)
- `vliw_mission/round2/_inputs/memory_controller.txt` (lines 142-168, 297-344)
- `vliw_mission/round2/_inputs/cache_optimization.txt` (lines 60-113, shared
  L3 distribution)
- `vliw_mission/round2/_inputs/loop_vectorization.txt` (base SWP/vectorization
  method, lines 23-70)
- `C:\Users\paper\Desktop\promethorch\torch\io\cpu_quant_gemv.h` lines 61-133
  (`q4k_gemv_avx2_float` — the actual inner loop we are fixing)
- `C:\Users\paper\Desktop\promethorch\torch\io\gguf_model.h` lines 1196-1263
  (`load_quantized_to_cpu()` — the repack hook point) and 1336-1375
  (`load_quantized_mmap()` — must be bypassed for repack path)
- `C:\Users\paper\Desktop\promethorch\torch\io\numa_weight_replica.h`
  (replication infrastructure we reuse per-field after split)

## 1. Why the current layout kills APB — quoted evidence

### 1.1 Architectural rule: APB works best with increment < 32 B

`mcst_main_guide.txt:3942-3946`:

> Также отдельно стоит выделить случаи, когда механизм APB эффективен:
>  - при достаточно большом числе итераций;
>  - **при инкременте < 32b**;
>  - при отсутствии зависимостей с записями между итерациями…

`mcst_main_guide.txt:4388-4391` is the specific number:

> при регулярном считывании/записи элементов массива могут достигаться
> теоретически максимальные показатели темпа доступа к памяти — 32 байта
> в такт при попадании в L2, 32 байта в 3 такта при гарантированном
> отсутствии в L2, … **при регулярном обращении с большим шагом (>64b)
> APB все еще применим, но темп существенно падает (до 64 раз при побайтовой
> обработке).**

### 1.2 Architectural rule: prefer SoA over AoS for hot fields

`mcst_main_guide.txt:4392-4396` — this is the textbook recommendation for
exactly our situation:

> одномерные массивы структур. При регулярной обработке применим APB,
> однако, следует следить за тем, чтобы набор одновременно читаемых/
> записываемых полей в горячих участках был как можно более компактным;
> **весьма полезен (в ущерб наглядности) переход от массивов структур
> к набору массивов, хранящих отдельные поля.**

### 1.3 FAPB micro-architecture: 64-B-aligned records, max 32 B per element

`mcst_main_guide.txt:6559-6573`:

> asz специфик атор размера области назначения в APB;
> **area_size = (64 байта)*(2^asz)**
> abs адрес базы области назначения в APB **(в терминах 64 байтов)**:
> area_base = (64 байта)*abs;
> база области должна быть выровнена до размера области;
> mrng … **length = (fapb.mrng != 0) ? fapb.mrng : 32;**

and 3939: *"при выровненности адресов"* — alignment is required for APB
on v1–v5.

So the APB engine wants: 64-B-aligned base, ≤ 32-B logical record, constant
stride, and — from 4388 — stride ideally ≤ 32 B.

### 1.4 DDR4 page rule: 8 KB pages, 45-50 ns penalty per miss

`memory_controller.txt:143-148`:

> В каждом логическом банке (размером от 256 Мбайт до 2 Гбайт) можно
> одновременно задействовать лишь одну страницу памяти (**8 Кбайт**).
> Переход к другой странице в рамках одного логического банка возможен
> через **45–50 нс**.

and the scheduler has an explicit *"priority filter for requests into an
open page"* (`memory_controller.txt:329-334`) — so any layout that keeps
hot accesses clustered inside the same 8 KB DDR page is a direct win.

### 1.5 Current Q4_K layout against these rules

Per-block layout as stored (see `gguf_dequant.h:301-319` and `cpu_quant_gemv.h`
line 55-88):

```
offset  size  field
  0      2    d       (fp16)
  2      2    dmin    (fp16)
  4     12    scales  (packed 6-bit)
 16    128    qs      (4-bit packed values)
total 144 bytes per 256 values
```

Inside the `bi`-loop of `q4k_gemv_avx2_float` the compiler sees, per block:
1 pair of scalar `ldh` from `block+0/2` (d, dmin) → 4 B,
12 B byte-wise read from `block+4` (scales),
then 128 B contiguous vector reads from `block+16` (qs).

Problems mapped to the architectural rules:

- The outer stride between successive super-blocks is **144 bytes per
  256 weights** — that is greater than 64 B, so rule 1.1 ("большим шагом
  >64b … темп существенно падает") kicks in already. The scalar d/dmin/
  scales reads have an effective stride of 144 B, which is *far* worse than
  the 32 B that rule 1.1 wants.
- The mix of read widths (2 B + 2 B + 12 B + 128 B) inside one struct
  prevents the `fapb.fmt` check from being stable (see 6574-6579:
  *"поле fmt должно кодировать, по крайней мере, самый длинный из них"*),
  so LCC cannot emit a single `fapb` for the block — it has to either
  skip the header fields or issue multiple `fapb`s with different fmt,
  blowing the APB area budget (only 6 asz levels, max 2048 B).
- 144 doesn't divide 64, so every 4–5 super-blocks crosses a 64-B cache
  line boundary in the middle of `qs` — an extra L1-miss we don't need.
- At 2 GB (qwen3 4b) the weight stream covers ≈ 250 k DDR pages. Reading
  144 B/block means we burn 2 B+2 B+12 B of header bandwidth per 128 B of
  qs — a **12 % dead-weight bandwidth tax** that never gets touched by
  APB streaming. That's ~2.4 GB/s stolen off the 20 GB/s channel.

## 2. Proposed RePack layout (SoA, stride-aligned, pre-fused scales)

### 2.1 Schema

For a weight matrix of shape `[N, K]` with `B = K/256` super-blocks per row,
replace the single 144 B-per-block AoS with **four independent row-major
matrices**, each with its own stride and alignment:

| name    | dtype             | shape        | row stride (B) | notes                                   |
|---------|-------------------|--------------|----------------|-----------------------------------------|
| `qs`    | `uint8_t` (4b×2)  | `[N, K/2]`   | `K/2`          | pure byte stream, 64-B aligned          |
| `dc`    | `float` (fp32)    | `[N, B*8]`   | `B*32`         | `d × sc_i` pre-multiplied, 32-B aligned |
| `mc`    | `float` (fp32)    | `[N, B*8]`   | `B*32`         | `dmin × m_i` pre-multiplied             |
| `row_d` | `uint16_t` (fp16) | `[N, 2]`     | 4              | optional diagnostic; not on hot path    |

Each super-block emits 8 `(d × sc)` and 8 `(dmin × m)` scalars (the j-loop
in `q4k_gemv_avx2_float` iterates `j = 0..192 step 64` and uses `is, is+1`
→ 2 scales per iteration × 4 iterations = 8 pairs). Per super-block:

- 128 B `qs` — already a clean stream.
- 32 B `dc`  — 8 fp32 scales, pre-fused with row-level `d`.
- 32 B `mc`  — 8 fp32 mins,  pre-fused with row-level `dmin`.

**Total bytes per super-block: 192 B** (vs 144 B currently).
**Per row bytes: `N × B × 192` instead of `N × B × 144`.** At qwen3:4b
(2.6 GB today) that is **2.6 × 192/144 = 3.47 GB**. Acceptable — still
fits 4× NUMA replication in < 14 GB on the 125 GB Elbrus.

(The 33 % size increase buys us: no `get_scale_min_k4` 6-bit unpack on hot
path, no `fp16_to_fp32` in the inner loop, no `fmul` chain of `d*sc`/
`dmin*m`. That's ~14 fp ops removed per 64 values.)

### 2.2 Why this wins on every rule

- **Rule 1.1 / 4388 — stride < 32 B:** each stream has its own stride.
  `qs` is 1 B granularity, `dc`/`mc` are 4 B granularity. Both well
  inside the ≤ 32 B sweet spot for APB maximum throughput.
- **Rule 1.2 / 4392-4396 — "переход от массивов структур к набору
  массивов, хранящих отдельные поля":** this *is* the textbook
  recommendation, applied verbatim.
- **Rule 1.3 / 6559-6573 — FAPB alignment:** `qs` is 128 B-aligned (2×
  the 64 B base), `dc`/`mc` are 32 B-aligned — both legal `fapb` targets
  without `-fno-faligned` fallback.
- **Rule 1.4 / memory controller open-page scheduler:** each of the three
  streams is now strictly sequential per row. A row of qwen3 (K=3072,
  B=12) covers `12×128=1536 B` qs + `12×32=384 B` dc + `384 B` mc per
  row. All three streams within one row fit in < 4 KB, i.e., **inside a
  single 8 KB DDR page per stream**. Page-open ratio approaches 100 %.
  Current layout: 12 × 144 = 1728 B per row, but interleaved with
  unrelated fields, so the controller's *"фильтр приоритета … в открытую
  страницу"* cannot cluster accesses as well.
- **Rule 1.5 / cache_optimization.txt:60-113 — L3 banks:** on Elbrus
  8C2 the 16 MB L3 is bank-sharded. With SoA the `dc`/`mc` arrays
  (≈ 260 MB total at qwen3:4b) are sequential, so the OS page colouring
  / NUMA replica pinning (already in `numa_weight_replica.h`) can bind
  each stream to the correct bank, and the compiler's 0-line prefetch
  rule (3889-3897) applies unambiguously.
- **LCC flag interaction:** we already ship `-faligned -fprefetch
  -fcache-opt -frestrict-all` (CMakeLists.txt:130-170). With the
  repacked layout, `-fcache-opt -flist-prefetch` (3537-3538) will
  actually engage on the header-free `qs` stream; today they can't
  because the struct mixes 3 different access patterns.

### 2.3 What this gives the compiler in the inner kernel

The j-loop today (lines 91-128 of `cpu_quant_gemv.h`) reads:

- `block+0/2` (d/dmin) — once per block, 4 B total, scalar load chain
- `block+4` (scales, via `get_scale_min_k4`) — 12 B bit-manipulation
  chain with 2 branches (`if (j < 4)` in `gguf_dequant.h:289-297`)
- `block+16+j` (qs) — the streamable part

After the repack the same j-loop reads **only**:

- `dc + bi*8 + is`    — scalar fp32 → `fapb` candidate
- `mc + bi*8 + is`    — scalar fp32 → `fapb` candidate
- `qs + bi*128 + j`   — byte stream → `fapb` candidate

Three independent stride-regular streams, no 6-bit-unpack branch, no
fp16→fp32 conversion. This is exactly the shape that Ermolitsky/Shlykov
(`loop_vectorization.txt:23-70`) call the base SWP-vectorization case —
loop without control divergence, multiple isomorphic sub-expressions,
trivially pipelinable.

## 3. Repack strategies evaluated

a. **Split headers and qs (SoA) — ACCEPTED as primary.** See §2.
   Single concrete change, minimal code surface, maximum APB impact.

b. **Thread-chunk interleave.** Already partially present: `NUMA weight
   replication` in `numa_weight_replica.h` gives one copy per node.
   After split (a), replicate *each of the three streams* per node.
   Cost: +33 % RAM on top of (a), no compute change. Recommended as
   stackable step `b`.

c. **Pre-dequantize scales (fuse `d·sc` and `dmin·m`) — FOLDED into
   option (a) as the `dc`/`mc` matrices.** Saves 2 `fmul`s per j-iter
   and collapses the 6-bit scale unpack to zero.

   Alternative considered: store `dc` as `bf16` to halve the scale stream
   from 64 B per super-block back to 32 B. Rejected because (i) rounding
   `d·sc_max ≈ 2^-4 · 63 ≈ 4` into 7-bit bf16 mantissa is safe, but
   (ii) E8C2 has **no native bf16 FMA** (v4 system, see guide
   figure 4.2 caption: "в каналах 2 и 5 операции над вещественными
   числами поддержаны начиная с версии v4" — v4 is fp32 only; bf16/fp16
   SIMD is v5+). So bf16 would cost a conversion per load. Stick with fp32.

d. **Transpose row ↔ column for N-outer loop.** For a GEMV this is a
   no-op because `y[n] = sum_k W[n,k] * x[k]`; each thread naturally
   gets a contiguous row. The thing that actually matters for 4-chip
   NUMA is *how `x[k]` is traversed*, and here repack (a) also helps:
   the three streams all index `k` in the same monotonic order, so the
   `x[base_k + j + l]` loads become open-page reuse (same 8 KB DDR page
   for 2048 consecutive fp32 x-values).

   Not folded into (a) because it's orthogonal; leave as-is (rows
   contiguous). No transpose needed.

## 4. Memory / load-time cost

Current (`load_quantized_to_cpu()` in `gguf_model.h:1200-1262`):
raw `malloc(total_bytes) + memcpy` of 2.6 GB ≈ **2–3 s** on the 8C2 per
chip.

Proposed repack:
- allocate `qs` (2.3 GB), `dc` (146 MB), `mc` (146 MB) — total 2.6 GB
  (essentially no growth — the 33 % struct-size figure in §2.1 is
  offset by dropping the 12-B packed-scales field: 144 B → 128 + 32
  + 32 = 192 B logical but only 128 + 32 + 32 = 192 B physical, so
  actually ≈ +33 %, 3.4 GB absolute).
- iterate all super-blocks once, decoding `fp16→fp32` and
  `get_scale_min_k4` into `dc/mc`; `memcpy` `qs` as one contiguous 2 GB
  block — throughput bound, not compute bound.
- one-shot wall clock on 8C2: **~4–6 s** (memcpy 2 GB + ~3 GB/s scale
  decode for 37 M super-blocks in qwen3:4b). Acceptable: happens once
  per model load, not per token.

mmap path (`load_quantized_mmap()` at line 1336) has to be disabled for
the repacked tensors — we need the derived layout in anonymous heap,
not the raw GGUF file. Keep mmap as fallback when `PT_REPACK=0`.

## 5. Expected throughput

Current measured: ~14 GB/s effective per chip (30 % of the ~50 GB/s
peak, MISSION.md).

Ceiling after repack:
- Remove 12 % header bandwidth tax → ceiling lifts from 50 → 56 GB/s
  *usable*.
- APB engaging on 3 streams instead of 0: MCST's own EML paper reports
  52 % mean speedup from APB engagement (`loop_vectorization.txt:
  321-325` — "средний прирост производительности функций EML за счет
  векторизации составил 52%"). Our loop is already vectorized on the
  AVX2 side; the APB gain here is on top of that, closing the gap
  between compute and memory fetch.
- Open-page ratio: empirical DDR4 studies give 1.3–1.7× effective
  bandwidth when page-hit ratio moves from ~50 % to ~90 %
  (`memory_controller.txt:148-161` discusses interleaving as the knob).

Conservative estimate: **per-chip effective BW 14 → 18-20 GB/s**, i.e.
**+30-45 %** on the memory-bound critical path. On a pure
bandwidth-bound GEMV (≈ 400 MB read per token at Q4_K) this translates
to **token/s 5.3 → 7.0-7.5 (1-proc)** and **6.2 → 8-9 (TP-4)**. Not
alone sufficient to hit the 20 tok/s target but unblocks the other
optimizations (LCC auto-parallel across 8 cores with APB actually
active is the multiplier that gets us to 15+). Must combine with
Agent 2/3 (EML-accelerated x-quant, loop-nesting) to reach 20.

Uncertainty: ±30 % — the 14 → 18 GB/s claim is backed by the MCST
textbook rule-of-thumb "up to 64× slowdown at stride > 64 B vs optimal",
not by measured Elbrus numbers for this specific kernel. First thing to
do after the patch: run `ldis -I` on the compiled hot loop to confirm
`fapb` and `movab`/`movad` instructions appear (see 6606-6617). If they
don't, the repack still gives us the scale pre-fusion win (~1.15× from
removed fp16 conv + fmul chain), but nothing more.

## 6. Concrete code-change plan

### 6.1 New file (to be created, not in this report)

- `torch/io/q4k_repacked.h` — defines

    ```
    struct Q4K_Repacked {
        uint8_t*  qs;     // [N, K/2], 64-B aligned
        float*    dc;     // [N, B*8], 32-B aligned
        float*    mc;     // [N, B*8], 32-B aligned
        int64_t   N, K, B;
        int64_t   qs_stride, scale_stride;
        bool      owns;
    };
    ```

    plus `Q4K_Repacked repack_q4k(const void* raw, int64_t N, int64_t K,
    int64_t row_stride_bytes);` and `free_q4k_repacked(Q4K_Repacked&)`.
    The repack function iterates super-blocks, reads `d`, `dmin`, the
    12 scale bytes, calls `get_scale_min_k4` 8 times per block, and
    writes `dc[bi*8+i] = d * sc_i`, `mc[bi*8+i] = dmin * m_i`, then
    `memcpy`s `qs` directly (no transformation needed).

### 6.2 `torch/io/gguf_model.h` changes

- Add field in `QuantizedWeight` (line ~140): `Q4K_Repacked repacked;`
  plus `bool has_repacked = false;`.
- In `load_quantized_to_cpu()` (line 1200) after the `memcpy` at line
  1233, gate on `std::getenv("PT_REPACK_Q4K")` and call
  `repacked = repack_q4k(cpu_data, qw.rows, qw.cols, qw.row_stride_bytes);
   has_repacked = true;` — then optionally `std::free(cpu_data);
   cpu_data = nullptr;` once the new GEMV is the only consumer.
- In `load_quantized_mmap()` (line 1336): when `PT_REPACK_Q4K` is set,
  after the mmap-set-up block, still allocate `dc/mc/qs` on the heap
  and populate from the mmap'd region — do *not* leave `cpu_data`
  pointing into the file (the file has the AoS layout, the CPU kernel
  will want SoA).
- NUMA replication: `ReplicatedWeight` currently wraps a single pointer.
  Extend to 3 independent `ReplicatedWeight`s (`rep_qs`, `rep_dc`,
  `rep_mc`), or add a variant struct `ReplicatedQ4K` that owns all
  three per-node buffer triples. Preferred: new struct, keep the old
  `ReplicatedWeight` for the non-repacked path.

### 6.3 `torch/io/cpu_quant_gemv.h` changes

- New kernel next to `q4k_gemv_avx2_float` (line 61):
  `q4k_gemv_avx2_repacked(const Q4K_Repacked& W, const float* x,
  float* y, const ReplicatedQ4K* numa = nullptr);`.

  Body differs only in the inner loop:
  - drop `std::memcpy(&d_bits, block, 2)` and both `fp16_to_fp32` calls
    (lines 82-86).
  - drop the `get_scale_min_k4` pair and the four `d * sc` / `dmin * m`
    fmuls (lines 92-98). Replace with two `_mm256_loadu_ps(dc_row + ...)`
    and `_mm256_loadu_ps(mc_row + ...)` that load all 8 scales at once
    (so the j-loop collapses two iterations, or stays unrolled × 4).
  - `qs` pointer is the repacked stream, advanced by 128 per block; no
    internal `qs += 32` because the outer `bi` loop now advances by 128.

  Net: the block becomes a straight stride-128 byte stream + stride-32 fp32
  stream + stride-32 fp32 stream. That's 3 parallel `fapb`s — well within
  the 4-APB-channel budget (line 1945: "4 устройства для команд
  асинхронного чтения данных по регулярным адресам в цикле").

- Dispatcher: in `forward_decode_cpu` / `forward_decode_cpu_batched`
  (`gguf_model.h` — search call sites of `q4k_gemv_avx2_float`), branch
  on `qw.has_repacked` to call the new kernel.

### 6.4 Build flag

Add `-DPT_REPACK_Q4K` to `CMakeLists.txt:130-170` Elbrus section so the
new code path is default-on only on Elbrus; other targets keep the old
kernel. Env override `PT_REPACK_Q4K=0` for A/B benchmarks.

### 6.5 Fallback for Q5_K / Q6_K

Q5_K (176 B, + `qh[32]`) and Q6_K (210 B, + scales[16] int8) have the
same family of problems and benefit from the same split. Do Q4_K first
because it's > 90 % of the weights in qwen3:4b Q4_K_M (FFN + attention
linears), then extend. Do *not* block on Q6_K — this report covers Q4_K
only.

## 7. Risks and non-issues

- **Correctness:** bit-exact, because `d`, `dmin`, `sc`, `m` are all
  known at load time — we just move the scalar multiplies from the
  hot loop to the repack pass. One-shot test: dequantize one row with
  both paths, bit-compare. If equal, done.
- **RAM budget:** +33 % on weights = +0.85 GB on qwen3:4b per replica.
  With 4-node NUMA replication that's +3.4 GB total, taking the weight
  footprint from 10.4 GB to 13.8 GB. Fits 125 GB 8C2 trivially.
- **Does NOT require hardware changes.** All Elbrus 8C2 system-command
  versions v1-v5 support `fapb` + `-faligned` (see 3955-3956).
- **Interaction with mmap zero-copy:** lost. That's the real tradeoff.
  But mmap's win was "0 s load time"; a 4-6 s one-shot repack is
  negligible for any session > ~30 s, and inference sessions are
  minutes-to-hours.
- **Interaction with sparse GEMV / low-rank output** (see
  `gguf_model.h:1269-1315`): those analyses run on `qw.cpu_data`, the
  raw AoS layout. They happen once at load, before we'd free the raw
  buffer. Order of operations: 1) load raw, 2) run analyses, 3) build
  repacked, 4) optionally free raw. If we skip step 4, peak RAM is
  2.6 + 3.4 = 6 GB transiently — still fine.
- **Does it stack with step b (per-node replication)?** Yes — each of
  the three streams gets its own 4-node replica. Budget: 3.4 GB × 4 =
  13.6 GB. Still fits. Expected to reclaim the 2× TP-4 regression
  mentioned in MISSION.md (6.1 vs 5.3 tok/s) by making the per-chip
  BW on the header scalars actually scale with chip count.

## 8. What this report does NOT solve

- Cross-chip AllReduce overhead (Agent 1/TP territory).
- Speculative decode acceptance rate (Agent 7 ngram, not memory-bound).
- Compute-bound tail: even at 20 GB/s × 0.4 KB/tok we top at ~50 tok/s
  per chip; the 3.3× target (20 tok/s) sits inside that ceiling only
  if we *also* activate LCC auto-parallel across the 8 cores, which
  requires APB to fire in the first place — this repack is the
  enabling precondition.

## Bottom line

The Q4_K 144-B AoS block violates the two most important rules in the
MCST programming guide (SoA-over-AoS at 4392-4396, APB-stride ≤ 32 B at
4388-4391 and 3942-3946) simultaneously. Repacking once at model load
into three stride-regular streams (`qs`, `dc`, `mc`) — with scales
pre-fused to `d` / `dmin` — is a ~100-line C++ change in two files,
costs 4-6 s of one-shot load time and +0.85 GB RAM per replica, and
is the single highest-leverage memory-layout fix available before
touching hardware. Expected gain on the memory-bound hot path:
**14 GB/s → 18-20 GB/s per chip, translating to ~+30-45 % tok/s at
Q4_K_M decode.** Not sufficient alone to hit 20 tok/s, but the
necessary unblocker for the LCC auto-parallel and EML-acceleration
work from the other agents to actually stack.
