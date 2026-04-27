# Agent 5 — SoA weight repack for multi-row VNNI ILP on E2K 8C2

Round 2's `qpmaddubsh` Q8_0 GEMV capped at 6.5 GOPS (1.91 ms / GEMV
K=2560 N=2432, 0.54× EML). Target **≤ 1.0 ms / GEMV**. Mechanism: lay
weights so one `qpmaddubsh` writes partial dots for several rows at
once, killing the per-block scalar extract.

## 1. Why current path is reduce-bound

```
qpmaddubsh w,a -> i16x8     # only useful work
qpmaddh    p16,1 -> i32x4   # forced fold (no qpidotsbwss on v5)
qpaddw
4× scalar extract + 3 adds  # killer, ~28 cy/block
```
Multi-row unroll didn't help — each row still ended in its own
horizontal reduce. Effective: 5.3 ops/cy (1/3 of raw peak).

## 2. Layout: 4-row interleaved Q8_SoA

Per "super-row" (4 N-rows × 32 K-elements):

```
offset  size  field
  0     16    s0..s3            fp32 scales (d×scale_a, pre-fused)
 16     16    corr0..corr3      fp32: 128 × sum_w × scale_a, pre-fused
 32    128    interleaved int8 weights:
              [r0[0] r1[0] r2[0] r3[0]
               r0[1] r1[1] r2[1] r3[1]
               r0[2] r1[2] r2[2] r3[2]
               r0[3] r1[3] r2[3] r3[3]
               ... 32 K positions × 4 rows = 128 B]
total 160 bytes per (4 rows × 32 K)
```

Old AoS: 4×34=136 B same coverage. Δ=+18 % (scales fp16→fp32 + corr
pre-folded). qwen3:4b = +0.5 GB/replica, ×4 NUMA = +2 GB. Fits.

### Why 4-row

`qpmaddubsh(a,b)` → 8 i16 lanes; each = `a[2k]*b[2k]+a[2k+1]*b[2k+1]`.
With 4-row interleave + activation broadcast `[a[0]×4, a[1]×4, a[2]×4,
a[3]×4]`, each lane's byte-pair stays in one row:
```
lane0 = a[0]*r0[0]+a[1]*r0[1]    lane4 = a[2]*r0[2]+a[3]*r0[3]
lane1 = a[0]*r1[0]+a[1]*r1[1]    lane5 = a[2]*r1[2]+a[3]*r1[3]
lane2 = a[0]*r2[0]+a[1]*r2[1]    lane6 = a[2]*r2[2]+a[3]*r2[3]
lane3 = a[0]*r3[0]+a[1]*r3[1]    lane7 = a[2]*r3[2]+a[3]*r3[3]
```
`qpmaddh(p16, ones)` → 4 i32 lanes, **one per row**. No mixing across
K-loop; accumulate into one v2di per super-row, **one** horizontal
split at the end. 8-row halves K per instruction; 2-row keeps 2
extracts per K-iter. 4-row is the sweet spot.

## 3. Hot loop pseudocode

```c
void q8_soa4_gemv(const Q8SoA4& W, const uint8_t* a_u8,
                  float* y, int K, int N) {
    for (int n = 0; n < N; n += 4) {
        v2di fp_acc = {0,0};      // 4 fp32 lanes
        const uint8_t* gp = W.w + (n>>2)*W.group_stride;
        for (int b = 0; b < K/32; ++b) {
            v2di acc = {0,0};     // 4 i32 lanes (one per row)
            const uint8_t* wb = gp + b*160 + 32;
            for (int k = 0; k < 32; k += 4) {
                v2di a4 = splat_pair_to_4(a_u8 + b*32, k);  // [a[k]*4, a[k+1]*4, a[k+2]*4, a[k+3]*4]
                v2di w4 = *(const v2di*)(wb + k*4);         // 16 B
                v2di p16 = __builtin_e2k_qpmaddubsh(a4, w4);
                v2di p32 = __builtin_e2k_qpmaddh(p16, ONES16);
                acc      = __builtin_e2k_qpaddw(acc, p32);
            }
            // i32 -> fp32, fold scale + sum_w correction (4 SIMD ops/block)
            v2di acc_f = __builtin_e2k_qpistofs(acc);
            v2di sc4   = *(const v2di*)(W.scales + (n>>2)*bpr*4 + b*4);
            v2di cr4   = *(const v2di*)(W.corr   + (n>>2)*bpr*4 + b*4);
            fp_acc = __builtin_e2k_qpfadds(fp_acc,
                       __builtin_e2k_qpfsubs(__builtin_e2k_qpfmuls(acc_f, sc4), cr4));
        }
        float lanes[4]; memcpy(lanes, &fp_acc, 16);
        y[n+0]=lanes[0]; y[n+1]=lanes[1]; y[n+2]=lanes[2]; y[n+3]=lanes[3];
    }
}
```

Inner body = 3 SIMD ops (qpmaddubsh, qpmaddh, qpaddw), chain length 3.
LCC SWP can double-buffer across 2 K-pair iters → 3 ops/cy = 16
byte-MADs across 4 rows = **24 GOPS @ 1.5 GHz**, 4× Round 2 single-row.

K=2560 N=2432: 12.5 Mops; ideal 0.52 ms; realistic 0.7-1.0 ms with
memory + branches — matches EML's 1.02 ms.

## 4. Q4_K: dequantise to Q8_SoA at load

Q4_K's nibble unpack + 6-bit sub-block scales would either need
nibble-interleaving (compute density halves) or preserve the
super-block machinery (extracts come back). Right call:
**dequantise Q4_K → Q8_SoA at model load**.

- Repack ~7 s one-shot (read 2.6 GB, write 3.4 GB).
- 1.32× storage (×4 NUMA = 13.6 GB, fits 125 GB).
- Numerics bit-exact: fold `d×sc×q4 - dmin×m` into fp32 scale
  `d×sc×scale_a` + per-block corr `(dmin×m + 128×d×sc)×sum_q×scale_a`.
- All 8 GEMVs (QKV, gate/up/down, attn_out, output) → same kernel.

The +32 % memory tax is fine: decode is compute-bound today (18 % BW
util), so a 4× faster kernel raises BW util.

## 5. Implementation plan

### 5.1 New file `torch/io/q8_soa_repack.h`

```cpp
struct Q8SoA4 {
    uint8_t* w;          // [N/4, K/32, 128] interleaved int8
    float*   scales;     // [N/4, K/32, 4]   d × scale_a, pre-fused
    float*   corr;       // [N/4, K/32, 4]   (dmin×m + 128×d×sc) × sum_q × scale_a
    int64_t  N, K, group_stride;
};
Q8SoA4 repack_q4k_to_q8soa4 (const void*, int64_t N, int64_t K, int64_t row_stride);
Q8SoA4 repack_q8_0_to_q8soa4(const void*, int64_t N, int64_t K, int64_t row_stride);
void   free_q8_soa4(Q8SoA4&);
void   q8_soa4_gemv(const Q8SoA4&, const uint8_t* a_u8, float scale_a,
                    float* y, const ReplicatedWeight* numa=nullptr);
```

### 5.2 Loader hook (`torch/io/gguf_model.h`)

- `QuantizedWeight` (line 139): add `Q8SoA4 soa; bool has_soa=false;`.
- `load_quantized_to_cpu()` (line 1201) after memcpy at 1234, gate on
  `getenv("PT_Q8_SOA")`. Q4_K → `repack_q4k_to_q8soa4`,
  Q8_0 → `repack_q8_0_to_q8soa4`. Then `free(cpu_data)`.
- NUMA: add sibling `ReplicatedQ8SoA4`, populated per node in parallel.

### 5.3 Dispatcher (`torch/io/cpu_quant_gemv.h`)

`cpu_quant_gemv()` (line 1568): if `qw.has_soa`, call `q8_soa4_gemv`
regardless of `quant_type`. K-slice variant (`cpu_quant_gemv_k_slice`,
line 1518) takes same branch — repack respects K-slicing
(group_stride is row-major). `PT_Q8_SOA=0` falls back.

### 5.4 Build

`__e2k__` ifdef (qpfmuls/qpfsubs/qpistofs/qpmaddubsh all v5 per
FINDINGS.md). x86 dev builds get scalar fallback over the same struct.

## 6. Expected uplift

TP-4, ms/token:

| Section | now | Q8_SoA4 | Δ |
|---|---|---|---|
| gate_up | 65.4 | 22 | -43 |
| ffn_down | 48.9 | 17 | -32 |
| attn_phase QKV | 29.9 | 10 | -20 |
| output_proj | 23.7 | 8 | -16 |
| attn_output | 15.0 | 5 | -10 |
| allreduce + tail | 12 | 12 | 0 |
| **TOTAL** | **211** | **~74** | **-137** |

Ideal `1000/74 ≈ 13.5 tok/s`. With 30 % shortfall (memory stalls, NUMA
cross-traffic), floor **9-11 tok/s** — meets Round 3 target.

Risks: (i) `qpfmuls/qpistofs` may not dual-issue with the qpmaddubsh
chain → ~1.5 ms / GEMV (still 8 tok/s). (ii) `splat_pair` memory
roundtrip ~3 cy/K-pair — mitigable by pre-permuting `a_u8` at activation
quantise time.

## 7. Bottom line

4-row SoA byte-interleave + fp32-fused scales + per-block fp accumulate
is the only software path on 8C2 matching EML FP32 GEMV
(≤ 1.0 ms / GEMV) at 4× useful work/cycle — i32 partial dots stay in
their own lanes for the whole K-loop, killing the horizontal reduce
that capped Round 2. **+5-7 tok/s**, `PT_Q8_SOA=1`, ~600 LoC across
two files, 7 s repack at load. Q4_K and Q8_0 collapse to one kernel.
