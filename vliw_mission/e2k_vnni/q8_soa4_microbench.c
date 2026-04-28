// Q8_SoA4 microbenchmark on E2K 8C2 (per Round 3 Agent 5 design).
//
// Layout (SIMD-friendly): for each K-group of 4 elements × 4 rows = 16 bytes:
//   bytes 0..3  = row0's [w[k+0], w[k+1], w[k+2], w[k+3]]
//   bytes 4..7  = row1's [w[k+0], w[k+1], w[k+2], w[k+3]]
//   bytes 8..11 = row2's
//   bytes 12..15= row3's
//
// Activation broadcast: per K-group of 4, store [a[k..k+3]] repeated 4 times.
//
// qpmaddubsh(s_v=w_v, u_v=a_v) gives 8 i16 lanes:
//   lane 0 = a[k+0]*w_r0[0] + a[k+1]*w_r0[1]   (row 0, pair k,k+1)
//   lane 1 = a[k+2]*w_r0[2] + a[k+3]*w_r0[3]   (row 0, pair k+2,k+3)
//   lane 2 = a[k+0]*w_r1[0] + a[k+1]*w_r1[1]   (row 1, pair k,k+1)
//   lane 3 = a[k+2]*w_r1[2] + a[k+3]*w_r1[3]   (row 1, pair k+2,k+3)
//   lanes 4-7: rows 2-3
//
// qpmaddh(p16, ONES16):
//   out 0 = lane0 + lane1 = row 0 sum over k..k+3
//   out 1 = lane2 + lane3 = row 1 sum
//   out 2 = lane4 + lane5 = row 2
//   out 3 = lane6 + lane7 = row 3
// Result: 4 i32 lanes, ONE per row. No horizontal reduce ever.
//
// Build: lcc -O3 -march=elbrus-v5 -I/usr/include/eml q8_soa4_microbench.c
//        -leml_algebra_mt -lm -o q8_soa4_microbench

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

typedef long long v2di __attribute__((vector_size(16)));

static const v2di ONES16 = {0x0001000100010001LL, 0x0001000100010001LL};

// =============================================================================
// Q8_SoA4 layout — per super-row (4 N-rows × 32 K-elements):
//   16B 4× fp32 d_w (per-row scale; 1 per row)
//   16B 4× fp32 dmin_m (per-row min; 1 per row, = 0 for Q8_0)
//   16B 4× int32 sum_q (per-row sum of int8 weights in this block)
//   128B byte-interleaved weights (8 K-groups × 16B/group, layout above)
// Total: 176 bytes per super-row.
// =============================================================================
#define SOA4_GROUP_BYTES 176

typedef struct {
    uint8_t* mem;
    int64_t  N, K, group_stride;
} Q8SoA4;

static Q8SoA4 q8_soa4_alloc(int64_t N, int64_t K) {
    int64_t bpr = K / 32;
    int64_t gpr = N / 4;
    Q8SoA4 w;
    w.N = N; w.K = K;
    w.group_stride = bpr * SOA4_GROUP_BYTES;
    w.mem = (uint8_t*)aligned_alloc(64, gpr * w.group_stride);
    return w;
}
static void q8_soa4_free(Q8SoA4* w){ free(w->mem); w->mem = NULL; }

static void q8_soa4_pack_group(Q8SoA4* w, int64_t group_idx,
                                const int8_t* src_rows[4],
                                const float* d_w_per_block[4],
                                const float* dmin_m_per_block[4]) {
    int64_t bpr = w->K / 32;
    uint8_t* gp = w->mem + group_idx * w->group_stride;
    for (int64_t b = 0; b < bpr; b++) {
        uint8_t* sb = gp + b * SOA4_GROUP_BYTES;
        float* scales = (float*)(sb + 0);
        float* dmins  = (float*)(sb + 16);
        int32_t* sum_q = (int32_t*)(sb + 32);
        int8_t* W = (int8_t*)(sb + 48);
        for (int r = 0; r < 4; r++) {
            scales[r] = d_w_per_block[r][b];
            dmins[r]  = dmin_m_per_block ? dmin_m_per_block[r][b] : 0.0f;
            int sum = 0;
            for (int k = 0; k < 32; k++) sum += (int)src_rows[r][b*32 + k];
            sum_q[r] = sum;
        }
        // Layout: per K-group of 4 elements × 4 rows = 16 bytes.
        // For K-group g (covers k = g*4..g*4+3), 8 K-groups per 32-block:
        //   bytes 0..3 = row0[g*4..g*4+3]
        //   bytes 4..7 = row1[g*4..g*4+3]
        //   bytes 8..11= row2[g*4..g*4+3]
        //   bytes 12..15= row3[g*4..g*4+3]
        for (int kg = 0; kg < 8; kg++) {  // 8 K-groups (each = 4 K elements)
            for (int r = 0; r < 4; r++) {
                for (int k = 0; k < 4; k++) {
                    W[kg*16 + r*4 + k] = src_rows[r][b*32 + kg*4 + k];
                }
            }
        }
    }
}

// =============================================================================
// Activation broadcast for SoA4 inner loop.
// For each K-group of 4 (8 groups per 32-block, bpr*8 total over GEMV):
//   16 bytes = a[k..k+3] repeated 4 times (one per row)
// Total: K*4 bytes (4× original a_u8). For K=2560 → 10 KB. L1 fits.
// =============================================================================
static void activation_broadcast(const uint8_t* a_u8, int K, uint8_t* a_b16) {
    int n_groups = K / 4;
    for (int g = 0; g < n_groups; g++) {
        const uint8_t* p = a_u8 + g*4;
        uint8_t* dst = a_b16 + g*16;
        // 4× repeat of (a[g*4..g*4+3])
        for (int r = 0; r < 4; r++) {
            dst[r*4 + 0] = p[0];
            dst[r*4 + 1] = p[1];
            dst[r*4 + 2] = p[2];
            dst[r*4 + 3] = p[3];
        }
    }
}

// =============================================================================
// q8_soa4_gemv: SoA4 GEMV with full SIMD inner loop.
//
// Per inner-K-group (4 K elements consumed):
//   v2di w_v = *(const v2di*)(W + kg*16)         // 16 bytes weights
//   v2di a_v = *(const v2di*)(a_b16 + b*128 + kg*16)  // 16 bytes activations
//   v2di p16 = qpmaddubsh(w_v, a_v)              // 8 i16 lanes
//   v2di p32 = qpmaddh(p16, ONES16)              // 4 i32 lanes (one per row)
//   acc      = qpaddw(acc, p32)
//
// 3 SIMD ops per K-group of 4 elements. 8 K-groups per block. 24 SIMD ops/block.
// Plus per-block fp32 fold: 5 ops.
// =============================================================================
static void q8_soa4_gemv(const Q8SoA4* w, const uint8_t* a_b16,
                          const int32_t* sum_a_per_block,
                          float scale_a, float* y) {
    int64_t bpr = w->K / 32;
    int64_t gpr = w->N / 4;
    // Pre-build broadcast scale_a vector
    v2di scale_a_v;
    {
        float arr[4] = {scale_a, scale_a, scale_a, scale_a};
        memcpy(&scale_a_v, arr, 16);
    }
    // shift-by-128 vector for sum_q correction
    v2di shift128 = {0x0000008000000080LL, 0x0000008000000080LL};

    for (int64_t g = 0; g < gpr; g++) {
        const uint8_t* gp = w->mem + g * w->group_stride;
        v2di fp_acc = {0, 0};  // 4 fp32 lanes
        for (int64_t b = 0; b < bpr; b++) {
            const uint8_t* sb = gp + b * SOA4_GROUP_BYTES;
            v2di scales_v = *(const v2di*)(sb + 0);
            v2di dmins_v  = *(const v2di*)(sb + 16);
            v2di sum_q_v  = *(const v2di*)(sb + 32);
            const v2di* W_v = (const v2di*)(sb + 48);
            const v2di* A_v = (const v2di*)(a_b16 + b*128);

            v2di acc_i32 = {0, 0};
            // Unrolled 8-iter K-group loop
            for (int kg = 0; kg < 8; kg++) {
                // qpmaddubsh: arg order (signed, unsigned) per LCC v5
                v2di p16 = __builtin_e2k_qpmaddubsh(W_v[kg], A_v[kg]);
                v2di p32 = __builtin_e2k_qpmaddh(p16, ONES16);
                acc_i32  = __builtin_e2k_qpaddw(acc_i32, p32);
            }

            // Apply per-block correction & scale (5 SIMD ops):
            // dot_signed = acc_i32 - 128 * sum_q
            v2di shift_v = __builtin_e2k_qpmullw(sum_q_v, shift128);
            v2di dot_signed = __builtin_e2k_qpsubw(acc_i32, shift_v);
            v2di acc_f = __builtin_e2k_qpistofs(dot_signed);
            // sum_a per block as broadcast fp32
            float sa_b = (float)sum_a_per_block[b];
            v2di sa_v;
            {
                float arr[4] = {sa_b, sa_b, sa_b, sa_b};
                memcpy(&sa_v, arr, 16);
            }
            // (d_w * acc_f - dmin_m * sum_a_block) * scale_a
            v2di term_w = __builtin_e2k_qpfmuls(scales_v, acc_f);
            v2di term_d = __builtin_e2k_qpfmuls(dmins_v, sa_v);
            v2di delta  = __builtin_e2k_qpfmuls(scale_a_v,
                          __builtin_e2k_qpfsubs(term_w, term_d));
            fp_acc = __builtin_e2k_qpfadds(fp_acc, delta);
        }
        float lanes[4]; memcpy(lanes, &fp_acc, 16);
        y[g*4 + 0] = lanes[0];
        y[g*4 + 1] = lanes[1];
        y[g*4 + 2] = lanes[2];
        y[g*4 + 3] = lanes[3];
    }
}

// =============================================================================
// Reference: FP32 GEMV from Q8 source. y[n] = sum_k(d_w[n,b] * w[n,k] * x[k]).
// For Q8_0 source (no dmin correction).
// =============================================================================
static void q8_ref_fp32(const int8_t* W, const float* d_w, const float* x,
                        int N, int K, float* y) {
    int bpr = K / 32;
    for (int n = 0; n < N; n++) {
        float s = 0;
        for (int b = 0; b < bpr; b++) {
            float dw = d_w[n*bpr + b];
            for (int k = 0; k < 32; k++) {
                s += dw * (float)W[n*K + b*32 + k] * x[b*32 + k];
            }
        }
        y[n] = s;
    }
}

int main(void) {
    const int K = 2560, N = 2432;
    const int bpr = K / 32;

    int8_t* W = (int8_t*)aligned_alloc(64, N * K);
    float* d_w = (float*)aligned_alloc(64, N * bpr * sizeof(float));
    float* x = (float*)aligned_alloc(64, K * sizeof(float));
    float* y_ref = (float*)aligned_alloc(64, N * sizeof(float));
    float* y_soa = (float*)aligned_alloc(64, N * sizeof(float));
    float* y_eml = (float*)aligned_alloc(64, N * sizeof(float));
    uint8_t* a_u8 = (uint8_t*)aligned_alloc(64, K);
    uint8_t* a_b16 = (uint8_t*)aligned_alloc(64, K * 4);
    int32_t* sum_a_pb = (int32_t*)aligned_alloc(64, bpr * sizeof(int32_t));
    float* w_fp32 = (float*)aligned_alloc(64, N * K * sizeof(float));

    srand(42);
    // Realistic Q8 weight distribution (-20..19)
    for (int i = 0; i < N*K; i++) W[i] = ((rand() % 41) - 20);
    for (int i = 0; i < N*bpr; i++) d_w[i] = 0.005f + 0.001f*((float)(rand()%100));
    for (int i = 0; i < K; i++) x[i] = sinf(i * 0.013f) + 0.5f;

    // Activation quant
    float max_a = 0;
    for (int i = 0; i < K; i++) if (fabsf(x[i]) > max_a) max_a = fabsf(x[i]);
    float scale_a = max_a > 0 ? max_a / 127.0f : 1.0f;
    float inv_a = 1.0f / scale_a;
    for (int i = 0; i < K; i++) {
        int v = (int)lrintf(x[i] * inv_a);
        if (v > 127) v = 127; if (v < -127) v = -127;
        a_u8[i] = (uint8_t)(v + 128);
    }
    activation_broadcast(a_u8, K, a_b16);
    for (int b = 0; b < bpr; b++) {
        int s = 0;
        for (int k = 0; k < 32; k++) s += ((int)a_u8[b*32 + k] - 128);
        sum_a_pb[b] = s;
    }

    // Pack into SoA4
    Q8SoA4 SW = q8_soa4_alloc(N, K);
    for (int g = 0; g < N/4; g++) {
        const int8_t* rows[4] = { W + (g*4+0)*K, W + (g*4+1)*K, W + (g*4+2)*K, W + (g*4+3)*K };
        const float* dws[4]   = { d_w + (g*4+0)*bpr, d_w + (g*4+1)*bpr, d_w + (g*4+2)*bpr, d_w + (g*4+3)*bpr };
        q8_soa4_pack_group(&SW, g, rows, dws, NULL);
    }

    // Build flat fp32 weight for EML baseline
    for (int n = 0; n < N; n++)
        for (int b = 0; b < bpr; b++)
            for (int k = 0; k < 32; k++)
                w_fp32[n*K + b*32 + k] = d_w[n*bpr + b] * (float)W[n*K + b*32 + k];

    struct timespec t0, t1;
    int ITERS = 200;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < ITERS; i++) q8_ref_fp32(W, d_w, x, N, K, y_ref);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_ref = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < ITERS; i++)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f,
                    w_fp32, K, x, 1, 0.0f, y_eml, 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_eml = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < ITERS; i++)
        q8_soa4_gemv(&SW, a_b16, sum_a_pb, scale_a, y_soa);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_soa = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;

    float max_err = 0, max_v = 0;
    for (int n = 0; n < N; n++) {
        float e = fabsf(y_soa[n] - y_ref[n]);
        if (e > max_err) max_err = e;
        if (fabsf(y_ref[n]) > max_v) max_v = fabsf(y_ref[n]);
    }
    long long ops = (long long)ITERS * N * K * 2;
    printf("K=%d N=%d iters=%d\n", K, N, ITERS);
    printf("FP32 ref dequant:    %.3f s, %6.2f GOPS, %.2f ms/GEMV\n",
           t_ref, ops/t_ref/1e9, t_ref/ITERS*1000);
    printf("EML cblas_sgemv:     %.3f s, %6.2f GOPS, %.2f ms/GEMV\n",
           t_eml, ops/t_eml/1e9, t_eml/ITERS*1000);
    printf("Q8_SoA4 qpmaddubsh:  %.3f s, %6.2f GOPS, %.2f ms/GEMV  (vs EML: %.2fx)\n",
           t_soa, ops/t_soa/1e9, t_soa/ITERS*1000, t_eml/t_soa);
    printf("correctness: max_err=%.6f (max_ref=%.3f, rel=%.4f%%)\n",
           max_err, max_v, 100.0f*max_err/max_v);
    free(W); free(d_w); free(x); free(y_ref); free(y_soa); free(y_eml);
    free(a_u8); free(a_b16); free(sum_a_pb); free(w_fp32);
    q8_soa4_free(&SW);
    return 0;
}
