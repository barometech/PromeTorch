// Q8_0 GEMV via qpmaddubsh on E2K 8C2 — proof of concept
//
// Q8_0 format: per 32-element block: fp16 scale + 32 int8 weights = 34 bytes.
// Weight matrix is N rows × (K/32) blocks per row.
//
// Algorithm:
//   1. Quantize activation once per GEMV (FP32 -> int8 symmetric per-row).
//   2. Pre-shift activation to uint8 (add 128) for qpmaddubsh.
//   3. Per weight block: compute INT32 dot via qpmaddubsh, correct for shift,
//      convert to FP32 with scale_w * scale_a.
//
// Build: lcc -O3 -march=elbrus-v5 q8_vnni_gemv.c -o q8_vnni_gemv -lm

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// EML for realistic FP32 GEMV baseline
#ifndef NO_EML
#include <cblas.h>
#endif

typedef long long v2di __attribute__((vector_size(16)));
typedef uint16_t ggml_fp16_t;

static const v2di ONES16 = {0x0001000100010001LL, 0x0001000100010001LL};

static inline float fp16_to_fp32(ggml_fp16_t h) {
    uint32_t sign = ((uint32_t)h >> 15) & 1;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float r; memcpy(&r, &f, 4); return r;
}

// Reference: FP32 dequant GEMV (what scalar Q8_0 kernel approximates).
static void q8_gemv_ref_fp32(const uint8_t* weight, const float* x,
                              int K, int N, int row_stride_bytes, float* y) {
    int bpr = K / 32;
    for (int n = 0; n < N; n++) {
        const uint8_t* row = weight + (size_t)n * row_stride_bytes;
        float acc = 0;
        for (int b = 0; b < bpr; b++) {
            const uint8_t* block = row + b * 34;
            float scale = fp16_to_fp32(*(const ggml_fp16_t*)block);
            const int8_t* w = (const int8_t*)(block + 2);
            for (int j = 0; j < 32; j++) acc += scale * (float)w[j] * x[b*32 + j];
        }
        y[n] = acc;
    }
}

// VNNI Q8 GEMV v1: sum_w inner loop (baseline).
static void q8_gemv_vnni(const uint8_t* weight,
                          const uint8_t* a_u8,
                          float scale_a,
                          int K, int N, int row_stride_bytes, float* y) {
    int bpr = K / 32;
    for (int n = 0; n < N; n++) {
        const uint8_t* row = weight + (size_t)n * row_stride_bytes;
        float acc = 0;
        for (int b = 0; b < bpr; b++) {
            const uint8_t* block = row + b * 34;
            float scale_w = fp16_to_fp32(*(const ggml_fp16_t*)block);
            const int8_t* w_s8 = (const int8_t*)(block + 2);
            v2di a0 = *(const v2di*)(a_u8 + b*32);
            v2di a1 = *(const v2di*)(a_u8 + b*32 + 16);
            v2di w0 = *(const v2di*)(w_s8);
            v2di w1 = *(const v2di*)(w_s8 + 16);
            v2di p16a = __builtin_e2k_qpmaddubsh(w0, a0);
            v2di p16b = __builtin_e2k_qpmaddubsh(w1, a1);
            v2di p32a = __builtin_e2k_qpmaddh(p16a, ONES16);
            v2di p32b = __builtin_e2k_qpmaddh(p16b, ONES16);
            v2di sum32 = __builtin_e2k_qpaddw(p32a, p32b);
            int dot_us = ((int*)&sum32)[0] + ((int*)&sum32)[1]
                       + ((int*)&sum32)[2] + ((int*)&sum32)[3];
            int sum_w = 0;
            for (int j = 0; j < 32; j++) sum_w += w_s8[j];
            int dot_ss = dot_us - 128 * sum_w;
            acc += scale_w * scale_a * (float)dot_ss;
        }
        y[n] = acc;
    }
}

// VNNI Q8 GEMV v2: pre-computed per-block sum_w passed externally. Expected
// to be ~5× faster because the inner sum_w loop dominated v1 (32 scalar
// adds × bpr blocks × N rows).
static void q8_gemv_vnni_presum(const uint8_t* weight,
                                 const int16_t* sum_w_table,  // N * bpr int16s, laid out row-major
                                 const uint8_t* a_u8,
                                 float scale_a,
                                 int K, int N, int row_stride_bytes, float* y) {
    int bpr = K / 32;
    for (int n = 0; n < N; n++) {
        const uint8_t* row = weight + (size_t)n * row_stride_bytes;
        const int16_t* sum_w_row = sum_w_table + n * bpr;
        float acc = 0;
        for (int b = 0; b < bpr; b++) {
            const uint8_t* block = row + b * 34;
            float scale_w = fp16_to_fp32(*(const ggml_fp16_t*)block);
            const int8_t* w_s8 = (const int8_t*)(block + 2);
            v2di a0 = *(const v2di*)(a_u8 + b*32);
            v2di a1 = *(const v2di*)(a_u8 + b*32 + 16);
            v2di w0 = *(const v2di*)(w_s8);
            v2di w1 = *(const v2di*)(w_s8 + 16);
            v2di p16a = __builtin_e2k_qpmaddubsh(w0, a0);
            v2di p16b = __builtin_e2k_qpmaddubsh(w1, a1);
            v2di p32a = __builtin_e2k_qpmaddh(p16a, ONES16);
            v2di p32b = __builtin_e2k_qpmaddh(p16b, ONES16);
            v2di sum32 = __builtin_e2k_qpaddw(p32a, p32b);
            int dot_us = ((int*)&sum32)[0] + ((int*)&sum32)[1]
                       + ((int*)&sum32)[2] + ((int*)&sum32)[3];
            int dot_ss = dot_us - 128 * (int)sum_w_row[b];
            acc += scale_w * scale_a * (float)dot_ss;
        }
        y[n] = acc;
    }
}

int main(void){
    const int K = 2560;
    const int N = 2432;
    const int bpr = K / 32;
    const int row_stride = bpr * 34;

    uint8_t* weight = (uint8_t*)aligned_alloc(64, (size_t)N * row_stride);
    float* x = (float*)aligned_alloc(64, K * sizeof(float));
    float* y_ref = (float*)aligned_alloc(64, N * sizeof(float));
    float* y_vnni = (float*)aligned_alloc(64, N * sizeof(float));
    uint8_t* a_u8 = (uint8_t*)aligned_alloc(64, K);

    // Realistic Q8_0 weight distribution: most values |w| < ~30, rare
    // outliers to ±80. This matches llama.cpp's actual Q8_0 quantization
    // of LLM weights — uniform ±127 is NOT realistic and causes int16
    // saturation in qpmaddubsh pair-sums.
    for (int n = 0; n < N; n++) {
        uint8_t* row = weight + (size_t)n * row_stride;
        for (int b = 0; b < bpr; b++) {
            float s = 0.01f + 0.001f * (float)((n + b*7) % 100);
            uint32_t u; memcpy(&u, &s, 4);
            uint32_t sign = (u >> 31) & 1, e32 = (u >> 23) & 0xFF, m32 = u & 0x7FFFFF;
            uint32_t e16 = e32 > 127 + 15 ? 31 : (e32 < 127 - 14 ? 0 : e32 - 127 + 15);
            uint32_t m16 = m32 >> 13;
            ggml_fp16_t h = (ggml_fp16_t)((sign << 15) | (e16 << 10) | m16);
            *(ggml_fp16_t*)(row + b*34) = h;
            for (int j = 0; j < 32; j++) {
                // Gaussian-like: sum of 3 uniforms mod 40, shifted, in (-30, 30)
                int raw = (n * 131 + b * 17 + j * 3) % 40
                        + (n * 7   + b * 5 + j * 11) % 40
                        + (n * 13  + b * 19 + j * 23) % 40;
                int w = (raw / 3) - 20;  // center on 0, range ~(-20, 19)
                row[b*34 + 2 + j] = (int8_t)w;
            }
        }
    }
    float max_abs = 0;
    for (int i = 0; i < K; i++) { x[i] = sinf((float)i * 0.01f) + 0.5f; if (fabsf(x[i]) > max_abs) max_abs = fabsf(x[i]); }
    float scale_a = max_abs > 0 ? max_abs / 127.0f : 1.0f;
    float inv = 1.0f / scale_a;
    for (int i = 0; i < K; i++) {
        int v = (int)lrintf(x[i] * inv);
        if (v > 127) v = 127; if (v < -127) v = -127;
        a_u8[i] = (uint8_t)(v + 128);
    }

    struct timespec t0, t1;
    int N_iters = 200;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < N_iters; it++) q8_gemv_ref_fp32(weight, x, K, N, row_stride, y_ref);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_ref = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < N_iters; it++) q8_gemv_vnni(weight, a_u8, scale_a, K, N, row_stride, y_vnni);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_vnni = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;

    // Pre-compute sum_w table (would normally be done once at weight-load)
    int16_t* sum_w_table = (int16_t*)aligned_alloc(64, (size_t)N * bpr * sizeof(int16_t));
    for (int n = 0; n < N; n++) {
        const uint8_t* row = weight + (size_t)n * row_stride;
        for (int b = 0; b < bpr; b++) {
            const int8_t* w = (const int8_t*)(row + b*34 + 2);
            int s = 0;
            for (int j = 0; j < 32; j++) s += w[j];
            sum_w_table[n*bpr + b] = (int16_t)s;
        }
    }
    float* y_presum = (float*)aligned_alloc(64, N * sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < N_iters; it++)
        q8_gemv_vnni_presum(weight, sum_w_table, a_u8, scale_a, K, N, row_stride, y_presum);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_presum = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    long long ops2 = (long long)N_iters * N * K * 2;
    printf("VNNI presum:      %.3f s, %.2f GOPS (per GEMV: %.2f ms, speedup %.2fx vs FP32)\n",
           t_presum, ops2/t_presum/1e9, t_presum/N_iters*1000, t_ref/t_presum);
    // correctness of presum
    float err_ps = 0, mr = 0;
    for (int n = 0; n < N; n++) {
        float e = fabsf(y_ref[n] - y_presum[n]);
        if (e > err_ps) err_ps = e;
        if (fabsf(y_ref[n]) > mr) mr = fabsf(y_ref[n]);
    }
    printf("presum correctness: max_err=%.6f rel=%.4f%%\n", err_ps, 100.0f*err_ps/mr);
    free(sum_w_table); free(y_presum);

#ifndef NO_EML
    // EML cblas_sgemv FP32 baseline: dequantize weight to FP32 first (one-off
    // amortized over 200 iters), then cblas_sgemv K iters.
    float* w_fp32 = (float*)aligned_alloc(64, (size_t)N * K * sizeof(float));
    for (int n = 0; n < N; n++) {
        const uint8_t* row = weight + (size_t)n * row_stride;
        for (int b = 0; b < bpr; b++) {
            const uint8_t* block = row + b * 34;
            float scale = fp16_to_fp32(*(const ggml_fp16_t*)block);
            const int8_t* w = (const int8_t*)(block + 2);
            for (int j = 0; j < 32; j++) w_fp32[n*K + b*32 + j] = scale * (float)w[j];
        }
    }
    float* y_eml = (float*)aligned_alloc(64, N * sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < N_iters; it++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f,
                    w_fp32, K, x, 1, 0.0f, y_eml, 1);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_eml = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double eml_gops = 2.0 * N * K * N_iters / t_eml / 1e9;
    printf("EML cblas_sgemv:  %.3f s, %.2f GOPS (per GEMV: %.2f ms, vnni/eml speedup %.2fx)\n",
           t_eml, eml_gops, t_eml/N_iters*1000, t_eml/t_vnni);
    free(w_fp32); free(y_eml);
#endif

    float max_err = 0, max_ref = 0;
    for (int n = 0; n < N; n++) {
        float e = fabsf(y_ref[n] - y_vnni[n]);
        if (e > max_err) max_err = e;
        if (fabsf(y_ref[n]) > max_ref) max_ref = fabsf(y_ref[n]);
    }
    // Debug: first 5 rows, ref vs vnni, and intermediate dot
    printf("DEBUG rows 0..4:\n");
    for (int n = 0; n < 5; n++) {
        printf("  row %d: ref=%12.4f vnni=%12.4f diff=%10.4f  (a_s8[0..3]=%d,%d,%d,%d w_s8[0..3]=%d,%d,%d,%d)\n",
               n, y_ref[n], y_vnni[n], y_vnni[n] - y_ref[n],
               (int)a_u8[0]-128, (int)a_u8[1]-128, (int)a_u8[2]-128, (int)a_u8[3]-128,
               (int)((int8_t*)(weight + (size_t)n * row_stride + 2))[0],
               (int)((int8_t*)(weight + (size_t)n * row_stride + 2))[1],
               (int)((int8_t*)(weight + (size_t)n * row_stride + 2))[2],
               (int)((int8_t*)(weight + (size_t)n * row_stride + 2))[3]);
    }
    // Manual compute row 0 block 0 with known arithmetic + compare intrin
    {
        const uint8_t* blk = weight;
        float scale = fp16_to_fp32(*(const ggml_fp16_t*)blk);
        const int8_t* w = (const int8_t*)(blk + 2);
        int dot_us_manual = 0, dot_ss_manual = 0;
        for (int j = 0; j < 32; j++) {
            dot_us_manual += (int)a_u8[j] * (int)w[j];
            dot_ss_manual += ((int)a_u8[j] - 128) * (int)w[j];
        }
        // Exercise the intrinsic the same way kernel does:
        v2di a0 = *(const v2di*)(a_u8);
        v2di a1 = *(const v2di*)(a_u8 + 16);
        v2di w0 = *(const v2di*)(w);
        v2di w1 = *(const v2di*)(w + 16);
        v2di p16a = __builtin_e2k_qpmaddubsh(a0, w0);
        v2di p16b = __builtin_e2k_qpmaddubsh(a1, w1);
        v2di p32a = __builtin_e2k_qpmaddh(p16a, ONES16);
        v2di p32b = __builtin_e2k_qpmaddh(p16b, ONES16);
        v2di sum32 = __builtin_e2k_qpaddw(p32a, p32b);
        int dot_us_intrin = ((int*)&sum32)[0] + ((int*)&sum32)[1]
                          + ((int*)&sum32)[2] + ((int*)&sum32)[3];
        printf("BLOCK 0 row 0: dot_us_manual=%d  dot_us_intrin=%d (should match)\n",
               dot_us_manual, dot_us_intrin);
        // Per-int16 lane inspection
        int16_t* p16a_ref = (int16_t*)&p16a;
        printf("  p16a first 4 halves (int16 MAD lanes): %d %d %d %d\n",
               p16a_ref[0], p16a_ref[1], p16a_ref[2], p16a_ref[3]);
        // Expected for lane 0 = a_u8[0]*w[0] + a_u8[1]*w[1] = (42+128)*(-20) + (43+128)*(-8) = 170*-20 + 171*-8 = -3400 + -1368 = -4768
        printf("  expected lane 0 = a_u8[0]*w[0] + a_u8[1]*w[1] = %d*%d + %d*%d = %d\n",
               (int)a_u8[0], (int)w[0], (int)a_u8[1], (int)w[1],
               (int)a_u8[0]*(int)w[0] + (int)a_u8[1]*(int)w[1]);
    }

    long long ops = (long long)N_iters * N * K * 2;
    printf("K=%d N=%d iters=%d\n", K, N, N_iters);
    printf("ref FP32 dequant: %.3f s, %.2f GOPS (per GEMV: %.2f ms)\n",
           t_ref, ops/t_ref/1e9, t_ref/N_iters*1000);
    printf("VNNI qpmaddubsh:  %.3f s, %.2f GOPS (per GEMV: %.2f ms, speedup %.2fx)\n",
           t_vnni, ops/t_vnni/1e9, t_vnni/N_iters*1000, t_ref/t_vnni);
    printf("correctness: max_err=%.6f (max_ref=%.3f, rel=%.4f%%)\n",
           max_err, max_ref, 100.0f*max_err/max_ref);

    free(weight); free(x); free(y_ref); free(y_vnni); free(a_u8);
    return 0;
}
