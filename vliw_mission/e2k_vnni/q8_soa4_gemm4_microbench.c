// q8_soa4_gemm_K4 microbench — Round 5
//
// Гипотеза (из disassembly q8_soa4_gemv):
//   В готовом бинаре wide instruction packing = 2× qpmaddubsh + 4× ldqp +
//   1 aaurwd. На 8 K-groups inner loop = 16 qpmaddubsh + 16 ldqp на block.
//   Density qpmaddubsh ~17% peak (40 ops в 240 cycles disassembled). Slots
//   заняты loads, не compute.
//
// Идея: batched GEMM с K=4 активациями reuses weight loads:
//   - 16 weight ldqp same as single-row
//   - 16×4=64 qpmaddubsh (4 acc chains, по одной на activation)
//   - density qpmaddubsh ~50% peak → 3× speedup на per-op cost
//
// Если speedup ≥ 2× per-op → интегрируем в speculative decode TP path
// (forward_decode_cpu_tp_batched) и получаем lossless 16-17 tok/s.
//
// Bench: K=2560, N=2432 (типичный output_proj k-slice). Compare:
//   - q8_soa4_gemv (single-row baseline) — 1.21 ms
//   - q8_soa4_gemm_K4 (4 activations batched) — target ≤ 1.5 ms (4× work in 1.25× time)
//   - speedup per-token = 4 / 1.5 = 2.67×

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef __e2k__
#include <e2kintrin.h>

typedef long long v2di __attribute__((vector_size(16)));

static const v2di ONES16   = {0x0001000100010001LL, 0x0001000100010001LL};
static const v2di SHIFT128 = {0x0000008000000080LL, 0x0000008000000080LL};

#define SOA4_GROUP_BYTES 176

// Производит 4 output vectors одновременно: y[k][n] = dot(x[k][:], w[n][:])
// для k=0..3, n=0..N-1.
static void q8_soa4_gemm_K4(
    const uint8_t* w_mem, int64_t N, int64_t K,
    /* per-K activations: 4 buffers a_b16[k], 4 sum_a[k], 4 scale_a[k] */
    const uint8_t* const a_b16[4],
    const int32_t* const sum_a_per_block[4],
    const float scale_a[4],
    float* const y[4]) {
    int64_t bpr = K / 32;
    int64_t gpr = N / 4;

    v2di scale_a_v[4];
    for (int k = 0; k < 4; k++) {
        float arr[4] = {scale_a[k], scale_a[k], scale_a[k], scale_a[k]};
        memcpy(&scale_a_v[k], arr, 16);
    }

    for (int64_t g = 0; g < gpr; g++) {
        const uint8_t* gp = w_mem + g * bpr * SOA4_GROUP_BYTES;
        v2di fp_acc[4];
        for (int k = 0; k < 4; k++) fp_acc[k] = (v2di){0, 0};

        for (int64_t b = 0; b < bpr; b++) {
            const uint8_t* sb = gp + b * SOA4_GROUP_BYTES;
            v2di scales_v = *(const v2di*)(sb + 0);
            v2di dmins_v  = *(const v2di*)(sb + 16);
            v2di sum_q_v  = *(const v2di*)(sb + 32);
            const v2di* W_v = (const v2di*)(sb + 48);

            // 4 independent accumulator chains, one per K-activation.
            // Weight loads SHARED across all 4 (the win).
            v2di acc_i32[4] = {{0,0},{0,0},{0,0},{0,0}};

            const v2di* A_v[4];
            for (int k = 0; k < 4; k++) A_v[k] = (const v2di*)(a_b16[k] + b*128);

            _Pragma("loop count(8)") _Pragma("ivdep")
            for (int kg = 0; kg < 8; kg++) {
                v2di W = W_v[kg];  // ONE load, shared across 4 activations
                // 4× independent qpmaddubsh — VLIW slots 1, 4 fill twice
                v2di p16_0 = __builtin_e2k_qpmaddubsh(W, A_v[0][kg]);
                v2di p16_1 = __builtin_e2k_qpmaddubsh(W, A_v[1][kg]);
                v2di p16_2 = __builtin_e2k_qpmaddubsh(W, A_v[2][kg]);
                v2di p16_3 = __builtin_e2k_qpmaddubsh(W, A_v[3][kg]);
                v2di p32_0 = __builtin_e2k_qpmaddh(p16_0, ONES16);
                v2di p32_1 = __builtin_e2k_qpmaddh(p16_1, ONES16);
                v2di p32_2 = __builtin_e2k_qpmaddh(p16_2, ONES16);
                v2di p32_3 = __builtin_e2k_qpmaddh(p16_3, ONES16);
                acc_i32[0] = __builtin_e2k_qpaddw(acc_i32[0], p32_0);
                acc_i32[1] = __builtin_e2k_qpaddw(acc_i32[1], p32_1);
                acc_i32[2] = __builtin_e2k_qpaddw(acc_i32[2], p32_2);
                acc_i32[3] = __builtin_e2k_qpaddw(acc_i32[3], p32_3);
            }

            v2di shift_v = __builtin_e2k_qpmullw(sum_q_v, SHIFT128);
            for (int k = 0; k < 4; k++) {
                v2di dot_signed = __builtin_e2k_qpsubw(acc_i32[k], shift_v);
                v2di acc_f = __builtin_e2k_qpistofs(dot_signed);
                float sa_b_val = (float)sum_a_per_block[k][b];
                v2di sa_v;
                float arr[4] = {sa_b_val, sa_b_val, sa_b_val, sa_b_val};
                memcpy(&sa_v, arr, 16);
                v2di term_w = __builtin_e2k_qpfmuls(scales_v, acc_f);
                v2di term_d = __builtin_e2k_qpfmuls(dmins_v, sa_v);
                v2di delta  = __builtin_e2k_qpfmuls(scale_a_v[k],
                              __builtin_e2k_qpfsubs(term_w, term_d));
                fp_acc[k]   = __builtin_e2k_qpfadds(fp_acc[k], delta);
            }
        }
        for (int k = 0; k < 4; k++) {
            float lanes[4]; memcpy(lanes, &fp_acc[k], 16);
            y[k][g*4 + 0] = lanes[0];
            y[k][g*4 + 1] = lanes[1];
            y[k][g*4 + 2] = lanes[2];
            y[k][g*4 + 3] = lanes[3];
        }
    }
}

// Single-row baseline (copy of production q8_soa4_gemv inner).
static void q8_soa4_gemv_baseline(
    const uint8_t* w_mem, int64_t N, int64_t K,
    const uint8_t* a_b16, const int32_t* sum_a_per_block, float scale_a,
    float* y) {
    int64_t bpr = K / 32;
    int64_t gpr = N / 4;

    v2di scale_a_v;
    {
        float arr[4] = {scale_a, scale_a, scale_a, scale_a};
        memcpy(&scale_a_v, arr, 16);
    }

    for (int64_t g = 0; g < gpr; g++) {
        const uint8_t* gp = w_mem + g * bpr * SOA4_GROUP_BYTES;
        v2di fp_acc = {0, 0};
        for (int64_t b = 0; b < bpr; b++) {
            const uint8_t* sb = gp + b * SOA4_GROUP_BYTES;
            v2di scales_v = *(const v2di*)(sb + 0);
            v2di dmins_v  = *(const v2di*)(sb + 16);
            v2di sum_q_v  = *(const v2di*)(sb + 32);
            const v2di* W_v = (const v2di*)(sb + 48);
            const v2di* A_v = (const v2di*)(a_b16 + b*128);
            v2di acc_i32 = {0, 0};
            _Pragma("loop count(8)") _Pragma("ivdep")
            for (int kg = 0; kg < 8; kg++) {
                v2di p16 = __builtin_e2k_qpmaddubsh(W_v[kg], A_v[kg]);
                v2di p32 = __builtin_e2k_qpmaddh(p16, ONES16);
                acc_i32  = __builtin_e2k_qpaddw(acc_i32, p32);
            }
            v2di shift_v = __builtin_e2k_qpmullw(sum_q_v, SHIFT128);
            v2di dot_signed = __builtin_e2k_qpsubw(acc_i32, shift_v);
            v2di acc_f = __builtin_e2k_qpistofs(dot_signed);
            float sa_b_val = (float)sum_a_per_block[b];
            v2di sa_v;
            float arr[4] = {sa_b_val, sa_b_val, sa_b_val, sa_b_val};
            memcpy(&sa_v, arr, 16);
            v2di term_w = __builtin_e2k_qpfmuls(scales_v, acc_f);
            v2di term_d = __builtin_e2k_qpfmuls(dmins_v, sa_v);
            v2di delta  = __builtin_e2k_qpfmuls(scale_a_v,
                          __builtin_e2k_qpfsubs(term_w, term_d));
            fp_acc = __builtin_e2k_qpfadds(fp_acc, delta);
        }
        float lanes[4]; memcpy(lanes, &fp_acc, 16);
        y[g*4 + 0] = lanes[0]; y[g*4 + 1] = lanes[1];
        y[g*4 + 2] = lanes[2]; y[g*4 + 3] = lanes[3];
    }
}

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main() {
    const int64_t N = 2432;   // typical k-sliced output_proj
    const int64_t K = 2560;
    const int64_t bpr = K / 32;
    const int64_t gpr = N / 4;
    const int64_t weight_bytes = gpr * bpr * SOA4_GROUP_BYTES;

    uint8_t* w_mem;
    if (posix_memalign((void**)&w_mem, 64, weight_bytes) != 0) {
        fprintf(stderr, "alloc weight\n"); return 1;
    }
    // Fill with random data — content does not affect timing.
    for (int64_t i = 0; i < weight_bytes; i++) w_mem[i] = (uint8_t)(i * 31 + 7);

    // 4 activations
    uint8_t* a_b16_arr[4];
    int32_t* sum_a_arr[4];
    float scale_a_arr[4] = {0.0123f, 0.0145f, 0.0167f, 0.0189f};
    for (int k = 0; k < 4; k++) {
        if (posix_memalign((void**)&a_b16_arr[k], 64, bpr * 128) != 0) return 2;
        if (posix_memalign((void**)&sum_a_arr[k], 64, bpr * 4) != 0) return 3;
        for (int64_t i = 0; i < bpr * 128; i++) a_b16_arr[k][i] = (uint8_t)(i * 17 + k);
        for (int64_t b = 0; b < bpr; b++) sum_a_arr[k][b] = (int32_t)(b - 8);
    }

    float* y_arr[4];
    for (int k = 0; k < 4; k++) {
        if (posix_memalign((void**)&y_arr[k], 64, N * 4) != 0) return 4;
    }

    // Warmup
    q8_soa4_gemv_baseline(w_mem, N, K, a_b16_arr[0], sum_a_arr[0], scale_a_arr[0], y_arr[0]);
    q8_soa4_gemm_K4(w_mem, N, K, (const uint8_t* const*)a_b16_arr,
                     (const int32_t* const*)sum_a_arr, scale_a_arr, y_arr);

    // Bench baseline: 100 iters, average.
    const int ITERS = 100;
    double t0 = now_ms();
    for (int it = 0; it < ITERS; it++) {
        q8_soa4_gemv_baseline(w_mem, N, K,
            a_b16_arr[0], sum_a_arr[0], scale_a_arr[0], y_arr[0]);
    }
    double t1 = now_ms();
    double baseline_ms = (t1 - t0) / ITERS;

    // Bench K=4 batched
    double t2 = now_ms();
    for (int it = 0; it < ITERS; it++) {
        q8_soa4_gemm_K4(w_mem, N, K,
            (const uint8_t* const*)a_b16_arr,
            (const int32_t* const*)sum_a_arr,
            scale_a_arr, y_arr);
    }
    double t3 = now_ms();
    double batched_ms = (t3 - t2) / ITERS;

    printf("=== q8_soa4 microbench K=2560 N=2432 ===\n");
    printf("baseline single-row:       %.3f ms/call (1 act × 1 weight set)\n", baseline_ms);
    printf("batched K=4 (4× работы):   %.3f ms/call (4 acts × 1 weight set)\n", batched_ms);
    printf("equivalent per-act:        %.3f ms (= batched / 4)\n", batched_ms / 4);
    printf("speedup per-act:           %.2fx\n", baseline_ms / (batched_ms / 4));
    printf("\n");
    printf("=== Trans interpretation ===\n");
    printf("Если speedup >= 2x per-act, batched GEMM имеет смысл интегрировать\n");
    printf("в speculative decode для TP path. Acceptance ~70%% × 4-token batch =\n");
    printf("expected lossless 16-17 tok/s vs current 11.4.\n");

    free(w_mem);
    for (int k = 0; k < 4; k++) {
        free(a_b16_arr[k]);
        free(sum_a_arr[k]);
        free(y_arr[k]);
    }
    return 0;
}

#else
int main() { printf("E2K only\n"); return 0; }
#endif
