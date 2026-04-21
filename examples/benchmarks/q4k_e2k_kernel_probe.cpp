// Single-thread Q4_K GEMV kernel shootout: LCC-translated AVX2 vs hand-rolled
// native E2K intrinsics. Goal: find out whether LCC's AVX2→qp* translation is
// already optimal, or whether direct qp intrinsics free up enough VLIW slots
// for a measurable speedup.
//
// Compile:
//   l++ -std=c++17 -O2 -mavx2 -I$HOME/promethorch examples/benchmarks/q4k_e2k_kernel_probe.cpp -o /tmp/q4kprobe -lpthread
//
// Run:
//   /tmp/q4kprobe          (default K=2560 N=9728 iters=20, mirrors ffn_gate shape)
//
// Metric: ns per row averaged across N × iters. Lower is better.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>

#include "torch/io/gguf_dequant.h"

#ifdef __AVX2__
#  include <immintrin.h>
#endif

// E2K native intrinsics header (LCC ships this at -1.29.16/e2k-8c2-linux/include)
#if __has_include(<e2kintrin.h>)
#  include <e2kintrin.h>
#  define HAVE_E2K_NATIVE 1
#else
#  define HAVE_E2K_NATIVE 0
#endif

// ----- Q8 block (mirror of cpu_quant_gemv.h) -----
struct alignas(16) Q8Block {
    float d;       // block scale
    float sum;     // sum of qs[i]
    int8_t qs[32]; // quantized x
};

// Prepare synthetic Q4_K row: 144 bytes per 256-element super-block.
// We fill scales + qs with pseudo-random bytes so the kernels do realistic work;
// numerical correctness between kernels is checked separately.
static void fill_q4k_blocks(uint8_t* buf, int64_t blocks, uint32_t seed) {
    std::mt19937 rng(seed);
    for (int64_t bi = 0; bi < blocks; ++bi) {
        uint8_t* blk = buf + bi * 144;
        uint16_t d = 0x3c00;    // fp16 1.0
        uint16_t dm = 0x0000;   // fp16 0
        std::memcpy(blk, &d, 2);
        std::memcpy(blk + 2, &dm, 2);
        for (int i = 4; i < 144; ++i) blk[i] = static_cast<uint8_t>(rng());
    }
}

static void fill_q8_blocks(Q8Block* blocks, int64_t nb, uint32_t seed) {
    std::mt19937 rng(seed);
    for (int64_t i = 0; i < nb; ++i) {
        blocks[i].d = 0.01f;
        int s = 0;
        for (int j = 0; j < 32; ++j) {
            blocks[i].qs[j] = static_cast<int8_t>(rng() % 127);
            s += blocks[i].qs[j];
        }
        blocks[i].sum = static_cast<float>(s);
    }
}

// ============================================================================
// Kernel 1 — current path: LCC translation of AVX2 intrinsics.
// Same structure as torch/io/cpu_quant_gemv.h q4k_gemv_avx2 2-row inner loop.
// ============================================================================
#ifdef __AVX2__
static __attribute__((noinline)) void q4k_gemv_avx2_row_pair(
    const uint8_t* row0, const uint8_t* row1,
    const Q8Block* x_q8, int64_t blocks_per_row,
    float& out0, float& out1) {
    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i ones_16  = _mm256_set1_epi16(1);
    float sum0 = 0.0f, sum1 = 0.0f;

    for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* blk0 = row0 + bi * 144;
        const uint8_t* blk1 = row1 + bi * 144;
        const Q8Block* xq = x_q8 + bi * 8;

        uint16_t d0b, dm0b, d1b, dm1b;
        std::memcpy(&d0b,  blk0,     2);
        std::memcpy(&dm0b, blk0 + 2, 2);
        std::memcpy(&d1b,  blk1,     2);
        std::memcpy(&dm1b, blk1 + 2, 2);
        float d0 = torch::io::gguf::fp16_to_fp32(d0b), dm0 = torch::io::gguf::fp16_to_fp32(dm0b);
        float d1 = torch::io::gguf::fp16_to_fp32(d1b), dm1 = torch::io::gguf::fp16_to_fp32(dm1b);
        const uint8_t* sc0 = blk0 + 4;
        const uint8_t* sc1 = blk1 + 4;
        const uint8_t* qs0 = blk0 + 16;
        const uint8_t* qs1 = blk1 + 16;

        // Prefetch NEXT block for both rows so HW/L1 are warm by the time we
        // advance. Hint T0 = keep in all cache levels.
        if (bi + 1 < blocks_per_row) {
            __builtin_prefetch(row0 + (bi + 1) * 144, 0, 3);
            __builtin_prefetch(row1 + (bi + 1) * 144, 0, 3);
            __builtin_prefetch(row0 + (bi + 1) * 144 + 64, 0, 3);
            __builtin_prefetch(row1 + (bi + 1) * 144 + 64, 0, 3);
        }

        int q8_idx = 0;
        for (int j = 0; j < 256; j += 64) {
            int is = j / 32;
            uint8_t sca0, ma0, scb0, mb0, sca1, ma1, scb1, mb1;
            torch::io::gguf::get_scale_min_k4(is,     sc0, &sca0, &ma0);
            torch::io::gguf::get_scale_min_k4(is + 1, sc0, &scb0, &mb0);
            torch::io::gguf::get_scale_min_k4(is,     sc1, &sca1, &ma1);
            torch::io::gguf::get_scale_min_k4(is + 1, sc1, &scb1, &mb1);

            __m256i q8_lo = _mm256_loadu_si256((const __m256i*)xq[q8_idx].qs);
            __m256i q8_hi = _mm256_loadu_si256((const __m256i*)xq[q8_idx+1].qs);
            float dx_lo = xq[q8_idx].d,  dx_hi = xq[q8_idx+1].d;
            float sx_lo = xq[q8_idx].sum, sx_hi = xq[q8_idx+1].sum;

            __m256i raw0 = _mm256_loadu_si256((const __m256i*)qs0);
            __m256i q4_lo0 = _mm256_and_si256(raw0, mask_lo4);
            __m256i q4_hi0 = _mm256_and_si256(_mm256_srli_epi16(raw0, 4), mask_lo4);
            int32_t is0_lo = _mm256_extract_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(q4_lo0, q8_lo), ones_16), 0);
            // full hsum done via scalar accum:
            __m256i p0_lo32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_lo0, q8_lo), ones_16);
            __m256i p0_hi32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_hi0, q8_hi), ones_16);
            alignas(32) int32_t tmp0_lo[8], tmp0_hi[8];
            _mm256_storeu_si256((__m256i*)tmp0_lo, p0_lo32);
            _mm256_storeu_si256((__m256i*)tmp0_hi, p0_hi32);
            int32_t is_lo0 = tmp0_lo[0]+tmp0_lo[1]+tmp0_lo[2]+tmp0_lo[3]+tmp0_lo[4]+tmp0_lo[5]+tmp0_lo[6]+tmp0_lo[7];
            int32_t is_hi0 = tmp0_hi[0]+tmp0_hi[1]+tmp0_hi[2]+tmp0_hi[3]+tmp0_hi[4]+tmp0_hi[5]+tmp0_hi[6]+tmp0_hi[7];
            (void)is0_lo;

            __m256i raw1 = _mm256_loadu_si256((const __m256i*)qs1);
            __m256i q4_lo1 = _mm256_and_si256(raw1, mask_lo4);
            __m256i q4_hi1 = _mm256_and_si256(_mm256_srli_epi16(raw1, 4), mask_lo4);
            __m256i p1_lo32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_lo1, q8_lo), ones_16);
            __m256i p1_hi32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_hi1, q8_hi), ones_16);
            alignas(32) int32_t tmp1_lo[8], tmp1_hi[8];
            _mm256_storeu_si256((__m256i*)tmp1_lo, p1_lo32);
            _mm256_storeu_si256((__m256i*)tmp1_hi, p1_hi32);
            int32_t is_lo1 = tmp1_lo[0]+tmp1_lo[1]+tmp1_lo[2]+tmp1_lo[3]+tmp1_lo[4]+tmp1_lo[5]+tmp1_lo[6]+tmp1_lo[7];
            int32_t is_hi1 = tmp1_hi[0]+tmp1_hi[1]+tmp1_hi[2]+tmp1_hi[3]+tmp1_hi[4]+tmp1_hi[5]+tmp1_hi[6]+tmp1_hi[7];

            sum0 += d0 * sca0 * dx_lo * (float)is_lo0 - dm0 * ma0 * sx_lo;
            sum0 += d0 * scb0 * dx_hi * (float)is_hi0 - dm0 * mb0 * sx_hi;
            sum1 += d1 * sca1 * dx_lo * (float)is_lo1 - dm1 * ma1 * sx_lo;
            sum1 += d1 * scb1 * dx_hi * (float)is_hi1 - dm1 * mb1 * sx_hi;

            qs0 += 32; qs1 += 32; q8_idx += 2;
        }
    }
    out0 = sum0; out1 = sum1;
}
#endif

// ============================================================================
// Kernel 2 — native E2K qp* intrinsics, 128-bit SIMD (half-width of AVX2 YMM,
// processes 16 bytes per op; 32-byte sub-blocks need 2 qp ops).
// ============================================================================
#if HAVE_E2K_NATIVE
static __attribute__((noinline)) void q4k_gemv_e2k_row_pair(
    const uint8_t* row0, const uint8_t* row1,
    const Q8Block* x_q8, int64_t blocks_per_row,
    float& out0, float& out1) {
    const __v16qu mask_lo4 = {0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,
                              0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F};
    const __v8hi ones16 = {1,1,1,1,1,1,1,1};
    float sum0 = 0.0f, sum1 = 0.0f;

    for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
        const uint8_t* blk0 = row0 + bi * 144;
        const uint8_t* blk1 = row1 + bi * 144;
        const Q8Block* xq = x_q8 + bi * 8;

        uint16_t d0b, dm0b, d1b, dm1b;
        std::memcpy(&d0b,  blk0,     2);
        std::memcpy(&dm0b, blk0 + 2, 2);
        std::memcpy(&d1b,  blk1,     2);
        std::memcpy(&dm1b, blk1 + 2, 2);
        float d0 = torch::io::gguf::fp16_to_fp32(d0b), dm0 = torch::io::gguf::fp16_to_fp32(dm0b);
        float d1 = torch::io::gguf::fp16_to_fp32(d1b), dm1 = torch::io::gguf::fp16_to_fp32(dm1b);
        const uint8_t* sc0 = blk0 + 4;
        const uint8_t* sc1 = blk1 + 4;
        const uint8_t* qs0 = blk0 + 16;
        const uint8_t* qs1 = blk1 + 16;

        int q8_idx = 0;
        // Process 256 elements as 4 × 64-element chunks. Each chunk uses 2
        // Q8Blocks (32 elements each) and 2 qp* passes per row (16 bytes each).
        for (int j = 0; j < 256; j += 64) {
            int is = j / 32;
            uint8_t sca0, ma0, scb0, mb0, sca1, ma1, scb1, mb1;
            torch::io::gguf::get_scale_min_k4(is,     sc0, &sca0, &ma0);
            torch::io::gguf::get_scale_min_k4(is + 1, sc0, &scb0, &mb0);
            torch::io::gguf::get_scale_min_k4(is,     sc1, &sca1, &ma1);
            torch::io::gguf::get_scale_min_k4(is + 1, sc1, &scb1, &mb1);

            // Each Q8Block holds 32 int8; we need both halves — 2 qp loads.
            __v2di q8_lo_a, q8_lo_b, q8_hi_a, q8_hi_b;
            std::memcpy(&q8_lo_a, xq[q8_idx].qs,        16);
            std::memcpy(&q8_lo_b, xq[q8_idx].qs + 16,   16);
            std::memcpy(&q8_hi_a, xq[q8_idx+1].qs,      16);
            std::memcpy(&q8_hi_b, xq[q8_idx+1].qs + 16, 16);
            float dx_lo = xq[q8_idx].d,  dx_hi = xq[q8_idx+1].d;
            float sx_lo = xq[q8_idx].sum, sx_hi = xq[q8_idx+1].sum;

            // Row 0
            __v2di raw0_a, raw0_b;
            std::memcpy(&raw0_a, qs0,      16);
            std::memcpy(&raw0_b, qs0 + 16, 16);
            __v16qu raw0_a_qu = (__v16qu)raw0_a;
            __v16qu raw0_b_qu = (__v16qu)raw0_b;
            __v16qu lo0_a = raw0_a_qu & mask_lo4;
            __v16qu hi0_a = (__v16qu)((__v8hi)raw0_a >> 4) & mask_lo4;
            __v16qu lo0_b = raw0_b_qu & mask_lo4;
            __v16qu hi0_b = (__v16qu)((__v8hi)raw0_b >> 4) & mask_lo4;

            // qpmaddubsh: 16 unsigned bytes × 16 signed bytes → 8 int16 with
            // pairwise add. Perfect match for int8×int8 dot extraction.
            __v8hi p0_lo16_a = (__v8hi)__builtin_e2k_qpmaddubsh((__v2di)lo0_a, q8_lo_a);
            __v8hi p0_lo16_b = (__v8hi)__builtin_e2k_qpmaddubsh((__v2di)lo0_b, q8_lo_b);
            __v8hi p0_hi16_a = (__v8hi)__builtin_e2k_qpmaddubsh((__v2di)hi0_a, q8_hi_a);
            __v8hi p0_hi16_b = (__v8hi)__builtin_e2k_qpmaddubsh((__v2di)hi0_b, q8_hi_b);
            // qpmaddh: 8 int16 × 8 int16 → 4 int32 pairwise. With ones, this
            // becomes int16 hsum into 4 int32 lanes.
            __v4si p0_lo32_a = (__v4si)__builtin_e2k_qpmaddh((__v2di)p0_lo16_a, (__v2di)ones16);
            __v4si p0_lo32_b = (__v4si)__builtin_e2k_qpmaddh((__v2di)p0_lo16_b, (__v2di)ones16);
            __v4si p0_hi32_a = (__v4si)__builtin_e2k_qpmaddh((__v2di)p0_hi16_a, (__v2di)ones16);
            __v4si p0_hi32_b = (__v4si)__builtin_e2k_qpmaddh((__v2di)p0_hi16_b, (__v2di)ones16);
            int32_t is_lo0 = p0_lo32_a[0]+p0_lo32_a[1]+p0_lo32_a[2]+p0_lo32_a[3]
                           + p0_lo32_b[0]+p0_lo32_b[1]+p0_lo32_b[2]+p0_lo32_b[3];
            int32_t is_hi0 = p0_hi32_a[0]+p0_hi32_a[1]+p0_hi32_a[2]+p0_hi32_a[3]
                           + p0_hi32_b[0]+p0_hi32_b[1]+p0_hi32_b[2]+p0_hi32_b[3];

            // Row 1
            __v2di raw1_a, raw1_b;
            std::memcpy(&raw1_a, qs1,      16);
            std::memcpy(&raw1_b, qs1 + 16, 16);
            __v16qu lo1_a = (__v16qu)raw1_a & mask_lo4;
            __v16qu hi1_a = (__v16qu)((__v8hi)raw1_a >> 4) & mask_lo4;
            __v16qu lo1_b = (__v16qu)raw1_b & mask_lo4;
            __v16qu hi1_b = (__v16qu)((__v8hi)raw1_b >> 4) & mask_lo4;
            __v8hi p1_lo16_a = (__v8hi)__builtin_e2k_qpmaddubsh((__v2di)lo1_a, q8_lo_a);
            __v8hi p1_lo16_b = (__v8hi)__builtin_e2k_qpmaddubsh((__v2di)lo1_b, q8_lo_b);
            __v8hi p1_hi16_a = (__v8hi)__builtin_e2k_qpmaddubsh((__v2di)hi1_a, q8_hi_a);
            __v8hi p1_hi16_b = (__v8hi)__builtin_e2k_qpmaddubsh((__v2di)hi1_b, q8_hi_b);
            __v4si p1_lo32_a = (__v4si)__builtin_e2k_qpmaddh((__v2di)p1_lo16_a, (__v2di)ones16);
            __v4si p1_lo32_b = (__v4si)__builtin_e2k_qpmaddh((__v2di)p1_lo16_b, (__v2di)ones16);
            __v4si p1_hi32_a = (__v4si)__builtin_e2k_qpmaddh((__v2di)p1_hi16_a, (__v2di)ones16);
            __v4si p1_hi32_b = (__v4si)__builtin_e2k_qpmaddh((__v2di)p1_hi16_b, (__v2di)ones16);
            int32_t is_lo1 = p1_lo32_a[0]+p1_lo32_a[1]+p1_lo32_a[2]+p1_lo32_a[3]
                           + p1_lo32_b[0]+p1_lo32_b[1]+p1_lo32_b[2]+p1_lo32_b[3];
            int32_t is_hi1 = p1_hi32_a[0]+p1_hi32_a[1]+p1_hi32_a[2]+p1_hi32_a[3]
                           + p1_hi32_b[0]+p1_hi32_b[1]+p1_hi32_b[2]+p1_hi32_b[3];

            sum0 += d0 * sca0 * dx_lo * (float)is_lo0 - dm0 * ma0 * sx_lo;
            sum0 += d0 * scb0 * dx_hi * (float)is_hi0 - dm0 * mb0 * sx_hi;
            sum1 += d1 * sca1 * dx_lo * (float)is_lo1 - dm1 * ma1 * sx_lo;
            sum1 += d1 * scb1 * dx_hi * (float)is_hi1 - dm1 * mb1 * sx_hi;

            qs0 += 32; qs1 += 32; q8_idx += 2;
        }
    }
    out0 = sum0; out1 = sum1;
}
#endif

int main(int argc, char** argv) {
    int64_t K = (argc > 1) ? std::atoll(argv[1]) : 2560;
    int64_t N = (argc > 2) ? std::atoll(argv[2]) : 9728;
    int iters = (argc > 3) ? std::atoi(argv[3]) : 20;

    const int64_t blocks_per_row = K / 256;
    const int64_t row_bytes = blocks_per_row * 144;
    const int64_t nb_q8 = K / 32;

    std::printf("Q4_K single-thread kernel probe: K=%ld N=%ld iters=%d\n", (long)K, (long)N, iters);
    std::printf("Per-row size: %ld bytes, %ld blocks\n", (long)row_bytes, (long)blocks_per_row);

    // Allocate weight matrix + x_q8 + y output
    std::vector<uint8_t> W(N * row_bytes);
    fill_q4k_blocks(W.data(), N * blocks_per_row, 0x1234);
    std::vector<Q8Block> x_q8(nb_q8);
    fill_q8_blocks(x_q8.data(), nb_q8, 0x5678);
    std::vector<float> y(N, 0.0f);

    // Quick self-correctness test: compute y for first 2 rows via both kernels
#if HAVE_E2K_NATIVE && defined(__AVX2__)
    float y0_avx=0, y1_avx=0, y0_e2k=0, y1_e2k=0;
    q4k_gemv_avx2_row_pair(W.data(), W.data() + row_bytes, x_q8.data(), blocks_per_row, y0_avx, y1_avx);
    q4k_gemv_e2k_row_pair (W.data(), W.data() + row_bytes, x_q8.data(), blocks_per_row, y0_e2k, y1_e2k);
    std::printf("Correctness row0: avx=%.4f  e2k=%.4f  diff=%.6f\n", y0_avx, y0_e2k, std::abs(y0_avx - y0_e2k));
    std::printf("Correctness row1: avx=%.4f  e2k=%.4f  diff=%.6f\n", y1_avx, y1_e2k, std::abs(y1_avx - y1_e2k));
#endif

    // Timing loop — process N rows in pairs, iters times, single thread.
#ifdef __AVX2__
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int64_t n = 0; n + 1 < N; n += 2) {
            float a, b;
            q4k_gemv_avx2_row_pair(W.data() + n * row_bytes, W.data() + (n+1) * row_bytes,
                                    x_q8.data(), blocks_per_row, a, b);
            y[n] = a; y[n+1] = b;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_avx = std::chrono::duration<double, std::milli>(t1-t0).count();
    double ns_per_row_avx = ms_avx * 1e6 / (N * iters);
    std::printf("[AVX2→LCC]  total=%.1f ms, %.0f ns/row\n", ms_avx, ns_per_row_avx);
#endif

#if HAVE_E2K_NATIVE
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int64_t n = 0; n + 1 < N; n += 2) {
            float a, b;
            q4k_gemv_e2k_row_pair(W.data() + n * row_bytes, W.data() + (n+1) * row_bytes,
                                   x_q8.data(), blocks_per_row, a, b);
            y[n] = a; y[n+1] = b;
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_e2k = std::chrono::duration<double, std::milli>(t3-t2).count();
    double ns_per_row_e2k = ms_e2k * 1e6 / (N * iters);
    std::printf("[E2K native] total=%.1f ms, %.0f ns/row\n", ms_e2k, ns_per_row_e2k);
#  ifdef __AVX2__
    std::printf("Ratio: %.3fx (%.1f%% %s)\n",
                ms_avx / ms_e2k,
                100.0 * (ms_avx - ms_e2k) / ms_avx,
                (ms_e2k < ms_avx) ? "faster" : "slower");
#  endif
#else
    std::printf("[E2K native] NOT AVAILABLE (no e2kintrin.h)\n");
#endif

    return 0;
}
