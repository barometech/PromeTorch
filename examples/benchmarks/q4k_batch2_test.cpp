// Correctness + perf test for q4k_gemv_avx2_batch2.
// Compares against 2 sequential q4k_gemv_avx2 calls on synthetic data.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

#include "torch/io/cpu_quant_gemv.h"

static void fill_q4k_blocks(uint8_t* buf, int64_t blocks, uint32_t seed) {
    std::mt19937 rng(seed);
    for (int64_t bi = 0; bi < blocks; ++bi) {
        uint8_t* blk = buf + bi * 144;
        uint16_t d = 0x3c00;    // fp16 1.0
        uint16_t dm = 0x0000;
        std::memcpy(blk, &d, 2);
        std::memcpy(blk + 2, &dm, 2);
        for (int i = 4; i < 144; ++i) blk[i] = static_cast<uint8_t>(rng());
    }
}

int main(int argc, char** argv) {
    int64_t K = (argc > 1) ? std::atoll(argv[1]) : 2560;
    int64_t N = (argc > 2) ? std::atoll(argv[2]) : 9728;
    int iters = (argc > 3) ? std::atoi(argv[3]) : 10;

    const int64_t blocks_per_row = K / 256;
    const int64_t row_bytes = blocks_per_row * 144;

    std::printf("q4k batch2 test: K=%ld N=%ld iters=%d\n", (long)K, (long)N, iters);

    std::vector<uint8_t> W(N * row_bytes);
    fill_q4k_blocks(W.data(), N * blocks_per_row, 0xCAFE);

    // Two random x vectors.
    std::vector<float> x0(K), x1(K);
    std::mt19937 rng(0xBEEF);
    for (auto& v : x0) v = (rng() % 2000 - 1000) / 100.0f;
    for (auto& v : x1) v = (rng() % 2000 - 1000) / 100.0f;

    std::vector<float> y0_ref(N), y1_ref(N);
    std::vector<float> y0_bat(N), y1_bat(N);

    // Reference: two separate calls.
    torch::io::cpu_quant::q4k_gemv_avx2(W.data(), x0.data(), y0_ref.data(), K, N, row_bytes);
    torch::io::cpu_quant::q4k_gemv_avx2(W.data(), x1.data(), y1_ref.data(), K, N, row_bytes);

    // Test: single batched call.
    torch::io::cpu_quant::q4k_gemv_avx2_batch2(W.data(), x0.data(), x1.data(),
                                    y0_bat.data(), y1_bat.data(),
                                    K, N, row_bytes);

    // Compare — allow small float-order difference (batched accum in different order).
    float max_rel = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
        float r0 = std::abs(y0_ref[i] - y0_bat[i]) / (std::abs(y0_ref[i]) + 1e-6f);
        float r1 = std::abs(y1_ref[i] - y1_bat[i]) / (std::abs(y1_ref[i]) + 1e-6f);
        if (r0 > max_rel) max_rel = r0;
        if (r1 > max_rel) max_rel = r1;
    }
    std::printf("max relative error: %.2e %s\n", max_rel,
                max_rel < 1e-4 ? "(OK)" : "(FAIL)");

    // Timing: serial vs batched.
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        torch::io::cpu_quant::q4k_gemv_avx2(W.data(), x0.data(), y0_ref.data(), K, N, row_bytes);
        torch::io::cpu_quant::q4k_gemv_avx2(W.data(), x1.data(), y1_ref.data(), K, N, row_bytes);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_serial = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        torch::io::cpu_quant::q4k_gemv_avx2_batch2(W.data(), x0.data(), x1.data(),
                                        y0_bat.data(), y1_bat.data(),
                                        K, N, row_bytes);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_batched = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("[serial  2× q4k_gemv_avx2] %.1f ms total, %.2f ms/call-pair\n",
                ms_serial, ms_serial / iters);
    std::printf("[batched q4k_gemv_batch2 ] %.1f ms total, %.2f ms/call\n",
                ms_batched, ms_batched / iters);
    std::printf("speedup (serial/batched):  %.3fx  %s\n",
                ms_serial / ms_batched,
                ms_batched < ms_serial ? "(batched wins)" : "(serial wins)");
    return 0;
}
