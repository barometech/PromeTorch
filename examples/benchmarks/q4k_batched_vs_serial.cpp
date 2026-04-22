// ============================================================================
// Phase 7.1 verification — batched Q4_K GEMV vs K serial calls
// ============================================================================
//
// Synthesises a Q4_K weight matrix of shape [N, H] and runs:
//   (a) K separate cpu_quant_gemv calls (old serial path)
//   (b) 1 cpu_quant_gemv_batched call with same K queries
// Compares outputs element-wise and reports speedup.
//
// Success criteria:
//   * max_abs < 1e-3 (float-associativity tolerance)
//   * batched_ms < serial_ms  (i.e., batched beats K separate calls)
//
// Usage: ./q4k_batched_vs_serial [K=2] [N=9728] [H=2560] [iters=20]
// (default shape = qwen3:4b ffn_gate — the biggest hot GEMV)
// ============================================================================

#include "torch/io/cpu_quant_gemv.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

using torch::io::cpu_quant::cpu_quant_gemv;
using torch::io::cpu_quant::cpu_quant_gemv_batched;

static void fill_q4k(uint8_t* buf, int64_t rows, int64_t cols, uint32_t seed) {
    int64_t blocks_per_row = cols / 256;
    std::mt19937 rng(seed);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            uint8_t* blk = buf + (r * blocks_per_row + bi) * 144;
            uint16_t d  = 0x3800;      // fp16 ~0.5
            uint16_t dm = 0x2c00;      // fp16 ~0.0625
            std::memcpy(blk,     &d,  2);
            std::memcpy(blk + 2, &dm, 2);
            for (int i = 4; i < 144; ++i) blk[i] = static_cast<uint8_t>(rng());
        }
    }
}

int main(int argc, char** argv) {
    int   K     = (argc >= 2) ? std::atoi(argv[1]) : 2;
    int64_t N   = (argc >= 3) ? std::atoll(argv[2]) : 9728;
    int64_t H   = (argc >= 4) ? std::atoll(argv[3]) : 2560;
    int   iters = (argc >= 5) ? std::atoi(argv[4]) : 20;
    if (K < 1) K = 1;
    if (K > 4) K = 4;

    std::printf("config: K=%d  N=%ld  H=%ld  iters=%d\n", K, (long)N, (long)H, iters);
    if (H % 256 != 0) { std::fprintf(stderr, "H must be multiple of 256\n"); return 1; }

    int64_t blocks_per_row = H / 256;
    int64_t row_stride_bytes = blocks_per_row * 144;
    std::vector<uint8_t> W(N * row_stride_bytes);
    fill_q4k(W.data(), N, H, 42);

    // K inputs
    std::vector<float> x(K * H);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : x) v = dist(rng);

    std::vector<float> y_serial (K * N, 0.f);
    std::vector<float> y_batched(K * N, 0.f);

    // Warm-up
    for (int k = 0; k < K; ++k) {
        cpu_quant_gemv(12, W.data(), x.data() + k * H,
                       y_serial.data() + k * N, H, N, row_stride_bytes);
    }
    cpu_quant_gemv_batched(12, W.data(), x.data(), y_batched.data(),
                           K, H, N, row_stride_bytes);

    // Correctness
    double max_abs = 0.0;
    for (int64_t i = 0; i < K * N; ++i) {
        double diff = std::fabs((double)y_serial[i] - (double)y_batched[i]);
        if (diff > max_abs) max_abs = diff;
    }
    std::printf("correctness: max_abs_diff = %.6g\n", max_abs);
    if (max_abs > 1e-3) {
        std::printf("FAIL: batched diverges from serial\n");
        return 3;
    }

    // Timing — serial path
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        for (int k = 0; k < K; ++k) {
            cpu_quant_gemv(12, W.data(), x.data() + k * H,
                           y_serial.data() + k * N, H, N, row_stride_bytes);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // Timing — batched
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cpu_quant_gemv_batched(12, W.data(), x.data(), y_batched.data(),
                               K, H, N, row_stride_bytes);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double batched_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    double speedup   = serial_ms / batched_ms;
    double weight_GB = (double)W.size() / 1e9;
    double serial_bw  = (weight_GB * K) / (serial_ms / 1000.0);
    double batched_bw =  weight_GB       / (batched_ms / 1000.0);

    std::printf("serial  (K=%d calls):  %7.2f ms/pass   effective BW = %.2f GB/s (K weight reads)\n",
                K, serial_ms, serial_bw);
    std::printf("batched (K=%d fused):  %7.2f ms/pass   effective BW = %.2f GB/s (1 weight read, K outputs)\n",
                K, batched_ms, batched_bw);
    std::printf("speedup: %.2fx   (theoretical max on BW-bound: K=%d → %dx)\n",
                speedup, K, K);

    return 0;
}
