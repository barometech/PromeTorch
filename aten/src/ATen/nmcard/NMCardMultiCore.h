#pragma once
// ============================================================================
// NMCardMultiCore.h — 16-core parallel dispatch for NM Card Mini
// ============================================================================
// Splits matmul across 4 clusters × 4 cores = 16 NMC4 cores.
// Each core computes a column range of the output matrix.
//
// Key discovery: each core must be read/written through ITS OWN
// PL_Access handle (per-cluster DDR mapping).
//
// dispatcher_mc.abs layout:
//   DDR_BASE + core_index * 32 = cmd block for core_index
//   core_index = cluster_id * 4 + core_id (0..15)
// ============================================================================

#include "aten/src/ATen/nmcard/NMCardHardware.h"
#include <algorithm>
#include <thread>
#include <chrono>

namespace at {
namespace nmcard {

constexpr int MAX_CORES = 16;
constexpr uint32_t OP_MATMUL_PARTIAL = 22;

class NMCardMultiCore {
public:
    static NMCardMultiCore& get() {
        static NMCardMultiCore instance;
        return instance;
    }

    // Initialize all 16 cores with dispatcher_mc.abs
    bool init(const std::string& dispatcher_mc_path) {
        if (initialized_) return true;

        auto& hw = NMCardHardware::get();
        if (!hw.is_available()) {
            // Need base hardware init first
            return false;
        }

        // dispatcher_mc.abs is loaded on all cores during init
        // Each core determines its own index via ncl_getCoreID/ncl_getClusterID
        num_active_ = hw.num_cores();  // Should be 16 after mc init
        initialized_ = (num_active_ > 1);
        return initialized_;
    }

    bool is_available() const { return initialized_; }
    int num_cores() const { return num_active_; }

    // Parallel matmul: split columns across cores
    // C[M,N] = A[M,K] @ B[K,N]
    // Core i computes C[:, col_start:col_end] where col_range = N/num_cores
    void parallel_matmul(const float* A, const float* B, float* C,
                         int64_t M, int64_t K, int64_t N) {
        if (!initialized_ || num_active_ <= 1) {
            // Fallback to single-core
            NMCardHardware::get().matmul(A, B, C, M, K, N);
            return;
        }

        auto& hw = NMCardHardware::get();

        // Upload A and B once (shared by all cores)
        hw.ddr().reset();
        uint32_t addr_A = hw.upload(A, M * K);
        uint32_t addr_B = hw.upload(B, K * N);
        uint32_t addr_C = hw.ddr().alloc(M * N * sizeof(float));

        // Divide columns among cores
        int cols_per_core = static_cast<int>(N) / num_active_;
        int remainder = static_cast<int>(N) % num_active_;

        // Send OP_MATMUL_PARTIAL to each core
        int col_start = 0;
        for (int core = 0; core < num_active_; ++core) {
            int cols = cols_per_core + (core < remainder ? 1 : 0);
            int col_end = col_start + cols;

            // args: [M, K, N, addr_A, addr_B, addr_C, col_start, col_end]
            uint32_t args[8] = {
                static_cast<uint32_t>(M),
                static_cast<uint32_t>(K),
                static_cast<uint32_t>(N),
                addr_A, addr_B, addr_C,
                static_cast<uint32_t>(col_start),
                static_cast<uint32_t>(col_end)
            };
            hw.dispatch_op(OP_MATMUL_PARTIAL, args, 8, core);
            col_start = col_end;
        }

        // Wait for all cores
        for (int core = 0; core < num_active_; ++core) {
            hw.wait_done(core, 10.0f);
        }

        // Download result
        hw.download(C, addr_C, M * N);
    }

private:
    NMCardMultiCore() = default;
    bool initialized_ = false;
    int num_active_ = 0;
};

} // namespace nmcard
} // namespace at
