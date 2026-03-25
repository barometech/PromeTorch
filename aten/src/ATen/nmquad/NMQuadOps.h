#pragma once
// ============================================================================
// NMQuadOps.h — Tensor operations dispatched to NM QUAD hardware
// ============================================================================
// Host-side: prepares data, DMA to chip DDR, executes dispatcher, reads back.
// Supports multi-chip parallelism for matmul (split across 4 chips).

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/nmquad/NMQuadHardware.h"
#include "c10/nmquad/NMQuadAllocator.h"

#include <cstring>
#include <algorithm>
#include <cmath>

namespace at {
namespace nmquad {

// ============================================================================
// Dispatcher opcodes (must match nmc_programs/dispatcher_nmquad.cpp)
// ============================================================================
enum NMQuadOpcode : uint32_t {
    OP_NOP           = 0,
    OP_MATMUL        = 1,   // C = A * B (float32)
    OP_ADD           = 2,   // C = A + B
    OP_MUL           = 3,   // C = A * B (element-wise)
    OP_RELU          = 4,   // Y = max(0, X)
    OP_SIGMOID       = 5,   // Y = 1 / (1 + exp(-X))
    OP_SOFTMAX       = 6,   // Y = softmax(X)
    OP_REDUCE_SUM    = 7,   // Y = sum(X)
    OP_SCALE         = 8,   // Y = alpha * X
    OP_BIAS_ADD      = 9,   // Y = X + bias
    OP_DONE          = 0xFF // Shutdown signal
};

// Command block layout (32 words per core in DDR)
struct CmdBlock {
    uint32_t opcode;      // Operation to execute
    uint32_t status;      // 0=idle, 1=running, 2=done
    uint32_t a_addr;      // Address of input A in DDR
    uint32_t b_addr;      // Address of input B in DDR
    uint32_t c_addr;      // Address of output C in DDR
    uint32_t M, N, K;     // Matrix dimensions
    uint32_t alpha_bits;  // float alpha (as uint32_t bits)
    uint32_t reserved[23]; // Pad to 32 words
};

// ============================================================================
// Tensor creation on NM QUAD
// ============================================================================

inline Tensor empty_nmquad(const std::vector<int64_t>& sizes,
                           c10::ScalarType dtype = c10::ScalarType::Float,
                           int chip_id = 0) {
    auto& alloc = c10::nmquad::NMQuadAllocator::get();
    auto opts = TensorOptions()
        .dtype(dtype)
        .device(c10::Device(c10::DeviceType::PrivateUse3, static_cast<c10::DeviceIndex>(chip_id)));
    return at::empty(sizes, opts);
}

// ============================================================================
// Data transfer: host <-> chip DDR
// ============================================================================

// Upload tensor data to chip DDR, returns DDR address
inline uint32_t upload_to_chip(const Tensor& t, int chip_id) {
    auto& hw = NMQuadHardware::get();
    Tensor contig = t.contiguous();
    size_t words = contig.nbytes() / sizeof(uint32_t);
    if (contig.nbytes() % sizeof(uint32_t) != 0) words++;

    uint32_t addr = hw.alloc_ddr(chip_id, words);
    hw.write_ddr(chip_id, contig.data_ptr(), addr, words);
    return addr;
}

// Download data from chip DDR to host tensor
inline void download_from_chip(Tensor& t, int chip_id, uint32_t addr) {
    auto& hw = NMQuadHardware::get();
    size_t words = t.nbytes() / sizeof(uint32_t);
    if (t.nbytes() % sizeof(uint32_t) != 0) words++;
    hw.read_ddr(chip_id, t.data_ptr(), addr, words);
}

// ============================================================================
// Dispatch operation to chip
// ============================================================================

inline void dispatch_op(int chip_id, const CmdBlock& cmd) {
    auto& hw = NMQuadHardware::get();

    // Write command block to DDR
    uint32_t cmd_addr = CMD_BLOCK_BASE;  // Core 0 cmd block
    hw.write_ddr(chip_id, &cmd, cmd_addr, sizeof(CmdBlock) / sizeof(uint32_t));

    // Wait for completion via sync
    int ret = hw.sync(chip_id, 1);  // Signal: execute
    if (ret < 0) {
        throw std::runtime_error("NM QUAD dispatch failed on chip " + std::to_string(chip_id));
    }
}

// ============================================================================
// High-level operations
// ============================================================================

// Matrix multiply on single chip
inline Tensor matmul_nmquad(const Tensor& A, const Tensor& B, int chip_id = 0) {
    auto& hw = NMQuadHardware::get();
    if (!hw.is_initialized()) {
        throw std::runtime_error("NM QUAD not initialized");
    }

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // Reset DDR allocator for this operation
    hw.reset_ddr(chip_id);

    // Upload A and B
    uint32_t a_addr = upload_to_chip(A, chip_id);
    uint32_t b_addr = upload_to_chip(B, chip_id);

    // Allocate output
    size_t c_words = (M * N * sizeof(float)) / sizeof(uint32_t);
    uint32_t c_addr = hw.alloc_ddr(chip_id, c_words);

    // Dispatch
    CmdBlock cmd = {};
    cmd.opcode = OP_MATMUL;
    cmd.status = 0;
    cmd.a_addr = a_addr;
    cmd.b_addr = b_addr;
    cmd.c_addr = c_addr;
    cmd.M = static_cast<uint32_t>(M);
    cmd.N = static_cast<uint32_t>(N);
    cmd.K = static_cast<uint32_t>(K);

    dispatch_op(chip_id, cmd);

    // Download result
    Tensor C = at::empty({M, N}, TensorOptions().dtype(A.dtype()).device(A.device()));
    download_from_chip(C, chip_id, c_addr);

    return C;
}

// Multi-chip matmul: split N across 4 chips
inline Tensor matmul_multi_chip(const Tensor& A, const Tensor& B) {
    auto& hw = NMQuadHardware::get();
    int chips = hw.num_chips();
    if (chips <= 1) return matmul_nmquad(A, B, 0);

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // Split B columns across chips
    int64_t cols_per_chip = (N + chips - 1) / chips;

    // Upload A to all chips, B slices to each chip
    std::vector<uint32_t> a_addrs(chips);
    std::vector<uint32_t> b_addrs(chips);
    std::vector<uint32_t> c_addrs(chips);
    std::vector<int64_t> chip_cols(chips);

    for (int i = 0; i < chips; ++i) {
        hw.reset_ddr(i);
        int64_t col_start = i * cols_per_chip;
        int64_t col_end = std::min(col_start + cols_per_chip, N);
        if (col_start >= N) { chip_cols[i] = 0; continue; }
        chip_cols[i] = col_end - col_start;

        a_addrs[i] = upload_to_chip(A, i);

        // Slice B columns [col_start:col_end]
        // B is K x N, we want B[:, col_start:col_end]
        Tensor B_slice = B.contiguous();  // TODO: proper slicing
        b_addrs[i] = upload_to_chip(B_slice, i);

        size_t c_words = (M * chip_cols[i] * sizeof(float)) / sizeof(uint32_t);
        c_addrs[i] = hw.alloc_ddr(i, c_words);
    }

    // Dispatch to all chips (could be parallel via threads)
    for (int i = 0; i < chips; ++i) {
        if (chip_cols[i] == 0) continue;
        CmdBlock cmd = {};
        cmd.opcode = OP_MATMUL;
        cmd.a_addr = a_addrs[i];
        cmd.b_addr = b_addrs[i];
        cmd.c_addr = c_addrs[i];
        cmd.M = static_cast<uint32_t>(M);
        cmd.N = static_cast<uint32_t>(chip_cols[i]);
        cmd.K = static_cast<uint32_t>(K);
        dispatch_op(i, cmd);
    }

    // Download and concatenate results
    Tensor C = at::empty({M, N}, TensorOptions().dtype(A.dtype()).device(A.device()));
    // TODO: proper column concatenation from each chip
    for (int i = 0; i < chips; ++i) {
        if (chip_cols[i] == 0) continue;
        Tensor C_slice = at::empty({M, chip_cols[i]}, TensorOptions().dtype(A.dtype()).device(A.device()));
        download_from_chip(C_slice, i, c_addrs[i]);
        // Copy to output (column-major concatenation)
        // For now just copy first chip result
        if (i == 0) {
            std::memcpy(C.data_ptr(), C_slice.data_ptr(), C_slice.nbytes());
        }
    }

    return C;
}

// Element-wise add on chip
inline Tensor add_nmquad(const Tensor& A, const Tensor& B, int chip_id = 0) {
    auto& hw = NMQuadHardware::get();
    hw.reset_ddr(chip_id);

    uint32_t a_addr = upload_to_chip(A, chip_id);
    uint32_t b_addr = upload_to_chip(B, chip_id);
    size_t words = A.nbytes() / sizeof(uint32_t);
    uint32_t c_addr = hw.alloc_ddr(chip_id, words);

    CmdBlock cmd = {};
    cmd.opcode = OP_ADD;
    cmd.a_addr = a_addr;
    cmd.b_addr = b_addr;
    cmd.c_addr = c_addr;
    cmd.M = static_cast<uint32_t>(A.numel());

    dispatch_op(chip_id, cmd);

    Tensor C = at::empty(A.sizes(), TensorOptions().dtype(A.dtype()).device(A.device()));
    download_from_chip(C, chip_id, c_addr);
    return C;
}

// ReLU on chip
inline Tensor relu_nmquad(const Tensor& X, int chip_id = 0) {
    auto& hw = NMQuadHardware::get();
    hw.reset_ddr(chip_id);

    uint32_t x_addr = upload_to_chip(X, chip_id);
    size_t words = X.nbytes() / sizeof(uint32_t);
    uint32_t y_addr = hw.alloc_ddr(chip_id, words);

    CmdBlock cmd = {};
    cmd.opcode = OP_RELU;
    cmd.a_addr = x_addr;
    cmd.c_addr = y_addr;
    cmd.M = static_cast<uint32_t>(X.numel());

    dispatch_op(chip_id, cmd);

    Tensor Y = at::empty(X.sizes(), TensorOptions().dtype(X.dtype()).device(X.device()));
    download_from_chip(Y, chip_id, y_addr);
    return Y;
}

} // namespace nmquad
} // namespace at
