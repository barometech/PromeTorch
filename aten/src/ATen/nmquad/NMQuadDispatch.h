#pragma once
// ============================================================================
// NMQuadDispatch.h — Tensor dispatch for NM QUAD backend
// ============================================================================
// Provides to_nmquad(), nmquad_to_cpu(), and op dispatch for NM QUAD tensors.
// Analogous to NMCardDispatch.h but targeting real NM6408 hardware.

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/nmquad/NMQuadHardware.h"
#include "aten/src/ATen/nmquad/NMQuadOps.h"
#include "c10/nmquad/NMQuadAllocator.h"

namespace at {

// ============================================================================
// Device transfer
// ============================================================================

// Move CPU tensor to NM QUAD (chip 0 by default)
inline Tensor to_nmquad(const Tensor& cpu_tensor, int chip_id = 0) {
    if (cpu_tensor.device().type() == c10::DeviceType::PrivateUse3) {
        return cpu_tensor;  // Already on NM QUAD
    }

    Tensor contig = cpu_tensor.contiguous();
    auto& alloc = c10::nmquad::NMQuadAllocator::get();
    auto data = alloc.allocate_on_chip(contig.nbytes(), chip_id);

    // Copy data
    std::memcpy(data.get(), contig.data_ptr(), contig.nbytes());

    // Create tensor with NM QUAD device
    auto impl = std::make_shared<c10::TensorImpl>(
        c10::Storage(std::move(data)),
        contig.dtype(),
        contig.sizes()
    );
    return Tensor(impl);
}

// Move NM QUAD tensor back to CPU
inline Tensor nmquad_to_cpu(const Tensor& nmquad_tensor) {
    if (nmquad_tensor.device().type() == c10::DeviceType::CPU) {
        return nmquad_tensor;
    }

    Tensor contig = nmquad_tensor.contiguous();
    Tensor cpu = at::empty(contig.sizes(), TensorOptions().dtype(contig.dtype()));
    std::memcpy(cpu.data_ptr(), contig.data_ptr(), contig.nbytes());
    return cpu;
}

// ============================================================================
// NM QUAD initialization
// ============================================================================

inline bool init_nmquad(int num_chips = 4) {
    // Register allocator
    c10::nmquad::register_nmquad_allocator();

    // Initialize hardware
    auto& hw = nmquad::NMQuadHardware::get();
    if (!hw.init(num_chips)) {
        std::cerr << "NM QUAD init failed: " << hw.last_error() << std::endl;
        return false;
    }

    std::cout << "NM QUAD initialized: " << hw.num_chips() << " chips" << std::endl;
    return true;
}

// Load dispatcher on all chips
inline bool load_nmquad_dispatcher(const std::string& abs_path) {
    auto& hw = nmquad::NMQuadHardware::get();
    for (int i = 0; i < hw.num_chips(); ++i) {
        if (!hw.load_dispatcher(i, abs_path)) {
            std::cerr << "Failed to load dispatcher on chip " << i
                      << ": " << hw.last_error() << std::endl;
            return false;
        }
    }
    std::cout << "Dispatcher loaded on all " << hw.num_chips() << " chips" << std::endl;
    return true;
}

// ============================================================================
// Operation dispatch (same signatures as NMCardDispatch)
// ============================================================================

// Matrix multiply — dispatch to NM6408
inline Tensor mm_nmquad(const Tensor& a, const Tensor& b) {
    return nmquad::matmul_nmquad(a, b, 0);
}

// Multi-chip matrix multiply
inline Tensor mm_nmquad_multi(const Tensor& a, const Tensor& b) {
    return nmquad::matmul_multi_chip(a, b);
}

// Element-wise add
inline Tensor add_nmquad(const Tensor& a, const Tensor& b) {
    return nmquad::add_nmquad(a, b, 0);
}

// ReLU
inline Tensor relu_nmquad(const Tensor& x) {
    return nmquad::relu_nmquad(x, 0);
}

} // namespace at
