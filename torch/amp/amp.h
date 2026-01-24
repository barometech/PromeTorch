#pragma once

// ============================================================================
// Automatic Mixed Precision (AMP) Module
// ============================================================================
//
// This module provides automatic mixed precision training support, allowing
// models to train with FP16/BF16 for faster computation while maintaining
// FP32 precision where needed for numerical stability.
//
// Key Components:
// - GradScaler: Dynamic loss scaling to prevent gradient underflow in FP16
// - Autocast: Automatic type casting based on operation requirements
//
// Benefits:
// - 2x memory reduction (FP16 vs FP32)
// - 2-3x speedup on Tensor Core GPUs (Volta+)
// - Maintains model accuracy
//
// ============================================================================
// Usage Example
// ============================================================================
//
//   #include "torch/amp/amp.h"
//
//   using namespace torch::amp;
//
//   // Create model and optimizer
//   auto model = MyModel();
//   auto optimizer = torch::optim::Adam(model.parameters(), 1e-3);
//
//   // Create gradient scaler
//   GradScaler scaler;
//
//   // Training loop
//   for (auto& batch : dataloader) {
//       auto [data, target] = batch;
//       optimizer.zero_grad();
//
//       // Forward pass with autocast
//       {
//           AutocastGuard guard(c10::ScalarType::Half);  // FP16
//           auto output = model(data);
//           auto loss = torch::nn::cross_entropy(output, target);
//
//           // Scale loss and backward (still in autocast scope is OK)
//           scaler.scale(loss).backward();
//       }
//
//       // Unscale gradients, clip, and optimizer step
//       scaler.unscale(optimizer);
//       torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
//       scaler.step(optimizer);
//
//       // Update scaler
//       scaler.update();
//   }
//
// ============================================================================
// BFloat16 on CPU
// ============================================================================
//
//   // CPU training with BF16 (supported on newer CPUs with AMX/AVX-512)
//   GradScaler scaler;
//   {
//       AutocastCPUGuard guard(c10::ScalarType::BFloat16);
//       auto output = model(data);
//       ...
//   }
//
// ============================================================================

#include "torch/amp/grad_scaler.h"
#include "torch/amp/autocast.h"

namespace torch {
namespace amp {

// ============================================================================
// Convenience Functions
// ============================================================================

// Create a GradScaler with default options
inline GradScaler make_grad_scaler() {
    return GradScaler();
}

// Create a GradScaler with custom initial scale
inline GradScaler make_grad_scaler(double init_scale) {
    GradScalerOptions opts;
    opts.init_scale = init_scale;
    return GradScaler(opts);
}

// Create disabled scaler (for FP32 training or inference)
inline GradScaler make_disabled_scaler() {
    GradScalerOptions opts;
    opts.enabled = false;
    return GradScaler(opts);
}

// ============================================================================
// Helper: Cast model parameters to half precision
// ============================================================================

// Cast all model parameters to FP16
template<typename Module>
void half(Module& module) {
    for (auto& [name, param] : module.named_parameters()) {
        if (param.data().dtype() == c10::ScalarType::Float ||
            param.data().dtype() == c10::ScalarType::Double) {
            param.data() = param.data().to(c10::ScalarType::Half);
        }
    }
}

// Cast all model parameters to BF16
template<typename Module>
void bfloat16(Module& module) {
    for (auto& [name, param] : module.named_parameters()) {
        if (param.data().dtype() == c10::ScalarType::Float ||
            param.data().dtype() == c10::ScalarType::Double) {
            param.data() = param.data().to(c10::ScalarType::BFloat16);
        }
    }
}

// Cast all model parameters to FP32
template<typename Module>
void float32(Module& module) {
    for (auto& [name, param] : module.named_parameters()) {
        if (param.data().dtype() == c10::ScalarType::Half ||
            param.data().dtype() == c10::ScalarType::BFloat16) {
            param.data() = param.data().to(c10::ScalarType::Float);
        }
    }
}

// ============================================================================
// Mixed Precision Training Strategy
// ============================================================================
//
// Recommended strategy for mixed precision training:
//
// 1. Keep master weights in FP32 (handled automatically by optimizers)
//    - Optimizer states (momentum, variance) stay in FP32
//    - Weight updates happen in FP32 then cast back
//
// 2. Forward pass in FP16:
//    - Inputs cast to FP16
//    - Linear layers, convolutions use FP16 (Tensor Cores)
//    - Activations stored in FP16
//
// 3. Loss computation in FP32:
//    - Softmax, cross-entropy need FP32 for stability
//    - Scale loss before backward
//
// 4. Backward pass in FP16:
//    - Gradients computed in FP16
//    - Accumulated in FP32
//
// 5. Gradient unscaling and update:
//    - Unscale gradients (divide by loss scale)
//    - Check for inf/nan
//    - Update weights in FP32
//    - Adjust loss scale dynamically
//
// ============================================================================

// ============================================================================
// Version and Feature Detection
// ============================================================================

// AMP version
constexpr int AMP_VERSION_MAJOR = 1;
constexpr int AMP_VERSION_MINOR = 0;

// Check if Tensor Cores are likely available
// (This is a heuristic - actual detection requires CUDA queries)
inline bool has_tensor_cores() {
#ifdef PT_USE_CUDA
    // Tensor Cores available on Volta (SM 7.0) and later
    // This is a compile-time check; runtime detection would need cudaDeviceGetAttribute
    return true;  // Assume modern GPU
#else
    return false;
#endif
}

// Recommended dtype for current hardware
inline c10::ScalarType recommended_autocast_dtype() {
    // FP16 is generally best for CUDA with Tensor Cores
    // BF16 is better for CPU (AMX) and some cases where dynamic range matters
#ifdef PT_USE_CUDA
    return c10::ScalarType::Half;
#else
    return c10::ScalarType::BFloat16;
#endif
}

} // namespace amp
} // namespace torch
