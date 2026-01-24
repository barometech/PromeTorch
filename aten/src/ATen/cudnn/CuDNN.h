#pragma once

// ============================================================================
// cuDNN Integration - Main Header
// ============================================================================
// High-performance deep learning primitives using NVIDIA cuDNN library.
//
// This header provides:
// - Thread-safe cuDNN handle management (CuDNNHandle)
// - Convolution operations (forward, backward data, backward filter)
// - Pooling operations (max pool, avg pool, adaptive pool)
// - Batch normalization (training, inference)
// - Activation functions (ReLU, sigmoid, tanh, ELU, Swish)
// - Softmax operations (softmax, log_softmax)
//
// All operations automatically select the best algorithm for the given
// input sizes and use Tensor Cores when available (FP16).
//
// Usage:
//   #include "aten/src/ATen/cudnn/CuDNN.h"
//
//   // Convolution
//   Tensor output = at::cudnn::cudnn_convolution_forward(
//       input, weight, padH, padW, strideH, strideW, dilationH, dilationW, groups
//   );
//
//   // Pooling
//   Tensor pooled = at::cudnn::cudnn_max_pool2d_forward(
//       input, kernelH, kernelW, strideH, strideW, padH, padW
//   );
//
//   // Batch Norm
//   auto [out, mean, var] = at::cudnn::cudnn_batch_norm_forward_training(
//       input, gamma, beta, running_mean, running_var, momentum, epsilon
//   );
//
//   // Activation
//   Tensor relu_out = at::cudnn::cudnn_relu_forward(input);
//
// ============================================================================

#ifdef PT_USE_CUDA

#include "aten/src/ATen/cudnn/CuDNNHandle.h"
#include "aten/src/ATen/cudnn/CuDNNConvolution.h"
#include "aten/src/ATen/cudnn/CuDNNPooling.h"
#include "aten/src/ATen/cudnn/CuDNNBatchNorm.h"
#include "aten/src/ATen/cudnn/CuDNNActivation.h"

namespace at {
namespace cudnn {

// ============================================================================
// cuDNN Version Info
// ============================================================================

inline int cudnn_version() {
    return CUDNN_VERSION;
}

inline int cudnn_major_version() {
    return CUDNN_MAJOR;
}

inline int cudnn_minor_version() {
    return CUDNN_MINOR;
}

inline int cudnn_patch_level() {
    return CUDNN_PATCHLEVEL;
}

// Runtime version
inline size_t cudnn_runtime_version() {
    return cudnnGetVersion();
}

// ============================================================================
// Device Detection for cuDNN Dispatch
// ============================================================================

// Check if tensor is on CUDA and cuDNN should be used
inline bool should_use_cudnn(const Tensor& t) {
    return t.device().type() == c10::DeviceType::CUDA;
}

// Check if multiple tensors are on CUDA
inline bool should_use_cudnn(const Tensor& a, const Tensor& b) {
    return a.device().type() == c10::DeviceType::CUDA &&
           b.device().type() == c10::DeviceType::CUDA;
}

inline bool should_use_cudnn(const Tensor& a, const Tensor& b, const Tensor& c) {
    return a.device().type() == c10::DeviceType::CUDA &&
           b.device().type() == c10::DeviceType::CUDA &&
           c.device().type() == c10::DeviceType::CUDA;
}

// ============================================================================
// High-Level Dispatch Functions
// ============================================================================
// These automatically choose between cuDNN and CPU implementations

// Conv2d with automatic dispatch
inline Tensor conv2d_dispatch(
    const Tensor& input,
    const Tensor& weight,
    int64_t padH, int64_t padW,
    int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW,
    int64_t groups
) {
    if (should_use_cudnn(input, weight)) {
        return cudnn_convolution_forward(
            input, weight, padH, padW, strideH, strideW, dilationH, dilationW, groups
        );
    }
    // Fallback to CPU implementation (im2col)
    PT_ERROR("CPU convolution fallback not implemented in cuDNN dispatch");
}

// MaxPool2d with automatic dispatch
inline Tensor max_pool2d_dispatch(
    const Tensor& input,
    int64_t kernelH, int64_t kernelW,
    int64_t strideH, int64_t strideW,
    int64_t padH, int64_t padW
) {
    if (should_use_cudnn(input)) {
        return cudnn_max_pool2d_forward(input, kernelH, kernelW, strideH, strideW, padH, padW);
    }
    PT_ERROR("CPU max_pool2d fallback not implemented in cuDNN dispatch");
}

// AvgPool2d with automatic dispatch
inline Tensor avg_pool2d_dispatch(
    const Tensor& input,
    int64_t kernelH, int64_t kernelW,
    int64_t strideH, int64_t strideW,
    int64_t padH, int64_t padW,
    bool count_include_pad = true
) {
    if (should_use_cudnn(input)) {
        return cudnn_avg_pool2d_forward(
            input, kernelH, kernelW, strideH, strideW, padH, padW, count_include_pad
        );
    }
    PT_ERROR("CPU avg_pool2d fallback not implemented in cuDNN dispatch");
}

// BatchNorm2d with automatic dispatch (training)
inline std::tuple<Tensor, Tensor, Tensor> batch_norm_training_dispatch(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    Tensor& running_mean,
    Tensor& running_var,
    double momentum,
    double epsilon
) {
    if (should_use_cudnn(input, gamma, beta)) {
        return cudnn_batch_norm_forward_training(
            input, gamma, beta, running_mean, running_var, momentum, epsilon
        );
    }
    PT_ERROR("CPU batch_norm_training fallback not implemented in cuDNN dispatch");
}

// BatchNorm2d with automatic dispatch (inference)
inline Tensor batch_norm_inference_dispatch(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    const Tensor& running_mean,
    const Tensor& running_var,
    double epsilon
) {
    if (should_use_cudnn(input, gamma, beta)) {
        return cudnn_batch_norm_forward_inference(
            input, gamma, beta, running_mean, running_var, epsilon
        );
    }
    PT_ERROR("CPU batch_norm_inference fallback not implemented in cuDNN dispatch");
}

// ReLU with automatic dispatch
inline Tensor relu_dispatch(const Tensor& input) {
    if (should_use_cudnn(input)) {
        return cudnn_relu_forward(input);
    }
    // CPU fallback using native implementation
    // Return call to native CPU relu
    PT_ERROR("CPU relu fallback not implemented in cuDNN dispatch");
}

// Softmax with automatic dispatch
inline Tensor softmax_dispatch(const Tensor& input, int64_t dim) {
    if (should_use_cudnn(input)) {
        return cudnn_softmax_forward(input, dim);
    }
    PT_ERROR("CPU softmax fallback not implemented in cuDNN dispatch");
}

} // namespace cudnn
} // namespace at

#endif // PT_USE_CUDA
