#pragma once

// ============================================================================
// cuDNN Batch Normalization Operations
// ============================================================================
// High-performance batch normalization using cuDNN.

#ifdef PT_USE_CUDA

#include "aten/src/ATen/cudnn/CuDNNHandle.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <tuple>

namespace at {
namespace cudnn {

// ============================================================================
// Batch Normalization Forward (Training)
// ============================================================================

inline std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_forward_training(
    const Tensor& input,        // [N, C, H, W]
    const Tensor& gamma,        // [C] scale
    const Tensor& beta,         // [C] bias
    Tensor& running_mean,       // [C] running mean (updated)
    Tensor& running_var,        // [C] running variance (updated)
    double momentum,
    double epsilon
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Output tensor
    Tensor output = empty({N, C, H, W},
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    // Save mean and inv_var for backward
    Tensor save_mean = empty({C},
                             TensorOptions().dtype(c10::ScalarType::Float).device(input.device()));
    Tensor save_inv_var = empty({C},
                                TensorOptions().dtype(c10::ScalarType::Float).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    // Descriptors
    TensorDescriptor inputDesc;
    TensorDescriptor bnDesc;

    inputDesc.set(dataType, N, C, H, W);

    // Derive batch norm descriptor from input descriptor
    cudnnTensorDescriptor_t bnDescRaw;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bnDescRaw));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bnDescRaw,
        inputDesc.get(),
        CUDNN_BATCHNORM_SPATIAL
    ));

    float alpha = 1.0f, beta_param = 0.0f;

    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha,
        &beta_param,
        inputDesc.get(),
        input.data_ptr<void>(),
        inputDesc.get(),  // output has same descriptor
        output.mutable_data_ptr<void>(),
        bnDescRaw,
        gamma.data_ptr<void>(),
        beta.data_ptr<void>(),
        momentum,
        running_mean.mutable_data_ptr<void>(),
        running_var.mutable_data_ptr<void>(),
        epsilon,
        save_mean.mutable_data_ptr<void>(),
        save_inv_var.mutable_data_ptr<void>()
    ));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bnDescRaw));

    return std::make_tuple(output, save_mean, save_inv_var);
}

// ============================================================================
// Batch Normalization Forward (Inference)
// ============================================================================

inline Tensor cudnn_batch_norm_forward_inference(
    const Tensor& input,        // [N, C, H, W]
    const Tensor& gamma,        // [C] scale
    const Tensor& beta,         // [C] bias
    const Tensor& running_mean, // [C]
    const Tensor& running_var,  // [C]
    double epsilon
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    Tensor output = empty({N, C, H, W},
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc;
    inputDesc.set(dataType, N, C, H, W);

    cudnnTensorDescriptor_t bnDescRaw;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bnDescRaw));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bnDescRaw,
        inputDesc.get(),
        CUDNN_BATCHNORM_SPATIAL
    ));

    float alpha = 1.0f, beta_param = 0.0f;

    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha,
        &beta_param,
        inputDesc.get(),
        input.data_ptr<void>(),
        inputDesc.get(),
        output.mutable_data_ptr<void>(),
        bnDescRaw,
        gamma.data_ptr<void>(),
        beta.data_ptr<void>(),
        running_mean.data_ptr<void>(),
        running_var.data_ptr<void>(),
        epsilon
    ));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bnDescRaw));

    return output;
}

// ============================================================================
// Batch Normalization Backward
// ============================================================================

inline std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& grad_output,  // [N, C, H, W]
    const Tensor& input,        // [N, C, H, W]
    const Tensor& gamma,        // [C]
    const Tensor& save_mean,    // [C]
    const Tensor& save_inv_var, // [C]
    double epsilon
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Gradients
    Tensor grad_input = empty({N, C, H, W},
                              TensorOptions().dtype(input.dtype()).device(input.device()));
    Tensor grad_gamma = empty({C},
                              TensorOptions().dtype(c10::ScalarType::Float).device(input.device()));
    Tensor grad_beta = empty({C},
                             TensorOptions().dtype(c10::ScalarType::Float).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc;
    inputDesc.set(dataType, N, C, H, W);

    cudnnTensorDescriptor_t bnDescRaw;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bnDescRaw));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bnDescRaw,
        inputDesc.get(),
        CUDNN_BATCHNORM_SPATIAL
    ));

    float alpha_data = 1.0f, beta_data = 0.0f;
    float alpha_param = 1.0f, beta_param = 0.0f;

    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha_data,
        &beta_data,
        &alpha_param,
        &beta_param,
        inputDesc.get(),
        input.data_ptr<void>(),
        inputDesc.get(),
        grad_output.data_ptr<void>(),
        inputDesc.get(),
        grad_input.mutable_data_ptr<void>(),
        bnDescRaw,
        gamma.data_ptr<void>(),
        grad_gamma.mutable_data_ptr<void>(),
        grad_beta.mutable_data_ptr<void>(),
        epsilon,
        save_mean.data_ptr<void>(),
        save_inv_var.data_ptr<void>()
    ));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bnDescRaw));

    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}

// ============================================================================
// Batch Normalization Per-Activation (for BatchNorm1d on [N, C] input)
// ============================================================================

inline Tensor cudnn_batch_norm1d_forward_inference(
    const Tensor& input,        // [N, C] or [N, C, L]
    const Tensor& gamma,        // [C]
    const Tensor& beta,         // [C]
    const Tensor& running_mean, // [C]
    const Tensor& running_var,  // [C]
    double epsilon
) {
    // Reshape to 4D for cuDNN: [N, C] -> [N, C, 1, 1]
    //                         [N, C, L] -> [N, C, L, 1]
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = (input.dim() == 3) ? input.size(2) : 1;
    int64_t W = 1;

    Tensor output = empty({N, C, H, W},
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc;
    inputDesc.set(dataType, N, C, H, W);

    cudnnTensorDescriptor_t bnDescRaw;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bnDescRaw));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bnDescRaw,
        inputDesc.get(),
        CUDNN_BATCHNORM_SPATIAL
    ));

    float alpha = 1.0f, beta_param = 0.0f;

    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha,
        &beta_param,
        inputDesc.get(),
        input.data_ptr<void>(),
        inputDesc.get(),
        output.mutable_data_ptr<void>(),
        bnDescRaw,
        gamma.data_ptr<void>(),
        beta.data_ptr<void>(),
        running_mean.data_ptr<void>(),
        running_var.data_ptr<void>(),
        epsilon
    ));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bnDescRaw));

    // Reshape back
    if (input.dim() == 2) {
        return output.view({N, C});
    } else {
        return output.view({N, C, H});
    }
}

} // namespace cudnn
} // namespace at

#endif // PT_USE_CUDA
