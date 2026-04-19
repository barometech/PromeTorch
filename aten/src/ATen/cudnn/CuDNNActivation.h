#pragma once

// ============================================================================
// cuDNN Activation Operations
// ============================================================================
// High-performance activation functions using cuDNN.

#ifdef PT_USE_CUDA

#include "aten/src/ATen/cudnn/CuDNNHandle.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

namespace at {
namespace cudnn {

// ============================================================================
// Activation Forward
// ============================================================================

inline Tensor cudnn_activation_forward(
    const Tensor& input,
    cudnnActivationMode_t mode,
    double coef = 0.0  // For CLIPPED_RELU or ELU
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = (input.dim() > 2) ? input.size(2) : 1;
    int64_t W = (input.dim() > 3) ? input.size(3) : 1;

    Tensor output = empty(input.sizes(),
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc;
    inputDesc.set(dataType, N, C, H, W);

    ActivationDescriptor actDesc;
    actDesc.set(mode, CUDNN_PROPAGATE_NAN, coef);

    float alpha = 1.0f, beta = 0.0f;

    CUDNN_CHECK(cudnnActivationForward(
        handle,
        actDesc.get(),
        &alpha,
        inputDesc.get(),
        input.data_ptr(),
        &beta,
        inputDesc.get(),
        output.mutable_data_ptr()
    ));

    return output;
}

// ============================================================================
// Activation Backward
// ============================================================================

inline Tensor cudnn_activation_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output,
    cudnnActivationMode_t mode,
    double coef = 0.0
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = (input.dim() > 2) ? input.size(2) : 1;
    int64_t W = (input.dim() > 3) ? input.size(3) : 1;

    Tensor grad_input = empty(input.sizes(),
                              TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor tensorDesc;
    tensorDesc.set(dataType, N, C, H, W);

    ActivationDescriptor actDesc;
    actDesc.set(mode, CUDNN_PROPAGATE_NAN, coef);

    float alpha = 1.0f, beta = 0.0f;

    CUDNN_CHECK(cudnnActivationBackward(
        handle,
        actDesc.get(),
        &alpha,
        tensorDesc.get(),
        output.data_ptr(),
        tensorDesc.get(),
        grad_output.data_ptr(),
        tensorDesc.get(),
        input.data_ptr(),
        &beta,
        tensorDesc.get(),
        grad_input.mutable_data_ptr()
    ));

    return grad_input;
}

// ============================================================================
// Convenience Functions for Specific Activations
// ============================================================================

// ReLU
inline Tensor cudnn_relu_forward(const Tensor& input) {
    return cudnn_activation_forward(input, CUDNN_ACTIVATION_RELU);
}

inline Tensor cudnn_relu_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output
) {
    return cudnn_activation_backward(grad_output, input, output, CUDNN_ACTIVATION_RELU);
}

// Sigmoid
inline Tensor cudnn_sigmoid_forward(const Tensor& input) {
    return cudnn_activation_forward(input, CUDNN_ACTIVATION_SIGMOID);
}

inline Tensor cudnn_sigmoid_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output
) {
    return cudnn_activation_backward(grad_output, input, output, CUDNN_ACTIVATION_SIGMOID);
}

// Tanh
inline Tensor cudnn_tanh_forward(const Tensor& input) {
    return cudnn_activation_forward(input, CUDNN_ACTIVATION_TANH);
}

inline Tensor cudnn_tanh_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output
) {
    return cudnn_activation_backward(grad_output, input, output, CUDNN_ACTIVATION_TANH);
}

// Clipped ReLU (ReLU6)
inline Tensor cudnn_relu6_forward(const Tensor& input) {
    return cudnn_activation_forward(input, CUDNN_ACTIVATION_CLIPPED_RELU, 6.0);
}

inline Tensor cudnn_relu6_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output
) {
    return cudnn_activation_backward(grad_output, input, output, CUDNN_ACTIVATION_CLIPPED_RELU, 6.0);
}

// ELU
inline Tensor cudnn_elu_forward(const Tensor& input, double alpha = 1.0) {
    return cudnn_activation_forward(input, CUDNN_ACTIVATION_ELU, alpha);
}

inline Tensor cudnn_elu_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output,
    double alpha = 1.0
) {
    return cudnn_activation_backward(grad_output, input, output, CUDNN_ACTIVATION_ELU, alpha);
}

// Swish (SiLU) - Note: cuDNN 8.0+ supports this
#if CUDNN_VERSION >= 8000
inline Tensor cudnn_swish_forward(const Tensor& input) {
    return cudnn_activation_forward(input, CUDNN_ACTIVATION_SWISH);
}

inline Tensor cudnn_swish_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output
) {
    return cudnn_activation_backward(grad_output, input, output, CUDNN_ACTIVATION_SWISH);
}
#endif

// ============================================================================
// Softmax Forward
// ============================================================================

inline Tensor cudnn_softmax_forward(
    const Tensor& input,
    int64_t dim,
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE
) {
    // Reshape input so that the softmax dimension is the channel dimension
    // cuDNN softmax operates on the C dimension in NCHW format

    // For now, assume input is 2D [N, C] or 4D [N, C, H, W] and dim=1
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = (input.dim() > 2) ? input.size(2) : 1;
    int64_t W = (input.dim() > 3) ? input.size(3) : 1;

    Tensor output = empty(input.sizes(),
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor tensorDesc;
    tensorDesc.set(dataType, N, C, H, W);

    float alpha = 1.0f, beta = 0.0f;

    CUDNN_CHECK(cudnnSoftmaxForward(
        handle,
        algo,
        CUDNN_SOFTMAX_MODE_CHANNEL,  // Apply softmax over C
        &alpha,
        tensorDesc.get(),
        input.data_ptr(),
        &beta,
        tensorDesc.get(),
        output.mutable_data_ptr()
    ));

    return output;
}

// ============================================================================
// Softmax Backward
// ============================================================================

inline Tensor cudnn_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,  // The softmax output (needed for backward)
    int64_t dim,
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE
) {
    int64_t N = output.size(0);
    int64_t C = output.size(1);
    int64_t H = (output.dim() > 2) ? output.size(2) : 1;
    int64_t W = (output.dim() > 3) ? output.size(3) : 1;

    Tensor grad_input = empty(output.sizes(),
                              TensorOptions().dtype(output.dtype()).device(output.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(output.dtype());

    TensorDescriptor tensorDesc;
    tensorDesc.set(dataType, N, C, H, W);

    float alpha = 1.0f, beta = 0.0f;

    CUDNN_CHECK(cudnnSoftmaxBackward(
        handle,
        algo,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        tensorDesc.get(),
        output.data_ptr(),
        tensorDesc.get(),
        grad_output.data_ptr(),
        &beta,
        tensorDesc.get(),
        grad_input.mutable_data_ptr()
    ));

    return grad_input;
}

// ============================================================================
// Log Softmax Forward
// ============================================================================

inline Tensor cudnn_log_softmax_forward(const Tensor& input, int64_t dim) {
    return cudnn_softmax_forward(input, dim, CUDNN_SOFTMAX_LOG);
}

// ============================================================================
// Log Softmax Backward
// ============================================================================

inline Tensor cudnn_log_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim
) {
    return cudnn_softmax_backward(grad_output, output, dim, CUDNN_SOFTMAX_LOG);
}

} // namespace cudnn
} // namespace at

#endif // PT_USE_CUDA
