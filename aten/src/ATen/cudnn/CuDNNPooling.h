#pragma once

// ============================================================================
// cuDNN Pooling Operations
// ============================================================================
// High-performance pooling using cuDNN.

#ifdef PT_USE_CUDA

#include "aten/src/ATen/cudnn/CuDNNHandle.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <vector>

namespace at {
namespace cudnn {

// ============================================================================
// Max Pooling 2D Forward
// ============================================================================

inline Tensor cudnn_max_pool2d_forward(
    const Tensor& input,      // [N, C, H, W]
    int64_t kernelH, int64_t kernelW,
    int64_t strideH, int64_t strideW,
    int64_t padH, int64_t padW
) {
    // Get dimensions
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    // Calculate output size
    int64_t H_out = (H_in + 2 * padH - kernelH) / strideH + 1;
    int64_t W_out = (W_in + 2 * padW - kernelW) / strideW + 1;

    // Create output tensor
    Tensor output = empty({N, C, H_out, W_out},
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    // Create descriptors
    TensorDescriptor inputDesc, outputDesc;
    PoolingDescriptor poolDesc;

    inputDesc.set(dataType, N, C, H_in, W_in);
    outputDesc.set(dataType, N, C, H_out, W_out);

    poolDesc.set(
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        kernelH, kernelW,
        padH, padW,
        strideH, strideW
    );

    // Perform pooling
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(
        handle,
        poolDesc.get(),
        &alpha,
        inputDesc.get(),
        input.data_ptr(),
        &beta,
        outputDesc.get(),
        output.mutable_data_ptr()
    ));

    return output;
}

// ============================================================================
// Max Pooling 2D Backward
// ============================================================================

inline Tensor cudnn_max_pool2d_backward(
    const Tensor& grad_output,  // [N, C, H_out, W_out]
    const Tensor& input,        // [N, C, H_in, W_in]
    const Tensor& output,       // [N, C, H_out, W_out]
    int64_t kernelH, int64_t kernelW,
    int64_t strideH, int64_t strideW,
    int64_t padH, int64_t padW
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    // Create gradient input tensor
    Tensor grad_input = empty({N, C, H_in, W_in},
                              TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc, outputDesc;
    PoolingDescriptor poolDesc;

    inputDesc.set(dataType, N, C, H_in, W_in);
    outputDesc.set(dataType, N, C, H_out, W_out);

    poolDesc.set(
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        kernelH, kernelW,
        padH, padW,
        strideH, strideW
    );

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingBackward(
        handle,
        poolDesc.get(),
        &alpha,
        outputDesc.get(),
        output.data_ptr(),
        outputDesc.get(),
        grad_output.data_ptr(),
        inputDesc.get(),
        input.data_ptr(),
        &beta,
        inputDesc.get(),
        grad_input.mutable_data_ptr()
    ));

    return grad_input;
}

// ============================================================================
// Average Pooling 2D Forward
// ============================================================================

inline Tensor cudnn_avg_pool2d_forward(
    const Tensor& input,      // [N, C, H, W]
    int64_t kernelH, int64_t kernelW,
    int64_t strideH, int64_t strideW,
    int64_t padH, int64_t padW,
    bool count_include_pad = true
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t H_out = (H_in + 2 * padH - kernelH) / strideH + 1;
    int64_t W_out = (W_in + 2 * padW - kernelW) / strideW + 1;

    Tensor output = empty({N, C, H_out, W_out},
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc, outputDesc;
    PoolingDescriptor poolDesc;

    inputDesc.set(dataType, N, C, H_in, W_in);
    outputDesc.set(dataType, N, C, H_out, W_out);

    cudnnPoolingMode_t mode = count_include_pad
        ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

    poolDesc.set(
        mode,
        CUDNN_PROPAGATE_NAN,
        kernelH, kernelW,
        padH, padW,
        strideH, strideW
    );

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(
        handle,
        poolDesc.get(),
        &alpha,
        inputDesc.get(),
        input.data_ptr(),
        &beta,
        outputDesc.get(),
        output.mutable_data_ptr()
    ));

    return output;
}

// ============================================================================
// Average Pooling 2D Backward
// ============================================================================

inline Tensor cudnn_avg_pool2d_backward(
    const Tensor& grad_output,  // [N, C, H_out, W_out]
    const Tensor& input,        // [N, C, H_in, W_in]
    int64_t kernelH, int64_t kernelW,
    int64_t strideH, int64_t strideW,
    int64_t padH, int64_t padW,
    bool count_include_pad = true
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    // For avg pooling backward, we don't need the actual output values
    // Just create a dummy tensor with the same shape
    Tensor output = empty({N, C, H_out, W_out},
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    Tensor grad_input = empty({N, C, H_in, W_in},
                              TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc, outputDesc;
    PoolingDescriptor poolDesc;

    inputDesc.set(dataType, N, C, H_in, W_in);
    outputDesc.set(dataType, N, C, H_out, W_out);

    cudnnPoolingMode_t mode = count_include_pad
        ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

    poolDesc.set(
        mode,
        CUDNN_PROPAGATE_NAN,
        kernelH, kernelW,
        padH, padW,
        strideH, strideW
    );

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingBackward(
        handle,
        poolDesc.get(),
        &alpha,
        outputDesc.get(),
        output.data_ptr(),     // Not used for avg pooling
        outputDesc.get(),
        grad_output.data_ptr(),
        inputDesc.get(),
        input.data_ptr(),      // Not used for avg pooling
        &beta,
        inputDesc.get(),
        grad_input.mutable_data_ptr()
    ));

    return grad_input;
}

// ============================================================================
// Adaptive Average Pooling 2D Forward
// ============================================================================
// Note: cuDNN doesn't have direct adaptive pooling, so we calculate the
// kernel/stride parameters to achieve the desired output size

inline Tensor cudnn_adaptive_avg_pool2d_forward(
    const Tensor& input,      // [N, C, H, W]
    int64_t output_H,
    int64_t output_W
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    // Calculate kernel size and stride for adaptive pooling
    // Formula: stride = input_size // output_size
    //          kernel = input_size - (output_size - 1) * stride
    int64_t strideH = H_in / output_H;
    int64_t strideW = W_in / output_W;
    int64_t kernelH = H_in - (output_H - 1) * strideH;
    int64_t kernelW = W_in - (output_W - 1) * strideW;

    // Use regular avg pooling with calculated parameters
    return cudnn_avg_pool2d_forward(
        input,
        kernelH, kernelW,
        strideH, strideW,
        0, 0,  // no padding
        true   // count_include_pad
    );
}

// ============================================================================
// Global Average Pooling 2D (output is 1x1)
// ============================================================================

inline Tensor cudnn_global_avg_pool2d_forward(const Tensor& input) {
    return cudnn_adaptive_avg_pool2d_forward(input, 1, 1);
}

} // namespace cudnn
} // namespace at

#endif // PT_USE_CUDA
