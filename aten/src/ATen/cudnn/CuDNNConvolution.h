#pragma once

// ============================================================================
// cuDNN Convolution Operations
// ============================================================================
// High-performance convolution using cuDNN with automatic algorithm selection.

#ifdef PT_USE_CUDA

#include "aten/src/ATen/cudnn/CuDNNHandle.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <vector>
#include <unordered_map>

namespace at {
namespace cudnn {

// ============================================================================
// Workspace Manager
// ============================================================================
// Manages GPU workspace memory for cuDNN algorithms

class WorkspaceManager {
public:
    static WorkspaceManager& instance() {
        static WorkspaceManager mgr;
        return mgr;
    }

    void* get(size_t size) {
        if (size > workspace_size_) {
            if (workspace_) {
                cudaFree(workspace_);
            }
            cudaMalloc(&workspace_, size);
            workspace_size_ = size;
        }
        return workspace_;
    }

    ~WorkspaceManager() {
        if (workspace_) {
            cudaFree(workspace_);
        }
    }

private:
    WorkspaceManager() : workspace_(nullptr), workspace_size_(0) {}
    void* workspace_;
    size_t workspace_size_;
};

// ============================================================================
// Convolution Forward
// ============================================================================

inline Tensor cudnn_convolution_forward(
    const Tensor& input,      // [N, C_in, H, W]
    const Tensor& weight,     // [C_out, C_in/groups, kH, kW]
    int64_t padH, int64_t padW,
    int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW,
    int64_t groups
) {
    // Get dimensions
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t C_out = weight.size(0);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);

    // Calculate output size
    int64_t H_out = (H_in + 2 * padH - dilationH * (kH - 1) - 1) / strideH + 1;
    int64_t W_out = (W_in + 2 * padW - dilationW * (kW - 1) - 1) / strideW + 1;

    // Create output tensor
    Tensor output = empty({N, C_out, H_out, W_out},
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    // Get cuDNN handle
    cudnnHandle_t handle = CuDNNHandle::get();

    // Get data type
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    // Create descriptors
    TensorDescriptor inputDesc, outputDesc;
    FilterDescriptor filterDesc;
    ConvolutionDescriptor convDesc;

    inputDesc.set(dataType, N, C_in, H_in, W_in);
    outputDesc.set(dataType, N, C_out, H_out, W_out);
    filterDesc.set(dataType, CUDNN_TENSOR_NCHW, C_out, C_in / groups, kH, kW);

    convDesc.set(padH, padW, strideH, strideW, dilationH, dilationW,
                 CUDNN_CROSS_CORRELATION, dataType);
    convDesc.setGroupCount(groups);

    // Use tensor cores if available (for FP16)
    if (dataType == CUDNN_DATA_HALF) {
        convDesc.setMathType(CUDNN_TENSOR_OP_MATH);
    }

    // Find best algorithm
    int requestedAlgoCount = 8;
    int returnedAlgoCount;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perfResults(requestedAlgoCount);

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        inputDesc.get(),
        filterDesc.get(),
        convDesc.get(),
        outputDesc.get(),
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResults.data()
    ));

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    size_t workspaceSize = perfResults[0].memory;

    // Get workspace
    void* workspace = WorkspaceManager::instance().get(workspaceSize);

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        inputDesc.get(),
        input.data_ptr<void>(),
        filterDesc.get(),
        weight.data_ptr<void>(),
        convDesc.get(),
        algo,
        workspace,
        workspaceSize,
        &beta,
        outputDesc.get(),
        output.mutable_data_ptr<void>()
    ));

    return output;
}

// ============================================================================
// Convolution Backward Data (gradient w.r.t. input)
// ============================================================================

inline Tensor cudnn_convolution_backward_data(
    const Tensor& grad_output,  // [N, C_out, H_out, W_out]
    const Tensor& weight,       // [C_out, C_in/groups, kH, kW]
    c10::IntArrayRef input_size, // [N, C_in, H_in, W_in]
    int64_t padH, int64_t padW,
    int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW,
    int64_t groups
) {
    int64_t N = input_size[0];
    int64_t C_in = input_size[1];
    int64_t H_in = input_size[2];
    int64_t W_in = input_size[3];

    int64_t C_out = weight.size(0);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);

    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    // Create gradient input tensor
    Tensor grad_input = empty({N, C_in, H_in, W_in},
                              TensorOptions().dtype(grad_output.dtype()).device(grad_output.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(grad_output.dtype());

    TensorDescriptor gradOutputDesc, gradInputDesc;
    FilterDescriptor filterDesc;
    ConvolutionDescriptor convDesc;

    gradOutputDesc.set(dataType, N, C_out, H_out, W_out);
    gradInputDesc.set(dataType, N, C_in, H_in, W_in);
    filterDesc.set(dataType, CUDNN_TENSOR_NCHW, C_out, C_in / groups, kH, kW);

    convDesc.set(padH, padW, strideH, strideW, dilationH, dilationW,
                 CUDNN_CROSS_CORRELATION, dataType);
    convDesc.setGroupCount(groups);

    if (dataType == CUDNN_DATA_HALF) {
        convDesc.setMathType(CUDNN_TENSOR_OP_MATH);
    }

    // Find best algorithm
    int requestedAlgoCount = 8;
    int returnedAlgoCount;
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perfResults(requestedAlgoCount);

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle,
        filterDesc.get(),
        gradOutputDesc.get(),
        convDesc.get(),
        gradInputDesc.get(),
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResults.data()
    ));

    cudnnConvolutionBwdDataAlgo_t algo = perfResults[0].algo;
    size_t workspaceSize = perfResults[0].memory;

    void* workspace = WorkspaceManager::instance().get(workspaceSize);

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        filterDesc.get(),
        weight.data_ptr<void>(),
        gradOutputDesc.get(),
        grad_output.data_ptr<void>(),
        convDesc.get(),
        algo,
        workspace,
        workspaceSize,
        &beta,
        gradInputDesc.get(),
        grad_input.mutable_data_ptr<void>()
    ));

    return grad_input;
}

// ============================================================================
// Convolution Backward Filter (gradient w.r.t. weights)
// ============================================================================

inline Tensor cudnn_convolution_backward_filter(
    const Tensor& input,        // [N, C_in, H, W]
    const Tensor& grad_output,  // [N, C_out, H_out, W_out]
    c10::IntArrayRef weight_size, // [C_out, C_in/groups, kH, kW]
    int64_t padH, int64_t padW,
    int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW,
    int64_t groups
) {
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t C_out = weight_size[0];
    int64_t kH = weight_size[2];
    int64_t kW = weight_size[3];

    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    // Create gradient weight tensor
    Tensor grad_weight = empty({C_out, C_in / groups, kH, kW},
                               TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc, gradOutputDesc;
    FilterDescriptor gradFilterDesc;
    ConvolutionDescriptor convDesc;

    inputDesc.set(dataType, N, C_in, H_in, W_in);
    gradOutputDesc.set(dataType, N, C_out, H_out, W_out);
    gradFilterDesc.set(dataType, CUDNN_TENSOR_NCHW, C_out, C_in / groups, kH, kW);

    convDesc.set(padH, padW, strideH, strideW, dilationH, dilationW,
                 CUDNN_CROSS_CORRELATION, dataType);
    convDesc.setGroupCount(groups);

    if (dataType == CUDNN_DATA_HALF) {
        convDesc.setMathType(CUDNN_TENSOR_OP_MATH);
    }

    // Find best algorithm
    int requestedAlgoCount = 8;
    int returnedAlgoCount;
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perfResults(requestedAlgoCount);

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle,
        inputDesc.get(),
        gradOutputDesc.get(),
        convDesc.get(),
        gradFilterDesc.get(),
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResults.data()
    ));

    cudnnConvolutionBwdFilterAlgo_t algo = perfResults[0].algo;
    size_t workspaceSize = perfResults[0].memory;

    void* workspace = WorkspaceManager::instance().get(workspaceSize);

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        handle,
        &alpha,
        inputDesc.get(),
        input.data_ptr<void>(),
        gradOutputDesc.get(),
        grad_output.data_ptr<void>(),
        convDesc.get(),
        algo,
        workspace,
        workspaceSize,
        &beta,
        gradFilterDesc.get(),
        grad_weight.mutable_data_ptr<void>()
    ));

    return grad_weight;
}

// ============================================================================
// Convolution with Bias and Activation (fused operation)
// ============================================================================

inline Tensor cudnn_convolution_bias_activation(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t padH, int64_t padW,
    int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW,
    int64_t groups,
    cudnnActivationMode_t activationMode = CUDNN_ACTIVATION_RELU
) {
    // Get dimensions
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t C_out = weight.size(0);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);

    int64_t H_out = (H_in + 2 * padH - dilationH * (kH - 1) - 1) / strideH + 1;
    int64_t W_out = (W_in + 2 * padW - dilationW * (kW - 1) - 1) / strideW + 1;

    Tensor output = empty({N, C_out, H_out, W_out},
                          TensorOptions().dtype(input.dtype()).device(input.device()));

    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dataType = getCudnnDataType(input.dtype());

    TensorDescriptor inputDesc, outputDesc, biasDesc;
    FilterDescriptor filterDesc;
    ConvolutionDescriptor convDesc;
    ActivationDescriptor actDesc;

    inputDesc.set(dataType, N, C_in, H_in, W_in);
    outputDesc.set(dataType, N, C_out, H_out, W_out);
    biasDesc.set(dataType, 1, C_out, 1, 1);  // Bias shape [1, C_out, 1, 1]
    filterDesc.set(dataType, CUDNN_TENSOR_NCHW, C_out, C_in / groups, kH, kW);

    convDesc.set(padH, padW, strideH, strideW, dilationH, dilationW,
                 CUDNN_CROSS_CORRELATION, dataType);
    convDesc.setGroupCount(groups);

    actDesc.set(activationMode, CUDNN_PROPAGATE_NAN, 0.0);

    // Find algorithm
    int requestedAlgoCount = 8;
    int returnedAlgoCount;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perfResults(requestedAlgoCount);

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        inputDesc.get(),
        filterDesc.get(),
        convDesc.get(),
        outputDesc.get(),
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResults.data()
    ));

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    size_t workspaceSize = perfResults[0].memory;
    void* workspace = WorkspaceManager::instance().get(workspaceSize);

    float alpha1 = 1.0f, alpha2 = 1.0f;

    // Fused convolution + bias + activation
    CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
        handle,
        &alpha1,
        inputDesc.get(),
        input.data_ptr<void>(),
        filterDesc.get(),
        weight.data_ptr<void>(),
        convDesc.get(),
        algo,
        workspace,
        workspaceSize,
        &alpha2,
        outputDesc.get(),  // z descriptor (for residual, we use output itself)
        output.data_ptr<void>(),  // z data
        biasDesc.get(),
        bias.data_ptr<void>(),
        actDesc.get(),
        outputDesc.get(),
        output.mutable_data_ptr<void>()
    ));

    return output;
}

} // namespace cudnn
} // namespace at

#endif // PT_USE_CUDA
