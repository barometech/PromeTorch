#pragma once

// ============================================================================
// cuDNN Handle Management
// ============================================================================
// Thread-safe cuDNN handle management with RAII semantics.
// Each thread gets its own cuDNN handle for thread safety.

#ifdef PT_USE_CUDA

#include <cudnn.h>
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <thread>
#include "c10/util/Exception.h"

namespace at {
namespace cudnn {

// ============================================================================
// cuDNN Error Checking
// ============================================================================

#define CUDNN_CHECK(status) \
    do { \
        cudnnStatus_t err = (status); \
        if (err != CUDNN_STATUS_SUCCESS) { \
            PT_ERROR("cuDNN error: ", cudnnGetErrorString(err), \
                     " at ", __FILE__, ":", __LINE__); \
        } \
    } while (0)

// ============================================================================
// cuDNN Handle - Thread-local singleton
// ============================================================================

class CuDNNHandle {
public:
    static cudnnHandle_t get() {
        static thread_local CuDNNHandle instance;
        return instance.handle_;
    }

    ~CuDNNHandle() {
        if (handle_) {
            cudnnDestroy(handle_);
        }
    }

    CuDNNHandle(const CuDNNHandle&) = delete;
    CuDNNHandle& operator=(const CuDNNHandle&) = delete;

private:
    CuDNNHandle() : handle_(nullptr) {
        CUDNN_CHECK(cudnnCreate(&handle_));
    }

    cudnnHandle_t handle_;
};

// ============================================================================
// Tensor Descriptor
// ============================================================================

class TensorDescriptor {
public:
    TensorDescriptor() : desc_(nullptr) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_));
    }

    ~TensorDescriptor() {
        if (desc_) {
            cudnnDestroyTensorDescriptor(desc_);
        }
    }

    // Set 4D tensor descriptor (NCHW format)
    void set(cudnnDataType_t dataType, int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            desc_,
            CUDNN_TENSOR_NCHW,  // format
            dataType,
            n, c, h, w
        ));
    }

    // Set ND tensor descriptor
    void setND(cudnnDataType_t dataType, int nbDims, const int* dims, const int* strides) {
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc_,
            dataType,
            nbDims,
            dims,
            strides
        ));
    }

    cudnnTensorDescriptor_t get() const { return desc_; }
    operator cudnnTensorDescriptor_t() const { return desc_; }

    TensorDescriptor(const TensorDescriptor&) = delete;
    TensorDescriptor& operator=(const TensorDescriptor&) = delete;

    TensorDescriptor(TensorDescriptor&& other) noexcept : desc_(other.desc_) {
        other.desc_ = nullptr;
    }

private:
    cudnnTensorDescriptor_t desc_;
};

// ============================================================================
// Filter Descriptor (for convolution weights)
// ============================================================================

class FilterDescriptor {
public:
    FilterDescriptor() : desc_(nullptr) {
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc_));
    }

    ~FilterDescriptor() {
        if (desc_) {
            cudnnDestroyFilterDescriptor(desc_);
        }
    }

    // Set 4D filter descriptor (out_channels, in_channels, kH, kW)
    void set(cudnnDataType_t dataType, cudnnTensorFormat_t format,
             int k, int c, int h, int w) {
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            desc_,
            dataType,
            format,
            k, c, h, w
        ));
    }

    cudnnFilterDescriptor_t get() const { return desc_; }
    operator cudnnFilterDescriptor_t() const { return desc_; }

    FilterDescriptor(const FilterDescriptor&) = delete;
    FilterDescriptor& operator=(const FilterDescriptor&) = delete;

private:
    cudnnFilterDescriptor_t desc_;
};

// ============================================================================
// Convolution Descriptor
// ============================================================================

class ConvolutionDescriptor {
public:
    ConvolutionDescriptor() : desc_(nullptr) {
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_));
    }

    ~ConvolutionDescriptor() {
        if (desc_) {
            cudnnDestroyConvolutionDescriptor(desc_);
        }
    }

    // Set 2D convolution descriptor
    void set(int padH, int padW, int strideH, int strideW,
             int dilationH, int dilationW, cudnnConvolutionMode_t mode,
             cudnnDataType_t computeType) {
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            desc_,
            padH, padW,
            strideH, strideW,
            dilationH, dilationW,
            mode,
            computeType
        ));
    }

    // Set group count for grouped convolutions
    void setGroupCount(int groupCount) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc_, groupCount));
    }

    // Set math type (for tensor cores)
    void setMathType(cudnnMathType_t mathType) {
        CUDNN_CHECK(cudnnSetConvolutionMathType(desc_, mathType));
    }

    cudnnConvolutionDescriptor_t get() const { return desc_; }
    operator cudnnConvolutionDescriptor_t() const { return desc_; }

    ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
    ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;

private:
    cudnnConvolutionDescriptor_t desc_;
};

// ============================================================================
// Pooling Descriptor
// ============================================================================

class PoolingDescriptor {
public:
    PoolingDescriptor() : desc_(nullptr) {
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc_));
    }

    ~PoolingDescriptor() {
        if (desc_) {
            cudnnDestroyPoolingDescriptor(desc_);
        }
    }

    // Set 2D pooling descriptor
    void set(cudnnPoolingMode_t mode, cudnnNanPropagation_t nanOpt,
             int windowH, int windowW, int padH, int padW,
             int strideH, int strideW) {
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(
            desc_,
            mode,
            nanOpt,
            windowH, windowW,
            padH, padW,
            strideH, strideW
        ));
    }

    cudnnPoolingDescriptor_t get() const { return desc_; }
    operator cudnnPoolingDescriptor_t() const { return desc_; }

    PoolingDescriptor(const PoolingDescriptor&) = delete;
    PoolingDescriptor& operator=(const PoolingDescriptor&) = delete;

private:
    cudnnPoolingDescriptor_t desc_;
};

// ============================================================================
// Activation Descriptor
// ============================================================================

class ActivationDescriptor {
public:
    ActivationDescriptor() : desc_(nullptr) {
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc_));
    }

    ~ActivationDescriptor() {
        if (desc_) {
            cudnnDestroyActivationDescriptor(desc_);
        }
    }

    void set(cudnnActivationMode_t mode, cudnnNanPropagation_t nanOpt, double coef) {
        CUDNN_CHECK(cudnnSetActivationDescriptor(desc_, mode, nanOpt, coef));
    }

    cudnnActivationDescriptor_t get() const { return desc_; }
    operator cudnnActivationDescriptor_t() const { return desc_; }

    ActivationDescriptor(const ActivationDescriptor&) = delete;
    ActivationDescriptor& operator=(const ActivationDescriptor&) = delete;

private:
    cudnnActivationDescriptor_t desc_;
};

// ============================================================================
// Batch Normalization Descriptor (uses TensorDescriptor)
// ============================================================================

// For batch normalization, we use the derived tensor descriptor
// cudnnDeriveBNTensorDescriptor to get the proper shape for scale/bias/mean/var

// ============================================================================
// Helper: Get cudnnDataType from c10::ScalarType
// ============================================================================

inline cudnnDataType_t getCudnnDataType(c10::ScalarType dtype) {
    switch (dtype) {
        case c10::ScalarType::Float:
            return CUDNN_DATA_FLOAT;
        case c10::ScalarType::Double:
            return CUDNN_DATA_DOUBLE;
        case c10::ScalarType::Half:
            return CUDNN_DATA_HALF;
        case c10::ScalarType::BFloat16:
            return CUDNN_DATA_BFLOAT16;
        default:
            PT_ERROR("Unsupported dtype for cuDNN: ", c10::toString(dtype));
    }
}

} // namespace cudnn
} // namespace at

#endif // PT_USE_CUDA
