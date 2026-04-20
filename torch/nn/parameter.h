#pragma once

#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"
#ifdef PT_USE_CUDA
#include <cuda_runtime.h>
#include "c10/cuda/CUDAAllocator.h"  // for CUDA_CHECK macro
#endif
#include <memory>
#include <string>
#include <cstring>  // memset for fast zero_grad

namespace torch {
namespace nn {

using at::Tensor;

// ============================================================================
// Parameter - A Tensor that is a module parameter
// ============================================================================
// Parameters are Tensors that are automatically registered when assigned as
// module attributes. They differ from regular Tensors in that they:
// - Have requires_grad=True by default
// - Are included in module.parameters()
// - Are saved and loaded with the module state

class Parameter {
public:
    // Default constructor - uninitialized parameter
    Parameter() : tensor_(), requires_grad_(true) {}

    // Construct from tensor
    explicit Parameter(Tensor tensor, bool requires_grad = true)
        : tensor_(std::move(tensor)), requires_grad_(requires_grad) {
        if (tensor_.defined() && requires_grad_) {
            // IMPORTANT: Use ensure_autograd_meta_impl to create AutogradMetaImpl
            // directly, not base AutogradMeta. This ensures grad_accumulator_
            // weak_ptr is available from the start and won't be lost during
            // metadata upgrades.
            tensor_.set_requires_grad(true);  // Create base metadata first
            autograd::ensure_autograd_meta_impl(tensor_);  // Upgrade to impl
        }
    }

    // Copy constructor
    Parameter(const Parameter& other)
        : tensor_(other.tensor_), requires_grad_(other.requires_grad_) {}

    // Move constructor
    Parameter(Parameter&& other) noexcept
        : tensor_(std::move(other.tensor_)), requires_grad_(other.requires_grad_) {}

    // Assignment
    Parameter& operator=(const Parameter& other) {
        if (this != &other) {
            tensor_ = other.tensor_;
            requires_grad_ = other.requires_grad_;
        }
        return *this;
    }

    Parameter& operator=(Parameter&& other) noexcept {
        if (this != &other) {
            tensor_ = std::move(other.tensor_);
            requires_grad_ = other.requires_grad_;
        }
        return *this;
    }

    // Set the tensor
    void set_data(Tensor tensor) {
        tensor_ = std::move(tensor);
        if (tensor_.defined() && requires_grad_) {
            tensor_.set_requires_grad(true);
            autograd::ensure_autograd_meta_impl(tensor_);
        }
    }

    // Access the underlying tensor
    Tensor& data() { return tensor_; }
    const Tensor& data() const { return tensor_; }

    // Implicit conversion to Tensor
    operator Tensor&() { return tensor_; }
    operator const Tensor&() const { return tensor_; }

    // Check if defined
    bool defined() const { return tensor_.defined(); }

    // Gradient access
    Tensor grad() const {
        if (!tensor_.defined()) return Tensor();
        // Access grad_ from base c10::AutogradMeta (no need for AutogradMetaImpl)
        auto* raw_meta = tensor_.autograd_meta();
        if (raw_meta && raw_meta->grad_) {
            return Tensor(raw_meta->grad_);
        }
        return Tensor();
    }

    // Set gradient
    void set_grad(const Tensor& grad) {
        if (!tensor_.defined()) return;
        // Use ensure_autograd_meta_impl to get/upgrade metadata
        auto* meta = autograd::ensure_autograd_meta_impl(tensor_);
        meta->grad_ = grad.getIntrusivePtr();
    }

    // Zero gradient
    void zero_grad() {
        if (!tensor_.defined()) return;
        // Access grad_ from base c10::AutogradMeta (no need for AutogradMetaImpl)
        auto* raw_meta = tensor_.autograd_meta();
        if (raw_meta && raw_meta->grad_) {
            // Gradient tensors from backward pass are always contiguous float32.
            auto* impl = raw_meta->grad_.get();
            if (impl && impl->is_contiguous()) {
#ifdef PT_USE_CUDA
                if (impl->is_cuda()) {
                    // Zero via cudaMemset; bare memset would dereference GPU ptr on host.
                    CUDA_CHECK(cudaMemset(impl->data(), 0, impl->nbytes()));
                } else {
                    std::memset(impl->data(), 0, impl->nbytes());
                }
#else
                std::memset(impl->data(), 0, impl->nbytes());
#endif
            } else if (impl) {
                Tensor g(raw_meta->grad_);
                g.zero_();
            }
        }
    }

    // Properties
    bool requires_grad() const { return requires_grad_; }

    void set_requires_grad(bool requires_grad) {
        requires_grad_ = requires_grad;
        if (tensor_.defined()) {
            tensor_.set_requires_grad(requires_grad_);
        }
    }

    // Shape info forwarding
    int64_t dim() const { return tensor_.dim(); }
    int64_t size(int64_t d) const { return tensor_.size(d); }
    c10::IntArrayRef sizes() const { return tensor_.sizes(); }
    int64_t numel() const { return tensor_.numel(); }

private:
    Tensor tensor_;
    bool requires_grad_;
};

// ============================================================================
// Buffer - A Tensor that is NOT a parameter (not trained)
// ============================================================================
// Buffers are tensors that are part of a module's state but should not be
// considered as parameters. Examples include:
// - Running mean/variance in BatchNorm
// - Attention masks
// - Positional encodings

class Buffer {
public:
    Buffer() : tensor_(), persistent_(true) {}

    explicit Buffer(Tensor tensor, bool persistent = true)
        : tensor_(std::move(tensor)), persistent_(persistent) {
        // Buffers typically don't require gradients
        if (tensor_.defined()) {
            tensor_.set_requires_grad(false);
        }
    }

    void set_data(Tensor tensor) {
        tensor_ = std::move(tensor);
        if (tensor_.defined()) {
            tensor_.set_requires_grad(false);
        }
    }

    Tensor& data() { return tensor_; }
    const Tensor& data() const { return tensor_; }

    operator Tensor&() { return tensor_; }
    operator const Tensor&() const { return tensor_; }

    bool defined() const { return tensor_.defined(); }

    // Whether this buffer should be saved in state_dict
    bool persistent() const { return persistent_; }
    void set_persistent(bool persistent) { persistent_ = persistent; }

private:
    Tensor tensor_;
    bool persistent_;
};

} // namespace nn
} // namespace torch
