#pragma once

#include "torch/csrc/autograd/node.h"
#include "aten/src/ATen/ATen.h"

namespace torch {
namespace autograd {

using at::Tensor;

// ============================================================================
// IndexWithTensorBackward
// Forward: result = self[index] along dim
// Backward: grad_self = zeros_like(self); grad_self.scatter_add_(dim, index, grad)
// ============================================================================

struct IndexWithTensorBackward : public Node {
    std::vector<int64_t> self_sizes_;
    Tensor index_;
    int64_t dim_;
    c10::ScalarType dtype_;

    IndexWithTensorBackward(const std::vector<int64_t>& self_sizes,
                            const Tensor& index, int64_t dim, c10::ScalarType dtype)
        : self_sizes_(self_sizes), index_(index), dim_(dim), dtype_(dtype) {}

    void release_saved_tensors() override {
        index_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Create zero tensor with original shape
        Tensor grad_self = at::zeros(self_sizes_, at::TensorOptions().dtype(dtype_));

        // Scatter-add gradient back to original positions
        // Need to reshape index to match grad's shape for scatter_add
        int64_t ndim = static_cast<int64_t>(self_sizes_.size());
        int64_t idx_numel = index_.numel();

        // For dim=0 simple case: grad is [idx_numel, ...], index is [idx_numel]
        // We need index to be [idx_numel, ...] with same value repeated along inner dims
        if (index_.dim() == 1 && ndim >= 1) {
            // Expand index to match grad shape
            std::vector<int64_t> expanded_sizes = grad.sizes().vec();
            Tensor idx_expanded = index_.clone();

            // Reshape index: [idx_numel] -> [idx_numel, 1, 1, ...] -> expand
            std::vector<int64_t> idx_shape(ndim, 1);
            idx_shape[dim_] = idx_numel;
            idx_expanded = idx_expanded.reshape(idx_shape);

            // Expand to match grad shape
            idx_expanded = idx_expanded.expand(expanded_sizes);
            idx_expanded = idx_expanded.contiguous();

            at::native::scatter_add_(grad_self, dim_, idx_expanded, grad.contiguous());
        } else {
            // General case: index already has correct shape
            at::native::scatter_add_(grad_self, dim_, index_, grad.contiguous());
        }

        index_ = Tensor();
        return {grad_self};
    }

    std::string name() const override { return "IndexWithTensorBackward"; }
};

// ============================================================================
// BooleanIndexBackward
// Forward: result = self[mask]
// Backward: grad_self = zeros_like(self); grad_self[mask] = grad
// ============================================================================

struct BooleanIndexBackward : public Node {
    std::vector<int64_t> self_sizes_;
    Tensor mask_;
    c10::ScalarType dtype_;

    BooleanIndexBackward(const std::vector<int64_t>& self_sizes,
                         const Tensor& mask, c10::ScalarType dtype)
        : self_sizes_(self_sizes), mask_(mask), dtype_(dtype) {}

    void release_saved_tensors() override {
        mask_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        Tensor grad_self = at::zeros(self_sizes_, at::TensorOptions().dtype(dtype_));
        at::native::boolean_index_put_(grad_self, mask_, grad);

        mask_ = Tensor();
        return {grad_self};
    }

    std::string name() const override { return "BooleanIndexBackward"; }
};

} // namespace autograd
} // namespace torch
