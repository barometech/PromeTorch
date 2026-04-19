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

// ============================================================================
// WhereBackward
// Forward: out = where(cond, x, y)
// Backward:
//   grad_x = where(cond, grad_out, 0)
//   grad_y = where(cond, 0, grad_out)
//   cond is bool/non-differentiable — no gradient.
// Both inputs participate, so num_outputs == 2.
// ============================================================================

struct WhereBackward : public Node {
    Tensor cond_;
    std::vector<int64_t> x_sizes_;
    std::vector<int64_t> y_sizes_;
    c10::ScalarType dtype_;

    WhereBackward(const Tensor& cond,
                  const std::vector<int64_t>& x_sizes,
                  const std::vector<int64_t>& y_sizes,
                  c10::ScalarType dtype)
        : cond_(cond), x_sizes_(x_sizes), y_sizes_(y_sizes), dtype_(dtype) {}

    void release_saved_tensors() override {
        cond_ = Tensor();
    }

    size_t num_outputs() const override { return 2; }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) { cond_ = Tensor(); return {Tensor(), Tensor()}; }

        auto opts = at::TensorOptions().dtype(dtype_).device(grad.device());
        // Zero template broadcast to grad's shape is handled implicitly by
        // where(): it accepts broadcasting. Create a zero scalar-like tensor
        // with the result shape so where() simply picks branch per element.
        Tensor zero = at::zeros(grad.sizes().vec(), opts);

        Tensor grad_x_full = at::native::where(cond_, grad, zero);
        Tensor grad_y_full = at::native::where(cond_, zero, grad);

        // If broadcasting happened during forward, reduce dims that were
        // expanded so the returned gradient matches each input's shape.
        auto maybe_reduce = [](Tensor g, const std::vector<int64_t>& target) -> Tensor {
            if (g.sizes().vec() == target) return g;
            // Reduce over leading broadcast dims first
            int64_t diff = static_cast<int64_t>(g.dim()) - static_cast<int64_t>(target.size());
            for (int64_t i = 0; i < diff; ++i) {
                g = g.sum(0, /*keepdim=*/false);
            }
            // Reduce over each dim where target is 1 but g has >1
            for (int64_t d = 0; d < static_cast<int64_t>(target.size()); ++d) {
                if (target[d] == 1 && g.size(d) != 1) {
                    g = g.sum(d, /*keepdim=*/true);
                }
            }
            return g;
        };

        Tensor grad_x = maybe_reduce(grad_x_full, x_sizes_);
        Tensor grad_y = maybe_reduce(grad_y_full, y_sizes_);

        cond_ = Tensor();
        return {grad_x, grad_y};
    }

    std::string name() const override { return "WhereBackward"; }
};

// ============================================================================
// MaskedFillBackward
// Forward: out = input.masked_fill(mask, value)   (scalar value)
// Backward:
//   grad_input = where(mask, 0, grad_out)
//   value is a scalar literal here, not a Tensor — nothing to return for it.
// ============================================================================

struct MaskedFillBackward : public Node {
    Tensor mask_;

    explicit MaskedFillBackward(const Tensor& mask) : mask_(mask) {}

    void release_saved_tensors() override { mask_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) { mask_ = Tensor(); return {Tensor()}; }

        // Zero template same shape/dtype/device as grad.
        Tensor zero = at::zeros(grad.sizes().vec(),
            at::TensorOptions().dtype(grad.dtype()).device(grad.device()));
        // where(mask, 0, grad_out): mask==true → 0, mask==false → grad
        Tensor grad_input = at::native::where(mask_, zero, grad);
        mask_ = Tensor();
        return {grad_input};
    }

    std::string name() const override { return "MaskedFillBackward"; }
};

// ============================================================================
// ScatterAddBackward
// Forward: out = self.clone(); out.scatter_add_(dim, idx, src)   (non-inplace form)
// Backward:
//   grad_self = grad_out                          (scatter_add leaves other positions identical)
//   grad_src  = gather(grad_out, dim, idx)
// ============================================================================

struct ScatterAddBackward : public Node {
    Tensor index_;
    int64_t dim_;
    bool has_self_grad_;
    bool has_src_grad_;

    ScatterAddBackward(const Tensor& index, int64_t dim,
                       bool has_self_grad, bool has_src_grad)
        : index_(index), dim_(dim),
          has_self_grad_(has_self_grad), has_src_grad_(has_src_grad) {}

    void release_saved_tensors() override { index_ = Tensor(); }

    size_t num_outputs() const override { return 2; }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) { index_ = Tensor(); return {Tensor(), Tensor()}; }

        Tensor grad_self, grad_src;

        if (has_self_grad_) {
            // Non-scattered positions: identity. Scattered positions: still
            // d out/d self == 1 (scatter_add adds src on top of self without
            // overwriting). Therefore grad_self == grad_out everywhere.
            grad_self = grad;
        }

        if (has_src_grad_) {
            // out[idx] += src  =>  d out/d src = 1 at gathered positions
            grad_src = at::native::gather(grad.contiguous(), dim_, index_);
        }

        index_ = Tensor();
        return {grad_self, grad_src};
    }

    std::string name() const override { return "ScatterAddBackward"; }
};

// ============================================================================
// GatherBackward
// Forward: out = gather(input, dim, idx)
// Backward: grad_input = zeros_like(input).scatter_add_(dim, idx, grad_out)
// ============================================================================

struct GatherBackward : public Node {
    std::vector<int64_t> self_sizes_;
    Tensor index_;
    int64_t dim_;
    c10::ScalarType dtype_;

    GatherBackward(const std::vector<int64_t>& self_sizes,
                   const Tensor& index, int64_t dim, c10::ScalarType dtype)
        : self_sizes_(self_sizes), index_(index), dim_(dim), dtype_(dtype) {}

    void release_saved_tensors() override { index_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) { index_ = Tensor(); return {Tensor()}; }

        Tensor grad_self = at::zeros(self_sizes_,
            at::TensorOptions().dtype(dtype_).device(grad.device()));
        // grad must be contiguous for scatter_add_ sequential src read.
        Tensor grad_c = grad.is_contiguous() ? grad : grad.contiguous();
        at::native::scatter_add_(grad_self, dim_, index_, grad_c);

        index_ = Tensor();
        return {grad_self};
    }

    std::string name() const override { return "GatherBackward"; }
};

} // namespace autograd
} // namespace torch
