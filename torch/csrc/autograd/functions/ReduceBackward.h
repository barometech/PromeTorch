#pragma once

#include "torch/csrc/autograd/node.h"
#include "aten/src/ATen/ATen.h"

namespace torch {
namespace autograd {

using at::Tensor;
using at::Scalar;

// ============================================================================
// Sum Backward Functions
// ============================================================================

// Sum all elements: d/dx[sum(x)] = 1 (expanded to input shape)
struct SumBackward : public Node {
    std::vector<int64_t> input_sizes_;

    explicit SumBackward(c10::IntArrayRef input_sizes)
        : input_sizes_(input_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Expand scalar gradient to input shape
        Tensor result = at::full(input_sizes_, grad.item());
#ifdef PT_USE_CUDA
        if (grad.is_cuda()) {
            result = at::to_cuda(result);
        }
#endif
        return {result};
    }

    std::string name() const override { return "SumBackward"; }
};

// Sum along dimension: gradient is broadcast along reduced dimension
struct SumDimBackward : public Node {
    std::vector<int64_t> input_sizes_;
    int64_t dim_;
    bool keepdim_;

    SumDimBackward(c10::IntArrayRef input_sizes, int64_t dim, bool keepdim)
        : input_sizes_(input_sizes.vec()), dim_(dim), keepdim_(keepdim) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        Tensor result = grad;

        // If keepdim was false, unsqueeze to add the dimension back
        if (!keepdim_) {
            result = result.unsqueeze(dim_);
        }

        // Expand to original shape
        return {result.expand(input_sizes_)};
    }

    std::string name() const override { return "SumDimBackward"; }
};

// ============================================================================
// Mean Backward Functions
// ============================================================================

// Mean all elements: d/dx[mean(x)] = 1/numel
struct MeanBackward : public Node {
    std::vector<int64_t> input_sizes_;
    int64_t numel_;

    explicit MeanBackward(c10::IntArrayRef input_sizes)
        : input_sizes_(input_sizes.vec()) {
        numel_ = 1;
        for (auto s : input_sizes_) {
            numel_ *= s;
        }
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Gradient is 1/numel at each position
        double scale = 1.0 / static_cast<double>(numel_);
        Tensor result = at::full(input_sizes_, Scalar(grad.item().toDouble() * scale));
#ifdef PT_USE_CUDA
        if (grad.is_cuda()) {
            result = at::to_cuda(result);
        }
#endif
        return {result};
    }

    std::string name() const override { return "MeanBackward"; }
};

// Mean along dimension
struct MeanDimBackward : public Node {
    std::vector<int64_t> input_sizes_;
    int64_t dim_;
    bool keepdim_;
    int64_t dim_size_;

    MeanDimBackward(c10::IntArrayRef input_sizes, int64_t dim, bool keepdim)
        : input_sizes_(input_sizes.vec()), dim_(dim), keepdim_(keepdim) {
        // Normalize negative dim
        if (dim_ < 0) {
            dim_ = dim_ + static_cast<int64_t>(input_sizes_.size());
        }
        dim_size_ = input_sizes_[dim_];
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        Tensor result = grad;

        // If keepdim was false, unsqueeze to add the dimension back
        if (!keepdim_) {
            result = result.unsqueeze(dim_);
        }

        // Expand and scale by 1/dim_size
        result = result.expand(input_sizes_);
        return {result.div(Scalar(static_cast<double>(dim_size_)))};
    }

    std::string name() const override { return "MeanDimBackward"; }
};

// ============================================================================
// Max/Min Backward Functions
// ============================================================================

// Max all elements: gradient flows only to the maximum element
struct MaxBackward : public Node {
    Tensor self_;
    Tensor result_;

    MaxBackward(const Tensor& self, const Tensor& result)
        : self_(self), result_(result) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        result_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Create mask where input equals max value
        Tensor max_val = result_;
        Tensor mask = self_.eq(max_val.expand(self_.sizes()));

        // Distribute gradient among max elements
        // If multiple maxes, divide gradient equally
        Tensor count = mask.to(c10::ScalarType::Float).sum();
        double num_maxes = count.item().toDouble();
        if (num_maxes == 0) num_maxes = 1;

        Tensor result = at::zeros(self_.sizes());
        auto grad_result = at::native::where(mask, at::full(self_.sizes(), Scalar(grad.item().toDouble() / num_maxes)), result);

        // Release saved tensors
        self_ = Tensor();
        result_ = Tensor();

        return {grad_result};
    }

    std::string name() const override { return "MaxBackward"; }
};

// Max along dimension
struct MaxDimBackward : public Node {
    Tensor self_;
    Tensor indices_;
    int64_t dim_;
    bool keepdim_;

    MaxDimBackward(const Tensor& self, const Tensor& indices, int64_t dim, bool keepdim)
        : self_(self), indices_(indices), dim_(dim), keepdim_(keepdim) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        indices_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        Tensor grad_expanded = grad;
        Tensor indices_expanded = indices_;

        // If keepdim was false, unsqueeze
        if (!keepdim_) {
            grad_expanded = grad_expanded.unsqueeze(dim_);
            indices_expanded = indices_expanded.unsqueeze(dim_);
        }

        // Scatter gradient to max positions
        Tensor result = at::zeros(self_.sizes());
        auto grad_result = at::native::scatter(result, dim_, indices_expanded, grad_expanded);

        // Release saved tensors
        self_ = Tensor();
        indices_ = Tensor();

        return {grad_result};
    }

    std::string name() const override { return "MaxDimBackward"; }
};

// Min all elements
struct MinBackward : public Node {
    Tensor self_;
    Tensor result_;

    MinBackward(const Tensor& self, const Tensor& result)
        : self_(self), result_(result) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        result_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        Tensor min_val = result_;
        Tensor mask = self_.eq(min_val.expand(self_.sizes()));

        Tensor count = mask.to(c10::ScalarType::Float).sum();
        double num_mins = count.item().toDouble();
        if (num_mins == 0) num_mins = 1;

        Tensor result = at::zeros(self_.sizes());
        auto grad_result = at::native::where(mask, at::full(self_.sizes(), Scalar(grad.item().toDouble() / num_mins)), result);

        // Release saved tensors
        self_ = Tensor();
        result_ = Tensor();

        return {grad_result};
    }

    std::string name() const override { return "MinBackward"; }
};

// Min along dimension
struct MinDimBackward : public Node {
    Tensor self_;
    Tensor indices_;
    int64_t dim_;
    bool keepdim_;

    MinDimBackward(const Tensor& self, const Tensor& indices, int64_t dim, bool keepdim)
        : self_(self), indices_(indices), dim_(dim), keepdim_(keepdim) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        indices_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        Tensor grad_expanded = grad;
        Tensor indices_expanded = indices_;

        if (!keepdim_) {
            grad_expanded = grad_expanded.unsqueeze(dim_);
            indices_expanded = indices_expanded.unsqueeze(dim_);
        }

        Tensor result = at::zeros(self_.sizes());
        auto grad_result = at::native::scatter(result, dim_, indices_expanded, grad_expanded);

        // Release saved tensors
        self_ = Tensor();
        indices_ = Tensor();

        return {grad_result};
    }

    std::string name() const override { return "MinDimBackward"; }
};

// ============================================================================
// Prod Backward
// ============================================================================

// Prod all elements: d/dx_i[prod(x)] = prod(x) / x_i
struct ProdBackward : public Node {
    Tensor self_;
    Tensor result_;

    ProdBackward(const Tensor& self, const Tensor& result)
        : self_(self), result_(result) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        result_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // grad * prod / x_i
        Tensor prod_val = result_.expand(self_.sizes());
        Tensor grad_input = prod_val.div(self_).mul(Scalar(grad.item()));

        // Release saved tensors
        self_ = Tensor();
        result_ = Tensor();

        return {grad_input};
    }

    std::string name() const override { return "ProdBackward"; }
};

// ============================================================================
// Var/Std Backward
// ============================================================================

// Var: d/dx[var(x)] = 2 * (x - mean(x)) / (n - 1) for unbiased
struct VarBackward : public Node {
    Tensor self_;
    bool unbiased_;

    VarBackward(const Tensor& self, bool unbiased)
        : self_(self), unbiased_(unbiased) {}

    void release_saved_tensors() override {
        self_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        int64_t n = self_.numel();
        double denom = unbiased_ ? (n - 1) : n;
        if (denom == 0) denom = 1;

        Tensor mean = self_.mean();
        Tensor diff = self_.sub(mean.expand(self_.sizes()));

        // d/dx = 2 * (x - mean) / denom * grad
        Tensor grad_input = diff.mul(Scalar(2.0 / denom * grad.item().toDouble()));

        // Release saved tensor
        self_ = Tensor();

        return {grad_input};
    }

    std::string name() const override { return "VarBackward"; }
};

// Std: d/dx[std(x)] = d/dx[sqrt(var(x))] = d_var / (2 * std)
struct StdBackward : public Node {
    Tensor self_;
    Tensor result_;
    bool unbiased_;

    StdBackward(const Tensor& self, const Tensor& result, bool unbiased)
        : self_(self), result_(result), unbiased_(unbiased) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        result_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        int64_t n = self_.numel();
        double denom = unbiased_ ? (n - 1) : n;
        if (denom == 0) denom = 1;

        double std_val = result_.item().toDouble();
        if (std_val == 0) std_val = 1e-12;  // Avoid division by zero

        Tensor mean = self_.mean();
        Tensor diff = self_.sub(mean.expand(self_.sizes()));

        // d/dx = (x - mean) / (denom * std) * grad
        Tensor grad_input = diff.mul(Scalar(grad.item().toDouble() / (denom * std_val)));

        // Release saved tensors
        self_ = Tensor();
        result_ = Tensor();

        return {grad_input};
    }

    std::string name() const override { return "StdBackward"; }
};

// ============================================================================
// Norm Backward
// ============================================================================

// L2 Norm: d/dx[||x||_2] = x / ||x||_2
struct NormBackward : public Node {
    Tensor self_;
    Tensor result_;
    double p_;

    NormBackward(const Tensor& self, const Tensor& result, double p)
        : self_(self), result_(result), p_(p) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        result_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        double norm_val = result_.item().toDouble();
        if (norm_val == 0) norm_val = 1e-12;

        Tensor grad_input;
        if (p_ == 2.0) {
            // L2: grad_input = grad * x / ||x||
            grad_input = self_.mul(Scalar(grad.item().toDouble() / norm_val));
        } else if (p_ == 1.0) {
            // L1: grad_input = grad * sign(x)
            grad_input = self_.sign().mul(Scalar(grad.item()));
        } else {
            // General p-norm: grad_input = grad * sign(x) * |x|^(p-1) / ||x||^(p-1)
            Tensor abs_x = self_.abs();
            Tensor sign_x = self_.sign();
            double norm_pow = std::pow(norm_val, p_ - 1);
            if (norm_pow == 0) norm_pow = 1e-12;

            grad_input = sign_x.mul(abs_x.pow(Scalar(p_ - 1)))
                               .mul(Scalar(grad.item().toDouble() / norm_pow));
        }

        // Release saved tensors
        self_ = Tensor();
        result_ = Tensor();

        return {grad_input};
    }

    std::string name() const override { return "NormBackward"; }
};

// ============================================================================
// Cumsum Backward: reverse cumsum
// backward of cumsum(x, dim) = flip(cumsum(flip(grad, dim), dim), dim)
// ============================================================================
struct CumsumBackward : public Node {
    int64_t dim_;
    std::vector<int64_t> input_sizes_;

    CumsumBackward(int64_t dim, c10::IntArrayRef input_sizes)
        : dim_(dim), input_sizes_(input_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // Reverse cumsum: for each position i, sum of grad[i..end]
        // = cumsum of reversed grad, then reverse back
        // Implement directly without flip (flip not yet available)
        Tensor input_grad = at::empty(input_sizes_, at::TensorOptions().dtype(grad.dtype()).device(grad.device()));
        Tensor g = grad.contiguous();

        PT_DISPATCH_FLOATING_TYPES(grad.dtype(), "cumsum_backward", [&] {
            const scalar_t* g_data = g.data_ptr<scalar_t>();
            scalar_t* out = input_grad.mutable_data_ptr<scalar_t>();

            int64_t ndim = grad.dim();
            int64_t actual_dim = dim_ < 0 ? dim_ + ndim : dim_;

            int64_t outer_size = 1;
            for (int64_t i = 0; i < actual_dim; ++i) outer_size *= grad.size(i);
            int64_t dim_size = grad.size(actual_dim);
            int64_t inner_size = 1;
            for (int64_t i = actual_dim + 1; i < ndim; ++i) inner_size *= grad.size(i);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    // Reverse cumulative sum
                    scalar_t running = 0;
                    for (int64_t r = dim_size - 1; r >= 0; --r) {
                        int64_t idx = (outer * dim_size + r) * inner_size + inner;
                        running += g_data[idx];
                        out[idx] = running;
                    }
                }
            }
        });

        return {input_grad};
    }
    std::string name() const override { return "CumsumBackward"; }
};

// ============================================================================
// Cumprod Backward
// d/dx[cumprod(x, dim)] involves cumprod(x) and cumsum of ratios
// ============================================================================
struct CumprodBackward : public Node {
    int64_t dim_;
    Tensor self_;     // saved input
    Tensor result_;   // saved cumprod output

    CumprodBackward(int64_t dim, const Tensor& self, const Tensor& result)
        : dim_(dim), self_(self), result_(result) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        result_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) {
            self_ = Tensor();
            result_ = Tensor();
            return {Tensor()};
        }

        // cumprod backward:
        // grad_input[i] = sum_{j>=i} grad_output[j] * cumprod_output[j] / input[i]
        // = (1/input[i]) * reverse_cumsum(grad * cumprod_output)[i]
        // But need to handle zeros carefully. For simplicity:
        // grad_input = reverse_cumsum(grad * result) / self
        // where division by zero gives 0

        Tensor grad_times_output = grad.mul(result_);

        // Reverse cumsum
        Tensor rev_cumsum = at::empty_like(grad_times_output);
        Tensor gto = grad_times_output.contiguous();

        PT_DISPATCH_FLOATING_TYPES(grad.dtype(), "cumprod_backward", [&] {
            const scalar_t* g_data = gto.data_ptr<scalar_t>();
            scalar_t* out = rev_cumsum.mutable_data_ptr<scalar_t>();

            int64_t ndim = grad.dim();
            int64_t actual_dim = dim_ < 0 ? dim_ + ndim : dim_;

            int64_t outer_size = 1;
            for (int64_t i = 0; i < actual_dim; ++i) outer_size *= grad.size(i);
            int64_t dim_size = grad.size(actual_dim);
            int64_t inner_size = 1;
            for (int64_t i = actual_dim + 1; i < ndim; ++i) inner_size *= grad.size(i);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    scalar_t running = 0;
                    for (int64_t r = dim_size - 1; r >= 0; --r) {
                        int64_t idx = (outer * dim_size + r) * inner_size + inner;
                        running += g_data[idx];
                        out[idx] = running;
                    }
                }
            }
        });

        // Divide by self (with safe division for zeros)
        Tensor self_safe = self_.clone();
        PT_DISPATCH_FLOATING_TYPES(self_safe.dtype(), "cumprod_bwd_safe", [&] {
            scalar_t* data = self_safe.mutable_data_ptr<scalar_t>();
            int64_t n = self_safe.numel();
            for (int64_t i = 0; i < n; ++i) {
                if (data[i] == 0) data[i] = 1;  // Avoid division by zero
            }
        });

        Tensor result = rev_cumsum.div(self_safe);
        self_ = Tensor();
        result_ = Tensor();
        return {result};
    }
    std::string name() const override { return "CumprodBackward"; }
};

// ============================================================================
// Sort Backward: scatter gradient back using indices
// ============================================================================
struct SortBackward : public Node {
    Tensor indices_;
    int64_t dim_;
    std::vector<int64_t> input_sizes_;

    SortBackward(const Tensor& indices, int64_t dim, c10::IntArrayRef input_sizes)
        : indices_(indices), dim_(dim), input_sizes_(input_sizes.vec()) {}

    void release_saved_tensors() override { indices_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) {
            indices_ = Tensor();
            return {Tensor()};
        }

        // Scatter gradient back to original positions
        int64_t ndim = grad.dim();
        int64_t actual_dim = dim_ < 0 ? dim_ + ndim : dim_;

        Tensor result = at::zeros(input_sizes_, at::TensorOptions().dtype(grad.dtype()).device(grad.device()));
        Tensor g = grad.contiguous();
        Tensor idx = indices_.contiguous();

        PT_DISPATCH_FLOATING_TYPES(grad.dtype(), "sort_backward", [&] {
            const scalar_t* g_data = g.data_ptr<scalar_t>();
            const int64_t* idx_data = idx.data_ptr<int64_t>();
            scalar_t* out = result.mutable_data_ptr<scalar_t>();

            int64_t outer_size = 1;
            for (int64_t i = 0; i < actual_dim; ++i) outer_size *= grad.size(i);
            int64_t dim_size = grad.size(actual_dim);
            int64_t inner_size = 1;
            for (int64_t i = actual_dim + 1; i < ndim; ++i) inner_size *= grad.size(i);
            int64_t orig_dim_size = input_sizes_[actual_dim];

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    for (int64_t r = 0; r < dim_size; ++r) {
                        int64_t grad_idx = (outer * dim_size + r) * inner_size + inner;
                        int64_t orig_pos = idx_data[grad_idx];
                        int64_t out_idx = (outer * orig_dim_size + orig_pos) * inner_size + inner;
                        out[out_idx] += g_data[grad_idx];
                    }
                }
            }
        });

        indices_ = Tensor();
        return {result};
    }
    std::string name() const override { return "SortBackward"; }
};

// ============================================================================
// Topk Backward: scatter gradient back using indices
// ============================================================================
struct TopkBackward : public Node {
    Tensor indices_;
    int64_t dim_;
    std::vector<int64_t> input_sizes_;

    TopkBackward(const Tensor& indices, int64_t dim, c10::IntArrayRef input_sizes)
        : indices_(indices), dim_(dim), input_sizes_(input_sizes.vec()) {}

    void release_saved_tensors() override { indices_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) {
            indices_ = Tensor();
            return {Tensor()};
        }

        // Same as sort backward: scatter gradient back
        int64_t ndim = grad.dim();
        int64_t actual_dim = dim_ < 0 ? dim_ + ndim : dim_;

        Tensor result = at::zeros(input_sizes_, at::TensorOptions().dtype(grad.dtype()).device(grad.device()));
        Tensor g = grad.contiguous();
        Tensor idx = indices_.contiguous();

        PT_DISPATCH_FLOATING_TYPES(grad.dtype(), "topk_backward", [&] {
            const scalar_t* g_data = g.data_ptr<scalar_t>();
            const int64_t* idx_data = idx.data_ptr<int64_t>();
            scalar_t* out = result.mutable_data_ptr<scalar_t>();

            int64_t outer_size = 1;
            for (int64_t i = 0; i < actual_dim; ++i) outer_size *= grad.size(i);
            int64_t k = grad.size(actual_dim);
            int64_t inner_size = 1;
            for (int64_t i = actual_dim + 1; i < ndim; ++i) inner_size *= grad.size(i);
            int64_t orig_dim_size = input_sizes_[actual_dim];

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    for (int64_t r = 0; r < k; ++r) {
                        int64_t grad_idx = (outer * k + r) * inner_size + inner;
                        int64_t orig_pos = idx_data[grad_idx];
                        int64_t out_idx = (outer * orig_dim_size + orig_pos) * inner_size + inner;
                        out[out_idx] += g_data[grad_idx];
                    }
                }
            }
        });

        indices_ = Tensor();
        return {result};
    }
    std::string name() const override { return "TopkBackward"; }
};

} // namespace autograd
} // namespace torch
