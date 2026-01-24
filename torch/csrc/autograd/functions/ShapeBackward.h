#pragma once

#include "torch/csrc/autograd/node.h"
#include "aten/src/ATen/ATen.h"

namespace torch {
namespace autograd {

using at::Tensor;

// ============================================================================
// View/Reshape Backward
// Simply reshape gradient back to original shape
// ============================================================================

struct ViewBackward : public Node {
    std::vector<int64_t> input_sizes_;

    explicit ViewBackward(c10::IntArrayRef input_sizes)
        : input_sizes_(input_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.reshape(input_sizes_)};
    }

    std::string name() const override { return "ViewBackward"; }
};

struct ReshapeBackward : public Node {
    std::vector<int64_t> input_sizes_;

    explicit ReshapeBackward(c10::IntArrayRef input_sizes)
        : input_sizes_(input_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.reshape(input_sizes_)};
    }

    std::string name() const override { return "ReshapeBackward"; }
};

// ============================================================================
// Flatten Backward
// ============================================================================

struct FlattenBackward : public Node {
    std::vector<int64_t> input_sizes_;

    explicit FlattenBackward(c10::IntArrayRef input_sizes)
        : input_sizes_(input_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.reshape(input_sizes_)};
    }

    std::string name() const override { return "FlattenBackward"; }
};

// ============================================================================
// Squeeze Backward
// ============================================================================

struct SqueezeBackward : public Node {
    std::vector<int64_t> input_sizes_;

    explicit SqueezeBackward(c10::IntArrayRef input_sizes)
        : input_sizes_(input_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.reshape(input_sizes_)};
    }

    std::string name() const override { return "SqueezeBackward"; }
};

struct SqueezeDimBackward : public Node {
    int64_t dim_;

    explicit SqueezeDimBackward(int64_t dim) : dim_(dim) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.unsqueeze(dim_)};
    }

    std::string name() const override { return "SqueezeDimBackward"; }
};

// ============================================================================
// Unsqueeze Backward
// ============================================================================

struct UnsqueezeBackward : public Node {
    int64_t dim_;

    explicit UnsqueezeBackward(int64_t dim) : dim_(dim) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.squeeze(dim_)};
    }

    std::string name() const override { return "UnsqueezeBackward"; }
};

// ============================================================================
// Permute Backward
// ============================================================================

struct PermuteBackward : public Node {
    std::vector<int64_t> dims_;

    explicit PermuteBackward(c10::IntArrayRef dims)
        : dims_(dims.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Compute inverse permutation
        std::vector<int64_t> inverse_dims(dims_.size());
        for (size_t i = 0; i < dims_.size(); ++i) {
            inverse_dims[dims_[i]] = static_cast<int64_t>(i);
        }

        return {grad.permute(inverse_dims)};
    }

    std::string name() const override { return "PermuteBackward"; }
};

// ============================================================================
// Expand Backward
// Sum over expanded dimensions
// ============================================================================

struct ExpandBackward : public Node {
    std::vector<int64_t> input_sizes_;

    explicit ExpandBackward(c10::IntArrayRef input_sizes)
        : input_sizes_(input_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        Tensor result = grad;

        // Handle leading dimensions that were added
        while (result.dim() > static_cast<int64_t>(input_sizes_.size())) {
            result = result.sum(0);
        }

        // Handle dimensions that were expanded from 1
        for (int64_t i = 0; i < static_cast<int64_t>(input_sizes_.size()); ++i) {
            if (input_sizes_[i] == 1 && result.size(i) != 1) {
                result = result.sum(i, /*keepdim=*/true);
            }
        }

        return {result};
    }

    std::string name() const override { return "ExpandBackward"; }
};

// ============================================================================
// Repeat Backward
// Sum over repeated sections
// ============================================================================

struct RepeatBackward : public Node {
    std::vector<int64_t> input_sizes_;
    std::vector<int64_t> repeats_;

    RepeatBackward(c10::IntArrayRef input_sizes, c10::IntArrayRef repeats)
        : input_sizes_(input_sizes.vec()), repeats_(repeats.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Reshape gradient to separate original and repeat dimensions
        // Then sum over repeat dimensions
        Tensor result = grad;

        // Add leading dimensions if repeats added dims
        int64_t dim_diff = static_cast<int64_t>(repeats_.size()) - static_cast<int64_t>(input_sizes_.size());
        for (int64_t i = 0; i < dim_diff; ++i) {
            result = result.sum(0);
        }

        // For each dimension, reshape and sum over repeats
        // This is a simplified version
        for (int64_t i = 0; i < static_cast<int64_t>(input_sizes_.size()); ++i) {
            int64_t repeat_idx = i + dim_diff;
            if (repeat_idx < static_cast<int64_t>(repeats_.size()) && repeats_[repeat_idx] > 1) {
                int64_t orig_size = input_sizes_[i];
                int64_t repeat_count = repeats_[repeat_idx];

                // Reshape to [repeat_count, orig_size] and sum
                auto sizes = result.sizes().vec();
                int64_t total_size = sizes[i];

                // Split into [repeat_count, orig_size]
                std::vector<int64_t> new_sizes;
                for (int64_t j = 0; j < i; ++j) {
                    new_sizes.push_back(sizes[j]);
                }
                new_sizes.push_back(repeat_count);
                new_sizes.push_back(orig_size);
                for (size_t j = i + 1; j < sizes.size(); ++j) {
                    new_sizes.push_back(sizes[j]);
                }

                result = result.reshape(new_sizes);
                result = result.sum(i);  // Sum over repeat dimension
            }
        }

        return {result.reshape(input_sizes_)};
    }

    std::string name() const override { return "RepeatBackward"; }
};

// ============================================================================
// Cat Backward
// Split gradient into pieces for each input
// ============================================================================

struct CatBackward : public Node {
    std::vector<int64_t> split_sizes_;
    int64_t dim_;

    CatBackward(const std::vector<int64_t>& split_sizes, int64_t dim)
        : split_sizes_(split_sizes), dim_(dim) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) {
            variable_list result(split_sizes_.size());
            return result;
        }

        // Split gradient along cat dimension
        variable_list result;
        int64_t start = 0;
        for (int64_t size : split_sizes_) {
            result.push_back(grad.narrow(dim_, start, size));
            start += size;
        }

        return result;
    }

    std::string name() const override { return "CatBackward"; }

    size_t num_outputs() const override {
        return split_sizes_.size();
    }
};

// ============================================================================
// Stack Backward
// Unstack gradient (select along stacked dimension)
// ============================================================================

struct StackBackward : public Node {
    size_t num_inputs_;
    int64_t dim_;

    StackBackward(size_t num_inputs, int64_t dim)
        : num_inputs_(num_inputs), dim_(dim) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) {
            variable_list result(num_inputs_);
            return result;
        }

        // Select along stack dimension for each input
        variable_list result;
        for (size_t i = 0; i < num_inputs_; ++i) {
            result.push_back(grad.select(dim_, static_cast<int64_t>(i)));
        }

        return result;
    }

    std::string name() const override { return "StackBackward"; }

    size_t num_outputs() const override {
        return num_inputs_;
    }
};

// ============================================================================
// Split/Chunk Backward
// Cat gradients back together
// ============================================================================

struct SplitBackward : public Node {
    size_t num_outputs_;
    int64_t dim_;

    SplitBackward(size_t num_outputs, int64_t dim)
        : num_outputs_(num_outputs), dim_(dim) {}

    variable_list apply(variable_list&& grads) override {
        // Cat all gradients together
        std::vector<Tensor> valid_grads;
        for (auto& g : grads) {
            if (g.defined()) {
                valid_grads.push_back(g);
            }
        }

        if (valid_grads.empty()) {
            return {Tensor()};
        }

        return {at::native::cat(valid_grads, dim_)};
    }

    std::string name() const override { return "SplitBackward"; }
};

// ============================================================================
// Select Backward
// Scatter gradient to selected position
// ============================================================================

struct SelectBackward : public Node {
    std::vector<int64_t> input_sizes_;
    int64_t dim_;
    int64_t index_;

    SelectBackward(c10::IntArrayRef input_sizes, int64_t dim, int64_t index)
        : input_sizes_(input_sizes.vec()), dim_(dim), index_(index) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Create zero tensor of original shape
        Tensor result = at::zeros(input_sizes_);

        // Scatter gradient to the selected position
        // Need to unsqueeze grad to match dimensions
        Tensor grad_expanded = grad.unsqueeze(dim_);

        // Use narrow to get a view and copy gradient
        result.narrow(dim_, index_, 1).copy_(grad_expanded);

        return {result};
    }

    std::string name() const override { return "SelectBackward"; }
};

// ============================================================================
// Narrow Backward
// ============================================================================

struct NarrowBackward : public Node {
    std::vector<int64_t> input_sizes_;
    int64_t dim_;
    int64_t start_;

    NarrowBackward(c10::IntArrayRef input_sizes, int64_t dim, int64_t start)
        : input_sizes_(input_sizes.vec()), dim_(dim), start_(start) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Create zero tensor of original shape
        Tensor result = at::zeros(input_sizes_);

        // Copy gradient to the narrowed region
        result.narrow(dim_, start_, grad.size(dim_)).copy_(grad);

        return {result};
    }

    std::string name() const override { return "NarrowBackward"; }
};

// ============================================================================
// Slice Backward
// ============================================================================

struct SliceBackward : public Node {
    std::vector<int64_t> input_sizes_;
    int64_t dim_;
    int64_t start_;
    int64_t end_;
    int64_t step_;

    SliceBackward(c10::IntArrayRef input_sizes, int64_t dim,
                  int64_t start, int64_t end, int64_t step)
        : input_sizes_(input_sizes.vec()), dim_(dim),
          start_(start), end_(end), step_(step) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Create zero tensor of original shape
        Tensor result = at::zeros(input_sizes_);

        if (step_ == 1) {
            // Simple case: contiguous slice
            result.narrow(dim_, start_, end_ - start_).copy_(grad);
        } else {
            // Strided case: need to scatter gradients
            // This is more complex - use index_put or loop
            int64_t n = grad.size(dim_);
            for (int64_t i = 0; i < n; ++i) {
                int64_t src_idx = start_ + i * step_;
                result.select(dim_, src_idx).copy_(grad.select(dim_, i));
            }
        }

        return {result};
    }

    std::string name() const override { return "SliceBackward"; }
};

// ============================================================================
// T (2D transpose) Backward
// ============================================================================

struct TBackward : public Node {
    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.t()};
    }

    std::string name() const override { return "TBackward"; }
};

// ============================================================================
// Contiguous Backward
// Gradient just passes through (contiguous is a memory operation)
// ============================================================================

struct ContiguousBackward : public Node {
    variable_list apply(variable_list&& grads) override {
        return {grads[0]};
    }

    std::string name() const override { return "ContiguousBackward"; }
};

// ============================================================================
// Detach Backward
// Detach stops gradient flow - returns undefined tensor
// ============================================================================

struct DetachBackward : public Node {
    variable_list apply(variable_list&& /*grads*/) override {
        return {Tensor()};  // No gradient flows through detach
    }

    std::string name() const override { return "DetachBackward"; }
};

} // namespace autograd
} // namespace torch
