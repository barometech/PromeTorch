#pragma once

#include "torch/csrc/autograd/node.h"
#include "aten/src/ATen/ATen.h"

namespace torch {
namespace autograd {

using at::Tensor;

// ============================================================================
// Matrix Multiplication Backward (mm)
// C = A @ B where A is [M, K] and B is [K, N]
// grad_A = grad_C @ B^T
// grad_B = A^T @ grad_C
// ============================================================================

struct MmBackward : public Node {
    Tensor self_;   // A
    Tensor other_;  // B

    MmBackward(const Tensor& self, const Tensor& other)
        : self_(self), other_(other) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        other_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // grad_C [M, N]
        if (!grad.defined()) return {Tensor(), Tensor()};

        // grad_A = grad_C @ B^T
        Tensor grad_self = grad.mm(other_.t());

        // grad_B = A^T @ grad_C
        Tensor grad_other = self_.t().mm(grad);

        // CRITICAL: Release saved tensors to prevent memory leak!
        self_ = Tensor();
        other_ = Tensor();

        return {grad_self, grad_other};
    }

    std::string name() const override { return "MmBackward"; }
};

// ============================================================================
// Matrix-Vector Multiplication Backward (mv)
// y = A @ x where A is [M, N] and x is [N]
// grad_A = outer(grad_y, x) = grad_y.unsqueeze(1) @ x.unsqueeze(0)
// grad_x = A^T @ grad_y
// ============================================================================

struct MvBackward : public Node {
    Tensor self_;  // A
    Tensor vec_;   // x

    MvBackward(const Tensor& self, const Tensor& vec)
        : self_(self), vec_(vec) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        vec_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // grad_y [M]
        if (!grad.defined()) return {Tensor(), Tensor()};

        // grad_A = outer(grad_y, x) [M, N]
        Tensor grad_self = at::native::outer(grad, vec_);

        // grad_x = A^T @ grad_y [N]
        Tensor grad_vec = self_.t().mv(grad);

        // CRITICAL: Release saved tensors to prevent memory leak!
        self_ = Tensor();
        vec_ = Tensor();

        return {grad_self, grad_vec};
    }

    std::string name() const override { return "MvBackward"; }
};

// ============================================================================
// Batched Matrix Multiplication Backward (bmm)
// C = A @ B where A is [B, M, K] and B is [B, K, N]
// grad_A = grad_C @ B^T (transpose last two dims)
// grad_B = A^T @ grad_C (transpose last two dims)
// ============================================================================

struct BmmBackward : public Node {
    Tensor self_;   // A
    Tensor other_;  // B

    BmmBackward(const Tensor& self, const Tensor& other)
        : self_(self), other_(other) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        other_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // grad_C [B, M, N]
        if (!grad.defined()) return {Tensor(), Tensor()};

        // grad_A = grad_C @ B^T (B transposed along last two dims)
        // B^T shape: [B, N, K]
        Tensor other_t = other_.transpose(1, 2);
        Tensor grad_self = grad.bmm(other_t);

        // grad_B = A^T @ grad_C
        // A^T shape: [B, K, M]
        Tensor self_t = self_.transpose(1, 2);
        Tensor grad_other = self_t.bmm(grad);

        // CRITICAL: Release saved tensors to prevent memory leak!
        self_ = Tensor();
        other_ = Tensor();

        return {grad_self, grad_other};
    }

    std::string name() const override { return "BmmBackward"; }
};

// ============================================================================
// Dot Product Backward
// c = a · b (scalar)
// grad_a = grad_c * b
// grad_b = grad_c * a
// ============================================================================

struct DotBackward : public Node {
    Tensor self_;
    Tensor other_;

    DotBackward(const Tensor& self, const Tensor& other)
        : self_(self), other_(other) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        other_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // scalar
        if (!grad.defined()) return {Tensor(), Tensor()};

        double grad_val = grad.item().toDouble();

        // grad_a = grad_c * b
        Tensor grad_self = other_.mul(at::Scalar(grad_val));

        // grad_b = grad_c * a
        Tensor grad_other = self_.mul(at::Scalar(grad_val));

        // Release saved tensors
        self_ = Tensor();
        other_ = Tensor();

        return {grad_self, grad_other};
    }

    std::string name() const override { return "DotBackward"; }
};

// ============================================================================
// General Matmul Backward
// Handles all tensor dimensions with broadcasting
// ============================================================================

struct MatmulBackward : public Node {
    Tensor self_;
    Tensor other_;
    int64_t self_dim_;
    int64_t other_dim_;

    MatmulBackward(const Tensor& self, const Tensor& other)
        : self_(self), other_(other), self_dim_(self.dim()), other_dim_(other.dim()) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        other_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor(), Tensor()};

        variable_list result;

        // Dispatch based on input dimensions
        // (same logic as forward matmul)

        // Vector @ Vector -> scalar
        if (self_dim_ == 1 && other_dim_ == 1) {
            result = dot_backward(grad);
        }
        // Matrix @ Vector -> vector
        else if (self_dim_ == 2 && other_dim_ == 1) {
            result = mv_backward(grad);
        }
        // Vector @ Matrix -> vector
        else if (self_dim_ == 1 && other_dim_ == 2) {
            result = vm_backward(grad);
        }
        // Matrix @ Matrix -> matrix
        else if (self_dim_ == 2 && other_dim_ == 2) {
            result = mm_backward(grad);
        }
        // Batched cases
        else if (self_dim_ >= 3 && other_dim_ >= 3) {
            result = batched_backward(grad);
        }
        else if (self_dim_ >= 3 && other_dim_ == 2) {
            result = batched_mat_backward(grad);
        }
        else if (self_dim_ == 2 && other_dim_ >= 3) {
            result = mat_batched_backward(grad);
        }
        else {
            result = {Tensor(), Tensor()};
        }

        // Release saved tensors
        self_ = Tensor();
        other_ = Tensor();

        return result;
    }

    std::string name() const override { return "MatmulBackward"; }

private:
    variable_list dot_backward(const Tensor& grad) {
        double grad_val = grad.item().toDouble();
        return {other_.mul(at::Scalar(grad_val)), self_.mul(at::Scalar(grad_val))};
    }

    variable_list mv_backward(const Tensor& grad) {
        // y = A @ x, grad_y shape [M]
        // grad_A = outer(grad_y, x)
        // grad_x = A^T @ grad_y
        return {at::native::outer(grad, other_), self_.t().mv(grad)};
    }

    variable_list vm_backward(const Tensor& grad) {
        // y = x @ A (where x is [N], A is [N, M])
        // Result is [M]
        // grad_x = grad_y @ A^T = A @ grad_y (A^T is [M, N])
        // grad_A = outer(x, grad_y)
        return {other_.mv(grad), at::native::outer(self_, grad)};
    }

    variable_list mm_backward(const Tensor& grad) {
        // C = A @ B
        // grad_A = grad_C @ B^T
        // grad_B = A^T @ grad_C
        return {grad.mm(other_.t()), self_.t().mm(grad)};
    }

    variable_list batched_backward(const Tensor& grad) {
        // For 3D tensors, use bmm backward logic
        if (self_dim_ == 3 && other_dim_ == 3) {
            Tensor other_t = other_.transpose(1, 2);
            Tensor self_t = self_.transpose(1, 2);
            return {grad.bmm(other_t), self_t.bmm(grad)};
        }

        // For higher dimensions, reshape and use bmm
        int64_t batch = 1;
        for (int64_t i = 0; i < self_dim_ - 2; ++i) {
            batch *= self_.size(i);
        }

        int64_t M = self_.size(-2);
        int64_t K = self_.size(-1);
        int64_t N = other_.size(-1);

        Tensor self_3d = self_.reshape({batch, M, K});
        Tensor other_3d = other_.reshape({batch, K, N});
        Tensor grad_3d = grad.reshape({batch, M, N});

        Tensor grad_self_3d = grad_3d.bmm(other_3d.transpose(1, 2));
        Tensor grad_other_3d = self_3d.transpose(1, 2).bmm(grad_3d);

        return {grad_self_3d.reshape(self_.sizes().vec()),
                grad_other_3d.reshape(other_.sizes().vec())};
    }

    variable_list batched_mat_backward(const Tensor& grad) {
        // self is batched [B..., M, K], other is [K, N]
        int64_t batch = 1;
        for (int64_t i = 0; i < self_dim_ - 2; ++i) {
            batch *= self_.size(i);
        }

        int64_t M = self_.size(-2);
        int64_t K = self_.size(-1);
        int64_t N = other_.size(-1);

        Tensor self_3d = self_.reshape({batch, M, K});
        Tensor other_3d = other_.unsqueeze(0).expand({batch, K, N});
        Tensor grad_3d = grad.reshape({batch, M, N});

        Tensor grad_self_3d = grad_3d.bmm(other_3d.transpose(1, 2));
        Tensor grad_other_3d = self_3d.transpose(1, 2).bmm(grad_3d);

        // Sum grad_other over batch dimension
        Tensor grad_other_sum = grad_other_3d.sum(0);

        return {grad_self_3d.reshape(self_.sizes().vec()), grad_other_sum};
    }

    variable_list mat_batched_backward(const Tensor& grad) {
        // self is [M, K], other is batched [B..., K, N]
        int64_t batch = 1;
        for (int64_t i = 0; i < other_dim_ - 2; ++i) {
            batch *= other_.size(i);
        }

        int64_t M = self_.size(0);
        int64_t K = self_.size(1);
        int64_t N = other_.size(-1);

        Tensor self_3d = self_.unsqueeze(0).expand({batch, M, K});
        Tensor other_3d = other_.reshape({batch, K, N});
        Tensor grad_3d = grad.reshape({batch, M, N});

        Tensor grad_self_3d = grad_3d.bmm(other_3d.transpose(1, 2));
        Tensor grad_other_3d = self_3d.transpose(1, 2).bmm(grad_3d);

        // Sum grad_self over batch dimension
        Tensor grad_self_sum = grad_self_3d.sum(0);

        return {grad_self_sum, grad_other_3d.reshape(other_.sizes().vec())};
    }
};

// ============================================================================
// Outer Product Backward
// C = outer(a, b) where a is [M], b is [N], result is [M, N]
// grad_a = grad_C @ b (sum over columns)
// grad_b = grad_C^T @ a = a^T @ grad_C (sum over rows)
// ============================================================================

struct OuterBackward : public Node {
    Tensor self_;
    Tensor other_;

    OuterBackward(const Tensor& self, const Tensor& other)
        : self_(self), other_(other) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        other_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // [M, N]
        if (!grad.defined()) return {Tensor(), Tensor()};

        // grad_a[i] = sum_j(grad_C[i,j] * b[j]) = grad_C @ b
        Tensor grad_self = grad.mv(other_);

        // grad_b[j] = sum_i(grad_C[i,j] * a[i]) = grad_C^T @ a = a^T @ grad_C
        Tensor grad_other = grad.t().mv(self_);

        // Release saved tensors
        self_ = Tensor();
        other_ = Tensor();

        return {grad_self, grad_other};
    }

    std::string name() const override { return "OuterBackward"; }
};

// ============================================================================
// Addmm Backward
// C = beta * input + alpha * A @ B
// grad_input = beta * grad_C
// grad_A = alpha * grad_C @ B^T
// grad_B = alpha * A^T @ grad_C
// ============================================================================

struct AddmmBackward : public Node {
    Tensor self_;   // input
    Tensor mat1_;   // A
    Tensor mat2_;   // B
    at::Scalar beta_;
    at::Scalar alpha_;

    AddmmBackward(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                  at::Scalar beta, at::Scalar alpha)
        : self_(self), mat1_(mat1), mat2_(mat2), beta_(beta), alpha_(alpha) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        mat1_ = Tensor();
        mat2_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // grad_C
        if (!grad.defined()) return {Tensor(), Tensor(), Tensor()};

        // grad_input = beta * grad_C (need to handle broadcasting)
        Tensor grad_self = grad.mul(beta_);
        // Reduce if self was broadcast
        if (self_.sizes() != grad_self.sizes()) {
            // Sum over dimensions that were broadcast
            while (grad_self.dim() > self_.dim()) {
                grad_self = grad_self.sum(0);
            }
            for (int64_t i = 0; i < self_.dim(); ++i) {
                if (self_.size(i) == 1 && grad_self.size(i) != 1) {
                    grad_self = grad_self.sum(i, true);
                }
            }
        }

        // grad_A = alpha * grad_C @ B^T
        Tensor grad_mat1 = grad.mm(mat2_.t()).mul(alpha_);

        // grad_B = alpha * A^T @ grad_C
        Tensor grad_mat2 = mat1_.t().mm(grad).mul(alpha_);

        // Release saved tensors
        self_ = Tensor();
        mat1_ = Tensor();
        mat2_ = Tensor();

        return {grad_self, grad_mat1, grad_mat2};
    }

    std::string name() const override { return "AddmmBackward"; }
};

// ============================================================================
// Transpose Backward
// B = A^T
// grad_A = grad_B^T
// ============================================================================

struct TransposeBackward : public Node {
    int64_t dim0_;
    int64_t dim1_;

    TransposeBackward(int64_t dim0, int64_t dim1)
        : dim0_(dim0), dim1_(dim1) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Transpose back
        return {grad.transpose(dim0_, dim1_)};
    }

    std::string name() const override { return "TransposeBackward"; }
};

} // namespace autograd
} // namespace torch
