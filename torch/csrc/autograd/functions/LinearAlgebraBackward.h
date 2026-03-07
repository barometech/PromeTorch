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

// ============================================================================
// Einsum Backward
// ============================================================================
// Backward of einsum is computed via einsum with rearranged subscripts

struct EinsumBackward : public Node {
    std::string equation_;
    std::vector<Tensor> saved_tensors_;
    std::vector<std::vector<int64_t>> input_shapes_;

    EinsumBackward(const std::string& equation, const std::vector<Tensor>& tensors)
        : equation_(equation), saved_tensors_(tensors) {
        for (const auto& t : tensors) {
            input_shapes_.push_back(t.sizes().vec());
        }
    }

    void release_saved_tensors() override {
        saved_tensors_.clear();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) {
            variable_list result(saved_tensors_.size());
            return result;
        }

        // Parse equation to build backward equations
        size_t arrow = equation_.find("->");
        std::string lhs = equation_.substr(0, arrow);
        std::string rhs = equation_.substr(arrow + 2);

        std::vector<std::string> input_subs;
        std::string current;
        for (char c : lhs) {
            if (c == ',') {
                input_subs.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        input_subs.push_back(current);

        variable_list result;

        // For each input, compute gradient via einsum
        // grad_i = einsum(rhs + "," + other_subs + "->" + input_subs[i], grad, other_tensors)
        for (size_t i = 0; i < saved_tensors_.size(); ++i) {
            if (saved_tensors_.size() == 2) {
                size_t other = 1 - i;
                std::string backward_eq = rhs + "," + input_subs[other] + "->" + input_subs[i];
                result.push_back(at::native::einsum(backward_eq, {grad, saved_tensors_[other]}));
            } else if (saved_tensors_.size() == 1) {
                // Single operand: grad of reduction or permutation
                // For permutation (ij->ji): backward is same permutation
                // For reduction (ij->i): backward is expand
                std::string backward_eq = rhs + "->" + input_subs[i];
                if (rhs.size() < input_subs[i].size()) {
                    // Reduction case: need to unsqueeze and expand
                    Tensor g = grad;
                    // Add missing dims
                    for (size_t d = 0; d < input_subs[i].size(); ++d) {
                        if (rhs.find(input_subs[i][d]) == std::string::npos) {
                            g = g.unsqueeze(d);
                        }
                    }
                    result.push_back(g.expand(input_shapes_[i]));
                } else {
                    result.push_back(at::native::einsum(backward_eq, {grad}));
                }
            }
        }

        saved_tensors_.clear();
        return result;
    }

    std::string name() const override { return "EinsumBackward"; }
};

// ============================================================================
// Inverse Backward
// grad_A = -A^{-T} @ grad @ A^{-T}
// ============================================================================

struct InverseBackward : public Node {
    Tensor result_;  // A^{-1}

    explicit InverseBackward(const Tensor& result) : result_(result) {}

    void release_saved_tensors() override { result_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // grad_A = -A^{-T} @ grad @ A^{-T}
        Tensor inv_t = result_.t();
        Tensor result = inv_t.mm(grad).mm(inv_t).neg();

        result_ = Tensor();
        return {result};
    }

    std::string name() const override { return "InverseBackward"; }
};

// ============================================================================
// Determinant Backward
// grad_A = grad * det(A) * A^{-T}
// ============================================================================

struct DetBackward : public Node {
    Tensor self_;
    Tensor det_val_;

    DetBackward(const Tensor& self, const Tensor& det_val)
        : self_(self), det_val_(det_val) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        det_val_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        double grad_val = grad.item().toDouble();
        double det_val = det_val_.item().toDouble();

        // grad_A = grad * det(A) * A^{-T}
        Tensor inv_t = at::native::inverse(self_).t();
        Tensor result = inv_t.mul(at::Scalar(grad_val * det_val));

        self_ = Tensor();
        det_val_ = Tensor();
        return {result};
    }

    std::string name() const override { return "DetBackward"; }
};

// ============================================================================
// Cholesky Backward
// Uses Phi operator (lower triangular part of symmetric gradient)
// ============================================================================

struct CholeskyBackward : public Node {
    Tensor L_;

    explicit CholeskyBackward(const Tensor& L) : L_(L) {}

    void release_saved_tensors() override { L_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Simplified Cholesky backward using inverse
        // grad_A = L^{-T} @ (L^T @ grad_L) tril @ L^{-1}
        // More precisely: grad_A = 0.5 * (S + S^T) where S = L^{-T} @ Phi(L^T @ grad_L) @ L^{-1}
        Tensor Lt = L_.t();
        Tensor S = Lt.mm(grad);

        // Phi operator: keep lower triangle, halve diagonal
        int64_t n = S.size(0);
        Tensor phi = at::native::tril(S);
        PT_DISPATCH_FLOATING_TYPES(S.dtype(), "cholesky_bwd", [&] {
            scalar_t* d = phi.mutable_data_ptr<scalar_t>();
            for (int64_t i = 0; i < n; ++i) d[i * n + i] *= 0.5;
        });

        Tensor L_inv = at::native::inverse(L_);
        Tensor L_inv_t = L_inv.t();
        Tensor result = L_inv_t.mm(phi).mm(L_inv);

        // Symmetrize
        result = result.add(result.t()).mul(at::Scalar(0.5));

        L_ = Tensor();
        return {result};
    }

    std::string name() const override { return "CholeskyBackward"; }
};

// ============================================================================
// Trace Backward
// grad_A = grad_scalar * I
// ============================================================================

struct TraceBackward : public Node {
    int64_t rows_, cols_;

    TraceBackward(int64_t rows, int64_t cols) : rows_(rows), cols_(cols) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        double g = grad.item().toDouble();
        int64_t n = std::min(rows_, cols_);
        Tensor result = at::zeros({rows_, cols_});
        float* data = result.mutable_data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) {
            data[i * cols_ + i] = static_cast<float>(g);
        }
        return {result};
    }

    std::string name() const override { return "TraceBackward"; }
};

} // namespace autograd
} // namespace torch
