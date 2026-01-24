#pragma once

#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "aten/src/ATen/ATen.h"
#include <cmath>

namespace torch {
namespace autograd {

using at::Tensor;
using at::Scalar;

// ============================================================================
// Unary Operation Backward Functions
// ============================================================================

// Neg: d/dx[-x] = -1
struct NegBackward : public Node {
    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        return {grad.defined() ? grad.neg() : Tensor()};
    }
    std::string name() const override { return "NegBackward"; }
};

// Abs: d/dx[|x|] = sign(x)
struct AbsBackward : public Node {
    Tensor self_;  // Save input for backward

    explicit AbsBackward(const Tensor& self) : self_(self) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        auto result = grad.mul(self_.sign());
        self_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "AbsBackward"; }
};

// Sqrt: d/dx[sqrt(x)] = 1 / (2 * sqrt(x))
struct SqrtBackward : public Node {
    Tensor result_;  // Save sqrt(x) for backward

    explicit SqrtBackward(const Tensor& result) : result_(result) {}

    void release_saved_tensors() override { result_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // grad / (2 * sqrt(x)) = grad / (2 * result)
        auto result = grad.div(result_.mul(Scalar(2.0)));
        result_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "SqrtBackward"; }
};

// Rsqrt: d/dx[1/sqrt(x)] = -1 / (2 * x^(3/2)) = -0.5 * rsqrt(x)^3
struct RsqrtBackward : public Node {
    Tensor result_;  // Save rsqrt(x) for backward

    explicit RsqrtBackward(const Tensor& result) : result_(result) {}

    void release_saved_tensors() override { result_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // -0.5 * grad * result^3
        auto result_cubed = result_.mul(result_).mul(result_);
        auto result = grad.mul(result_cubed).mul(Scalar(-0.5));
        result_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "RsqrtBackward"; }
};

// Square: d/dx[x^2] = 2x
struct SquareBackward : public Node {
    Tensor self_;

    explicit SquareBackward(const Tensor& self) : self_(self) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        auto result = grad.mul(self_).mul(Scalar(2.0));
        self_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "SquareBackward"; }
};

// Exp: d/dx[exp(x)] = exp(x)
struct ExpBackward : public Node {
    Tensor result_;  // Save exp(x) for backward

    explicit ExpBackward(const Tensor& result) : result_(result) {}

    void release_saved_tensors() override { result_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        auto result = grad.mul(result_);
        result_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "ExpBackward"; }
};

// Log: d/dx[log(x)] = 1/x
struct LogBackward : public Node {
    Tensor self_;

    explicit LogBackward(const Tensor& self) : self_(self) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        auto result = grad.div(self_);
        self_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "LogBackward"; }
};

// Log2: d/dx[log2(x)] = 1/(x * ln(2))
struct Log2Backward : public Node {
    Tensor self_;
    static constexpr double LN2 = 0.6931471805599453;

    explicit Log2Backward(const Tensor& self) : self_(self) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        auto result = grad.div(self_.mul(Scalar(LN2)));
        self_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "Log2Backward"; }
};

// Log10: d/dx[log10(x)] = 1/(x * ln(10))
struct Log10Backward : public Node {
    Tensor self_;
    static constexpr double LN10 = 2.302585092994046;

    explicit Log10Backward(const Tensor& self) : self_(self) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        auto result = grad.div(self_.mul(Scalar(LN10)));
        self_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "Log10Backward"; }
};

// Sin: d/dx[sin(x)] = cos(x)
struct SinBackward : public Node {
    Tensor self_;

    explicit SinBackward(const Tensor& self) : self_(self) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        auto result = grad.mul(self_.cos());
        self_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "SinBackward"; }
};

// Cos: d/dx[cos(x)] = -sin(x)
struct CosBackward : public Node {
    Tensor self_;

    explicit CosBackward(const Tensor& self) : self_(self) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        auto result = grad.mul(self_.sin().neg());
        self_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "CosBackward"; }
};

// Tan: d/dx[tan(x)] = sec^2(x) = 1 + tan^2(x)
struct TanBackward : public Node {
    Tensor result_;  // tan(x)

    explicit TanBackward(const Tensor& result) : result_(result) {}

    void release_saved_tensors() override { result_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // 1 + tan^2(x)
        auto sec_squared = result_.square().add(Scalar(1.0));
        auto result = grad.mul(sec_squared);
        result_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "TanBackward"; }
};

// Tanh: d/dx[tanh(x)] = 1 - tanh^2(x)
struct TanhBackward : public Node {
    Tensor result_;  // tanh(x)

    explicit TanhBackward(const Tensor& result) : result_(result) {}

    void release_saved_tensors() override { result_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // 1 - tanh^2(x) = -(tanh^2(x)) + 1
        // Use tensor ops that auto-dispatch to CUDA
        auto one_minus_tanh_sq = result_.square().neg().add(Scalar(1.0));
        auto result = grad.mul(one_minus_tanh_sq);
        result_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "TanhBackward"; }
};

// Sigmoid: d/dx[sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))
struct SigmoidBackward : public Node {
    Tensor result_;  // sigmoid(x)

    explicit SigmoidBackward(const Tensor& result) : result_(result) {}

    void release_saved_tensors() override { result_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // sigmoid(x) * (1 - sigmoid(x))
        // Use tensor ops: (1 - result_) = -result_ + 1
        auto one_minus = result_.neg().add(Scalar(1.0));
        auto result = grad.mul(result_).mul(one_minus);
        result_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "SigmoidBackward"; }
};

// ReLU: d/dx[relu(x)] = x > 0 ? 1 : 0
struct ReluBackward : public Node {
    Tensor self_;

    explicit ReluBackward(const Tensor& self) : self_(self) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // grad * (self > 0)
        // Create zeros on same device by multiplying grad by 0
        auto mask = self_.gt(Scalar(0.0));
        auto zeros_t = grad.mul(Scalar(0.0));
        auto result = at::native::where(mask, grad, zeros_t);
        self_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "ReluBackward"; }
};

// Reciprocal: d/dx[1/x] = -1/x^2
struct ReciprocalBackward : public Node {
    Tensor result_;  // 1/x

    explicit ReciprocalBackward(const Tensor& result) : result_(result) {}

    void release_saved_tensors() override { result_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // -grad * result^2
        auto result = grad.mul(result_.square()).neg();
        result_ = Tensor();  // Release saved tensor
        return {result};
    }
    std::string name() const override { return "ReciprocalBackward"; }
};

// ============================================================================
// Binary Operation Backward Functions
// ============================================================================

// Add: d/da[a + alpha*b] = 1, d/db[a + alpha*b] = alpha
struct AddBackward : public Node {
    Scalar alpha_;
    std::vector<int64_t> self_sizes_;
    std::vector<int64_t> other_sizes_;

    AddBackward(Scalar alpha, c10::IntArrayRef self_sizes, c10::IntArrayRef other_sizes)
        : alpha_(alpha), self_sizes_(self_sizes.vec()), other_sizes_(other_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor(), Tensor()};

        Tensor grad_self = grad;
        Tensor grad_other = grad.mul(alpha_);

        // Handle broadcasting: sum over broadcasted dimensions
        grad_self = maybe_reduce(grad_self, self_sizes_);
        grad_other = maybe_reduce(grad_other, other_sizes_);

        return {grad_self, grad_other};
    }

    std::string name() const override { return "AddBackward"; }

private:
    // Reduce gradient if input was broadcast
    Tensor maybe_reduce(const Tensor& grad, const std::vector<int64_t>& original_shape) {
        if (grad.sizes().vec() == original_shape) {
            return grad;
        }

        // Sum over dimensions that were broadcast
        Tensor result = grad;
        int64_t grad_dim = grad.dim();
        int64_t orig_dim = static_cast<int64_t>(original_shape.size());

        // Handle leading dimensions that were added
        while (result.dim() > orig_dim) {
            result = result.sum(0);
        }

        // Handle dimensions that were expanded
        for (int64_t i = 0; i < orig_dim; ++i) {
            if (original_shape[i] == 1 && result.size(i) != 1) {
                result = result.sum(i, true);
            }
        }

        return result;
    }
};

// Sub: d/da[a - alpha*b] = 1, d/db[a - alpha*b] = -alpha
struct SubBackward : public Node {
    Scalar alpha_;
    std::vector<int64_t> self_sizes_;
    std::vector<int64_t> other_sizes_;

    SubBackward(Scalar alpha, c10::IntArrayRef self_sizes, c10::IntArrayRef other_sizes)
        : alpha_(alpha), self_sizes_(self_sizes.vec()), other_sizes_(other_sizes.vec()) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor(), Tensor()};

        Tensor grad_self = grad;
        Tensor grad_other = grad.mul(Scalar(-alpha_.toDouble()));

        grad_self = maybe_reduce(grad_self, self_sizes_);
        grad_other = maybe_reduce(grad_other, other_sizes_);

        return {grad_self, grad_other};
    }

    std::string name() const override { return "SubBackward"; }

private:
    Tensor maybe_reduce(const Tensor& grad, const std::vector<int64_t>& original_shape) {
        if (grad.sizes().vec() == original_shape) {
            return grad;
        }
        Tensor result = grad;
        int64_t orig_dim = static_cast<int64_t>(original_shape.size());
        while (result.dim() > orig_dim) {
            result = result.sum(0);
        }
        for (int64_t i = 0; i < orig_dim; ++i) {
            if (original_shape[i] == 1 && result.size(i) != 1) {
                result = result.sum(i, true);
            }
        }
        return result;
    }
};

// Mul: d/da[a * b] = b, d/db[a * b] = a
struct MulBackward : public Node {
    Tensor self_;
    Tensor other_;

    MulBackward(const Tensor& self, const Tensor& other)
        : self_(self), other_(other) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        other_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor(), Tensor()};

        Tensor grad_self = grad.mul(other_);
        Tensor grad_other = grad.mul(self_);

        // Handle broadcasting
        grad_self = maybe_reduce(grad_self, self_.sizes().vec());
        grad_other = maybe_reduce(grad_other, other_.sizes().vec());

        // Release saved tensors
        self_ = Tensor();
        other_ = Tensor();

        return {grad_self, grad_other};
    }

    std::string name() const override { return "MulBackward"; }

private:
    Tensor maybe_reduce(const Tensor& grad, const std::vector<int64_t>& original_shape) {
        if (grad.sizes().vec() == original_shape) {
            return grad;
        }
        Tensor result = grad;
        int64_t orig_dim = static_cast<int64_t>(original_shape.size());
        while (result.dim() > orig_dim) {
            result = result.sum(0);
        }
        for (int64_t i = 0; i < orig_dim; ++i) {
            if (original_shape[i] == 1 && result.size(i) != 1) {
                result = result.sum(i, true);
            }
        }
        return result;
    }
};

// Div: d/da[a / b] = 1/b, d/db[a / b] = -a / b^2
struct DivBackward : public Node {
    Tensor self_;
    Tensor other_;

    DivBackward(const Tensor& self, const Tensor& other)
        : self_(self), other_(other) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        other_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor(), Tensor()};

        Tensor grad_self = grad.div(other_);
        Tensor grad_other = grad.mul(self_).div(other_.square()).neg();

        grad_self = maybe_reduce(grad_self, self_.sizes().vec());
        grad_other = maybe_reduce(grad_other, other_.sizes().vec());

        // Release saved tensors
        self_ = Tensor();
        other_ = Tensor();

        return {grad_self, grad_other};
    }

    std::string name() const override { return "DivBackward"; }

private:
    Tensor maybe_reduce(const Tensor& grad, const std::vector<int64_t>& original_shape) {
        if (grad.sizes().vec() == original_shape) {
            return grad;
        }
        Tensor result = grad;
        int64_t orig_dim = static_cast<int64_t>(original_shape.size());
        while (result.dim() > orig_dim) {
            result = result.sum(0);
        }
        for (int64_t i = 0; i < orig_dim; ++i) {
            if (original_shape[i] == 1 && result.size(i) != 1) {
                result = result.sum(i, true);
            }
        }
        return result;
    }
};

// Pow (tensor ^ tensor): d/da[a^b] = b * a^(b-1), d/db[a^b] = a^b * log(a)
struct PowBackward : public Node {
    Tensor self_;
    Tensor exponent_;
    Tensor result_;

    PowBackward(const Tensor& self, const Tensor& exponent, const Tensor& result)
        : self_(self), exponent_(exponent), result_(result) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        exponent_ = Tensor();
        result_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor(), Tensor()};

        // d/da = b * a^(b-1) = b * result / a
        Tensor grad_self = grad.mul(exponent_).mul(result_).div(self_);

        // d/db = a^b * log(a) = result * log(a)
        Tensor grad_exponent = grad.mul(result_).mul(self_.log());

        grad_self = maybe_reduce(grad_self, self_.sizes().vec());
        grad_exponent = maybe_reduce(grad_exponent, exponent_.sizes().vec());

        // Release saved tensors
        self_ = Tensor();
        exponent_ = Tensor();
        result_ = Tensor();

        return {grad_self, grad_exponent};
    }

    std::string name() const override { return "PowBackward"; }

private:
    Tensor maybe_reduce(const Tensor& grad, const std::vector<int64_t>& original_shape) {
        if (grad.sizes().vec() == original_shape) {
            return grad;
        }
        Tensor result = grad;
        int64_t orig_dim = static_cast<int64_t>(original_shape.size());
        while (result.dim() > orig_dim) {
            result = result.sum(0);
        }
        for (int64_t i = 0; i < orig_dim; ++i) {
            if (original_shape[i] == 1 && result.size(i) != 1) {
                result = result.sum(i, true);
            }
        }
        return result;
    }
};

// Pow (tensor ^ scalar): d/da[a^n] = n * a^(n-1)
struct PowScalarBackward : public Node {
    Tensor self_;
    Scalar exponent_;

    PowScalarBackward(const Tensor& self, Scalar exponent)
        : self_(self), exponent_(exponent) {}

    void release_saved_tensors() override { self_ = Tensor(); }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        double exp_val = exponent_.toDouble();
        // n * a^(n-1)
        Tensor grad_self = grad.mul(Scalar(exp_val)).mul(self_.pow(Scalar(exp_val - 1.0)));

        // Release saved tensor
        self_ = Tensor();

        return {grad_self};
    }

    std::string name() const override { return "PowScalarBackward"; }
};

// ============================================================================
// Scalar Operation Backward Functions
// ============================================================================

// Add scalar: d/da[a + s] = 1
struct AddScalarBackward : public Node {
    variable_list apply(variable_list&& grads) override {
        return {grads[0]};
    }
    std::string name() const override { return "AddScalarBackward"; }
};

// Mul scalar: d/da[a * s] = s
struct MulScalarBackward : public Node {
    Scalar scalar_;

    explicit MulScalarBackward(Scalar scalar) : scalar_(scalar) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.mul(scalar_)};
    }
    std::string name() const override { return "MulScalarBackward"; }
};

// Div scalar: d/da[a / s] = 1/s
struct DivScalarBackward : public Node {
    Scalar scalar_;

    explicit DivScalarBackward(Scalar scalar) : scalar_(scalar) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        return {grad.div(scalar_)};
    }
    std::string name() const override { return "DivScalarBackward"; }
};

// ============================================================================
// Clone/Copy Backward
// ============================================================================

struct CloneBackward : public Node {
    variable_list apply(variable_list&& grads) override {
        return {grads[0]};
    }
    std::string name() const override { return "CloneBackward"; }
};

// ============================================================================
// LogSoftmax Backward
// ============================================================================
// d(log_softmax(x))/dx = I - softmax(x)
// For loss backprop: grad_input = grad_output - softmax * sum(grad_output)

struct LogSoftmaxBackward : public Node {
    Tensor softmax_;  // Cached softmax output
    int64_t dim_;

    LogSoftmaxBackward(const Tensor& softmax, int64_t dim)
        : softmax_(softmax), dim_(dim) {}

    void release_saved_tensors() override {
        softmax_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // grad_input = grad_output - softmax * sum(grad_output, dim, keepdim=true)
        // But sum reduces, so we need to broadcast back
        auto sizes = softmax_.sizes().vec();

        // Compute sum of grad_output along dim
        Tensor sum_grad = grad.sum(dim_, true);  // keepdim

        // grad_input = grad - softmax * sum_grad
        Tensor grad_input = grad.sub(softmax_.mul(sum_grad));

        // Release saved tensor
        softmax_ = Tensor();

        return {grad_input};
    }
    std::string name() const override { return "LogSoftmaxBackward"; }
};

// ============================================================================
// Cross-Entropy Backward (for hard labels)
// ============================================================================
// L = -sum(one_hot(target) * log_softmax(x)) / N
// dL/dx = (softmax(x) - one_hot(target)) / N

struct CrossEntropyBackward : public Node {
    Tensor softmax_;      // Cached softmax output (on CPU)
    Tensor targets_;      // Target class indices (on CPU)
    int64_t ignore_index_;
    int64_t num_classes_;
    int64_t num_valid_;   // Number of valid (non-ignored) samples
    bool output_cuda_;    // Whether output should be on CUDA

    CrossEntropyBackward(
        const Tensor& softmax,
        const Tensor& targets,
        int64_t ignore_index,
        int64_t num_classes,
        int64_t num_valid,
        bool output_cuda = false
    ) : softmax_(softmax), targets_(targets),
        ignore_index_(ignore_index), num_classes_(num_classes),
        num_valid_(num_valid), output_cuda_(output_cuda) {}

    void release_saved_tensors() override {
        softmax_ = Tensor();
        targets_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];

        // Use saved flag for output device (softmax_ is always on CPU for computation)
        bool output_cuda = output_cuda_;

        // Move saved tensors to CPU for computation
        Tensor softmax_cpu = softmax_;
        Tensor targets_cpu = targets_;
#ifdef PT_USE_CUDA
        if (softmax_.is_cuda()) {
            softmax_cpu = at::to_cpu(softmax_);
        }
        if (targets_.is_cuda()) {
            targets_cpu = at::to_cpu(targets_);
        }
#endif

        // Move grad to CPU if needed
        Tensor grad_cpu = grad;
#ifdef PT_USE_CUDA
        if (grad.defined() && grad.is_cuda()) {
            grad_cpu = at::to_cpu(grad);
        }
#endif

        auto sizes = softmax_cpu.sizes().vec();
        int64_t total = softmax_cpu.numel() / num_classes_;

        // grad_input = softmax - one_hot(target)
        Tensor grad_input = softmax_cpu.clone();
        float* grad_data = grad_input.mutable_data_ptr<float>();
        const float* target_data = targets_cpu.data_ptr<float>();

        // Scale by grad_output (usually 1.0 for scalar loss)
        float scale = 1.0f;
        if (grad_cpu.defined() && grad_cpu.numel() == 1) {
            scale = grad_cpu.data_ptr<float>()[0];
        }

        // Subtract 1 from correct class position
        for (int64_t i = 0; i < total; ++i) {
            int64_t target_idx = static_cast<int64_t>(target_data[i]);
            if (target_idx == ignore_index_ || target_idx < 0 || target_idx >= num_classes_) {
                // Zero out gradient for ignored samples
                for (int64_t c = 0; c < num_classes_; ++c) {
                    grad_data[i * num_classes_ + c] = 0.0f;
                }
            } else {
                grad_data[i * num_classes_ + target_idx] -= 1.0f;
            }
        }

        // Scale by 1/N and grad_output
        float final_scale = scale / static_cast<float>(num_valid_ > 0 ? num_valid_ : 1);
        for (int64_t i = 0; i < grad_input.numel(); ++i) {
            grad_data[i] *= final_scale;
        }

        // Move result back to CUDA if saved tensors were on CUDA
#ifdef PT_USE_CUDA
        if (output_cuda) {
            grad_input = at::to_cuda(grad_input);
        }
#endif

        // CRITICAL: Release saved tensors to prevent memory leak!
        softmax_ = Tensor();
        targets_ = Tensor();

        return {grad_input};
    }
    std::string name() const override { return "CrossEntropyBackward"; }
};

// ============================================================================
// SiLU (Swish) Backward
// ============================================================================
// y = x * sigmoid(x)
// dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//       = y/x + sigmoid(x) * (1 - sigmoid(x)) * x  [when x != 0]

struct SiLUBackward : public Node {
    Tensor input_;   // Original input
    Tensor sigmoid_; // Cached sigmoid(input)

    SiLUBackward(const Tensor& input, const Tensor& sigmoid)
        : input_(input), sigmoid_(sigmoid) {}

    void release_saved_tensors() override {
        input_ = Tensor();
        sigmoid_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        bool is_cuda = grad.is_cuda();

        // Move all tensors to CPU for computation
        Tensor grad_cpu = grad;
        Tensor input_cpu = input_;
        Tensor sigmoid_cpu = sigmoid_;
#ifdef PT_USE_CUDA
        if (is_cuda) {
            grad_cpu = at::to_cpu(grad);
            if (input_.is_cuda()) input_cpu = at::to_cpu(input_);
            if (sigmoid_.is_cuda()) sigmoid_cpu = at::to_cpu(sigmoid_);
        }
#endif

        Tensor grad_input = at::empty(input_cpu.sizes());
        const float* grad_data = grad_cpu.data_ptr<float>();
        const float* in_data = input_cpu.data_ptr<float>();
        const float* sig_data = sigmoid_cpu.data_ptr<float>();
        float* result_data = grad_input.mutable_data_ptr<float>();

        int64_t numel = input_cpu.numel();
        for (int64_t i = 0; i < numel; ++i) {
            float sig = sig_data[i];
            float x = in_data[i];
            // dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            result_data[i] = grad_data[i] * sig * (1.0f + x * (1.0f - sig));
        }

#ifdef PT_USE_CUDA
        if (is_cuda) {
            grad_input = at::to_cuda(grad_input);
        }
#endif

        // CRITICAL: Release saved tensors to prevent memory leak!
        input_ = Tensor();
        sigmoid_ = Tensor();

        return {grad_input};
    }
    std::string name() const override { return "SiLUBackward"; }
};

// ============================================================================
// RMSNorm Backward
// ============================================================================
// y = x * rsqrt(mean(x^2) + eps) * weight
// Gradient is complex - we need to track input, weight, and rms

struct RMSNormBackward : public Node {
    Tensor input_;
    Tensor weight_;
    Tensor inv_rms_;  // 1/rms for each position
    int64_t dim_;
    double eps_;

    RMSNormBackward(const Tensor& input, const Tensor& weight,
                    const Tensor& inv_rms, int64_t dim, double eps)
        : input_(input), weight_(weight), inv_rms_(inv_rms), dim_(dim), eps_(eps) {}

    void release_saved_tensors() override {
        input_ = Tensor();
        weight_ = Tensor();
        inv_rms_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];
        if (!grad_output.defined()) return {Tensor()};

        bool is_cuda = grad_output.is_cuda();

        // Move all tensors to CPU for computation
        Tensor grad_out_cpu = grad_output;
        Tensor input_cpu = input_;
        Tensor weight_cpu = weight_;
        Tensor inv_rms_cpu = inv_rms_;
#ifdef PT_USE_CUDA
        if (is_cuda) {
            grad_out_cpu = at::to_cpu(grad_output);
            if (input_.is_cuda()) input_cpu = at::to_cpu(input_);
            if (weight_.is_cuda()) weight_cpu = at::to_cpu(weight_);
            if (inv_rms_.is_cuda()) inv_rms_cpu = at::to_cpu(inv_rms_);
        }
#endif

        auto sizes = input_cpu.sizes().vec();
        int64_t last_dim = sizes.back();
        int64_t outer_size = input_cpu.numel() / last_dim;

        Tensor grad_input = at::empty(sizes);

        const float* grad_out_data = grad_out_cpu.data_ptr<float>();
        const float* in_data = input_cpu.data_ptr<float>();
        const float* weight_data = weight_cpu.data_ptr<float>();
        const float* inv_rms_data = inv_rms_cpu.data_ptr<float>();
        float* grad_in_data = grad_input.mutable_data_ptr<float>();

        // Get or create gradient for weight via autograd metadata
        auto* meta = ensure_autograd_meta_impl(const_cast<Tensor&>(weight_));
        Tensor grad_weight;
        if (meta->grad_) {
            grad_weight = Tensor(meta->grad_);
#ifdef PT_USE_CUDA
            if (grad_weight.is_cuda()) grad_weight = at::to_cpu(grad_weight);
#endif
        } else {
            grad_weight = at::zeros({last_dim});
        }
        float* grad_w_data = grad_weight.mutable_data_ptr<float>();

        for (int64_t i = 0; i < outer_size; ++i) {
            int64_t offset = i * last_dim;
            float inv_rms = inv_rms_data[i];

            // Compute grad_weight contribution
            for (int64_t j = 0; j < last_dim; ++j) {
                grad_w_data[j] += grad_out_data[offset + j] * in_data[offset + j] * inv_rms;
            }

            // Compute terms for grad_input
            // grad_input = inv_rms * weight * grad_output
            //            - inv_rms^3 * x * mean(x * weight * grad_output)
            float sum_xwg = 0.0f;
            for (int64_t j = 0; j < last_dim; ++j) {
                sum_xwg += in_data[offset + j] * weight_data[j] * grad_out_data[offset + j];
            }
            float mean_xwg = sum_xwg / static_cast<float>(last_dim);

            for (int64_t j = 0; j < last_dim; ++j) {
                float term1 = inv_rms * weight_data[j] * grad_out_data[offset + j];
                float term2 = inv_rms * inv_rms * inv_rms * in_data[offset + j] * mean_xwg;
                grad_in_data[offset + j] = term1 - term2;
            }
        }

        // Store weight gradient (move to CUDA if needed)
#ifdef PT_USE_CUDA
        if (is_cuda) {
            grad_weight = at::to_cuda(grad_weight);
            grad_input = at::to_cuda(grad_input);
        }
#endif
        meta->grad_ = grad_weight.getIntrusivePtr();

        // CRITICAL: Release saved tensors to prevent memory leak!
        input_ = Tensor();
        weight_ = Tensor();
        inv_rms_ = Tensor();

        return {grad_input};
    }
    std::string name() const override { return "RMSNormBackward"; }
};

// ============================================================================
// Parallel Scan Backward
// ============================================================================
// Forward: h[t] = gate[t] * h[t-1] + x[t]
// Backward requires reverse scan

struct ParallelScanBackward : public Node {
    Tensor input_;       // x
    Tensor gates_;       // gate values (after modulation)
    Tensor gate_logits_; // Original gate logits
    Tensor base_decay_;
    Tensor hidden_;      // h (scan output)
    bool input_requires_grad_;
    bool gate_logits_requires_grad_;

    ParallelScanBackward(
        const Tensor& input,
        const Tensor& gates,
        const Tensor& gate_logits,
        const Tensor& base_decay,
        const Tensor& hidden,
        bool input_rg,
        bool gate_logits_rg
    ) : input_(input), gates_(gates), gate_logits_(gate_logits),
        base_decay_(base_decay), hidden_(hidden),
        input_requires_grad_(input_rg), gate_logits_requires_grad_(gate_logits_rg) {}

    void release_saved_tensors() override {
        input_ = Tensor();
        gates_ = Tensor();
        gate_logits_ = Tensor();
        base_decay_ = Tensor();
        hidden_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];

        variable_list result;
        if (!grad_output.defined()) {
            if (input_requires_grad_) result.push_back(Tensor());
            if (gate_logits_requires_grad_) result.push_back(Tensor());
            return result;
        }

        bool is_cuda = grad_output.is_cuda();

        // Move all tensors to CPU for computation
        Tensor grad_out_cpu = grad_output;
        Tensor input_cpu = input_;
        Tensor gates_cpu = gates_;
        Tensor gate_logits_cpu = gate_logits_;
        Tensor base_decay_cpu = base_decay_;
        Tensor hidden_cpu = hidden_;

#ifdef PT_USE_CUDA
        if (is_cuda) {
            grad_out_cpu = at::to_cpu(grad_output);
            if (input_.is_cuda()) input_cpu = at::to_cpu(input_);
            if (gates_.is_cuda()) gates_cpu = at::to_cpu(gates_);
            if (gate_logits_.is_cuda()) gate_logits_cpu = at::to_cpu(gate_logits_);
            if (base_decay_.is_cuda()) base_decay_cpu = at::to_cpu(base_decay_);
            if (hidden_.is_cuda()) hidden_cpu = at::to_cpu(hidden_);
        }
#endif

        auto sizes = input_cpu.sizes().vec();
        int64_t B = sizes[0];
        int64_t T = sizes[1];
        int64_t D = sizes[2];

        Tensor grad_x = at::zeros(sizes);
        Tensor grad_gate_logits = at::zeros(sizes);

        float* grad_x_data = grad_x.mutable_data_ptr<float>();
        float* grad_gate_data = grad_gate_logits.mutable_data_ptr<float>();
        const float* grad_out_data = grad_out_cpu.data_ptr<float>();
        const float* gates_data = gates_cpu.data_ptr<float>();
        const float* gate_logits_data = gate_logits_cpu.data_ptr<float>();
        const float* hidden_data = hidden_cpu.data_ptr<float>();
        const float* base_decay_data = base_decay_cpu.data_ptr<float>();

        // Backward scan: accumulate gradients from end to start
        for (int64_t b = 0; b < B; ++b) {
            std::vector<float> grad_h(D, 0.0f);

            for (int64_t t = T - 1; t >= 0; --t) {
                int64_t offset = (b * T + t) * D;

                for (int64_t d = 0; d < D; ++d) {
                    // Add current gradient
                    grad_h[d] += grad_out_data[offset + d];

                    // Gradient w.r.t. x: grad_x[t] = grad_h[t]
                    grad_x_data[offset + d] = grad_h[d];

                    // Gradient w.r.t. gate
                    if (t > 0) {
                        int64_t prev_offset = (b * T + t - 1) * D;
                        float h_prev = hidden_data[prev_offset + d];

                        // gate = base_decay * (1 + tanh(gate_logit) * 0.1)
                        // d_gate/d_gate_logit = base_decay * 0.1 * (1 - tanh^2(gate_logit))
                        float gate_logit = gate_logits_data[offset + d];
                        float tanh_val = std::tanh(gate_logit);
                        float d_gate_d_logit = base_decay_data[d] * 0.1f * (1.0f - tanh_val * tanh_val);

                        grad_gate_data[offset + d] = grad_h[d] * h_prev * d_gate_d_logit;

                        // Propagate gradient: grad_h[t-1] += grad_h[t] * gate[t]
                        grad_h[d] = grad_h[d] * gates_data[offset + d];
                    } else {
                        grad_gate_data[offset + d] = 0.0f;
                    }
                }
            }
        }

        // Move results back to CUDA if needed
#ifdef PT_USE_CUDA
        if (is_cuda) {
            grad_x = at::to_cuda(grad_x);
            grad_gate_logits = at::to_cuda(grad_gate_logits);
        }
#endif

        // Return gradients in the same order edges were added
        if (input_requires_grad_) result.push_back(grad_x);
        if (gate_logits_requires_grad_) result.push_back(grad_gate_logits);

        // CRITICAL: Release saved tensors to prevent memory leak!
        input_ = Tensor();
        gates_ = Tensor();
        gate_logits_ = Tensor();
        base_decay_ = Tensor();
        hidden_ = Tensor();

        return result;
    }
    std::string name() const override { return "ParallelScanBackward"; }
};

// ============================================================================
// Rotary Embedding Backward
// ============================================================================
// RoPE is just rotation, so backward is inverse rotation

struct RotaryEmbeddingBackward : public Node {
    Tensor cos_cache_;
    Tensor sin_cache_;
    int64_t seq_len_;
    int64_t dim_;
    bool batch_first_;

    RotaryEmbeddingBackward(
        const Tensor& cos_cache,
        const Tensor& sin_cache,
        int64_t seq_len,
        int64_t dim,
        bool batch_first
    ) : cos_cache_(cos_cache), sin_cache_(sin_cache),
        seq_len_(seq_len), dim_(dim), batch_first_(batch_first) {}

    void release_saved_tensors() override {
        cos_cache_ = Tensor();
        sin_cache_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};

        // Track if output should be CUDA
        bool output_cuda = grad.is_cuda();

        // Move to CPU for computation
        Tensor grad_cpu = grad;
        Tensor cos_cpu = cos_cache_;
        Tensor sin_cpu = sin_cache_;
#ifdef PT_USE_CUDA
        if (grad.is_cuda()) grad_cpu = at::to_cpu(grad);
        if (cos_cache_.is_cuda()) cos_cpu = at::to_cpu(cos_cache_);
        if (sin_cache_.is_cuda()) sin_cpu = at::to_cpu(sin_cache_);
#endif

        auto sizes = grad_cpu.sizes().vec();
        int64_t batch_size = batch_first_ ? sizes[0] : sizes[1];
        int64_t dim = sizes.back();
        int64_t half_dim = dim / 2;

        Tensor grad_input = at::empty(sizes);
        float* grad_in_data = grad_input.mutable_data_ptr<float>();
        const float* grad_out_data = grad_cpu.data_ptr<float>();
        const float* cos_data = cos_cpu.data_ptr<float>();
        const float* sin_data = sin_cpu.data_ptr<float>();

        // Apply inverse rotation (negate sin)
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len_; ++s) {
                int64_t offset = batch_first_ ?
                    (b * seq_len_ + s) * dim :
                    (s * batch_size + b) * dim;

                for (int64_t i = 0; i < half_dim; ++i) {
                    float g1 = grad_out_data[offset + i];
                    float g2 = grad_out_data[offset + half_dim + i];
                    float cos_val = cos_data[s * dim_ + i];
                    float sin_val = sin_data[s * dim_ + i];

                    // Inverse rotation: transpose rotation matrix
                    grad_in_data[offset + i] = g1 * cos_val + g2 * sin_val;
                    grad_in_data[offset + half_dim + i] = -g1 * sin_val + g2 * cos_val;
                }
            }
        }

        // Move result back to CUDA if needed
#ifdef PT_USE_CUDA
        if (output_cuda) {
            grad_input = at::to_cuda(grad_input);
        }
#endif

        // CRITICAL: Release saved tensors to prevent memory leak!
        cos_cache_ = Tensor();
        sin_cache_ = Tensor();

        return {grad_input};
    }
    std::string name() const override { return "RotaryEmbeddingBackward"; }
};

// ============================================================================
// Element-wise Multiply Backward (for autograd mul)
// ============================================================================

struct MulTensorBackward : public Node {
    Tensor self_;
    Tensor other_;
    bool self_requires_grad_;
    bool other_requires_grad_;

    MulTensorBackward(const Tensor& self, const Tensor& other,
                      bool self_rg, bool other_rg)
        : self_(self), other_(other),
          self_requires_grad_(self_rg), other_requires_grad_(other_rg) {}

    void release_saved_tensors() override {
        self_ = Tensor();
        other_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        variable_list result;

        if (!grad.defined()) {
            // Return empty tensors matching the number of edges
            if (self_requires_grad_) result.push_back(Tensor());
            if (other_requires_grad_) result.push_back(Tensor());
            return result;
        }

        // Return gradients in the same order edges were added
        if (self_requires_grad_) {
            result.push_back(grad.mul(other_));
        }
        if (other_requires_grad_) {
            result.push_back(grad.mul(self_));
        }

        // CRITICAL: Release saved tensors to prevent memory leak!
        self_ = Tensor();
        other_ = Tensor();

        return result;
    }
    std::string name() const override { return "MulTensorBackward"; }
};

// ============================================================================
// Embedding Backward
// ============================================================================
// Forward: output[i] = weight[indices[i]]
// Backward: grad_weight[indices[i]] += grad_output[i] (scatter-add)
//
// Note: Embedding is a sparse operation, so we directly accumulate into
// the weight's gradient rather than returning it through the normal flow.

struct EmbeddingBackward : public Node {
    Tensor indices_;      // The lookup indices
    Tensor weight_;       // Reference to weight tensor for gradient accumulation
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
    bool has_padding_idx_;

    EmbeddingBackward(
        const Tensor& indices,
        const Tensor& weight,
        int64_t num_embeddings,
        int64_t embedding_dim,
        int64_t padding_idx,
        bool has_padding_idx
    ) : indices_(indices), weight_(weight), num_embeddings_(num_embeddings),
        embedding_dim_(embedding_dim), padding_idx_(padding_idx),
        has_padding_idx_(has_padding_idx) {}

    void release_saved_tensors() override {
        indices_ = Tensor();
        weight_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];
        if (!grad_output.defined()) return {};

        // Track if weight is on CUDA
        bool weight_cuda = weight_.is_cuda();

        // Move tensors to CPU for computation
        Tensor grad_out_cpu = grad_output;
        Tensor indices_cpu = indices_;
#ifdef PT_USE_CUDA
        if (grad_output.is_cuda()) grad_out_cpu = at::to_cpu(grad_output);
        if (indices_.is_cuda()) indices_cpu = at::to_cpu(indices_);
#endif

        const float* grad_out_data = grad_out_cpu.data_ptr<float>();
        const float* indices_data = indices_cpu.data_ptr<float>();
        int64_t num_indices = indices_cpu.numel();

        // Get or create gradient for weight via autograd metadata
        auto* meta = ensure_autograd_meta_impl(const_cast<Tensor&>(weight_));
        Tensor grad_weight;
        if (meta->grad_) {
            grad_weight = Tensor(meta->grad_);
#ifdef PT_USE_CUDA
            // Move existing grad to CPU for accumulation
            if (grad_weight.is_cuda()) {
                grad_weight = at::to_cpu(grad_weight);
            }
#endif
        } else {
            grad_weight = at::zeros({num_embeddings_, embedding_dim_});
        }
        float* grad_weight_data = grad_weight.mutable_data_ptr<float>();

        // Scatter-add: grad_weight[idx] += grad_output[i]
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = static_cast<int64_t>(indices_data[i]);

            // Skip padding index
            if (has_padding_idx_ && idx == padding_idx_) {
                continue;
            }

            if (idx >= 0 && idx < num_embeddings_) {
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    grad_weight_data[idx * embedding_dim_ + j] +=
                        grad_out_data[i * embedding_dim_ + j];
                }
            }
        }

        // Move grad_weight back to CUDA if weight is on CUDA
#ifdef PT_USE_CUDA
        if (weight_cuda) {
            grad_weight = at::to_cuda(grad_weight);
        }
#endif

        // Set the gradient back to the weight tensor via metadata
        meta->grad_ = grad_weight.getIntrusivePtr();

        // CRITICAL: Release saved tensors to prevent memory leak!
        indices_ = Tensor();
        weight_ = Tensor();

        // Return empty - we've accumulated directly into weight.grad()
        return {};
    }
    std::string name() const override { return "EmbeddingBackward"; }
};

} // namespace autograd
} // namespace torch
