#pragma once

#include "torch/nn/module.h"
#include "torch/csrc/autograd/autograd.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include <cmath>

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif

namespace torch {
namespace nn {

// ============================================================================
// ReLU - Rectified Linear Unit
// ============================================================================
// ReLU(x) = max(0, x)

class ReLU : public Module {
public:
    explicit ReLU(bool inplace = false)
        : Module("ReLU"), inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        if (inplace_) {
            // For inplace operations, we need to cast away const
            // This is safe because inplace flag indicates caller expects modification
            return const_cast<Tensor&>(input).relu_();
        }
        // Use autograd-aware relu to maintain gradient flow
        return torch::autograd::relu_autograd(input);
    }

    std::string extra_repr() const override {
        return inplace_ ? "inplace=True" : "";
    }

private:
    bool inplace_;
};

// ============================================================================
// ReLU6 - ReLU clamped at 6
// ============================================================================
// ReLU6(x) = min(max(0, x), 6)

class ReLU6 : public Module {
public:
    explicit ReLU6(bool inplace = false)
        : Module("ReLU6"), inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // ReLU6 = hardtanh(x, 0, 6) — use autograd-aware hardtanh
        return torch::autograd::hardtanh_autograd(input, 0.0, 6.0);
    }

private:
    bool inplace_;
};

// ============================================================================
// LeakyReLU
// ============================================================================
// LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)

class LeakyReLU : public Module {
public:
    explicit LeakyReLU(double negative_slope = 0.01, bool inplace = false)
        : Module("LeakyReLU")
        , negative_slope_(negative_slope)
        , inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware leaky_relu to maintain gradient flow
        return torch::autograd::leaky_relu_autograd(input, negative_slope_);
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "negative_slope=" << negative_slope_;
        if (inplace_) ss << ", inplace=True";
        return ss.str();
    }

private:
    double negative_slope_;
    bool inplace_;
};

// ============================================================================
// PReLU - Parametric ReLU
// ============================================================================
// PReLU(x) = max(0, x) + a * min(0, x)
// where a is a learnable parameter

class PReLU : public Module {
public:
    explicit PReLU(int64_t num_parameters = 1, double init = 0.25)
        : Module("PReLU")
        , num_parameters_(num_parameters)
        , init_(init)
    {
        Tensor weight = at::full({num_parameters}, at::Scalar(init));
        register_parameter("weight", Parameter(weight));
    }

    Tensor forward(const Tensor& input) override {
        // PReLU: x if x >= 0, else weight * x
        // Use leaky_relu_autograd with learned slope for single-parameter case
        if (num_parameters_ == 1) {
            auto* weight = get_parameter("weight");
            float slope = weight->data().data_ptr<float>()[0];
            return torch::autograd::leaky_relu_autograd(input, static_cast<double>(slope));
        }
        // Multi-parameter PReLU: manual implementation (no autograd for now)
        Tensor result = input.clone();
        auto* weight = get_parameter("weight");
        const float* w = weight->data().data_ptr<float>();
        float* data = result.mutable_data_ptr<float>();

        for (int64_t i = 0; i < result.numel(); ++i) {
            if (data[i] < 0) {
                int64_t w_idx = i % num_parameters_;
                data[i] *= w[w_idx];
            }
        }
        return result;
    }

    std::string extra_repr() const override {
        return "num_parameters=" + std::to_string(num_parameters_);
    }

private:
    int64_t num_parameters_;
    double init_;
};

// ============================================================================
// ELU - Exponential Linear Unit
// ============================================================================
// ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0

class ELU : public Module {
public:
    explicit ELU(double alpha = 1.0, bool inplace = false)
        : Module("ELU"), alpha_(alpha), inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware elu to maintain gradient flow
        return torch::autograd::elu_autograd(input, alpha_);
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "alpha=" << alpha_;
        if (inplace_) ss << ", inplace=True";
        return ss.str();
    }

private:
    double alpha_;
    bool inplace_;
};

// ============================================================================
// SELU - Scaled Exponential Linear Unit
// ============================================================================
// SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
// Self-normalizing neural networks

class SELU : public Module {
public:
    explicit SELU(bool inplace = false)
        : Module("SELU"), inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware selu to maintain gradient flow
        return torch::autograd::selu_autograd(input);
    }

private:
    bool inplace_;
};

// ============================================================================
// GELU - Gaussian Error Linear Unit
// ============================================================================
// GELU(x) = x * Phi(x) where Phi is the CDF of standard normal
// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

class GELU : public Module {
public:
    explicit GELU(const std::string& approximate = "none")
        : Module("GELU"), approximate_(approximate) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware gelu to maintain gradient flow
        return torch::autograd::gelu_autograd(input, approximate_ == "tanh");
    }

    std::string extra_repr() const override {
        return "approximate='" + approximate_ + "'";
    }

private:
    std::string approximate_;
};

// ============================================================================
// Sigmoid
// ============================================================================
// Sigmoid(x) = 1 / (1 + exp(-x))

class Sigmoid : public Module {
public:
    Sigmoid() : Module("Sigmoid") {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware sigmoid to maintain gradient flow
        return torch::autograd::sigmoid_autograd(input);
    }
};

// ============================================================================
// Tanh
// ============================================================================

class Tanh : public Module {
public:
    Tanh() : Module("Tanh") {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware tanh to maintain gradient flow
        return torch::autograd::tanh_autograd(input);
    }
};

// ============================================================================
// Softmax
// ============================================================================
// Softmax(x_i) = exp(x_i) / sum(exp(x_j))

class Softmax : public Module {
public:
    explicit Softmax(int64_t dim = -1)
        : Module("Softmax"), dim_(dim) {}

    Tensor forward(const Tensor& input) override {
        int64_t actual_dim = dim_;
        if (actual_dim < 0) {
            actual_dim = input.dim() + actual_dim;
        }

        // Subtract max for numerical stability
        Tensor max_val = std::get<0>(input.max(actual_dim, true));
        Tensor shifted = input.sub(max_val.expand(input.sizes()));
        Tensor exp_shifted = shifted.exp();
        Tensor sum_exp = exp_shifted.sum(actual_dim, true);
        return exp_shifted.div(sum_exp.expand(input.sizes()));
    }

    std::string extra_repr() const override {
        return "dim=" + std::to_string(dim_);
    }

private:
    int64_t dim_;
};

// ============================================================================
// LogSoftmax
// ============================================================================
// LogSoftmax(x_i) = log(exp(x_i) / sum(exp(x_j)))
//                 = x_i - log(sum(exp(x_j)))

class LogSoftmax : public Module {
public:
    explicit LogSoftmax(int64_t dim = -1)
        : Module("LogSoftmax"), dim_(dim) {}

    Tensor forward(const Tensor& input) override {
        int64_t actual_dim = dim_;
        if (actual_dim < 0) {
            actual_dim = input.dim() + actual_dim;
        }

        // For numerical stability: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        Tensor max_val = std::get<0>(input.max(actual_dim, true));
        Tensor shifted = input.sub(max_val.expand(input.sizes()));
        Tensor log_sum_exp = shifted.exp().sum(actual_dim, true).log();
        return shifted.sub(log_sum_exp.expand(input.sizes()));
    }

    std::string extra_repr() const override {
        return "dim=" + std::to_string(dim_);
    }

private:
    int64_t dim_;
};

// ============================================================================
// Softplus
// ============================================================================
// Softplus(x) = (1/beta) * log(1 + exp(beta * x))

class Softplus : public Module {
public:
    explicit Softplus(double beta = 1.0, double threshold = 20.0)
        : Module("Softplus"), beta_(beta), threshold_(threshold) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware softplus to maintain gradient flow
        return torch::autograd::softplus_autograd(input, beta_, threshold_);
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "beta=" << beta_ << ", threshold=" << threshold_;
        return ss.str();
    }

private:
    double beta_;
    double threshold_;
};

// ============================================================================
// Softsign
// ============================================================================
// Softsign(x) = x / (1 + |x|)

class Softsign : public Module {
public:
    Softsign() : Module("Softsign") {}

    Tensor forward(const Tensor& input) override {
        Tensor result = input.clone();
        float* data = result.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        for (int64_t i = 0; i < result.numel(); ++i) {
            float x = in_data[i];
            data[i] = x / (1.0f + std::abs(x));
        }
        return result;
    }
};

// ============================================================================
// Hardtanh
// ============================================================================
// Hardtanh(x) = max(min_val, min(max_val, x))

class Hardtanh : public Module {
public:
    explicit Hardtanh(double min_val = -1.0, double max_val = 1.0, bool inplace = false)
        : Module("Hardtanh")
        , min_val_(min_val)
        , max_val_(max_val)
        , inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware hardtanh to maintain gradient flow
        return torch::autograd::hardtanh_autograd(input, min_val_, max_val_);
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "min_val=" << min_val_ << ", max_val=" << max_val_;
        if (inplace_) ss << ", inplace=True";
        return ss.str();
    }

private:
    double min_val_;
    double max_val_;
    bool inplace_;
};

// ============================================================================
// Hardsigmoid
// ============================================================================
// Hardsigmoid(x) = 0 if x <= -3, 1 if x >= 3, else x/6 + 0.5

class Hardsigmoid : public Module {
public:
    explicit Hardsigmoid(bool inplace = false)
        : Module("Hardsigmoid"), inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware hardsigmoid to maintain gradient flow
        return torch::autograd::hardsigmoid_autograd(input);
    }

private:
    bool inplace_;
};

// ============================================================================
// Hardswish
// ============================================================================
// Hardswish(x) = x * Hardsigmoid(x)

class Hardswish : public Module {
public:
    explicit Hardswish(bool inplace = false)
        : Module("Hardswish"), inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware hardswish to maintain gradient flow
        return torch::autograd::hardswish_autograd(input);
    }

private:
    bool inplace_;
};

// ============================================================================
// SiLU (Swish)
// ============================================================================
// SiLU(x) = x * sigmoid(x)

class SiLU : public Module {
public:
    explicit SiLU(bool inplace = false)
        : Module("SiLU"), inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware silu to maintain gradient flow
        return torch::autograd::silu_autograd(input);
    }

private:
    bool inplace_;
};

// ============================================================================
// Mish
// ============================================================================
// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

class Mish : public Module {
public:
    explicit Mish(bool inplace = false)
        : Module("Mish"), inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        // Use autograd-aware mish to maintain gradient flow
        return torch::autograd::mish_autograd(input);
    }

private:
    bool inplace_;
};

// ============================================================================
// Threshold
// ============================================================================
// Threshold(x) = x if x > threshold, else value

class Threshold : public Module {
public:
    Threshold(double threshold, double value, bool inplace = false)
        : Module("Threshold")
        , threshold_(threshold)
        , value_(value)
        , inplace_(inplace) {}

    Tensor forward(const Tensor& input) override {
        Tensor result = inplace_ ? input : input.clone();
        float* data = result.mutable_data_ptr<float>();
        float thresh = static_cast<float>(threshold_);
        float val = static_cast<float>(value_);

        for (int64_t i = 0; i < result.numel(); ++i) {
            if (data[i] <= thresh) {
                data[i] = val;
            }
        }
        return result;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "threshold=" << threshold_ << ", value=" << value_;
        if (inplace_) ss << ", inplace=True";
        return ss.str();
    }

private:
    double threshold_;
    double value_;
    bool inplace_;
};

} // namespace nn
} // namespace torch
