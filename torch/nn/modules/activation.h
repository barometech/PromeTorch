#pragma once

#include "torch/nn/module.h"
#include "torch/csrc/autograd/autograd.h"
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
#ifdef PT_USE_CUDA
        if (input.is_cuda()) {
            // Use CUDA clamp kernel: clamp(x, 0, 6)
            return at::cuda_ops::clamp(input, 0.0f, 6.0f);
        }
#endif
        // CPU path
        Tensor result = input.relu();
        float* data = result.mutable_data_ptr<float>();
        for (int64_t i = 0; i < result.numel(); ++i) {
            if (data[i] > 6.0f) data[i] = 6.0f;
        }
        return result;
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
#ifdef PT_USE_CUDA
        if (input.is_cuda()) {
            return at::cuda_ops::leaky_relu(input, static_cast<float>(negative_slope_));
        }
#endif
        // CPU path
        Tensor result = inplace_ ? input : input.clone();
        float* data = result.mutable_data_ptr<float>();
        float slope = static_cast<float>(negative_slope_);

        for (int64_t i = 0; i < result.numel(); ++i) {
            if (data[i] < 0) {
                data[i] *= slope;
            }
        }
        return result;
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
        // PReLU needs custom CUDA kernel - for now CPU implementation
        // TODO: Add CUDA kernel for PReLU
        Tensor result = input.clone();
        auto* weight = get_parameter("weight");
        const float* w = weight->data().data_ptr<float>();
        float* data = result.mutable_data_ptr<float>();

        for (int64_t i = 0; i < result.numel(); ++i) {
            if (data[i] < 0) {
                int64_t w_idx = (num_parameters_ == 1) ? 0 : (i % num_parameters_);
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
        Tensor result = inplace_ ? input : input.clone();
        float* data = result.mutable_data_ptr<float>();
        float alpha = static_cast<float>(alpha_);

        for (int64_t i = 0; i < result.numel(); ++i) {
            if (data[i] <= 0) {
                data[i] = alpha * (std::exp(data[i]) - 1.0f);
            }
        }
        return result;
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
        // SELU constants
        constexpr float alpha = 1.6732632423543772848170429916717f;
        constexpr float scale = 1.0507009873554804934193349852946f;

        Tensor result = inplace_ ? input : input.clone();
        float* data = result.mutable_data_ptr<float>();

        for (int64_t i = 0; i < result.numel(); ++i) {
            if (data[i] > 0) {
                data[i] *= scale;
            } else {
                data[i] = scale * alpha * (std::exp(data[i]) - 1.0f);
            }
        }
        return result;
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
#ifdef PT_USE_CUDA
        if (input.is_cuda()) {
            return at::cuda_ops::gelu(input);
        }
#endif
        // CPU path
        Tensor result = input.clone();
        float* data = result.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        if (approximate_ == "tanh") {
            // Tanh approximation
            constexpr float sqrt_2_over_pi = 0.7978845608028654f;
            constexpr float coef = 0.044715f;

            for (int64_t i = 0; i < result.numel(); ++i) {
                float x = in_data[i];
                float inner = sqrt_2_over_pi * (x + coef * x * x * x);
                data[i] = 0.5f * x * (1.0f + std::tanh(inner));
            }
        } else {
            // Exact (using erf)
            constexpr float sqrt_half = 0.7071067811865476f;

            for (int64_t i = 0; i < result.numel(); ++i) {
                float x = in_data[i];
                data[i] = 0.5f * x * (1.0f + std::erf(x * sqrt_half));
            }
        }
        return result;
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
        return input.sigmoid();
    }
};

// ============================================================================
// Tanh
// ============================================================================

class Tanh : public Module {
public:
    Tanh() : Module("Tanh") {}

    Tensor forward(const Tensor& input) override {
        return input.tanh();
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
        Tensor result = input.clone();
        float* data = result.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();
        float beta = static_cast<float>(beta_);
        float threshold = static_cast<float>(threshold_);
        float inv_beta = 1.0f / beta;

        for (int64_t i = 0; i < result.numel(); ++i) {
            float x = in_data[i];
            if (x * beta > threshold) {
                data[i] = x;  // Linear for large values
            } else {
                data[i] = inv_beta * std::log(1.0f + std::exp(beta * x));
            }
        }
        return result;
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
        Tensor result = inplace_ ? input : input.clone();
        float* data = result.mutable_data_ptr<float>();
        float min_v = static_cast<float>(min_val_);
        float max_v = static_cast<float>(max_val_);

        for (int64_t i = 0; i < result.numel(); ++i) {
            if (data[i] < min_v) data[i] = min_v;
            else if (data[i] > max_v) data[i] = max_v;
        }
        return result;
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
        Tensor result = inplace_ ? input : input.clone();
        float* data = result.mutable_data_ptr<float>();

        for (int64_t i = 0; i < result.numel(); ++i) {
            float x = data[i];
            if (x <= -3.0f) {
                data[i] = 0.0f;
            } else if (x >= 3.0f) {
                data[i] = 1.0f;
            } else {
                data[i] = x / 6.0f + 0.5f;
            }
        }
        return result;
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
        Tensor result = inplace_ ? input : input.clone();
        float* data = result.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        for (int64_t i = 0; i < result.numel(); ++i) {
            float x = in_data[i];
            float hs;
            if (x <= -3.0f) {
                hs = 0.0f;
            } else if (x >= 3.0f) {
                hs = 1.0f;
            } else {
                hs = x / 6.0f + 0.5f;
            }
            data[i] = x * hs;
        }
        return result;
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
        Tensor sigmoid_x = input.sigmoid();
        return input.mul(sigmoid_x);
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
        Tensor result = input.clone();
        float* data = result.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        for (int64_t i = 0; i < result.numel(); ++i) {
            float x = in_data[i];
            float sp = std::log(1.0f + std::exp(x));  // softplus
            data[i] = x * std::tanh(sp);
        }
        return result;
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
