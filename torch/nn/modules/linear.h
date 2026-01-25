#pragma once

#include "torch/nn/module.h"
#include "torch/nn/init.h"
#include "torch/csrc/autograd/autograd.h"
#include <cmath>

namespace torch {
namespace nn {

// ============================================================================
// Identity - A placeholder identity operator
// ============================================================================
// A placeholder layer that returns the input unchanged.
// Useful for skip connections or as a placeholder in Sequential.

class Identity : public Module {
public:
    Identity() : Module("Identity") {}

    Tensor forward(const Tensor& input) override {
        return input;
    }
};

// ============================================================================
// Linear - Applies a linear transformation: y = xW^T + b
// ============================================================================
// Input: (*, in_features) where * means any number of dimensions
// Output: (*, out_features)

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true)
        : Module("Linear")
        , in_features_(in_features)
        , out_features_(out_features)
        , has_bias_(bias)
    {
        // Initialize weight: [out_features, in_features]
        Tensor weight = at::empty({out_features, in_features});
        register_parameter("weight", Parameter(weight));

        if (has_bias_) {
            Tensor bias_tensor = at::empty({out_features});
            register_parameter("bias", Parameter(bias_tensor));
        }

        reset_parameters();
    }

    void reset_parameters() override {
        // Kaiming uniform initialization
        // Same as PyTorch's default
        double fan_in = static_cast<double>(in_features_);
        double std = 1.0 / std::sqrt(fan_in);
        double bound = std::sqrt(3.0) * std;

        // Uniform(-bound, bound)
        auto* weight = get_parameter("weight");
        if (weight && weight->defined()) {
            Tensor w = weight->data();
            float* data = w.mutable_data_ptr<float>();
            for (int64_t i = 0; i < w.numel(); ++i) {
                data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
            }
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            if (bias && bias->defined()) {
                Tensor b = bias->data();
                float* data = b.mutable_data_ptr<float>();
                for (int64_t i = 0; i < b.numel(); ++i) {
                    data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
                }
            }
        }
    }

    Tensor forward(const Tensor& input) override {
        // y = xW^T + b
        // Use autograd-tracked operations for gradient computation
        auto* weight = get_parameter("weight");
        // CRITICAL: Use t_autograd() to maintain gradient flow through transpose!
        Tensor weight_t = torch::autograd::t_autograd(weight->data());
        Tensor output;

        if (input.dim() == 1) {
            // Input is [in_features], output is [out_features]
            output = torch::autograd::mv_autograd(weight->data(), input);
        } else if (input.dim() == 2) {
            // Input is [batch, in_features], output is [batch, out_features]
            // y = x @ W^T
            output = torch::autograd::mm_autograd(input, weight_t);
        } else {
            // Input is [*, in_features], output is [*, out_features]
            // Flatten to 2D, compute, reshape back
            // IMPORTANT: Use autograd-aware reshape to maintain gradient flow!
            auto input_shape = input.sizes().vec();
            int64_t batch = 1;
            for (size_t i = 0; i < input_shape.size() - 1; ++i) {
                batch *= input_shape[i];
            }

            Tensor input_2d = torch::autograd::reshape_autograd(input, {batch, in_features_});
            Tensor output_2d = torch::autograd::mm_autograd(input_2d, weight_t);

            // Reshape back - use autograd-aware reshape
            std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
            output_shape.push_back(out_features_);
            output = torch::autograd::reshape_autograd(output_2d, output_shape);
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            output = torch::autograd::add_autograd(output, bias->data());
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "in_features=" << in_features_
           << ", out_features=" << out_features_
           << ", bias=" << (has_bias_ ? "True" : "False");
        return ss.str();
    }

    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    bool has_bias_;
};

// ============================================================================
// Bilinear - Applies a bilinear transformation: y = x1^T A x2 + b
// ============================================================================
// Input1: (*, in1_features)
// Input2: (*, in2_features)
// Output: (*, out_features)

class Bilinear : public Module {
public:
    Bilinear(int64_t in1_features, int64_t in2_features, int64_t out_features, bool bias = true)
        : Module("Bilinear")
        , in1_features_(in1_features)
        , in2_features_(in2_features)
        , out_features_(out_features)
        , has_bias_(bias)
    {
        // Weight shape: [out_features, in1_features, in2_features]
        Tensor weight = at::empty({out_features, in1_features, in2_features});
        register_parameter("weight", Parameter(weight));

        if (has_bias_) {
            Tensor bias_tensor = at::empty({out_features});
            register_parameter("bias", Parameter(bias_tensor));
        }

        reset_parameters();
    }

    void reset_parameters() override {
        double bound = 1.0 / std::sqrt(static_cast<double>(in1_features_));

        auto* weight = get_parameter("weight");
        if (weight && weight->defined()) {
            Tensor w = weight->data();
            float* data = w.mutable_data_ptr<float>();
            for (int64_t i = 0; i < w.numel(); ++i) {
                data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
            }
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            if (bias && bias->defined()) {
                Tensor b = bias->data();
                float* data = b.mutable_data_ptr<float>();
                for (int64_t i = 0; i < b.numel(); ++i) {
                    data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
                }
            }
        }
    }

    Tensor forward(const Tensor& input1, const Tensor& input2) override {
        // y_k = x1^T W_k x2 + b_k
        // For each output feature k, compute x1^T W[k] x2

        auto* weight = get_parameter("weight");
        Tensor W = weight->data();  // [out_features, in1_features, in2_features]

        int64_t batch_size = input1.size(0);
        Tensor output = at::zeros({batch_size, out_features_});

        // Simple implementation (can be optimized with einsum)
        float* out_data = output.mutable_data_ptr<float>();
        const float* x1_data = input1.data_ptr<float>();
        const float* x2_data = input2.data_ptr<float>();
        const float* w_data = W.data_ptr<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t k = 0; k < out_features_; ++k) {
                float sum = 0.0f;
                for (int64_t i = 0; i < in1_features_; ++i) {
                    for (int64_t j = 0; j < in2_features_; ++j) {
                        // W[k, i, j] * x1[b, i] * x2[b, j]
                        int64_t w_idx = k * in1_features_ * in2_features_ + i * in2_features_ + j;
                        sum += w_data[w_idx] * x1_data[b * in1_features_ + i] * x2_data[b * in2_features_ + j];
                    }
                }
                out_data[b * out_features_ + k] = sum;
            }
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            output = output.add(bias->data());
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "in1_features=" << in1_features_
           << ", in2_features=" << in2_features_
           << ", out_features=" << out_features_
           << ", bias=" << (has_bias_ ? "True" : "False");
        return ss.str();
    }

private:
    int64_t in1_features_;
    int64_t in2_features_;
    int64_t out_features_;
    bool has_bias_;
};

// ============================================================================
// LazyLinear - Linear layer with lazy initialization
// ============================================================================
// The in_features is inferred from the first input.

class LazyLinear : public Module {
public:
    explicit LazyLinear(int64_t out_features, bool bias = true)
        : Module("LazyLinear")
        , out_features_(out_features)
        , has_bias_(bias)
        , initialized_(false)
        , in_features_(0)
    {}

    Tensor forward(const Tensor& input) override {
        if (!initialized_) {
            in_features_ = input.size(-1);
            initialize();
        }

        auto* weight = get_parameter("weight");
        Tensor output = input.mm(weight->data().t());

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            output = output.add(bias->data());
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "out_features=" << out_features_
           << ", bias=" << (has_bias_ ? "True" : "False");
        if (initialized_) {
            ss << ", in_features=" << in_features_;
        }
        return ss.str();
    }

private:
    void initialize() {
        Tensor weight = at::empty({out_features_, in_features_});
        register_parameter("weight", Parameter(weight));

        if (has_bias_) {
            Tensor bias = at::empty({out_features_});
            register_parameter("bias", Parameter(bias));
        }

        reset_parameters();
        initialized_ = true;
    }

    void reset_parameters() override {
        double fan_in = static_cast<double>(in_features_);
        double std = 1.0 / std::sqrt(fan_in);
        double bound = std::sqrt(3.0) * std;

        auto* weight = get_parameter("weight");
        if (weight && weight->defined()) {
            Tensor w = weight->data();
            float* data = w.mutable_data_ptr<float>();
            for (int64_t i = 0; i < w.numel(); ++i) {
                data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
            }
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            if (bias && bias->defined()) {
                Tensor b = bias->data();
                float* data = b.mutable_data_ptr<float>();
                for (int64_t i = 0; i < b.numel(); ++i) {
                    data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
                }
            }
        }
    }

    int64_t out_features_;
    bool has_bias_;
    bool initialized_;
    int64_t in_features_;
};

} // namespace nn
} // namespace torch
