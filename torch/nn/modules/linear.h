#pragma once

#include "torch/nn/module.h"
#include "torch/nn/init.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "aten/src/ATen/native/cpu/PromeBLAS.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
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
    Linear(int64_t in_features, int64_t out_features, bool bias = true,
           bool fused_relu = false)
        : Module("Linear")
        , in_features_(in_features)
        , out_features_(out_features)
        , has_bias_(bias)
        , fused_relu_(fused_relu)
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
        // PyTorch Linear uses simple uniform initialization:
        // bound = 1 / sqrt(fan_in)
        // NOT Kaiming uniform (which uses sqrt(3) * 1/sqrt(fan_in))
        double fan_in = static_cast<double>(in_features_);
        double bound = 1.0 / std::sqrt(fan_in);

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
        auto* weight_param = get_parameter("weight");
        Tensor W = weight_param->data();  // [out_features, in_features]

        // ================================================================
        // Fast path: no autograd needed (inference or NoGradGuard)
        // Bypass all autograd wrappers — direct sgemm_nt + AVX2 bias
        // ================================================================
        bool need_grad = torch::autograd::GradMode::is_enabled() &&
                         (input.requires_grad() || W.requires_grad());

        if (!need_grad && input.dim() >= 2 && W.dtype() == c10::ScalarType::Float) {
            Tensor x = input.contiguous();
            const float* x_data = x.data_ptr<float>();
            const float* w_data = W.data_ptr<float>();

            // Flatten to 2D if needed
            int64_t M = 1;
            for (int64_t d = 0; d < input.dim() - 1; ++d) M *= input.size(d);
            int64_t K = in_features_;
            int64_t N = out_features_;

            Tensor result = at::empty({M, N});
            float* out = result.mutable_data_ptr<float>();

            // y = x @ W^T  (W is [N, K], we want x[M,K] @ W^T[K,N])
            at::native::blas::sgemm_nt(M, K, N, 1.0f, x_data, K, w_data, K, 0.0f, out, N);

            // Fused bias add with AVX2
            if (has_bias_) {
                auto* bias_param = get_parameter("bias");
                const float* b = bias_param->data().data_ptr<float>();
                for (int64_t i = 0; i < M; ++i) {
                    float* row = out + i * N;
                    int64_t j = 0;
                    constexpr int W = at::native::tuda::VecF::width;
                    for (; j + W <= N; j += W)
                        (at::native::tuda::VecF::load(row + j) + at::native::tuda::VecF::load(b + j)).store(row + j);
                    for (; j < N; ++j) row[j] += b[j];
                }
            }

            // Fused relu in inference fast path
            if (fused_relu_) {
                int64_t total = M * N;
                for (int64_t i = 0; i < total; ++i) {
                    out[i] = out[i] > 0.0f ? out[i] : 0.0f;
                }
            }

            // Reshape back if input was >2D
            if (input.dim() > 2) {
                std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end() - 1);
                out_shape.push_back(N);
                result = result.reshape(out_shape);
            }
            return result;
        }

        // ================================================================
        // Autograd path: full gradient tracking
        // ================================================================

        // Fast fused path for 2D inputs (most common: MLP training)
        // Fuses mm + bias_add (+ optional relu) into a single op and backward node
        if (input.dim() == 2 && W.dtype() == c10::ScalarType::Float) {
            Tensor bias_data;
            if (has_bias_) {
                auto* bias_param = get_parameter("bias");
                bias_data = bias_param->data();
            }

            if (fused_relu_) {
                return torch::autograd::fused_linear_relu_autograd(
                    input, W, bias_data, has_bias_);
            } else {
                return torch::autograd::fused_linear_autograd(
                    input, W, bias_data, has_bias_);
            }
        }

        // Fallback path for 1D and >2D inputs (unchanged)
        Tensor weight_t = torch::autograd::t_autograd(W);
        Tensor output;

        if (input.dim() == 1) {
            output = torch::autograd::mv_autograd(W, input);
        } else {
            auto input_shape = input.sizes().vec();
            int64_t batch = 1;
            for (size_t i = 0; i < input_shape.size() - 1; ++i) {
                batch *= input_shape[i];
            }
            Tensor input_2d = torch::autograd::reshape_autograd(input, {batch, in_features_});
            Tensor output_2d = torch::autograd::mm_autograd(input_2d, weight_t);
            std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
            output_shape.push_back(out_features_);
            output = torch::autograd::reshape_autograd(output_2d, output_shape);
        }

        if (has_bias_) {
            auto* bias_param = get_parameter("bias");
            output = torch::autograd::add_autograd(output, bias_param->data());
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "in_features=" << in_features_
           << ", out_features=" << out_features_
           << ", bias=" << (has_bias_ ? "True" : "False");
        if (fused_relu_) ss << ", fused_relu=True";
        return ss.str();
    }

    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    bool fused_relu() const { return fused_relu_; }
    void set_fused_relu(bool v) { fused_relu_ = v; }

private:
    int64_t in_features_;
    int64_t out_features_;
    bool has_bias_;
    bool fused_relu_;
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
        double bound = 1.0 / std::sqrt(fan_in);

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

// ============================================================================
// Flatten - Flattens a contiguous range of dims
// ============================================================================
// Input: (N, *dims) -> Output: (N, prod(dims[start_dim:end_dim+1]))

class Flatten : public Module {
public:
    Flatten(int64_t start_dim = 1, int64_t end_dim = -1)
        : Module("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {}

    Tensor forward(const Tensor& input) override {
        return input.flatten(start_dim_, end_dim_);
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "start_dim=" << start_dim_ << ", end_dim=" << end_dim_;
        return ss.str();
    }

private:
    int64_t start_dim_;
    int64_t end_dim_;
};

// ============================================================================
// Unflatten - Unflattens a dim into multiple dims
// ============================================================================

class Unflatten : public Module {
public:
    Unflatten(int64_t dim, std::vector<int64_t> unflattened_size)
        : Module("Unflatten"), dim_(dim), unflattened_size_(std::move(unflattened_size)) {}

    Tensor forward(const Tensor& input) override {
        int64_t actual_dim = dim_ < 0 ? dim_ + input.dim() : dim_;

        // Build new shape: dims before | unflattened | dims after
        std::vector<int64_t> new_shape;
        for (int64_t i = 0; i < actual_dim; ++i) {
            new_shape.push_back(input.size(i));
        }
        for (auto s : unflattened_size_) {
            new_shape.push_back(s);
        }
        for (int64_t i = actual_dim + 1; i < input.dim(); ++i) {
            new_shape.push_back(input.size(i));
        }

        return input.reshape(new_shape);
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "dim=" << dim_ << ", unflattened_size=(";
        for (size_t i = 0; i < unflattened_size_.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << unflattened_size_[i];
        }
        ss << ")";
        return ss.str();
    }

private:
    int64_t dim_;
    std::vector<int64_t> unflattened_size_;
};

} // namespace nn
} // namespace torch
