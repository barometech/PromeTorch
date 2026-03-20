#pragma once

#include "torch/nn/module.h"
#include "torch/nn/init.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "aten/src/ATen/native/cpu/PromeBLAS.h"
#include "aten/src/ATen/native/cpu/FastOps.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/LinearAlgebra.h"
#include <cmath>
#include <iomanip>
#include <cstring>

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
        , cached_weight_(nullptr)
        , cached_bias_(nullptr)
    {
        // Initialize weight: [out_features, in_features]
        Tensor weight = at::empty({out_features, in_features});
        register_parameter("weight", Parameter(weight));

        if (has_bias_) {
            Tensor bias_tensor = at::empty({out_features});
            register_parameter("bias", Parameter(bias_tensor));
        }

        // Cache parameter pointers to avoid map lookup on every forward()
        cached_weight_ = get_parameter("weight");
        if (has_bias_) cached_bias_ = get_parameter("bias");

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
        // Use cached pointers (avoid map lookup on every forward call)
        Tensor W = cached_weight_->data();  // [out_features, in_features]

        // ================================================================
        // FAST PATH: float32, no autograd (inference or NoGradGuard)
        // Zero dispatch overhead — calls FastOps directly.
        // On Elbrus E8C2 this eliminates ~50% overhead for small tensors.
        // ================================================================
        bool need_grad = torch::autograd::GradMode::is_enabled() &&
                         (input.requires_grad() || W.requires_grad());

        if (!need_grad && W.dtype() == c10::ScalarType::Float) {
            // 2D fast path: most common (MLP inference)
            if (input.dim() == 2 && input.is_contiguous()) {
                if (fused_relu_ && has_bias_) {
                    return at::native::fast::fused_linear_relu_f32(
                        input, W, cached_bias_->data());
                } else if (has_bias_) {
                    return at::native::fast::fused_linear_f32(
                        input, W, cached_bias_->data());
                } else if (fused_relu_) {
                    return at::native::fast::fused_linear_relu_nobias_f32(input, W);
                } else {
                    return at::native::fast::fused_linear_nobias_f32(input, W);
                }
            }

            // >2D or non-contiguous: flatten + fast path + reshape
            if (input.dim() >= 2) {
                Tensor x = input.contiguous();
                int64_t M = 1;
                for (int64_t d = 0; d < input.dim() - 1; ++d) M *= input.size(d);
                Tensor x_2d = x.reshape({M, in_features_});

                Tensor result;
                if (fused_relu_ && has_bias_) {
                    result = at::native::fast::fused_linear_relu_f32(
                        x_2d, W, cached_bias_->data());
                } else if (has_bias_) {
                    result = at::native::fast::fused_linear_f32(
                        x_2d, W, cached_bias_->data());
                } else if (fused_relu_) {
                    result = at::native::fast::fused_linear_relu_nobias_f32(x_2d, W);
                } else {
                    result = at::native::fast::fused_linear_nobias_f32(x_2d, W);
                }

                if (input.dim() > 2) {
                    std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end() - 1);
                    out_shape.push_back(out_features_);
                    result = result.reshape(out_shape);
                }
                return result;
            }
        }

        // ================================================================
        // Autograd path: full gradient tracking
        // ================================================================

        // Fast fused path for 2D inputs (most common: MLP training)
        // Fuses mm + bias_add (+ optional relu) into a single op and backward node
        if (input.dim() == 2 && W.dtype() == c10::ScalarType::Float) {
            Tensor bias_data;
            if (has_bias_) {
                bias_data = cached_bias_->data();
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
            output = torch::autograd::add_autograd(output, cached_bias_->data());
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
    // Cached parameter pointers — avoids std::map lookup on every forward() call.
    // Linear's parameters never change identity (only data), so caching is safe.
    Parameter* cached_weight_;
    Parameter* cached_bias_;
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

// ============================================================================
// LowRankLinear - Factored linear: W ≈ A @ B, forward: x @ B^T @ A^T + bias
// ============================================================================
// Instead of storing W[out, in], stores A[out, rank] and B[rank, in].
// When rank << min(out, in), uses less memory and computes faster
// (2 smaller matmuls vs 1 large).
//
// Can be created:
// 1. Directly: LowRankLinear(in, out, rank)
// 2. From existing Linear: LowRankLinear::from_linear(linear, rank)
//    Uses randomized SVD to decompose the weight matrix.

class LowRankLinear : public Module {
public:
    LowRankLinear(int64_t in_features, int64_t out_features, int64_t rank, bool bias = true)
        : Module("LowRankLinear")
        , in_features_(in_features)
        , out_features_(out_features)
        , rank_(rank)
        , has_bias_(bias)
    {
        // A: [out_features, rank]
        Tensor A_tensor = at::empty({out_features, rank});
        register_parameter("A", Parameter(A_tensor));

        // B: [rank, in_features]
        Tensor B_tensor = at::empty({rank, in_features});
        register_parameter("B", Parameter(B_tensor));

        if (has_bias_) {
            Tensor bias_tensor = at::empty({out_features});
            register_parameter("bias", Parameter(bias_tensor));
        }

        reset_parameters();
    }

    void reset_parameters() override {
        // Initialize so that A @ B has variance ~ 1/in_features (like Linear)
        // A ~ N(0, 1/sqrt(rank)), B ~ N(0, 1/sqrt(in_features))
        // Var(AB) = rank * (1/rank) * (1/in_features) = 1/in_features
        double bound_A = 1.0 / std::sqrt(static_cast<double>(rank_));
        double bound_B = 1.0 / std::sqrt(static_cast<double>(in_features_));

        auto* A_param = get_parameter("A");
        if (A_param && A_param->defined()) {
            float* data = A_param->data().mutable_data_ptr<float>();
            for (int64_t i = 0; i < A_param->data().numel(); ++i) {
                data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound_A);
            }
        }

        auto* B_param = get_parameter("B");
        if (B_param && B_param->defined()) {
            float* data = B_param->data().mutable_data_ptr<float>();
            for (int64_t i = 0; i < B_param->data().numel(); ++i) {
                data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound_B);
            }
        }

        if (has_bias_) {
            double bound = 1.0 / std::sqrt(static_cast<double>(in_features_));
            auto* bias_param = get_parameter("bias");
            if (bias_param && bias_param->defined()) {
                float* data = bias_param->data().mutable_data_ptr<float>();
                for (int64_t i = 0; i < bias_param->data().numel(); ++i) {
                    data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
                }
            }
        }
    }

    // Create LowRankLinear from an existing Linear by SVD compression
    static std::shared_ptr<LowRankLinear> from_linear(
        std::shared_ptr<Linear> linear, int64_t rank)
    {
        int64_t out_f = linear->out_features();
        int64_t in_f = linear->in_features();
        bool has_b = (linear->get_parameter("bias") != nullptr);

        auto lr = std::make_shared<LowRankLinear>(in_f, out_f, rank, has_b);

        // Decompose weight W[out, in] into A[out, rank] @ B[rank, in]
        Tensor W = linear->get_parameter("weight")->data();
        auto cw = at::native::compress_weight(W, rank);

        // Copy decomposed factors
        auto* A_param = lr->get_parameter("A");
        auto* B_param = lr->get_parameter("B");
        {
            Tensor A_src = cw.A.contiguous();  // [out, rank]
            Tensor B_src = cw.B.contiguous();  // [rank, in]
            float* a_dst = A_param->data().mutable_data_ptr<float>();
            float* b_dst = B_param->data().mutable_data_ptr<float>();
            const float* a_src = A_src.data_ptr<float>();
            const float* b_src = B_src.data_ptr<float>();
            std::memcpy(a_dst, a_src, A_src.numel() * sizeof(float));
            std::memcpy(b_dst, b_src, B_src.numel() * sizeof(float));
        }

        // Copy bias if present
        if (has_b) {
            auto* src_bias = linear->get_parameter("bias");
            auto* dst_bias = lr->get_parameter("bias");
            const float* src = src_bias->data().data_ptr<float>();
            float* dst = dst_bias->data().mutable_data_ptr<float>();
            std::memcpy(dst, src, src_bias->data().numel() * sizeof(float));
        }

        return lr;
    }

    Tensor forward(const Tensor& input) override {
        auto* A_param = get_parameter("A");
        auto* B_param = get_parameter("B");
        Tensor A = A_param->data();  // [out_features, rank]
        Tensor B = B_param->data();  // [rank, in_features]

        // ================================================================
        // Fast path: no autograd (inference or NoGradGuard)
        // ================================================================
        bool need_grad = torch::autograd::GradMode::is_enabled() &&
                         (input.requires_grad() || A.requires_grad() || B.requires_grad());

        if (!need_grad && input.dim() >= 2 && A.dtype() == c10::ScalarType::Float) {
            Tensor x = input.contiguous();
            const float* x_data = x.data_ptr<float>();
            const float* b_data = B.data_ptr<float>();
            const float* a_data = A.data_ptr<float>();

            int64_t M = 1;
            for (int64_t d = 0; d < input.dim() - 1; ++d) M *= input.size(d);
            int64_t K = in_features_;
            int64_t R = rank_;
            int64_t N = out_features_;

            // temp = x @ B^T  [M, K] @ [K, R] = [M, R]
            Tensor temp = at::empty({M, R});
            float* t_data = temp.mutable_data_ptr<float>();
            at::native::hot::sgemm_nt(M, K, R, 1.0f, x_data, K, b_data, K, 0.0f, t_data, R);

            // result = temp @ A^T  [M, R] @ [R, N] = [M, N]
            Tensor result = at::empty({M, N});
            float* out = result.mutable_data_ptr<float>();
            at::native::hot::sgemm_nt(M, R, N, 1.0f, t_data, R, a_data, R, 0.0f, out, N);

            // Bias add
            if (has_bias_) {
                auto* bias_param = get_parameter("bias");
                const float* b = bias_param->data().data_ptr<float>();
                for (int64_t i = 0; i < M; ++i) {
                    float* row = out + i * N;
                    for (int64_t j = 0; j < N; ++j) row[j] += b[j];
                }
            }

            if (input.dim() > 2) {
                std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end() - 1);
                out_shape.push_back(N);
                result = result.reshape(out_shape);
            }
            return result;
        }

        // ================================================================
        // Autograd path
        // ================================================================
        if (input.dim() == 2 && A.dtype() == c10::ScalarType::Float) {
            Tensor bias_data;
            if (has_bias_) {
                auto* bias_param = get_parameter("bias");
                bias_data = bias_param->data();
            }
            return torch::autograd::low_rank_linear_autograd(input, A, B, bias_data, has_bias_);
        }

        // Fallback: use mm autograd ops
        Tensor Bt = torch::autograd::t_autograd(B);
        Tensor At = torch::autograd::t_autograd(A);

        Tensor x2d = input;
        if (input.dim() > 2) {
            int64_t batch = 1;
            for (int64_t d = 0; d < input.dim() - 1; ++d) batch *= input.size(d);
            x2d = torch::autograd::reshape_autograd(input, {batch, in_features_});
        }

        Tensor temp = torch::autograd::mm_autograd(x2d, Bt);   // [M, rank]
        Tensor result = torch::autograd::mm_autograd(temp, At); // [M, out]

        if (has_bias_) {
            auto* bias_param = get_parameter("bias");
            result = torch::autograd::add_autograd(result, bias_param->data());
        }

        if (input.dim() > 2) {
            std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end() - 1);
            out_shape.push_back(out_features_);
            result = torch::autograd::reshape_autograd(result, out_shape);
        }

        return result;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "in_features=" << in_features_
           << ", out_features=" << out_features_
           << ", rank=" << rank_
           << ", bias=" << (has_bias_ ? "True" : "False")
           << ", compression=" << std::fixed << std::setprecision(1)
           << (100.0 * (1.0 - (double)(out_features_ * rank_ + rank_ * in_features_)
                              / (double)(out_features_ * in_features_))) << "%";
        return ss.str();
    }

    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    int64_t rank() const { return rank_; }
    bool has_bias() const { return has_bias_; }

    // Reconstruct the full weight matrix W ≈ A @ B
    Tensor weight_reconstructed() const {
        auto* A_param = const_cast<LowRankLinear*>(this)->get_parameter("A");
        auto* B_param = const_cast<LowRankLinear*>(this)->get_parameter("B");
        return at::native::mm(A_param->data(), B_param->data());
    }

    // Parameter count savings
    int64_t full_params() const { return out_features_ * in_features_; }
    int64_t compressed_params() const { return out_features_ * rank_ + rank_ * in_features_; }
    double compression_ratio() const {
        return static_cast<double>(compressed_params()) / static_cast<double>(full_params());
    }

private:
    int64_t in_features_;
    int64_t out_features_;
    int64_t rank_;
    bool has_bias_;
};

} // namespace nn
} // namespace torch
