#pragma once

#include "../module.h"
#include "torch/csrc/autograd/autograd.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include <cmath>
#include <algorithm>
#include <limits>

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif

namespace torch {
namespace nn {

// ============================================================================
// Reduction Mode for Loss Functions
// ============================================================================

enum class Reduction {
    None,   // No reduction, return loss per element
    Mean,   // Mean of all losses
    Sum     // Sum of all losses
};

// ============================================================================
// L1Loss - Mean Absolute Error
// ============================================================================
// Creates a criterion that measures the mean absolute error (MAE) between
// each element in the input x and target y.
// L = |x - y|

class L1Loss : public Module {
private:
    Reduction reduction_;

public:
    explicit L1Loss(Reduction reduction = Reduction::Mean)
        : reduction_(reduction) {}

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        int64_t numel = input.numel();

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                output_data[i] = std::abs(input_data[i] - target_data[i]);
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            sum += std::abs(input_data[i] - target_data[i]);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("L1Loss requires both input and target tensors");
    }

    std::string name() const override { return "L1Loss"; }
    Reduction reduction() const { return reduction_; }
};

// ============================================================================
// MSELoss - Mean Squared Error
// ============================================================================
// Creates a criterion that measures the mean squared error (squared L2 norm)
// between each element in the input x and target y.
// L = (x - y)^2

class MSELoss : public Module {
private:
    Reduction reduction_;

public:
    explicit MSELoss(Reduction reduction = Reduction::Mean)
        : reduction_(reduction) {}

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        int64_t numel = input.numel();

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                float diff = input_data[i] - target_data[i];
                output_data[i] = diff * diff;
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            double diff = input_data[i] - target_data[i];
            sum += diff * diff;
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("MSELoss requires both input and target tensors");
    }

    std::string name() const override { return "MSELoss"; }
    Reduction reduction() const { return reduction_; }
};

// ============================================================================
// SmoothL1Loss - Huber Loss
// ============================================================================
// Creates a criterion that uses a squared term if the absolute element-wise
// error falls below beta and an L1 term otherwise.
// L = 0.5 * (x - y)^2 / beta  if |x - y| < beta
//   = |x - y| - 0.5 * beta    otherwise

class SmoothL1Loss : public Module {
private:
    Reduction reduction_;
    double beta_;

public:
    explicit SmoothL1Loss(Reduction reduction = Reduction::Mean, double beta = 1.0)
        : reduction_(reduction), beta_(beta) {
        if (beta_ <= 0) {
            throw std::invalid_argument("beta must be positive");
        }
    }

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        int64_t numel = input.numel();

        auto compute_loss = [this](float diff) -> float {
            float abs_diff = std::abs(diff);
            if (abs_diff < beta_) {
                return 0.5f * diff * diff / static_cast<float>(beta_);
            } else {
                return abs_diff - 0.5f * static_cast<float>(beta_);
            }
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                output_data[i] = compute_loss(input_data[i] - target_data[i]);
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            sum += compute_loss(input_data[i] - target_data[i]);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("SmoothL1Loss requires both input and target tensors");
    }

    std::string name() const override { return "SmoothL1Loss"; }
    Reduction reduction() const { return reduction_; }
    double beta() const { return beta_; }
};

// ============================================================================
// HuberLoss - Huber Loss (alias for SmoothL1Loss with configurable delta)
// ============================================================================

class HuberLoss : public Module {
private:
    Reduction reduction_;
    double delta_;

public:
    explicit HuberLoss(Reduction reduction = Reduction::Mean, double delta = 1.0)
        : reduction_(reduction), delta_(delta) {
        if (delta_ <= 0) {
            throw std::invalid_argument("delta must be positive");
        }
    }

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        int64_t numel = input.numel();

        auto compute_loss = [this](float diff) -> float {
            float abs_diff = std::abs(diff);
            if (abs_diff <= delta_) {
                return 0.5f * diff * diff;
            } else {
                return static_cast<float>(delta_) * (abs_diff - 0.5f * static_cast<float>(delta_));
            }
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                output_data[i] = compute_loss(input_data[i] - target_data[i]);
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            sum += compute_loss(input_data[i] - target_data[i]);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("HuberLoss requires both input and target tensors");
    }

    std::string name() const override { return "HuberLoss"; }
    Reduction reduction() const { return reduction_; }
    double delta() const { return delta_; }
};

// ============================================================================
// BCELoss - Binary Cross Entropy Loss
// ============================================================================
// Creates a criterion that measures the Binary Cross Entropy between
// the target and the input probabilities.
// L = -(y * log(x) + (1 - y) * log(1 - x))
// Note: Input must be in range [0, 1]

class BCELoss : public Module {
private:
    Reduction reduction_;
    bool has_weight_;
    Tensor weight_;

public:
    explicit BCELoss(Reduction reduction = Reduction::Mean, const Tensor* weight = nullptr)
        : reduction_(reduction), has_weight_(weight != nullptr) {
        if (weight) {
            weight_ = *weight;
        }
    }

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        const float* weight_data = has_weight_ ? weight_.data_ptr<float>() : nullptr;
        int64_t numel = input.numel();

        // Clamp for numerical stability
        constexpr float eps = 1e-7f;

        auto compute_bce = [eps](float x, float y) -> float {
            float x_clamped = std::max(eps, std::min(1.0f - eps, x));
            return -(y * std::log(x_clamped) + (1.0f - y) * std::log(1.0f - x_clamped));
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                float loss = compute_bce(input_data[i], target_data[i]);
                if (weight_data) {
                    loss *= weight_data[i % weight_.numel()];
                }
                output_data[i] = loss;
            }
            return output;
        }

        double sum = 0.0;
        double weight_sum = 0.0;

        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            float loss = compute_bce(input_data[i], target_data[i]);
            float w = weight_data ? weight_data[i % weight_.numel()] : 1.0f;
            sum += loss * w;
            weight_sum += w;
        }

        if (reduction_ == Reduction::Mean) {
            sum = has_weight_ ? sum / weight_sum : sum / numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("BCELoss requires both input and target tensors");
    }

    std::string name() const override { return "BCELoss"; }
    Reduction reduction() const { return reduction_; }
};

// ============================================================================
// BCEWithLogitsLoss - Binary Cross Entropy with Logits
// ============================================================================
// Combines a Sigmoid layer and BCELoss in one single class.
// More numerically stable than using a plain Sigmoid followed by BCELoss.
// L = -w * (y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x)))
//   = w * (max(x, 0) - x * y + log(1 + exp(-|x|)))

class BCEWithLogitsLoss : public Module {
private:
    Reduction reduction_;
    bool has_weight_;
    Tensor weight_;
    bool has_pos_weight_;
    Tensor pos_weight_;

public:
    explicit BCEWithLogitsLoss(
        Reduction reduction = Reduction::Mean,
        const Tensor* weight = nullptr,
        const Tensor* pos_weight = nullptr
    ) : reduction_(reduction),
        has_weight_(weight != nullptr),
        has_pos_weight_(pos_weight != nullptr) {
        if (weight) {
            weight_ = *weight;
        }
        if (pos_weight) {
            pos_weight_ = *pos_weight;
        }
    }

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        const float* weight_data = has_weight_ ? weight_.data_ptr<float>() : nullptr;
        const float* pos_weight_data = has_pos_weight_ ? pos_weight_.data_ptr<float>() : nullptr;
        int64_t numel = input.numel();

        // Numerically stable formulation:
        // loss = max(x, 0) - x * y + log(1 + exp(-|x|))
        // With pos_weight: loss = max(x, 0) - x * y * p + log(1 + exp(-|x|)) * (p - 1) * y + log(1 + exp(-|x|))
        //                       = (1 - y) * x + (1 + (p - 1) * y) * log(1 + exp(-|x|))  (for x >= 0)

        auto compute_loss = [pos_weight_data, numel](float x, float y, int64_t i) -> float {
            float max_val = std::max(x, 0.0f);
            float log_term = std::log(1.0f + std::exp(-std::abs(x)));

            if (pos_weight_data) {
                float p = pos_weight_data[i % numel];
                // (1 - y) * max(x, 0) + (1 + (p - 1) * y) * (-log(sigmoid(x)))
                // For numerical stability: max(x, 0) - x * y + (1 + (p - 1) * y) * log(1 + exp(-|x|))
                return max_val - x * y + (1.0f + (p - 1.0f) * y) * log_term;
            } else {
                return max_val - x * y + log_term;
            }
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                float loss = compute_loss(input_data[i], target_data[i], i);
                if (weight_data) {
                    loss *= weight_data[i % weight_.numel()];
                }
                output_data[i] = loss;
            }
            return output;
        }

        double sum = 0.0;
        double weight_sum = 0.0;

        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            float loss = compute_loss(input_data[i], target_data[i], i);
            float w = weight_data ? weight_data[i % weight_.numel()] : 1.0f;
            sum += loss * w;
            weight_sum += w;
        }

        if (reduction_ == Reduction::Mean) {
            sum = has_weight_ ? sum / weight_sum : sum / numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("BCEWithLogitsLoss requires both input and target tensors");
    }

    std::string name() const override { return "BCEWithLogitsLoss"; }
    Reduction reduction() const { return reduction_; }
};

// ============================================================================
// NLLLoss - Negative Log Likelihood Loss
// ============================================================================
// The negative log likelihood loss. Useful for training a classification
// problem with C classes.
// Input: (N, C) log-probabilities of each class
// Target: (N) class indices in [0, C-1]
// L = -weight[class] * input[class]

class NLLLoss : public Module {
private:
    Reduction reduction_;
    int64_t ignore_index_;
    bool has_weight_;
    Tensor weight_;

public:
    explicit NLLLoss(
        Reduction reduction = Reduction::Mean,
        int64_t ignore_index = -100,
        const Tensor* weight = nullptr
    ) : reduction_(reduction),
        ignore_index_(ignore_index),
        has_weight_(weight != nullptr) {
        if (weight) {
            weight_ = *weight;
        }
    }

    Tensor forward(const Tensor& input, const Tensor& target) {
        // Input: (N, C) or (N, C, d1, d2, ...) log-probabilities
        // Target: (N) or (N, d1, d2, ...) class indices

        if (input.dim() < 2) {
            throw std::runtime_error("NLLLoss expects at least 2D input");
        }

        int64_t batch_size = input.size(0);
        int64_t num_classes = input.size(1);
        int64_t spatial_size = 1;

        for (int64_t d = 2; d < input.dim(); ++d) {
            spatial_size *= input.size(d);
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        const float* weight_data = has_weight_ ? weight_.data_ptr<float>() : nullptr;

        int64_t total_elements = batch_size * spatial_size;

        if (reduction_ == Reduction::None) {
            std::vector<int64_t> output_shape = target.sizes().vec();
            Tensor output = at::empty(output_shape);
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < total_elements; ++i) {
                int64_t b = i / spatial_size;
                int64_t s = i % spatial_size;

                int64_t class_idx = static_cast<int64_t>(target_data[i]);

                if (class_idx == ignore_index_) {
                    output_data[i] = 0.0f;
                    continue;
                }

                if (class_idx < 0 || class_idx >= num_classes) {
                    throw std::out_of_range("Target class index out of range");
                }

                int64_t input_idx = b * num_classes * spatial_size + class_idx * spatial_size + s;
                float loss = -input_data[input_idx];

                if (weight_data) {
                    loss *= weight_data[class_idx];
                }

                output_data[i] = loss;
            }
            return output;
        }

        double sum = 0.0;
        double weight_sum = 0.0;
        int64_t count = 0;

        for (int64_t i = 0; i < total_elements; ++i) {
            int64_t b = i / spatial_size;
            int64_t s = i % spatial_size;

            int64_t class_idx = static_cast<int64_t>(target_data[i]);

            if (class_idx == ignore_index_) {
                continue;
            }

            if (class_idx < 0 || class_idx >= num_classes) {
                throw std::out_of_range("Target class index out of range");
            }

            int64_t input_idx = b * num_classes * spatial_size + class_idx * spatial_size + s;
            float loss = -input_data[input_idx];
            float w = weight_data ? weight_data[class_idx] : 1.0f;

            sum += loss * w;
            weight_sum += w;
            count++;
        }

        if (reduction_ == Reduction::Mean) {
            sum = has_weight_ ? sum / weight_sum : sum / count;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("NLLLoss requires both input and target tensors");
    }

    std::string name() const override { return "NLLLoss"; }
    Reduction reduction() const { return reduction_; }
    int64_t ignore_index() const { return ignore_index_; }
};

// ============================================================================
// CrossEntropyLoss - Combines LogSoftmax and NLLLoss
// ============================================================================
// This criterion computes the cross entropy loss between input logits and target.
// Input: (N, C) raw logits (before softmax)
// Target: (N) class indices or (N, C) class probabilities
// L = -sum(y * log(softmax(x)))

class CrossEntropyLoss : public Module {
private:
    Reduction reduction_;
    int64_t ignore_index_;
    double label_smoothing_;
    bool has_weight_;
    Tensor weight_;

public:
    explicit CrossEntropyLoss(
        Reduction reduction = Reduction::Mean,
        int64_t ignore_index = -100,
        double label_smoothing = 0.0,
        const Tensor* weight = nullptr
    ) : reduction_(reduction),
        ignore_index_(ignore_index),
        label_smoothing_(label_smoothing),
        has_weight_(weight != nullptr) {
        if (label_smoothing < 0.0 || label_smoothing > 1.0) {
            throw std::invalid_argument("label_smoothing must be in [0, 1]");
        }
        if (weight) {
            weight_ = *weight;
        }
    }

    // Hard labels version (class indices)
    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.dim() < 2) {
            throw std::runtime_error("CrossEntropyLoss expects at least 2D input");
        }

#ifdef PT_USE_CUDA
        // Use CUDA kernel for GPU tensors WITH autograd support
        if (input.is_cuda()) {
            // Convert reduction enum to int: None=0, Mean=1, Sum=2
            int reduction_int = 1;  // Default: Mean
            if (reduction_ == Reduction::None) reduction_int = 0;
            else if (reduction_ == Reduction::Sum) reduction_int = 2;

            // Ensure target is on same device
            Tensor target_cuda = target.is_cuda() ? target : at::to_cuda(target);

            // Compute softmax for backward (needed for gradient computation)
            Tensor softmax_cuda = at::cuda_ops::softmax(input, /*dim=*/1);

            // Compute loss using CUDA kernel
            Tensor loss = at::cuda_ops::cross_entropy_loss(input, target_cuda, reduction_int);

            // Setup autograd if input requires grad
            if (torch::autograd::compute_requires_grad(input)) {
                int64_t batch_size = input.size(0);
                int64_t num_classes = input.size(1);
                int64_t num_valid = batch_size;  // Simplified: assume all samples valid

                auto grad_fn = torch::autograd::NodePool<torch::autograd::CrossEntropyBackward>::make_shared(
                    softmax_cuda, target_cuda, ignore_index_, num_classes, num_valid, /*output_cuda=*/true);
                grad_fn->add_input_metadata(input);
                torch::autograd::set_grad_fn(loss, grad_fn);
                loss.set_requires_grad(true);
            }

            return loss;
        }
#endif

        // ================================================================
        // FUSED FAST PATH: float32, 2D input, hard labels, no smoothing,
        // no weights, no ignore_index, Mean reduction
        // Single pass over data — critical for E2K with 64KB L1 cache
        // ================================================================
        if (input.dtype() == c10::ScalarType::Float &&
            input.dim() == 2 &&
            label_smoothing_ == 0.0 &&
            !has_weight_ &&
            ignore_index_ == -100 &&
            reduction_ == Reduction::Mean) {

            Tensor input_c = input.contiguous();
            int64_t batch_size = input_c.size(0);
            int64_t num_classes = input_c.size(1);

            // Convert targets to int64_t array (existing code stores as float)
            const float* target_data = target.data_ptr<float>();
            std::vector<int64_t> targets_i64(batch_size);
            bool targets_valid = true;
            for (int64_t i = 0; i < batch_size; ++i) {
                targets_i64[i] = static_cast<int64_t>(target_data[i]);
                if (targets_i64[i] == ignore_index_ || targets_i64[i] < 0 || targets_i64[i] >= num_classes) {
                    targets_valid = false;
                    break;
                }
            }

            if (targets_valid) {
                Tensor output = at::empty({});
                Tensor grad_buf = at::empty({batch_size, num_classes});

                at::native::hot::cross_entropy_fused(
                    input_c.data_ptr<float>(), targets_i64.data(),
                    output.mutable_data_ptr<float>(), grad_buf.mutable_data_ptr<float>(),
                    batch_size, num_classes);

                // Wire autograd: backward just returns the pre-computed gradient
                if (torch::autograd::compute_requires_grad(input)) {
                    auto grad_fn = torch::autograd::NodePool<torch::autograd::PrecomputedGradBackward>::make_shared(grad_buf);
                    grad_fn->add_input_metadata(input);
                    torch::autograd::set_grad_fn(output, grad_fn);
                    output.set_requires_grad(true);
                }

                return output;
            }
            // Fall through to generic CPU path if targets invalid
        }

        // CPU implementation (generic path)
        Tensor input_cpu = input;
        Tensor target_cpu = target;
        Tensor weight_cpu = weight_;

        int64_t batch_size = input_cpu.size(0);
        int64_t num_classes = input_cpu.size(1);
        int64_t spatial_size = 1;

        for (int64_t d = 2; d < input_cpu.dim(); ++d) {
            spatial_size *= input_cpu.size(d);
        }

        const float* input_data = input_cpu.data_ptr<float>();
        const float* target_data = target_cpu.data_ptr<float>();
        const float* weight_data = has_weight_ ? weight_cpu.data_ptr<float>() : nullptr;

        int64_t total_elements = batch_size * spatial_size;

        // Check if target is soft labels (same shape as input)
        bool soft_labels = (target_cpu.dim() == input_cpu.dim() && target_cpu.size(1) == num_classes);

        if (reduction_ == Reduction::None) {
            std::vector<int64_t> output_shape;
            if (soft_labels) {
                output_shape.push_back(batch_size);
                for (int64_t d = 2; d < input_cpu.dim(); ++d) {
                    output_shape.push_back(input_cpu.size(d));
                }
            } else {
                output_shape = target_cpu.sizes().vec();
            }
            Tensor output = at::empty(output_shape);
            float* output_data = output.mutable_data_ptr<float>();

            for (int64_t i = 0; i < total_elements; ++i) {
                int64_t b = i / spatial_size;
                int64_t s = i % spatial_size;

                // Compute log-softmax for this position
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t c = 0; c < num_classes; ++c) {
                    int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                    max_val = std::max(max_val, input_data[idx]);
                }

                double sum_exp = 0.0;
                for (int64_t c = 0; c < num_classes; ++c) {
                    int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                    sum_exp += std::exp(input_data[idx] - max_val);
                }
                float log_sum_exp = max_val + static_cast<float>(std::log(sum_exp));

                if (soft_labels) {
                    // Soft labels: L = -sum(target * log_softmax)
                    float loss = 0.0f;
                    for (int64_t c = 0; c < num_classes; ++c) {
                        int64_t input_idx = b * num_classes * spatial_size + c * spatial_size + s;
                        int64_t target_idx = b * num_classes * spatial_size + c * spatial_size + s;
                        float log_prob = input_data[input_idx] - log_sum_exp;
                        float target_prob = target_data[target_idx];

                        // Apply label smoothing
                        if (label_smoothing_ > 0) {
                            target_prob = target_prob * (1.0f - static_cast<float>(label_smoothing_)) +
                                         static_cast<float>(label_smoothing_) / num_classes;
                        }

                        loss -= target_prob * log_prob;
                    }
                    output_data[i] = loss;
                } else {
                    // Hard labels
                    int64_t class_idx = static_cast<int64_t>(target_data[i]);

                    if (class_idx == ignore_index_) {
                        output_data[i] = 0.0f;
                        continue;
                    }

                    int64_t input_idx = b * num_classes * spatial_size + class_idx * spatial_size + s;
                    float log_prob = input_data[input_idx] - log_sum_exp;

                    float loss;
                    if (label_smoothing_ > 0) {
                        // With label smoothing: smooth_loss = (1 - ls) * ce + ls * mean(-log_prob)
                        float mean_log_prob = 0.0f;
                        for (int64_t c = 0; c < num_classes; ++c) {
                            int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                            mean_log_prob += input_data[idx] - log_sum_exp;
                        }
                        mean_log_prob /= num_classes;

                        loss = (1.0f - static_cast<float>(label_smoothing_)) * (-log_prob) +
                               static_cast<float>(label_smoothing_) * (-mean_log_prob);
                    } else {
                        loss = -log_prob;
                    }

                    if (weight_data) {
                        loss *= weight_data[class_idx];
                    }

                    output_data[i] = loss;
                }
            }
            return output;
        }

        // Reduction::Sum or Reduction::Mean
        double sum = 0.0;
        double weight_sum = 0.0;
        int64_t count = 0;

        for (int64_t i = 0; i < total_elements; ++i) {
            int64_t b = i / spatial_size;
            int64_t s = i % spatial_size;

            // Compute log-softmax
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < num_classes; ++c) {
                int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                max_val = std::max(max_val, input_data[idx]);
            }

            double sum_exp = 0.0;
            for (int64_t c = 0; c < num_classes; ++c) {
                int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                sum_exp += std::exp(input_data[idx] - max_val);
            }
            float log_sum_exp = max_val + static_cast<float>(std::log(sum_exp));

            float loss;
            float w = 1.0f;

            if (soft_labels) {
                loss = 0.0f;
                for (int64_t c = 0; c < num_classes; ++c) {
                    int64_t input_idx = b * num_classes * spatial_size + c * spatial_size + s;
                    int64_t target_idx = b * num_classes * spatial_size + c * spatial_size + s;
                    float log_prob = input_data[input_idx] - log_sum_exp;
                    float target_prob = target_data[target_idx];

                    if (label_smoothing_ > 0) {
                        target_prob = target_prob * (1.0f - static_cast<float>(label_smoothing_)) +
                                     static_cast<float>(label_smoothing_) / num_classes;
                    }

                    loss -= target_prob * log_prob;
                }
                count++;
            } else {
                int64_t class_idx = static_cast<int64_t>(target_data[i]);

                if (class_idx == ignore_index_) {
                    continue;
                }

                int64_t input_idx = b * num_classes * spatial_size + class_idx * spatial_size + s;
                float log_prob = input_data[input_idx] - log_sum_exp;

                if (label_smoothing_ > 0) {
                    float mean_log_prob = 0.0f;
                    for (int64_t c = 0; c < num_classes; ++c) {
                        int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                        mean_log_prob += input_data[idx] - log_sum_exp;
                    }
                    mean_log_prob /= num_classes;

                    loss = (1.0f - static_cast<float>(label_smoothing_)) * (-log_prob) +
                           static_cast<float>(label_smoothing_) * (-mean_log_prob);
                } else {
                    loss = -log_prob;
                }

                if (weight_data) {
                    w = weight_data[class_idx];
                }
                count++;
            }

            sum += loss * w;
            weight_sum += w;
        }

        if (reduction_ == Reduction::Mean) {
            sum = has_weight_ ? sum / weight_sum : sum / count;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);

        // CRITICAL: Setup autograd for CPU if input requires grad
        if (torch::autograd::compute_requires_grad(input)) {
            // Compute softmax for backward (needed for gradient computation)
            Tensor softmax_cpu = at::empty(input.sizes());
            float* softmax_data = softmax_cpu.mutable_data_ptr<float>();

            for (int64_t i = 0; i < total_elements; ++i) {
                int64_t b = i / spatial_size;
                int64_t s = i % spatial_size;

                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t c = 0; c < num_classes; ++c) {
                    int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                    max_val = std::max(max_val, input_data[idx]);
                }

                double sum_exp = 0.0;
                for (int64_t c = 0; c < num_classes; ++c) {
                    int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                    sum_exp += std::exp(input_data[idx] - max_val);
                }

                for (int64_t c = 0; c < num_classes; ++c) {
                    int64_t idx = b * num_classes * spatial_size + c * spatial_size + s;
                    softmax_data[idx] = static_cast<float>(std::exp(input_data[idx] - max_val) / sum_exp);
                }
            }

            int64_t num_valid = count > 0 ? count : 1;

            auto grad_fn = torch::autograd::NodePool<torch::autograd::CrossEntropyBackward>::make_shared(
                softmax_cpu, target_cpu, ignore_index_, num_classes, num_valid, /*output_cuda=*/false);
            grad_fn->add_input_metadata(input);
            torch::autograd::set_grad_fn(output, grad_fn);
            output.set_requires_grad(true);
        }

        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("CrossEntropyLoss requires both input and target tensors");
    }

    std::string name() const override { return "CrossEntropyLoss"; }
    Reduction reduction() const { return reduction_; }
    int64_t ignore_index() const { return ignore_index_; }
    double label_smoothing() const { return label_smoothing_; }
};

// ============================================================================
// KLDivLoss - Kullback-Leibler Divergence Loss
// ============================================================================
// Measures the Kullback-Leibler divergence between the target and input distributions.
// L = target * (log(target) - input)
// Note: Input is expected to be log-probabilities, target is probabilities

class KLDivLoss : public Module {
private:
    Reduction reduction_;
    bool log_target_;

public:
    explicit KLDivLoss(Reduction reduction = Reduction::Mean, bool log_target = false)
        : reduction_(reduction), log_target_(log_target) {}

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        int64_t numel = input.numel();

        // KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        // If log_target: input is log(Q), target is log(P)
        // Else: input is log(Q), target is P

        auto compute_kl = [this](float log_q, float p_or_log_p) -> float {
            if (log_target_) {
                float log_p = p_or_log_p;
                float p = std::exp(log_p);
                return p * (log_p - log_q);
            } else {
                float p = p_or_log_p;
                if (p > 0) {
                    return p * (std::log(p) - log_q);
                }
                return 0.0f;
            }
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                output_data[i] = compute_kl(input_data[i], target_data[i]);
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            sum += compute_kl(input_data[i], target_data[i]);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("KLDivLoss requires both input and target tensors");
    }

    std::string name() const override { return "KLDivLoss"; }
    Reduction reduction() const { return reduction_; }
    bool log_target() const { return log_target_; }
};

// ============================================================================
// CosineEmbeddingLoss
// ============================================================================
// Measures the loss given input tensors x1, x2 and a Tensor label y with values 1 or -1.
// L = 1 - cos(x1, x2)              if y = 1
// L = max(0, cos(x1, x2) - margin) if y = -1

class CosineEmbeddingLoss : public Module {
private:
    Reduction reduction_;
    double margin_;

public:
    explicit CosineEmbeddingLoss(Reduction reduction = Reduction::Mean, double margin = 0.0)
        : reduction_(reduction), margin_(margin) {
        if (margin_ < -1.0 || margin_ > 1.0) {
            throw std::invalid_argument("margin should be in range [-1, 1]");
        }
    }

    Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& target) {
        if (input1.sizes() != input2.sizes()) {
            throw std::runtime_error("input1 and input2 must have the same shape");
        }

        // Input: (N, D), Target: (N)
        int64_t batch_size = input1.size(0);
        int64_t dim = input1.numel() / batch_size;

        const float* x1_data = input1.data_ptr<float>();
        const float* x2_data = input2.data_ptr<float>();
        const float* y_data = target.data_ptr<float>();

        auto compute_cosine = [](const float* x1, const float* x2, int64_t dim) -> float {
            double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
            for (int64_t j = 0; j < dim; ++j) {
                dot += x1[j] * x2[j];
                norm1 += x1[j] * x1[j];
                norm2 += x2[j] * x2[j];
            }
            double denom = std::sqrt(norm1) * std::sqrt(norm2);
            return denom > 0 ? static_cast<float>(dot / denom) : 0.0f;
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty({batch_size});
            float* output_data = output.mutable_data_ptr<float>();

            for (int64_t i = 0; i < batch_size; ++i) {
                float cos_sim = compute_cosine(x1_data + i * dim, x2_data + i * dim, dim);
                float y = y_data[i];

                if (y > 0) {
                    output_data[i] = 1.0f - cos_sim;
                } else {
                    output_data[i] = std::max(0.0f, cos_sim - static_cast<float>(margin_));
                }
            }
            return output;
        }

        double sum = 0.0;
        for (int64_t i = 0; i < batch_size; ++i) {
            float cos_sim = compute_cosine(x1_data + i * dim, x2_data + i * dim, dim);
            float y = y_data[i];

            if (y > 0) {
                sum += 1.0 - cos_sim;
            } else {
                sum += std::max(0.0, static_cast<double>(cos_sim) - margin_);
            }
        }

        if (reduction_ == Reduction::Mean) {
            sum /= batch_size;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("CosineEmbeddingLoss requires input1, input2, and target tensors");
    }

    std::string name() const override { return "CosineEmbeddingLoss"; }
    Reduction reduction() const { return reduction_; }
    double margin() const { return margin_; }
};

// ============================================================================
// MarginRankingLoss
// ============================================================================
// Creates a criterion that measures the loss given inputs x1, x2 and label y.
// L = max(0, -y * (x1 - x2) + margin)

class MarginRankingLoss : public Module {
private:
    Reduction reduction_;
    double margin_;

public:
    explicit MarginRankingLoss(Reduction reduction = Reduction::Mean, double margin = 0.0)
        : reduction_(reduction), margin_(margin) {}

    Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& target) {
        if (input1.sizes() != input2.sizes() || input1.sizes() != target.sizes()) {
            throw std::runtime_error("input1, input2, and target must have the same shape");
        }

        const float* x1_data = input1.data_ptr<float>();
        const float* x2_data = input2.data_ptr<float>();
        const float* y_data = target.data_ptr<float>();
        int64_t numel = input1.numel();

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input1.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                float loss = -y_data[i] * (x1_data[i] - x2_data[i]) + static_cast<float>(margin_);
                output_data[i] = std::max(0.0f, loss);
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            double loss = -y_data[i] * (x1_data[i] - x2_data[i]) + margin_;
            sum += std::max(0.0, loss);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("MarginRankingLoss requires input1, input2, and target tensors");
    }

    std::string name() const override { return "MarginRankingLoss"; }
    Reduction reduction() const { return reduction_; }
    double margin() const { return margin_; }
};

// ============================================================================
// TripletMarginLoss
// ============================================================================
// Measures the triplet loss given anchor, positive, and negative samples.
// L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
// where d(x, y) = ||x - y||_p

class TripletMarginLoss : public Module {
private:
    Reduction reduction_;
    double margin_;
    double p_;
    double eps_;
    bool swap_;

public:
    explicit TripletMarginLoss(
        Reduction reduction = Reduction::Mean,
        double margin = 1.0,
        double p = 2.0,
        double eps = 1e-6,
        bool swap = false
    ) : reduction_(reduction), margin_(margin), p_(p), eps_(eps), swap_(swap) {}

    Tensor forward(const Tensor& anchor, const Tensor& positive, const Tensor& negative) {
        if (anchor.sizes() != positive.sizes() || anchor.sizes() != negative.sizes()) {
            throw std::runtime_error("anchor, positive, and negative must have the same shape");
        }

        // Input: (N, D)
        int64_t batch_size = anchor.size(0);
        int64_t dim = anchor.numel() / batch_size;

        const float* a_data = anchor.data_ptr<float>();
        const float* p_data = positive.data_ptr<float>();
        const float* n_data = negative.data_ptr<float>();

        auto compute_dist = [this](const float* x, const float* y, int64_t dim) -> float {
            double sum = 0.0;
            for (int64_t j = 0; j < dim; ++j) {
                sum += std::pow(std::abs(x[j] - y[j]), p_);
            }
            return static_cast<float>(std::pow(sum + eps_, 1.0 / p_));
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty({batch_size});
            float* output_data = output.mutable_data_ptr<float>();

            for (int64_t i = 0; i < batch_size; ++i) {
                float d_ap = compute_dist(a_data + i * dim, p_data + i * dim, dim);
                float d_an = compute_dist(a_data + i * dim, n_data + i * dim, dim);

                if (swap_) {
                    float d_pn = compute_dist(p_data + i * dim, n_data + i * dim, dim);
                    d_an = std::min(d_an, d_pn);
                }

                output_data[i] = std::max(0.0f, d_ap - d_an + static_cast<float>(margin_));
            }
            return output;
        }

        double sum = 0.0;
        for (int64_t i = 0; i < batch_size; ++i) {
            float d_ap = compute_dist(a_data + i * dim, p_data + i * dim, dim);
            float d_an = compute_dist(a_data + i * dim, n_data + i * dim, dim);

            if (swap_) {
                float d_pn = compute_dist(p_data + i * dim, n_data + i * dim, dim);
                d_an = std::min(d_an, d_pn);
            }

            sum += std::max(0.0, static_cast<double>(d_ap - d_an) + margin_);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= batch_size;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("TripletMarginLoss requires anchor, positive, and negative tensors");
    }

    std::string name() const override { return "TripletMarginLoss"; }
    Reduction reduction() const { return reduction_; }
    double margin() const { return margin_; }
    double p() const { return p_; }
};

// ============================================================================
// MultiMarginLoss
// ============================================================================
// Creates a criterion that optimizes a multi-class classification hinge loss.
// L = 1/C * sum(max(0, margin - x[y] + x[j])^p) for j != y

class MultiMarginLoss : public Module {
private:
    Reduction reduction_;
    double margin_;
    int64_t p_;
    bool has_weight_;
    Tensor weight_;

public:
    explicit MultiMarginLoss(
        Reduction reduction = Reduction::Mean,
        double margin = 1.0,
        int64_t p = 1,
        const Tensor* weight = nullptr
    ) : reduction_(reduction), margin_(margin), p_(p), has_weight_(weight != nullptr) {
        if (p != 1 && p != 2) {
            throw std::invalid_argument("p must be 1 or 2");
        }
        if (weight) {
            weight_ = *weight;
        }
    }

    Tensor forward(const Tensor& input, const Tensor& target) {
        // Input: (N, C), Target: (N)
        if (input.dim() != 2) {
            throw std::runtime_error("MultiMarginLoss expects 2D input");
        }

        int64_t batch_size = input.size(0);
        int64_t num_classes = input.size(1);

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        const float* weight_data = has_weight_ ? weight_.data_ptr<float>() : nullptr;

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty({batch_size});
            float* output_data = output.mutable_data_ptr<float>();

            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t y = static_cast<int64_t>(target_data[i]);
                float x_y = input_data[i * num_classes + y];
                double loss = 0.0;

                for (int64_t j = 0; j < num_classes; ++j) {
                    if (j == y) continue;

                    float margin_loss = static_cast<float>(margin_) - x_y + input_data[i * num_classes + j];
                    if (margin_loss > 0) {
                        float w = weight_data ? weight_data[y] : 1.0f;
                        if (p_ == 1) {
                            loss += w * margin_loss;
                        } else {
                            loss += w * margin_loss * margin_loss;
                        }
                    }
                }

                output_data[i] = static_cast<float>(loss / num_classes);
            }
            return output;
        }

        double sum = 0.0;
        for (int64_t i = 0; i < batch_size; ++i) {
            int64_t y = static_cast<int64_t>(target_data[i]);
            float x_y = input_data[i * num_classes + y];
            double sample_loss = 0.0;

            for (int64_t j = 0; j < num_classes; ++j) {
                if (j == y) continue;

                float margin_loss = static_cast<float>(margin_) - x_y + input_data[i * num_classes + j];
                if (margin_loss > 0) {
                    float w = weight_data ? weight_data[y] : 1.0f;
                    if (p_ == 1) {
                        sample_loss += w * margin_loss;
                    } else {
                        sample_loss += w * margin_loss * margin_loss;
                    }
                }
            }

            sum += sample_loss / num_classes;
        }

        if (reduction_ == Reduction::Mean) {
            sum /= batch_size;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("MultiMarginLoss requires both input and target tensors");
    }

    std::string name() const override { return "MultiMarginLoss"; }
    Reduction reduction() const { return reduction_; }
    double margin() const { return margin_; }
    int64_t p() const { return p_; }
};

// ============================================================================
// PoissonNLLLoss - Poisson Negative Log Likelihood Loss
// ============================================================================
// Target is Poisson distribution, input is log of expected counts.
// L = exp(input) - target * input

class PoissonNLLLoss : public Module {
private:
    Reduction reduction_;
    bool log_input_;
    bool full_;
    double eps_;

public:
    explicit PoissonNLLLoss(
        Reduction reduction = Reduction::Mean,
        bool log_input = true,
        bool full = false,
        double eps = 1e-8
    ) : reduction_(reduction), log_input_(log_input), full_(full), eps_(eps) {}

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        int64_t numel = input.numel();

        // Stirling approximation for log(target!)
        auto stirling = [this](float t) -> float {
            if (!full_) return 0.0f;
            if (t <= 1) return 0.0f;
            return t * std::log(t) - t + 0.5f * std::log(2.0f * 3.14159265f * t);
        };

        auto compute_loss = [this, &stirling](float x, float t) -> float {
            float loss;
            if (log_input_) {
                loss = std::exp(x) - t * x;
            } else {
                loss = x - t * std::log(x + static_cast<float>(eps_));
            }
            return loss + stirling(t);
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                output_data[i] = compute_loss(input_data[i], target_data[i]);
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            sum += compute_loss(input_data[i], target_data[i]);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("PoissonNLLLoss requires both input and target tensors");
    }

    std::string name() const override { return "PoissonNLLLoss"; }
    Reduction reduction() const { return reduction_; }
    bool log_input() const { return log_input_; }
};

// ============================================================================
// GaussianNLLLoss - Gaussian Negative Log Likelihood Loss
// ============================================================================
// L = 0.5 * (log(var) + (input - target)^2 / var)

class GaussianNLLLoss : public Module {
private:
    Reduction reduction_;
    bool full_;
    double eps_;

public:
    explicit GaussianNLLLoss(
        Reduction reduction = Reduction::Mean,
        bool full = false,
        double eps = 1e-6
    ) : reduction_(reduction), full_(full), eps_(eps) {}

    Tensor forward(const Tensor& input, const Tensor& target, const Tensor& var) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        const float* var_data = var.data_ptr<float>();
        int64_t numel = input.numel();

        // Determine if var is per-element or scalar
        int64_t var_numel = var.numel();
        bool broadcast_var = (var_numel == 1);

        auto compute_loss = [this](float x, float t, float v) -> float {
            float var_clamped = std::max(v, static_cast<float>(eps_));
            float diff = x - t;
            float loss = 0.5f * (std::log(var_clamped) + diff * diff / var_clamped);
            if (full_) {
                loss += 0.5f * std::log(2.0f * 3.14159265f);
            }
            return loss;
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                float v = broadcast_var ? var_data[0] : var_data[i];
                output_data[i] = compute_loss(input_data[i], target_data[i], v);
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            float v = broadcast_var ? var_data[0] : var_data[i];
            sum += compute_loss(input_data[i], target_data[i], v);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("GaussianNLLLoss requires input, target, and var tensors");
    }

    std::string name() const override { return "GaussianNLLLoss"; }
    Reduction reduction() const { return reduction_; }
    bool full() const { return full_; }
};

// ============================================================================
// CTCLoss - Connectionist Temporal Classification Loss
// ============================================================================
// Note: This is a simplified placeholder. Full CTC requires dynamic programming.

class CTCLoss : public Module {
private:
    Reduction reduction_;
    int64_t blank_;
    bool zero_infinity_;

public:
    explicit CTCLoss(
        int64_t blank = 0,
        Reduction reduction = Reduction::Mean,
        bool zero_infinity = false
    ) : reduction_(reduction), blank_(blank), zero_infinity_(zero_infinity) {}

    Tensor forward(
        const Tensor& log_probs,
        const Tensor& targets,
        const Tensor& input_lengths,
        const Tensor& target_lengths
    ) {
        // This would require a full CTC implementation with forward-backward algorithm
        // For now, we provide the interface structure
        throw std::runtime_error(
            "CTCLoss requires a full forward-backward dynamic programming implementation. "
            "This is a complex algorithm - please use a dedicated CTC library for production."
        );
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("CTCLoss requires log_probs, targets, input_lengths, and target_lengths");
    }

    std::string name() const override { return "CTCLoss"; }
    Reduction reduction() const { return reduction_; }
    int64_t blank() const { return blank_; }
};

// ============================================================================
// FocalLoss - For handling class imbalance
// ============================================================================
// L = -alpha * (1 - p_t)^gamma * log(p_t)
// where p_t = p if y = 1, else 1 - p

class FocalLoss : public Module {
private:
    Reduction reduction_;
    double alpha_;
    double gamma_;

public:
    explicit FocalLoss(
        Reduction reduction = Reduction::Mean,
        double alpha = 0.25,
        double gamma = 2.0
    ) : reduction_(reduction), alpha_(alpha), gamma_(gamma) {}

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();
        int64_t numel = input.numel();

        constexpr float eps = 1e-7f;

        auto compute_focal = [this, eps](float p, float y) -> float {
            float p_clamped = std::max(eps, std::min(1.0f - eps, p));
            float p_t = y * p_clamped + (1.0f - y) * (1.0f - p_clamped);
            float alpha_t = y * static_cast<float>(alpha_) + (1.0f - y) * (1.0f - static_cast<float>(alpha_));
            return -alpha_t * std::pow(1.0f - p_t, static_cast<float>(gamma_)) * std::log(p_t);
        };

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty(input.sizes());
            float* output_data = output.mutable_data_ptr<float>();

            // omp removed for LCC compatibility
            for (int64_t i = 0; i < numel; ++i) {
                output_data[i] = compute_focal(input_data[i], target_data[i]);
            }
            return output;
        }

        double sum = 0.0;
        // omp removed for LCC compatibility
        for (int64_t i = 0; i < numel; ++i) {
            sum += compute_focal(input_data[i], target_data[i]);
        }

        if (reduction_ == Reduction::Mean) {
            sum /= numel;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("FocalLoss requires both input and target tensors");
    }

    std::string name() const override { return "FocalLoss"; }
    Reduction reduction() const { return reduction_; }
    double alpha() const { return alpha_; }
    double gamma() const { return gamma_; }
};

// ============================================================================
// DiceLoss - For segmentation tasks
// ============================================================================
// L = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

class DiceLoss : public Module {
private:
    Reduction reduction_;
    double smooth_;

public:
    explicit DiceLoss(Reduction reduction = Reduction::Mean, double smooth = 1.0)
        : reduction_(reduction), smooth_(smooth) {}

    Tensor forward(const Tensor& input, const Tensor& target) {
        if (input.sizes() != target.sizes()) {
            throw std::runtime_error("Input and target must have the same shape");
        }

        // Assuming input and target are flattened per sample
        // Input: (N, *), Target: (N, *)
        int64_t batch_size = input.size(0);
        int64_t spatial_size = input.numel() / batch_size;

        const float* input_data = input.data_ptr<float>();
        const float* target_data = target.data_ptr<float>();

        if (reduction_ == Reduction::None) {
            Tensor output = at::empty({batch_size});
            float* output_data = output.mutable_data_ptr<float>();

            for (int64_t b = 0; b < batch_size; ++b) {
                double intersection = 0.0;
                double sum_input = 0.0;
                double sum_target = 0.0;

                for (int64_t i = 0; i < spatial_size; ++i) {
                    int64_t idx = b * spatial_size + i;
                    intersection += input_data[idx] * target_data[idx];
                    sum_input += input_data[idx];
                    sum_target += target_data[idx];
                }

                float dice = static_cast<float>(
                    (2.0 * intersection + smooth_) / (sum_input + sum_target + smooth_)
                );
                output_data[b] = 1.0f - dice;
            }
            return output;
        }

        double sum = 0.0;
        for (int64_t b = 0; b < batch_size; ++b) {
            double intersection = 0.0;
            double sum_input = 0.0;
            double sum_target = 0.0;

            for (int64_t i = 0; i < spatial_size; ++i) {
                int64_t idx = b * spatial_size + i;
                intersection += input_data[idx] * target_data[idx];
                sum_input += input_data[idx];
                sum_target += target_data[idx];
            }

            double dice = (2.0 * intersection + smooth_) / (sum_input + sum_target + smooth_);
            sum += 1.0 - dice;
        }

        if (reduction_ == Reduction::Mean) {
            sum /= batch_size;
        }

        Tensor output = at::empty({});
        output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
        return output;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("DiceLoss requires both input and target tensors");
    }

    std::string name() const override { return "DiceLoss"; }
    Reduction reduction() const { return reduction_; }
    double smooth() const { return smooth_; }
};

} // namespace nn
} // namespace torch
