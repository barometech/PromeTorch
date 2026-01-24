#pragma once

#include "../module.h"
#include <random>

namespace torch {
namespace nn {

// ============================================================================
// Dropout - Randomly zeroes elements during training
// ============================================================================
// During training, randomly zeroes some of the elements of the input tensor
// with probability p using samples from a Bernoulli distribution.
// The outputs are scaled by a factor of 1/(1-p) during training.
// This means that during evaluation the module simply computes an identity function.

class Dropout : public Module {
private:
    double p_;
    bool inplace_;
    mutable std::mt19937 gen_;

public:
    explicit Dropout(double p = 0.5, bool inplace = false)
        : p_(p), inplace_(inplace), gen_(std::random_device{}()) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "dropout probability has to be between 0 and 1, but got " + std::to_string(p_)
            );
        }
    }

    Tensor forward(const Tensor& input) override {
        if (!is_training() || p_ == 0.0) {
            return input;
        }

        if (p_ == 1.0) {
            Tensor output = inplace_ ? input : at::zeros(input.sizes());
            if (inplace_) {
                output.zero_();
            }
            return output;
        }

        Tensor output = inplace_ ? input : at::empty(input.sizes());
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();
        int64_t numel = input.numel();

        std::bernoulli_distribution dist(1.0 - p_);
        float scale = static_cast<float>(1.0 / (1.0 - p_));

        for (int64_t i = 0; i < numel; ++i) {
            if (dist(gen_)) {
                out_data[i] = in_data[i] * scale;
            } else {
                out_data[i] = 0.0f;
            }
        }

        return output;
    }

    std::string name() const override { return "Dropout"; }

    double p() const { return p_; }
    bool inplace() const { return inplace_; }
};

// ============================================================================
// Dropout1d - Randomly zeroes entire channels (for 3D input: NCL)
// ============================================================================
// Randomly zero out entire channels (a channel is a 1D feature map).
// Each channel will be zeroed out independently on every forward call
// with probability p using samples from a Bernoulli distribution.

class Dropout1d : public Module {
private:
    double p_;
    bool inplace_;
    mutable std::mt19937 gen_;

public:
    explicit Dropout1d(double p = 0.5, bool inplace = false)
        : p_(p), inplace_(inplace), gen_(std::random_device{}()) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "dropout probability has to be between 0 and 1, but got " + std::to_string(p_)
            );
        }
    }

    Tensor forward(const Tensor& input) override {
        if (input.dim() != 2 && input.dim() != 3) {
            throw std::runtime_error(
                "Dropout1d: Expected 2D or 3D input (NC or NCL), got " +
                std::to_string(input.dim()) + "D"
            );
        }

        if (!is_training() || p_ == 0.0) {
            return input;
        }

        // Treat 2D input as 3D with L=1
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t length = input.dim() == 3 ? input.size(2) : 1;

        Tensor output = inplace_ ? input : at::empty(input.sizes());
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        std::bernoulli_distribution dist(1.0 - p_);
        float scale = static_cast<float>(1.0 / (1.0 - p_));

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t c = 0; c < channels; ++c) {
                bool keep = dist(gen_);
                int64_t channel_offset = (b * channels + c) * length;

                for (int64_t l = 0; l < length; ++l) {
                    if (keep) {
                        out_data[channel_offset + l] = in_data[channel_offset + l] * scale;
                    } else {
                        out_data[channel_offset + l] = 0.0f;
                    }
                }
            }
        }

        return output;
    }

    std::string name() const override { return "Dropout1d"; }

    double p() const { return p_; }
    bool inplace() const { return inplace_; }
};

// ============================================================================
// Dropout2d - Randomly zeroes entire channels (for 4D input: NCHW)
// ============================================================================
// Randomly zero out entire channels (a channel is a 2D feature map).
// Each channel will be zeroed out independently on every forward call
// with probability p using samples from a Bernoulli distribution.
// Usually the input comes from Conv2d modules.

class Dropout2d : public Module {
private:
    double p_;
    bool inplace_;
    mutable std::mt19937 gen_;

public:
    explicit Dropout2d(double p = 0.5, bool inplace = false)
        : p_(p), inplace_(inplace), gen_(std::random_device{}()) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "dropout probability has to be between 0 and 1, but got " + std::to_string(p_)
            );
        }
    }

    Tensor forward(const Tensor& input) override {
        if (input.dim() != 3 && input.dim() != 4) {
            throw std::runtime_error(
                "Dropout2d: Expected 3D or 4D input (NCH or NCHW), got " +
                std::to_string(input.dim()) + "D"
            );
        }

        if (!is_training() || p_ == 0.0) {
            return input;
        }

        // For 3D input: NCH, treat as NCHW with W=1
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t height = input.size(2);
        int64_t width = input.dim() == 4 ? input.size(3) : 1;
        int64_t spatial_size = height * width;

        Tensor output = inplace_ ? input : at::empty(input.sizes());
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        std::bernoulli_distribution dist(1.0 - p_);
        float scale = static_cast<float>(1.0 / (1.0 - p_));

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t c = 0; c < channels; ++c) {
                bool keep = dist(gen_);
                int64_t channel_offset = (b * channels + c) * spatial_size;

                for (int64_t s = 0; s < spatial_size; ++s) {
                    if (keep) {
                        out_data[channel_offset + s] = in_data[channel_offset + s] * scale;
                    } else {
                        out_data[channel_offset + s] = 0.0f;
                    }
                }
            }
        }

        return output;
    }

    std::string name() const override { return "Dropout2d"; }

    double p() const { return p_; }
    bool inplace() const { return inplace_; }
};

// ============================================================================
// Dropout3d - Randomly zeroes entire channels (for 5D input: NCDHW)
// ============================================================================
// Randomly zero out entire channels (a channel is a 3D feature map).
// Each channel will be zeroed out independently on every forward call
// with probability p using samples from a Bernoulli distribution.
// Usually the input comes from Conv3d modules.

class Dropout3d : public Module {
private:
    double p_;
    bool inplace_;
    mutable std::mt19937 gen_;

public:
    explicit Dropout3d(double p = 0.5, bool inplace = false)
        : p_(p), inplace_(inplace), gen_(std::random_device{}()) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "dropout probability has to be between 0 and 1, but got " + std::to_string(p_)
            );
        }
    }

    Tensor forward(const Tensor& input) override {
        if (input.dim() != 4 && input.dim() != 5) {
            throw std::runtime_error(
                "Dropout3d: Expected 4D or 5D input (NCDH or NCDHW), got " +
                std::to_string(input.dim()) + "D"
            );
        }

        if (!is_training() || p_ == 0.0) {
            return input;
        }

        // For 4D input: NCDH, treat as NCDHW with W=1
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t depth = input.size(2);
        int64_t height = input.size(3);
        int64_t width = input.dim() == 5 ? input.size(4) : 1;
        int64_t spatial_size = depth * height * width;

        Tensor output = inplace_ ? input : at::empty(input.sizes());
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        std::bernoulli_distribution dist(1.0 - p_);
        float scale = static_cast<float>(1.0 / (1.0 - p_));

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t c = 0; c < channels; ++c) {
                bool keep = dist(gen_);
                int64_t channel_offset = (b * channels + c) * spatial_size;

                for (int64_t s = 0; s < spatial_size; ++s) {
                    if (keep) {
                        out_data[channel_offset + s] = in_data[channel_offset + s] * scale;
                    } else {
                        out_data[channel_offset + s] = 0.0f;
                    }
                }
            }
        }

        return output;
    }

    std::string name() const override { return "Dropout3d"; }

    double p() const { return p_; }
    bool inplace() const { return inplace_; }
};

// ============================================================================
// AlphaDropout - Alpha Dropout for SELU activations
// ============================================================================
// Applies Alpha Dropout over the input.
// Alpha Dropout is a type of Dropout that maintains the self-normalizing property.
// For an input with zero mean and unit standard deviation, the output of
// Alpha Dropout maintains the original mean and standard deviation of the input.
// Alpha Dropout goes hand-in-hand with SELU activation function.
//
// During training, it randomly sets some elements to the negative saturation value
// -lambda * alpha where:
//   lambda ≈ 1.0507
//   alpha ≈ 1.6733

class AlphaDropout : public Module {
private:
    double p_;
    bool inplace_;
    mutable std::mt19937 gen_;

    // SELU parameters
    static constexpr double ALPHA = 1.6732632423543772848170429916717;
    static constexpr double SCALE = 1.0507009873554804934193349852946;
    static constexpr double ALPHA_PRIME = -ALPHA * SCALE;

public:
    explicit AlphaDropout(double p = 0.5, bool inplace = false)
        : p_(p), inplace_(inplace), gen_(std::random_device{}()) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "dropout probability has to be between 0 and 1, but got " + std::to_string(p_)
            );
        }
    }

    Tensor forward(const Tensor& input) override {
        if (!is_training() || p_ == 0.0) {
            return input;
        }

        // Alpha dropout parameters
        // Maintain self-normalizing properties
        double a = 1.0 / std::sqrt(1.0 - p_ + p_ * ALPHA_PRIME * ALPHA_PRIME);
        double b = -a * p_ * ALPHA_PRIME;

        Tensor output = inplace_ ? input : at::empty(input.sizes());
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();
        int64_t numel = input.numel();

        std::bernoulli_distribution dist(1.0 - p_);

        float scale = static_cast<float>(a);
        float shift = static_cast<float>(b);
        float saturation = static_cast<float>(ALPHA_PRIME);

        for (int64_t i = 0; i < numel; ++i) {
            if (dist(gen_)) {
                out_data[i] = scale * in_data[i] + shift;
            } else {
                out_data[i] = scale * saturation + shift;
            }
        }

        return output;
    }

    std::string name() const override { return "AlphaDropout"; }

    double p() const { return p_; }
    bool inplace() const { return inplace_; }
};

// ============================================================================
// FeatureAlphaDropout - Alpha Dropout that drops entire channels
// ============================================================================
// Randomly masks out entire channels (feature maps) using Alpha Dropout.
// Each channel will be masked independently on every forward call
// with probability p.

class FeatureAlphaDropout : public Module {
private:
    double p_;
    bool inplace_;
    mutable std::mt19937 gen_;

    static constexpr double ALPHA = 1.6732632423543772848170429916717;
    static constexpr double SCALE = 1.0507009873554804934193349852946;
    static constexpr double ALPHA_PRIME = -ALPHA * SCALE;

public:
    explicit FeatureAlphaDropout(double p = 0.5, bool inplace = false)
        : p_(p), inplace_(inplace), gen_(std::random_device{}()) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument(
                "dropout probability has to be between 0 and 1, but got " + std::to_string(p_)
            );
        }
    }

    Tensor forward(const Tensor& input) override {
        if (input.dim() < 2) {
            throw std::runtime_error(
                "FeatureAlphaDropout: Expected at least 2D input, got " +
                std::to_string(input.dim()) + "D"
            );
        }

        if (!is_training() || p_ == 0.0) {
            return input;
        }

        double a = 1.0 / std::sqrt(1.0 - p_ + p_ * ALPHA_PRIME * ALPHA_PRIME);
        double b = -a * p_ * ALPHA_PRIME;

        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t feature_size = input.numel() / (batch_size * channels);

        Tensor output = inplace_ ? input : at::empty(input.sizes());
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        std::bernoulli_distribution dist(1.0 - p_);

        float scale = static_cast<float>(a);
        float shift = static_cast<float>(b);
        float saturation = static_cast<float>(ALPHA_PRIME);

        for (int64_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            for (int64_t c = 0; c < channels; ++c) {
                bool keep = dist(gen_);
                int64_t channel_offset = (b_idx * channels + c) * feature_size;

                for (int64_t f = 0; f < feature_size; ++f) {
                    if (keep) {
                        out_data[channel_offset + f] = scale * in_data[channel_offset + f] + shift;
                    } else {
                        out_data[channel_offset + f] = scale * saturation + shift;
                    }
                }
            }
        }

        return output;
    }

    std::string name() const override { return "FeatureAlphaDropout"; }

    double p() const { return p_; }
    bool inplace() const { return inplace_; }
};

} // namespace nn
} // namespace torch
