#pragma once

#include "torch/nn/module.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/functions/ConvBackward.h"
#include <cmath>

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif

namespace torch {
namespace nn {

// ============================================================================
// BatchNorm1d
// ============================================================================
// Applies Batch Normalization over a 2D or 3D input
// Input: (N, C) or (N, C, L)
// y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

class BatchNorm1d : public Module {
public:
    explicit BatchNorm1d(
        int64_t num_features,
        double eps = 1e-5,
        double momentum = 0.1,
        bool affine = true,
        bool track_running_stats = true
    )
        : Module("BatchNorm1d")
        , num_features_(num_features)
        , eps_(eps)
        , momentum_(momentum)
        , affine_(affine)
        , track_running_stats_(track_running_stats)
    {
        if (affine_) {
            register_parameter("weight", Parameter(at::ones({num_features})));
            register_parameter("bias", Parameter(at::zeros({num_features})));
        }

        if (track_running_stats_) {
            register_buffer("running_mean", Buffer(at::zeros({num_features})));
            register_buffer("running_var", Buffer(at::ones({num_features})));
            register_buffer("num_batches_tracked", Buffer(at::zeros({})));
        }
    }

    Tensor forward(const Tensor& input) override {
        // Input shape: [N, C] or [N, C, L]
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t spatial = input.dim() > 2 ? input.size(2) : 1;

        Tensor output = input.clone();
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        Buffer* running_mean_buf = get_buffer("running_mean");
        Buffer* running_var_buf = get_buffer("running_var");

        std::vector<float> mean(channels, 0.0f);
        std::vector<float> var(channels, 0.0f);

        if (is_training() || !track_running_stats_) {
            // Compute batch statistics
            int64_t count = batch_size * spatial;

            for (int64_t c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int64_t n = 0; n < batch_size; ++n) {
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * channels * spatial + c * spatial + s;
                        sum += in_data[idx];
                    }
                }
                mean[c] = sum / static_cast<float>(count);
            }

            for (int64_t c = 0; c < channels; ++c) {
                float sum_sq = 0.0f;
                for (int64_t n = 0; n < batch_size; ++n) {
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * channels * spatial + c * spatial + s;
                        float diff = in_data[idx] - mean[c];
                        sum_sq += diff * diff;
                    }
                }
                var[c] = sum_sq / static_cast<float>(count);
            }

            // Update running stats
            if (track_running_stats_ && is_training()) {
                float* rm = running_mean_buf->data().mutable_data_ptr<float>();
                float* rv = running_var_buf->data().mutable_data_ptr<float>();
                float mom = static_cast<float>(momentum_);

                for (int64_t c = 0; c < channels; ++c) {
                    rm[c] = (1.0f - mom) * rm[c] + mom * mean[c];
                    // PyTorch stores UNbiased variance in running_var
                    float var_unbiased = var[c] * static_cast<float>(count) / static_cast<float>(count - 1);
                    rv[c] = (1.0f - mom) * rv[c] + mom * var_unbiased;
                }
            }
        } else {
            // Use running statistics
            const float* rm = running_mean_buf->data().data_ptr<float>();
            const float* rv = running_var_buf->data().data_ptr<float>();
            for (int64_t c = 0; c < channels; ++c) {
                mean[c] = rm[c];
                var[c] = rv[c];
            }
        }

        // Normalize
        const float* gamma = affine_ ? get_parameter("weight")->data().data_ptr<float>() : nullptr;
        const float* beta = affine_ ? get_parameter("bias")->data().data_ptr<float>() : nullptr;

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                float inv_std = 1.0f / std::sqrt(var[c] + static_cast<float>(eps_));
                float g = affine_ ? gamma[c] : 1.0f;
                float b = affine_ ? beta[c] : 0.0f;

                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * channels * spatial + c * spatial + s;
                    out_data[idx] = (in_data[idx] - mean[c]) * inv_std * g + b;
                }
            }
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << num_features_ << ", eps=" << eps_ << ", momentum=" << momentum_;
        if (!affine_) ss << ", affine=False";
        if (!track_running_stats_) ss << ", track_running_stats=False";
        return ss.str();
    }

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;
};

// ============================================================================
// BatchNorm2d
// ============================================================================

class BatchNorm2d : public Module {
public:
    explicit BatchNorm2d(
        int64_t num_features,
        double eps = 1e-5,
        double momentum = 0.1,
        bool affine = true,
        bool track_running_stats = true
    )
        : Module("BatchNorm2d")
        , num_features_(num_features)
        , eps_(eps)
        , momentum_(momentum)
        , affine_(affine)
        , track_running_stats_(track_running_stats)
    {
        if (affine_) {
            register_parameter("weight", Parameter(at::ones({num_features})));
            register_parameter("bias", Parameter(at::zeros({num_features})));
        }

        if (track_running_stats_) {
            register_buffer("running_mean", Buffer(at::zeros({num_features})));
            register_buffer("running_var", Buffer(at::ones({num_features})));
        }
    }

    Tensor forward(const Tensor& input) override {
#ifdef PT_USE_CUDA
        // Use CUDA kernel for inference mode on GPU
        if (input.is_cuda() && !is_training() && track_running_stats_) {
            Buffer* rm_buf = get_buffer("running_mean");
            Buffer* rv_buf = get_buffer("running_var");
            Tensor gamma = affine_ ? get_parameter("weight")->data() : at::ones({num_features_});
            Tensor beta = affine_ ? get_parameter("bias")->data() : at::zeros({num_features_});

            // Move gamma, beta, running stats to CUDA if not already
            if (!gamma.is_cuda()) gamma = at::to_cuda(gamma);
            if (!beta.is_cuda()) beta = at::to_cuda(beta);
            Tensor rm = rm_buf->data().is_cuda() ? rm_buf->data() : at::to_cuda(rm_buf->data());
            Tensor rv = rv_buf->data().is_cuda() ? rv_buf->data() : at::to_cuda(rv_buf->data());

            return at::cuda_ops::batch_norm2d_forward(
                input, gamma, beta, rm, rv, static_cast<float>(eps_)
            );
        }
#endif
        // Input: [N, C, H, W]
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t height = input.size(2);
        int64_t width = input.size(3);
        int64_t spatial = height * width;

        Tensor output = input.clone();
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        std::vector<float> mean(channels, 0.0f);
        std::vector<float> var(channels, 0.0f);

        if (is_training() || !track_running_stats_) {
            int64_t count = batch_size * spatial;

            // Compute mean
            for (int64_t c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int64_t n = 0; n < batch_size; ++n) {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            int64_t idx = n * channels * spatial + c * spatial + h * width + w;
                            sum += in_data[idx];
                        }
                    }
                }
                mean[c] = sum / static_cast<float>(count);
            }

            // Compute variance
            for (int64_t c = 0; c < channels; ++c) {
                float sum_sq = 0.0f;
                for (int64_t n = 0; n < batch_size; ++n) {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            int64_t idx = n * channels * spatial + c * spatial + h * width + w;
                            float diff = in_data[idx] - mean[c];
                            sum_sq += diff * diff;
                        }
                    }
                }
                var[c] = sum_sq / static_cast<float>(count);
            }

            // Update running stats
            if (track_running_stats_ && is_training()) {
                Buffer* rm_buf = get_buffer("running_mean");
                Buffer* rv_buf = get_buffer("running_var");
                float* rm = rm_buf->data().mutable_data_ptr<float>();
                float* rv = rv_buf->data().mutable_data_ptr<float>();
                float mom = static_cast<float>(momentum_);

                for (int64_t c = 0; c < channels; ++c) {
                    rm[c] = (1.0f - mom) * rm[c] + mom * mean[c];
                    // PyTorch stores UNbiased variance in running_var
                    float var_unbiased = var[c] * static_cast<float>(count) / static_cast<float>(count - 1);
                    rv[c] = (1.0f - mom) * rv[c] + mom * var_unbiased;
                }
            }
        } else {
            Buffer* rm_buf = get_buffer("running_mean");
            Buffer* rv_buf = get_buffer("running_var");
            const float* rm = rm_buf->data().data_ptr<float>();
            const float* rv = rv_buf->data().data_ptr<float>();
            for (int64_t c = 0; c < channels; ++c) {
                mean[c] = rm[c];
                var[c] = rv[c];
            }
        }

        // Normalize
        const float* gamma = affine_ ? get_parameter("weight")->data().data_ptr<float>() : nullptr;
        const float* beta = affine_ ? get_parameter("bias")->data().data_ptr<float>() : nullptr;

        // omp removed for LCC
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                float inv_std = 1.0f / std::sqrt(var[c] + static_cast<float>(eps_));
                float g = affine_ ? gamma[c] : 1.0f;
                float b = affine_ ? beta[c] : 0.0f;

                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        int64_t idx = n * channels * spatial + c * spatial + h * width + w;
                        out_data[idx] = (in_data[idx] - mean[c]) * inv_std * g + b;
                    }
                }
            }
        }

        // Wire autograd backward
        if (autograd::GradMode::is_enabled() && is_training()) {
            bool needs_grad = input.requires_grad();
            if (affine_) {
                needs_grad = needs_grad ||
                    get_parameter("weight")->data().requires_grad() ||
                    get_parameter("bias")->data().requires_grad();
            }
            if (needs_grad) {
                Tensor weight_tensor = affine_ ? get_parameter("weight")->data() : Tensor();
                auto grad_fn = std::make_shared<autograd::BatchNorm2dBackward>(
                    input, weight_tensor, mean, var, eps_, affine_
                );
                grad_fn->add_input_metadata(input);
                if (affine_) {
                    grad_fn->add_input_metadata(get_parameter("weight")->data());
                    grad_fn->add_input_metadata(get_parameter("bias")->data());
                }
                autograd::set_grad_fn(output, grad_fn);
                output.set_requires_grad(true);
            }
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << num_features_ << ", eps=" << eps_ << ", momentum=" << momentum_;
        return ss.str();
    }

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;
};

// ============================================================================
// LayerNorm
// ============================================================================
// Applies Layer Normalization
// y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
// Normalization is applied across the last D dimensions

class LayerNorm : public Module {
public:
    LayerNorm(
        std::vector<int64_t> normalized_shape,
        double eps = 1e-5,
        bool elementwise_affine = true
    )
        : Module("LayerNorm")
        , normalized_shape_(std::move(normalized_shape))
        , eps_(eps)
        , elementwise_affine_(elementwise_affine)
    {
        if (elementwise_affine_) {
            int64_t numel = 1;
            for (auto s : normalized_shape_) numel *= s;

            register_parameter("weight", Parameter(at::ones(normalized_shape_)));
            register_parameter("bias", Parameter(at::zeros(normalized_shape_)));
        }
    }

    Tensor forward(const Tensor& input) override {
        // Normalize over the last D dimensions
        int64_t norm_dims = static_cast<int64_t>(normalized_shape_.size());
        int64_t batch_dims = input.dim() - norm_dims;

        // Compute number of elements in batch and norm dimensions
        int64_t batch_size = 1;
        for (int64_t i = 0; i < batch_dims; ++i) {
            batch_size *= input.size(i);
        }

        int64_t norm_size = 1;
        for (int64_t i = batch_dims; i < input.dim(); ++i) {
            norm_size *= input.size(i);
        }

        Tensor output = input.clone();
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        const float* gamma = elementwise_affine_ ? get_parameter("weight")->data().data_ptr<float>() : nullptr;
        const float* beta = elementwise_affine_ ? get_parameter("bias")->data().data_ptr<float>() : nullptr;

        // Normalize each batch element
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t offset = b * norm_size;

            // Compute mean
            float sum = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                sum += in_data[offset + i];
            }
            float mean = sum / static_cast<float>(norm_size);

            // Compute variance
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float diff = in_data[offset + i] - mean;
                sum_sq += diff * diff;
            }
            float var = sum_sq / static_cast<float>(norm_size);
            float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps_));

            // Normalize
            for (int64_t i = 0; i < norm_size; ++i) {
                float g = elementwise_affine_ ? gamma[i] : 1.0f;
                float bi = elementwise_affine_ ? beta[i] : 0.0f;
                out_data[offset + i] = (in_data[offset + i] - mean) * inv_std * g + bi;
            }
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < normalized_shape_.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << normalized_shape_[i];
        }
        ss << "], eps=" << eps_;
        if (!elementwise_affine_) ss << ", elementwise_affine=False";
        return ss.str();
    }

private:
    std::vector<int64_t> normalized_shape_;
    double eps_;
    bool elementwise_affine_;
};

// ============================================================================
// GroupNorm
// ============================================================================
// Applies Group Normalization
// Divides channels into groups and normalizes within each group

class GroupNorm : public Module {
public:
    GroupNorm(
        int64_t num_groups,
        int64_t num_channels,
        double eps = 1e-5,
        bool affine = true
    )
        : Module("GroupNorm")
        , num_groups_(num_groups)
        , num_channels_(num_channels)
        , eps_(eps)
        , affine_(affine)
    {
        if (num_channels % num_groups != 0) {
            throw std::runtime_error("num_channels must be divisible by num_groups");
        }

        if (affine_) {
            register_parameter("weight", Parameter(at::ones({num_channels})));
            register_parameter("bias", Parameter(at::zeros({num_channels})));
        }
    }

    Tensor forward(const Tensor& input) override {
        // Input: [N, C, *]
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t spatial = input.numel() / (batch_size * channels);

        int64_t channels_per_group = channels / num_groups_;

        Tensor output = input.clone();
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        const float* gamma = affine_ ? get_parameter("weight")->data().data_ptr<float>() : nullptr;
        const float* beta = affine_ ? get_parameter("bias")->data().data_ptr<float>() : nullptr;

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t g = 0; g < num_groups_; ++g) {
                // Compute mean and variance for this group
                float sum = 0.0f;
                int64_t count = channels_per_group * spatial;

                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t channel = g * channels_per_group + c;
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * channels * spatial + channel * spatial + s;
                        sum += in_data[idx];
                    }
                }
                float mean = sum / static_cast<float>(count);

                float sum_sq = 0.0f;
                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t channel = g * channels_per_group + c;
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * channels * spatial + channel * spatial + s;
                        float diff = in_data[idx] - mean;
                        sum_sq += diff * diff;
                    }
                }
                float var = sum_sq / static_cast<float>(count);
                float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps_));

                // Normalize
                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t channel = g * channels_per_group + c;
                    float gi = affine_ ? gamma[channel] : 1.0f;
                    float bi = affine_ ? beta[channel] : 0.0f;

                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * channels * spatial + channel * spatial + s;
                        out_data[idx] = (in_data[idx] - mean) * inv_std * gi + bi;
                    }
                }
            }
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << num_groups_ << ", " << num_channels_ << ", eps=" << eps_;
        if (!affine_) ss << ", affine=False";
        return ss.str();
    }

private:
    int64_t num_groups_;
    int64_t num_channels_;
    double eps_;
    bool affine_;
};

// ============================================================================
// InstanceNorm2d
// ============================================================================
// Instance Normalization - normalizes each channel of each sample independently

class InstanceNorm2d : public Module {
public:
    explicit InstanceNorm2d(
        int64_t num_features,
        double eps = 1e-5,
        double momentum = 0.1,
        bool affine = false,
        bool track_running_stats = false
    )
        : Module("InstanceNorm2d")
        , num_features_(num_features)
        , eps_(eps)
        , affine_(affine)
    {
        if (affine_) {
            register_parameter("weight", Parameter(at::ones({num_features})));
            register_parameter("bias", Parameter(at::zeros({num_features})));
        }
    }

    Tensor forward(const Tensor& input) override {
        // Input: [N, C, H, W]
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t height = input.size(2);
        int64_t width = input.size(3);
        int64_t spatial = height * width;

        Tensor output = input.clone();
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        const float* gamma = affine_ ? get_parameter("weight")->data().data_ptr<float>() : nullptr;
        const float* beta = affine_ ? get_parameter("bias")->data().data_ptr<float>() : nullptr;

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                // Compute mean for this instance-channel
                float sum = 0.0f;
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * channels * spatial + c * spatial + s;
                    sum += in_data[idx];
                }
                float mean = sum / static_cast<float>(spatial);

                // Compute variance
                float sum_sq = 0.0f;
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * channels * spatial + c * spatial + s;
                    float diff = in_data[idx] - mean;
                    sum_sq += diff * diff;
                }
                float var = sum_sq / static_cast<float>(spatial);
                float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps_));

                // Normalize
                float g = affine_ ? gamma[c] : 1.0f;
                float b = affine_ ? beta[c] : 0.0f;

                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * channels * spatial + c * spatial + s;
                    out_data[idx] = (in_data[idx] - mean) * inv_std * g + b;
                }
            }
        }

        return output;
    }

private:
    int64_t num_features_;
    double eps_;
    bool affine_;
};

} // namespace nn
} // namespace torch
