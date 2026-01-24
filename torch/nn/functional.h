#pragma once

#include "aten/src/ATen/ATen.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace torch {
namespace nn {
namespace functional {

using at::Tensor;

// ============================================================================
// Activation Functions
// ============================================================================

inline Tensor relu(const Tensor& input, bool inplace = false) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::relu(input);
    }
#endif
    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = std::max(0.0f, in_data[i]);
    }
    return output;
}

inline Tensor relu6(const Tensor& input, bool inplace = false) {
    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = std::min(6.0f, std::max(0.0f, in_data[i]));
    }
    return output;
}

inline Tensor leaky_relu(const Tensor& input, double negative_slope = 0.01, bool inplace = false) {
    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();
    float slope = static_cast<float>(negative_slope);

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = in_data[i] >= 0 ? in_data[i] : slope * in_data[i];
    }
    return output;
}

inline Tensor elu(const Tensor& input, double alpha = 1.0, bool inplace = false) {
    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();
    float a = static_cast<float>(alpha);

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = in_data[i] >= 0 ? in_data[i] : a * (std::exp(in_data[i]) - 1.0f);
    }
    return output;
}

inline Tensor selu(const Tensor& input, bool inplace = false) {
    constexpr float ALPHA = 1.6732632423543772848170429916717f;
    constexpr float SCALE = 1.0507009873554804934193349852946f;

    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = SCALE * (in_data[i] >= 0 ? in_data[i] : ALPHA * (std::exp(in_data[i]) - 1.0f));
    }
    return output;
}

inline Tensor gelu(const Tensor& input, const std::string& approximate = "none") {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::gelu(input);
    }
#endif
    Tensor output = at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    if (approximate == "tanh") {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
            out_data[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    } else {
        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
        }
    }
    return output;
}

inline Tensor sigmoid(const Tensor& input) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::sigmoid(input);
    }
#endif
    Tensor output = at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = 1.0f / (1.0f + std::exp(-in_data[i]));
    }
    return output;
}

inline Tensor tanh(const Tensor& input) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::tanh(input);
    }
#endif
    Tensor output = at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = std::tanh(in_data[i]);
    }
    return output;
}

inline Tensor silu(const Tensor& input, bool inplace = false) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::silu(input);
    }
#endif
    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = in_data[i] / (1.0f + std::exp(-in_data[i]));
    }
    return output;
}

inline Tensor mish(const Tensor& input, bool inplace = false) {
    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        float sp = std::log(1.0f + std::exp(in_data[i]));
        out_data[i] = in_data[i] * std::tanh(sp);
    }
    return output;
}

inline Tensor hardswish(const Tensor& input, bool inplace = false) {
    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        float x = in_data[i];
        if (x <= -3.0f) {
            out_data[i] = 0.0f;
        } else if (x >= 3.0f) {
            out_data[i] = x;
        } else {
            out_data[i] = x * (x + 3.0f) / 6.0f;
        }
    }
    return output;
}

inline Tensor hardsigmoid(const Tensor& input, bool inplace = false) {
    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        float x = in_data[i];
        if (x <= -3.0f) {
            out_data[i] = 0.0f;
        } else if (x >= 3.0f) {
            out_data[i] = 1.0f;
        } else {
            out_data[i] = (x + 3.0f) / 6.0f;
        }
    }
    return output;
}

inline Tensor softplus(const Tensor& input, double beta = 1.0, double threshold = 20.0) {
    Tensor output = at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();
    float b = static_cast<float>(beta);
    float th = static_cast<float>(threshold);

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        float bx = b * in_data[i];
        out_data[i] = bx > th ? in_data[i] : std::log(1.0f + std::exp(bx)) / b;
    }
    return output;
}

inline Tensor softsign(const Tensor& input) {
    Tensor output = at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = in_data[i] / (1.0f + std::abs(in_data[i]));
    }
    return output;
}

// ============================================================================
// Softmax and LogSoftmax
// ============================================================================

inline Tensor softmax(const Tensor& input, int64_t dim = -1) {
    if (dim < 0) dim = input.dim() + dim;

#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::softmax(input, static_cast<int>(dim));
    }
#endif

    std::vector<int64_t> sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    // Calculate strides
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= sizes[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < static_cast<int64_t>(sizes.size()); ++i) inner_size *= sizes[i];

    Tensor output = at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1000)
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t i = 0; i < inner_size; ++i) {
            // Find max
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                max_val = std::max(max_val, in_data[idx]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                float exp_val = std::exp(in_data[idx] - max_val);
                out_data[idx] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                out_data[idx] /= sum;
            }
        }
    }

    return output;
}

inline Tensor log_softmax(const Tensor& input, int64_t dim = -1) {
    if (dim < 0) dim = input.dim() + dim;

    std::vector<int64_t> sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= sizes[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < static_cast<int64_t>(sizes.size()); ++i) inner_size *= sizes[i];

    Tensor output = at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1000)
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t i = 0; i < inner_size; ++i) {
            // Find max
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                max_val = std::max(max_val, in_data[idx]);
            }

            // Compute sum of exp
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                sum += std::exp(in_data[idx] - max_val);
            }

            float log_sum = max_val + std::log(sum);

            // Compute log_softmax
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                out_data[idx] = in_data[idx] - log_sum;
            }
        }
    }

    return output;
}

// ============================================================================
// Dropout
// ============================================================================

inline Tensor dropout(const Tensor& input, double p = 0.5, bool training = true, bool inplace = false) {
    if (!training || p == 0.0) {
        return input;
    }

    if (p == 1.0) {
        return at::zeros(input.sizes());
    }

    static std::mt19937 gen(std::random_device{}());
    std::bernoulli_distribution dist(1.0 - p);

    Tensor output = inplace ? input : at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();
    int64_t n = input.numel();
    float scale = static_cast<float>(1.0 / (1.0 - p));

    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = dist(gen) ? in_data[i] * scale : 0.0f;
    }

    return output;
}

// ============================================================================
// Linear
// ============================================================================

inline Tensor linear(const Tensor& input, const Tensor& weight, const Tensor* bias = nullptr) {
    // input: (*, in_features)
    // weight: (out_features, in_features)
    // output: (*, out_features)

    int64_t in_features = weight.size(1);
    int64_t out_features = weight.size(0);

    std::vector<int64_t> output_shape = input.sizes().vec();
    output_shape.back() = out_features;

    int64_t batch_size = input.numel() / in_features;

    // Reshape input to 2D for matmul: [batch_size, in_features]
    Tensor input_2d = input.view({batch_size, in_features});

    // weight is [out_features, in_features]
    // y = x @ W^T, so we need mm(input, weight.t())
    Tensor weight_t = weight.t();

    // Use matmul which has CUDA dispatch
    Tensor output = at::mm(input_2d, weight_t);

    // Add bias if present
    if (bias) {
        output = at::add(output, *bias);
    }

    // Reshape output back to original batch dimensions
    return output.view(output_shape);
}

// ============================================================================
// Batch Normalization
// ============================================================================

inline Tensor batch_norm(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor* weight = nullptr,
    const Tensor* bias = nullptr,
    bool training = false,
    double momentum = 0.1,
    double eps = 1e-5
) {
    // For simplicity, assume input is NCHW format
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t spatial_size = input.numel() / (batch_size * channels);

    Tensor output = at::empty(input.sizes());

    const float* in_data = input.data_ptr<float>();
    const float* mean_data = running_mean.data_ptr<float>();
    const float* var_data = running_var.data_ptr<float>();
    const float* weight_data = weight ? weight->data_ptr<float>() : nullptr;
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* out_data = output.mutable_data_ptr<float>();

    #pragma omp parallel for if(channels > 16)
    for (int64_t c = 0; c < channels; ++c) {
        float mean = mean_data[c];
        float var = var_data[c];
        float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps));
        float gamma = weight_data ? weight_data[c] : 1.0f;
        float beta = bias_data ? bias_data[c] : 0.0f;

        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t channel_offset = (b * channels + c) * spatial_size;
            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = channel_offset + s;
                out_data[idx] = gamma * (in_data[idx] - mean) * inv_std + beta;
            }
        }
    }

    return output;
}

// ============================================================================
// Layer Normalization
// ============================================================================

inline Tensor layer_norm(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor* weight = nullptr,
    const Tensor* bias = nullptr,
    double eps = 1e-5
) {
    int64_t normalized_size = 1;
    for (int64_t s : normalized_shape) {
        normalized_size *= s;
    }

    int64_t outer_size = input.numel() / normalized_size;

    Tensor output = at::empty(input.sizes());
    const float* in_data = input.data_ptr<float>();
    const float* weight_data = weight ? weight->data_ptr<float>() : nullptr;
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* out_data = output.mutable_data_ptr<float>();

    #pragma omp parallel for if(outer_size > 100)
    for (int64_t o = 0; o < outer_size; ++o) {
        // Compute mean
        float mean = 0.0f;
        for (int64_t i = 0; i < normalized_size; ++i) {
            mean += in_data[o * normalized_size + i];
        }
        mean /= normalized_size;

        // Compute variance
        float var = 0.0f;
        for (int64_t i = 0; i < normalized_size; ++i) {
            float diff = in_data[o * normalized_size + i] - mean;
            var += diff * diff;
        }
        var /= normalized_size;

        float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps));

        // Normalize
        for (int64_t i = 0; i < normalized_size; ++i) {
            float normalized = (in_data[o * normalized_size + i] - mean) * inv_std;
            float gamma = weight_data ? weight_data[i] : 1.0f;
            float beta = bias_data ? bias_data[i] : 0.0f;
            out_data[o * normalized_size + i] = gamma * normalized + beta;
        }
    }

    return output;
}

// ============================================================================
// Pooling Functions
// ============================================================================

inline Tensor max_pool2d(
    const Tensor& input,
    const std::array<int64_t, 2>& kernel_size,
    const std::array<int64_t, 2>& stride,
    const std::array<int64_t, 2>& padding = {0, 0}
) {
    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    int64_t out_height = (height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
    int64_t out_width = (width + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;

    Tensor output = at::empty({batch, channels, out_height, out_width});
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    #pragma omp parallel for collapse(4)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_height; ++oh) {
                for (int64_t ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int64_t kh = 0; kh < kernel_size[0]; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size[1]; ++kw) {
                            int64_t ih = oh * stride[0] - padding[0] + kh;
                            int64_t iw = ow * stride[1] - padding[1] + kw;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int64_t idx = ((b * channels + c) * height + ih) * width + iw;
                                max_val = std::max(max_val, in_data[idx]);
                            }
                        }
                    }

                    int64_t out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    out_data[out_idx] = max_val;
                }
            }
        }
    }

    return output;
}

inline Tensor avg_pool2d(
    const Tensor& input,
    const std::array<int64_t, 2>& kernel_size,
    const std::array<int64_t, 2>& stride,
    const std::array<int64_t, 2>& padding = {0, 0},
    bool count_include_pad = true
) {
    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    int64_t out_height = (height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
    int64_t out_width = (width + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;

    Tensor output = at::empty({batch, channels, out_height, out_width});
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    #pragma omp parallel for collapse(4)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_height; ++oh) {
                for (int64_t ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t kh = 0; kh < kernel_size[0]; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size[1]; ++kw) {
                            int64_t ih = oh * stride[0] - padding[0] + kh;
                            int64_t iw = ow * stride[1] - padding[1] + kw;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int64_t idx = ((b * channels + c) * height + ih) * width + iw;
                                sum += in_data[idx];
                                count++;
                            } else if (count_include_pad) {
                                count++;
                            }
                        }
                    }

                    int64_t out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    out_data[out_idx] = count > 0 ? sum / count : 0.0f;
                }
            }
        }
    }

    return output;
}

inline Tensor adaptive_avg_pool2d(const Tensor& input, const std::array<int64_t, 2>& output_size) {
    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);

    int64_t out_height = output_size[0];
    int64_t out_width = output_size[1];

    Tensor output = at::empty({batch, channels, out_height, out_width});
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    #pragma omp parallel for collapse(4)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_height; ++oh) {
                for (int64_t ow = 0; ow < out_width; ++ow) {
                    int64_t ih_start = oh * in_height / out_height;
                    int64_t ih_end = (oh + 1) * in_height / out_height;
                    int64_t iw_start = ow * in_width / out_width;
                    int64_t iw_end = (ow + 1) * in_width / out_width;

                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t ih = ih_start; ih < ih_end; ++ih) {
                        for (int64_t iw = iw_start; iw < iw_end; ++iw) {
                            int64_t idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                            sum += in_data[idx];
                            count++;
                        }
                    }

                    int64_t out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    out_data[out_idx] = count > 0 ? sum / count : 0.0f;
                }
            }
        }
    }

    return output;
}

// ============================================================================
// Padding Functions
// ============================================================================

inline Tensor pad(
    const Tensor& input,
    const std::vector<int64_t>& padding,
    const std::string& mode = "constant",
    double value = 0.0
) {
    // Padding format: (left, right, top, bottom, front, back, ...)
    // For 2D: (left, right, top, bottom)

    if (input.dim() < 2) {
        throw std::runtime_error("pad requires at least 2D input");
    }

    std::vector<int64_t> new_shape = input.sizes().vec();
    int64_t ndim = input.dim();

    // Apply padding to last dimensions
    size_t num_pad_dims = padding.size() / 2;
    for (size_t i = 0; i < num_pad_dims; ++i) {
        int64_t dim_idx = ndim - 1 - i;
        new_shape[dim_idx] += padding[2 * i] + padding[2 * i + 1];
    }

    Tensor output = at::full(new_shape, static_cast<float>(value));

    // Copy input data to appropriate location
    // This is simplified for 2D case
    if (input.dim() == 4 && padding.size() == 4) {
        int64_t batch = input.size(0);
        int64_t channels = input.size(1);
        int64_t height = input.size(2);
        int64_t width = input.size(3);

        int64_t pad_left = padding[0];
        int64_t pad_top = padding[2];

        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        int64_t new_height = new_shape[2];
        int64_t new_width = new_shape[3];

        #pragma omp parallel for collapse(4)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        int64_t in_idx = ((b * channels + c) * height + h) * width + w;
                        int64_t out_idx = ((b * channels + c) * new_height + (h + pad_top)) * new_width + (w + pad_left);
                        out_data[out_idx] = in_data[in_idx];
                    }
                }
            }
        }
    }

    return output;
}

// ============================================================================
// Embedding
// ============================================================================

inline Tensor embedding(
    const Tensor& input,
    const Tensor& weight,
    int64_t padding_idx = -1,
    double max_norm = -1.0,
    double norm_type = 2.0,
    bool scale_grad_by_freq = false,
    bool sparse = false
) {
    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);

    std::vector<int64_t> output_sizes = input.sizes().vec();
    output_sizes.push_back(embedding_dim);

    Tensor output = at::empty(output_sizes);

    const float* indices_data = input.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    float* output_data = output.mutable_data_ptr<float>();

    int64_t num_indices = input.numel();

    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = static_cast<int64_t>(indices_data[i]);

        if (idx < 0 || idx >= num_embeddings) {
            throw std::out_of_range("Embedding index out of range");
        }

        for (int64_t j = 0; j < embedding_dim; ++j) {
            output_data[i * embedding_dim + j] = weight_data[idx * embedding_dim + j];
        }
    }

    return output;
}

// ============================================================================
// One-Hot Encoding
// ============================================================================

inline Tensor one_hot(const Tensor& input, int64_t num_classes = -1) {
    const float* data = input.data_ptr<float>();
    int64_t numel = input.numel();

    if (num_classes < 0) {
        int64_t max_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            max_idx = std::max(max_idx, static_cast<int64_t>(data[i]));
        }
        num_classes = max_idx + 1;
    }

    std::vector<int64_t> output_sizes = input.sizes().vec();
    output_sizes.push_back(num_classes);

    Tensor output = at::zeros(output_sizes);
    float* out_data = output.mutable_data_ptr<float>();

    for (int64_t i = 0; i < numel; ++i) {
        int64_t idx = static_cast<int64_t>(data[i]);
        if (idx >= 0 && idx < num_classes) {
            out_data[i * num_classes + idx] = 1.0f;
        }
    }

    return output;
}

// ============================================================================
// Loss Functions (Functional Interface)
// ============================================================================

inline Tensor mse_loss(const Tensor& input, const Tensor& target, const std::string& reduction = "mean") {
    if (input.sizes() != target.sizes()) {
        throw std::runtime_error("Input and target must have the same shape");
    }

    const float* in_data = input.data_ptr<float>();
    const float* tgt_data = target.data_ptr<float>();
    int64_t numel = input.numel();

    if (reduction == "none") {
        Tensor output = at::empty(input.sizes());
        float* out_data = output.mutable_data_ptr<float>();

        #pragma omp parallel for if(numel > 10000)
        for (int64_t i = 0; i < numel; ++i) {
            float diff = in_data[i] - tgt_data[i];
            out_data[i] = diff * diff;
        }
        return output;
    }

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) if(numel > 10000)
    for (int64_t i = 0; i < numel; ++i) {
        double diff = in_data[i] - tgt_data[i];
        sum += diff * diff;
    }

    if (reduction == "mean") {
        sum /= numel;
    }

    Tensor output = at::empty({});
    output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
    return output;
}

inline Tensor l1_loss(const Tensor& input, const Tensor& target, const std::string& reduction = "mean") {
    if (input.sizes() != target.sizes()) {
        throw std::runtime_error("Input and target must have the same shape");
    }

    const float* in_data = input.data_ptr<float>();
    const float* tgt_data = target.data_ptr<float>();
    int64_t numel = input.numel();

    if (reduction == "none") {
        Tensor output = at::empty(input.sizes());
        float* out_data = output.mutable_data_ptr<float>();

        #pragma omp parallel for if(numel > 10000)
        for (int64_t i = 0; i < numel; ++i) {
            out_data[i] = std::abs(in_data[i] - tgt_data[i]);
        }
        return output;
    }

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) if(numel > 10000)
    for (int64_t i = 0; i < numel; ++i) {
        sum += std::abs(in_data[i] - tgt_data[i]);
    }

    if (reduction == "mean") {
        sum /= numel;
    }

    Tensor output = at::empty({});
    output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
    return output;
}

inline Tensor cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor* weight = nullptr,
    int64_t ignore_index = -100,
    double label_smoothing = 0.0,
    const std::string& reduction = "mean"
) {
    // Simplified implementation - assumes input is (N, C) and target is (N)
    int64_t batch_size = input.size(0);
    int64_t num_classes = input.size(1);

    const float* in_data = input.data_ptr<float>();
    const float* tgt_data = target.data_ptr<float>();
    const float* w_data = weight ? weight->data_ptr<float>() : nullptr;

    double total_loss = 0.0;
    double weight_sum = 0.0;
    int64_t count = 0;

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t class_idx = static_cast<int64_t>(tgt_data[b]);

        if (class_idx == ignore_index) {
            continue;
        }

        // Log-softmax
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < num_classes; ++c) {
            max_val = std::max(max_val, in_data[b * num_classes + c]);
        }

        double sum_exp = 0.0;
        for (int64_t c = 0; c < num_classes; ++c) {
            sum_exp += std::exp(in_data[b * num_classes + c] - max_val);
        }
        float log_sum_exp = max_val + static_cast<float>(std::log(sum_exp));

        float log_prob = in_data[b * num_classes + class_idx] - log_sum_exp;
        float loss = -log_prob;

        if (label_smoothing > 0) {
            float mean_log_prob = 0.0f;
            for (int64_t c = 0; c < num_classes; ++c) {
                mean_log_prob += in_data[b * num_classes + c] - log_sum_exp;
            }
            mean_log_prob /= num_classes;
            loss = (1.0f - static_cast<float>(label_smoothing)) * loss +
                   static_cast<float>(label_smoothing) * (-mean_log_prob);
        }

        float w = w_data ? w_data[class_idx] : 1.0f;
        total_loss += loss * w;
        weight_sum += w;
        count++;
    }

    if (reduction == "none") {
        // Would need per-sample losses
        throw std::runtime_error("reduction='none' not implemented for cross_entropy");
    }

    if (reduction == "mean") {
        total_loss = w_data ? total_loss / weight_sum : total_loss / count;
    }

    Tensor output = at::empty({});
    output.mutable_data_ptr<float>()[0] = static_cast<float>(total_loss);
    return output;
}

inline Tensor binary_cross_entropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor* weight = nullptr,
    const std::string& reduction = "mean"
) {
    const float* in_data = input.data_ptr<float>();
    const float* tgt_data = target.data_ptr<float>();
    const float* w_data = weight ? weight->data_ptr<float>() : nullptr;
    int64_t numel = input.numel();

    constexpr float eps = 1e-7f;

    if (reduction == "none") {
        Tensor output = at::empty(input.sizes());
        float* out_data = output.mutable_data_ptr<float>();

        for (int64_t i = 0; i < numel; ++i) {
            float p = std::max(eps, std::min(1.0f - eps, in_data[i]));
            float loss = -(tgt_data[i] * std::log(p) + (1.0f - tgt_data[i]) * std::log(1.0f - p));
            if (w_data) loss *= w_data[i];
            out_data[i] = loss;
        }
        return output;
    }

    double sum = 0.0;
    double weight_sum = 0.0;

    for (int64_t i = 0; i < numel; ++i) {
        float p = std::max(eps, std::min(1.0f - eps, in_data[i]));
        float loss = -(tgt_data[i] * std::log(p) + (1.0f - tgt_data[i]) * std::log(1.0f - p));
        float w = w_data ? w_data[i] : 1.0f;
        sum += loss * w;
        weight_sum += w;
    }

    if (reduction == "mean") {
        sum = w_data ? sum / weight_sum : sum / numel;
    }

    Tensor output = at::empty({});
    output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
    return output;
}

inline Tensor nll_loss(
    const Tensor& input,
    const Tensor& target,
    const Tensor* weight = nullptr,
    int64_t ignore_index = -100,
    const std::string& reduction = "mean"
) {
    int64_t batch_size = input.size(0);
    int64_t num_classes = input.size(1);

    const float* in_data = input.data_ptr<float>();
    const float* tgt_data = target.data_ptr<float>();
    const float* w_data = weight ? weight->data_ptr<float>() : nullptr;

    double sum = 0.0;
    double weight_sum = 0.0;
    int64_t count = 0;

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t class_idx = static_cast<int64_t>(tgt_data[b]);

        if (class_idx == ignore_index) continue;

        float loss = -in_data[b * num_classes + class_idx];
        float w = w_data ? w_data[class_idx] : 1.0f;

        sum += loss * w;
        weight_sum += w;
        count++;
    }

    if (reduction == "mean") {
        sum = w_data ? sum / weight_sum : sum / count;
    }

    Tensor output = at::empty({});
    output.mutable_data_ptr<float>()[0] = static_cast<float>(sum);
    return output;
}

} // namespace functional

// Namespace alias for convenience
namespace F = functional;

} // namespace nn
} // namespace torch
