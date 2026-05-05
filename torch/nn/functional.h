#pragma once

#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/tuda/TudaMath.h"
#include "aten/src/ATen/native/Attention.h"
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

    Tensor inp = input.is_contiguous() ? input : input.contiguous();
    std::vector<int64_t> sizes = inp.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= sizes[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < static_cast<int64_t>(sizes.size()); ++i) inner_size *= sizes[i];

    Tensor output = at::empty(inp.sizes());
    const float* in_data = inp.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    if (inner_size == 1) {
        // Fused vectorized softmax for contiguous rows (most common: softmax over last dim)
        namespace tuda = at::native::tuda;  // namespace alias (LCC-strict syntax)
        constexpr int W = tuda::VecF::width;
        for (int64_t o = 0; o < outer_size; ++o) {
            const float* row_in = in_data + o * dim_size;
            float* row_out = out_data + o * dim_size;

            // Pass 1: vectorized max
            float max_val = tuda::vec_max(row_in, dim_size);
            tuda::VecF vmax = tuda::VecF::broadcast(max_val);

            // Pass 2: fused exp(x - max) + sum
            tuda::VecF vsum0 = tuda::VecF::zero(), vsum1 = tuda::VecF::zero();
            int64_t d = 0;
            for (; d + 2*W <= dim_size; d += 2*W) {
                tuda::VecF e0 = tuda::exp_vec(tuda::VecF::load(row_in + d) - vmax);
                tuda::VecF e1 = tuda::exp_vec(tuda::VecF::load(row_in + d + W) - vmax);
                e0.store(row_out + d);
                e1.store(row_out + d + W);
                vsum0 = vsum0 + e0;
                vsum1 = vsum1 + e1;
            }
            for (; d + W <= dim_size; d += W) {
                tuda::VecF e = tuda::exp_vec(tuda::VecF::load(row_in + d) - vmax);
                e.store(row_out + d);
                vsum0 = vsum0 + e;
            }
            float sum = (vsum0 + vsum1).hsum();
            for (; d < dim_size; ++d) {
                float e = std::exp(row_in[d] - max_val);
                row_out[d] = e;
                sum += e;
            }

            // Pass 3: vectorized normalize
            tuda::VecF vinv = tuda::VecF::broadcast(1.0f / sum);
            d = 0;
            for (; d + W <= dim_size; d += W) {
                (tuda::VecF::load(row_out + d) * vinv).store(row_out + d);
            }
            for (; d < dim_size; ++d) row_out[d] /= sum;
        }
    } else {
        // General case: non-contiguous softmax dimension
        for (int64_t o = 0; o < outer_size; ++o) {
            for (int64_t i = 0; i < inner_size; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    max_val = std::max(max_val, in_data[idx]);
                }
                float sum = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    float exp_val = std::exp(in_data[idx] - max_val);
                    out_data[idx] = exp_val;
                    sum += exp_val;
                }
                float inv_sum = 1.0f / sum;
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    out_data[idx] *= inv_sum;
                }
            }
        }
    }

    return output;
}

inline Tensor log_softmax(const Tensor& input, int64_t dim = -1) {
    if (dim < 0) dim = input.dim() + dim;

    Tensor inp = input.is_contiguous() ? input : input.contiguous();
    std::vector<int64_t> sizes = inp.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= sizes[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < static_cast<int64_t>(sizes.size()); ++i) inner_size *= sizes[i];

    Tensor output = at::empty(inp.sizes());
    const float* in_data = inp.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    if (inner_size == 1) {
        // Fused vectorized log_softmax for contiguous rows
        namespace tuda = at::native::tuda;  // namespace alias (LCC-strict syntax)
        constexpr int W = tuda::VecF::width;
        for (int64_t o = 0; o < outer_size; ++o) {
            const float* row_in = in_data + o * dim_size;
            float* row_out = out_data + o * dim_size;

            // Pass 1: vectorized max
            float max_val = tuda::vec_max(row_in, dim_size);
            tuda::VecF vmax = tuda::VecF::broadcast(max_val);

            // Pass 2: fused exp(x - max) + sum
            tuda::VecF vsum0 = tuda::VecF::zero(), vsum1 = tuda::VecF::zero();
            int64_t d = 0;
            for (; d + 2*W <= dim_size; d += 2*W) {
                tuda::VecF e0 = tuda::exp_vec(tuda::VecF::load(row_in + d) - vmax);
                tuda::VecF e1 = tuda::exp_vec(tuda::VecF::load(row_in + d + W) - vmax);
                vsum0 = vsum0 + e0;
                vsum1 = vsum1 + e1;
            }
            for (; d + W <= dim_size; d += W) {
                tuda::VecF e = tuda::exp_vec(tuda::VecF::load(row_in + d) - vmax);
                vsum0 = vsum0 + e;
            }
            float sum = (vsum0 + vsum1).hsum();
            for (; d < dim_size; ++d) sum += std::exp(row_in[d] - max_val);

            float log_sum = max_val + std::log(sum);
            tuda::VecF vlog_sum = tuda::VecF::broadcast(log_sum);

            // Pass 3: log_softmax = x - log_sum_exp
            d = 0;
            for (; d + W <= dim_size; d += W) {
                (tuda::VecF::load(row_in + d) - vlog_sum).store(row_out + d);
            }
            for (; d < dim_size; ++d) row_out[d] = row_in[d] - log_sum;
        }
    } else {
        // General case
        for (int64_t o = 0; o < outer_size; ++o) {
            for (int64_t i = 0; i < inner_size; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    max_val = std::max(max_val, in_data[idx]);
                }
                float sum = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    sum += std::exp(in_data[idx] - max_val);
                }
                float log_sum = max_val + std::log(sum);
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    out_data[idx] = in_data[idx] - log_sum;
                }
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
    Tensor output = at::native::mm(input_2d, weight_t);

    // Add bias if present
    if (bias) {
        output = output + *bias;
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
    // Padding format: pairs from last dim to first: (left, right, top, bottom, ...)
    if (input.dim() < 1) {
        throw std::runtime_error("pad requires at least 1D input");
    }

    int64_t ndim = input.dim();
    size_t num_pad_dims = padding.size() / 2;

    // Compute new shape
    std::vector<int64_t> new_shape = input.sizes().vec();
    for (size_t i = 0; i < num_pad_dims; ++i) {
        int64_t dim_idx = ndim - 1 - static_cast<int64_t>(i);
        if (dim_idx >= 0) {
            new_shape[dim_idx] += padding[2 * i] + padding[2 * i + 1];
        }
    }

    Tensor output = at::full(new_shape, static_cast<float>(value));
    Tensor inp = input.contiguous();

    // Build pad_before array for each dimension
    std::vector<int64_t> pad_before(ndim, 0);
    for (size_t i = 0; i < num_pad_dims; ++i) {
        int64_t dim_idx = ndim - 1 - static_cast<int64_t>(i);
        if (dim_idx >= 0) {
            pad_before[dim_idx] = padding[2 * i];
        }
    }

    int64_t total = inp.numel();
    auto in_sizes = inp.sizes();

    if (mode == "constant") {
        // Copy input to padded position
        const float* in_data = inp.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        for (int64_t idx = 0; idx < total; ++idx) {
            // Convert flat index to multi-dim coords
            int64_t remaining = idx;
            std::vector<int64_t> coords(ndim);
            for (int64_t d = ndim - 1; d >= 0; --d) {
                coords[d] = remaining % in_sizes[d];
                remaining /= in_sizes[d];
            }

            // Shift by padding
            int64_t out_idx = 0;
            int64_t stride = 1;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                out_idx += (coords[d] + pad_before[d]) * stride;
                stride *= new_shape[d];
            }

            out_data[out_idx] = in_data[idx];
        }
    } else if (mode == "reflect") {
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = inp.data_ptr<float>();
        int64_t out_total = output.numel();

        for (int64_t idx = 0; idx < out_total; ++idx) {
            int64_t remaining = idx;
            std::vector<int64_t> out_coords(ndim);
            for (int64_t d = ndim - 1; d >= 0; --d) {
                out_coords[d] = remaining % new_shape[d];
                remaining /= new_shape[d];
            }

            // Map output coord to input coord with reflection
            int64_t in_idx = 0;
            int64_t stride = 1;
            bool valid = true;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t c = out_coords[d] - pad_before[d];
                int64_t s = in_sizes[d];
                // Reflect
                if (c < 0) c = -c;
                if (c >= s) c = 2 * s - 2 - c;
                if (c < 0 || c >= s) { valid = false; break; }
                in_idx += c * stride;
                stride *= in_sizes[d];
            }

            out_data[idx] = valid ? in_data[in_idx] : 0.0f;
        }
    } else if (mode == "replicate") {
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = inp.data_ptr<float>();
        int64_t out_total = output.numel();

        for (int64_t idx = 0; idx < out_total; ++idx) {
            int64_t remaining = idx;
            std::vector<int64_t> out_coords(ndim);
            for (int64_t d = ndim - 1; d >= 0; --d) {
                out_coords[d] = remaining % new_shape[d];
                remaining /= new_shape[d];
            }

            int64_t in_idx = 0;
            int64_t stride = 1;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t c = out_coords[d] - pad_before[d];
                c = std::max((int64_t)0, std::min(c, in_sizes[d] - 1));
                in_idx += c * stride;
                stride *= in_sizes[d];
            }

            out_data[idx] = in_data[in_idx];
        }
    } else if (mode == "circular") {
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = inp.data_ptr<float>();
        int64_t out_total = output.numel();

        for (int64_t idx = 0; idx < out_total; ++idx) {
            int64_t remaining = idx;
            std::vector<int64_t> out_coords(ndim);
            for (int64_t d = ndim - 1; d >= 0; --d) {
                out_coords[d] = remaining % new_shape[d];
                remaining /= new_shape[d];
            }

            int64_t in_idx = 0;
            int64_t stride = 1;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t c = out_coords[d] - pad_before[d];
                c = ((c % in_sizes[d]) + in_sizes[d]) % in_sizes[d];
                in_idx += c * stride;
                stride *= in_sizes[d];
            }

            out_data[idx] = in_data[in_idx];
        }
    } else {
        throw std::runtime_error("pad: unsupported mode '" + mode + "'. Use constant, reflect, replicate, or circular.");
    }

    return output;
}

// ============================================================================
// Unfold (im2col) and Fold (col2im)
// ============================================================================

inline Tensor unfold(const Tensor& input,
                     std::array<int64_t, 2> kernel_size,
                     std::array<int64_t, 2> dilation = {1, 1},
                     std::array<int64_t, 2> padding = {0, 0},
                     std::array<int64_t, 2> stride = {1, 1}) {
    return at::native::unfold_im2col(input, kernel_size, dilation, padding, stride);
}

inline Tensor fold(const Tensor& input,
                   std::array<int64_t, 2> output_size,
                   std::array<int64_t, 2> kernel_size,
                   std::array<int64_t, 2> dilation = {1, 1},
                   std::array<int64_t, 2> padding = {0, 0},
                   std::array<int64_t, 2> stride = {1, 1}) {
    return at::native::fold_col2im(input, output_size, kernel_size, dilation, padding, stride);
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
    // input: [N, C], target: [N] (float holding int class). Supports mean/sum/none.
    int64_t batch_size = input.size(0);
    int64_t num_classes = input.size(1);

    const float* in_data = input.data_ptr<float>();
    const float* tgt_data = target.data_ptr<float>();
    const float* w_data = weight ? weight->data_ptr<float>() : nullptr;

    // Per-sample loss buffer (also used for reduction=none output).
    std::vector<float> per_sample(batch_size, 0.0f);
    std::vector<bool>  valid(batch_size, false);
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
        per_sample[b] = loss * w;
        valid[b] = true;
        total_loss += loss * w;
        weight_sum += w;
        count++;
    }

    if (reduction == "none") {
        Tensor out = at::empty({batch_size}, at::TensorOptions().dtype(c10::ScalarType::Float));
        float* od = out.mutable_data_ptr<float>();
        for (int64_t b = 0; b < batch_size; ++b) od[b] = valid[b] ? per_sample[b] : 0.0f;
        return out;
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

// ============================================================================
// Interpolate — Resize tensor (nearest or bilinear)
// ============================================================================
// Input: (N, C, H, W) or (C, H, W) or (H, W)
// Output: resized to target size or scaled by scale_factor

inline Tensor interpolate(const Tensor& input,
                           std::vector<int64_t> size = {},
                           std::vector<double> scale_factor = {},
                           const std::string& mode = "nearest",
                           bool align_corners = false) {
    int64_t ndim = input.dim();
    PT_CHECK_MSG(ndim >= 2, "interpolate: input must have at least 2 dimensions");

    // Determine spatial dimensions (last 2)
    int64_t in_h = input.size(ndim - 2);
    int64_t in_w = input.size(ndim - 1);

    int64_t out_h, out_w;

    if (!size.empty()) {
        PT_CHECK(size.size() == 2);
        out_h = size[0];
        out_w = size[1];
    } else if (!scale_factor.empty()) {
        if (scale_factor.size() == 1) {
            out_h = static_cast<int64_t>(in_h * scale_factor[0]);
            out_w = static_cast<int64_t>(in_w * scale_factor[0]);
        } else {
            PT_CHECK(scale_factor.size() == 2);
            out_h = static_cast<int64_t>(in_h * scale_factor[0]);
            out_w = static_cast<int64_t>(in_w * scale_factor[1]);
        }
    } else {
        PT_CHECK_MSG(false, "interpolate: either size or scale_factor must be provided");
        out_h = out_w = 0;
    }

    // Build output shape
    std::vector<int64_t> out_shape = input.sizes().vec();
    out_shape[ndim - 2] = out_h;
    out_shape[ndim - 1] = out_w;

    Tensor result = at::empty(out_shape, at::TensorOptions().dtype(input.dtype()).device(input.device()));
    Tensor in_contig = input.contiguous();

    // Number of outer elements (batch * channels or just channels)
    int64_t outer = 1;
    for (int64_t d = 0; d < ndim - 2; ++d) {
        outer *= input.size(d);
    }

    if (mode == "nearest") {
        PT_DISPATCH_FLOATING_TYPES(input.dtype(), "interpolate_nearest", [&] {
            const scalar_t* src = in_contig.data_ptr<scalar_t>();
            scalar_t* dst = result.mutable_data_ptr<scalar_t>();

            for (int64_t o = 0; o < outer; ++o) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    int64_t ih = static_cast<int64_t>(std::floor(
                        static_cast<double>(oh) * in_h / out_h));
                    ih = std::min(ih, in_h - 1);

                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        int64_t iw = static_cast<int64_t>(std::floor(
                            static_cast<double>(ow) * in_w / out_w));
                        iw = std::min(iw, in_w - 1);

                        dst[o * out_h * out_w + oh * out_w + ow] =
                            src[o * in_h * in_w + ih * in_w + iw];
                    }
                }
            }
        });
    } else if (mode == "bilinear") {
        PT_DISPATCH_FLOATING_TYPES(input.dtype(), "interpolate_bilinear", [&] {
            const scalar_t* src = in_contig.data_ptr<scalar_t>();
            scalar_t* dst = result.mutable_data_ptr<scalar_t>();

            for (int64_t o = 0; o < outer; ++o) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    double src_h;
                    if (align_corners && out_h > 1) {
                        src_h = static_cast<double>(oh) * (in_h - 1) / (out_h - 1);
                    } else {
                        src_h = (static_cast<double>(oh) + 0.5) * in_h / out_h - 0.5;
                    }

                    int64_t h0 = std::max(static_cast<int64_t>(std::floor(src_h)), (int64_t)0);
                    int64_t h1 = std::min(h0 + 1, in_h - 1);
                    double dh = src_h - h0;
                    if (dh < 0) dh = 0;

                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        double src_w;
                        if (align_corners && out_w > 1) {
                            src_w = static_cast<double>(ow) * (in_w - 1) / (out_w - 1);
                        } else {
                            src_w = (static_cast<double>(ow) + 0.5) * in_w / out_w - 0.5;
                        }

                        int64_t w0 = std::max(static_cast<int64_t>(std::floor(src_w)), (int64_t)0);
                        int64_t w1 = std::min(w0 + 1, in_w - 1);
                        double dw = src_w - w0;
                        if (dw < 0) dw = 0;

                        scalar_t v00 = src[o * in_h * in_w + h0 * in_w + w0];
                        scalar_t v01 = src[o * in_h * in_w + h0 * in_w + w1];
                        scalar_t v10 = src[o * in_h * in_w + h1 * in_w + w0];
                        scalar_t v11 = src[o * in_h * in_w + h1 * in_w + w1];

                        dst[o * out_h * out_w + oh * out_w + ow] = static_cast<scalar_t>(
                            (1.0 - dh) * (1.0 - dw) * v00 +
                            (1.0 - dh) * dw * v01 +
                            dh * (1.0 - dw) * v10 +
                            dh * dw * v11
                        );
                    }
                }
            }
        });
    } else {
        PT_CHECK_MSG(false, "interpolate: unsupported mode. Use 'nearest' or 'bilinear'");
    }

    return result;
}

// ============================================================================
// Normalize — L_p normalization along a dimension
// ============================================================================

inline Tensor normalize(const Tensor& input, double p = 2.0, int64_t dim = 1, double eps = 1e-12) {
    Tensor norm_val = input.norm(at::Scalar(p), dim, /*keepdim=*/true);
    // Clamp to avoid division by zero
    Tensor denom = at::native::clamp_min(norm_val, at::Scalar(eps));
    return input / denom;
}

// ============================================================================
// Cosine Similarity — pairwise cosine similarity along a dimension
// ============================================================================

inline Tensor cosine_similarity(const Tensor& x1, const Tensor& x2, int64_t dim = 1, double eps = 1e-8) {
    Tensor dot = (x1 * x2).sum(dim);
    Tensor norm1 = x1.norm(at::Scalar(2.0), dim);
    Tensor norm2 = x2.norm(at::Scalar(2.0), dim);
    Tensor denom = at::native::clamp_min(norm1 * norm2, at::Scalar(eps));
    return dot / denom;
}

// ============================================================================
// Pairwise Distance — Euclidean (or Lp) distance between pairs
// ============================================================================

inline Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p = 2.0, double eps = 1e-6, bool keepdim = false) {
    Tensor diff = x1 - x2;
    return diff.norm(at::Scalar(p), /*dim=*/1, keepdim);
}

// ============================================================================
// Grid Sample — bilinear sampling from a grid of coordinates
// ============================================================================

inline Tensor grid_sample(const Tensor& input, const Tensor& grid,
                          const std::string& mode = "bilinear",
                          const std::string& padding_mode = "zeros",
                          bool align_corners = false) {
    PT_CHECK_MSG(input.dim() == 4, "grid_sample: input must be 4D (N,C,H,W)");
    PT_CHECK_MSG(grid.dim() == 4, "grid_sample: grid must be 4D (N,H_out,W_out,2)");
    PT_CHECK_MSG(grid.size(3) == 2, "grid_sample: last dim of grid must be 2");
    PT_CHECK_MSG(input.size(0) == grid.size(0), "grid_sample: batch size must match");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t in_H = input.size(2);
    int64_t in_W = input.size(3);
    int64_t out_H = grid.size(1);
    int64_t out_W = grid.size(2);

    Tensor output = at::zeros({N, C, out_H, out_W}, at::TensorOptions().dtype(input.dtype()));

    Tensor inp = input.contiguous();
    Tensor grd = grid.contiguous();

    PT_DISPATCH_FLOATING_TYPES(input.dtype(), "grid_sample", [&] {
        const scalar_t* in_data = inp.data_ptr<scalar_t>();
        const scalar_t* g_data = grd.data_ptr<scalar_t>();
        scalar_t* out_data = output.mutable_data_ptr<scalar_t>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t oh = 0; oh < out_H; ++oh) {
                for (int64_t ow = 0; ow < out_W; ++ow) {
                    // Grid values in [-1, 1]
                    scalar_t gx = g_data[n * out_H * out_W * 2 + oh * out_W * 2 + ow * 2 + 0];
                    scalar_t gy = g_data[n * out_H * out_W * 2 + oh * out_W * 2 + ow * 2 + 1];

                    // Convert to pixel coordinates
                    scalar_t ix, iy;
                    if (align_corners) {
                        ix = (gx + 1) * (in_W - 1) / 2;
                        iy = (gy + 1) * (in_H - 1) / 2;
                    } else {
                        ix = ((gx + 1) * in_W - 1) / 2;
                        iy = ((gy + 1) * in_H - 1) / 2;
                    }

                    if (mode == "bilinear") {
                        int64_t ix0 = static_cast<int64_t>(std::floor(ix));
                        int64_t iy0 = static_cast<int64_t>(std::floor(iy));
                        int64_t ix1 = ix0 + 1;
                        int64_t iy1 = iy0 + 1;

                        scalar_t dx = ix - ix0;
                        scalar_t dy = iy - iy0;

                        auto get_val = [&](int64_t c, int64_t h, int64_t w) -> scalar_t {
                            if (padding_mode == "zeros") {
                                if (h < 0 || h >= in_H || w < 0 || w >= in_W) return 0;
                            } else if (padding_mode == "border") {
                                h = std::max(int64_t(0), std::min(h, in_H - 1));
                                w = std::max(int64_t(0), std::min(w, in_W - 1));
                            }
                            return in_data[n * C * in_H * in_W + c * in_H * in_W + h * in_W + w];
                        };

                        for (int64_t c = 0; c < C; ++c) {
                            scalar_t val = (1 - dy) * (1 - dx) * get_val(c, iy0, ix0) +
                                           (1 - dy) * dx       * get_val(c, iy0, ix1) +
                                           dy       * (1 - dx) * get_val(c, iy1, ix0) +
                                           dy       * dx       * get_val(c, iy1, ix1);
                            out_data[n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow] = val;
                        }
                    } else { // nearest
                        int64_t nix = static_cast<int64_t>(std::round(ix));
                        int64_t niy = static_cast<int64_t>(std::round(iy));
                        for (int64_t c = 0; c < C; ++c) {
                            scalar_t val = 0;
                            if (niy >= 0 && niy < in_H && nix >= 0 && nix < in_W) {
                                val = in_data[n * C * in_H * in_W + c * in_H * in_W + niy * in_W + nix];
                            }
                            out_data[n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow] = val;
                        }
                    }
                }
            }
        }
    });

    return output;
}

// ============================================================================
// Affine Grid — generate sampling grid from affine transformation matrix
// ============================================================================

inline Tensor affine_grid(const Tensor& theta, const std::vector<int64_t>& size, bool align_corners = false) {
    PT_CHECK_MSG(theta.dim() == 3 && theta.size(1) == 2 && theta.size(2) == 3,
        "affine_grid: theta must be Nx2x3");
    PT_CHECK_MSG(size.size() == 4, "affine_grid: size must be (N,C,H,W)");

    int64_t N = size[0];
    int64_t H = size[2];
    int64_t W = size[3];

    PT_CHECK_MSG(theta.size(0) == N, "affine_grid: theta batch must match size[0]");

    // Create base grid with normalized coordinates [-1, 1]
    Tensor grid = at::empty({N, H, W, 2}, at::TensorOptions().dtype(theta.dtype()));
    Tensor th = theta.contiguous();

    PT_DISPATCH_FLOATING_TYPES(theta.dtype(), "affine_grid", [&] {
        const scalar_t* t_data = th.data_ptr<scalar_t>();
        scalar_t* g_data = grid.mutable_data_ptr<scalar_t>();

        for (int64_t n = 0; n < N; ++n) {
            const scalar_t* t = t_data + n * 6; // 2x3 matrix
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    scalar_t y, x;
                    if (align_corners) {
                        y = (H > 1) ? (2.0 * h / (H - 1) - 1.0) : 0;
                        x = (W > 1) ? (2.0 * w / (W - 1) - 1.0) : 0;
                    } else {
                        y = (2.0 * (h + 0.5) / H - 1.0);
                        x = (2.0 * (w + 0.5) / W - 1.0);
                    }
                    // [x', y'] = theta @ [x, y, 1]^T
                    scalar_t gx = t[0] * x + t[1] * y + t[2];
                    scalar_t gy = t[3] * x + t[4] * y + t[5];
                    int64_t idx = n * H * W * 2 + h * W * 2 + w * 2;
                    g_data[idx + 0] = gx;
                    g_data[idx + 1] = gy;
                }
            }
        }
    });

    return grid;
}

// ============================================================================
// scaled_dot_product_attention — thin wrapper around at::scaled_dot_product_attention.
// On CUDA-compiled builds with head_dim in {64,128}, delegates to FlashAttention.
// CPU (and any other head_dim): explicit softmax(QK^T / sqrt(d))V reference impl.
// Q/K/V shape: [B, N, H, D].
// ============================================================================
inline at::Tensor scaled_dot_product_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_mask = at::Tensor(),
    float dropout_p = 0.0f,
    bool is_causal = false,
    float scale = -1.0f)
{
    return at::scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale);
}

} // namespace functional

// Namespace alias for convenience
namespace F = functional;

} // namespace nn
} // namespace torch
