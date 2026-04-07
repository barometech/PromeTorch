#pragma once

#include "torch/csrc/autograd/node.h"
#include "aten/src/ATen/ATen.h"
#include <array>
#include <vector>
#include <cstring>

namespace torch {
namespace autograd {

using at::Tensor;

// ============================================================================
// Conv2d Backward
// ============================================================================
// Forward: output = conv2d(input, weight, bias)
//   using im2col + GEMM: out[M,N] = W[M,K] * col[K,N]
//   where M = out_channels/groups, K = in_channels/groups * kH * kW, N = OH*OW
//
// Backward:
//   grad_input:  col2im(W^T @ grad_col)
//   grad_weight: grad_col @ input_col^T  (accumulated over batch)
//   grad_bias:   grad.sum(dims={0,2,3})
// ============================================================================

struct Conv2dBackward : public Node {
    Tensor saved_input_;
    Tensor saved_weight_;
    bool has_bias_;
    int64_t in_channels_, out_channels_, groups_;
    std::array<int64_t, 2> kernel_size_, stride_, padding_, dilation_;
    std::array<int64_t, 4> input_shape_;  // [N, C, H, W]

    Conv2dBackward(const Tensor& input, const Tensor& weight, bool has_bias,
                   int64_t in_channels, int64_t out_channels, int64_t groups,
                   std::array<int64_t, 2> kernel_size,
                   std::array<int64_t, 2> stride,
                   std::array<int64_t, 2> padding,
                   std::array<int64_t, 2> dilation)
        : saved_input_(input), saved_weight_(weight), has_bias_(has_bias)
        , in_channels_(in_channels), out_channels_(out_channels), groups_(groups)
        , kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation)
        , input_shape_({input.size(0), input.size(1), input.size(2), input.size(3)})
    {}

    void release_saved_tensors() override {
        saved_input_ = Tensor();
        saved_weight_ = Tensor();
    }

    // col2im: reverse of im2col — accumulate column data back into image
    static void col2im(
        const float* __restrict col,      // [K, OH*OW]
        float* __restrict output,          // [C_per_group, H, W]
        int64_t channels_per_group,
        int64_t in_height, int64_t in_width,
        int64_t kH, int64_t kW,
        int64_t padH, int64_t padW,
        int64_t strH, int64_t strW,
        int64_t dilH, int64_t dilW,
        int64_t out_height, int64_t out_width)
    {
        const int64_t col_width = out_height * out_width;
        for (int64_t c = 0; c < channels_per_group; ++c) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t col_row = (c * kH + kh) * kW + kw;
                    const float* col_ptr = col + col_row * col_width;
                    float* out_c = output + c * in_height * in_width;

                    for (int64_t oh = 0; oh < out_height; ++oh) {
                        int64_t ih = oh * strH - padH + kh * dilH;
                        if (ih < 0 || ih >= in_height) continue;
                        for (int64_t ow = 0; ow < out_width; ++ow) {
                            int64_t iw = ow * strW - padW + kw * dilW;
                            if (iw >= 0 && iw < in_width) {
                                out_c[ih * in_width + iw] += col_ptr[oh * out_width + ow];
                            }
                        }
                    }
                }
            }
        }
    }

    // im2col: same as Conv2d::im2col (duplicate to avoid coupling)
    static void im2col(
        const float* __restrict input,
        float* __restrict col,
        int64_t channels_per_group,
        int64_t in_height, int64_t in_width,
        int64_t kH, int64_t kW,
        int64_t padH, int64_t padW,
        int64_t strH, int64_t strW,
        int64_t dilH, int64_t dilW,
        int64_t out_height, int64_t out_width)
    {
        const int64_t col_width = out_height * out_width;
        for (int64_t c = 0; c < channels_per_group; ++c) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t col_row = (c * kH + kh) * kW + kw;
                    float* col_ptr = col + col_row * col_width;
                    const float* in_c = input + c * in_height * in_width;

                    for (int64_t oh = 0; oh < out_height; ++oh) {
                        int64_t ih = oh * strH - padH + kh * dilH;
                        if (ih < 0 || ih >= in_height) {
                            std::memset(col_ptr + oh * out_width, 0, out_width * sizeof(float));
                        } else {
                            for (int64_t ow = 0; ow < out_width; ++ow) {
                                int64_t iw = ow * strW - padW + kw * dilW;
                                col_ptr[oh * out_width + ow] =
                                    (iw >= 0 && iw < in_width) ? in_c[ih * in_width + iw] : 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];  // [N, C_out, OH, OW]
        if (!grad_output.defined()) {
            return {Tensor(), Tensor(), Tensor()};
        }

        Tensor grad_out = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
        Tensor weight = saved_weight_.is_contiguous() ? saved_weight_ : saved_weight_.contiguous();
        Tensor input = saved_input_.is_contiguous() ? saved_input_ : saved_input_.contiguous();

        int64_t N = input_shape_[0];
        int64_t in_H = input_shape_[2], in_W = input_shape_[3];
        int64_t kH = kernel_size_[0], kW = kernel_size_[1];
        int64_t out_H = grad_out.size(2), out_W = grad_out.size(3);

        int64_t group_in_ch = in_channels_ / groups_;
        int64_t group_out_ch = out_channels_ / groups_;
        int64_t col_height = group_in_ch * kH * kW;  // K
        int64_t col_width = out_H * out_W;            // N (spatial)

        const float* grad_out_data = grad_out.data_ptr<float>();
        const float* weight_data = weight.data_ptr<float>();
        const float* input_data = input.data_ptr<float>();

        // --- grad_input: col2im(W^T @ grad_col) ---
        Tensor grad_input;
        bool need_input_grad = should_compute_output(0);
        if (need_input_grad) {
            grad_input = at::zeros({N, in_channels_, in_H, in_W});
            float* gi_data = grad_input.mutable_data_ptr<float>();
            std::vector<float> col_buf(col_height * col_width);

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t g = 0; g < groups_; ++g) {
                    // grad_col = grad_output for this sample/group: [group_out_ch, col_width]
                    const float* go_ptr = grad_out_data +
                        n * out_channels_ * col_width +
                        g * group_out_ch * col_width;

                    // W for this group: [group_out_ch, col_height] (row-major)
                    const float* w_ptr = weight_data + g * group_out_ch * col_height;

                    // col_buf = W^T @ grad_col  →  [col_height, col_width]
                    // col_buf[k][j] = sum_m W[m][k] * go[m][j]
                    std::memset(col_buf.data(), 0, col_height * col_width * sizeof(float));
                    for (int64_t m = 0; m < group_out_ch; ++m) {
                        const float* w_row = w_ptr + m * col_height;   // W[m, :]
                        const float* go_row = go_ptr + m * col_width;  // go[m, :]
                        for (int64_t k = 0; k < col_height; ++k) {
                            float w_mk = w_row[k];
                            float* cb_row = col_buf.data() + k * col_width;
                            for (int64_t j = 0; j < col_width; ++j) {
                                cb_row[j] += w_mk * go_row[j];
                            }
                        }
                    }

                    // col2im: accumulate col_buf into grad_input
                    float* gi_ptr = gi_data +
                        n * in_channels_ * in_H * in_W +
                        g * group_in_ch * in_H * in_W;

                    col2im(col_buf.data(), gi_ptr,
                           group_in_ch, in_H, in_W, kH, kW,
                           padding_[0], padding_[1],
                           stride_[0], stride_[1],
                           dilation_[0], dilation_[1],
                           out_H, out_W);
                }
            }
        }

        // --- grad_weight: sum over batch of grad_col @ input_col^T ---
        // gw[m][k] += sum_n sum_j go[m][j] * input_col[k][j]
        // where go: [group_out_ch, col_width], input_col: [col_height, col_width]
        Tensor grad_weight;
        bool need_weight_grad = should_compute_output(1);
        if (need_weight_grad) {
            grad_weight = at::zeros({out_channels_, group_in_ch, kH, kW});
            float* gw_data = grad_weight.mutable_data_ptr<float>();
            std::vector<float> input_col(col_height * col_width);

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t g = 0; g < groups_; ++g) {
                    const float* in_ptr = input_data +
                        n * in_channels_ * in_H * in_W +
                        g * group_in_ch * in_H * in_W;

                    im2col(in_ptr, input_col.data(),
                           group_in_ch, in_H, in_W, kH, kW,
                           padding_[0], padding_[1],
                           stride_[0], stride_[1],
                           dilation_[0], dilation_[1],
                           out_H, out_W);

                    const float* go_ptr = grad_out_data +
                        n * out_channels_ * col_width +
                        g * group_out_ch * col_width;

                    float* gw_ptr = gw_data + g * group_out_ch * col_height;

                    // gw[m][k] += sum_j go[m][j] * input_col[k][j]
                    for (int64_t m = 0; m < group_out_ch; ++m) {
                        const float* go_row = go_ptr + m * col_width;
                        float* gw_row = gw_ptr + m * col_height;
                        for (int64_t k = 0; k < col_height; ++k) {
                            const float* ic_row = input_col.data() + k * col_width;
                            float sum = 0.0f;
                            for (int64_t j = 0; j < col_width; ++j) {
                                sum += go_row[j] * ic_row[j];
                            }
                            gw_row[k] += sum;
                        }
                    }
                }
            }
        }

        // --- grad_bias: sum grad_output over N, H, W dims ---
        Tensor grad_bias;
        if (has_bias_ && should_compute_output(2)) {
            grad_bias = at::zeros({out_channels_});
            float* gb_data = grad_bias.mutable_data_ptr<float>();
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < out_channels_; ++c) {
                    const float* go_c = grad_out_data + n * out_channels_ * col_width + c * col_width;
                    float sum = 0.0f;
                    for (int64_t j = 0; j < col_width; ++j) {
                        sum += go_c[j];
                    }
                    gb_data[c] += sum;
                }
            }
        }

        // Release saved tensors
        saved_input_ = Tensor();
        saved_weight_ = Tensor();

        return {grad_input, grad_weight, grad_bias};
    }

    std::string name() const override { return "Conv2dBackward"; }
};

// ============================================================================
// BatchNorm2d Backward
// ============================================================================
// Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Backward (training mode with batch stats):
//   dx = (1/sqrt(var+eps)) * gamma * (dout - mean(dout) - x_hat * mean(dout * x_hat))
//   dgamma = sum(dout * x_hat, dims={0,2,3})
//   dbeta  = sum(dout, dims={0,2,3})
//   where x_hat = (x - mean) / sqrt(var + eps)
// ============================================================================

struct BatchNorm2dBackward : public Node {
    Tensor saved_input_;
    Tensor saved_weight_;  // gamma
    std::vector<float> saved_mean_;
    std::vector<float> saved_var_;
    double eps_;
    bool affine_;

    BatchNorm2dBackward(const Tensor& input, const Tensor& weight,
                        std::vector<float> mean, std::vector<float> var,
                        double eps, bool affine)
        : saved_input_(input), saved_weight_(weight)
        , saved_mean_(std::move(mean)), saved_var_(std::move(var))
        , eps_(eps), affine_(affine)
    {}

    void release_saved_tensors() override {
        saved_input_ = Tensor();
        saved_weight_ = Tensor();
        saved_mean_.clear();
        saved_var_.clear();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];  // [N, C, H, W]
        if (!grad_output.defined()) {
            return {Tensor(), Tensor(), Tensor()};
        }

        Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
        Tensor input = saved_input_.is_contiguous() ? saved_input_ : saved_input_.contiguous();

        int64_t N = input.size(0);
        int64_t C = input.size(1);
        int64_t H = input.size(2);
        int64_t W = input.size(3);
        int64_t spatial = H * W;
        int64_t count = N * spatial;

        const float* go_data = go.data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        const float* gamma = nullptr;
        if (affine_ && saved_weight_.defined()) {
            gamma = saved_weight_.data_ptr<float>();
        }

        // --- grad_input ---
        Tensor grad_input = at::zeros({N, C, H, W});
        float* gi_data = grad_input.mutable_data_ptr<float>();

        // --- grad_weight (gamma) and grad_bias (beta) ---
        Tensor grad_weight, grad_bias;
        float* gw_data = nullptr;
        float* gb_data = nullptr;
        if (affine_) {
            grad_weight = at::zeros({C});
            grad_bias = at::zeros({C});
            gw_data = grad_weight.mutable_data_ptr<float>();
            gb_data = grad_bias.mutable_data_ptr<float>();
        }

        for (int64_t c = 0; c < C; ++c) {
            float mean_c = saved_mean_[c];
            float var_c = saved_var_[c];
            float inv_std = 1.0f / std::sqrt(var_c + static_cast<float>(eps_));
            float g = (affine_ && gamma) ? gamma[c] : 1.0f;

            // Compute: sum_dout, sum_dout_xhat for this channel
            float sum_dout = 0.0f;
            float sum_dout_xhat = 0.0f;

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    float dout = go_data[idx];
                    float x_hat = (in_data[idx] - mean_c) * inv_std;
                    sum_dout += dout;
                    sum_dout_xhat += dout * x_hat;

                    // Accumulate grad_weight and grad_bias
                    if (affine_) {
                        gw_data[c] += dout * x_hat;
                        gb_data[c] += dout;
                    }
                }
            }

            // grad_input = gamma * inv_std * (dout - mean(dout) - x_hat * mean(dout * x_hat))
            float mean_dout = sum_dout / static_cast<float>(count);
            float mean_dout_xhat = sum_dout_xhat / static_cast<float>(count);

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    float dout = go_data[idx];
                    float x_hat = (in_data[idx] - mean_c) * inv_std;
                    gi_data[idx] = g * inv_std * (dout - mean_dout - x_hat * mean_dout_xhat);
                }
            }
        }

        saved_input_ = Tensor();
        saved_weight_ = Tensor();
        saved_mean_.clear();
        saved_var_.clear();

        return {grad_input, grad_weight, grad_bias};
    }

    std::string name() const override { return "BatchNorm2dBackward"; }
};

// ============================================================================
// MaxPool2d Backward
// ============================================================================
// Forward saved argmax indices. Backward scatters gradient to max positions.
// ============================================================================

struct MaxPool2dBackward : public Node {
    Tensor saved_indices_;  // [N, C, OH, OW] — flattened index into input spatial
    int64_t in_height_, in_width_;
    int64_t channels_;
    int64_t batch_size_;

    MaxPool2dBackward(const Tensor& indices, int64_t batch_size, int64_t channels,
                      int64_t in_height, int64_t in_width)
        : saved_indices_(indices), batch_size_(batch_size), channels_(channels)
        , in_height_(in_height), in_width_(in_width)
    {}

    void release_saved_tensors() override {
        saved_indices_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];  // [N, C, OH, OW]
        if (!grad_output.defined()) return {Tensor()};

        Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
        Tensor indices = saved_indices_.is_contiguous() ? saved_indices_ : saved_indices_.contiguous();

        int64_t N = batch_size_;
        int64_t C = channels_;
        int64_t out_H = go.size(2), out_W = go.size(3);
        int64_t out_spatial = out_H * out_W;
        int64_t in_spatial = in_height_ * in_width_;

        Tensor grad_input = at::zeros({N, C, in_height_, in_width_});
        float* gi_data = grad_input.mutable_data_ptr<float>();
        const float* go_data = go.data_ptr<float>();
        const float* idx_data = indices.data_ptr<float>();  // stored as float for compatibility

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t j = 0; j < out_spatial; ++j) {
                    int64_t offset = n * C * out_spatial + c * out_spatial + j;
                    int64_t max_idx = static_cast<int64_t>(idx_data[offset]);
                    float grad_val = go_data[offset];
                    // Scatter gradient to the max position
                    gi_data[n * C * in_spatial + c * in_spatial + max_idx] += grad_val;
                }
            }
        }

        saved_indices_ = Tensor();
        return {grad_input};
    }

    std::string name() const override { return "MaxPool2dBackward"; }
};

// ============================================================================
// AvgPool2d Backward
// ============================================================================
// Backward: distribute gradient evenly across the pooling window
// ============================================================================

struct AvgPool2dBackward : public Node {
    int64_t batch_size_, channels_;
    int64_t in_height_, in_width_;
    std::array<int64_t, 2> kernel_size_, stride_, padding_;
    bool count_include_pad_;

    AvgPool2dBackward(int64_t batch_size, int64_t channels,
                      int64_t in_height, int64_t in_width,
                      std::array<int64_t, 2> kernel_size,
                      std::array<int64_t, 2> stride,
                      std::array<int64_t, 2> padding,
                      bool count_include_pad)
        : batch_size_(batch_size), channels_(channels)
        , in_height_(in_height), in_width_(in_width)
        , kernel_size_(kernel_size), stride_(stride), padding_(padding)
        , count_include_pad_(count_include_pad)
    {}

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];  // [N, C, OH, OW]
        if (!grad_output.defined()) return {Tensor()};

        Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

        int64_t out_H = go.size(2), out_W = go.size(3);
        int64_t pool_size = kernel_size_[0] * kernel_size_[1];

        Tensor grad_input = at::zeros({batch_size_, channels_, in_height_, in_width_});
        float* gi_data = grad_input.mutable_data_ptr<float>();
        const float* go_data = go.data_ptr<float>();

        for (int64_t n = 0; n < batch_size_; ++n) {
            for (int64_t c = 0; c < channels_; ++c) {
                for (int64_t oh = 0; oh < out_H; ++oh) {
                    for (int64_t ow = 0; ow < out_W; ++ow) {
                        float grad_val = go_data[n * channels_ * out_H * out_W +
                                                 c * out_H * out_W +
                                                 oh * out_W + ow];

                        // Count valid positions
                        int64_t count = 0;
                        for (int64_t kh = 0; kh < kernel_size_[0]; ++kh) {
                            for (int64_t kw = 0; kw < kernel_size_[1]; ++kw) {
                                int64_t ih = oh * stride_[0] - padding_[0] + kh;
                                int64_t iw = ow * stride_[1] - padding_[1] + kw;
                                if (ih >= 0 && ih < in_height_ && iw >= 0 && iw < in_width_) {
                                    count++;
                                }
                            }
                        }
                        int64_t divisor = count_include_pad_ ? pool_size : std::max(count, int64_t(1));
                        float distributed = grad_val / static_cast<float>(divisor);

                        // Distribute gradient
                        for (int64_t kh = 0; kh < kernel_size_[0]; ++kh) {
                            for (int64_t kw = 0; kw < kernel_size_[1]; ++kw) {
                                int64_t ih = oh * stride_[0] - padding_[0] + kh;
                                int64_t iw = ow * stride_[1] - padding_[1] + kw;
                                if (ih >= 0 && ih < in_height_ && iw >= 0 && iw < in_width_) {
                                    gi_data[n * channels_ * in_height_ * in_width_ +
                                            c * in_height_ * in_width_ +
                                            ih * in_width_ + iw] += distributed;
                                }
                            }
                        }
                    }
                }
            }
        }

        return {grad_input};
    }

    std::string name() const override { return "AvgPool2dBackward"; }
};

// ============================================================================
// LayerNorm Backward
// ============================================================================
// y = (x - mean) / sqrt(var + eps) * gamma + beta
// grad_input, grad_weight, grad_bias

struct LayerNormBackward : public Node {
    Tensor saved_input_;
    Tensor saved_weight_;  // gamma, may be empty if !elementwise_affine
    int64_t norm_size_;
    double eps_;
    bool elementwise_affine_;

    LayerNormBackward(const Tensor& input, const Tensor& weight,
                      int64_t norm_size, double eps, bool elementwise_affine)
        : saved_input_(input), saved_weight_(weight)
        , norm_size_(norm_size), eps_(eps)
        , elementwise_affine_(elementwise_affine)
    {}

    void release_saved_tensors() override {
        saved_input_ = Tensor();
        saved_weight_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];
        if (!grad_output.defined()) {
            return {Tensor(), Tensor(), Tensor()};
        }

        Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
        Tensor input = saved_input_.is_contiguous() ? saved_input_ : saved_input_.contiguous();

        int64_t total = input.numel();
        int64_t batch_size = total / norm_size_;

        const float* go_data = go.data_ptr<float>();
        const float* in_data = input.data_ptr<float>();
        const float* gamma = nullptr;
        if (elementwise_affine_ && saved_weight_.defined()) {
            gamma = saved_weight_.data_ptr<float>();
        }

        Tensor grad_input = at::zeros(input.sizes().vec());
        float* gi_data = grad_input.mutable_data_ptr<float>();

        Tensor grad_weight, grad_bias;
        float* gw_data = nullptr;
        float* gb_data = nullptr;
        if (elementwise_affine_) {
            grad_weight = at::zeros({norm_size_});
            grad_bias = at::zeros({norm_size_});
            gw_data = grad_weight.mutable_data_ptr<float>();
            gb_data = grad_bias.mutable_data_ptr<float>();
        }

        float inv_norm = 1.0f / static_cast<float>(norm_size_);

        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t offset = b * norm_size_;

            // Recompute mean and variance for this batch element
            float mean = 0.0f;
            for (int64_t i = 0; i < norm_size_; ++i) {
                mean += in_data[offset + i];
            }
            mean *= inv_norm;

            float var = 0.0f;
            for (int64_t i = 0; i < norm_size_; ++i) {
                float diff = in_data[offset + i] - mean;
                var += diff * diff;
            }
            var *= inv_norm;
            float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps_));

            // Accumulate grad_weight, grad_bias and intermediate sums
            float sum_dout_gamma = 0.0f;
            float sum_dout_gamma_xhat = 0.0f;

            for (int64_t i = 0; i < norm_size_; ++i) {
                float dout = go_data[offset + i];
                float x_hat = (in_data[offset + i] - mean) * inv_std;
                float g = (elementwise_affine_ && gamma) ? gamma[i] : 1.0f;

                sum_dout_gamma += dout * g;
                sum_dout_gamma_xhat += dout * g * x_hat;

                if (elementwise_affine_) {
                    gw_data[i] += dout * x_hat;
                    gb_data[i] += dout;
                }
            }

            // grad_input = gamma * inv_std * (dout - mean(dout*gamma) - x_hat * mean(dout*gamma*x_hat))
            float mean_dg = sum_dout_gamma * inv_norm;
            float mean_dg_xhat = sum_dout_gamma_xhat * inv_norm;

            for (int64_t i = 0; i < norm_size_; ++i) {
                float dout = go_data[offset + i];
                float x_hat = (in_data[offset + i] - mean) * inv_std;
                float g = (elementwise_affine_ && gamma) ? gamma[i] : 1.0f;
                gi_data[offset + i] = inv_std * (dout * g - mean_dg - x_hat * mean_dg_xhat);
            }
        }

        saved_input_ = Tensor();
        saved_weight_ = Tensor();

        return {grad_input, grad_weight, grad_bias};
    }

    std::string name() const override { return "LayerNormBackward"; }
};

// ============================================================================
// GroupNorm Backward
// ============================================================================
// y = (x - mean_g) / sqrt(var_g + eps) * gamma + beta
// where mean_g, var_g are computed per (N, group)

struct GroupNormBackward : public Node {
    Tensor saved_input_;
    Tensor saved_weight_;  // gamma [C], may be empty
    int64_t num_groups_;
    double eps_;
    bool affine_;

    GroupNormBackward(const Tensor& input, const Tensor& weight,
                     int64_t num_groups, double eps, bool affine)
        : saved_input_(input), saved_weight_(weight)
        , num_groups_(num_groups), eps_(eps), affine_(affine)
    {}

    void release_saved_tensors() override {
        saved_input_ = Tensor();
        saved_weight_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];
        if (!grad_output.defined()) {
            return {Tensor(), Tensor(), Tensor()};
        }

        Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
        Tensor input = saved_input_.is_contiguous() ? saved_input_ : saved_input_.contiguous();

        int64_t N = input.size(0);
        int64_t C = input.size(1);
        int64_t spatial = input.numel() / (N * C);
        int64_t channels_per_group = C / num_groups_;
        int64_t group_size = channels_per_group * spatial;

        const float* go_data = go.data_ptr<float>();
        const float* in_data = input.data_ptr<float>();
        const float* gamma = nullptr;
        if (affine_ && saved_weight_.defined()) {
            gamma = saved_weight_.data_ptr<float>();
        }

        Tensor grad_input = at::zeros(input.sizes().vec());
        float* gi_data = grad_input.mutable_data_ptr<float>();

        Tensor grad_weight, grad_bias;
        float* gw_data = nullptr;
        float* gb_data = nullptr;
        if (affine_) {
            grad_weight = at::zeros({C});
            grad_bias = at::zeros({C});
            gw_data = grad_weight.mutable_data_ptr<float>();
            gb_data = grad_bias.mutable_data_ptr<float>();
        }

        float inv_group_size = 1.0f / static_cast<float>(group_size);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < num_groups_; ++g) {
                // Recompute mean and variance for this group
                float mean = 0.0f;
                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t ch = g * channels_per_group + c;
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * C * spatial + ch * spatial + s;
                        mean += in_data[idx];
                    }
                }
                mean *= inv_group_size;

                float var = 0.0f;
                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t ch = g * channels_per_group + c;
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * C * spatial + ch * spatial + s;
                        float diff = in_data[idx] - mean;
                        var += diff * diff;
                    }
                }
                var *= inv_group_size;
                float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps_));

                // Intermediate sums
                float sum_dout_gamma = 0.0f;
                float sum_dout_gamma_xhat = 0.0f;

                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t ch = g * channels_per_group + c;
                    float gi = (affine_ && gamma) ? gamma[ch] : 1.0f;
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * C * spatial + ch * spatial + s;
                        float dout = go_data[idx];
                        float x_hat = (in_data[idx] - mean) * inv_std;
                        sum_dout_gamma += dout * gi;
                        sum_dout_gamma_xhat += dout * gi * x_hat;

                        if (affine_) {
                            gw_data[ch] += dout * x_hat;
                            gb_data[ch] += dout;
                        }
                    }
                }

                float mean_dg = sum_dout_gamma * inv_group_size;
                float mean_dg_xhat = sum_dout_gamma_xhat * inv_group_size;

                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t ch = g * channels_per_group + c;
                    float gi = (affine_ && gamma) ? gamma[ch] : 1.0f;
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * C * spatial + ch * spatial + s;
                        float dout = go_data[idx];
                        float x_hat = (in_data[idx] - mean) * inv_std;
                        gi_data[idx] = inv_std * (dout * gi - mean_dg - x_hat * mean_dg_xhat);
                    }
                }
            }
        }

        saved_input_ = Tensor();
        saved_weight_ = Tensor();

        return {grad_input, grad_weight, grad_bias};
    }

    std::string name() const override { return "GroupNormBackward"; }
};

} // namespace autograd
} // namespace torch
