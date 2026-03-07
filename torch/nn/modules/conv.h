#pragma once

#include "torch/nn/module.h"
#include "torch/nn/init.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include <array>
#include <cmath>
#include <tuple>

namespace torch {
namespace nn {

// ============================================================================
// Convolution Utilities
// ============================================================================

// Helper to expand single value to tuple
template<size_t N>
std::array<int64_t, N> expand_to_tuple(int64_t value) {
    std::array<int64_t, N> result;
    result.fill(value);
    return result;
}

template<size_t N>
std::array<int64_t, N> expand_to_tuple(const std::vector<int64_t>& values) {
    std::array<int64_t, N> result;
    if (values.size() == 1) {
        result.fill(values[0]);
    } else {
        for (size_t i = 0; i < N && i < values.size(); ++i) {
            result[i] = values[i];
        }
    }
    return result;
}

// Padding mode
enum class PaddingMode {
    Zeros,
    Reflect,
    Replicate,
    Circular
};

// ============================================================================
// Conv1d
// ============================================================================
// Applies a 1D convolution over an input signal.
// Input: (N, C_in, L)
// Output: (N, C_out, L_out)
// where L_out = floor((L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)

class Conv1d : public Module {
public:
    Conv1d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t dilation = 1,
        int64_t groups = 1,
        bool bias = true,
        PaddingMode padding_mode = PaddingMode::Zeros
    )
        : Module("Conv1d")
        , in_channels_(in_channels)
        , out_channels_(out_channels)
        , kernel_size_(kernel_size)
        , stride_(stride)
        , padding_(padding)
        , dilation_(dilation)
        , groups_(groups)
        , has_bias_(bias)
        , padding_mode_(padding_mode)
    {
        if (in_channels % groups != 0) {
            throw std::runtime_error("in_channels must be divisible by groups");
        }
        if (out_channels % groups != 0) {
            throw std::runtime_error("out_channels must be divisible by groups");
        }

        // Weight shape: [out_channels, in_channels/groups, kernel_size]
        Tensor weight = at::empty({out_channels, in_channels / groups, kernel_size});
        register_parameter("weight", Parameter(weight));

        if (has_bias_) {
            Tensor bias_tensor = at::empty({out_channels});
            register_parameter("bias", Parameter(bias_tensor));
        }

        reset_parameters();
    }

    void reset_parameters() override {
        auto* weight = get_parameter("weight");
        if (weight && weight->defined()) {
            init::kaiming_uniform_(weight->data(), std::sqrt(5.0));
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            if (bias && bias->defined()) {
                auto [fan_in, _] = init::calculate_fan_in_and_fan_out(get_parameter("weight")->data());
                double bound = 1.0 / std::sqrt(static_cast<double>(fan_in));
                init::uniform_(bias->data(), -bound, bound);
            }
        }
    }

    Tensor forward(const Tensor& input) override {
        // Input: [N, C_in, L]
        // Weight: [C_out, C_in/groups, K]
        // Output: [N, C_out, L_out]

        int64_t batch_size = input.size(0);
        int64_t in_length = input.size(2);

        int64_t out_length = (in_length + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;

        auto* weight = get_parameter("weight");
        Tensor W = weight->data();

        Tensor output = at::zeros({batch_size, out_channels_, out_length});

        // Simple direct convolution implementation
        const float* input_data = input.data_ptr<float>();
        const float* weight_data = W.data_ptr<float>();
        float* output_data = output.mutable_data_ptr<float>();

        int64_t group_in_channels = in_channels_ / groups_;
        int64_t group_out_channels = out_channels_ / groups_;

        #pragma omp parallel for collapse(3) if(batch_size * out_channels_ > 16)
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t g = 0; g < groups_; ++g) {
                for (int64_t oc = 0; oc < group_out_channels; ++oc) {
                    int64_t out_c = g * group_out_channels + oc;

                    for (int64_t ol = 0; ol < out_length; ++ol) {
                        float sum = 0.0f;

                        for (int64_t ic = 0; ic < group_in_channels; ++ic) {
                            int64_t in_c = g * group_in_channels + ic;

                            for (int64_t k = 0; k < kernel_size_; ++k) {
                                int64_t il = ol * stride_ - padding_ + k * dilation_;

                                if (il >= 0 && il < in_length) {
                                    int64_t in_idx = n * in_channels_ * in_length + in_c * in_length + il;
                                    int64_t w_idx = out_c * group_in_channels * kernel_size_ + ic * kernel_size_ + k;
                                    sum += input_data[in_idx] * weight_data[w_idx];
                                }
                            }
                        }

                        int64_t out_idx = n * out_channels_ * out_length + out_c * out_length + ol;
                        output_data[out_idx] = sum;
                    }
                }
            }
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            Tensor b = bias->data();
            // Add bias to each channel
            for (int64_t n = 0; n < batch_size; ++n) {
                for (int64_t c = 0; c < out_channels_; ++c) {
                    float bias_val = b.data_ptr<float>()[c];
                    for (int64_t l = 0; l < out_length; ++l) {
                        output_data[n * out_channels_ * out_length + c * out_length + l] += bias_val;
                    }
                }
            }
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << in_channels_ << ", " << out_channels_
           << ", kernel_size=" << kernel_size_
           << ", stride=" << stride_;
        if (padding_ != 0) ss << ", padding=" << padding_;
        if (dilation_ != 1) ss << ", dilation=" << dilation_;
        if (groups_ != 1) ss << ", groups=" << groups_;
        if (!has_bias_) ss << ", bias=False";
        return ss.str();
    }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
    bool has_bias_;
    PaddingMode padding_mode_;
};

// ============================================================================
// Conv2d
// ============================================================================
// Applies a 2D convolution over an input signal.
// Input: (N, C_in, H, W)
// Output: (N, C_out, H_out, W_out)

class Conv2d : public Module {
public:
    Conv2d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t dilation = 1,
        int64_t groups = 1,
        bool bias = true,
        PaddingMode padding_mode = PaddingMode::Zeros
    )
        : Conv2d(in_channels, out_channels,
                 {kernel_size, kernel_size},
                 {stride, stride},
                 {padding, padding},
                 {dilation, dilation},
                 groups, bias, padding_mode) {}

    Conv2d(
        int64_t in_channels,
        int64_t out_channels,
        std::array<int64_t, 2> kernel_size,
        std::array<int64_t, 2> stride = {1, 1},
        std::array<int64_t, 2> padding = {0, 0},
        std::array<int64_t, 2> dilation = {1, 1},
        int64_t groups = 1,
        bool bias = true,
        PaddingMode padding_mode = PaddingMode::Zeros
    )
        : Module("Conv2d")
        , in_channels_(in_channels)
        , out_channels_(out_channels)
        , kernel_size_(kernel_size)
        , stride_(stride)
        , padding_(padding)
        , dilation_(dilation)
        , groups_(groups)
        , has_bias_(bias)
        , padding_mode_(padding_mode)
    {
        if (in_channels % groups != 0) {
            throw std::runtime_error("in_channels must be divisible by groups");
        }
        if (out_channels % groups != 0) {
            throw std::runtime_error("out_channels must be divisible by groups");
        }

        // Weight shape: [out_channels, in_channels/groups, kH, kW]
        Tensor weight = at::empty({out_channels, in_channels / groups, kernel_size[0], kernel_size[1]});
        register_parameter("weight", Parameter(weight));

        if (has_bias_) {
            Tensor bias_tensor = at::empty({out_channels});
            register_parameter("bias", Parameter(bias_tensor));
        }

        reset_parameters();
    }

    void reset_parameters() override {
        auto* weight = get_parameter("weight");
        if (weight && weight->defined()) {
            init::kaiming_uniform_(weight->data(), std::sqrt(5.0));
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            if (bias && bias->defined()) {
                auto [fan_in, _] = init::calculate_fan_in_and_fan_out(get_parameter("weight")->data());
                double bound = 1.0 / std::sqrt(static_cast<double>(fan_in));
                init::uniform_(bias->data(), -bound, bound);
            }
        }
    }

    Tensor forward(const Tensor& input) override {
        // Input: [N, C_in, H, W]
        auto* weight_param = get_parameter("weight");
        Tensor W = weight_param->data();

#ifdef PT_USE_CUDA
        // CUDA dispatch
        if (input.is_cuda()) {
            Tensor bias_tensor;
            if (has_bias_) {
                auto* bias_param = get_parameter("bias");
                bias_tensor = bias_param->data();
            }
            return at::cuda_ops::conv2d_forward(
                input, W, bias_tensor,
                static_cast<int>(stride_[0]), static_cast<int>(stride_[1]),
                static_cast<int>(padding_[0]), static_cast<int>(padding_[1]),
                static_cast<int>(dilation_[0]), static_cast<int>(dilation_[1]),
                static_cast<int>(groups_)
            );
        }
#endif

        // CPU implementation
        int64_t batch_size = input.size(0);
        int64_t in_height = input.size(2);
        int64_t in_width = input.size(3);

        int64_t out_height = (in_height + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
        int64_t out_width = (in_width + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;

        Tensor output = at::zeros({batch_size, out_channels_, out_height, out_width});

        const float* input_data = input.data_ptr<float>();
        const float* weight_data = W.data_ptr<float>();
        float* output_data = output.mutable_data_ptr<float>();

        int64_t group_in_channels = in_channels_ / groups_;
        int64_t group_out_channels = out_channels_ / groups_;

        #pragma omp parallel for collapse(2) if(batch_size * out_channels_ > 16)
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t g = 0; g < groups_; ++g) {
                for (int64_t oc = 0; oc < group_out_channels; ++oc) {
                    int64_t out_c = g * group_out_channels + oc;

                    for (int64_t oh = 0; oh < out_height; ++oh) {
                        for (int64_t ow = 0; ow < out_width; ++ow) {
                            float sum = 0.0f;

                            for (int64_t ic = 0; ic < group_in_channels; ++ic) {
                                int64_t in_c = g * group_in_channels + ic;

                                for (int64_t kh = 0; kh < kernel_size_[0]; ++kh) {
                                    for (int64_t kw = 0; kw < kernel_size_[1]; ++kw) {
                                        int64_t ih = oh * stride_[0] - padding_[0] + kh * dilation_[0];
                                        int64_t iw = ow * stride_[1] - padding_[1] + kw * dilation_[1];

                                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                            int64_t in_idx = n * in_channels_ * in_height * in_width +
                                                            in_c * in_height * in_width +
                                                            ih * in_width + iw;
                                            int64_t w_idx = out_c * group_in_channels * kernel_size_[0] * kernel_size_[1] +
                                                           ic * kernel_size_[0] * kernel_size_[1] +
                                                           kh * kernel_size_[1] + kw;
                                            sum += input_data[in_idx] * weight_data[w_idx];
                                        }
                                    }
                                }
                            }

                            int64_t out_idx = n * out_channels_ * out_height * out_width +
                                             out_c * out_height * out_width +
                                             oh * out_width + ow;
                            output_data[out_idx] = sum;
                        }
                    }
                }
            }
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            const float* bias_data = bias->data().data_ptr<float>();

            for (int64_t n = 0; n < batch_size; ++n) {
                for (int64_t c = 0; c < out_channels_; ++c) {
                    float bias_val = bias_data[c];
                    for (int64_t h = 0; h < out_height; ++h) {
                        for (int64_t w = 0; w < out_width; ++w) {
                            int64_t idx = n * out_channels_ * out_height * out_width +
                                         c * out_height * out_width +
                                         h * out_width + w;
                            output_data[idx] += bias_val;
                        }
                    }
                }
            }
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << in_channels_ << ", " << out_channels_
           << ", kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")"
           << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";
        if (padding_[0] != 0 || padding_[1] != 0) {
            ss << ", padding=(" << padding_[0] << ", " << padding_[1] << ")";
        }
        if (dilation_[0] != 1 || dilation_[1] != 1) {
            ss << ", dilation=(" << dilation_[0] << ", " << dilation_[1] << ")";
        }
        if (groups_ != 1) ss << ", groups=" << groups_;
        if (!has_bias_) ss << ", bias=False";
        return ss.str();
    }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
    std::array<int64_t, 2> dilation_;
    int64_t groups_;
    bool has_bias_;
    PaddingMode padding_mode_;
};

// ============================================================================
// Conv3d
// ============================================================================
// Applies a 3D convolution over an input signal.
// Input: (N, C_in, D, H, W)
// Output: (N, C_out, D_out, H_out, W_out)

class Conv3d : public Module {
public:
    Conv3d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t dilation = 1,
        int64_t groups = 1,
        bool bias = true
    )
        : Module("Conv3d")
        , in_channels_(in_channels)
        , out_channels_(out_channels)
        , kernel_size_({kernel_size, kernel_size, kernel_size})
        , stride_({stride, stride, stride})
        , padding_({padding, padding, padding})
        , dilation_({dilation, dilation, dilation})
        , groups_(groups)
        , has_bias_(bias)
    {
        // Weight shape: [out_channels, in_channels/groups, kD, kH, kW]
        Tensor weight = at::empty({out_channels, in_channels / groups, kernel_size, kernel_size, kernel_size});
        register_parameter("weight", Parameter(weight));

        if (has_bias_) {
            Tensor bias_tensor = at::empty({out_channels});
            register_parameter("bias", Parameter(bias_tensor));
        }

        reset_parameters();
    }

    void reset_parameters() override {
        auto* weight = get_parameter("weight");
        if (weight && weight->defined()) {
            init::kaiming_uniform_(weight->data(), std::sqrt(5.0));
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            if (bias && bias->defined()) {
                auto [fan_in, _] = init::calculate_fan_in_and_fan_out(get_parameter("weight")->data());
                double bound = 1.0 / std::sqrt(static_cast<double>(fan_in));
                init::uniform_(bias->data(), -bound, bound);
            }
        }
    }

    Tensor forward(const Tensor& input) override {
        // Simplified 3D convolution - similar structure to Conv2d
        int64_t batch_size = input.size(0);
        int64_t in_depth = input.size(2);
        int64_t in_height = input.size(3);
        int64_t in_width = input.size(4);

        int64_t out_depth = (in_depth + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
        int64_t out_height = (in_height + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;
        int64_t out_width = (in_width + 2 * padding_[2] - dilation_[2] * (kernel_size_[2] - 1) - 1) / stride_[2] + 1;

        Tensor output = at::zeros({batch_size, out_channels_, out_depth, out_height, out_width});

        // Implementation similar to Conv2d but with extra dimension
        // (Simplified for brevity - full implementation would follow same pattern)

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << in_channels_ << ", " << out_channels_
           << ", kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ", " << kernel_size_[2] << ")";
        return ss.str();
    }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    std::array<int64_t, 3> kernel_size_;
    std::array<int64_t, 3> stride_;
    std::array<int64_t, 3> padding_;
    std::array<int64_t, 3> dilation_;
    int64_t groups_;
    bool has_bias_;
};

// ============================================================================
// ConvTranspose2d (Deconvolution / Transposed Convolution)
// ============================================================================

class ConvTranspose2d : public Module {
public:
    ConvTranspose2d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t output_padding = 0,
        int64_t groups = 1,
        bool bias = true,
        int64_t dilation = 1
    )
        : Module("ConvTranspose2d")
        , in_channels_(in_channels)
        , out_channels_(out_channels)
        , kernel_size_({kernel_size, kernel_size})
        , stride_({stride, stride})
        , padding_({padding, padding})
        , output_padding_({output_padding, output_padding})
        , dilation_({dilation, dilation})
        , groups_(groups)
        , has_bias_(bias)
    {
        // Weight shape: [in_channels, out_channels/groups, kH, kW]
        // Note: different from Conv2d
        Tensor weight = at::empty({in_channels, out_channels / groups, kernel_size, kernel_size});
        register_parameter("weight", Parameter(weight));

        if (has_bias_) {
            Tensor bias_tensor = at::empty({out_channels});
            register_parameter("bias", Parameter(bias_tensor));
        }

        reset_parameters();
    }

    void reset_parameters() override {
        auto* weight = get_parameter("weight");
        if (weight && weight->defined()) {
            init::kaiming_uniform_(weight->data(), std::sqrt(5.0));
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            if (bias && bias->defined()) {
                auto [fan_in, _] = init::calculate_fan_in_and_fan_out(get_parameter("weight")->data());
                double bound = 1.0 / std::sqrt(static_cast<double>(fan_in));
                init::uniform_(bias->data(), -bound, bound);
            }
        }
    }

    Tensor forward(const Tensor& input) override {
        // Transposed convolution (deconvolution)
        int64_t batch_size = input.size(0);
        int64_t in_channels = input.size(1);
        int64_t in_height = input.size(2);
        int64_t in_width = input.size(3);

        // Output size calculation for transposed conv
        int64_t out_height = (in_height - 1) * stride_[0] - 2 * padding_[0] +
                            dilation_[0] * (kernel_size_[0] - 1) + output_padding_[0] + 1;
        int64_t out_width = (in_width - 1) * stride_[1] - 2 * padding_[1] +
                           dilation_[1] * (kernel_size_[1] - 1) + output_padding_[1] + 1;

        auto* weight_param = get_parameter("weight");
        PT_CHECK_MSG(weight_param && weight_param->defined(), "ConvTranspose2d: weight not initialized");
        Tensor weight = weight_param->data().contiguous();

        int64_t out_channels_per_group = out_channels_ / groups_;
        int64_t in_channels_per_group = in_channels_ / groups_;

        Tensor output = at::zeros({batch_size, out_channels_, out_height, out_width});
        Tensor inp = input.contiguous();

        const float* in_data = inp.data_ptr<float>();
        const float* w_data = weight.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        int64_t kH = kernel_size_[0];
        int64_t kW = kernel_size_[1];

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t g = 0; g < groups_; ++g) {
                for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                    int64_t c_in = g * in_channels_per_group + ic;
                    for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                        int64_t c_out = g * out_channels_per_group + oc;
                        for (int64_t ih = 0; ih < in_height; ++ih) {
                            for (int64_t iw = 0; iw < in_width; ++iw) {
                                float val = in_data[((n * in_channels + c_in) * in_height + ih) * in_width + iw];
                                for (int64_t kh = 0; kh < kH; ++kh) {
                                    for (int64_t kw = 0; kw < kW; ++kw) {
                                        int64_t oh = ih * stride_[0] - padding_[0] + kh * dilation_[0];
                                        int64_t ow = iw * stride_[1] - padding_[1] + kw * dilation_[1];
                                        if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                                            // weight layout: [in_channels, out_channels/groups, kH, kW]
                                            int64_t w_idx = ((c_in * out_channels_per_group + oc) * kH + kh) * kW + kw;
                                            out_data[((n * out_channels_ + c_out) * out_height + oh) * out_width + ow] += val * w_data[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        if (has_bias_) {
            auto* bias_param = get_parameter("bias");
            if (bias_param && bias_param->defined()) {
                const float* b_data = bias_param->data().data_ptr<float>();
                for (int64_t n = 0; n < batch_size; ++n) {
                    for (int64_t c = 0; c < out_channels_; ++c) {
                        for (int64_t h = 0; h < out_height; ++h) {
                            for (int64_t w = 0; w < out_width; ++w) {
                                out_data[((n * out_channels_ + c) * out_height + h) * out_width + w] += b_data[c];
                            }
                        }
                    }
                }
            }
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << in_channels_ << ", " << out_channels_
           << ", kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")"
           << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";
        return ss.str();
    }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
    std::array<int64_t, 2> output_padding_;
    std::array<int64_t, 2> dilation_;
    int64_t groups_;
    bool has_bias_;
};

} // namespace nn
} // namespace torch
