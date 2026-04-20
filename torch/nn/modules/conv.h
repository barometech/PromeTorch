#pragma once

#include "torch/nn/module.h"
#include "torch/nn/init.h"
#include "torch/amp/autocast.h"
#include "torch/amp/autocast_policy.h"
#include "aten/src/ATen/native/cpu/PromeBLAS.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/functions/ConvBackward.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#ifdef PT_USE_CUDNN
#include "aten/src/ATen/cudnn/CuDNN.h"
#endif
#endif
#include <array>
#include <iostream>
#include <cmath>
#include <tuple>
#include <vector>

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

    // im2col for 1D: unfold patches into column matrix [IC/g * K, OL]
    static void im2col_1d(
        const float* __restrict input,   // [C_per_group, L]
        float* __restrict col,           // [C_per_group * K, OL]
        int64_t channels_per_group,
        int64_t in_length,
        int64_t kernel_size,
        int64_t pad, int64_t stride, int64_t dilation,
        int64_t out_length)
    {
        for (int64_t c = 0; c < channels_per_group; ++c) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t col_row = c * kernel_size + k;
                float* col_ptr = col + col_row * out_length;
                const float* in_c = input + c * in_length;

                for (int64_t ol = 0; ol < out_length; ++ol) {
                    int64_t il = ol * stride - pad + k * dilation;
                    col_ptr[ol] = (il >= 0 && il < in_length) ? in_c[il] : 0.0f;
                }
            }
        }
    }

    Tensor forward(const Tensor& input) override {
        // Input: [N, C_in, L]  Weight: [C_out, C_in/groups, K]
        Tensor inp = input.is_contiguous() ? input : input.contiguous();
        auto* weight = get_parameter("weight");
        Tensor W = weight->data().is_contiguous() ? weight->data() : weight->data().contiguous();

        int64_t batch_size = inp.size(0);
        int64_t in_length = inp.size(2);
        int64_t out_length = (in_length + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;

        int64_t group_in_channels = in_channels_ / groups_;
        int64_t group_out_channels = out_channels_ / groups_;
        int64_t col_height = group_in_channels * kernel_size_;

        Tensor output = at::empty({batch_size, out_channels_, out_length});

        const float* input_data = inp.data_ptr<float>();
        const float* weight_data = W.data_ptr<float>();
        float* output_data = output.mutable_data_ptr<float>();

        std::vector<float> col_buf(col_height * out_length);

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t g = 0; g < groups_; ++g) {
                const float* in_ptr = input_data +
                    n * in_channels_ * in_length +
                    g * group_in_channels * in_length;

                im2col_1d(in_ptr, col_buf.data(),
                         group_in_channels, in_length, kernel_size_,
                         padding_, stride_, dilation_, out_length);

                const float* w_ptr = weight_data + g * group_out_channels * col_height;
                float* out_ptr = output_data +
                    n * out_channels_ * out_length +
                    g * group_out_channels * out_length;

                at::native::hot::sgemm(
                    group_out_channels, col_height, out_length,
                    1.0f, w_ptr, col_height,
                    col_buf.data(), out_length,
                    0.0f, out_ptr, out_length
                );
            }
        }

        if (has_bias_) {
            auto* bias = get_parameter("bias");
            const float* bias_data = bias->data().data_ptr<float>();
            for (int64_t n = 0; n < batch_size; ++n) {
                for (int64_t c = 0; c < out_channels_; ++c) {
                    float* out_c = output_data + n * out_channels_ * out_length + c * out_length;
                    constexpr int W = at::native::tuda::VecF::width;
                    auto vb = at::native::tuda::VecF::broadcast(bias_data[c]);
                    int64_t j = 0;
                    for (; j + W <= out_length; j += W)
                        (at::native::tuda::VecF::load(out_c + j) + vb).store(out_c + j);
                    for (; j < out_length; ++j) out_c[j] += bias_data[c];
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

    // im2col: unfold input patches into column matrix for GEMM-based convolution
    // Output: col_buf[IC/groups * KH * KW, OH * OW]
    static void im2col(
        const float* __restrict input,   // [C_in, H, W] for single sample
        float* __restrict col,           // [IC/g * KH * KW, OH * OW]
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
                    const float* in_channel = input + c * in_height * in_width;

                    for (int64_t oh = 0; oh < out_height; ++oh) {
                        int64_t ih = oh * strH - padH + kh * dilH;
                        if (ih < 0 || ih >= in_height) {
                            // Entire row is padding — zero fill
                            int64_t col_offset = oh * out_width;
                            std::memset(col_ptr + col_offset, 0, out_width * sizeof(float));
                        } else {
                            for (int64_t ow = 0; ow < out_width; ++ow) {
                                int64_t iw = ow * strW - padW + kw * dilW;
                                int64_t col_idx = oh * out_width + ow;
                                if (iw >= 0 && iw < in_width) {
                                    col_ptr[col_idx] = in_channel[ih * in_width + iw];
                                } else {
                                    col_ptr[col_idx] = 0.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor forward(const Tensor& input_orig) override {
        // Input: [N, C_in, H, W]
        auto* weight_param = get_parameter("weight");
        Tensor W_orig = weight_param->data();

        // ================================================================
        // Autocast preamble: AMP dispatch boundary
        // ================================================================
        // If an AutocastGuard is active on this device AND the policy table
        // maps "conv2d" -> FP16, cast input + weight + bias to the autocast
        // dtype before entering the forward body.  Uses `to_autograd` so
        // backward flows through ToBackward nodes to the FP32 master weights.
        //
        // When autocast is off or policy is Unchanged, the original tensors
        // pass through by reference — zero overhead on the hot path.
        Tensor input_cast;
        Tensor W_cast;
        Tensor bias_cast;
        const Tensor* input_p = &input_orig;
        const Tensor* W_p = &W_orig;
        // bias tensor obtained lazily per-branch below; track the effective
        // one here so CUDA / CPU paths can share the cast result.
        bool bias_overridden = false;

        if (torch::amp::is_autocast_enabled(input_orig.device().type()) &&
            torch::amp::policy_for("conv2d") == torch::amp::CastPolicy::FP16)
        {
            auto target = torch::amp::get_autocast_dtype(input_orig.device().type());
            if (c10::isFloatingType(input_orig.dtype()) &&
                input_orig.dtype() != target)
            {
                input_cast = torch::autograd::to_autograd(input_orig, target);
                W_cast     = torch::autograd::to_autograd(W_orig, target);
                input_p = &input_cast;
                W_p     = &W_cast;
                if (has_bias_) {
                    auto* bias_param = get_parameter("bias");
                    bias_cast = torch::autograd::to_autograd(
                        bias_param->data(), target);
                    bias_overridden = true;
                }
            }
        }

        const Tensor& input = *input_p;
        const Tensor& W     = *W_p;

#ifdef PT_USE_CUDA
        // CUDA dispatch: prefer cuDNN for float32/groups=any; fall back to custom kernel
        // if cuDNN wrapper errors. Bias is added post-hoc since cuDNN conv here has no bias.
        if (input.is_cuda()) {
            Tensor bias_tensor;
            if (has_bias_) {
                if (bias_overridden) {
                    bias_tensor = bias_cast;
                } else {
                    auto* bias_param = get_parameter("bias");
                    bias_tensor = bias_param->data();
                }
            }
#ifdef PT_USE_CUDNN
            if (input.dtype() == c10::ScalarType::Float) {
                Tensor out = at::cudnn::cudnn_convolution_forward(
                    input, W,
                    padding_[0], padding_[1],
                    stride_[0], stride_[1],
                    dilation_[0], dilation_[1],
                    groups_);
                if (has_bias_ && bias_tensor.defined()) {
                    // Broadcast-add bias along channel dim: reshape bias [C] -> [1,C,1,1].
                    Tensor bias_reshaped = bias_tensor.view(
                        {1, bias_tensor.size(0), 1, 1});
                    out = out.add(bias_reshaped);
                }
                return out;
            }
#endif
            return at::cuda_ops::conv2d_forward(
                input, W, bias_tensor,
                static_cast<int>(stride_[0]), static_cast<int>(stride_[1]),
                static_cast<int>(padding_[0]), static_cast<int>(padding_[1]),
                static_cast<int>(dilation_[0]), static_cast<int>(dilation_[1]),
                static_cast<int>(groups_)
            );
        }
#endif

        // CPU im2col + GEMM implementation
        Tensor inp = input.is_contiguous() ? input : input.contiguous();
        Tensor weight = W.is_contiguous() ? W : W.contiguous();

        int64_t batch_size = inp.size(0);
        int64_t in_height = inp.size(2);
        int64_t in_width = inp.size(3);

        int64_t kH = kernel_size_[0], kW = kernel_size_[1];
        int64_t out_height = (in_height + 2 * padding_[0] - dilation_[0] * (kH - 1) - 1) / stride_[0] + 1;
        int64_t out_width = (in_width + 2 * padding_[1] - dilation_[1] * (kW - 1) - 1) / stride_[1] + 1;

        int64_t group_in_channels = in_channels_ / groups_;
        int64_t group_out_channels = out_channels_ / groups_;
        int64_t col_height = group_in_channels * kH * kW;  // K dimension for GEMM
        int64_t col_width = out_height * out_width;         // N dimension for GEMM

        Tensor output = at::empty({batch_size, out_channels_, out_height, out_width});

        const float* input_data = inp.data_ptr<float>();
        const float* weight_data = weight.data_ptr<float>();
        float* output_data = output.mutable_data_ptr<float>();

        // Allocate im2col buffer (reused per sample)
        std::vector<float> col_buf(col_height * col_width);

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t g = 0; g < groups_; ++g) {
                // Input pointer for this sample/group
                const float* in_ptr = input_data +
                    n * in_channels_ * in_height * in_width +
                    g * group_in_channels * in_height * in_width;

                // im2col: unfold patches into col_buf [col_height, col_width]
                im2col(in_ptr, col_buf.data(),
                       group_in_channels, in_height, in_width,
                       kH, kW,
                       padding_[0], padding_[1],
                       stride_[0], stride_[1],
                       dilation_[0], dilation_[1],
                       out_height, out_width);

                // Weight pointer for this group: [group_out_channels, col_height]
                const float* w_ptr = weight_data +
                    g * group_out_channels * col_height;

                // Output pointer: [group_out_channels, col_width]
                float* out_ptr = output_data +
                    n * out_channels_ * col_width +
                    g * group_out_channels * col_width;

                // GEMM: out[M,N] = W[M,K] × col[K,N]
                // M = group_out_channels, K = col_height, N = col_width
                at::native::hot::sgemm(
                    group_out_channels,  // M
                    col_height,          // K
                    col_width,           // N
                    1.0f,                // alpha
                    w_ptr, col_height,   // A [M, K], lda = K
                    col_buf.data(), col_width, // B [K, N], ldb = N
                    0.0f,                // beta
                    out_ptr, col_width   // C [M, N], ldc = N
                );
            }
        }

        // Fused bias addition with AVX2
        if (has_bias_) {
            Tensor bias_data_tensor = bias_overridden
                ? bias_cast
                : get_parameter("bias")->data();
            const float* bias_data = bias_data_tensor.data_ptr<float>();

            for (int64_t n = 0; n < batch_size; ++n) {
                for (int64_t c = 0; c < out_channels_; ++c) {
                    float* out_channel = output_data +
                        n * out_channels_ * col_width + c * col_width;
                    constexpr int W = at::native::tuda::VecF::width;
                    auto vbias = at::native::tuda::VecF::broadcast(bias_data[c]);
                    int64_t j = 0;
                    for (; j + W <= col_width; j += W)
                        (at::native::tuda::VecF::load(out_channel + j) + vbias).store(out_channel + j);
                    for (; j < col_width; ++j) {
                        out_channel[j] += bias_data[c];
                    }
                }
            }
        }

        // Wire autograd backward
        if (autograd::GradMode::is_enabled() &&
            (input.requires_grad() || weight_param->data().requires_grad())) {
            auto grad_fn = std::make_shared<autograd::Conv2dBackward>(
                inp, weight, has_bias_,
                in_channels_, out_channels_, groups_,
                kernel_size_, stride_, padding_, dilation_
            );
            grad_fn->add_input_metadata(input);
            grad_fn->add_input_metadata(weight_param->data());
            if (has_bias_) {
                grad_fn->add_input_metadata(get_parameter("bias")->data());
            }
            autograd::set_grad_fn(output, grad_fn);
            output.set_requires_grad(true);
        }

        // Preserve input memory format: if input was channels_last (NHWC), return
        // output in channels_last too. Internal compute remained NCHW (im2col path)
        // but the format contract is honored — user code that relies on NHWC will
        // see NHWC output. For a true NHWC-native fast path, a dedicated kernel
        // would be needed — flagged in Known Limitations.
        if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) && !input.is_contiguous()) {
            output = at::native::contiguous(output, c10::MemoryFormat::ChannelsLast);
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

    // Accessors (used by ONNX export and other introspection tools)
    int64_t in_channels() const { return in_channels_; }
    int64_t out_channels() const { return out_channels_; }
    const std::array<int64_t, 2>& kernel_size() const { return kernel_size_; }
    const std::array<int64_t, 2>& stride() const { return stride_; }
    const std::array<int64_t, 2>& padding() const { return padding_; }
    const std::array<int64_t, 2>& dilation() const { return dilation_; }
    int64_t groups() const { return groups_; }
    bool has_bias() const { return has_bias_; }

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
        // 3D convolution via im2col-style direct nested loops.
        // Previous version returned zeros — fixed 2026-04-18.
        PT_CHECK_MSG(input.dim() == 5, "Conv3d: input must be 5D [N, C, D, H, W]");
        int64_t N = input.size(0);
        int64_t C_in = input.size(1);
        int64_t Di = input.size(2);
        int64_t Hi = input.size(3);
        int64_t Wi = input.size(4);
        PT_CHECK(C_in == in_channels_);

        const int64_t kD = kernel_size_[0], kH = kernel_size_[1], kW = kernel_size_[2];
        const int64_t sD = stride_[0], sH = stride_[1], sW = stride_[2];
        const int64_t pD = padding_[0], pH = padding_[1], pW = padding_[2];
        const int64_t dD = dilation_[0], dH = dilation_[1], dW = dilation_[2];

        const int64_t Do = (Di + 2 * pD - dD * (kD - 1) - 1) / sD + 1;
        const int64_t Ho = (Hi + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        const int64_t Wo = (Wi + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

        Tensor output = at::zeros({N, out_channels_, Do, Ho, Wo},
                                  at::TensorOptions().dtype(input.dtype()).device(input.device()));

        const Tensor& weight = get_parameter("weight")->data();
        auto bias_ref = get_parameter("bias");
        const bool has_bias = bias_ref && bias_ref->defined();

        const float* x = input.is_contiguous() ? input.data_ptr<float>()
                                               : input.contiguous().data_ptr<float>();
        const float* w = weight.is_contiguous() ? weight.data_ptr<float>()
                                                : weight.contiguous().data_ptr<float>();
        float* y = output.mutable_data_ptr<float>();
        const float* b = has_bias ? bias_ref->data().data_ptr<float>() : nullptr;

        // weight shape: [C_out, C_in, kD, kH, kW]
        const int64_t wstride_oc = C_in * kD * kH * kW;
        const int64_t wstride_ic = kD * kH * kW;
        const int64_t wstride_kd = kH * kW;
        const int64_t xstride_n  = C_in * Di * Hi * Wi;
        const int64_t xstride_c  = Di * Hi * Wi;
        const int64_t xstride_d  = Hi * Wi;
        const int64_t ystride_n  = out_channels_ * Do * Ho * Wo;
        const int64_t ystride_c  = Do * Ho * Wo;
        const int64_t ystride_d  = Ho * Wo;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t oc = 0; oc < out_channels_; ++oc) {
                float bias_v = has_bias ? b[oc] : 0.0f;
                for (int64_t od = 0; od < Do; ++od) {
                    for (int64_t oh = 0; oh < Ho; ++oh) {
                        for (int64_t ow = 0; ow < Wo; ++ow) {
                            float sum = bias_v;
                            for (int64_t ic = 0; ic < C_in; ++ic) {
                                for (int64_t ki = 0; ki < kD; ++ki) {
                                    int64_t id_ = od * sD - pD + ki * dD;
                                    if (id_ < 0 || id_ >= Di) continue;
                                    for (int64_t kj = 0; kj < kH; ++kj) {
                                        int64_t ih_ = oh * sH - pH + kj * dH;
                                        if (ih_ < 0 || ih_ >= Hi) continue;
                                        for (int64_t kk = 0; kk < kW; ++kk) {
                                            int64_t iw_ = ow * sW - pW + kk * dW;
                                            if (iw_ < 0 || iw_ >= Wi) continue;
                                            float xv = x[n*xstride_n + ic*xstride_c +
                                                         id_*xstride_d + ih_*Wi + iw_];
                                            float wv = w[oc*wstride_oc + ic*wstride_ic +
                                                         ki*wstride_kd + kj*kW + kk];
                                            sum += xv * wv;
                                        }
                                    }
                                }
                            }
                            y[n*ystride_n + oc*ystride_c + od*ystride_d + oh*Wo + ow] = sum;
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
        // Transposed convolution (deconvolution).
        // NOTE: Compute is CPU-only — if the input lives on CUDA we bounce through
        // host memory. That's fine for small models (DCGAN / debugging); a native
        // cuDNN path can be wired in later.
        const bool input_on_cuda = input.is_cuda();

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
#ifdef PT_USE_CUDA
        Tensor weight = weight_param->data().is_cuda()
            ? at::to_cpu(weight_param->data()).contiguous()
            : weight_param->data().contiguous();
#else
        Tensor weight = weight_param->data().contiguous();
#endif

        int64_t out_channels_per_group = out_channels_ / groups_;
        int64_t in_channels_per_group = in_channels_ / groups_;

        Tensor output = at::zeros({batch_size, out_channels_, out_height, out_width});
#ifdef PT_USE_CUDA
        Tensor inp = input_on_cuda ? at::to_cpu(input).contiguous() : input.contiguous();
#else
        Tensor inp = input.contiguous();
#endif

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

        // Wire autograd backward — saved tensors are always on CPU (compute path).
        // ConvTranspose2dBackward will produce CPU grads which the engine then
        // forwards to the upstream grad sinks (parameter.grad_). For CUDA params,
        // the caller is responsible for ensuring grads end up on the correct device;
        // in practice our optimizers handle CPU grads for CPU params and this
        // bounce matches the CPU-bounce of the forward.
        const bool needs_grad = autograd::GradMode::is_enabled() &&
            (input.requires_grad() || weight_param->data().requires_grad());
        if (needs_grad) {
            auto grad_fn = std::make_shared<autograd::ConvTranspose2dBackward>(
                inp, weight, has_bias_,
                in_channels_, out_channels_, groups_,
                kernel_size_, stride_, padding_, dilation_,
                out_height, out_width
            );
            grad_fn->add_input_metadata(input);
            grad_fn->add_input_metadata(weight_param->data());
            if (has_bias_) {
                grad_fn->add_input_metadata(get_parameter("bias")->data());
            }
            autograd::set_grad_fn(output, grad_fn);
            output.set_requires_grad(true);
        }

#ifdef PT_USE_CUDA
        if (input_on_cuda) {
            output = at::to_cuda(output);
        }
#endif

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
