#pragma once

#include "torch/nn/module.h"
#include "torch/quantization/quantize.h"
#include "torch/quantization/observer.h"
#include "aten/src/ATen/ATen.h"
#include <memory>

namespace torch {
namespace nn {

using at::Tensor;
using at::TensorOptions;

// ============================================================================
// QuantizedLinear — quantized version of Linear layer
// Uses fake quantization: dequant weights -> matmul -> result
// ============================================================================

class QuantizedLinear : public Module {
public:
    QuantizedLinear(int64_t in_features, int64_t out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias) {}

    // Initialize from a float Linear layer's parameters
    void from_float(const Tensor& weight, const Tensor& bias, double scale, int64_t zero_point) {
        weight_q_ = std::make_unique<quantization::QuantizedTensor>(
            quantization::quantize_per_tensor(weight, scale, zero_point));

        if (has_bias_ && bias.defined()) {
            bias_ = bias.clone();  // bias stays in fp32
        }

        scale_ = scale;
        zero_point_ = zero_point;
    }

    Tensor forward(const Tensor& input) override {
        PT_CHECK_MSG(weight_q_, "QuantizedLinear: weights not initialized. Call from_float() first.");

        // Dequantize weight for computation (fake quant approach)
        Tensor weight = weight_q_->dequantize();

        // x @ W^T + b
        Tensor output = input.mm(weight.t());

        if (has_bias_ && bias_.defined()) {
            output = output.add(bias_);
        }

        return output;
    }

    std::string name() const override { return "QuantizedLinear"; }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << in_features_ << ", " << out_features_;
        ss << ", scale=" << scale_ << ", zero_point=" << zero_point_;
        return ss.str();
    }

    double scale() const { return scale_; }
    int64_t zero_point() const { return zero_point_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    bool has_bias_;
    std::unique_ptr<quantization::QuantizedTensor> weight_q_;
    Tensor bias_;
    double scale_ = 0;
    int64_t zero_point_ = 0;
};

// ============================================================================
// QuantizedConv2d — quantized version of Conv2d layer
// ============================================================================

class QuantizedConv2d : public Module {
public:
    QuantizedConv2d(int64_t in_channels, int64_t out_channels,
                    std::array<int64_t, 2> kernel_size,
                    std::array<int64_t, 2> stride = {1, 1},
                    std::array<int64_t, 2> padding = {0, 0},
                    int64_t groups = 1, bool bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          groups_(groups), has_bias_(bias) {}

    void from_float(const Tensor& weight, const Tensor& bias, double scale, int64_t zero_point) {
        weight_q_ = std::make_unique<quantization::QuantizedTensor>(
            quantization::quantize_per_tensor(weight, scale, zero_point));

        if (has_bias_ && bias.defined()) {
            bias_ = bias.clone();
        }

        scale_ = scale;
        zero_point_ = zero_point;
    }

    Tensor forward(const Tensor& input) override {
        PT_CHECK_MSG(weight_q_, "QuantizedConv2d: weights not initialized");

        // Dequantize and do regular conv (fake quant)
        Tensor weight = weight_q_->dequantize();
        Tensor inp = input.contiguous();

        int64_t batch = inp.size(0);
        int64_t iH = inp.size(2);
        int64_t iW = inp.size(3);
        int64_t kH = kernel_size_[0];
        int64_t kW = kernel_size_[1];
        int64_t oH = (iH + 2 * padding_[0] - kH) / stride_[0] + 1;
        int64_t oW = (iW + 2 * padding_[1] - kW) / stride_[1] + 1;

        Tensor output = at::zeros({batch, out_channels_, oH, oW});

        const float* in_data = inp.data_ptr<float>();
        const float* w_data = weight.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        int64_t in_channels_per_group = in_channels_ / groups_;
        int64_t out_channels_per_group = out_channels_ / groups_;

        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t g = 0; g < groups_; ++g) {
                for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
                    int64_t c_out = g * out_channels_per_group + oc;
                    for (int64_t oh = 0; oh < oH; ++oh) {
                        for (int64_t ow = 0; ow < oW; ++ow) {
                            float sum = 0.0f;
                            for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
                                int64_t c_in = g * in_channels_per_group + ic;
                                for (int64_t kh = 0; kh < kH; ++kh) {
                                    for (int64_t kw = 0; kw < kW; ++kw) {
                                        int64_t ih = oh * stride_[0] - padding_[0] + kh;
                                        int64_t iw = ow * stride_[1] - padding_[1] + kw;
                                        if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                                            sum += in_data[((n * in_channels_ + c_in) * iH + ih) * iW + iw] *
                                                   w_data[((c_out * in_channels_per_group + ic) * kH + kh) * kW + kw];
                                        }
                                    }
                                }
                            }
                            out_data[((n * out_channels_ + c_out) * oH + oh) * oW + ow] = sum;
                        }
                    }
                }
            }
        }

        // Add bias
        if (has_bias_ && bias_.defined()) {
            const float* b_data = bias_.data_ptr<float>();
            for (int64_t n = 0; n < batch; ++n) {
                for (int64_t c = 0; c < out_channels_; ++c) {
                    for (int64_t h = 0; h < oH; ++h) {
                        for (int64_t w = 0; w < oW; ++w) {
                            out_data[((n * out_channels_ + c) * oH + h) * oW + w] += b_data[c];
                        }
                    }
                }
            }
        }

        return output;
    }

    std::string name() const override { return "QuantizedConv2d"; }

private:
    int64_t in_channels_, out_channels_;
    std::array<int64_t, 2> kernel_size_, stride_, padding_;
    int64_t groups_;
    bool has_bias_;
    std::unique_ptr<quantization::QuantizedTensor> weight_q_;
    Tensor bias_;
    double scale_ = 0;
    int64_t zero_point_ = 0;
};

} // namespace nn
} // namespace torch
