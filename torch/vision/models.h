// torch/vision/models.h
// torchvision-compatible model definitions for PromeTorch.
// Currently provides MobileNetV2 (Sandler et al., 2018).
//
// Architecture follows the canonical reference:
//   stem:   3x3 conv (stride 2) -> BN -> ReLU6
//   body:   17 InvertedResidual blocks (config table below)
//   head:   1x1 conv (stride 1) -> BN -> ReLU6 -> AdaptiveAvgPool(1) -> Linear(num_classes)
//
// CPU-only, header-only.

#pragma once

#include "torch/nn/module.h"
#include "torch/nn/modules/conv.h"
#include "torch/nn/modules/normalization.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/pooling.h"
#include "torch/serialization.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace vision {
namespace models {

using torch::nn::Module;
using torch::nn::ModulePtr;
using torch::nn::Conv2d;
using torch::nn::BatchNorm2d;
using torch::nn::ReLU6;
using torch::nn::Linear;
using torch::nn::AdaptiveAvgPool2d;
using at::Tensor;

// Round channels to nearest multiple of `divisor` (TF/torchvision convention).
inline int64_t make_divisible(double v, int64_t divisor = 8, int64_t min_value = 0) {
    int64_t mn = (min_value > 0) ? min_value : divisor;
    int64_t new_v = std::max(mn, static_cast<int64_t>((v + divisor / 2.0)) / divisor * divisor);
    if (new_v < 0.9 * v) new_v += divisor;
    return new_v;
}

// ---------------------------------------------------------------------------
// ConvBNReLU: 3x3 (or 1x1) Conv -> BN -> ReLU6
// ---------------------------------------------------------------------------
class ConvBNReLU : public Module {
public:
    ConvBNReLU(int64_t in_planes, int64_t out_planes,
               int64_t kernel_size = 3, int64_t stride = 1, int64_t groups = 1)
        : Module("ConvBNReLU") {
        const int64_t padding = (kernel_size - 1) / 2;
        conv_ = std::make_shared<Conv2d>(
            in_planes, out_planes, kernel_size, stride, padding, /*dilation=*/1,
            groups, /*bias=*/false);
        bn_ = std::make_shared<BatchNorm2d>(out_planes);
        relu_ = std::make_shared<ReLU6>();
        register_module("0", conv_);
        register_module("1", bn_);
        register_module("2", relu_);
    }

    Tensor forward(const Tensor& x) override {
        Tensor y = conv_->forward(x);
        y = bn_->forward(y);
        y = relu_->forward(y);
        return y;
    }

private:
    std::shared_ptr<Conv2d> conv_;
    std::shared_ptr<BatchNorm2d> bn_;
    std::shared_ptr<ReLU6> relu_;
};

// ---------------------------------------------------------------------------
// InvertedResidual: the MobileNetV2 building block.
//   pw expand (1x1) -> dw conv (3x3, stride s, groups=hidden) -> pw project (1x1, no act)
//   if stride==1 and in==out, add residual connection.
// ---------------------------------------------------------------------------
class InvertedResidual : public Module {
public:
    InvertedResidual(int64_t inp, int64_t oup, int64_t stride, int64_t expand_ratio)
        : Module("InvertedResidual"), stride_(stride), use_res_(stride == 1 && inp == oup) {
        const int64_t hidden_dim = inp * expand_ratio;

        int idx = 0;
        if (expand_ratio != 1) {
            expand_ = std::make_shared<ConvBNReLU>(inp, hidden_dim, /*ks=*/1);
            register_module(std::to_string(idx++), expand_);
        }
        // depthwise
        dw_ = std::make_shared<ConvBNReLU>(hidden_dim, hidden_dim, /*ks=*/3,
                                           stride, /*groups=*/hidden_dim);
        register_module(std::to_string(idx++), dw_);
        // pointwise project (no activation)
        pw_conv_ = std::make_shared<Conv2d>(hidden_dim, oup, /*ks=*/1, /*stride=*/1,
                                            /*pad=*/0, /*dilation=*/1, /*groups=*/1,
                                            /*bias=*/false);
        pw_bn_ = std::make_shared<BatchNorm2d>(oup);
        register_module(std::to_string(idx++), pw_conv_);
        register_module(std::to_string(idx++), pw_bn_);
    }

    Tensor forward(const Tensor& x) override {
        Tensor y = x;
        if (expand_) y = expand_->forward(y);
        y = dw_->forward(y);
        y = pw_conv_->forward(y);
        y = pw_bn_->forward(y);
        if (use_res_) y = y.add(x);
        return y;
    }

private:
    int64_t stride_;
    bool use_res_;
    std::shared_ptr<ConvBNReLU> expand_;  // may be null for first block (expand_ratio==1)
    std::shared_ptr<ConvBNReLU> dw_;
    std::shared_ptr<Conv2d> pw_conv_;
    std::shared_ptr<BatchNorm2d> pw_bn_;
};

// ---------------------------------------------------------------------------
// MobileNetV2.
//   Default: ImageNet (1000 classes), width_mult=1.0, input 224x224x3.
// ---------------------------------------------------------------------------
class MobileNetV2 : public Module {
public:
    explicit MobileNetV2(int64_t num_classes = 1000,
                         double width_mult = 1.0,
                         int64_t round_nearest = 8)
        : Module("MobileNetV2") {
        // Inverted residual config: t, c, n, s   (expand, channels, repeat, stride)
        const std::vector<std::vector<int64_t>> cfg = {
            {1,  16, 1, 1},
            {6,  24, 2, 2},
            {6,  32, 3, 2},
            {6,  64, 4, 2},
            {6,  96, 3, 1},
            {6, 160, 3, 2},
            {6, 320, 1, 1},
        };

        int64_t input_channel = make_divisible(32 * width_mult, round_nearest);
        int64_t last_channel  = make_divisible(1280 * std::max(1.0, width_mult), round_nearest);
        last_channel_ = last_channel;

        // ---- Build features Sequential equivalent (ordered child modules) ----
        int feat_idx = 0;
        auto add_feature = [&](ModulePtr m) {
            features_.push_back(m);
            register_module("features." + std::to_string(feat_idx++), m);
        };

        // Stem: 3x3 conv s=2.
        add_feature(std::make_shared<ConvBNReLU>(3, input_channel, /*ks=*/3, /*stride=*/2));

        // Body
        for (const auto& row : cfg) {
            int64_t t = row[0], c = row[1], n = row[2], s = row[3];
            int64_t output_channel = make_divisible(c * width_mult, round_nearest);
            for (int64_t i = 0; i < n; ++i) {
                int64_t stride = (i == 0) ? s : 1;
                add_feature(std::make_shared<InvertedResidual>(
                    input_channel, output_channel, stride, t));
                input_channel = output_channel;
            }
        }
        // Head conv: 1x1 to last_channel
        add_feature(std::make_shared<ConvBNReLU>(input_channel, last_channel, /*ks=*/1));

        // Classifier: AvgPool(1) → flatten → Linear
        avgpool_ = std::make_shared<AdaptiveAvgPool2d>(1);
        register_module("avgpool", avgpool_);
        classifier_ = std::make_shared<Linear>(last_channel, num_classes);
        register_module("classifier", classifier_);
    }

    Tensor forward(const Tensor& input) override {
        Tensor x = input;
        for (auto& m : features_) {
            x = m->forward(x);
        }
        x = avgpool_->forward(x);                 // [N, C, 1, 1]
        x = x.view({x.size(0), x.size(1)});       // flatten
        x = classifier_->forward(x);
        return x;
    }

    // Load pretrained weights from a PromeTorch state_dict file (.bin format,
    // produced by torch::save_state_dict). Delegates to Module::load_state_dict.
    void load_pretrained(const std::string& path, bool strict = true) {
        auto sd = torch::load_state_dict(path);
        // torch::StateDict is the same alias used by Module.
        load_state_dict(sd, strict);
    }

    int64_t last_channel() const { return last_channel_; }

private:
    std::vector<ModulePtr> features_;
    std::shared_ptr<AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<Linear> classifier_;
    int64_t last_channel_ = 0;
};

} // namespace models
} // namespace vision
} // namespace torch
