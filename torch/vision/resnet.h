// torch/vision/resnet.h
// ResNet-20 for CIFAR-10, following He et al. 2015 ("Deep Residual Learning for
// Image Recognition", Section 4.2 "CIFAR-10 and Analysis").
//
//   stem:    3x3 conv (3 -> 16), stride 1, pad 1, no bias  +  BN  +  ReLU
//   stage 1: 3 BasicBlocks, 16 channels, stride 1
//   stage 2: 3 BasicBlocks, 32 channels, first block stride 2 (downsample)
//   stage 3: 3 BasicBlocks, 64 channels, first block stride 2 (downsample)
//   head:    GlobalAvgPool (8x8 -> 1x1) + Linear(64, num_classes)
//
// BasicBlock: 3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN ( + skip ) -> ReLU
// Skip path: identity when channels + stride match, else 1x1 Conv + BN
// (projection shortcut, option B in the original paper).
//
// All Conv weights are Kaiming-normal fan_out / ReLU (the reference init for
// ResNet). BN gammas are 1, betas are 0 (framework defaults). Kaiming init is
// applied in-place right after the Conv2d is constructed so we never need
// accessors into submodules from the outside.
//
// Total params for ResNet-20 on CIFAR-10 ≈ 0.27M. Target test accuracy 91.25%.
// CPU + CUDA (cuDNN) ready: all underlying modules already dispatch on device.

#pragma once

#include "torch/nn/module.h"
#include "torch/nn/modules/conv.h"
#include "torch/nn/modules/normalization.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/pooling.h"
#include "torch/nn/init.h"

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
using torch::nn::ReLU;
using torch::nn::Linear;
using torch::nn::AdaptiveAvgPool2d;
using at::Tensor;

// Kaiming-normal (fan_out, ReLU gain) init for a Conv2d's weight parameter.
// Applied in-place on the tensor.
inline void kaiming_init_conv_(Conv2d& c) {
    auto* w = c.get_parameter("weight");
    if (w && w->defined()) {
        Tensor& t = w->data();
        torch::nn::init::kaiming_normal_(t, /*a=*/0.0,
            torch::nn::init::FanMode::FanOut, /*nonlinearity=*/"relu");
    }
}

// Make a Conv2d with no bias and immediately apply Kaiming init.
inline std::shared_ptr<Conv2d> make_conv(int64_t in_c, int64_t out_c,
                                         int64_t ks, int64_t stride, int64_t pad) {
    auto c = std::make_shared<Conv2d>(in_c, out_c, ks, stride, pad,
                                      /*dilation=*/1, /*groups=*/1, /*bias=*/false);
    kaiming_init_conv_(*c);
    return c;
}

// ---------------------------------------------------------------------------
// CifarBasicBlock
//   Two 3x3 convs (bias=False) with BN + ReLU, plus identity/projection skip.
// ---------------------------------------------------------------------------
class CifarBasicBlock : public Module {
public:
    CifarBasicBlock(int64_t in_planes, int64_t planes, int64_t stride = 1)
        : Module("CifarBasicBlock"), stride_(stride), in_planes_(in_planes), planes_(planes) {
        conv1_ = make_conv(in_planes, planes, /*ks=*/3, /*stride=*/stride, /*pad=*/1);
        bn1_ = std::make_shared<BatchNorm2d>(planes);
        relu_ = std::make_shared<ReLU>();
        conv2_ = make_conv(planes, planes, /*ks=*/3, /*stride=*/1, /*pad=*/1);
        bn2_ = std::make_shared<BatchNorm2d>(planes);

        register_module("conv1", conv1_);
        register_module("bn1", bn1_);
        register_module("conv2", conv2_);
        register_module("bn2", bn2_);

        if (stride != 1 || in_planes != planes) {
            shortcut_conv_ = make_conv(in_planes, planes, /*ks=*/1, /*stride=*/stride, /*pad=*/0);
            shortcut_bn_ = std::make_shared<BatchNorm2d>(planes);
            register_module("shortcut_conv", shortcut_conv_);
            register_module("shortcut_bn", shortcut_bn_);
        }
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = conv1_->forward(x);
        out = bn1_->forward(out);
        out = relu_->forward(out);
        out = conv2_->forward(out);
        out = bn2_->forward(out);

        Tensor residual = x;
        if (shortcut_conv_) {
            residual = shortcut_conv_->forward(x);
            residual = shortcut_bn_->forward(residual);
        }
        out = out.add(residual);
        out = relu_->forward(out);
        return out;
    }

    int64_t stride() const { return stride_; }
    int64_t in_planes() const { return in_planes_; }
    int64_t planes() const { return planes_; }

private:
    int64_t stride_;
    int64_t in_planes_;
    int64_t planes_;
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<BatchNorm2d> bn1_;
    std::shared_ptr<Conv2d> conv2_;
    std::shared_ptr<BatchNorm2d> bn2_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<Conv2d> shortcut_conv_;       // null when identity skip
    std::shared_ptr<BatchNorm2d> shortcut_bn_;    // null when identity skip
};

// ---------------------------------------------------------------------------
// ResNetCifar — depth = 6n + 2. For ResNet-20, n = 3.
// ---------------------------------------------------------------------------
class ResNetCifar : public Module {
public:
    // n_blocks_per_stage: 3 for ResNet-20, 5 for ResNet-32, 7 for ResNet-44, ...
    explicit ResNetCifar(int64_t num_classes = 10, int64_t n_blocks_per_stage = 3)
        : Module("ResNetCifar"), num_classes_(num_classes), n_(n_blocks_per_stage) {
        // Stem: 3x3 conv (3 -> 16) with stride 1 (NOT 2 — the CIFAR variant keeps
        // spatial resolution until the first downsampling block).
        stem_conv_ = make_conv(3, 16, /*ks=*/3, /*stride=*/1, /*pad=*/1);
        stem_bn_ = std::make_shared<BatchNorm2d>(16);
        stem_relu_ = std::make_shared<ReLU>();
        register_module("conv1", stem_conv_);
        register_module("bn1", stem_bn_);

        // Three stages of n blocks each.
        build_stage(stage1_, /*in=*/16, /*out=*/16, /*first_stride=*/1, "layer1");
        build_stage(stage2_, /*in=*/16, /*out=*/32, /*first_stride=*/2, "layer2");
        build_stage(stage3_, /*in=*/32, /*out=*/64, /*first_stride=*/2, "layer3");

        avgpool_ = std::make_shared<AdaptiveAvgPool2d>(1);
        register_module("avgpool", avgpool_);
        fc_ = std::make_shared<Linear>(64, num_classes);
        register_module("fc", fc_);
    }

    Tensor forward(const Tensor& input) override {
        Tensor x = stem_conv_->forward(input);
        x = stem_bn_->forward(x);
        x = stem_relu_->forward(x);

        for (auto& b : stage1_) x = b->forward(x);
        for (auto& b : stage2_) x = b->forward(x);
        for (auto& b : stage3_) x = b->forward(x);

        x = avgpool_->forward(x);                           // [N, 64, 1, 1]
        x = x.view({x.size(0), x.size(1)});                 // flatten to [N, 64]
        x = fc_->forward(x);
        return x;
    }

    int64_t num_classes() const { return num_classes_; }

private:
    void build_stage(std::vector<std::shared_ptr<CifarBasicBlock>>& blocks,
                     int64_t in_planes, int64_t out_planes,
                     int64_t first_stride, const std::string& name_prefix) {
        for (int64_t i = 0; i < n_; ++i) {
            int64_t stride = (i == 0) ? first_stride : 1;
            int64_t in_c = (i == 0) ? in_planes : out_planes;
            auto blk = std::make_shared<CifarBasicBlock>(in_c, out_planes, stride);
            blocks.push_back(blk);
            register_module(name_prefix + "." + std::to_string(i), blk);
        }
    }

    int64_t num_classes_;
    int64_t n_;
    std::shared_ptr<Conv2d> stem_conv_;
    std::shared_ptr<BatchNorm2d> stem_bn_;
    std::shared_ptr<ReLU> stem_relu_;
    std::vector<std::shared_ptr<CifarBasicBlock>> stage1_;
    std::vector<std::shared_ptr<CifarBasicBlock>> stage2_;
    std::vector<std::shared_ptr<CifarBasicBlock>> stage3_;
    std::shared_ptr<AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<Linear> fc_;
};

// Convenience factories.
inline std::shared_ptr<ResNetCifar> resnet20(int64_t num_classes = 10) {
    return std::make_shared<ResNetCifar>(num_classes, /*n_blocks_per_stage=*/3);
}
inline std::shared_ptr<ResNetCifar> resnet32(int64_t num_classes = 10) {
    return std::make_shared<ResNetCifar>(num_classes, /*n_blocks_per_stage=*/5);
}
inline std::shared_ptr<ResNetCifar> resnet44(int64_t num_classes = 10) {
    return std::make_shared<ResNetCifar>(num_classes, /*n_blocks_per_stage=*/7);
}
inline std::shared_ptr<ResNetCifar> resnet56(int64_t num_classes = 10) {
    return std::make_shared<ResNetCifar>(num_classes, /*n_blocks_per_stage=*/9);
}

} // namespace models
} // namespace vision
} // namespace torch
