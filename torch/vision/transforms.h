// torch/vision/transforms.h
// torchvision-style image transforms for PromeTorch.
//
// All transforms are callable classes:  Tensor operator()(const Tensor& img);
// Tensors are expected in either:
//   - HWC uint8 (raw image, e.g. straight out of ImageFolder loader)
//   - CHW float (after ToTensor)
// Compose chains them in order.
//
// CPU-only, header-only. Safe on Elbrus (LCC, no SSE/AVX intrinsics here).
#pragma once

#include "aten/src/ATen/ATen.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace torch {
namespace vision {
namespace transforms {

using at::Tensor;
using at::Scalar;

// Base interface.
class Transform {
public:
    virtual ~Transform() = default;
    virtual Tensor operator()(const Tensor& input) const = 0;
};

using TransformPtr = std::shared_ptr<Transform>;

// ---------------------------------------------------------------------------
// Compose — sequential pipeline.
// ---------------------------------------------------------------------------
class Compose : public Transform {
public:
    Compose() = default;
    Compose(std::vector<TransformPtr> transforms) : transforms_(std::move(transforms)) {}

    void push_back(TransformPtr t) { transforms_.push_back(std::move(t)); }

    Tensor operator()(const Tensor& input) const override {
        Tensor x = input;
        for (const auto& t : transforms_) {
            x = (*t)(x);
        }
        return x;
    }

private:
    std::vector<TransformPtr> transforms_;
};

inline std::shared_ptr<Compose> compose(std::initializer_list<TransformPtr> ts) {
    return std::make_shared<Compose>(std::vector<TransformPtr>(ts));
}

// ---------------------------------------------------------------------------
// ToTensor — uint8 [H,W,C] in [0,255]  →  float32 [C,H,W] in [0,1].
// Also handles uint8 [H,W] grayscale → float32 [1,H,W].
// If input is already float CHW, returns it unchanged (idempotent).
// ---------------------------------------------------------------------------
class ToTensor : public Transform {
public:
    Tensor operator()(const Tensor& input) const override {
        // Already float CHW? Pass through.
        if (input.dtype() == c10::ScalarType::Float && input.dim() == 3 &&
            input.size(0) <= 4) {
            return input.contiguous();
        }

        Tensor x = input;
        bool is_int = (x.dtype() != c10::ScalarType::Float &&
                       x.dtype() != c10::ScalarType::Double);

        if (is_int) {
            x = x.to(c10::ScalarType::Float);
            x = x.div(Scalar(255.0));
        }

        if (x.dim() == 3) {
            // Assume HWC -> CHW
            x = x.permute({2, 0, 1}).contiguous();
        } else if (x.dim() == 2) {
            x = x.unsqueeze(0).contiguous();  // [H,W] -> [1,H,W]
        }

        return x;
    }
};

// ---------------------------------------------------------------------------
// Normalize — per-channel: (x - mean) / std.
// Expects CHW float tensor.
// ---------------------------------------------------------------------------
class Normalize : public Transform {
public:
    Normalize(std::vector<float> mean, std::vector<float> std_dev)
        : mean_(std::move(mean)), std_(std::move(std_dev)) {}

    Tensor operator()(const Tensor& input) const override {
        PT_CHECK_MSG(input.dim() == 3,
            "Normalize expects CHW tensor, got dim=", input.dim());
        Tensor x = input.contiguous();
        const int64_t C = x.size(0);
        const int64_t H = x.size(1);
        const int64_t W = x.size(2);
        const int64_t spatial = H * W;
        PT_CHECK_MSG(static_cast<size_t>(C) <= mean_.size() &&
                     static_cast<size_t>(C) <= std_.size(),
            "Normalize: channel count > mean/std vector length");

        Tensor out = at::empty(x.sizes(), at::TensorOptions().dtype(c10::ScalarType::Float));
        const float* src = x.data_ptr<float>();
        float* dst = out.mutable_data_ptr<float>();
        for (int64_t c = 0; c < C; ++c) {
            const float m = mean_[c];
            const float s = (std_[c] != 0.0f ? std_[c] : 1.0f);
            const float inv = 1.0f / s;
            const float* sp = src + c * spatial;
            float* dp = dst + c * spatial;
            for (int64_t i = 0; i < spatial; ++i) {
                dp[i] = (sp[i] - m) * inv;
            }
        }
        return out;
    }

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

// ---------------------------------------------------------------------------
// Resize — bilinear resample to (H,W). Works on CHW or HWC, float or uint8.
// If input is uint8 we resample directly in uint8 space (no allocation overhead).
// ---------------------------------------------------------------------------
namespace detail {

// Bilinear sample one channel-plane.  Generic over T.
template <typename T>
inline void bilinear_plane(const T* src, int64_t in_h, int64_t in_w,
                           T* dst, int64_t out_h, int64_t out_w) {
    const double scale_h = (out_h > 0) ? static_cast<double>(in_h) / out_h : 1.0;
    const double scale_w = (out_w > 0) ? static_cast<double>(in_w) / out_w : 1.0;
    for (int64_t oh = 0; oh < out_h; ++oh) {
        double sh = (oh + 0.5) * scale_h - 0.5;
        if (sh < 0) sh = 0;
        if (sh > in_h - 1) sh = in_h - 1;
        int64_t h0 = static_cast<int64_t>(std::floor(sh));
        int64_t h1 = std::min(h0 + 1, in_h - 1);
        double dh = sh - h0;
        for (int64_t ow = 0; ow < out_w; ++ow) {
            double sw = (ow + 0.5) * scale_w - 0.5;
            if (sw < 0) sw = 0;
            if (sw > in_w - 1) sw = in_w - 1;
            int64_t w0 = static_cast<int64_t>(std::floor(sw));
            int64_t w1 = std::min(w0 + 1, in_w - 1);
            double dw = sw - w0;
            double v00 = static_cast<double>(src[h0 * in_w + w0]);
            double v01 = static_cast<double>(src[h0 * in_w + w1]);
            double v10 = static_cast<double>(src[h1 * in_w + w0]);
            double v11 = static_cast<double>(src[h1 * in_w + w1]);
            double v = (1.0 - dh) * ((1.0 - dw) * v00 + dw * v01) +
                       dh * ((1.0 - dw) * v10 + dw * v11);
            if constexpr (std::is_integral_v<T>) {
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                dst[oh * out_w + ow] = static_cast<T>(std::round(v));
            } else {
                dst[oh * out_w + ow] = static_cast<T>(v);
            }
        }
    }
}

} // namespace detail

class Resize : public Transform {
public:
    Resize(int64_t height, int64_t width) : height_(height), width_(width) {}
    explicit Resize(int64_t size) : height_(size), width_(size) {}

    Tensor operator()(const Tensor& input) const override {
        const int64_t ndim = input.dim();
        PT_CHECK_MSG(ndim == 2 || ndim == 3, "Resize: expected 2D or 3D tensor");

        Tensor x = input.contiguous();
        std::vector<int64_t> out_shape = x.sizes().vec();

        // Determine layout: HWC if last dim is small (<=4); CHW if first dim is small.
        bool hwc = (ndim == 3 && x.size(2) <= 4 && x.size(0) > 4);
        // In HW it's just [H,W].
        int64_t in_h, in_w, channels;
        if (ndim == 2) { in_h = x.size(0); in_w = x.size(1); channels = 1; }
        else if (hwc)  { in_h = x.size(0); in_w = x.size(1); channels = x.size(2); }
        else           { channels = x.size(0); in_h = x.size(1); in_w = x.size(2); }

        if (ndim == 2) { out_shape[0] = height_; out_shape[1] = width_; }
        else if (hwc)  { out_shape[0] = height_; out_shape[1] = width_; }
        else           { out_shape[1] = height_; out_shape[2] = width_; }

        Tensor out = at::empty(out_shape, at::TensorOptions().dtype(x.dtype()));

        if (x.dtype() == c10::ScalarType::Float) {
            const float* src = x.data_ptr<float>();
            float* dst = out.mutable_data_ptr<float>();
            if (ndim == 2 || !hwc) {
                // CHW (or single plane): each channel contiguous.
                for (int64_t c = 0; c < channels; ++c) {
                    detail::bilinear_plane<float>(
                        src + c * in_h * in_w, in_h, in_w,
                        dst + c * height_ * width_, height_, width_);
                }
            } else {
                // HWC: interleaved channels — resample channel-by-channel via copy.
                std::vector<float> plane_in(in_h * in_w), plane_out(height_ * width_);
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t y = 0; y < in_h; ++y)
                        for (int64_t xx = 0; xx < in_w; ++xx)
                            plane_in[y * in_w + xx] = src[(y * in_w + xx) * channels + c];
                    detail::bilinear_plane<float>(plane_in.data(), in_h, in_w,
                                                  plane_out.data(), height_, width_);
                    for (int64_t y = 0; y < height_; ++y)
                        for (int64_t xx = 0; xx < width_; ++xx)
                            dst[(y * width_ + xx) * channels + c] = plane_out[y * width_ + xx];
                }
            }
        } else if (x.dtype() == c10::ScalarType::Byte) {
            const uint8_t* src = x.data_ptr<uint8_t>();
            uint8_t* dst = out.mutable_data_ptr<uint8_t>();
            if (ndim == 2 || !hwc) {
                for (int64_t c = 0; c < channels; ++c) {
                    detail::bilinear_plane<uint8_t>(
                        src + c * in_h * in_w, in_h, in_w,
                        dst + c * height_ * width_, height_, width_);
                }
            } else {
                std::vector<uint8_t> plane_in(in_h * in_w), plane_out(height_ * width_);
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t y = 0; y < in_h; ++y)
                        for (int64_t xx = 0; xx < in_w; ++xx)
                            plane_in[y * in_w + xx] = src[(y * in_w + xx) * channels + c];
                    detail::bilinear_plane<uint8_t>(plane_in.data(), in_h, in_w,
                                                    plane_out.data(), height_, width_);
                    for (int64_t y = 0; y < height_; ++y)
                        for (int64_t xx = 0; xx < width_; ++xx)
                            dst[(y * width_ + xx) * channels + c] = plane_out[y * width_ + xx];
                }
            }
        } else {
            PT_CHECK_MSG(false, "Resize: unsupported dtype (need float or uint8)");
        }
        return out;
    }

private:
    int64_t height_, width_;
};

// ---------------------------------------------------------------------------
// CenterCrop — crop H,W centered. Works on CHW or HW.
// ---------------------------------------------------------------------------
class CenterCrop : public Transform {
public:
    CenterCrop(int64_t height, int64_t width) : height_(height), width_(width) {}
    explicit CenterCrop(int64_t size) : height_(size), width_(size) {}

    Tensor operator()(const Tensor& input) const override {
        const int64_t ndim = input.dim();
        const int64_t h = input.size(ndim - 2);
        const int64_t w = input.size(ndim - 1);
        PT_CHECK_MSG(h >= height_ && w >= width_, "CenterCrop: image too small");
        const int64_t top = (h - height_) / 2;
        const int64_t left = (w - width_) / 2;
        return input.narrow(ndim - 2, top, height_)
                    .narrow(ndim - 1, left, width_)
                    .contiguous();
    }

private:
    int64_t height_, width_;
};

// ---------------------------------------------------------------------------
// RandomCrop — random crop with optional zero-padding.
// ---------------------------------------------------------------------------
class RandomCrop : public Transform {
public:
    RandomCrop(int64_t height, int64_t width, int64_t padding = 0)
        : height_(height), width_(width), padding_(padding) {}
    explicit RandomCrop(int64_t size, int64_t padding = 0)
        : height_(size), width_(size), padding_(padding) {}

    Tensor operator()(const Tensor& input) const override {
        thread_local std::mt19937 gen(std::random_device{}());

        Tensor padded = input;
        const int64_t ndim = input.dim();
        if (padding_ > 0) {
            std::vector<int64_t> padded_shape = input.sizes().vec();
            padded_shape[ndim - 2] += 2 * padding_;
            padded_shape[ndim - 1] += 2 * padding_;
            padded = at::zeros(padded_shape,
                at::TensorOptions().dtype(input.dtype()).device(input.device()));
            Tensor center = padded.narrow(ndim - 2, padding_, input.size(ndim - 2))
                                   .narrow(ndim - 1, padding_, input.size(ndim - 1));
            center.copy_(input);
        }

        const int64_t h = padded.size(ndim - 2);
        const int64_t w = padded.size(ndim - 1);
        PT_CHECK_MSG(h >= height_ && w >= width_, "RandomCrop: padded image too small");

        std::uniform_int_distribution<int64_t> hd(0, h - height_);
        std::uniform_int_distribution<int64_t> wd(0, w - width_);
        const int64_t top = hd(gen);
        const int64_t left = wd(gen);
        return padded.narrow(ndim - 2, top, height_)
                     .narrow(ndim - 1, left, width_)
                     .contiguous();
    }

private:
    int64_t height_, width_;
    int64_t padding_;
};

// ---------------------------------------------------------------------------
// RandomHorizontalFlip — flip last dim with probability p.
// ---------------------------------------------------------------------------
class RandomHorizontalFlip : public Transform {
public:
    explicit RandomHorizontalFlip(double p = 0.5) : p_(p) {}

    Tensor operator()(const Tensor& input) const override {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(gen) >= p_) return input;

        const int64_t ndim = input.dim();
        const int64_t w = input.size(ndim - 1);
        const int64_t outer = input.numel() / w;

        Tensor x = input.contiguous();
        Tensor out = at::empty_like(x);

        if (x.dtype() == c10::ScalarType::Float) {
            const float* src = x.data_ptr<float>();
            float* dst = out.mutable_data_ptr<float>();
            for (int64_t o = 0; o < outer; ++o)
                for (int64_t i = 0; i < w; ++i)
                    dst[o * w + i] = src[o * w + (w - 1 - i)];
        } else if (x.dtype() == c10::ScalarType::Byte) {
            const uint8_t* src = x.data_ptr<uint8_t>();
            uint8_t* dst = out.mutable_data_ptr<uint8_t>();
            for (int64_t o = 0; o < outer; ++o)
                for (int64_t i = 0; i < w; ++i)
                    dst[o * w + i] = src[o * w + (w - 1 - i)];
        } else {
            PT_CHECK_MSG(false, "RandomHorizontalFlip: dtype not supported");
        }
        return out;
    }

private:
    double p_;
};

} // namespace transforms
} // namespace vision
} // namespace torch
