#pragma once

#define _USE_MATH_DEFINES

#include "aten/src/ATen/ATen.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace torch {
namespace data {
namespace transforms {

using at::Tensor;
using at::Scalar;

// ============================================================================
// Transform - Base class for all transforms
// ============================================================================

class Transform {
public:
    virtual ~Transform() = default;
    virtual Tensor operator()(const Tensor& input) const = 0;
};

// ============================================================================
// Compose - Chain multiple transforms
// ============================================================================

class Compose : public Transform {
public:
    Compose(std::vector<std::shared_ptr<Transform>> transforms)
        : transforms_(std::move(transforms)) {}

    Tensor operator()(const Tensor& input) const override {
        Tensor result = input;
        for (const auto& t : transforms_) {
            result = (*t)(result);
        }
        return result;
    }

private:
    std::vector<std::shared_ptr<Transform>> transforms_;
};

// ============================================================================
// ToTensor - Convert uint8 [H,W,C] to float [C,H,W], scale to [0,1]
// ============================================================================

class ToTensor : public Transform {
public:
    Tensor operator()(const Tensor& input) const override {
        Tensor result = input;

        // If input is integer type, convert to float and divide by 255
        if (input.dtype() != c10::ScalarType::Float && input.dtype() != c10::ScalarType::Double) {
            result = result.to(c10::ScalarType::Float);
            result = result.div(Scalar(255.0));
        }

        // If HWC format (3 dims with last dim as channels), permute to CHW
        if (result.dim() == 3) {
            // Assume HWC -> CHW
            result = result.permute({2, 0, 1}).contiguous();
        }

        return result;
    }
};

// ============================================================================
// Normalize - Per-channel normalization: (x - mean) / std
// ============================================================================

class Normalize : public Transform {
public:
    Normalize(std::vector<float> mean, std::vector<float> std_dev)
        : mean_(std::move(mean)), std_(std::move(std_dev)) {}

    Tensor operator()(const Tensor& input) const override {
        Tensor result = input.clone();

        PT_CHECK(input.dim() >= 1);

        // For CHW (3D) or NCHW (4D): normalize along channel dimension
        int64_t channels = input.size(0);  // C is first dim for CHW
        PT_CHECK(static_cast<size_t>(channels) <= mean_.size());

        for (int64_t c = 0; c < channels; ++c) {
            // Select channel slice
            float m = (c < static_cast<int64_t>(mean_.size())) ? mean_[c] : 0.0f;
            float s = (c < static_cast<int64_t>(std_.size())) ? std_[c] : 1.0f;

            // (channel - mean) / std
            Tensor ch_contig = result.select(0, c).contiguous();
            const float* src = ch_contig.data_ptr<float>();
            int64_t n = ch_contig.numel();

            // Create new data for this channel
            Tensor normalized = at::empty(ch_contig.sizes(), at::TensorOptions().dtype(c10::ScalarType::Float));
            float* dst = normalized.mutable_data_ptr<float>();
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = (src[i] - m) / s;
            }

            // Copy back - use the view into result
            Tensor result_channel = result.select(0, c);
            result_channel.copy_(normalized);
        }

        return result;
    }

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

// ============================================================================
// RandomHorizontalFlip - Flip image with probability p
// ============================================================================

class RandomHorizontalFlip : public Transform {
public:
    explicit RandomHorizontalFlip(double p = 0.5) : p_(p) {}

    Tensor operator()(const Tensor& input) const override {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        if (dist(gen) < p_) {
            // Flip along width dimension (last dim)
            // For CHW: flip dim 2
            // For HW: flip dim 1
            int64_t flip_dim = input.dim() - 1;
            int64_t w = input.size(flip_dim);

            Tensor result = at::empty_like(input);
            Tensor in_contig = input.contiguous();

            // General flip via slicing
            // Create reversed copy along flip_dim
            PT_DISPATCH_ALL_TYPES(input.dtype(), "random_hflip", [&] {
                const scalar_t* src = in_contig.data_ptr<scalar_t>();
                scalar_t* dst = result.mutable_data_ptr<scalar_t>();

                // Calculate outer stride relative to flip_dim
                int64_t outer = input.numel() / w;

                // For each row of elements along the flip axis
                for (int64_t o = 0; o < outer; ++o) {
                    for (int64_t i = 0; i < w; ++i) {
                        dst[o * w + i] = src[o * w + (w - 1 - i)];
                    }
                }
            });

            return result;
        }
        return input;
    }

private:
    double p_;
};

// ============================================================================
// RandomVerticalFlip - Flip image vertically with probability p
// ============================================================================

class RandomVerticalFlip : public Transform {
public:
    explicit RandomVerticalFlip(double p = 0.5) : p_(p) {}

    Tensor operator()(const Tensor& input) const override {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        if (dist(gen) < p_) {
            // Flip along height dimension (second-to-last dim)
            int64_t flip_dim = input.dim() - 2;
            if (flip_dim < 0) flip_dim = 0;
            int64_t h = input.size(flip_dim);

            Tensor result = at::empty_like(input);
            Tensor in_contig = input.contiguous();

            PT_DISPATCH_ALL_TYPES(input.dtype(), "random_vflip", [&] {
                const scalar_t* src = in_contig.data_ptr<scalar_t>();
                scalar_t* dst = result.mutable_data_ptr<scalar_t>();

                int64_t outer = 1;
                for (int64_t d = 0; d < flip_dim; ++d) outer *= input.size(d);
                int64_t inner = 1;
                for (int64_t d = flip_dim + 1; d < input.dim(); ++d) inner *= input.size(d);

                for (int64_t o = 0; o < outer; ++o) {
                    for (int64_t i = 0; i < h; ++i) {
                        int64_t src_offset = (o * h + (h - 1 - i)) * inner;
                        int64_t dst_offset = (o * h + i) * inner;
                        std::memcpy(dst + dst_offset, src + src_offset, inner * sizeof(scalar_t));
                    }
                }
            });

            return result;
        }
        return input;
    }

private:
    double p_;
};

// ============================================================================
// RandomCrop - Random crop with optional padding
// ============================================================================

class RandomCrop : public Transform {
public:
    RandomCrop(int64_t height, int64_t width, int64_t padding = 0)
        : height_(height), width_(width), padding_(padding) {}

    // Single size (square crop)
    explicit RandomCrop(int64_t size, int64_t padding = 0)
        : height_(size), width_(size), padding_(padding) {}

    Tensor operator()(const Tensor& input) const override {
        thread_local std::mt19937 gen(std::random_device{}());

        Tensor padded = input;

        // Apply zero-padding if needed
        if (padding_ > 0) {
            // Input is CHW or HW
            int64_t ndim = input.dim();
            std::vector<int64_t> padded_shape = input.sizes().vec();
            padded_shape[ndim - 2] += 2 * padding_;
            padded_shape[ndim - 1] += 2 * padding_;

            padded = at::zeros(padded_shape, at::TensorOptions().dtype(input.dtype()).device(input.device()));

            // Copy input into center of padded tensor
            // Use narrow to get the center region and copy into it
            Tensor center = padded.narrow(ndim - 2, padding_, input.size(ndim - 2))
                                   .narrow(ndim - 1, padding_, input.size(ndim - 1));
            center.copy_(input);
        }

        int64_t h = padded.size(padded.dim() - 2);
        int64_t w = padded.size(padded.dim() - 1);

        PT_CHECK_MSG(h >= height_ && w >= width_,
            "RandomCrop: image size (", h, "x", w,
            ") is smaller than crop size (", height_, "x", width_, ")");

        std::uniform_int_distribution<int64_t> h_dist(0, h - height_);
        std::uniform_int_distribution<int64_t> w_dist(0, w - width_);

        int64_t top = h_dist(gen);
        int64_t left = w_dist(gen);

        // Crop using narrow
        int64_t ndim = padded.dim();
        return padded.narrow(ndim - 2, top, height_)
                      .narrow(ndim - 1, left, width_)
                      .contiguous();
    }

private:
    int64_t height_, width_;
    int64_t padding_;
};

// ============================================================================
// CenterCrop - Center crop
// ============================================================================

class CenterCrop : public Transform {
public:
    CenterCrop(int64_t height, int64_t width)
        : height_(height), width_(width) {}

    explicit CenterCrop(int64_t size)
        : height_(size), width_(size) {}

    Tensor operator()(const Tensor& input) const override {
        int64_t ndim = input.dim();
        int64_t h = input.size(ndim - 2);
        int64_t w = input.size(ndim - 1);

        int64_t top = (h - height_) / 2;
        int64_t left = (w - width_) / 2;

        return input.narrow(ndim - 2, top, height_)
                     .narrow(ndim - 1, left, width_)
                     .contiguous();
    }

private:
    int64_t height_, width_;
};

// ============================================================================
// Resize - Resize to target size (nearest neighbor interpolation)
// ============================================================================

class Resize : public Transform {
public:
    Resize(int64_t height, int64_t width, const std::string& mode = "nearest")
        : height_(height), width_(width), mode_(mode) {}

    explicit Resize(int64_t size, const std::string& mode = "nearest")
        : height_(size), width_(size), mode_(mode) {}

    Tensor operator()(const Tensor& input) const override {
        int64_t ndim = input.dim();
        int64_t in_h = input.size(ndim - 2);
        int64_t in_w = input.size(ndim - 1);

        std::vector<int64_t> out_shape = input.sizes().vec();
        out_shape[ndim - 2] = height_;
        out_shape[ndim - 1] = width_;

        Tensor result = at::empty(out_shape, at::TensorOptions().dtype(input.dtype()).device(input.device()));
        Tensor in_contig = input.contiguous();

        if (mode_ == "nearest") {
            PT_DISPATCH_FLOATING_TYPES(input.dtype(), "resize_nearest", [&] {
                const scalar_t* src = in_contig.data_ptr<scalar_t>();
                scalar_t* dst = result.mutable_data_ptr<scalar_t>();

                int64_t outer = 1;
                for (int64_t d = 0; d < ndim - 2; ++d) outer *= input.size(d);

                for (int64_t o = 0; o < outer; ++o) {
                    for (int64_t oh = 0; oh < height_; ++oh) {
                        int64_t ih = static_cast<int64_t>(std::floor(static_cast<double>(oh) * in_h / height_));
                        ih = std::min(ih, in_h - 1);
                        for (int64_t ow = 0; ow < width_; ++ow) {
                            int64_t iw = static_cast<int64_t>(std::floor(static_cast<double>(ow) * in_w / width_));
                            iw = std::min(iw, in_w - 1);
                            dst[o * height_ * width_ + oh * width_ + ow] =
                                src[o * in_h * in_w + ih * in_w + iw];
                        }
                    }
                }
            });
        } else if (mode_ == "bilinear") {
            PT_DISPATCH_FLOATING_TYPES(input.dtype(), "resize_bilinear", [&] {
                const scalar_t* src = in_contig.data_ptr<scalar_t>();
                scalar_t* dst = result.mutable_data_ptr<scalar_t>();

                int64_t outer = 1;
                for (int64_t d = 0; d < ndim - 2; ++d) outer *= input.size(d);

                for (int64_t o = 0; o < outer; ++o) {
                    for (int64_t oh = 0; oh < height_; ++oh) {
                        double src_h = (static_cast<double>(oh) + 0.5) * in_h / height_ - 0.5;
                        int64_t h0 = std::max(static_cast<int64_t>(std::floor(src_h)), (int64_t)0);
                        int64_t h1 = std::min(h0 + 1, in_h - 1);
                        double dh = src_h - h0;
                        if (dh < 0) dh = 0;

                        for (int64_t ow = 0; ow < width_; ++ow) {
                            double src_w = (static_cast<double>(ow) + 0.5) * in_w / width_ - 0.5;
                            int64_t w0 = std::max(static_cast<int64_t>(std::floor(src_w)), (int64_t)0);
                            int64_t w1 = std::min(w0 + 1, in_w - 1);
                            double dw = src_w - w0;
                            if (dw < 0) dw = 0;

                            scalar_t v00 = src[o * in_h * in_w + h0 * in_w + w0];
                            scalar_t v01 = src[o * in_h * in_w + h0 * in_w + w1];
                            scalar_t v10 = src[o * in_h * in_w + h1 * in_w + w0];
                            scalar_t v11 = src[o * in_h * in_w + h1 * in_w + w1];

                            dst[o * height_ * width_ + oh * width_ + ow] = static_cast<scalar_t>(
                                (1.0 - dh) * (1.0 - dw) * v00 +
                                (1.0 - dh) * dw * v01 +
                                dh * (1.0 - dw) * v10 +
                                dh * dw * v11
                            );
                        }
                    }
                }
            });
        }

        return result;
    }

private:
    int64_t height_, width_;
    std::string mode_;
};

// ============================================================================
// ColorJitter - Random brightness, contrast, saturation, hue adjustments
// ============================================================================

class ColorJitter : public Transform {
public:
    ColorJitter(float brightness = 0.0f, float contrast = 0.0f,
                float saturation = 0.0f, float hue = 0.0f)
        : brightness_(brightness), contrast_(contrast)
        , saturation_(saturation), hue_(hue) {}

    Tensor operator()(const Tensor& input) const override {
        thread_local std::mt19937 gen(std::random_device{}());
        Tensor result = input.clone();

        // Brightness: multiply by random factor
        if (brightness_ > 0) {
            std::uniform_real_distribution<float> dist(
                std::max(0.0f, 1.0f - brightness_), 1.0f + brightness_);
            float factor = dist(gen);
            result = result.mul(Scalar(factor));
        }

        // Contrast: blend with mean (grayscale)
        if (contrast_ > 0) {
            std::uniform_real_distribution<float> dist(
                std::max(0.0f, 1.0f - contrast_), 1.0f + contrast_);
            float factor = dist(gen);
            double mean_val = result.mean().item().toDouble();
            // result = factor * result + (1 - factor) * mean
            result = result.mul(Scalar(factor)).add(
                at::full(result.sizes(), Scalar(mean_val * (1.0 - factor)),
                         at::TensorOptions().dtype(result.dtype())));
        }

        return result;
    }

private:
    float brightness_, contrast_, saturation_, hue_;
};

// ============================================================================
// Grayscale - Convert to grayscale
// ============================================================================

class Grayscale : public Transform {
public:
    explicit Grayscale(int64_t num_output_channels = 1)
        : num_output_channels_(num_output_channels) {}

    Tensor operator()(const Tensor& input) const override {
        PT_CHECK(input.dim() >= 2);

        if (input.dim() == 2) return input;  // Already grayscale

        // CHW format: average over channel dimension
        // gray = 0.299*R + 0.587*G + 0.114*B
        Tensor gray;
        if (input.size(0) == 3) {
            Tensor r = input.select(0, 0);
            Tensor g = input.select(0, 1);
            Tensor b = input.select(0, 2);
            gray = r.mul(Scalar(0.299f)).add(g.mul(Scalar(0.587f))).add(b.mul(Scalar(0.114f)));
        } else {
            gray = input.mean(0);  // Average over channels
        }

        if (num_output_channels_ == 1) {
            return gray.unsqueeze(0);  // [1, H, W]
        } else if (num_output_channels_ == 3) {
            // Stack same grayscale 3 times
            return torch::stack({gray, gray, gray}, 0);
        }

        return gray.unsqueeze(0);
    }

private:
    int64_t num_output_channels_;
};

// ============================================================================
// RandomErasing - Random rectangle erasing
// ============================================================================

class RandomErasing : public Transform {
public:
    RandomErasing(double p = 0.5, double scale_min = 0.02, double scale_max = 0.33,
                  double ratio_min = 0.3, double ratio_max = 3.3, float value = 0.0f)
        : p_(p), scale_min_(scale_min), scale_max_(scale_max)
        , ratio_min_(ratio_min), ratio_max_(ratio_max), value_(value) {}

    Tensor operator()(const Tensor& input) const override {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> prob(0.0, 1.0);

        if (prob(gen) >= p_) return input;

        Tensor result = input.clone();
        int64_t ndim = input.dim();
        int64_t h = input.size(ndim - 2);
        int64_t w = input.size(ndim - 1);
        int64_t area = h * w;

        std::uniform_real_distribution<double> scale_dist(scale_min_, scale_max_);
        std::uniform_real_distribution<double> ratio_dist(std::log(ratio_min_), std::log(ratio_max_));

        for (int attempt = 0; attempt < 10; ++attempt) {
            double target_area = scale_dist(gen) * area;
            double aspect_ratio = std::exp(ratio_dist(gen));

            int64_t eh = static_cast<int64_t>(std::round(std::sqrt(target_area * aspect_ratio)));
            int64_t ew = static_cast<int64_t>(std::round(std::sqrt(target_area / aspect_ratio)));

            if (eh < h && ew < w) {
                std::uniform_int_distribution<int64_t> h_dist(0, h - eh);
                std::uniform_int_distribution<int64_t> w_dist(0, w - ew);
                int64_t top = h_dist(gen);
                int64_t left = w_dist(gen);

                // Fill rectangle with value
                Tensor region = result.narrow(ndim - 2, top, eh).narrow(ndim - 1, left, ew);
                region.fill_(Scalar(value_));
                break;
            }
        }

        return result;
    }

private:
    double p_, scale_min_, scale_max_, ratio_min_, ratio_max_;
    float value_;
};

// ============================================================================
// RandomRotation - Rotate image by random angle
// ============================================================================

class RandomRotation : public Transform {
public:
    explicit RandomRotation(double degrees) : degrees_(std::abs(degrees)) {}

    Tensor operator()(const Tensor& input) const override {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-degrees_, degrees_);
        double angle = dist(gen) * M_PI / 180.0;

        int64_t ndim = input.dim();
        int64_t h = input.size(ndim - 2);
        int64_t w = input.size(ndim - 1);

        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);
        double cx = w / 2.0;
        double cy = h / 2.0;

        Tensor result = at::zeros_like(input);
        Tensor in_contig = input.contiguous();

        PT_DISPATCH_FLOATING_TYPES(input.dtype(), "random_rotation", [&] {
            const scalar_t* src = in_contig.data_ptr<scalar_t>();
            scalar_t* dst = result.mutable_data_ptr<scalar_t>();

            int64_t outer = 1;
            for (int64_t d = 0; d < ndim - 2; ++d) outer *= input.size(d);

            for (int64_t o = 0; o < outer; ++o) {
                for (int64_t oh = 0; oh < h; ++oh) {
                    for (int64_t ow = 0; ow < w; ++ow) {
                        // Rotate backwards from dst to src
                        double dx = ow - cx;
                        double dy = oh - cy;
                        double sx = cos_a * dx + sin_a * dy + cx;
                        double sy = -sin_a * dx + cos_a * dy + cy;

                        int64_t ix = static_cast<int64_t>(std::round(sx));
                        int64_t iy = static_cast<int64_t>(std::round(sy));

                        if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
                            dst[o * h * w + oh * w + ow] = src[o * h * w + iy * w + ix];
                        }
                    }
                }
            }
        });

        return result;
    }

private:
    double degrees_;
};

// ============================================================================
// Lambda - Apply custom function transform
// ============================================================================

class Lambda : public Transform {
public:
    explicit Lambda(std::function<Tensor(const Tensor&)> fn)
        : fn_(std::move(fn)) {}

    Tensor operator()(const Tensor& input) const override {
        return fn_(input);
    }

private:
    std::function<Tensor(const Tensor&)> fn_;
};

} // namespace transforms
} // namespace data
} // namespace torch
