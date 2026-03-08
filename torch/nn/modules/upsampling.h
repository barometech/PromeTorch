#pragma once

#include "torch/nn/module.h"
#include "torch/nn/functional.h"

namespace torch {
namespace nn {

// ============================================================================
// Upsample - Upsamples a given multi-channel 2D data
// ============================================================================
// Applies nearest-neighbor or bilinear upsampling to spatial dimensions.
// Input: (N, C, H, W) or any tensor with at least 2 spatial dims at the end.
// Either size or scale_factor must be provided (not both).

class Upsample : public Module {
public:
    // Constructor with target size
    Upsample(std::vector<int64_t> size, const std::string& mode = "nearest", bool align_corners = false)
        : Module("Upsample"), size_(std::move(size)), mode_(mode), align_corners_(align_corners) {}

    // Constructor with single scale factor (applied to both H and W)
    Upsample(double scale_factor, const std::string& mode = "nearest", bool align_corners = false)
        : Module("Upsample"), scale_factor_({scale_factor}), mode_(mode), align_corners_(align_corners) {}

    // Constructor with per-dimension scale factors
    Upsample(std::vector<double> scale_factor, const std::string& mode = "nearest", bool align_corners = false)
        : Module("Upsample"), scale_factor_(std::move(scale_factor)), mode_(mode), align_corners_(align_corners) {}

    Tensor forward(const Tensor& input) override {
        return functional::interpolate(input, size_, scale_factor_, mode_, align_corners_);
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        if (!size_.empty()) {
            ss << "size=(";
            for (size_t i = 0; i < size_.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << size_[i];
            }
            ss << ")";
        }
        if (!scale_factor_.empty()) {
            ss << "scale_factor=(";
            for (size_t i = 0; i < scale_factor_.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << scale_factor_[i];
            }
            ss << ")";
        }
        ss << ", mode=" << mode_;
        if (mode_ == "bilinear") {
            ss << ", align_corners=" << (align_corners_ ? "True" : "False");
        }
        return ss.str();
    }

private:
    std::vector<int64_t> size_;
    std::vector<double> scale_factor_;
    std::string mode_;
    bool align_corners_;
};

} // namespace nn
} // namespace torch
