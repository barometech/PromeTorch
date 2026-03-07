#pragma once

#include "aten/src/ATen/ATen.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

namespace torch {
namespace quantization {

using at::Tensor;

// ============================================================================
// Observer base class
// ============================================================================

class Observer {
public:
    virtual ~Observer() = default;

    // Observe a tensor (update statistics)
    virtual void forward(const Tensor& input) = 0;

    // Calculate scale and zero_point from observed statistics
    virtual std::pair<double, int64_t> calculate_qparams() const = 0;

    // Reset observer state
    virtual void reset() = 0;
};

// ============================================================================
// MinMaxObserver — tracks global min/max for scale/zero_point
// ============================================================================

class MinMaxObserver : public Observer {
public:
    MinMaxObserver(int64_t quant_min = 0, int64_t quant_max = 255)
        : quant_min_(quant_min), quant_max_(quant_max),
          min_val_(std::numeric_limits<float>::max()),
          max_val_(std::numeric_limits<float>::lowest()),
          initialized_(false) {}

    void forward(const Tensor& input) override {
        Tensor inp = input.contiguous();
        const float* data = inp.data_ptr<float>();
        int64_t n = inp.numel();

        for (int64_t i = 0; i < n; ++i) {
            if (data[i] < min_val_) min_val_ = data[i];
            if (data[i] > max_val_) max_val_ = data[i];
        }
        initialized_ = true;
    }

    std::pair<double, int64_t> calculate_qparams() const override {
        PT_CHECK_MSG(initialized_, "MinMaxObserver: no data observed yet");

        float min_val = min_val_;
        float max_val = max_val_;

        // Ensure zero is representable
        min_val = std::min(min_val, 0.0f);
        max_val = std::max(max_val, 0.0f);

        double scale = (static_cast<double>(max_val) - min_val) / (quant_max_ - quant_min_);
        if (scale < 1e-10) scale = 1e-10;  // avoid division by zero

        int64_t zero_point = static_cast<int64_t>(std::round(quant_min_ - min_val / scale));
        zero_point = std::max(quant_min_, std::min(quant_max_, zero_point));

        return {scale, zero_point};
    }

    void reset() override {
        min_val_ = std::numeric_limits<float>::max();
        max_val_ = std::numeric_limits<float>::lowest();
        initialized_ = false;
    }

    float min_val() const { return min_val_; }
    float max_val() const { return max_val_; }

private:
    int64_t quant_min_;
    int64_t quant_max_;
    float min_val_;
    float max_val_;
    bool initialized_;
};

// ============================================================================
// HistogramObserver — uses histogram for better quantization range
// ============================================================================

class HistogramObserver : public Observer {
public:
    HistogramObserver(int64_t bins = 2048, int64_t quant_min = 0, int64_t quant_max = 255)
        : bins_(bins), quant_min_(quant_min), quant_max_(quant_max),
          histogram_(bins, 0),
          min_val_(std::numeric_limits<float>::max()),
          max_val_(std::numeric_limits<float>::lowest()),
          initialized_(false) {}

    void forward(const Tensor& input) override {
        Tensor inp = input.contiguous();
        const float* data = inp.data_ptr<float>();
        int64_t n = inp.numel();

        // Update min/max
        for (int64_t i = 0; i < n; ++i) {
            if (data[i] < min_val_) min_val_ = data[i];
            if (data[i] > max_val_) max_val_ = data[i];
        }

        if (max_val_ <= min_val_) {
            initialized_ = true;
            return;
        }

        // Build histogram
        double bin_width = (static_cast<double>(max_val_) - min_val_) / bins_;
        for (int64_t i = 0; i < n; ++i) {
            int64_t bin = static_cast<int64_t>((data[i] - min_val_) / bin_width);
            bin = std::max((int64_t)0, std::min(bins_ - 1, bin));
            histogram_[bin]++;
        }

        initialized_ = true;
    }

    std::pair<double, int64_t> calculate_qparams() const override {
        PT_CHECK_MSG(initialized_, "HistogramObserver: no data observed yet");

        // Use percentile-based clipping (0.1% and 99.9%)
        int64_t total = 0;
        for (auto c : histogram_) total += c;

        int64_t low_threshold = static_cast<int64_t>(total * 0.001);
        int64_t high_threshold = static_cast<int64_t>(total * 0.999);

        float min_val = min_val_;
        float max_val = max_val_;
        double bin_width = (static_cast<double>(max_val_) - min_val_) / bins_;

        int64_t cumsum = 0;
        for (int64_t i = 0; i < bins_; ++i) {
            cumsum += histogram_[i];
            if (cumsum >= low_threshold) {
                min_val = static_cast<float>(min_val_ + i * bin_width);
                break;
            }
        }

        cumsum = 0;
        for (int64_t i = bins_ - 1; i >= 0; --i) {
            cumsum += histogram_[i];
            if (cumsum >= (total - high_threshold)) {
                max_val = static_cast<float>(min_val_ + (i + 1) * bin_width);
                break;
            }
        }

        min_val = std::min(min_val, 0.0f);
        max_val = std::max(max_val, 0.0f);

        double scale = (static_cast<double>(max_val) - min_val) / (quant_max_ - quant_min_);
        if (scale < 1e-10) scale = 1e-10;

        int64_t zero_point = static_cast<int64_t>(std::round(quant_min_ - min_val / scale));
        zero_point = std::max(quant_min_, std::min(quant_max_, zero_point));

        return {scale, zero_point};
    }

    void reset() override {
        std::fill(histogram_.begin(), histogram_.end(), 0);
        min_val_ = std::numeric_limits<float>::max();
        max_val_ = std::numeric_limits<float>::lowest();
        initialized_ = false;
    }

private:
    int64_t bins_;
    int64_t quant_min_;
    int64_t quant_max_;
    std::vector<int64_t> histogram_;
    float min_val_;
    float max_val_;
    bool initialized_;
};

// ============================================================================
// PerChannelMinMaxObserver — per-channel quantization
// ============================================================================

class PerChannelMinMaxObserver : public Observer {
public:
    PerChannelMinMaxObserver(int64_t axis = 0, int64_t quant_min = 0, int64_t quant_max = 255)
        : axis_(axis), quant_min_(quant_min), quant_max_(quant_max), initialized_(false) {}

    void forward(const Tensor& input) override {
        Tensor inp = input.contiguous();
        int64_t ndim = inp.dim();
        int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;
        int64_t num_channels = inp.size(axis);

        if (!initialized_) {
            min_vals_.assign(num_channels, std::numeric_limits<float>::max());
            max_vals_.assign(num_channels, std::numeric_limits<float>::lowest());
        }

        const float* data = inp.data_ptr<float>();
        int64_t n = inp.numel();

        int64_t axis_stride = 1;
        for (int64_t d = ndim - 1; d > axis; --d) {
            axis_stride *= inp.size(d);
        }

        for (int64_t i = 0; i < n; ++i) {
            int64_t channel = (i / axis_stride) % num_channels;
            if (data[i] < min_vals_[channel]) min_vals_[channel] = data[i];
            if (data[i] > max_vals_[channel]) max_vals_[channel] = data[i];
        }

        initialized_ = true;
    }

    std::pair<double, int64_t> calculate_qparams() const override {
        // Returns average scale/zp for compatibility
        PT_CHECK_MSG(initialized_, "PerChannelMinMaxObserver: no data observed");
        auto [scales, zps] = calculate_per_channel_qparams();
        double avg_scale = 0;
        const float* s = scales.data_ptr<float>();
        for (int64_t i = 0; i < scales.numel(); ++i) avg_scale += s[i];
        avg_scale /= scales.numel();
        return {avg_scale, 0};
    }

    std::pair<Tensor, Tensor> calculate_per_channel_qparams() const {
        PT_CHECK_MSG(initialized_, "PerChannelMinMaxObserver: no data observed");

        int64_t num_channels = static_cast<int64_t>(min_vals_.size());
        Tensor scales = at::empty({num_channels});
        Tensor zero_points = at::empty({num_channels}, at::TensorOptions().dtype(c10::ScalarType::Long));

        float* sc = scales.mutable_data_ptr<float>();
        int64_t* zp = zero_points.mutable_data_ptr<int64_t>();

        for (int64_t c = 0; c < num_channels; ++c) {
            float mn = std::min(min_vals_[c], 0.0f);
            float mx = std::max(max_vals_[c], 0.0f);

            double scale = (static_cast<double>(mx) - mn) / (quant_max_ - quant_min_);
            if (scale < 1e-10) scale = 1e-10;

            int64_t z = static_cast<int64_t>(std::round(quant_min_ - mn / scale));
            z = std::max(quant_min_, std::min(quant_max_, z));

            sc[c] = static_cast<float>(scale);
            zp[c] = z;
        }

        return {scales, zero_points};
    }

    void reset() override {
        min_vals_.clear();
        max_vals_.clear();
        initialized_ = false;
    }

    int64_t axis() const { return axis_; }

private:
    int64_t axis_;
    int64_t quant_min_;
    int64_t quant_max_;
    std::vector<float> min_vals_;
    std::vector<float> max_vals_;
    bool initialized_;
};

} // namespace quantization
} // namespace torch
