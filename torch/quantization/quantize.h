#pragma once

#include "aten/src/ATen/ATen.h"
#include <cmath>
#include <algorithm>

namespace torch {
namespace quantization {

using at::Tensor;
using at::TensorOptions;

// ============================================================================
// QuantizedTensor — holds quantized data + scale/zero_point
// ============================================================================

struct QuantizedTensor {
    Tensor int_repr_;      // uint8 or int8 data
    double scale_;
    int64_t zero_point_;
    bool per_channel_;
    Tensor scales_;        // per-channel scales
    Tensor zero_points_;   // per-channel zero points
    int64_t axis_;         // quantization axis (for per-channel)
    std::vector<int64_t> original_shape_;
    c10::ScalarType original_dtype_;

    // Per-tensor constructor
    QuantizedTensor(Tensor int_repr, double scale, int64_t zero_point,
                    std::vector<int64_t> shape, c10::ScalarType dtype)
        : int_repr_(std::move(int_repr)), scale_(scale), zero_point_(zero_point),
          per_channel_(false), axis_(0),
          original_shape_(std::move(shape)), original_dtype_(dtype) {}

    // Per-channel constructor
    QuantizedTensor(Tensor int_repr, Tensor scales, Tensor zero_points,
                    int64_t axis, std::vector<int64_t> shape, c10::ScalarType dtype)
        : int_repr_(std::move(int_repr)), scale_(0), zero_point_(0),
          per_channel_(true), scales_(std::move(scales)),
          zero_points_(std::move(zero_points)), axis_(axis),
          original_shape_(std::move(shape)), original_dtype_(dtype) {}

    // Dequantize: convert back to float
    Tensor dequantize() const {
        Tensor result = at::empty(original_shape_, TensorOptions().dtype(original_dtype_));

        if (!per_channel_) {
            // Per-tensor dequantization: float = (q - zero_point) * scale
            const uint8_t* q_data = int_repr_.data_ptr<uint8_t>();
            float* f_data = result.mutable_data_ptr<float>();
            int64_t n = result.numel();

            for (int64_t i = 0; i < n; ++i) {
                f_data[i] = static_cast<float>((static_cast<int32_t>(q_data[i]) - zero_point_) * scale_);
            }
        } else {
            // Per-channel dequantization
            const uint8_t* q_data = int_repr_.data_ptr<uint8_t>();
            float* f_data = result.mutable_data_ptr<float>();
            const float* sc = scales_.data_ptr<float>();
            const int64_t* zp = zero_points_.data_ptr<int64_t>();

            int64_t n = result.numel();
            int64_t ndim = static_cast<int64_t>(original_shape_.size());
            int64_t axis_size = original_shape_[axis_];

            // Compute stride for axis
            int64_t axis_stride = 1;
            for (int64_t d = ndim - 1; d > axis_; --d) {
                axis_stride *= original_shape_[d];
            }

            for (int64_t i = 0; i < n; ++i) {
                int64_t channel = (i / axis_stride) % axis_size;
                f_data[i] = static_cast<float>((static_cast<int32_t>(q_data[i]) - zp[channel]) * sc[channel]);
            }
        }

        return result;
    }

    // Get the int representation
    const Tensor& int_repr() const { return int_repr_; }
    double q_scale() const { return scale_; }
    int64_t q_zero_point() const { return zero_point_; }
};

// ============================================================================
// quantize_per_tensor — quantize a float tensor to uint8
// ============================================================================

inline QuantizedTensor quantize_per_tensor(const Tensor& self, double scale, int64_t zero_point) {
    Tensor input = self.contiguous();
    int64_t n = input.numel();

    Tensor q = at::empty(input.sizes(), TensorOptions().dtype(c10::ScalarType::Byte));

    const float* f_data = input.data_ptr<float>();
    uint8_t* q_data = q.mutable_data_ptr<uint8_t>();

    for (int64_t i = 0; i < n; ++i) {
        int32_t val = static_cast<int32_t>(std::round(f_data[i] / scale)) + static_cast<int32_t>(zero_point);
        q_data[i] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
    }

    return QuantizedTensor(q, scale, zero_point, input.sizes().vec(), input.dtype());
}

// ============================================================================
// quantize_per_channel — per-channel quantization
// ============================================================================

inline QuantizedTensor quantize_per_channel(const Tensor& self, const Tensor& scales,
                                             const Tensor& zero_points, int64_t axis) {
    Tensor input = self.contiguous();
    int64_t n = input.numel();
    int64_t ndim = input.dim();

    if (axis < 0) axis += ndim;
    int64_t axis_size = input.size(axis);

    PT_CHECK_MSG(scales.numel() == axis_size, "scales must match axis size");
    PT_CHECK_MSG(zero_points.numel() == axis_size, "zero_points must match axis size");

    Tensor q = at::empty(input.sizes(), TensorOptions().dtype(c10::ScalarType::Byte));

    const float* f_data = input.data_ptr<float>();
    uint8_t* q_data = q.mutable_data_ptr<uint8_t>();
    const float* sc = scales.contiguous().data_ptr<float>();
    const int64_t* zp = zero_points.contiguous().data_ptr<int64_t>();

    int64_t axis_stride = 1;
    for (int64_t d = ndim - 1; d > axis; --d) {
        axis_stride *= input.size(d);
    }

    for (int64_t i = 0; i < n; ++i) {
        int64_t channel = (i / axis_stride) % axis_size;
        int32_t val = static_cast<int32_t>(std::round(f_data[i] / sc[channel])) + static_cast<int32_t>(zp[channel]);
        q_data[i] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
    }

    return QuantizedTensor(q, scales.clone(), zero_points.clone(), axis,
                           input.sizes().vec(), input.dtype());
}

} // namespace quantization
} // namespace torch
