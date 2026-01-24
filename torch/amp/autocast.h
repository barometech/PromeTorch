#pragma once

// ============================================================================
// Autocast for Automatic Mixed Precision
// ============================================================================
// Automatically cast operations to lower precision (FP16/BF16) where safe.
//
// Usage:
//   {
//       AutocastGuard guard(c10::ScalarType::Half);
//       // All eligible ops will run in FP16
//       auto y = model(x);  // FP16 matmuls, convolutions
//   }
//   // Back to FP32
//
// Or with enable/disable:
//   set_autocast_enabled(true);
//   auto y = model(x);
//   set_autocast_enabled(false);

#include "c10/core/ScalarType.h"
#include "c10/core/Device.h"
#include <stack>

namespace torch {
namespace amp {

// ============================================================================
// Thread-Local Autocast State
// ============================================================================

namespace detail {

// Autocast state for a single device type
struct AutocastState {
    bool enabled = false;
    c10::ScalarType dtype = c10::ScalarType::Half;  // Default: FP16
    bool cache_enabled = true;
};

// Get thread-local autocast state
inline AutocastState& get_autocast_state(c10::DeviceType device_type) {
    // Thread-local storage for each device type
    thread_local AutocastState cuda_state;
    thread_local AutocastState cpu_state;

    switch (device_type) {
        case c10::DeviceType::CUDA:
            return cuda_state;
        case c10::DeviceType::CPU:
            return cpu_state;
        default:
            PT_ERROR("Autocast not supported for device: ", c10::deviceTypeToString(device_type));
    }
}

} // namespace detail

// ============================================================================
// Autocast Mode API
// ============================================================================

// Check if autocast is enabled for CUDA
inline bool is_autocast_enabled() {
    return detail::get_autocast_state(c10::DeviceType::CUDA).enabled;
}

// Check if autocast is enabled for specific device type
inline bool is_autocast_enabled(c10::DeviceType device_type) {
    return detail::get_autocast_state(device_type).enabled;
}

// Enable/disable autocast for CUDA
inline void set_autocast_enabled(bool enabled) {
    detail::get_autocast_state(c10::DeviceType::CUDA).enabled = enabled;
}

// Enable/disable autocast for specific device type
inline void set_autocast_enabled(c10::DeviceType device_type, bool enabled) {
    detail::get_autocast_state(device_type).enabled = enabled;
}

// Get autocast dtype for CUDA
inline c10::ScalarType get_autocast_dtype() {
    return detail::get_autocast_state(c10::DeviceType::CUDA).dtype;
}

// Get autocast dtype for specific device type
inline c10::ScalarType get_autocast_dtype(c10::DeviceType device_type) {
    return detail::get_autocast_state(device_type).dtype;
}

// Set autocast dtype for CUDA
inline void set_autocast_dtype(c10::ScalarType dtype) {
    detail::get_autocast_state(c10::DeviceType::CUDA).dtype = dtype;
}

// Set autocast dtype for specific device type
inline void set_autocast_dtype(c10::DeviceType device_type, c10::ScalarType dtype) {
    detail::get_autocast_state(device_type).dtype = dtype;
}

// ============================================================================
// Autocast CPU Mode API
// ============================================================================

inline bool is_autocast_cpu_enabled() {
    return detail::get_autocast_state(c10::DeviceType::CPU).enabled;
}

inline void set_autocast_cpu_enabled(bool enabled) {
    detail::get_autocast_state(c10::DeviceType::CPU).enabled = enabled;
}

inline c10::ScalarType get_autocast_cpu_dtype() {
    return detail::get_autocast_state(c10::DeviceType::CPU).dtype;
}

inline void set_autocast_cpu_dtype(c10::ScalarType dtype) {
    detail::get_autocast_state(c10::DeviceType::CPU).dtype = dtype;
}

// ============================================================================
// AutocastGuard (RAII)
// ============================================================================

class AutocastGuard {
public:
    // Enable autocast for CUDA with specified dtype
    explicit AutocastGuard(
        c10::ScalarType dtype = c10::ScalarType::Half,
        bool enabled = true,
        c10::DeviceType device_type = c10::DeviceType::CUDA
    ) : device_type_(device_type) {
        auto& state = detail::get_autocast_state(device_type_);
        prev_enabled_ = state.enabled;
        prev_dtype_ = state.dtype;

        state.enabled = enabled;
        state.dtype = dtype;
    }

    // Disable copy
    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;

    // Move is OK
    AutocastGuard(AutocastGuard&& other) noexcept
        : device_type_(other.device_type_),
          prev_enabled_(other.prev_enabled_),
          prev_dtype_(other.prev_dtype_) {
        other.moved_ = true;
    }

    ~AutocastGuard() {
        if (!moved_) {
            auto& state = detail::get_autocast_state(device_type_);
            state.enabled = prev_enabled_;
            state.dtype = prev_dtype_;
        }
    }

private:
    c10::DeviceType device_type_;
    bool prev_enabled_;
    c10::ScalarType prev_dtype_;
    bool moved_ = false;
};

// ============================================================================
// Autocast CPU Guard
// ============================================================================

class AutocastCPUGuard {
public:
    explicit AutocastCPUGuard(
        c10::ScalarType dtype = c10::ScalarType::BFloat16,
        bool enabled = true
    ) : guard_(dtype, enabled, c10::DeviceType::CPU) {}

private:
    AutocastGuard guard_;
};

// ============================================================================
// Operation Categories for Autocast
// ============================================================================
// Operations are categorized by their precision requirements:
//
// 1. FP16-safe ops (should be cast to FP16):
//    - matmul, conv2d, linear, bmm
//    - These benefit from Tensor Cores
//
// 2. FP32-required ops (must stay in FP32):
//    - softmax, log_softmax (numerical stability)
//    - batch_norm, layer_norm (statistics)
//    - loss functions
//    - exp, log, pow (precision)
//
// 3. Promotion ops (output in highest precision of inputs):
//    - add, sub, mul, div
//    - cat, stack

enum class AutocastCategory {
    // Run in lower precision (FP16/BF16)
    LowerPrecision,

    // Must run in FP32 for numerical stability
    FP32Required,

    // Promote to highest precision input
    Promote,

    // No change
    Unchanged
};

// Get autocast category for an operation
inline AutocastCategory get_autocast_category(const std::string& op_name) {
    // FP16-safe operations (benefit from Tensor Cores)
    static const std::unordered_set<std::string> lower_precision_ops = {
        "mm", "matmul", "bmm", "addmm", "addmv",
        "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
        "linear",
        "prelu",
        "baddbmm", "addbmm",
        // Attention
        "scaled_dot_product_attention",
        "_scaled_dot_product_efficient_attention",
        // RNN
        "gru", "gru_cell", "lstm", "lstm_cell",
        "rnn_relu", "rnn_relu_cell", "rnn_tanh", "rnn_tanh_cell",
    };

    // FP32-required operations
    static const std::unordered_set<std::string> fp32_ops = {
        // Normalization
        "softmax", "log_softmax",
        "batch_norm", "layer_norm", "group_norm", "instance_norm",
        // Loss
        "cross_entropy", "nll_loss", "mse_loss", "l1_loss",
        "binary_cross_entropy", "binary_cross_entropy_with_logits",
        "kl_div", "smooth_l1_loss", "huber_loss",
        // Precision-sensitive
        "exp", "exp2", "expm1",
        "log", "log2", "log10", "log1p",
        "pow", "rsqrt", "reciprocal",
        "cumprod", "cumsum",
        // Reduction with accumulation
        "sum", "mean", "prod",
        "var", "std",
        "norm",
    };

    // Promote ops
    static const std::unordered_set<std::string> promote_ops = {
        "add", "sub", "mul", "div", "true_divide", "floor_divide",
        "cat", "stack",
        "index_put", "scatter", "gather",
        "where",
    };

    if (lower_precision_ops.count(op_name)) {
        return AutocastCategory::LowerPrecision;
    } else if (fp32_ops.count(op_name)) {
        return AutocastCategory::FP32Required;
    } else if (promote_ops.count(op_name)) {
        return AutocastCategory::Promote;
    } else {
        return AutocastCategory::Unchanged;
    }
}

// ============================================================================
// Tensor Casting Utilities
// ============================================================================

// Cast tensor to autocast dtype if autocast is enabled
inline at::Tensor autocast_cast(
    const at::Tensor& tensor,
    AutocastCategory category,
    c10::DeviceType device_type = c10::DeviceType::CUDA
) {
    if (!is_autocast_enabled(device_type)) {
        return tensor;
    }

    auto autocast_dtype = get_autocast_dtype(device_type);

    switch (category) {
        case AutocastCategory::LowerPrecision:
            // Cast to lower precision if not already
            if (tensor.dtype() != autocast_dtype &&
                c10::isFloatingType(tensor.dtype())) {
                return tensor.to(autocast_dtype);
            }
            break;

        case AutocastCategory::FP32Required:
            // Cast to FP32 if not already
            if (tensor.dtype() != c10::ScalarType::Float &&
                c10::isFloatingType(tensor.dtype())) {
                return tensor.to(c10::ScalarType::Float);
            }
            break;

        case AutocastCategory::Promote:
        case AutocastCategory::Unchanged:
            // No change
            break;
    }

    return tensor;
}

// Promote multiple tensors to their common dtype
inline c10::ScalarType promote_types(
    const std::vector<at::Tensor>& tensors
) {
    c10::ScalarType result = c10::ScalarType::Half;

    for (const auto& t : tensors) {
        if (t.defined()) {
            auto t_dtype = t.dtype();
            // FP32/FP64 > BF16 > FP16
            if (t_dtype == c10::ScalarType::Double) {
                return c10::ScalarType::Double;
            } else if (t_dtype == c10::ScalarType::Float) {
                result = c10::ScalarType::Float;
            } else if (t_dtype == c10::ScalarType::BFloat16 &&
                       result == c10::ScalarType::Half) {
                result = c10::ScalarType::BFloat16;
            }
        }
    }

    return result;
}

// ============================================================================
// Autocast-aware Operations
// ============================================================================

// Example: autocast-aware matmul
inline at::Tensor autocast_matmul(
    const at::Tensor& a,
    const at::Tensor& b
) {
    if (is_autocast_enabled(a.device().type())) {
        auto dtype = get_autocast_dtype(a.device().type());
        at::Tensor a_cast = (a.dtype() != dtype) ? a.to(dtype) : a;
        at::Tensor b_cast = (b.dtype() != dtype) ? b.to(dtype) : b;
        return at::native::matmul(a_cast, b_cast);
    }
    return at::native::matmul(a, b);
}

// Example: autocast-aware softmax (always FP32)
inline at::Tensor autocast_softmax(
    const at::Tensor& input,
    int64_t dim
) {
    // Softmax requires FP32 for stability
    at::Tensor input_fp32 = (input.dtype() != c10::ScalarType::Float)
        ? input.to(c10::ScalarType::Float)
        : input;

    at::Tensor result = at::native::softmax(input_fp32, dim);

    // Cast back to input dtype if autocast is enabled
    if (is_autocast_enabled(input.device().type()) &&
        input.dtype() != c10::ScalarType::Float) {
        return result.to(input.dtype());
    }

    return result;
}

} // namespace amp
} // namespace torch
