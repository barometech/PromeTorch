#pragma once

// ============================================================================
// GradScaler for Mixed Precision Training
// ============================================================================
// Loss scaling to prevent gradient underflow in FP16/BF16 training.
//
// Usage:
//   GradScaler scaler;
//   for (auto& batch : dataloader) {
//       optimizer.zero_grad();
//
//       // Forward pass with autocast (FP16)
//       auto loss = model(input);
//
//       // Scale loss and backward
//       scaler.scale(loss).backward();
//
//       // Unscale, clip, and step
//       scaler.unscale(optimizer);
//       clip_grad_norm_(model.parameters(), 1.0);
//       scaler.step(optimizer);
//
//       // Update scale factor
//       scaler.update();
//   }

#include "aten/src/ATen/ATen.h"
#include "torch/optim/optimizer.h"
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace amp {

// ============================================================================
// GradScaler Configuration
// ============================================================================

struct GradScalerOptions {
    // Initial loss scale
    double init_scale = 65536.0;  // 2^16

    // Factor to multiply scale when increasing
    double growth_factor = 2.0;

    // Factor to multiply scale when decreasing
    double backoff_factor = 0.5;

    // Number of consecutive non-inf/nan iterations before increasing scale
    int growth_interval = 2000;

    // Enable scaler (set to false to disable scaling)
    bool enabled = true;
};

// ============================================================================
// GradScaler Class
// ============================================================================

class GradScaler {
public:
    explicit GradScaler(const GradScalerOptions& options = GradScalerOptions())
        : scale_(options.init_scale),
          growth_factor_(options.growth_factor),
          backoff_factor_(options.backoff_factor),
          growth_interval_(options.growth_interval),
          enabled_(options.enabled),
          growth_tracker_(0),
          found_inf_overall_(false) {}

    // Get current scale factor
    double get_scale() const {
        return scale_;
    }

    // Check if scaler is enabled
    bool is_enabled() const {
        return enabled_;
    }

    // Scale a loss tensor for backward pass
    // Returns: scaled_loss = loss * scale_factor
    at::Tensor scale(const at::Tensor& loss) {
        if (!enabled_) {
            return loss;
        }

        // Create scale tensor on same device as loss
        at::Tensor scale_tensor = at::full({}, scale_,
            at::TensorOptions().dtype(loss.dtype()).device(loss.device()));

        return loss * scale_tensor;
    }

    // Unscale gradients of all parameters in optimizer
    // Must be called before gradient clipping
    void unscale(optim::Optimizer& optimizer) {
        if (!enabled_) {
            return;
        }

        // Check if already unscaled for this optimizer
        auto opt_id = reinterpret_cast<uintptr_t>(&optimizer);
        if (unscaled_optimizers_.count(opt_id)) {
            return;  // Already unscaled
        }

        found_inf_overall_ = false;
        double inv_scale = 1.0 / scale_;

        for (auto& param_ptr : optimizer.param_groups()[0].params) {
            if (!param_ptr->grad().defined()) {
                continue;
            }

            at::Tensor& grad = param_ptr->grad();

            // Check for inf/nan
            if (has_inf_or_nan(grad)) {
                found_inf_overall_ = true;
            }

            // Unscale gradient: grad = grad / scale
            at::Tensor inv_scale_tensor = at::full({}, inv_scale,
                at::TensorOptions().dtype(grad.dtype()).device(grad.device()));
            grad = grad * inv_scale_tensor;
        }

        unscaled_optimizers_.insert(opt_id);
    }

    // Step optimizer if gradients are valid (no inf/nan)
    // Returns true if step was taken
    bool step(optim::Optimizer& optimizer) {
        if (!enabled_) {
            optimizer.step();
            return true;
        }

        // Ensure gradients are unscaled
        unscale(optimizer);

        if (found_inf_overall_) {
            // Skip this step due to inf/nan gradients
            return false;
        }

        optimizer.step();
        return true;
    }

    // Update scale factor based on gradient validity
    void update() {
        if (!enabled_) {
            return;
        }

        // Reset unscaled optimizers set for next iteration
        unscaled_optimizers_.clear();

        if (found_inf_overall_) {
            // Found inf/nan: decrease scale
            scale_ *= backoff_factor_;
            growth_tracker_ = 0;
        } else {
            // No inf/nan: potentially increase scale
            growth_tracker_++;
            if (growth_tracker_ >= growth_interval_) {
                scale_ *= growth_factor_;
                growth_tracker_ = 0;

                // Cap scale to prevent overflow
                if (scale_ > max_scale_) {
                    scale_ = max_scale_;
                }
            }
        }

        found_inf_overall_ = false;
    }

    // ========================================================================
    // State Dict for Checkpointing
    // ========================================================================

    struct State {
        double scale;
        int growth_tracker;

        State() : scale(65536.0), growth_tracker(0) {}
        State(double s, int g) : scale(s), growth_tracker(g) {}
    };

    State state_dict() const {
        return State(scale_, growth_tracker_);
    }

    void load_state_dict(const State& state) {
        scale_ = state.scale;
        growth_tracker_ = state.growth_tracker;
    }

    // ========================================================================
    // Advanced Configuration
    // ========================================================================

    void set_growth_factor(double factor) {
        growth_factor_ = factor;
    }

    void set_backoff_factor(double factor) {
        backoff_factor_ = factor;
    }

    void set_growth_interval(int interval) {
        growth_interval_ = interval;
    }

    // Found inf/nan in last unscale call?
    bool found_inf() const {
        return found_inf_overall_;
    }

private:
    // Check if tensor contains inf or nan values
    bool has_inf_or_nan(const at::Tensor& tensor) {
        if (!tensor.defined()) {
            return false;
        }

        // For simplicity, check by iterating (CPU only for now)
        // In production, this should be a CUDA kernel
        at::Tensor contiguous = tensor.contiguous();
        int64_t numel = contiguous.numel();

        PT_DISPATCH_FLOATING_TYPES(contiguous.dtype(), "has_inf_or_nan", [&] {
            const scalar_t* data = contiguous.data_ptr<scalar_t>();
            for (int64_t i = 0; i < numel; ++i) {
                if (std::isinf(data[i]) || std::isnan(data[i])) {
                    return true;
                }
            }
            return false;
        });

        return false;
    }

    double scale_;
    double growth_factor_;
    double backoff_factor_;
    int growth_interval_;
    bool enabled_;
    int growth_tracker_;
    bool found_inf_overall_;

    // Track which optimizers have been unscaled this step
    std::unordered_set<uintptr_t> unscaled_optimizers_;

    // Maximum scale to prevent overflow
    static constexpr double max_scale_ = 65536.0 * 65536.0;  // 2^32
};

// ============================================================================
// Helper Macros for Mixed Precision
// ============================================================================

// Dispatch for floating point types (FP16, BF16, FP32, FP64)
#define PT_DISPATCH_FLOATING_TYPES(DTYPE, NAME, ...)                          \
    [&] {                                                                      \
        switch (DTYPE) {                                                       \
            case c10::ScalarType::Half: {                                      \
                using scalar_t = c10::Half;                                    \
                return __VA_ARGS__();                                          \
            }                                                                  \
            case c10::ScalarType::BFloat16: {                                  \
                using scalar_t = c10::BFloat16;                                \
                return __VA_ARGS__();                                          \
            }                                                                  \
            case c10::ScalarType::Float: {                                     \
                using scalar_t = float;                                        \
                return __VA_ARGS__();                                          \
            }                                                                  \
            case c10::ScalarType::Double: {                                    \
                using scalar_t = double;                                       \
                return __VA_ARGS__();                                          \
            }                                                                  \
            default:                                                           \
                PT_ERROR(NAME, ": unsupported dtype");                         \
        }                                                                      \
    }()

} // namespace amp
} // namespace torch
