#pragma once

#include "torch/nn/parameter.h"
#include "aten/src/ATen/ATen.h"
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>

namespace torch {
namespace nn {
namespace utils {

using at::Tensor;

// ============================================================================
// clip_grad_norm_  (Parameter-vector overloads, mirroring PyTorch API)
// ============================================================================
// Computes the total L^p norm over all gradients and, if it exceeds max_norm,
// multiplies each gradient in-place by max_norm / (total_norm + eps).
// Returns the pre-clip total norm (as PyTorch does).
//
// `norm_type`:
//   - 2.0  (default)           : L2 norm, fast single-pass
//   - +inf                     : max absolute value
//   - any other p > 0          : general Lp

inline float clip_grad_norm_(
    const std::vector<std::shared_ptr<Parameter>>& params,
    float max_norm,
    float norm_type = 2.0f
) {
    if (params.empty()) return 0.0f;

    const float eps = 1e-6f;
    double total_norm = 0.0;

    if (std::isinf(norm_type)) {
        for (const auto& p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            const float* gd = g.data_ptr<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i) {
                total_norm = std::max(total_norm, static_cast<double>(std::abs(gd[i])));
            }
        }
    } else if (norm_type == 2.0f) {
        double total_sq = 0.0;
        for (const auto& p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            const float* gd = g.data_ptr<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i) {
                total_sq += static_cast<double>(gd[i]) * static_cast<double>(gd[i]);
            }
        }
        total_norm = std::sqrt(total_sq);
    } else {
        double acc = 0.0;
        for (const auto& p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            const float* gd = g.data_ptr<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i) {
                acc += std::pow(std::abs(static_cast<double>(gd[i])),
                                static_cast<double>(norm_type));
            }
        }
        total_norm = std::pow(acc, 1.0 / static_cast<double>(norm_type));
    }

    double clip_coef = static_cast<double>(max_norm) / (total_norm + eps);
    if (clip_coef < 1.0) {
        float scale = static_cast<float>(clip_coef);
        for (const auto& p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            float* gd = g.mutable_data_ptr<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i) {
                gd[i] *= scale;
            }
        }
    }

    return static_cast<float>(total_norm);
}

// Raw pointer overload (Parameter*). Useful when params came from
// Module::parameters() which returns std::vector<Parameter*>.
inline float clip_grad_norm_(
    const std::vector<Parameter*>& params,
    float max_norm,
    float norm_type = 2.0f
) {
    if (params.empty()) return 0.0f;

    const float eps = 1e-6f;
    double total_norm = 0.0;

    if (std::isinf(norm_type)) {
        for (auto* p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            const float* gd = g.data_ptr<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i) {
                total_norm = std::max(total_norm, static_cast<double>(std::abs(gd[i])));
            }
        }
    } else if (norm_type == 2.0f) {
        double total_sq = 0.0;
        for (auto* p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            const float* gd = g.data_ptr<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i) {
                total_sq += static_cast<double>(gd[i]) * static_cast<double>(gd[i]);
            }
        }
        total_norm = std::sqrt(total_sq);
    } else {
        double acc = 0.0;
        for (auto* p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            const float* gd = g.data_ptr<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i) {
                acc += std::pow(std::abs(static_cast<double>(gd[i])),
                                static_cast<double>(norm_type));
            }
        }
        total_norm = std::pow(acc, 1.0 / static_cast<double>(norm_type));
    }

    double clip_coef = static_cast<double>(max_norm) / (total_norm + eps);
    if (clip_coef < 1.0) {
        float scale = static_cast<float>(clip_coef);
        for (auto* p : params) {
            if (!p || !p->defined()) continue;
            Tensor g = p->grad();
            if (!g.defined()) continue;
            float* gd = g.mutable_data_ptr<float>();
            int64_t n = g.numel();
            for (int64_t i = 0; i < n; ++i) {
                gd[i] *= scale;
            }
        }
    }

    return static_cast<float>(total_norm);
}

// ============================================================================
// clip_grad_value_  — element-wise clamp of each gradient to [-clip_value, clip_value]
// ============================================================================

inline void clip_grad_value_(
    const std::vector<std::shared_ptr<Parameter>>& params,
    float clip_value
) {
    float cv = std::abs(clip_value);
    for (const auto& p : params) {
        if (!p || !p->defined()) continue;
        Tensor g = p->grad();
        if (!g.defined()) continue;
        float* gd = g.mutable_data_ptr<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; ++i) {
            gd[i] = std::max(-cv, std::min(cv, gd[i]));
        }
    }
}

inline void clip_grad_value_(
    const std::vector<Parameter*>& params,
    float clip_value
) {
    float cv = std::abs(clip_value);
    for (auto* p : params) {
        if (!p || !p->defined()) continue;
        Tensor g = p->grad();
        if (!g.defined()) continue;
        float* gd = g.mutable_data_ptr<float>();
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; ++i) {
            gd[i] = std::max(-cv, std::min(cv, gd[i]));
        }
    }
}

} // namespace utils
} // namespace nn
} // namespace torch
