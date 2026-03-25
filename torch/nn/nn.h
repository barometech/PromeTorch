#pragma once

// ============================================================================
// PromeTorch Neural Network Module
// ============================================================================
// This is the main header file for the torch::nn namespace.
// It provides all the building blocks needed to construct neural networks.
//
// Usage:
//   #include "torch/nn/nn.h"
//
//   using namespace torch::nn;
//
//   // Create a simple network
//   auto model = std::make_shared<Sequential>();
//   model->add(std::make_shared<Linear>(784, 256));
//   model->add(std::make_shared<ReLU>());
//   model->add(std::make_shared<Linear>(256, 10));
//
//   // Forward pass
//   Tensor output = model->forward(input);
// ============================================================================

// Core module components
#include "parameter.h"
#include "module.h"
#include "init.h"

// Container modules
#include "modules/container.h"

// Layer modules
#include "modules/linear.h"
#include "modules/conv.h"
#include "modules/pooling.h"
#include "modules/normalization.h"
#include "modules/dropout.h"
#include "modules/activation.h"
#include "modules/sparse.h"

// Attention and Transformer modules
#include "modules/attention.h"
#include "modules/transformer.h"

// Recurrent modules
#include "modules/rnn.h"

// Utilities
#include "utils/rnn.h"

// Loss functions
#include "modules/loss.h"

namespace torch {
namespace nn {

// ============================================================================
// Convenient type aliases
// ============================================================================

using ModulePtr = std::shared_ptr<Module>;

// ============================================================================
// Module printing utilities
// ============================================================================

inline std::string module_repr(const Module& module, int indent = 0) {
    std::string indent_str(indent * 2, ' ');
    std::string result = indent_str + "(" + module.name() + ")";

    // Check for parameters
    auto params = const_cast<Module&>(module).named_parameters("", false);
    if (!params.empty()) {
        result += " [";
        bool first = true;
        for (const auto& [name, param] : params) {
            if (!first) result += ", ";
            first = false;

            std::vector<int64_t> sizes = param->data().sizes().vec();
            result += name + ": ";
            if (sizes.empty()) {
                result += "scalar";
            } else {
                result += "(";
                for (size_t i = 0; i < sizes.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += std::to_string(sizes[i]);
                }
                result += ")";
            }
        }
        result += "]";
    }

    return result;
}

// ============================================================================
// Count parameters utility
// ============================================================================

inline int64_t count_parameters(Module& module, bool only_trainable = true) {
    int64_t count = 0;
    for (auto* param : module.parameters()) {
        if (!only_trainable || param->requires_grad()) {
            count += param->data().numel();
        }
    }
    return count;
}

// ============================================================================
// Freeze/Unfreeze utilities
// ============================================================================

inline void freeze(Module& module) {
    for (auto* param : module.parameters()) {
        param->set_requires_grad(false);
    }
}

inline void unfreeze(Module& module) {
    for (auto* param : module.parameters()) {
        param->set_requires_grad(true);
    }
}

// ============================================================================
// Gradient clipping utilities
// ============================================================================

// Fast L2-only clip_grad_norm: single/two-pass over raw float*, no intermediates
// Returns the total L2 gradient norm BEFORE clipping
inline float fast_clip_grad_norm_(std::vector<Parameter*>& params, float max_norm) {
    float total_sq = 0.0f;
    for (auto* p : params) {
        if (!p->grad().defined()) continue;
        const float* g = p->grad().data_ptr<float>();
        int64_t n = p->grad().numel();
        // Unrolled accumulation for ILP
        float sq0 = 0, sq1 = 0, sq2 = 0, sq3 = 0;
        int64_t i = 0;
        for (; i + 3 < n; i += 4) {
            sq0 += g[i]   * g[i];
            sq1 += g[i+1] * g[i+1];
            sq2 += g[i+2] * g[i+2];
            sq3 += g[i+3] * g[i+3];
        }
        for (; i < n; i++) sq0 += g[i] * g[i];
        total_sq += sq0 + sq1 + sq2 + sq3;
    }
    float norm = std::sqrt(total_sq);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        for (auto* p : params) {
            if (!p->grad().defined()) continue;
            float* g = p->grad().mutable_data_ptr<float>();
            int64_t n = p->grad().numel();
            for (int64_t i = 0; i < n; i++) g[i] *= scale;
        }
    }
    return norm;
}

// Module-based convenience wrapper
inline float fast_clip_grad_norm_(Module& module, float max_norm) {
    auto params = module.parameters();
    return fast_clip_grad_norm_(params, max_norm);
}

inline double clip_grad_norm_(Module& module, double max_norm, double norm_type = 2.0) {
    std::vector<Parameter*> params = module.parameters();
    if (params.empty()) return 0.0;

    double total_norm = 0.0;

    if (norm_type == std::numeric_limits<double>::infinity()) {
        // Max norm
        for (auto* param : params) {
            if (param->grad().defined()) {
                // Move grad to CPU for computing norm
                Tensor grad_cpu = param->grad();
#ifdef PT_USE_CUDA
                if (grad_cpu.is_cuda()) {
                    grad_cpu = at::to_cpu(grad_cpu);
                }
#endif
                const float* grad_data = grad_cpu.data_ptr<float>();
                int64_t numel = grad_cpu.numel();
                for (int64_t i = 0; i < numel; ++i) {
                    total_norm = std::max(total_norm, static_cast<double>(std::abs(grad_data[i])));
                }
            }
        }
    } else if (norm_type == 2.0) {
        // L2 norm fast path: x*x instead of pow(abs(x), 2.0)
        for (auto* param : params) {
            if (param->grad().defined()) {
                Tensor grad_cpu = param->grad();
#ifdef PT_USE_CUDA
                if (grad_cpu.is_cuda()) {
                    grad_cpu = at::to_cpu(grad_cpu);
                }
#endif
                const float* grad_data = grad_cpu.data_ptr<float>();
                int64_t numel = grad_cpu.numel();
                for (int64_t i = 0; i < numel; ++i) {
                    float g = grad_data[i];
                    total_norm += static_cast<double>(g * g);
                }
            }
        }
        total_norm = std::sqrt(total_norm);
    } else {
        // General Lp norm
        for (auto* param : params) {
            if (param->grad().defined()) {
                // Move grad to CPU for computing norm
                Tensor grad_cpu = param->grad();
#ifdef PT_USE_CUDA
                if (grad_cpu.is_cuda()) {
                    grad_cpu = at::to_cpu(grad_cpu);
                }
#endif
                const float* grad_data = grad_cpu.data_ptr<float>();
                int64_t numel = grad_cpu.numel();
                for (int64_t i = 0; i < numel; ++i) {
                    total_norm += std::pow(std::abs(grad_data[i]), norm_type);
                }
            }
        }
        total_norm = std::pow(total_norm, 1.0 / norm_type);
    }

    double clip_coef = max_norm / (total_norm + 1e-6);
    if (clip_coef < 1.0) {
        for (auto* param : params) {
            if (param->grad().defined()) {
                // mul_ with Scalar has CUDA dispatch
                param->grad().mul_(at::Scalar(clip_coef));
            }
        }
    }

    return total_norm;
}

inline void clip_grad_value_(Module& module, double clip_value) {
    for (auto* param : module.parameters()) {
        if (param->grad().defined()) {
            float* grad_data = param->grad().mutable_data_ptr<float>();
            int64_t numel = param->grad().numel();
            float clip_val = static_cast<float>(clip_value);

            for (int64_t i = 0; i < numel; ++i) {
                grad_data[i] = std::max(-clip_val, std::min(clip_val, grad_data[i]));
            }
        }
    }
}

// ============================================================================
// Model compression — replace Linear layers with LowRankLinear via SVD
// ============================================================================
// rank_ratio: fraction of min(in, out) to keep as rank (e.g., 0.5 = 50%)
// min_size: only compress layers with in*out > min_size (skip small layers)
//
// Usage:
//   auto stats = compress_model(model, 0.5);
//   // model now uses LowRankLinear for large layers
//
// Returns: number of layers compressed

struct CompressionStats {
    int layers_compressed = 0;
    int layers_skipped = 0;
    int64_t params_before = 0;
    int64_t params_after = 0;
};

inline CompressionStats compress_model(Module& model, double rank_ratio = 0.5,
                                        int64_t min_size = 1000) {
    CompressionStats stats;

    // Iterate over direct submodules
    // Collect replacements first, then apply (can't modify during iteration)
    auto named = model.named_children();
    std::vector<std::pair<std::string, std::shared_ptr<LowRankLinear>>> replacements;

    for (auto& [name, submodule] : named) {
        if (!submodule) continue;

        // Check if it's a Linear layer
        auto linear = std::dynamic_pointer_cast<Linear>(submodule);
        if (!linear) {
            // Recursively compress submodules that are not Linear
            auto sub_stats = compress_model(*submodule, rank_ratio, min_size);
            stats.layers_compressed += sub_stats.layers_compressed;
            stats.layers_skipped += sub_stats.layers_skipped;
            stats.params_before += sub_stats.params_before;
            stats.params_after += sub_stats.params_after;
            continue;
        }

        int64_t in_f = linear->in_features();
        int64_t out_f = linear->out_features();
        int64_t full_size = in_f * out_f;

        // Skip small layers (not worth compressing)
        if (full_size <= min_size) {
            stats.layers_skipped++;
            stats.params_before += full_size;
            stats.params_after += full_size;
            continue;
        }

        // Compute rank
        int64_t min_dim = std::min(in_f, out_f);
        int64_t rank = std::max(int64_t(1), static_cast<int64_t>(min_dim * rank_ratio));

        // Compress
        auto lr = LowRankLinear::from_linear(linear, rank);

        stats.layers_compressed++;
        stats.params_before += full_size;
        stats.params_after += lr->compressed_params();

        replacements.push_back({name, lr});
    }

    // Apply replacements
    for (auto& [name, lr] : replacements) {
        model.replace_module(name, lr);
    }

    return stats;
}

} // namespace nn
} // namespace torch
