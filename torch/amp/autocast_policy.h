#pragma once

// ============================================================================
// Autocast Policy Table (Automatic Mixed Precision)
// ============================================================================
// Centralized op categorization for autocast. Given an op name, returns the
// policy that the dispatcher should apply at op entry.
//
// Three categories follow PyTorch's design:
//
//   FP16  (LowerPrecision): Matmul-family ops that benefit from Tensor Cores
//                           and have benign numerics. Inputs cast to autocast
//                           dtype (Half or BFloat16).
//
//   FP32  (FP32Required):   Ops that need full precision for stability: losses,
//                           normalizations, softmax, log/exp, reductions. Inputs
//                           upcast to Float.
//
//   PROMOTE:                Element-wise ops where we follow the widest input
//                           dtype. `add(fp16, fp32)` -> fp32 etc.
//
//   UNCHANGED:              Everything else. Shape ops, indexing, views.
//
// Typical call pattern from an op entry point:
//
//     if (autocast::is_enabled(tensor.device().type())) {
//         auto policy = autocast::policy_for(kOpName);
//         auto processed = autocast::apply_policy(policy, {a, b});
//         // then call the kernel with processed[0], processed[1]
//     }

#include "torch/amp/autocast.h"
#include "aten/src/ATen/ATen.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace amp {

// ============================================================================
// Policy enum (mirrors AutocastCategory in autocast.h)
// ============================================================================

enum class CastPolicy {
    FP16,        // cast to autocast lower-precision dtype (Half or BFloat16)
    FP32,        // cast to Float32
    Promote,     // promote to widest input dtype
    Unchanged    // leave as-is
};

// ============================================================================
// Op -> Policy table
// ============================================================================
// Keep this in sync with PyTorch's torch/csrc/autograd/autocast_mode.cpp.
// Names are bare op names (no namespace prefix); the dispatcher should match
// by op name.

namespace detail {

inline const std::unordered_map<std::string, CastPolicy>& policy_table() {
    static const std::unordered_map<std::string, CastPolicy> table = {
        // ====== FP16: matmul-like / Tensor-Core friendly ======
        {"linear",                              CastPolicy::FP16},
        {"matmul",                              CastPolicy::FP16},
        {"mm",                                  CastPolicy::FP16},
        {"bmm",                                 CastPolicy::FP16},
        {"addmm",                               CastPolicy::FP16},
        {"addbmm",                              CastPolicy::FP16},
        {"baddbmm",                             CastPolicy::FP16},
        {"addmv",                               CastPolicy::FP16},
        {"mv",                                  CastPolicy::FP16},

        {"conv1d",                              CastPolicy::FP16},
        {"conv2d",                              CastPolicy::FP16},
        {"conv3d",                              CastPolicy::FP16},
        {"conv_transpose1d",                    CastPolicy::FP16},
        {"conv_transpose2d",                    CastPolicy::FP16},
        {"conv_transpose3d",                    CastPolicy::FP16},

        {"scaled_dot_product_attention",        CastPolicy::FP16},
        {"flash_attention",                     CastPolicy::FP16},
        {"_scaled_dot_product_efficient_attention", CastPolicy::FP16},

        {"prelu",                               CastPolicy::FP16},
        {"gru",                                 CastPolicy::FP16},
        {"gru_cell",                            CastPolicy::FP16},
        {"lstm",                                CastPolicy::FP16},
        {"lstm_cell",                           CastPolicy::FP16},
        {"rnn_relu",                            CastPolicy::FP16},
        {"rnn_relu_cell",                       CastPolicy::FP16},
        {"rnn_tanh",                            CastPolicy::FP16},
        {"rnn_tanh_cell",                       CastPolicy::FP16},

        // ====== FP32: numerical stability ======
        // Normalization & softmax — reduction stability
        {"softmax",                             CastPolicy::FP32},
        {"log_softmax",                         CastPolicy::FP32},
        {"layer_norm",                          CastPolicy::FP32},
        {"batch_norm",                          CastPolicy::FP32},
        {"group_norm",                          CastPolicy::FP32},
        {"instance_norm",                       CastPolicy::FP32},
        // NOTE: intentionally NOT putting rms_norm in FP32 because our kernels
        // already accumulate in FP32 internally; keep it FP16 eligible.

        // Losses — almost always implemented with FP32 reductions
        {"cross_entropy",                       CastPolicy::FP32},
        {"cross_entropy_loss",                  CastPolicy::FP32},
        {"nll_loss",                            CastPolicy::FP32},
        {"mse_loss",                            CastPolicy::FP32},
        {"l1_loss",                             CastPolicy::FP32},
        {"smooth_l1_loss",                      CastPolicy::FP32},
        {"huber_loss",                          CastPolicy::FP32},
        {"binary_cross_entropy",                CastPolicy::FP32},
        {"binary_cross_entropy_with_logits",    CastPolicy::FP32},
        {"kl_div",                              CastPolicy::FP32},
        {"poisson_nll_loss",                    CastPolicy::FP32},

        // Precision-sensitive unary
        {"exp",                                 CastPolicy::FP32},
        {"exp2",                                CastPolicy::FP32},
        {"expm1",                               CastPolicy::FP32},
        {"log",                                 CastPolicy::FP32},
        {"log2",                                CastPolicy::FP32},
        {"log10",                               CastPolicy::FP32},
        {"log1p",                               CastPolicy::FP32},
        {"pow",                                 CastPolicy::FP32},
        {"reciprocal",                          CastPolicy::FP32},
        {"rsqrt",                               CastPolicy::FP32},
        // GELU and tanh are sensitive enough that PyTorch upcasts them for AMP.
        {"gelu",                                CastPolicy::FP32},
        {"tanh",                                CastPolicy::FP32},

        // Reductions
        {"sum",                                 CastPolicy::FP32},
        {"mean",                                CastPolicy::FP32},
        {"prod",                                CastPolicy::FP32},
        {"cumsum",                              CastPolicy::FP32},
        {"cumprod",                             CastPolicy::FP32},
        {"var",                                 CastPolicy::FP32},
        {"std",                                 CastPolicy::FP32},
        {"norm",                                CastPolicy::FP32},
        {"dist",                                CastPolicy::FP32},

        // ====== Promote (result follows widest input) ======
        {"add",                                 CastPolicy::Promote},
        {"sub",                                 CastPolicy::Promote},
        {"mul",                                 CastPolicy::Promote},
        {"div",                                 CastPolicy::Promote},
        {"true_divide",                         CastPolicy::Promote},
        {"floor_divide",                        CastPolicy::Promote},
        {"cat",                                 CastPolicy::Promote},
        {"stack",                               CastPolicy::Promote},
        {"where",                               CastPolicy::Promote},
        {"index_put",                           CastPolicy::Promote},
        {"scatter",                             CastPolicy::Promote},
    };
    return table;
}

} // namespace detail

// ============================================================================
// Public API
// ============================================================================

// Look up policy for an op. Unknown names default to Unchanged.
inline CastPolicy policy_for(const std::string& op_name) {
    const auto& table = detail::policy_table();
    auto it = table.find(op_name);
    if (it == table.end()) return CastPolicy::Unchanged;
    return it->second;
}

// Map our CastPolicy -> AutocastCategory from autocast.h (bridge for existing
// helpers that already accept AutocastCategory).
inline AutocastCategory policy_to_category(CastPolicy p) {
    switch (p) {
        case CastPolicy::FP16:      return AutocastCategory::LowerPrecision;
        case CastPolicy::FP32:      return AutocastCategory::FP32Required;
        case CastPolicy::Promote:   return AutocastCategory::Promote;
        case CastPolicy::Unchanged: return AutocastCategory::Unchanged;
    }
    return AutocastCategory::Unchanged;
}

// Cast a single tensor according to policy (used for simple single-input ops).
// Returns the possibly-casted tensor, leaves integer / non-float tensors alone.
inline at::Tensor apply_policy(
    CastPolicy policy,
    const at::Tensor& t,
    c10::DeviceType device_type = c10::DeviceType::CUDA)
{
    if (!is_autocast_enabled(device_type)) return t;
    if (!t.defined()) return t;
    if (!c10::isFloatingType(t.dtype())) return t;

    auto autocast_dtype = get_autocast_dtype(device_type);

    switch (policy) {
        case CastPolicy::FP16:
            return (t.dtype() == autocast_dtype) ? t : t.to(autocast_dtype);
        case CastPolicy::FP32:
            return (t.dtype() == c10::ScalarType::Float) ? t : t.to(c10::ScalarType::Float);
        case CastPolicy::Promote:
        case CastPolicy::Unchanged:
            return t;
    }
    return t;
}

// Cast a list of tensors according to policy. For Promote, all tensors are
// cast to the widest dtype among the inputs.
inline std::vector<at::Tensor> apply_policy(
    CastPolicy policy,
    const std::vector<at::Tensor>& tensors,
    c10::DeviceType device_type = c10::DeviceType::CUDA)
{
    if (!is_autocast_enabled(device_type) || tensors.empty()) return tensors;

    std::vector<at::Tensor> out;
    out.reserve(tensors.size());

    if (policy == CastPolicy::Promote) {
        c10::ScalarType widest = promote_types(tensors);
        for (const auto& t : tensors) {
            if (!t.defined() || !c10::isFloatingType(t.dtype()) || t.dtype() == widest) {
                out.push_back(t);
            } else {
                out.push_back(t.to(widest));
            }
        }
        return out;
    }

    for (const auto& t : tensors) {
        out.push_back(apply_policy(policy, t, device_type));
    }
    return out;
}

// Convenience: look up by name and apply in one call.
inline std::vector<at::Tensor> autocast_inputs(
    const std::string& op_name,
    const std::vector<at::Tensor>& tensors,
    c10::DeviceType device_type = c10::DeviceType::CUDA)
{
    return apply_policy(policy_for(op_name), tensors, device_type);
}

} // namespace amp
} // namespace torch
