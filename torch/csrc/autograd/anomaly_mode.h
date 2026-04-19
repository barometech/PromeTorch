#pragma once

// ============================================================================
// AnomalyMode - NaN/Inf detection during backward
// ============================================================================
// When enabled, the Engine validates gradients before and after each
// Node::apply(). Any NaN/Inf triggers a runtime_error carrying:
//   - The offending Node's name()
//   - The node's captured forward-creation stack (if any)
//   - Which input/output was bad
//
// Usage:
//   {
//     torch::autograd::AnomalyGuard g;     // scope-local enable
//     loss.backward();                     // throws on NaN/Inf
//   }
//
// Or globally:
//   torch::autograd::AnomalyMode::set_enabled(true);

#include "aten/src/ATen/core/Tensor.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch {
namespace autograd {

class AnomalyMode {
public:
    static bool is_enabled() noexcept {
        return enabled_.load(std::memory_order_relaxed);
    }
    static void set_enabled(bool v) noexcept {
        enabled_.store(v, std::memory_order_relaxed);
    }

private:
    inline static std::atomic<bool> enabled_{false};
};

// Scope-local RAII guard: pushes enabled=true on construction, restores on dtor.
// Nested guards are supported (each guard saves and restores the prior state).
class AnomalyGuard {
public:
    AnomalyGuard() : prev_(AnomalyMode::is_enabled()) {
        AnomalyMode::set_enabled(true);
    }
    ~AnomalyGuard() {
        AnomalyMode::set_enabled(prev_);
    }
    AnomalyGuard(const AnomalyGuard&) = delete;
    AnomalyGuard& operator=(const AnomalyGuard&) = delete;

private:
    bool prev_;
};

namespace debug {

// Simple stack frame: a textual location string.
// Populated when a Node is constructed under AnomalyMode (engine sets this).
using StackFrame = std::string;
using Stack = std::vector<StackFrame>;

// Capture a lightweight "stack" — on Elbrus LCC we don't have portable
// std::stacktrace yet; we fall back to a single frame string that callers
// may pass at Node construction (e.g. __FILE__ ":" __LINE__).
inline Stack capture_stack(const std::string& hint = {}) {
    Stack s;
    if (!hint.empty()) s.push_back(hint);
    return s;
}

inline std::string format_stack(const Stack& s) {
    if (s.empty()) return "(no stack captured)";
    std::ostringstream oss;
    for (size_t i = 0; i < s.size(); ++i) {
        oss << "  #" << i << " " << s[i] << "\n";
    }
    return oss.str();
}

} // namespace debug

// ============================================================================
// NaN / Inf validation
// ============================================================================
// Scans a float32 contiguous tensor for NaN/Inf. Skips undefined / non-float
// tensors silently. Returns true if bad value found (and sets offset).

inline bool tensor_has_nan_or_inf(const at::Tensor& t, int64_t* bad_offset = nullptr) {
    if (!t.defined()) return false;
    if (t.dtype() != c10::ScalarType::Float) return false;
    if (!t.is_contiguous()) return false;  // conservative: skip weird layouts
    const float* p = t.data_ptr<float>();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        float v = p[i];
        if (std::isnan(v) || std::isinf(v)) {
            if (bad_offset) *bad_offset = i;
            return true;
        }
    }
    return false;
}

// Throws std::runtime_error with node context when a bad value is detected.
// `phase` is "input" or "output" for the error message.
inline void validate_no_nan_or_inf(
    const std::vector<at::Tensor>& tensors,
    const std::string& node_name,
    const debug::Stack& forward_stack,
    const char* phase)
{
    for (size_t i = 0; i < tensors.size(); ++i) {
        int64_t off = -1;
        if (tensor_has_nan_or_inf(tensors[i], &off)) {
            std::ostringstream oss;
            oss << "[AnomalyMode] NaN/Inf detected in " << phase
                << " tensor #" << i << " of node '" << node_name
                << "' (element offset " << off << ")\n"
                << "Forward creation stack:\n"
                << debug::format_stack(forward_stack);
            throw std::runtime_error(oss.str());
        }
    }
}

} // namespace autograd
} // namespace torch
