#pragma once

// ============================================================================
// Gradient Hooks — tensor-level, called before gradient accumulation.
// ============================================================================
// Usage:
//   auto h = torch::autograd::register_hook(param, [](const at::Tensor& g) {
//       return g.mul(at::scalar_tensor(2.0f));   // double the gradient
//   });
//   ... loss.backward() ...
//   torch::autograd::remove_hook(h);             // optional
//
// Semantics (PyTorch-compatible):
//   * Registered on a leaf tensor that requires_grad.
//   * Called during backward() with the incoming gradient BEFORE it is
//     accumulated into tensor.grad_.
//   * Return value replaces the gradient (an UNDEFINED Tensor means "no
//     change", i.e. keep original gradient).
//   * Multiple hooks fire in registration order, each seeing the output of
//     the previous hook.
//
// These hooks live on AutogradMetaImpl.grad_hooks_ (added below).

#include "aten/src/ATen/core/Tensor.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include <functional>
#include <memory>
#include <stdexcept>

namespace torch {
namespace autograd {

using GradHookFn = std::function<at::Tensor(const at::Tensor&)>;

// Shared handle — users hold this to remove the hook.
// Internally we keep a shared_ptr to the function object; removal sets the
// shared pointer's target to null, so stored copies on the tensor also
// short-circuit.
using GradHookHandle = std::shared_ptr<GradHookFn>;

// Register a hook that runs before the gradient is accumulated into `tensor`.
// Throws if `tensor` does not require grad. Upgrades metadata as needed.
inline GradHookHandle register_hook(at::Tensor& tensor, GradHookFn fn) {
    if (!tensor.defined()) {
        throw std::runtime_error("register_hook: tensor is undefined");
    }
    if (!tensor.requires_grad()) {
        throw std::runtime_error(
            "register_hook: tensor does not require grad — cannot hook");
    }
    auto* meta = ensure_autograd_meta_impl(tensor);
    auto handle = std::make_shared<GradHookFn>(std::move(fn));
    meta->grad_hooks_.push_back(handle);
    return handle;
}

// Remove a previously registered hook. Safe to call more than once.
// Implemented by clearing the function target; the engine checks for null.
inline void remove_hook(GradHookHandle h) {
    if (h) {
        *h = GradHookFn();  // target() now returns nullptr; hook is a no-op
    }
}

// Apply all registered hooks on `tensor` to `grad`, returning the
// (possibly modified) gradient. Called from the engine / AccumulateGrad.
// If there are no hooks, returns `grad` unchanged (zero-copy).
inline at::Tensor run_grad_hooks(
    const AutogradMetaImpl* meta,
    const at::Tensor& grad)
{
    if (!meta || meta->grad_hooks_.empty() || !grad.defined()) return grad;
    at::Tensor cur = grad;
    for (auto& h : meta->grad_hooks_) {
        if (!h) continue;
        auto& fn = *h;
        if (!fn) continue;                // removed hook
        at::Tensor next = fn(cur);
        if (next.defined()) cur = next;   // undefined => keep previous grad
    }
    return cur;
}

} // namespace autograd
} // namespace torch
