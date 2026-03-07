#pragma once

#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/engine.h"
#include "aten/src/ATen/core/Tensor.h"
#include <functional>
#include <memory>

namespace torch {
namespace utils {

using at::Tensor;
using autograd::Node;
using autograd::variable_list;

// ============================================================================
// CheckpointBackward - Recomputes forward during backward
// ============================================================================
// Instead of saving intermediate activations, saves the function and input.
// During backward, re-runs forward to reconstruct the graph, then backprops.

struct CheckpointBackward : public Node {
    using ForwardFn = std::function<Tensor(const Tensor&)>;

    ForwardFn fn_;
    Tensor input_;  // detached copy of input

    CheckpointBackward(ForwardFn fn, const Tensor& input)
        : fn_(std::move(fn)), input_(input.detach().clone()) {}

    void release_saved_tensors() override {
        input_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];
        if (!grad_output.defined()) return {Tensor()};

        // Re-run forward with grad enabled to build the graph
        Tensor input_with_grad = input_.detach();
        input_with_grad.set_requires_grad(true);

        Tensor output;
        {
            autograd::EnableGradGuard enable;
            output = fn_(input_with_grad);
        }

        // Backprop through the recomputed graph
        autograd::backward({output}, {grad_output}, /*retain_graph=*/false);

        // Get gradient of input
        Tensor grad_input = input_with_grad.grad();

        // Cleanup
        input_ = Tensor();

        return {grad_input.defined() ? grad_input : Tensor()};
    }

    std::string name() const override { return "CheckpointBackward"; }
};

// ============================================================================
// checkpoint - Memory-efficient forward that recomputes during backward
// ============================================================================
// Usage:
//   auto output = torch::utils::checkpoint(
//       [&](const Tensor& x) { return expensive_fn(x); },
//       input
//   );

inline Tensor checkpoint(
    std::function<Tensor(const Tensor&)> fn,
    const Tensor& input
) {
    // Run forward WITHOUT recording the graph (saves memory)
    Tensor output;
    {
        autograd::NoGradGuard no_grad;
        output = fn(input);
    }

    // If input doesn't need grad, just return
    if (!input.requires_grad() || !autograd::GradMode::is_enabled()) {
        return output;
    }

    // Create a backward node that will recompute forward during backward
    auto grad_fn = std::make_shared<CheckpointBackward>(fn, input);
    grad_fn->add_input_metadata(input);

    autograd::set_grad_fn(output, grad_fn);
    output.set_requires_grad(true);

    return output;
}

// Multi-input version
struct CheckpointMultiBackward : public Node {
    using ForwardFn = std::function<Tensor(const std::vector<Tensor>&)>;

    ForwardFn fn_;
    std::vector<Tensor> inputs_;

    CheckpointMultiBackward(ForwardFn fn, const std::vector<Tensor>& inputs)
        : fn_(std::move(fn)) {
        inputs_.reserve(inputs.size());
        for (const auto& inp : inputs) {
            inputs_.push_back(inp.detach().clone());
        }
    }

    void release_saved_tensors() override {
        inputs_.clear();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_output = grads[0];
        if (!grad_output.defined()) {
            return variable_list(inputs_.size(), Tensor());
        }

        // Prepare inputs with grad
        std::vector<Tensor> inputs_with_grad;
        inputs_with_grad.reserve(inputs_.size());
        for (auto& inp : inputs_) {
            Tensor t = inp.detach();
            t.set_requires_grad(true);
            inputs_with_grad.push_back(t);
        }

        // Recompute forward
        Tensor output;
        {
            autograd::EnableGradGuard enable;
            output = fn_(inputs_with_grad);
        }

        // Backprop
        autograd::backward({output}, {grad_output}, /*retain_graph=*/false);

        // Collect gradients
        variable_list grad_inputs;
        for (auto& t : inputs_with_grad) {
            grad_inputs.push_back(t.grad());
        }

        inputs_.clear();
        return grad_inputs;
    }

    std::string name() const override { return "CheckpointMultiBackward"; }
};

inline Tensor checkpoint(
    std::function<Tensor(const std::vector<Tensor>&)> fn,
    const std::vector<Tensor>& inputs
) {
    Tensor output;
    {
        autograd::NoGradGuard no_grad;
        output = fn(inputs);
    }

    bool needs_grad = false;
    if (autograd::GradMode::is_enabled()) {
        for (const auto& inp : inputs) {
            if (inp.requires_grad()) { needs_grad = true; break; }
        }
    }

    if (!needs_grad) return output;

    auto grad_fn = std::make_shared<CheckpointMultiBackward>(fn, inputs);
    for (const auto& inp : inputs) {
        grad_fn->add_input_metadata(inp);
    }

    autograd::set_grad_fn(output, grad_fn);
    output.set_requires_grad(true);

    return output;
}

} // namespace utils
} // namespace torch
