#pragma once

#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "aten/src/ATen/core/Tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <any>
#include <unordered_map>
#include <functional>
#include <initializer_list>

namespace torch {
namespace autograd {

using at::Tensor;

// ============================================================================
// FunctionCtx - Context for saving data between forward and backward
// ============================================================================
// Users save tensors and arbitrary data in forward(), retrieve in backward().

struct FunctionCtx {
    // Save tensors for backward
    void save_for_backward(std::initializer_list<Tensor> tensors) {
        saved_tensors_.assign(tensors.begin(), tensors.end());
    }

    void save_for_backward(const std::vector<Tensor>& tensors) {
        saved_tensors_ = tensors;
    }

    // Retrieve saved tensors in backward
    const std::vector<Tensor>& get_saved_tensors() const {
        return saved_tensors_;
    }

    // Save arbitrary data by key
    template<typename T>
    void save_data(const std::string& key, T value) {
        saved_data_[key] = std::any(std::move(value));
    }

    // Retrieve saved data
    template<typename T>
    T get_data(const std::string& key) const {
        auto it = saved_data_.find(key);
        if (it == saved_data_.end()) {
            throw std::runtime_error("FunctionCtx: key '" + key + "' not found");
        }
        return std::any_cast<T>(it->second);
    }

    // Check if a key exists
    bool has_data(const std::string& key) const {
        return saved_data_.find(key) != saved_data_.end();
    }

    // Flags indicating which inputs need gradients
    bool needs_input_grad_[8] = {};

    // Set the number of inputs and their grad requirements
    void set_needs_input_grad(const variable_list& inputs) {
        for (size_t i = 0; i < inputs.size() && i < 8; ++i) {
            needs_input_grad_[i] = inputs[i].defined() && inputs[i].requires_grad();
        }
    }

    // Release all saved state
    void release() {
        saved_tensors_.clear();
        saved_data_.clear();
    }

private:
    std::vector<Tensor> saved_tensors_;
    std::unordered_map<std::string, std::any> saved_data_;
};

// ============================================================================
// CustomFunctionNode - Node wrapping user's backward as std::function
// ============================================================================

struct CustomFunctionNode : public Node {
    using BackwardFn = std::function<variable_list(FunctionCtx&, variable_list&&)>;

    FunctionCtx ctx_;
    BackwardFn backward_fn_;
    std::string name_;

    CustomFunctionNode(BackwardFn backward_fn, const std::string& name = "CustomFunction")
        : backward_fn_(std::move(backward_fn)), name_(name) {}

    variable_list apply(variable_list&& grads) override {
        auto result = backward_fn_(ctx_, std::move(grads));
        // Release saved tensors after backward
        ctx_.release();
        return result;
    }

    void release_saved_tensors() override {
        ctx_.release();
    }

    std::string name() const override { return name_ + "Backward"; }
};

// ============================================================================
// Function<Derived> - CRTP base class for custom autograd functions
// ============================================================================
// Users implement:
//   static variable_list forward(FunctionCtx& ctx, variable_list&& inputs)
//   static variable_list backward(FunctionCtx& ctx, variable_list&& grad_outputs)
//
// Usage:
//   struct MyReLU : Function<MyReLU> {
//       static variable_list forward(FunctionCtx& ctx, variable_list&& inputs) { ... }
//       static variable_list backward(FunctionCtx& ctx, variable_list&& grad_outputs) { ... }
//   };
//   auto result = MyReLU::apply(std::move(inputs));

template<typename Derived>
struct Function {
    static variable_list apply(variable_list&& inputs) {
        // Check if any input requires grad and grad mode is enabled
        bool requires_grad = false;
        if (GradMode::is_enabled()) {
            for (const auto& t : inputs) {
                if (t.defined() && t.requires_grad()) {
                    requires_grad = true;
                    break;
                }
            }
        }

        if (!requires_grad) {
            // No grad needed — just run forward without graph
            FunctionCtx ctx;
            ctx.set_needs_input_grad(inputs);
            return Derived::forward(ctx, std::move(inputs));
        }

        // Create backward node
        auto grad_fn = std::make_shared<CustomFunctionNode>(
            [](FunctionCtx& ctx, variable_list&& grads) -> variable_list {
                return Derived::backward(ctx, std::move(grads));
            },
            typeid(Derived).name()
        );

        // Set up context
        grad_fn->ctx_.set_needs_input_grad(inputs);

        // Record input edges for backward graph
        for (const auto& input : inputs) {
            grad_fn->add_input_metadata(input);
        }

        // Run forward (under current grad mode — user can use NoGradGuard if needed)
        variable_list outputs = Derived::forward(grad_fn->ctx_, std::move(inputs));

        // Wire outputs to the backward node
        for (auto& output : outputs) {
            if (output.defined()) {
                set_grad_fn(output, grad_fn);
                output.set_requires_grad(true);
            }
        }

        return outputs;
    }
};

} // namespace autograd
} // namespace torch
