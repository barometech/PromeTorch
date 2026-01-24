#pragma once

#include "torch/nn/parameter.h"
#include "aten/src/ATen/ATen.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <string>

namespace torch {
namespace optim {

using at::Tensor;
using nn::Parameter;

// ============================================================================
// ParamGroup - Groups parameters with shared optimizer options
// ============================================================================
// Allows different learning rates or other options for different parameter groups

struct ParamGroup {
    std::vector<Parameter*> params;

    // Common options
    double lr = 0.01;
    double weight_decay = 0.0;

    ParamGroup() = default;

    explicit ParamGroup(std::vector<Parameter*> params_, double lr_ = 0.01)
        : params(std::move(params_)), lr(lr_) {}
};

// ============================================================================
// OptimizerState - Per-parameter state storage
// ============================================================================

struct OptimizerParamState {
    virtual ~OptimizerParamState() = default;
};

// ============================================================================
// Optimizer - Base class for all optimizers
// ============================================================================
// Optimizers update model parameters based on computed gradients.
// All optimizers inherit from this base class and implement step().
//
// Usage:
//   auto optimizer = SGD(model.parameters(), /*lr=*/0.01);
//   for (auto& batch : dataloader) {
//       optimizer.zero_grad();
//       auto loss = model.forward(batch);
//       loss.backward();
//       optimizer.step();
//   }

class Optimizer {
public:
    Optimizer() = default;

    explicit Optimizer(std::vector<Parameter*> params, double lr = 0.01)
        : param_groups_{ParamGroup(std::move(params), lr)} {}

    explicit Optimizer(std::vector<ParamGroup> param_groups)
        : param_groups_(std::move(param_groups)) {}

    virtual ~Optimizer() = default;

    // ========================================================================
    // Core Interface
    // ========================================================================

    // Perform a single optimization step
    virtual void step() = 0;

    // Zero out all parameter gradients
    virtual void zero_grad(bool set_to_none = false) {
        for (auto& group : param_groups_) {
            for (auto* param : group.params) {
                if (param->defined()) {
                    if (set_to_none) {
                        // Set gradient to undefined
                        param->set_grad(Tensor());
                    } else {
                        // Zero the gradient
                        param->zero_grad();
                    }
                }
            }
        }
    }

    // ========================================================================
    // Parameter Groups
    // ========================================================================

    void add_param_group(ParamGroup group) {
        param_groups_.push_back(std::move(group));
    }

    std::vector<ParamGroup>& param_groups() {
        return param_groups_;
    }

    const std::vector<ParamGroup>& param_groups() const {
        return param_groups_;
    }

    // ========================================================================
    // State Management
    // ========================================================================

    // Get state for a specific parameter
    template<typename StateT>
    StateT* get_state(Parameter* param) {
        auto it = state_.find(param);
        if (it == state_.end()) {
            return nullptr;
        }
        return static_cast<StateT*>(it->second.get());
    }

    // Set state for a specific parameter
    template<typename StateT>
    void set_state(Parameter* param, std::unique_ptr<StateT> state) {
        state_[param] = std::move(state);
    }

    // Get or create state for a parameter
    template<typename StateT>
    StateT* get_or_create_state(Parameter* param) {
        auto* existing = get_state<StateT>(param);
        if (existing) {
            return existing;
        }
        auto state = std::make_unique<StateT>();
        auto* ptr = state.get();
        set_state(param, std::move(state));
        return ptr;
    }

    // ========================================================================
    // Learning Rate
    // ========================================================================

    // Get the learning rate of the first parameter group
    double get_lr() const {
        if (param_groups_.empty()) return 0.0;
        return param_groups_[0].lr;
    }

    // Set learning rate for all parameter groups
    void set_lr(double lr) {
        for (auto& group : param_groups_) {
            group.lr = lr;
        }
    }

protected:
    std::vector<ParamGroup> param_groups_;
    std::unordered_map<Parameter*, std::unique_ptr<OptimizerParamState>> state_;
};

} // namespace optim
} // namespace torch
