#pragma once

#include "torch/nn/parameter.h"
#include "aten/src/ATen/ATen.h"
#include "torch/serialization.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
#include <sstream>

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

    // Serialize this state into a map of named tensors + scalar metadata.
    // Scalars are stored as 0-dim tensors.
    virtual std::unordered_map<std::string, Tensor> save() const {
        return {};
    }

    // Restore state from a map of named tensors.
    virtual void load(const std::unordered_map<std::string, Tensor>& /*m*/) {}
};

// ============================================================================
// OptimizerStateDict — serializable optimizer checkpoint
// ============================================================================
// Format:
//   param_states: map from param index (as string) to per-param state tensors
//   Scalar values (step counts, etc.) stored as 0-dim float tensors.

struct OptimizerStateDict {
    // Per-parameter states: key = param index "0", "1", ...
    // Value = map of named tensors (e.g. "exp_avg", "exp_avg_sq", "step")
    std::unordered_map<std::string, std::unordered_map<std::string, Tensor>> param_states;

    // Flatten to a single StateDict for binary serialization
    StateDict flatten() const {
        StateDict flat;
        for (const auto& [idx, state_map] : param_states) {
            for (const auto& [key, tensor] : state_map) {
                flat["state." + idx + "." + key] = tensor;
            }
        }
        return flat;
    }

    // Unflatten from a single StateDict
    static OptimizerStateDict unflatten(const StateDict& flat) {
        OptimizerStateDict osd;
        const std::string prefix = "state.";
        for (const auto& [full_key, tensor] : flat) {
            if (full_key.substr(0, prefix.size()) != prefix) continue;
            // Parse "state.<idx>.<key>"
            auto rest = full_key.substr(prefix.size());
            auto dot = rest.find('.');
            if (dot == std::string::npos) continue;
            std::string idx = rest.substr(0, dot);
            std::string key = rest.substr(dot + 1);
            osd.param_states[idx][key] = tensor;
        }
        return osd;
    }
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

    // ========================================================================
    // State Dict — save/load optimizer state for checkpoint resume
    // ========================================================================

    // Build a linear list of all parameters across all param groups (stable ordering).
    std::vector<Parameter*> all_params() const {
        std::vector<Parameter*> result;
        for (const auto& group : param_groups_) {
            for (auto* p : group.params) {
                result.push_back(p);
            }
        }
        return result;
    }

    // Save optimizer state to an OptimizerStateDict.
    // Each parameter that has state gets an entry keyed by its linear index.
    virtual OptimizerStateDict state_dict() const {
        OptimizerStateDict osd;
        auto params = all_params();
        for (size_t i = 0; i < params.size(); ++i) {
            auto it = state_.find(params[i]);
            if (it != state_.end() && it->second) {
                auto saved = it->second->save();
                if (!saved.empty()) {
                    osd.param_states[std::to_string(i)] = std::move(saved);
                }
            }
        }
        return osd;
    }

    // Restore optimizer state from an OptimizerStateDict.
    // Subclasses must override create_state_from_map() to produce the right
    // state type from the saved tensor map.
    virtual void load_state_dict(const OptimizerStateDict& osd) {
        auto params = all_params();
        for (const auto& [idx_str, state_map] : osd.param_states) {
            size_t idx = std::stoull(idx_str);
            if (idx >= params.size()) continue;
            Parameter* param = params[idx];
            // Get or create state, then load
            auto it = state_.find(param);
            if (it != state_.end() && it->second) {
                it->second->load(state_map);
            } else {
                // Create new state via virtual factory
                auto new_state = create_param_state();
                if (new_state) {
                    new_state->load(state_map);
                    state_[param] = std::move(new_state);
                }
            }
        }
    }

    // Save optimizer state to a binary file (PTOR format).
    void save_optimizer_state(const std::string& path) const {
        auto osd = state_dict();
        auto flat = osd.flatten();
        torch::save_state_dict(flat, path);
    }

    // Load optimizer state from a binary file (PTOR format).
    void load_optimizer_state(const std::string& path) {
        auto flat = torch::load_state_dict(path);
        auto osd = OptimizerStateDict::unflatten(flat);
        load_state_dict(osd);
    }

protected:
    // Factory for creating the correct OptimizerParamState subclass.
    // Subclasses override this to return their specific state type.
    virtual std::unique_ptr<OptimizerParamState> create_param_state() const {
        return nullptr;
    }

    std::vector<ParamGroup> param_groups_;
    std::unordered_map<Parameter*, std::unique_ptr<OptimizerParamState>> state_;
};

} // namespace optim
} // namespace torch
