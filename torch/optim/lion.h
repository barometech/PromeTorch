#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// LionOptions
// ============================================================================

struct LionOptions {
    double lr = 1e-4;            // Lion typically uses lr ~10x smaller than AdamW
    double beta1 = 0.9;          // Coefficient for interpolation in update direction
    double beta2 = 0.99;         // Coefficient for momentum EMA
    double weight_decay = 0.0;   // Decoupled weight decay

    LionOptions(double lr_ = 1e-4) : lr(lr_) {}

    LionOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    LionOptions& betas(double b1, double b2) { beta1 = b1; beta2 = b2; return *this; }
    LionOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
};

// ============================================================================
// LionParamState
// ============================================================================

struct LionParamState : public OptimizerParamState {
    Tensor exp_avg;  // Momentum EMA

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        if (exp_avg.defined()) m["exp_avg"] = exp_avg;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("exp_avg");
        if (it != m.end()) exp_avg = it->second.clone();
    }
};

// ============================================================================
// Lion - EvoLved Sign Momentum (Chen et al 2023)
// ============================================================================
// update = sign(beta1 * m + (1 - beta1) * g)
// m      = beta2 * m + (1 - beta2) * g
// p      = p - lr * (update + wd * p)
// Uses sign() so update magnitude is uniform — typical lr ~3e-5..1e-4.

class Lion : public Optimizer {
public:
    Lion(std::vector<Parameter*> params, LionOptions options = LionOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    Lion(std::vector<Parameter*> params, double lr)
        : Lion(std::move(params), LionOptions(lr)) {}

    Lion(std::vector<ParamGroup> param_groups, LionOptions options = LionOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        double beta1 = options_.beta1;
        double beta2 = options_.beta2;

        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<LionParamState>(param);
                if (!state->exp_avg.defined()) {
                    state->exp_avg = at::zeros(param->sizes());
                }

                // Decoupled weight decay
                if (wd != 0.0) {
                    param->data().mul_(at::Scalar(1.0 - lr * wd));
                }

                // update = sign(beta1 * m + (1 - beta1) * g)
                Tensor update = state->exp_avg.mul(at::Scalar(beta1))
                    .add(grad, at::Scalar(1.0 - beta1))
                    .sign();

                // p -= lr * update
                param->data().sub_(update, at::Scalar(lr));

                // m = beta2 * m + (1 - beta2) * g
                state->exp_avg.mul_(at::Scalar(beta2));
                state->exp_avg.add_(grad, at::Scalar(1.0 - beta2));
            }
        }
    }

    LionOptions& options() { return options_; }
    const LionOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<LionParamState>();
    }

private:
    LionOptions options_;
};

} // namespace optim
} // namespace torch
