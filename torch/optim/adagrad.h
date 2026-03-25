#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// AdagradOptions
// ============================================================================

struct AdagradOptions {
    double lr = 0.01;
    double lr_decay = 0.0;
    double weight_decay = 0.0;
    double eps = 1e-10;
    double initial_accumulator_value = 0.0;

    AdagradOptions(double lr_ = 0.01) : lr(lr_) {}

    AdagradOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    AdagradOptions& lr_decay_(double d) { lr_decay = d; return *this; }
    AdagradOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    AdagradOptions& eps_(double e) { eps = e; return *this; }
    AdagradOptions& initial_accumulator_value_(double v) { initial_accumulator_value = v; return *this; }
};

// ============================================================================
// AdagradParamState
// ============================================================================

struct AdagradParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor sum;  // Accumulated squared gradients

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        Tensor step_t = at::zeros({});
        step_t.mutable_data_ptr<float>()[0] = static_cast<float>(step);
        m["step"] = step_t;
        if (sum.defined()) m["sum"] = sum;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("step");
        if (it != m.end()) step = static_cast<int64_t>(it->second.data_ptr<float>()[0]);
        it = m.find("sum");
        if (it != m.end()) sum = it->second.clone();
    }
};

// ============================================================================
// Adagrad - Adaptive Gradient Algorithm
// ============================================================================
// Duchi, Hazan, Singer (2011). Adapts learning rate per parameter based on
// historical gradient magnitudes. Good for sparse gradients.

class Adagrad : public Optimizer {
public:
    Adagrad(std::vector<Parameter*> params, AdagradOptions options = AdagradOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    Adagrad(std::vector<Parameter*> params, double lr)
        : Adagrad(std::move(params), AdagradOptions(lr)) {}

    Adagrad(std::vector<ParamGroup> param_groups, AdagradOptions options = AdagradOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<AdagradParamState>(param);
                state->step++;

                // Initialize accumulator
                if (!state->sum.defined()) {
                    state->sum = at::full(param->sizes(), at::Scalar(options_.initial_accumulator_value));
                }

                // Weight decay (L2 regularization added to gradient)
                if (wd != 0.0) {
                    grad = grad.add(param->data(), at::Scalar(wd));
                }

                // Learning rate with decay: clr = lr / (1 + (step-1) * lr_decay)
                double clr = lr / (1.0 + (state->step - 1) * options_.lr_decay);

                // sum += grad^2
                state->sum.addcmul_(grad, grad, at::Scalar(1.0));

                // param -= clr * grad / (sqrt(sum) + eps)
                Tensor denom = state->sum.sqrt().add(at::Scalar(options_.eps));
                param->data().addcdiv_(grad, denom, at::Scalar(-clr));
            }
        }
    }

    AdagradOptions& options() { return options_; }
    const AdagradOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<AdagradParamState>();
    }

private:
    AdagradOptions options_;
};

} // namespace optim
} // namespace torch
