#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// ASGDOptions - Options for Averaged SGD optimizer
// ============================================================================

struct ASGDOptions {
    double lr = 0.01;            // Learning rate
    double lambd = 1e-4;         // Decay term
    double alpha = 0.75;         // Power for eta update
    double t0 = 1e6;             // Point at which to start averaging
    double weight_decay = 0.0;   // Weight decay (L2 penalty)

    ASGDOptions(double lr_ = 0.01) : lr(lr_) {}

    ASGDOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    ASGDOptions& lambd_(double l) { lambd = l; return *this; }
    ASGDOptions& alpha_(double a) { alpha = a; return *this; }
    ASGDOptions& t0_(double t) { t0 = t; return *this; }
    ASGDOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
};

// ============================================================================
// ASGDParamState - Per-parameter state for ASGD
// ============================================================================

struct ASGDParamState : public OptimizerParamState {
    int64_t step = 0;
    double eta = 0.0;   // Current effective learning rate
    double mu = 1.0;    // Averaging coefficient
    Tensor ax;          // Averaged parameters

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        Tensor step_t = at::zeros({});
        step_t.mutable_data_ptr<float>()[0] = static_cast<float>(step);
        m["step"] = step_t;
        Tensor eta_t = at::zeros({});
        eta_t.mutable_data_ptr<float>()[0] = static_cast<float>(eta);
        m["eta"] = eta_t;
        Tensor mu_t = at::zeros({});
        mu_t.mutable_data_ptr<float>()[0] = static_cast<float>(mu);
        m["mu"] = mu_t;
        if (ax.defined()) m["ax"] = ax;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("step");
        if (it != m.end()) step = static_cast<int64_t>(it->second.data_ptr<float>()[0]);
        it = m.find("eta");
        if (it != m.end()) eta = static_cast<double>(it->second.data_ptr<float>()[0]);
        it = m.find("mu");
        if (it != m.end()) mu = static_cast<double>(it->second.data_ptr<float>()[0]);
        it = m.find("ax");
        if (it != m.end()) ax = it->second.clone();
    }
};

// ============================================================================
// ASGD - Averaged Stochastic Gradient Descent (Polyak averaging)
// ============================================================================
// Implements ASGD from "Acceleration of Stochastic Approximation by Averaging"
// by B. T. Polyak and A. B. Juditsky (1992).
//
// Algorithm (per step):
//   p = p * (1 - lambd * eta) - eta * g            // SGD step with weight decay & lambd
//   if mu != 1:
//       ax += (p - ax) * mu
//   else:
//       ax = p.clone()
//   eta = lr / (1 + lambd * lr * step)^alpha
//   mu = 1 / max(1, step - t0)
//
// The averaged parameters `ax` are returned at the end of training as a
// lower-variance estimator of the optimum (Polyak averaging).

class ASGD : public Optimizer {
public:
    ASGD(std::vector<Parameter*> params, ASGDOptions options = ASGDOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    ASGD(std::vector<Parameter*> params, double lr)
        : ASGD(std::move(params), ASGDOptions(lr)) {}

    ASGD(std::vector<ParamGroup> param_groups, ASGDOptions options = ASGDOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        double lambd = options_.lambd;
        double alpha = options_.alpha;
        double t0 = options_.t0;
        double base_lr = options_.lr;

        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : base_lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<ASGDParamState>(param);
                state->step++;

                if (!state->ax.defined()) {
                    state->ax = at::zeros(param->sizes());
                    state->ax.copy_(param->data());
                    state->eta = lr;
                    state->mu = 1.0;
                }

                // Apply L2 weight decay to gradient
                if (wd != 0.0) {
                    grad = grad.add(param->data(), at::Scalar(wd));
                }

                // Decay term: p *= (1 - lambd * eta)
                double decay = 1.0 - lambd * state->eta;
                if (decay != 1.0) {
                    param->data().mul_(at::Scalar(decay));
                }

                // SGD step: p -= eta * g
                param->data().sub_(grad, at::Scalar(state->eta));

                // Averaging: ax = ax + mu * (p - ax)
                if (state->mu != 1.0) {
                    Tensor diff = param->data().sub(state->ax);
                    state->ax.add_(diff, at::Scalar(state->mu));
                } else {
                    state->ax.copy_(param->data());
                }

                // Update eta and mu for next step
                double denom = std::pow(1.0 + lambd * lr * state->step, alpha);
                state->eta = lr / denom;
                double m = static_cast<double>(state->step) - t0;
                if (m < 1.0) m = 1.0;
                state->mu = 1.0 / m;
            }
        }
    }

    // Copy averaged parameters (ax) back into the actual parameters.
    // Call this after training is complete to use the Polyak-averaged solution.
    void swap_in_averaged_params() {
        for (auto& group : param_groups_) {
            for (auto* param : group.params) {
                if (!param->defined()) continue;
                auto* state = get_state<ASGDParamState>(param);
                if (state && state->ax.defined()) {
                    param->data().copy_(state->ax);
                }
            }
        }
    }

    ASGDOptions& options() { return options_; }
    const ASGDOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<ASGDParamState>();
    }

private:
    ASGDOptions options_;
};

} // namespace optim
} // namespace torch
