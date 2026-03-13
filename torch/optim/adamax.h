#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// AdamaxOptions
// ============================================================================

struct AdamaxOptions {
    double lr = 0.002;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;

    AdamaxOptions(double lr_ = 0.002) : lr(lr_) {}

    AdamaxOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    AdamaxOptions& betas(double b1, double b2) { beta1 = b1; beta2 = b2; return *this; }
    AdamaxOptions& eps_(double e) { eps = e; return *this; }
    AdamaxOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
};

// ============================================================================
// AdamaxParamState
// ============================================================================

struct AdamaxParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor exp_avg;  // First moment (m)
    Tensor exp_inf;  // Infinity norm (u)
};

// ============================================================================
// Adamax - Adam variant using infinity norm
// ============================================================================
// Kingma and Ba (2014), Section 7. Generalizes Adam to the L-infinity norm,
// replacing the v_t update with an exponentially weighted infinity norm.

class Adamax : public Optimizer {
public:
    Adamax(std::vector<Parameter*> params, AdamaxOptions options = AdamaxOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    Adamax(std::vector<Parameter*> params, double lr)
        : Adamax(std::move(params), AdamaxOptions(lr)) {}

    Adamax(std::vector<ParamGroup> param_groups, AdamaxOptions options = AdamaxOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        double beta1 = options_.beta1;
        double beta2 = options_.beta2;
        double eps = options_.eps;

        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<AdamaxParamState>(param);
                state->step++;

                // Initialize state
                if (!state->exp_avg.defined()) {
                    state->exp_avg = at::zeros(param->sizes());
                    state->exp_inf = at::zeros(param->sizes());
                }

                // Weight decay (L2)
                if (wd != 0.0) {
                    grad = grad.add(param->data(), at::Scalar(wd));
                }

                // m = beta1 * m + (1-beta1) * g
                state->exp_avg.mul_(at::Scalar(beta1));
                state->exp_avg.add_(grad, at::Scalar(1.0 - beta1));

                // u = max(beta2 * u, |g|)  -- element-wise
                // Compute via raw pointers since we need element-wise max
                Tensor scaled_inf = state->exp_inf.mul(at::Scalar(beta2));
                Tensor abs_grad = grad.abs();
                float* u_data = scaled_inf.mutable_data_ptr<float>();
                const float* ag_data = abs_grad.data_ptr<float>();
                int64_t n = scaled_inf.numel();
                for (int64_t i = 0; i < n; ++i) {
                    u_data[i] = std::max(u_data[i], ag_data[i]);
                }
                state->exp_inf = scaled_inf;

                // Bias-corrected first moment: m_hat = m / (1 - beta1^t)
                double bias_correction1 = 1.0 - std::pow(beta1, state->step);
                double step_size = lr / bias_correction1;

                // param -= step_size * m / (u + eps)
                // Using m directly (step_size includes bias correction)
                Tensor denom = state->exp_inf.add(at::Scalar(eps));
                param->data().addcdiv_(state->exp_avg, denom, at::Scalar(-step_size));
            }
        }
    }

    AdamaxOptions& options() { return options_; }
    const AdamaxOptions& options() const { return options_; }

private:
    AdamaxOptions options_;
};

} // namespace optim
} // namespace torch
