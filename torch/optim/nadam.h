#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// NAdamOptions
// ============================================================================

struct NAdamOptions {
    double lr = 0.002;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;
    double momentum_decay = 0.004;

    NAdamOptions(double lr_ = 0.002) : lr(lr_) {}

    NAdamOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    NAdamOptions& betas(double b1, double b2) { beta1 = b1; beta2 = b2; return *this; }
    NAdamOptions& eps_(double e) { eps = e; return *this; }
    NAdamOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    NAdamOptions& momentum_decay_(double md) { momentum_decay = md; return *this; }
};

// ============================================================================
// NAdamParamState
// ============================================================================

struct NAdamParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor exp_avg;     // First moment (m)
    Tensor exp_avg_sq;  // Second moment (v)
    double mu_product = 1.0;  // Running product of mu_t values
};

// ============================================================================
// NAdam - Nesterov-accelerated Adam
// ============================================================================
// Dozat (2016). Incorporates Nesterov momentum into Adam by looking ahead
// with the next step's momentum coefficient.

class NAdam : public Optimizer {
public:
    NAdam(std::vector<Parameter*> params, NAdamOptions options = NAdamOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    NAdam(std::vector<Parameter*> params, double lr)
        : NAdam(std::move(params), NAdamOptions(lr)) {}

    NAdam(std::vector<ParamGroup> param_groups, NAdamOptions options = NAdamOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        double beta1 = options_.beta1;
        double beta2 = options_.beta2;
        double eps = options_.eps;
        double momentum_decay = options_.momentum_decay;

        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<NAdamParamState>(param);
                state->step++;

                // Initialize moments
                if (!state->exp_avg.defined()) {
                    state->exp_avg = at::zeros(param->sizes());
                    state->exp_avg_sq = at::zeros(param->sizes());
                }

                // Weight decay (L2)
                if (wd != 0.0) {
                    grad = grad.add(param->data(), at::Scalar(wd));
                }

                int64_t t = state->step;

                // mu_t = beta1 * (1 - 0.5 * 0.96^(t * momentum_decay))
                double mu_t = beta1 * (1.0 - 0.5 * std::pow(0.96, t * momentum_decay));
                double mu_t1 = beta1 * (1.0 - 0.5 * std::pow(0.96, (t + 1) * momentum_decay));

                // Update mu_product
                state->mu_product *= mu_t;

                // m = beta1 * m + (1-beta1) * g
                state->exp_avg.mul_(at::Scalar(beta1));
                state->exp_avg.add_(grad, at::Scalar(1.0 - beta1));

                // v = beta2 * v + (1-beta2) * g^2
                state->exp_avg_sq.mul_(at::Scalar(beta2));
                state->exp_avg_sq.addcmul_(grad, grad, at::Scalar(1.0 - beta2));

                // Bias-corrected second moment
                double bias_correction2 = 1.0 - std::pow(beta2, t);
                Tensor v_hat = state->exp_avg_sq.div(at::Scalar(bias_correction2));

                // Nesterov-corrected first moment:
                // m_hat = mu_{t+1} * m / (1 - mu_product * mu_{t+1}) + (1 - mu_t) * g / (1 - mu_product)
                double mu_product_next = state->mu_product * mu_t1;
                Tensor m_hat = state->exp_avg.mul(at::Scalar(mu_t1 / (1.0 - mu_product_next)))
                    .add(grad, at::Scalar((1.0 - mu_t) / (1.0 - state->mu_product)));

                // param -= lr * m_hat / (sqrt(v_hat) + eps)
                Tensor denom = v_hat.sqrt().add(at::Scalar(eps));
                param->data().addcdiv_(m_hat, denom, at::Scalar(-lr));
            }
        }
    }

    NAdamOptions& options() { return options_; }
    const NAdamOptions& options() const { return options_; }

private:
    NAdamOptions options_;
};

} // namespace optim
} // namespace torch
