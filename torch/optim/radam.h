#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// RAdamOptions
// ============================================================================

struct RAdamOptions {
    double lr = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;

    RAdamOptions(double lr_ = 0.001) : lr(lr_) {}

    RAdamOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    RAdamOptions& betas(double b1, double b2) { beta1 = b1; beta2 = b2; return *this; }
    RAdamOptions& eps_(double e) { eps = e; return *this; }
    RAdamOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
};

// ============================================================================
// RAdamParamState
// ============================================================================

struct RAdamParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor exp_avg;     // First moment (m)
    Tensor exp_avg_sq;  // Second moment (v)

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        Tensor step_t = at::zeros({});
        step_t.mutable_data_ptr<float>()[0] = static_cast<float>(step);
        m["step"] = step_t;
        if (exp_avg.defined()) m["exp_avg"] = exp_avg;
        if (exp_avg_sq.defined()) m["exp_avg_sq"] = exp_avg_sq;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("step");
        if (it != m.end()) step = static_cast<int64_t>(it->second.data_ptr<float>()[0]);
        it = m.find("exp_avg");
        if (it != m.end()) exp_avg = it->second.clone();
        it = m.find("exp_avg_sq");
        if (it != m.end()) exp_avg_sq = it->second.clone();
    }
};

// ============================================================================
// RAdam - Rectified Adam
// ============================================================================
// Liu et al. (2019). Provides an automated, dynamic warm-up by computing
// the variance rectification term. When variance is tractable (rho_t > 5),
// uses full Adam; otherwise falls back to SGD with momentum.

class RAdam : public Optimizer {
public:
    RAdam(std::vector<Parameter*> params, RAdamOptions options = RAdamOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    RAdam(std::vector<Parameter*> params, double lr)
        : RAdam(std::move(params), RAdamOptions(lr)) {}

    RAdam(std::vector<ParamGroup> param_groups, RAdamOptions options = RAdamOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        double beta1 = options_.beta1;
        double beta2 = options_.beta2;
        double eps = options_.eps;
        double rho_inf = 2.0 / (1.0 - beta2) - 1.0;

        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<RAdamParamState>(param);
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

                // m = beta1 * m + (1-beta1) * g
                state->exp_avg.mul_(at::Scalar(beta1));
                state->exp_avg.add_(grad, at::Scalar(1.0 - beta1));

                // v = beta2 * v + (1-beta2) * g^2
                state->exp_avg_sq.mul_(at::Scalar(beta2));
                state->exp_avg_sq.addcmul_(grad, grad, at::Scalar(1.0 - beta2));

                // Bias-corrected first moment
                double bias_correction1 = 1.0 - std::pow(beta1, t);
                Tensor m_hat = state->exp_avg.div(at::Scalar(bias_correction1));

                // Compute rho_t
                double beta2_t = std::pow(beta2, t);
                double rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t);

                if (rho_t > 5.0) {
                    // Variance is tractable: use rectified Adam update
                    double bias_correction2 = 1.0 - beta2_t;
                    double r_t = std::sqrt(
                        (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                        ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                    );
                    double step_size = lr * r_t;
                    Tensor denom = state->exp_avg_sq.div(at::Scalar(bias_correction2)).sqrt().add(at::Scalar(eps));
                    param->data().addcdiv_(m_hat, denom, at::Scalar(-step_size));
                } else {
                    // Variance not tractable: SGD with bias-corrected momentum
                    param->data().sub_(m_hat, at::Scalar(lr));
                }
            }
        }
    }

    RAdamOptions& options() { return options_; }
    const RAdamOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<RAdamParamState>();
    }

private:
    RAdamOptions options_;
};

} // namespace optim
} // namespace torch
