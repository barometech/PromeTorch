#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// SophiaGOptions
// ============================================================================

struct SophiaGOptions {
    double lr = 1e-4;
    double beta1 = 0.965;
    double beta2 = 0.99;         // EMA for Hessian diagonal estimate
    double eps = 1e-12;
    double weight_decay = 0.0;
    double rho = 0.04;           // Per-coordinate clipping threshold
    int hessian_update_interval = 10;  // k: Hessian re-estimated every k steps

    SophiaGOptions(double lr_ = 1e-4) : lr(lr_) {}

    SophiaGOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    SophiaGOptions& betas(double b1, double b2) { beta1 = b1; beta2 = b2; return *this; }
    SophiaGOptions& eps_(double e) { eps = e; return *this; }
    SophiaGOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    SophiaGOptions& rho_(double r) { rho = r; return *this; }
};

// ============================================================================
// SophiaGParamState
// ============================================================================

struct SophiaGParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor exp_avg;   // First moment m
    Tensor hessian;   // EMA of Hessian diagonal estimate (h)

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        Tensor step_t = at::zeros({});
        step_t.mutable_data_ptr<float>()[0] = static_cast<float>(step);
        m["step"] = step_t;
        if (exp_avg.defined()) m["exp_avg"] = exp_avg;
        if (hessian.defined()) m["hessian"] = hessian;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("step");
        if (it != m.end()) step = static_cast<int64_t>(it->second.data_ptr<float>()[0]);
        it = m.find("exp_avg");
        if (it != m.end()) exp_avg = it->second.clone();
        it = m.find("hessian");
        if (it != m.end()) hessian = it->second.clone();
    }
};

// ============================================================================
// SophiaG - Liu et al 2023, Gauss-Newton-Bartlett variant
// ============================================================================
// Maintains diagonal Hessian estimate; per-coordinate clipped second-order step.
//   m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
//   h_t   = beta2 * h_{t-1} + (1 - beta2) * g_t * g_t   (GNB approximation)
//   p_t   = p_{t-1} - lr * clip(m_t / max(h_t, eps), -rho, rho)   (with WD)
// Hessian only refreshed every k steps in original paper; here updated each
// step using grad^2 (cheap GNB proxy) which still preserves coord clipping.

class SophiaG : public Optimizer {
public:
    SophiaG(std::vector<Parameter*> params, SophiaGOptions options = SophiaGOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    SophiaG(std::vector<Parameter*> params, double lr)
        : SophiaG(std::move(params), SophiaGOptions(lr)) {}

    SophiaG(std::vector<ParamGroup> param_groups, SophiaGOptions options = SophiaGOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        double beta1 = options_.beta1;
        double beta2 = options_.beta2;
        double eps = options_.eps;
        double rho = options_.rho;

        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<SophiaGParamState>(param);
                state->step++;

                if (!state->exp_avg.defined()) {
                    state->exp_avg = at::zeros(param->sizes());
                    state->hessian = at::zeros(param->sizes());
                }

                // Decoupled weight decay
                if (wd != 0.0) {
                    param->data().mul_(at::Scalar(1.0 - lr * wd));
                }

                // m = beta1 * m + (1 - beta1) * g
                state->exp_avg.mul_(at::Scalar(beta1));
                state->exp_avg.add_(grad, at::Scalar(1.0 - beta1));

                // h = beta2 * h + (1 - beta2) * g^2  (GNB diagonal proxy)
                state->hessian.mul_(at::Scalar(beta2));
                state->hessian.addcmul_(grad, grad, at::Scalar(1.0 - beta2));

                // ratio = m / max(h, eps); then clip to [-rho, rho]
                Tensor denom = state->hessian.clamp_min(at::Scalar(eps));
                Tensor ratio = state->exp_avg.div(denom).clamp(at::Scalar(-rho), at::Scalar(rho));

                // p -= lr * ratio
                param->data().sub_(ratio, at::Scalar(lr));
            }
        }
    }

    SophiaGOptions& options() { return options_; }
    const SophiaGOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<SophiaGParamState>();
    }

private:
    SophiaGOptions options_;
};

// Alias for convenience
using Sophia = SophiaG;

} // namespace optim
} // namespace torch
