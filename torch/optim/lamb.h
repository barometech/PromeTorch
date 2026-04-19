#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// LAMBOptions
// ============================================================================

struct LAMBOptions {
    double lr = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-6;
    double weight_decay = 0.01;
    bool bias_correction = true;

    LAMBOptions(double lr_ = 1e-3) : lr(lr_) {}

    LAMBOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    LAMBOptions& betas(double b1, double b2) { beta1 = b1; beta2 = b2; return *this; }
    LAMBOptions& eps_(double e) { eps = e; return *this; }
    LAMBOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    LAMBOptions& bias_correction_(bool b) { bias_correction = b; return *this; }
};

// ============================================================================
// LAMBParamState
// ============================================================================

struct LAMBParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor exp_avg;     // m
    Tensor exp_avg_sq;  // v

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
// LAMB - You et al 2019, layer-wise adaptive scaling
// ============================================================================
// m_t = beta1 * m + (1-beta1) * g
// v_t = beta2 * v + (1-beta2) * g^2
// r   = m_hat / (sqrt(v_hat) + eps) + wd * p
// trust = ||p|| / ||r||  (1.0 when either norm is 0)
// p   = p - lr * trust * r

class LAMB : public Optimizer {
public:
    LAMB(std::vector<Parameter*> params, LAMBOptions options = LAMBOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    LAMB(std::vector<Parameter*> params, double lr)
        : LAMB(std::move(params), LAMBOptions(lr)) {}

    LAMB(std::vector<ParamGroup> param_groups, LAMBOptions options = LAMBOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        double beta1 = options_.beta1;
        double beta2 = options_.beta2;
        double eps = options_.eps;
        bool bc = options_.bias_correction;

        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<LAMBParamState>(param);
                state->step++;

                if (!state->exp_avg.defined()) {
                    state->exp_avg = at::zeros(param->sizes());
                    state->exp_avg_sq = at::zeros(param->sizes());
                }

                int64_t t = state->step;

                // m = beta1 * m + (1 - beta1) * g
                state->exp_avg.mul_(at::Scalar(beta1));
                state->exp_avg.add_(grad, at::Scalar(1.0 - beta1));

                // v = beta2 * v + (1 - beta2) * g^2
                state->exp_avg_sq.mul_(at::Scalar(beta2));
                state->exp_avg_sq.addcmul_(grad, grad, at::Scalar(1.0 - beta2));

                double bc1 = bc ? (1.0 - std::pow(beta1, t)) : 1.0;
                double bc2 = bc ? (1.0 - std::pow(beta2, t)) : 1.0;

                // adam_step = (m / bc1) / (sqrt(v / bc2) + eps)
                Tensor m_hat = state->exp_avg.div(at::Scalar(bc1));
                Tensor denom = state->exp_avg_sq.div(at::Scalar(bc2)).sqrt().add(at::Scalar(eps));
                Tensor r = m_hat.div(denom);

                // Decoupled weight decay (added inside r so trust ratio uses combined update)
                if (wd != 0.0) {
                    r.add_(param->data(), at::Scalar(wd));
                }

                // Trust ratio: ||p|| / ||r||
                float w_norm = param->data().norm(at::Scalar(2)).data_ptr<float>()[0];
                float r_norm = r.norm(at::Scalar(2)).data_ptr<float>()[0];
                double trust = 1.0;
                if (w_norm > 0.0f && r_norm > 0.0f) {
                    trust = static_cast<double>(w_norm) / static_cast<double>(r_norm);
                }

                // p = p - lr * trust * r
                param->data().sub_(r, at::Scalar(lr * trust));
            }
        }
    }

    LAMBOptions& options() { return options_; }
    const LAMBOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<LAMBParamState>();
    }

private:
    LAMBOptions options_;
};

} // namespace optim
} // namespace torch
