#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// AdadeltaOptions
// ============================================================================

struct AdadeltaOptions {
    double lr = 1.0;
    double rho = 0.9;
    double eps = 1e-6;
    double weight_decay = 0.0;

    AdadeltaOptions(double lr_ = 1.0) : lr(lr_) {}

    AdadeltaOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    AdadeltaOptions& rho_(double r) { rho = r; return *this; }
    AdadeltaOptions& eps_(double e) { eps = e; return *this; }
    AdadeltaOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
};

// ============================================================================
// AdadeltaParamState
// ============================================================================

struct AdadeltaParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor square_avg;  // EMA of g^2
    Tensor acc_delta;   // EMA of delta^2

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        Tensor step_t = at::zeros({});
        step_t.mutable_data_ptr<float>()[0] = static_cast<float>(step);
        m["step"] = step_t;
        if (square_avg.defined()) m["square_avg"] = square_avg;
        if (acc_delta.defined()) m["acc_delta"] = acc_delta;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("step");
        if (it != m.end()) step = static_cast<int64_t>(it->second.data_ptr<float>()[0]);
        it = m.find("square_avg");
        if (it != m.end()) square_avg = it->second.clone();
        it = m.find("acc_delta");
        if (it != m.end()) acc_delta = it->second.clone();
    }
};

// ============================================================================
// Adadelta - Adaptive Learning Rate Method
// ============================================================================
// Zeiler (2012). Uses ratio of RMS of parameter updates to RMS of gradients,
// eliminating the need to set a default learning rate.

class Adadelta : public Optimizer {
public:
    Adadelta(std::vector<Parameter*> params, AdadeltaOptions options = AdadeltaOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    Adadelta(std::vector<Parameter*> params, double lr)
        : Adadelta(std::move(params), AdadeltaOptions(lr)) {}

    Adadelta(std::vector<ParamGroup> param_groups, AdadeltaOptions options = AdadeltaOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;
            double rho = options_.rho;
            double eps = options_.eps;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<AdadeltaParamState>(param);
                state->step++;

                // Initialize state
                if (!state->square_avg.defined()) {
                    state->square_avg = at::zeros(param->sizes());
                    state->acc_delta = at::zeros(param->sizes());
                }

                // Weight decay
                if (wd != 0.0) {
                    grad = grad.add(param->data(), at::Scalar(wd));
                }

                // square_avg = rho * square_avg + (1-rho) * g^2
                state->square_avg.mul_(at::Scalar(rho));
                state->square_avg.addcmul_(grad, grad, at::Scalar(1.0 - rho));

                // std_delta = sqrt(acc_delta + eps)
                Tensor std_delta = state->acc_delta.add(at::Scalar(eps)).sqrt();

                // std_grad = sqrt(square_avg + eps)
                Tensor std_grad = state->square_avg.add(at::Scalar(eps)).sqrt();

                // delta = (std_delta / std_grad) * grad
                Tensor delta = std_delta.div(std_grad).mul(grad);

                // acc_delta = rho * acc_delta + (1-rho) * delta^2
                state->acc_delta.mul_(at::Scalar(rho));
                state->acc_delta.addcmul_(delta, delta, at::Scalar(1.0 - rho));

                // param -= lr * delta
                param->data().sub_(delta, at::Scalar(lr));
            }
        }
    }

    AdadeltaOptions& options() { return options_; }
    const AdadeltaOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<AdadeltaParamState>();
    }

private:
    AdadeltaOptions options_;
};

} // namespace optim
} // namespace torch
