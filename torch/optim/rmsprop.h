#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// RMSpropOptions - Options for RMSprop optimizer
// ============================================================================

struct RMSpropOptions {
    double lr = 0.01;           // Learning rate
    double alpha = 0.99;        // Smoothing constant (decay rate)
    double eps = 1e-8;          // Term added to denominator for numerical stability
    double weight_decay = 0.0;  // Weight decay (L2 penalty)
    double momentum = 0.0;      // Momentum factor
    bool centered = false;      // If true, compute centered RMSProp

    RMSpropOptions(double lr_ = 0.01) : lr(lr_) {}

    RMSpropOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    RMSpropOptions& alpha_(double a) { alpha = a; return *this; }
    RMSpropOptions& eps_(double e) { eps = e; return *this; }
    RMSpropOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    RMSpropOptions& momentum_(double m) { momentum = m; return *this; }
    RMSpropOptions& centered_(bool c) { centered = c; return *this; }
};

// ============================================================================
// RMSpropState - Per-parameter state for RMSprop
// ============================================================================

struct RMSpropParamState : public OptimizerParamState {
    Tensor square_avg;       // Running average of squared gradients
    Tensor grad_avg;         // Running average of gradients (for centered)
    Tensor momentum_buffer;  // Momentum buffer

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        if (square_avg.defined()) m["square_avg"] = square_avg;
        if (grad_avg.defined()) m["grad_avg"] = grad_avg;
        if (momentum_buffer.defined()) m["momentum_buffer"] = momentum_buffer;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("square_avg");
        if (it != m.end()) square_avg = it->second.clone();
        it = m.find("grad_avg");
        if (it != m.end()) grad_avg = it->second.clone();
        it = m.find("momentum_buffer");
        if (it != m.end()) momentum_buffer = it->second.clone();
    }
};

// ============================================================================
// RMSprop - Root Mean Square Propagation Optimizer
// ============================================================================
// Implements RMSprop algorithm proposed by G. Hinton in his Coursera class.
//
// Standard RMSprop:
//   v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
//   p_t = p_{t-1} - lr * g_t / (sqrt(v_t) + eps)
//
// Centered RMSprop (divides by uncentered variance):
//   v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
//   g_avg_t = alpha * g_avg_{t-1} + (1 - alpha) * g_t
//   p_t = p_{t-1} - lr * g_t / (sqrt(v_t - g_avg_t^2) + eps)
//
// With momentum:
//   buf_t = momentum * buf_{t-1} + g_t / (sqrt(v_t) + eps)
//   p_t = p_{t-1} - lr * buf_t

class RMSprop : public Optimizer {
public:
    RMSprop(std::vector<Parameter*> params, RMSpropOptions options = RMSpropOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    RMSprop(std::vector<Parameter*> params, double lr)
        : RMSprop(std::move(params), RMSpropOptions(lr)) {}

    RMSprop(std::vector<ParamGroup> param_groups, RMSpropOptions options = RMSpropOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;
            double alpha = options_.alpha;
            double eps = group.resolve_eps(options_.eps);
            double momentum = group.resolve_momentum(options_.momentum);
            bool centered = options_.centered;

            for (auto* param : group.params) {
                if (!param->defined()) continue;

                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                // Apply weight decay
                if (wd != 0.0) {
                    grad = grad.add(param->data(), at::Scalar(wd));
                }

                // Get or create state
                auto* state = get_or_create_state<RMSpropParamState>(param);

                // Initialize state buffers
                if (!state->square_avg.defined()) {
                    state->square_avg = at::zeros(param->sizes());
                    if (momentum > 0) {
                        state->momentum_buffer = at::zeros(param->sizes());
                    }
                    if (centered) {
                        state->grad_avg = at::zeros(param->sizes());
                    }
                }

                // Update running average of squared gradients
                // v = alpha * v + (1 - alpha) * g^2
                state->square_avg.mul_(at::Scalar(alpha));
                state->square_avg.addcmul_(grad, grad, at::Scalar(1.0 - alpha));

                // Compute denominator
                Tensor avg;
                if (centered) {
                    // Update running average of gradients
                    state->grad_avg.mul_(at::Scalar(alpha));
                    state->grad_avg.add_(grad, at::Scalar(1.0 - alpha));

                    // avg = sqrt(v - g_avg^2) + eps
                    avg = state->square_avg.addcmul(state->grad_avg, state->grad_avg, at::Scalar(-1.0)).sqrt();
                } else {
                    // avg = sqrt(v) + eps
                    avg = state->square_avg.sqrt();
                }
                avg.add_(at::Scalar(eps));

                if (momentum > 0) {
                    // buf = momentum * buf + g / avg
                    state->momentum_buffer.mul_(at::Scalar(momentum));
                    state->momentum_buffer.addcdiv_(grad, avg, at::Scalar(1.0));

                    // p = p - lr * buf
                    param->data().sub_(state->momentum_buffer, at::Scalar(lr));
                } else {
                    // p = p - lr * g / avg
                    param->data().addcdiv_(grad, avg, at::Scalar(-lr));
                }
            }
        }
    }

    // Get options
    RMSpropOptions& options() { return options_; }
    const RMSpropOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<RMSpropParamState>();
    }

private:
    RMSpropOptions options_;
};

} // namespace optim
} // namespace torch
