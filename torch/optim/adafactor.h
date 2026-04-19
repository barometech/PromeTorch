#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>
#include <algorithm>

namespace torch {
namespace optim {

// ============================================================================
// AdafactorOptions
// ============================================================================
// lr<=0 enables relative_step (lr derived from step).
// scale_parameter rescales lr by max(eps2, RMS(p)) for parameter-relative steps.

struct AdafactorOptions {
    double lr = 0.0;             // 0 => relative step schedule
    double eps1 = 1e-30;         // Stability for v
    double eps2 = 1e-3;          // Stability for parameter scale
    double clip_threshold = 1.0; // Update RMS clip
    double decay_rate = -0.8;    // beta2 schedule exponent
    double beta1 = 0.0;          // 0 disables first-moment EMA
    double weight_decay = 0.0;
    bool scale_parameter = true;
    bool relative_step = true;
    bool warmup_init = false;

    AdafactorOptions(double lr_ = 0.0) : lr(lr_) {}

    AdafactorOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    AdafactorOptions& eps_(double e1, double e2) { eps1 = e1; eps2 = e2; return *this; }
    AdafactorOptions& clip_threshold_(double c) { clip_threshold = c; return *this; }
    AdafactorOptions& decay_rate_(double d) { decay_rate = d; return *this; }
    AdafactorOptions& beta1_(double b) { beta1 = b; return *this; }
    AdafactorOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    AdafactorOptions& scale_parameter_(bool b) { scale_parameter = b; return *this; }
    AdafactorOptions& relative_step_(bool b) { relative_step = b; return *this; }
    AdafactorOptions& warmup_init_(bool b) { warmup_init = b; return *this; }
};

// ============================================================================
// AdafactorParamState
// ============================================================================
// For tensors with rank>=2: factored second moment via row + col stats.
// For tensors with rank<2: fall back to dense exp_avg_sq.

struct AdafactorParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor exp_avg;          // Optional m (only when beta1>0)
    Tensor exp_avg_sq;       // Dense v (rank<2 path)
    Tensor exp_avg_sq_row;   // Row factor (rank>=2 path), shape = param.sizes minus last dim
    Tensor exp_avg_sq_col;   // Col factor (rank>=2 path), shape = param.sizes minus second-to-last dim
    double rms_p = 0.0;      // Cached parameter RMS (informational)

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        Tensor step_t = at::zeros({});
        step_t.mutable_data_ptr<float>()[0] = static_cast<float>(step);
        m["step"] = step_t;
        if (exp_avg.defined()) m["exp_avg"] = exp_avg;
        if (exp_avg_sq.defined()) m["exp_avg_sq"] = exp_avg_sq;
        if (exp_avg_sq_row.defined()) m["exp_avg_sq_row"] = exp_avg_sq_row;
        if (exp_avg_sq_col.defined()) m["exp_avg_sq_col"] = exp_avg_sq_col;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("step");
        if (it != m.end()) step = static_cast<int64_t>(it->second.data_ptr<float>()[0]);
        it = m.find("exp_avg");
        if (it != m.end()) exp_avg = it->second.clone();
        it = m.find("exp_avg_sq");
        if (it != m.end()) exp_avg_sq = it->second.clone();
        it = m.find("exp_avg_sq_row");
        if (it != m.end()) exp_avg_sq_row = it->second.clone();
        it = m.find("exp_avg_sq_col");
        if (it != m.end()) exp_avg_sq_col = it->second.clone();
    }
};

// ============================================================================
// Adafactor - Shazeer & Stern 2018
// ============================================================================
// Memory-efficient adaptive optimizer using factored second-moment estimates
// (row + col EMAs) for matrices instead of full v. Used to train T5/PaLM.

class Adafactor : public Optimizer {
public:
    Adafactor(std::vector<Parameter*> params, AdafactorOptions options = AdafactorOptions())
        : Optimizer(std::move(params), options.lr > 0 ? options.lr : 1e-3),
          options_(options) {}

    Adafactor(std::vector<Parameter*> params, double lr)
        : Adafactor(std::move(params), AdafactorOptions(lr)) {}

    Adafactor(std::vector<ParamGroup> param_groups, AdafactorOptions options = AdafactorOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        double clip_threshold = options_.clip_threshold;
        double decay_rate = options_.decay_rate;
        double beta1 = options_.beta1;
        double eps1 = options_.eps1;
        double eps2 = options_.eps2;

        for (auto& group : param_groups_) {
            double base_lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;
                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                auto* state = get_or_create_state<AdafactorParamState>(param);
                state->step++;
                int64_t t = state->step;

                auto sizes = param->sizes();
                int64_t ndim = static_cast<int64_t>(sizes.size());
                bool factored = (ndim >= 2);

                if (factored) {
                    if (!state->exp_avg_sq_row.defined()) {
                        std::vector<int64_t> row_shape(sizes.begin(), sizes.end() - 1);
                        std::vector<int64_t> col_shape(sizes.begin(), sizes.end());
                        col_shape.erase(col_shape.end() - 2);
                        state->exp_avg_sq_row = at::zeros(row_shape);
                        state->exp_avg_sq_col = at::zeros(col_shape);
                    }
                } else {
                    if (!state->exp_avg_sq.defined()) {
                        state->exp_avg_sq = at::zeros(param->sizes());
                    }
                }
                if (beta1 > 0.0 && !state->exp_avg.defined()) {
                    state->exp_avg = at::zeros(param->sizes());
                }

                // Parameter RMS for scale_parameter / lr_t
                double rms_p = 0.0;
                {
                    Tensor norm_p = param->data().norm(at::Scalar(2));
                    double n = static_cast<double>(norm_p.data_ptr<float>()[0]);
                    int64_t numel = param->numel();
                    rms_p = numel > 0 ? n / std::sqrt(static_cast<double>(numel)) : 0.0;
                }
                state->rms_p = rms_p;

                // Effective lr
                double lr_t = base_lr;
                if (options_.relative_step || base_lr <= 0.0) {
                    double min_step = options_.warmup_init
                        ? 1e-6 * static_cast<double>(t)
                        : 1e-2;
                    double rel = std::min(min_step, 1.0 / std::sqrt(static_cast<double>(t)));
                    lr_t = rel;
                }
                if (options_.scale_parameter) {
                    lr_t *= std::max(eps2, rms_p);
                }

                // beta2_t = 1 - t^decay_rate
                double beta2_t = 1.0 - std::pow(static_cast<double>(t), decay_rate);

                // grad^2 + eps1 (avoid log(0))
                Tensor g2 = grad.mul(grad).add(at::Scalar(eps1));

                Tensor update;  // Per-coord step direction (sign of -gradient already encoded later)
                if (factored) {
                    // Row factor: mean over last dim
                    Tensor g2_row = g2.mean(ndim - 1, /*keepdim=*/false);
                    // Col factor: mean over second-to-last dim
                    Tensor g2_col = g2.mean(ndim - 2, /*keepdim=*/false);

                    state->exp_avg_sq_row.mul_(at::Scalar(beta2_t));
                    state->exp_avg_sq_row.add_(g2_row, at::Scalar(1.0 - beta2_t));
                    state->exp_avg_sq_col.mul_(at::Scalar(beta2_t));
                    state->exp_avg_sq_col.add_(g2_col, at::Scalar(1.0 - beta2_t));

                    // Reconstruct v ~= row * col / mean(row)
                    // mean over last remaining dim of row gives normalizer (a scalar per leading slice).
                    Tensor row_mean = state->exp_avg_sq_row.mean(ndim - 2, /*keepdim=*/true);
                    Tensor row_factor = state->exp_avg_sq_row.unsqueeze(ndim - 1)
                        .div(row_mean.unsqueeze(ndim - 1).clamp_min(at::Scalar(eps1)));
                    Tensor col_factor = state->exp_avg_sq_col.unsqueeze(ndim - 2);
                    // v_approx = row_factor (broadcast last dim) * col_factor (broadcast second-to-last)
                    Tensor v_approx = row_factor.mul(col_factor);
                    // update = grad / sqrt(v_approx)
                    update = grad.div(v_approx.sqrt().clamp_min(at::Scalar(eps1)));
                } else {
                    state->exp_avg_sq.mul_(at::Scalar(beta2_t));
                    state->exp_avg_sq.add_(g2, at::Scalar(1.0 - beta2_t));
                    update = grad.div(state->exp_avg_sq.sqrt().clamp_min(at::Scalar(eps1)));
                }

                // Clip update by RMS / clip_threshold
                {
                    Tensor norm_u = update.norm(at::Scalar(2));
                    double n = static_cast<double>(norm_u.data_ptr<float>()[0]);
                    int64_t numel = param->numel();
                    double rms_u = numel > 0 ? n / std::sqrt(static_cast<double>(numel)) : 0.0;
                    double scale = std::max(1.0, rms_u / clip_threshold);
                    if (scale > 1.0) {
                        update.div_(at::Scalar(scale));
                    }
                }

                if (beta1 > 0.0) {
                    state->exp_avg.mul_(at::Scalar(beta1));
                    state->exp_avg.add_(update, at::Scalar(1.0 - beta1));
                    update = state->exp_avg;
                }

                // Decoupled weight decay
                if (wd != 0.0) {
                    param->data().mul_(at::Scalar(1.0 - lr_t * wd));
                }

                // p -= lr_t * update
                param->data().sub_(update, at::Scalar(lr_t));
            }
        }
    }

    AdafactorOptions& options() { return options_; }
    const AdafactorOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<AdafactorParamState>();
    }

private:
    AdafactorOptions options_;
};

} // namespace optim
} // namespace torch
