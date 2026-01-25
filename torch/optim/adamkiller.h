#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace torch {
namespace optim {

// ============================================================================
// ADAM-KILLER: Next-Generation Optimizer
// ============================================================================
// Target: 4-10x faster convergence than Adam
//
// Key innovations:
// 1. Adaptive momentum (curvature-aware beta1)
// 2. Per-layer learning rate scaling
// 3. Gradient prediction (lookahead with EMA)
// 4. Warm restarts for escaping local minima
// 5. Trust region-like update clipping
//
// Why this beats Adam:
// - Adam uses fixed beta1=0.9 regardless of loss landscape
// - Adam applies same lr to all layers (bad for deep networks)
// - Adam has no lookahead - reactive, not proactive
// - Adam can get stuck in sharp minima

struct AdamKillerOptions {
    double lr = 0.001;              // Base learning rate
    double beta1_min = 0.7;         // Minimum momentum (high curvature)
    double beta1_max = 0.95;        // Maximum momentum (low curvature)
    double beta2 = 0.999;           // EMA for second moment (like Adam)
    double eps = 1e-8;              // Numerical stability
    double weight_decay = 0.01;     // Decoupled weight decay (like AdamW)

    // Adaptive features
    bool use_adaptive_momentum = true;   // Curvature-aware beta1
    bool use_per_layer_lr = true;        // Auto-scale lr per layer
    bool use_gradient_prediction = true; // Lookahead with EMA
    bool use_warm_restarts = false;      // Periodic momentum reset
    int restart_period = 1000;           // Steps between restarts

    // Stability
    double max_grad_norm = 1.0;          // Gradient clipping
    double lr_scale_min = 0.5;           // Min per-layer lr scale (was 0.1)
    double lr_scale_max = 2.0;           // Max per-layer lr scale (was 10.0)

    AdamKillerOptions(double lr_ = 0.001) : lr(lr_) {}

    AdamKillerOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    AdamKillerOptions& beta1_range(double min_, double max_) {
        beta1_min = min_; beta1_max = max_; return *this;
    }
    AdamKillerOptions& beta2_(double b2) { beta2 = b2; return *this; }
    AdamKillerOptions& eps_(double e) { eps = e; return *this; }
    AdamKillerOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    AdamKillerOptions& adaptive_momentum(bool v) { use_adaptive_momentum = v; return *this; }
    AdamKillerOptions& per_layer_lr(bool v) { use_per_layer_lr = v; return *this; }
    AdamKillerOptions& gradient_prediction(bool v) { use_gradient_prediction = v; return *this; }
    AdamKillerOptions& warm_restarts(bool v, int period = 1000) {
        use_warm_restarts = v; restart_period = period; return *this;
    }
};

// ============================================================================
// AdamKillerState - Per-parameter state
// ============================================================================

struct AdamKillerParamState : public OptimizerParamState {
    int64_t step = 0;
    Tensor m;              // First moment (momentum)
    Tensor v;              // Second moment (adaptive lr)
    Tensor prev_grad;      // Previous gradient (for curvature estimation)
    Tensor grad_diff_ema;  // EMA of gradient differences (for prediction)
    double layer_lr_scale = 1.0;  // Per-layer lr multiplier
    double current_beta1 = 0.9;   // Current adaptive beta1
};

// ============================================================================
// ADAM-KILLER Implementation
// ============================================================================

class AdamKiller : public Optimizer {
public:
    AdamKiller(std::vector<Parameter*> params, AdamKillerOptions options = AdamKillerOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    AdamKiller(std::vector<Parameter*> params, double lr)
        : AdamKiller(std::move(params), AdamKillerOptions(lr)) {}

    void step() override {
        for (auto& group : param_groups_) {
            double base_lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;

            for (auto* param : group.params) {
                if (!param->defined()) continue;

                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                // Get or create state
                auto* state = get_or_create_state<AdamKillerParamState>(param);
                state->step++;

                // Initialize buffers
                if (!state->m.defined()) {
                    state->m = at::zeros(param->sizes());
                    state->v = at::zeros(param->sizes());
                    if (options_.use_gradient_prediction) {
                        state->grad_diff_ema = at::zeros(param->sizes());
                    }
                }

                // Work on CPU for numerical operations
                Tensor grad_cpu = grad;
                Tensor param_cpu = param->data();
                Tensor m_cpu = state->m;
                Tensor v_cpu = state->v;

                int64_t numel = param_cpu.numel();
                float* param_data = param_cpu.mutable_data_ptr<float>();
                float* m_data = m_cpu.mutable_data_ptr<float>();
                float* v_data = v_cpu.mutable_data_ptr<float>();
                const float* grad_data = grad_cpu.data_ptr<float>();

                // ============================================================
                // STEP 1: Gradient clipping (per-param norm)
                // ============================================================
                double grad_norm = 0.0;
                for (int64_t i = 0; i < numel; ++i) {
                    grad_norm += grad_data[i] * grad_data[i];
                }
                grad_norm = std::sqrt(grad_norm);

                double grad_scale = 1.0;
                if (grad_norm > options_.max_grad_norm && grad_norm > 1e-12) {
                    grad_scale = options_.max_grad_norm / grad_norm;
                }

                // ============================================================
                // STEP 2: Adaptive Momentum (curvature-aware beta1)
                // ============================================================
                double beta1 = options_.beta1_max;  // Default: high momentum

                if (options_.use_adaptive_momentum && state->prev_grad.defined()) {
                    // Estimate curvature as ||grad - prev_grad|| / lr
                    const float* prev_grad_data = state->prev_grad.data_ptr<float>();
                    double curvature = 0.0;
                    for (int64_t i = 0; i < numel; ++i) {
                        double diff = (grad_data[i] * grad_scale) - prev_grad_data[i];
                        curvature += diff * diff;
                    }
                    curvature = std::sqrt(curvature) / (base_lr + 1e-12);

                    // Normalize curvature (empirically tuned)
                    double norm_curv = std::min(curvature / 100.0, 1.0);

                    // High curvature -> low beta1 (react faster)
                    // Low curvature -> high beta1 (accelerate)
                    beta1 = options_.beta1_max - norm_curv * (options_.beta1_max - options_.beta1_min);
                    beta1 = std::max(options_.beta1_min, std::min(options_.beta1_max, beta1));
                }
                state->current_beta1 = beta1;

                // ============================================================
                // STEP 3: Gradient Prediction (lookahead)
                // ============================================================
                std::vector<float> effective_grad(numel);
                for (int64_t i = 0; i < numel; ++i) {
                    effective_grad[i] = grad_data[i] * static_cast<float>(grad_scale);
                }

                if (options_.use_gradient_prediction && state->prev_grad.defined()) {
                    float* diff_ema_data = state->grad_diff_ema.mutable_data_ptr<float>();
                    const float* prev_grad_data = state->prev_grad.data_ptr<float>();

                    double beta_diff = 0.5;  // EMA coefficient for gradient difference
                    for (int64_t i = 0; i < numel; ++i) {
                        // Update EMA of gradient difference
                        float diff = effective_grad[i] - prev_grad_data[i];
                        diff_ema_data[i] = static_cast<float>(beta_diff) * diff_ema_data[i] +
                                          static_cast<float>(1.0 - beta_diff) * diff;

                        // Predict next gradient: current + trend
                        // This is the "lookahead" - we anticipate where the gradient is going
                        effective_grad[i] += 0.5f * diff_ema_data[i];
                    }
                }

                // Save current gradient for next iteration
                if (!state->prev_grad.defined()) {
                    state->prev_grad = at::zeros(param->sizes());
                }
                float* prev_grad_data = state->prev_grad.mutable_data_ptr<float>();
                for (int64_t i = 0; i < numel; ++i) {
                    prev_grad_data[i] = grad_data[i] * static_cast<float>(grad_scale);
                }

                // ============================================================
                // STEP 4: Update moments (like Adam but with adaptive beta1)
                // ============================================================
                double beta2 = options_.beta2;

                for (int64_t i = 0; i < numel; ++i) {
                    // First moment (momentum)
                    m_data[i] = static_cast<float>(beta1) * m_data[i] +
                               static_cast<float>(1.0 - beta1) * effective_grad[i];

                    // Second moment (adaptive lr)
                    v_data[i] = static_cast<float>(beta2) * v_data[i] +
                               static_cast<float>(1.0 - beta2) * effective_grad[i] * effective_grad[i];
                }

                // ============================================================
                // STEP 5: Per-layer learning rate scaling (LARS-style)
                // ============================================================
                double layer_lr = base_lr;

                if (options_.use_per_layer_lr) {
                    // Compute weight norm and gradient norm
                    double weight_norm = 0.0;
                    double effective_grad_norm = 0.0;
                    for (int64_t i = 0; i < numel; ++i) {
                        weight_norm += param_data[i] * param_data[i];
                        effective_grad_norm += effective_grad[i] * effective_grad[i];
                    }
                    weight_norm = std::sqrt(weight_norm);
                    effective_grad_norm = std::sqrt(effective_grad_norm);

                    // LARS-style trust ratio: scale = weight_norm / (grad_norm + wd * weight_norm)
                    // This prevents lr from exploding when gradients are small
                    if (effective_grad_norm > 1e-8 && weight_norm > 1e-8) {
                        double denominator = effective_grad_norm + options_.weight_decay * weight_norm;
                        double trust_ratio = weight_norm / denominator;

                        // Clamp trust ratio to reasonable range
                        trust_ratio = std::max(options_.lr_scale_min,
                                              std::min(options_.lr_scale_max, trust_ratio));
                        state->layer_lr_scale = trust_ratio;
                        layer_lr = base_lr * trust_ratio;
                    }
                }

                // ============================================================
                // STEP 6: Bias correction
                // ============================================================
                double bc1 = 1.0 - std::pow(beta1, state->step);
                double bc2 = 1.0 - std::pow(beta2, state->step);

                // ============================================================
                // STEP 7: Decoupled weight decay (like AdamW)
                // ============================================================
                if (wd > 0) {
                    for (int64_t i = 0; i < numel; ++i) {
                        param_data[i] *= static_cast<float>(1.0 - layer_lr * wd);
                    }
                }

                // ============================================================
                // STEP 8: Parameter update with trust region
                // ============================================================
                double step_size = layer_lr / bc1;
                double update_norm_sq = 0.0;
                std::vector<float> updates(numel);

                for (int64_t i = 0; i < numel; ++i) {
                    float m_hat = m_data[i] / static_cast<float>(bc1);
                    float v_hat = v_data[i] / static_cast<float>(bc2);

                    // Stable denominator: sqrt(v_hat + eps)
                    float denom = std::sqrt(v_hat + static_cast<float>(options_.eps));

                    updates[i] = static_cast<float>(step_size) * m_hat / denom;
                    update_norm_sq += updates[i] * updates[i];
                }

                // Trust region: clip total update norm
                double update_norm = std::sqrt(update_norm_sq);
                double max_update = layer_lr * 2.0;  // Allow 2x lr as max update
                double update_scale = 1.0;
                if (update_norm > max_update && update_norm > 1e-12) {
                    update_scale = max_update / update_norm;
                }

                // Apply updates
                for (int64_t i = 0; i < numel; ++i) {
                    param_data[i] -= updates[i] * static_cast<float>(update_scale);
                }

                // ============================================================
                // STEP 9: Warm restarts (optional)
                // ============================================================
                if (options_.use_warm_restarts &&
                    options_.restart_period > 0 &&
                    state->step % options_.restart_period == 0) {
                    // Reset momentum but keep v (second moment)
                    state->m.zero_();
                    if (state->grad_diff_ema.defined()) {
                        state->grad_diff_ema.zero_();
                    }
                }
            }
        }
    }

    // Getters
    AdamKillerOptions& options() { return options_; }
    const AdamKillerOptions& options() const { return options_; }

    // Debug: print adaptive parameters
    void print_stats() const {
        std::cout << "[AdamKiller Stats]" << std::endl;
        int param_idx = 0;
        for (const auto& [param, state_ptr] : state_) {
            auto* state = static_cast<AdamKillerParamState*>(state_ptr.get());
            std::cout << "  Param " << param_idx++
                      << ": step=" << state->step
                      << " beta1=" << state->current_beta1
                      << " lr_scale=" << state->layer_lr_scale
                      << std::endl;
        }
    }

private:
    AdamKillerOptions options_;
};

// ============================================================================
// Convenience functions
// ============================================================================

inline AdamKiller make_adamkiller(std::vector<Parameter*> params, double lr = 0.001) {
    return AdamKiller(std::move(params), AdamKillerOptions(lr));
}

inline AdamKiller make_adamkiller_aggressive(std::vector<Parameter*> params, double lr = 0.001) {
    // Aggressive settings for fast convergence (but still stable)
    auto options = AdamKillerOptions(lr)
        .beta1_range(0.7, 0.95)   // Adaptive momentum range
        .gradient_prediction(true)
        .adaptive_momentum(true)
        .per_layer_lr(false);     // Disabled for stability
    return AdamKiller(std::move(params), options);
}

inline AdamKiller make_adamkiller_stable(std::vector<Parameter*> params, double lr = 0.001) {
    // Conservative settings for stability
    auto options = AdamKillerOptions(lr)
        .beta1_range(0.85, 0.95)  // Higher momentum range
        .gradient_prediction(false)
        .adaptive_momentum(true)
        .per_layer_lr(false);     // Disabled for maximum stability
    return AdamKiller(std::move(params), options);
}

} // namespace optim
} // namespace torch
