#pragma once

#include "torch/optim/optimizer.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include <cmath>

namespace torch {
namespace optim {

// ============================================================================
// AdamOptions - Options for Adam optimizer
// ============================================================================

struct AdamOptions {
    double lr = 0.001;          // Learning rate
    double beta1 = 0.9;         // Coefficient for first moment estimates
    double beta2 = 0.999;       // Coefficient for second moment estimates
    double eps = 1e-8;          // Term added to denominator for numerical stability
    double weight_decay = 0.0;  // Weight decay (L2 penalty)
    bool amsgrad = false;       // Whether to use AMSGrad variant

    AdamOptions(double lr_ = 0.001) : lr(lr_) {}

    AdamOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    AdamOptions& betas(double b1, double b2) { beta1 = b1; beta2 = b2; return *this; }
    AdamOptions& eps_(double e) { eps = e; return *this; }
    AdamOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    AdamOptions& amsgrad_(bool a) { amsgrad = a; return *this; }
};

// ============================================================================
// AdamState - Per-parameter state for Adam
// ============================================================================

struct AdamParamState : public OptimizerParamState {
    int64_t step = 0;           // Number of steps
    Tensor exp_avg;             // First moment estimate (m)
    Tensor exp_avg_sq;          // Second moment estimate (v)
    Tensor max_exp_avg_sq;      // Max of second moment (for AMSGrad)
};

// ============================================================================
// Adam - Adaptive Moment Estimation Optimizer
// ============================================================================
// Implements Adam algorithm from "Adam: A Method for Stochastic Optimization"
// by Kingma and Ba (2014).
//
// Algorithm:
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//   m_hat = m_t / (1 - beta1^t)
//   v_hat = v_t / (1 - beta2^t)
//   p_t = p_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
//
// With AMSGrad:
//   v_hat = max(v_hat, v_{t-1})

class Adam : public Optimizer {
public:
    Adam(std::vector<Parameter*> params, AdamOptions options = AdamOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    Adam(std::vector<Parameter*> params, double lr)
        : Adam(std::move(params), AdamOptions(lr)) {}

    Adam(std::vector<ParamGroup> param_groups, AdamOptions options = AdamOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;
            double beta1 = options_.beta1;
            double beta2 = options_.beta2;
            double eps = options_.eps;
            bool amsgrad = options_.amsgrad;

            for (auto* param : group.params) {
                if (!param->defined()) continue;

                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                // Get or create state
                auto* state = get_or_create_state<AdamParamState>(param);

                // Increment step counter
                state->step++;

                // Initialize moment buffers
                if (!state->exp_avg.defined()) {
                    state->exp_avg = at::zeros(param->sizes());
                    state->exp_avg_sq = at::zeros(param->sizes());
                    if (amsgrad) {
                        state->max_exp_avg_sq = at::zeros(param->sizes());
                    }
#ifdef PT_USE_CUDA
                    if (param->data().is_cuda()) {
                        state->exp_avg = at::to_cuda(state->exp_avg);
                        state->exp_avg_sq = at::to_cuda(state->exp_avg_sq);
                        if (amsgrad) {
                            state->max_exp_avg_sq = at::to_cuda(state->max_exp_avg_sq);
                        }
                    }
#endif
                }

                // CUDA path: fallback to tensor ops
                bool is_cuda = false;
#ifdef PT_USE_CUDA
                is_cuda = param->data().is_cuda();
#endif
                if (is_cuda) {
                    step_cuda(param, grad, state, lr, wd, beta1, beta2, eps, amsgrad);
                    continue;
                }

                // ================================================================
                // CPU fast path: fused AVX2 Adam step
                // ================================================================
                Tensor grad_c = grad.contiguous();
                float* p_data = param->data().mutable_data_ptr<float>();
                const float* g_data = grad_c.data_ptr<float>();
                float* m_data = state->exp_avg.mutable_data_ptr<float>();
                float* v_data = state->exp_avg_sq.mutable_data_ptr<float>();
                int64_t n = param->numel();

                double bias_correction1 = 1.0 - std::pow(beta1, state->step);
                double bias_correction2 = 1.0 - std::pow(beta2, state->step);
                float step_size = static_cast<float>(lr / bias_correction1);
                float inv_sqrt_bc2 = static_cast<float>(1.0 / std::sqrt(bias_correction2));
                float b1f = static_cast<float>(beta1);
                float b2f = static_cast<float>(beta2);
                float one_minus_b1 = static_cast<float>(1.0 - beta1);
                float one_minus_b2 = static_cast<float>(1.0 - beta2);
                float wdf = static_cast<float>(wd);
                float epsf = static_cast<float>(eps);

                if (!amsgrad) {
                    using VF = at::native::tuda::VecF;
                    VF vb1 = VF::broadcast(b1f);
                    VF vb2 = VF::broadcast(b2f);
                    VF v1mb1 = VF::broadcast(one_minus_b1);
                    VF v1mb2 = VF::broadcast(one_minus_b2);
                    VF vneg_ss = VF::broadcast(-step_size);
                    VF veps = VF::broadcast(epsf);
                    VF v_isbc2 = VF::broadcast(inv_sqrt_bc2);
                    VF vwd = VF::broadcast(wdf);

                    int64_t i = 0;
                    constexpr int W = VF::width;
                    for (; i + W <= n; i += W) {
                        VF g = VF::load(g_data + i);
                        VF p = VF::load(p_data + i);

                        if (wdf != 0.0f) {
                            g = VF::fmadd(vwd, p, g);
                        }

                        VF m = VF::load(m_data + i);
                        VF v = VF::load(v_data + i);

                        // m = beta1 * m + (1-beta1) * g
                        m = VF::fmadd(vb1, m, v1mb1 * g);
                        // v = beta2 * v + (1-beta2) * g^2
                        v = VF::fmadd(vb2, v, v1mb2 * (g * g));

                        // denom = sqrt(v) * inv_sqrt_bc2 + eps
                        VF denom = VF::fmadd(v.sqrt(), v_isbc2, veps);
                        // p -= step_size * m / denom
                        p = VF::fmadd(vneg_ss, m / denom, p);

                        m.store(m_data + i);
                        v.store(v_data + i);
                        p.store(p_data + i);
                    }
                    // Scalar tail
                    for (; i < n; ++i) {
                        float g = g_data[i];
                        if (wdf != 0.0f) g += wdf * p_data[i];
                        m_data[i] = b1f * m_data[i] + one_minus_b1 * g;
                        v_data[i] = b2f * v_data[i] + one_minus_b2 * g * g;
                        float denom = std::sqrt(v_data[i]) * inv_sqrt_bc2 + epsf;
                        p_data[i] -= step_size * m_data[i] / denom;
                    }
                } else {
                    // AMSGrad path: scalar (rare case)
                    float* max_v_data = state->max_exp_avg_sq.mutable_data_ptr<float>();
                    for (int64_t i = 0; i < n; ++i) {
                        float g = g_data[i];
                        if (wdf != 0.0f) g += wdf * p_data[i];
                        m_data[i] = b1f * m_data[i] + one_minus_b1 * g;
                        v_data[i] = b2f * v_data[i] + one_minus_b2 * g * g;
                        max_v_data[i] = std::max(max_v_data[i], v_data[i]);
                        float denom = std::sqrt(max_v_data[i]) * inv_sqrt_bc2 + epsf;
                        p_data[i] -= step_size * m_data[i] / denom;
                    }
                }
            }
        }
    }

private:
    // CUDA fallback: uses tensor ops + CPU transfer
    void step_cuda(Parameter* param, const Tensor& grad, AdamParamState* state,
                   double lr, double wd, double beta1, double beta2, double eps, bool amsgrad) {
#ifdef PT_USE_CUDA
        Tensor grad_wd = grad;
        if (wd != 0.0) grad_wd = grad.add(param->data(), at::Scalar(wd));
        Tensor grad_cpu = at::to_cpu(grad_wd);
        Tensor exp_avg_cpu = at::to_cpu(state->exp_avg);
        Tensor exp_avg_sq_cpu = at::to_cpu(state->exp_avg_sq);

        exp_avg_cpu.mul_(at::Scalar(beta1));
        exp_avg_cpu.add_(grad_cpu, at::Scalar(1.0 - beta1));
        exp_avg_sq_cpu.mul_(at::Scalar(beta2));
        exp_avg_sq_cpu.addcmul_(grad_cpu, grad_cpu, at::Scalar(1.0 - beta2));

        state->exp_avg = at::to_cuda(exp_avg_cpu);
        state->exp_avg_sq = at::to_cuda(exp_avg_sq_cpu);

        double bias_correction1 = 1.0 - std::pow(beta1, state->step);
        double bias_correction2 = 1.0 - std::pow(beta2, state->step);
        double step_size = lr / bias_correction1;

        Tensor denom = exp_avg_sq_cpu.sqrt().div(at::Scalar(std::sqrt(bias_correction2)));
        denom.add_(at::Scalar(eps));

        Tensor param_cpu = at::to_cpu(param->data());
        param_cpu.addcdiv_(exp_avg_cpu, denom, at::Scalar(-step_size));
        at::cuda_ops::copy_(param->data(), at::to_cuda(param_cpu));
#endif
    }

public:

    // Get options
    AdamOptions& options() { return options_; }
    const AdamOptions& options() const { return options_; }

private:
    AdamOptions options_;
};

// ============================================================================
// AdamWOptions - Options for AdamW optimizer
// ============================================================================

struct AdamWOptions {
    double lr = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.01;  // Default non-zero for AdamW
    bool amsgrad = false;

    AdamWOptions(double lr_ = 0.001) : lr(lr_) {}

    AdamWOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    AdamWOptions& betas(double b1, double b2) { beta1 = b1; beta2 = b2; return *this; }
    AdamWOptions& eps_(double e) { eps = e; return *this; }
    AdamWOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    AdamWOptions& amsgrad_(bool a) { amsgrad = a; return *this; }
};

// ============================================================================
// AdamW - Adam with Decoupled Weight Decay
// ============================================================================
// Implements AdamW algorithm from "Decoupled Weight Decay Regularization"
// by Loshchilov and Hutter (2019).
//
// The key difference from Adam is that weight decay is applied directly to
// the parameters, not to the gradient. This is the "decoupled" weight decay.
//
// Algorithm:
//   p_t = (1 - lr * wd) * p_{t-1}    // Weight decay step
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//   m_hat = m_t / (1 - beta1^t)
//   v_hat = v_t / (1 - beta2^t)
//   p_t = p_t - lr * m_hat / (sqrt(v_hat) + eps)

class AdamW : public Optimizer {
public:
    AdamW(std::vector<Parameter*> params, AdamWOptions options = AdamWOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    AdamW(std::vector<Parameter*> params, double lr)
        : AdamW(std::move(params), AdamWOptions(lr)) {}

    AdamW(std::vector<ParamGroup> param_groups, AdamWOptions options = AdamWOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;
            double beta1 = options_.beta1;
            double beta2 = options_.beta2;
            double eps = options_.eps;
            bool amsgrad = options_.amsgrad;

            for (auto* param : group.params) {
                if (!param->defined()) continue;

                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                // Get or create state
                auto* state = get_or_create_state<AdamParamState>(param);

                // Increment step counter
                state->step++;

                // Initialize moment buffers - must be on same device as param
                if (!state->exp_avg.defined()) {
                    state->exp_avg = at::zeros(param->sizes());
                    state->exp_avg_sq = at::zeros(param->sizes());
                    if (amsgrad) {
                        state->max_exp_avg_sq = at::zeros(param->sizes());
                    }
#ifdef PT_USE_CUDA
                    // Move state tensors to same device as param
                    if (param->data().is_cuda()) {
                        state->exp_avg = at::to_cuda(state->exp_avg);
                        state->exp_avg_sq = at::to_cuda(state->exp_avg_sq);
                        if (amsgrad) {
                            state->max_exp_avg_sq = at::to_cuda(state->max_exp_avg_sq);
                        }
                    }
#endif
                }

                // DECOUPLED weight decay: apply directly to parameters
                // p = (1 - lr * wd) * p = p - lr * wd * p
                if (wd != 0.0) {
                    param->data().mul_(at::Scalar(1.0 - lr * wd));
                }

                // For CUDA tensors, move to CPU for optimizer math, then back
                Tensor grad_cpu = grad;
                Tensor exp_avg_cpu = state->exp_avg;
                Tensor exp_avg_sq_cpu = state->exp_avg_sq;
                bool is_cuda = grad.is_cuda();
#ifdef PT_USE_CUDA
                if (is_cuda) {
                    grad_cpu = at::to_cpu(grad);
                    exp_avg_cpu = at::to_cpu(state->exp_avg);
                    exp_avg_sq_cpu = at::to_cpu(state->exp_avg_sq);
                }
#endif

                // Update biased first moment estimate
                exp_avg_cpu.mul_(at::Scalar(beta1));
                exp_avg_cpu.add_(grad_cpu, at::Scalar(1.0 - beta1));

                // Update biased second moment estimate
                exp_avg_sq_cpu.mul_(at::Scalar(beta2));
                exp_avg_sq_cpu.addcmul_(grad_cpu, grad_cpu, at::Scalar(1.0 - beta2));

#ifdef PT_USE_CUDA
                // Move state back to CUDA
                if (is_cuda) {
                    state->exp_avg = at::to_cuda(exp_avg_cpu);
                    state->exp_avg_sq = at::to_cuda(exp_avg_sq_cpu);
                }
#else
                state->exp_avg = exp_avg_cpu;
                state->exp_avg_sq = exp_avg_sq_cpu;
#endif

                // Compute bias correction
                double bias_correction1 = 1.0 - std::pow(beta1, state->step);
                double bias_correction2 = 1.0 - std::pow(beta2, state->step);

                // For CUDA: do denom computation on CPU
                Tensor exp_avg_sq_for_denom = exp_avg_sq_cpu;  // Already on CPU from above
                Tensor exp_avg_for_update = exp_avg_cpu;       // Already on CPU from above

                // Compute denominator
                Tensor denom;
                if (amsgrad) {
                    Tensor max_exp_avg_sq_cpu = state->max_exp_avg_sq;
#ifdef PT_USE_CUDA
                    if (is_cuda) max_exp_avg_sq_cpu = at::to_cpu(state->max_exp_avg_sq);
#endif
                    // max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq)
                    float* max_data = max_exp_avg_sq_cpu.mutable_data_ptr<float>();
                    const float* sq_data = exp_avg_sq_for_denom.data_ptr<float>();
                    int64_t n = max_exp_avg_sq_cpu.numel();
                    for (int64_t i = 0; i < n; ++i) {
                        max_data[i] = std::max(max_data[i], sq_data[i]);
                    }
#ifdef PT_USE_CUDA
                    if (is_cuda) state->max_exp_avg_sq = at::to_cuda(max_exp_avg_sq_cpu);
                    else state->max_exp_avg_sq = max_exp_avg_sq_cpu;
#else
                    state->max_exp_avg_sq = max_exp_avg_sq_cpu;
#endif
                    denom = max_exp_avg_sq_cpu.sqrt().div(at::Scalar(std::sqrt(bias_correction2)));
                } else {
                    denom = exp_avg_sq_for_denom.sqrt().div(at::Scalar(std::sqrt(bias_correction2)));
                }
                denom.add_(at::Scalar(eps));

                // Compute step size
                double step_size = lr / bias_correction1;

                // Update parameters on CPU: p = p - step_size * m / denom
                Tensor param_cpu = param->data();
#ifdef PT_USE_CUDA
                if (is_cuda) param_cpu = at::to_cpu(param->data());
#endif
                param_cpu.addcdiv_(exp_avg_for_update, denom, at::Scalar(-step_size));
#ifdef PT_USE_CUDA
                if (is_cuda) {
                    // Copy result back to GPU param
                    at::cuda_ops::copy_(param->data(), at::to_cuda(param_cpu));
                }
#endif
            }
        }
    }

    // Get options
    AdamWOptions& options() { return options_; }
    const AdamWOptions& options() const { return options_; }

private:
    AdamWOptions options_;
};

} // namespace optim
} // namespace torch
