#pragma once

#include "torch/optim/optimizer.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
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

    std::unordered_map<std::string, Tensor> save() const override {
        std::unordered_map<std::string, Tensor> m;
        // Store step as a 0-dim float tensor
        Tensor step_t = at::zeros({});
        step_t.mutable_data_ptr<float>()[0] = static_cast<float>(step);
        m["step"] = step_t;
        if (exp_avg.defined()) m["exp_avg"] = exp_avg;
        if (exp_avg_sq.defined()) m["exp_avg_sq"] = exp_avg_sq;
        if (max_exp_avg_sq.defined()) m["max_exp_avg_sq"] = max_exp_avg_sq;
        return m;
    }

    void load(const std::unordered_map<std::string, Tensor>& m) override {
        auto it = m.find("step");
        if (it != m.end()) step = static_cast<int64_t>(it->second.data_ptr<float>()[0]);
        it = m.find("exp_avg");
        if (it != m.end()) exp_avg = it->second.clone();
        it = m.find("exp_avg_sq");
        if (it != m.end()) exp_avg_sq = it->second.clone();
        it = m.find("max_exp_avg_sq");
        if (it != m.end()) max_exp_avg_sq = it->second.clone();
    }
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
            double beta1 = group.resolve_beta1(options_.beta1);
            double beta2 = group.resolve_beta2(options_.beta2);
            double eps = group.resolve_eps(options_.eps);
            bool amsgrad = group.resolve_amsgrad(options_.amsgrad);

            // ================================================================
            // Phase 1: Initialize states, handle CUDA params individually
            // ================================================================
            std::vector<at::native::hot::AdamParamPack> cpu_packs;
            std::vector<float*> amsgrad_ptrs;  // max_exp_avg_sq pointers
            // Keep contiguous grads alive until fused_adam_multi completes
            std::vector<Tensor> grad_holders;
            int64_t step_val = 0;

            for (auto* param : group.params) {
                if (!param->defined()) continue;

                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                // Get or create state
                auto* state = get_or_create_state<AdamParamState>(param);

                // Increment step counter
                state->step++;
                step_val = state->step;  // All params in group share step count

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

                // CUDA path: fallback to tensor ops (per-parameter)
                bool is_cuda = false;
#ifdef PT_USE_CUDA
                is_cuda = param->data().is_cuda();
#endif
                if (is_cuda) {
                    step_cuda(param, grad, state, lr, wd, beta1, beta2, eps, amsgrad);
                    continue;
                }

                // ================================================================
                // CPU path: collect into packs for fused multi-param step
                // ================================================================
                // Avoid contiguous() copy if grad is already contiguous
                Tensor grad_c = grad.is_contiguous() ? grad : grad.contiguous();
                if (!grad.is_contiguous()) grad_holders.push_back(grad_c);

                at::native::hot::AdamParamPack pack;
                pack.param = param->data().mutable_data_ptr<float>();
                pack.grad = grad_c.data_ptr<float>();
                pack.exp_avg = state->exp_avg.mutable_data_ptr<float>();
                pack.exp_avg_sq = state->exp_avg_sq.mutable_data_ptr<float>();
                pack.numel = param->numel();
                cpu_packs.push_back(pack);

                if (amsgrad) {
                    amsgrad_ptrs.push_back(state->max_exp_avg_sq.mutable_data_ptr<float>());
                }
            }

            // ================================================================
            // Phase 2: Fused multi-parameter Adam step (ONE call for all CPU params)
            // ================================================================
            if (!cpu_packs.empty()) {
                at::native::hot::fused_adam_multi(
                    cpu_packs.data(), static_cast<int>(cpu_packs.size()),
                    static_cast<float>(lr),
                    static_cast<float>(beta1), static_cast<float>(beta2),
                    static_cast<float>(eps), static_cast<float>(wd),
                    static_cast<int>(step_val),
                    amsgrad, amsgrad ? amsgrad_ptrs.data() : nullptr);
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

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<AdamParamState>();
    }

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
            double beta1 = group.resolve_beta1(options_.beta1);
            double beta2 = group.resolve_beta2(options_.beta2);
            double eps = group.resolve_eps(options_.eps);
            bool amsgrad = group.resolve_amsgrad(options_.amsgrad);

            // ================================================================
            // Phase 1: Initialize states, collect CPU float params for fused step
            // ================================================================
            std::vector<at::native::hot::AdamWParamPack> cpu_packs;
            std::vector<Tensor> grad_holders;
            int64_t step_val = 0;

            for (auto* param : group.params) {
                if (!param->defined()) continue;

                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                // Get or create state
                auto* state = get_or_create_state<AdamParamState>(param);

                // Increment step counter
                state->step++;
                step_val = state->step;

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

                // CUDA path: fallback to per-parameter tensor ops
                bool is_cuda = false;
#ifdef PT_USE_CUDA
                is_cuda = param->data().is_cuda();
#endif
                if (is_cuda || amsgrad ||
                    param->data().dtype() != c10::ScalarType::Float) {
                    // Fallback: original per-parameter path
                    if (wd != 0.0) {
                        param->data().mul_(at::Scalar(1.0 - lr * wd));
                    }
                    Tensor grad_cpu = grad;
                    Tensor exp_avg_cpu = state->exp_avg;
                    Tensor exp_avg_sq_cpu = state->exp_avg_sq;
#ifdef PT_USE_CUDA
                    if (is_cuda) {
                        grad_cpu = at::to_cpu(grad);
                        exp_avg_cpu = at::to_cpu(state->exp_avg);
                        exp_avg_sq_cpu = at::to_cpu(state->exp_avg_sq);
                    }
#endif
                    exp_avg_cpu.mul_(at::Scalar(beta1));
                    exp_avg_cpu.add_(grad_cpu, at::Scalar(1.0 - beta1));
                    exp_avg_sq_cpu.mul_(at::Scalar(beta2));
                    exp_avg_sq_cpu.addcmul_(grad_cpu, grad_cpu, at::Scalar(1.0 - beta2));
#ifdef PT_USE_CUDA
                    if (is_cuda) {
                        state->exp_avg = at::to_cuda(exp_avg_cpu);
                        state->exp_avg_sq = at::to_cuda(exp_avg_sq_cpu);
                    }
#endif
                    double bc1 = 1.0 - std::pow(beta1, state->step);
                    double bc2 = 1.0 - std::pow(beta2, state->step);
                    Tensor denom = exp_avg_sq_cpu.sqrt().div(at::Scalar(std::sqrt(bc2)));
                    denom.add_(at::Scalar(eps));
                    double step_size = lr / bc1;
                    Tensor param_cpu = param->data();
#ifdef PT_USE_CUDA
                    if (is_cuda) param_cpu = at::to_cpu(param->data());
#endif
                    param_cpu.addcdiv_(exp_avg_cpu, denom, at::Scalar(-step_size));
#ifdef PT_USE_CUDA
                    if (is_cuda) at::cuda_ops::copy_(param->data(), at::to_cuda(param_cpu));
#endif
                    continue;
                }

                // ================================================================
                // CPU float path: collect for FUSED multi-param AdamW step
                // Single pass does BOTH decoupled weight decay AND Adam update
                // (eliminates separate mul_ pass over params)
                // ================================================================
                Tensor grad_c = grad.is_contiguous() ? grad : grad.contiguous();
                if (!grad.is_contiguous()) grad_holders.push_back(grad_c);

                at::native::hot::AdamWParamPack pack;
                pack.param = param->data().mutable_data_ptr<float>();
                pack.grad = grad_c.data_ptr<float>();
                pack.exp_avg = state->exp_avg.mutable_data_ptr<float>();
                pack.exp_avg_sq = state->exp_avg_sq.mutable_data_ptr<float>();
                pack.numel = param->numel();
                cpu_packs.push_back(pack);
            }

            // ================================================================
            // Phase 2: Fused multi-parameter AdamW step (ONE call for all CPU params)
            // Weight decay + moment updates + param update in SINGLE pass per element
            // ================================================================
            if (!cpu_packs.empty()) {
                at::native::hot::fused_adamw_multi(
                    cpu_packs.data(), static_cast<int>(cpu_packs.size()),
                    static_cast<float>(lr),
                    static_cast<float>(beta1), static_cast<float>(beta2),
                    static_cast<float>(eps), static_cast<float>(wd),
                    static_cast<int>(step_val));
            }
        }
    }

    // Get options
    AdamWOptions& options() { return options_; }
    const AdamWOptions& options() const { return options_; }

protected:
    std::unique_ptr<OptimizerParamState> create_param_state() const override {
        return std::make_unique<AdamParamState>();
    }

private:
    AdamWOptions options_;
};

} // namespace optim
} // namespace torch
