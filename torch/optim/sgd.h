#pragma once

#include "torch/optim/optimizer.h"
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace torch {
namespace optim {

// ============================================================================
// SGDOptions - Options for SGD optimizer
// ============================================================================

struct SGDOptions {
    double lr = 0.01;           // Learning rate
    double momentum = 0.0;       // Momentum factor
    double dampening = 0.0;      // Dampening for momentum
    double weight_decay = 0.0;   // Weight decay (L2 penalty)
    bool nesterov = false;       // Enables Nesterov momentum

    SGDOptions(double lr_ = 0.01) : lr(lr_) {}

    SGDOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    SGDOptions& momentum_(double m) { momentum = m; return *this; }
    SGDOptions& dampening_(double d) { dampening = d; return *this; }
    SGDOptions& weight_decay_(double wd) { weight_decay = wd; return *this; }
    SGDOptions& nesterov_(bool n) { nesterov = n; return *this; }
};

// ============================================================================
// SGDState - Per-parameter state for SGD
// ============================================================================

struct SGDParamState : public OptimizerParamState {
    Tensor momentum_buffer;  // Momentum buffer (velocity)
};

// ============================================================================
// SGD - Stochastic Gradient Descent Optimizer
// ============================================================================
// Implements SGD with optional momentum, weight decay, and Nesterov momentum.
//
// Algorithm:
//   v_t = momentum * v_{t-1} + (1 - dampening) * g_t
//   if nesterov:
//       p_t = p_{t-1} - lr * (g_t + momentum * v_t)
//   else:
//       p_t = p_{t-1} - lr * v_t
//
// Where:
//   - g_t is the gradient (with optional weight decay: g_t = grad + wd * param)
//   - v_t is the velocity (momentum buffer)
//   - p_t is the parameter

class SGD : public Optimizer {
public:
    SGD(std::vector<Parameter*> params, SGDOptions options = SGDOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    SGD(std::vector<Parameter*> params, double lr)
        : SGD(std::move(params), SGDOptions(lr)) {}

    SGD(std::vector<ParamGroup> param_groups, SGDOptions options = SGDOptions())
        : Optimizer(std::move(param_groups)), options_(options) {}

    void step() override {
        for (auto& group : param_groups_) {
            double lr = group.lr > 0 ? group.lr : options_.lr;
            double wd = group.weight_decay > 0 ? group.weight_decay : options_.weight_decay;
            double momentum = options_.momentum;
            double dampening = options_.dampening;
            bool nesterov = options_.nesterov;

            for (auto* param : group.params) {
                if (!param->defined()) continue;

                Tensor grad = param->grad();
                if (!grad.defined()) continue;

                // ================================================================
                // Fast path: no momentum, CPU, float — fused AVX2
                // ================================================================
                if (momentum == 0.0 && param->data().dtype() == c10::ScalarType::Float &&
                    !param->data().is_cuda()) {
                    Tensor grad_c = grad.contiguous();
                    float* p_data = param->data().mutable_data_ptr<float>();
                    const float* g_data = grad_c.data_ptr<float>();
                    int64_t n = param->numel();
                    float lrf = static_cast<float>(lr);
                    float wdf = static_cast<float>(wd);

                    if (wdf != 0.0f) {
                        // param = param * (1 - lr*wd) - lr * grad
                        __m256 decay = _mm256_set1_ps(1.0f - lrf * wdf);
                        __m256 neg_lr = _mm256_set1_ps(-lrf);
                        int64_t i = 0;
                        for (; i + 8 <= n; i += 8) {
                            __m256 p = _mm256_loadu_ps(p_data + i);
                            __m256 g = _mm256_loadu_ps(g_data + i);
                            p = _mm256_fmadd_ps(neg_lr, g, _mm256_mul_ps(p, decay));
                            _mm256_storeu_ps(p_data + i, p);
                        }
                        for (; i < n; ++i) {
                            p_data[i] = p_data[i] * (1.0f - lrf * wdf) - lrf * g_data[i];
                        }
                    } else {
                        // param -= lr * grad
                        __m256 neg_lr = _mm256_set1_ps(-lrf);
                        int64_t i = 0;
                        for (; i + 8 <= n; i += 8) {
                            __m256 p = _mm256_loadu_ps(p_data + i);
                            __m256 g = _mm256_loadu_ps(g_data + i);
                            p = _mm256_fmadd_ps(neg_lr, g, p);
                            _mm256_storeu_ps(p_data + i, p);
                        }
                        for (; i < n; ++i) {
                            p_data[i] -= lrf * g_data[i];
                        }
                    }
                    continue;
                }

                // ================================================================
                // General path: momentum, nesterov, non-float
                // ================================================================
                // Apply weight decay
                if (wd != 0.0) {
                    grad = grad.add(param->data(), at::Scalar(wd));
                }

                if (momentum != 0.0) {
                    auto* state = get_or_create_state<SGDParamState>(param);

                    if (!state->momentum_buffer.defined()) {
                        state->momentum_buffer = grad.clone();
                    } else {
                        state->momentum_buffer.mul_(at::Scalar(momentum));
                        state->momentum_buffer.add_(grad, at::Scalar(1.0 - dampening));
                    }

                    if (nesterov) {
                        grad = grad.add(state->momentum_buffer, at::Scalar(momentum));
                    } else {
                        grad = state->momentum_buffer;
                    }
                }

                // Update parameters: param = param - lr * grad
                param->data().sub_(grad, at::Scalar(lr));
            }
        }
    }

    // Get options
    SGDOptions& options() { return options_; }
    const SGDOptions& options() const { return options_; }

private:
    SGDOptions options_;
};

} // namespace optim
} // namespace torch
