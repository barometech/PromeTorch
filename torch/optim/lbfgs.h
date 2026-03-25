#pragma once

#include "torch/optim/optimizer.h"
#include <cmath>
#include <deque>
#include <functional>
#include <algorithm>

namespace torch {
namespace optim {

// ============================================================================
// LBFGSOptions - Options for L-BFGS optimizer
// ============================================================================

struct LBFGSOptions {
    double lr = 1.0;                  // Learning rate (step size)
    int max_iter = 20;                // Max iterations per step()
    int max_eval = 25;                // Max function evaluations per step() (0 = max_iter * 5/4)
    double tolerance_grad = 1e-7;     // Termination tolerance on gradient norm
    double tolerance_change = 1e-9;   // Termination tolerance on function/x change
    int history_size = 100;           // Number of (s, y) pairs to keep
    // Line search parameters (Strong Wolfe)
    double c1 = 1e-4;                // Sufficient decrease (Armijo)
    double c2 = 0.9;                 // Curvature condition
    int max_ls = 25;                 // Max line search iterations

    LBFGSOptions(double lr_ = 1.0) : lr(lr_) {}

    LBFGSOptions& learning_rate(double lr_) { lr = lr_; return *this; }
    LBFGSOptions& max_iter_(int m) { max_iter = m; return *this; }
    LBFGSOptions& max_eval_(int m) { max_eval = m; return *this; }
    LBFGSOptions& tolerance_grad_(double t) { tolerance_grad = t; return *this; }
    LBFGSOptions& tolerance_change_(double t) { tolerance_change = t; return *this; }
    LBFGSOptions& history_size_(int h) { history_size = h; return *this; }
};

// ============================================================================
// LBFGS - Limited-memory Broyden-Fletcher-Goldfarb-Shanno Optimizer
// ============================================================================
// Implements L-BFGS with Strong Wolfe line search.
//
// Unlike SGD/Adam, step() requires a closure that recomputes the loss:
//   auto closure = [&]() -> float {
//       optimizer.zero_grad();
//       auto loss = model.forward(input);
//       loss.backward();
//       return loss.item();
//   };
//   optimizer.step(closure);
//
// L-BFGS approximates the inverse Hessian using a history of parameter
// and gradient differences (s and y vectors), then uses two-loop recursion
// to compute H*g efficiently without storing the full matrix.

class LBFGS : public Optimizer {
public:
    using ClosureFn = std::function<float()>;

    LBFGS(std::vector<Parameter*> params, LBFGSOptions options = LBFGSOptions())
        : Optimizer(std::move(params), options.lr), options_(options) {}

    LBFGS(std::vector<Parameter*> params, double lr)
        : LBFGS(std::move(params), LBFGSOptions(lr)) {}

    // L-BFGS requires a closure; this no-arg step() throws
    void step() override {
        throw std::runtime_error("LBFGS requires a closure. Use step(closure) instead.");
    }

    // Main L-BFGS step with closure
    float step(ClosureFn closure) {
        if (!closure) {
            throw std::runtime_error("LBFGS::step requires a non-null closure");
        }

        int max_eval = options_.max_eval > 0 ? options_.max_eval : (options_.max_iter * 5 / 4);
        int n_iter = 0;
        int n_eval = 0;

        // Gather all parameters into a flat vector for L-BFGS
        auto params = all_params_with_grad();
        int64_t total_numel = 0;
        for (auto* p : params) total_numel += p->numel();

        if (total_numel == 0) return 0.0f;

        // Evaluate initial loss and gradient
        float loss = closure();
        n_eval++;

        // Flatten current gradient
        std::vector<float> grad_flat = flatten_grad(params, total_numel);

        float grad_norm = vec_norm(grad_flat);
        if (grad_norm <= options_.tolerance_grad) {
            return loss;
        }

        // Flatten current parameters
        std::vector<float> x_flat = flatten_params(params, total_numel);

        // Previous values for convergence check
        float prev_loss = loss;
        std::vector<float> prev_x = x_flat;
        std::vector<float> prev_grad = grad_flat;

        // L-BFGS history
        std::deque<std::vector<float>> s_history; // s_k = x_{k+1} - x_k
        std::deque<std::vector<float>> y_history; // y_k = g_{k+1} - g_k
        std::deque<float> rho_history;            // 1 / (y_k^T s_k)

        // Initial Hessian scaling
        float H0_scale = 1.0f;

        for (n_iter = 0; n_iter < options_.max_iter; ++n_iter) {
            // ================================================================
            // Two-loop recursion: compute search direction d = -H * g
            // ================================================================
            std::vector<float> q = grad_flat;  // start with current gradient
            int hist_len = static_cast<int>(s_history.size());
            std::vector<float> alpha_hist(hist_len);

            // First loop (backward through history)
            for (int i = hist_len - 1; i >= 0; --i) {
                alpha_hist[i] = rho_history[i] * vec_dot(s_history[i], q);
                vec_axpy(q, y_history[i], -alpha_hist[i]);
            }

            // Apply initial Hessian approximation: r = H0 * q
            std::vector<float> r(total_numel);
            for (int64_t j = 0; j < total_numel; ++j) {
                r[j] = H0_scale * q[j];
            }

            // Second loop (forward through history)
            for (int i = 0; i < hist_len; ++i) {
                float beta = rho_history[i] * vec_dot(y_history[i], r);
                vec_axpy(r, s_history[i], alpha_hist[i] - beta);
            }

            // d = -r (descent direction)
            std::vector<float> d(total_numel);
            for (int64_t j = 0; j < total_numel; ++j) {
                d[j] = -r[j];
            }

            // Check if d is a descent direction
            float dg = vec_dot(d, grad_flat);
            if (dg > 0) {
                // Not a descent direction — reset to steepest descent
                for (int64_t j = 0; j < total_numel; ++j) {
                    d[j] = -grad_flat[j];
                }
                dg = vec_dot(d, grad_flat);
                s_history.clear();
                y_history.clear();
                rho_history.clear();
                H0_scale = 1.0f;
            }

            // ================================================================
            // Strong Wolfe line search
            // ================================================================
            float step_size = (n_iter == 0 && s_history.empty())
                ? std::min(1.0f, 1.0f / grad_norm)
                : 1.0f;
            step_size *= static_cast<float>(options_.lr);

            // Save state before line search
            prev_x = x_flat;
            prev_grad = grad_flat;
            prev_loss = loss;

            float new_loss = loss;
            bool ls_success = strong_wolfe_line_search(
                closure, params, total_numel,
                x_flat, grad_flat, d, step_size,
                loss, dg, new_loss, n_eval, max_eval);

            loss = new_loss;

            // ================================================================
            // Update history
            // ================================================================
            // s = x_new - x_old, y = g_new - g_old
            std::vector<float> s_k(total_numel);
            std::vector<float> y_k(total_numel);
            for (int64_t j = 0; j < total_numel; ++j) {
                s_k[j] = x_flat[j] - prev_x[j];
                y_k[j] = grad_flat[j] - prev_grad[j];
            }

            float ys = vec_dot(y_k, s_k);
            if (ys > 1e-10f) {
                // Valid curvature — update history
                if (static_cast<int>(s_history.size()) >= options_.history_size) {
                    s_history.pop_front();
                    y_history.pop_front();
                    rho_history.pop_front();
                }
                s_history.push_back(std::move(s_k));
                y_history.push_back(std::move(y_k));
                rho_history.push_back(1.0f / ys);

                // Update H0 scaling: H0 = (y^T s) / (y^T y)
                float yy = vec_dot(y_history.back(), y_history.back());
                if (yy > 1e-10f) {
                    H0_scale = ys / yy;
                }
            }

            // ================================================================
            // Convergence checks
            // ================================================================
            grad_norm = vec_norm(grad_flat);
            if (grad_norm <= options_.tolerance_grad) break;

            float x_change = 0.0f;
            for (int64_t j = 0; j < total_numel; ++j) {
                float diff = x_flat[j] - prev_x[j];
                x_change += diff * diff;
            }
            x_change = std::sqrt(x_change);
            if (x_change <= options_.tolerance_change) break;

            float loss_change = std::abs(loss - prev_loss);
            if (loss_change <= options_.tolerance_change) break;

            if (n_eval >= max_eval) break;
        }

        return loss;
    }

    LBFGSOptions& options() { return options_; }
    const LBFGSOptions& options() const { return options_; }

private:
    LBFGSOptions options_;

    // ========================================================================
    // Helper: get parameters that have gradients
    // ========================================================================
    std::vector<Parameter*> all_params_with_grad() {
        std::vector<Parameter*> result;
        for (auto& group : param_groups_) {
            for (auto* p : group.params) {
                if (p->defined() && p->grad().defined()) {
                    result.push_back(p);
                }
            }
        }
        return result;
    }

    // ========================================================================
    // Flatten/unflatten parameter and gradient vectors
    // ========================================================================
    static std::vector<float> flatten_params(const std::vector<Parameter*>& params, int64_t n) {
        std::vector<float> flat(n);
        int64_t offset = 0;
        for (auto* p : params) {
            Tensor t = p->data().contiguous();
            const float* src = t.data_ptr<float>();
            int64_t numel = p->numel();
            std::copy(src, src + numel, flat.data() + offset);
            offset += numel;
        }
        return flat;
    }

    static std::vector<float> flatten_grad(const std::vector<Parameter*>& params, int64_t n) {
        std::vector<float> flat(n);
        int64_t offset = 0;
        for (auto* p : params) {
            Tensor g = p->grad().contiguous();
            const float* src = g.data_ptr<float>();
            int64_t numel = p->numel();
            std::copy(src, src + numel, flat.data() + offset);
            offset += numel;
        }
        return flat;
    }

    static void unflatten_to_params(const std::vector<Parameter*>& params,
                                     const std::vector<float>& flat) {
        int64_t offset = 0;
        for (auto* p : params) {
            Tensor t = p->data().contiguous();
            float* dst = t.mutable_data_ptr<float>();
            int64_t numel = p->numel();
            std::copy(flat.data() + offset, flat.data() + offset + numel, dst);
            // If param was non-contiguous, copy back
            if (!p->data().is_contiguous()) {
                p->data().copy_(t);
            }
            offset += numel;
        }
    }

    // ========================================================================
    // Vector operations on flat std::vector<float>
    // ========================================================================
    static float vec_dot(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    }

    static float vec_norm(const std::vector<float>& a) {
        return std::sqrt(vec_dot(a, a));
    }

    // a += alpha * b
    static void vec_axpy(std::vector<float>& a, const std::vector<float>& b, float alpha) {
        for (size_t i = 0; i < a.size(); ++i) a[i] += alpha * b[i];
    }

    // ========================================================================
    // Strong Wolfe line search
    // ========================================================================
    // Finds step_size satisfying:
    //   f(x + t*d) <= f(x) + c1 * t * g^T d    (sufficient decrease)
    //   |g_new^T d| <= c2 * |g^T d|             (curvature condition)

    bool strong_wolfe_line_search(
        ClosureFn& closure,
        std::vector<Parameter*>& params,
        int64_t total_numel,
        std::vector<float>& x_flat,        // updated in-place to x + t*d
        std::vector<float>& grad_flat,     // updated in-place to new gradient
        const std::vector<float>& d,       // search direction
        float initial_step,
        float f0,                          // f(x)
        float dg0,                         // g^T d at x
        float& f_new,                      // output: f(x + t*d)
        int& n_eval,
        int max_eval)
    {
        float c1 = static_cast<float>(options_.c1);
        float c2 = static_cast<float>(options_.c2);
        int max_ls = options_.max_ls;

        float t = initial_step;
        float t_prev = 0.0f;
        float f_prev = f0;
        float dg_prev = dg0;

        // Save original x
        std::vector<float> x0 = x_flat;
        bool first_iter = true;

        for (int ls = 0; ls < max_ls && n_eval < max_eval; ++ls) {
            // Set parameters to x0 + t * d
            for (int64_t j = 0; j < total_numel; ++j) {
                x_flat[j] = x0[j] + t * d[j];
            }
            unflatten_to_params(params, x_flat);

            // Evaluate
            f_new = closure();
            n_eval++;
            grad_flat = flatten_grad(params, total_numel);

            float dg_new = vec_dot(grad_flat, d);

            // Check Armijo (sufficient decrease)
            if (f_new > f0 + c1 * t * dg0 || (!first_iter && f_new >= f_prev)) {
                // Zoom between t_prev and t
                return zoom(closure, params, total_numel, x0, x_flat, grad_flat, d,
                           t_prev, t, f0, dg0, f_prev, f_new, dg_prev, dg_new,
                           f_new, n_eval, max_eval, c1, c2);
            }

            // Check curvature condition (strong Wolfe)
            if (std::abs(dg_new) <= c2 * std::abs(dg0)) {
                // Both conditions satisfied
                return true;
            }

            if (dg_new >= 0) {
                // Zoom between t and t_prev
                return zoom(closure, params, total_numel, x0, x_flat, grad_flat, d,
                           t, t_prev, f0, dg0, f_new, f_prev, dg_new, dg_prev,
                           f_new, n_eval, max_eval, c1, c2);
            }

            // Expand step
            t_prev = t;
            f_prev = f_new;
            dg_prev = dg_new;
            t *= 2.0f;
            first_iter = false;
        }

        // Max line search iterations reached — accept current point
        return false;
    }

    // Zoom phase of Strong Wolfe line search (bisection between lo and hi)
    bool zoom(
        ClosureFn& closure,
        std::vector<Parameter*>& params,
        int64_t total_numel,
        const std::vector<float>& x0,
        std::vector<float>& x_flat,
        std::vector<float>& grad_flat,
        const std::vector<float>& d,
        float t_lo, float t_hi,
        float f0, float dg0,
        float f_lo, float f_hi,
        float dg_lo, float dg_hi,
        float& f_new,
        int& n_eval, int max_eval,
        float c1, float c2)
    {
        for (int z = 0; z < 10 && n_eval < max_eval; ++z) {
            // Bisection (cubic interpolation is better but bisection is robust)
            float t = 0.5f * (t_lo + t_hi);

            for (int64_t j = 0; j < total_numel; ++j) {
                x_flat[j] = x0[j] + t * d[j];
            }
            unflatten_to_params(params, x_flat);

            f_new = closure();
            n_eval++;
            grad_flat = flatten_grad(params, total_numel);
            float dg_new = vec_dot(grad_flat, d);

            if (f_new > f0 + c1 * t * dg0 || f_new >= f_lo) {
                t_hi = t;
                f_hi = f_new;
                dg_hi = dg_new;
            } else {
                if (std::abs(dg_new) <= c2 * std::abs(dg0)) {
                    return true; // Found it
                }
                if (dg_new * (t_hi - t_lo) >= 0) {
                    t_hi = t_lo;
                    f_hi = f_lo;
                    dg_hi = dg_lo;
                }
                t_lo = t;
                f_lo = f_new;
                dg_lo = dg_new;
            }

            if (std::abs(t_hi - t_lo) < 1e-12f) break;
        }
        return false;
    }
};

} // namespace optim
} // namespace torch
