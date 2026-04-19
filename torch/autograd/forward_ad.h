#pragma once

// ============================================================================
// forward_ad — Forward-mode automatic differentiation (JVP)
// ============================================================================
//
// Implements PyTorch-compatible API:
//   * DualLevel — RAII scope guard for a forward-AD level
//   * make_dual(primal, tangent) — attaches a tangent to a tensor
//   * unpack_dual(t) -> {primal, tangent}
//   * jvp(f, primal, tangent) — Jacobian-vector product
//
// IMPLEMENTATION:
//   Tangents are stored on a side-map keyed by TensorImpl*. When ops in this
//   namespace (add/sub/mul/div/mm/matmul/relu/sigmoid/tanh/exp/log/softmax/
//   linear) run, they query the side-map for input tangents, compute the
//   primal via at::native::*, then compute the output tangent via the op's
//   JVP rule and register it.
//
// LIMITATION (documented):
//   The user-supplied callable `f` passed to jvp() MUST use the wrapped ops
//   in `torch::autograd::forward_ad::` (e.g. forward_ad::mul) instead of
//   at::mul / torch::mul, because C++ has no monkey-patching. Plain at::*
//   calls inside f will compute correct primals but will NOT propagate the
//   tangent (the output gets a zero tangent).
//
// CPU-only. Float32 primary dtype. Compiles on Elbrus (LCC) and MSVC.
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/MathOps.h"
#include "aten/src/ATen/native/cpu/LinearAlgebra.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"
#include "aten/src/ATen/native/cpu/IndexOps.h"
#include "aten/src/ATen/native/cpu/ReduceOps.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "c10/macros/Macros.h"

#include <cmath>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace torch {
namespace autograd {
namespace forward_ad {

using at::Tensor;

// ============================================================================
// Side-map: TensorImpl* -> tangent Tensor, scoped to a stack of DualLevels
// ============================================================================
namespace detail {

struct TangentMap {
    // One map per active level. level 0 = bottom, top() = current.
    std::vector<std::unordered_map<c10::TensorImpl*, Tensor>> levels_;
    std::mutex mutex_;

    static TangentMap& instance() {
        static TangentMap inst;
        return inst;
    }

    int push() {
        std::lock_guard<std::mutex> g(mutex_);
        levels_.emplace_back();
        return static_cast<int>(levels_.size()) - 1;
    }

    void pop(int level) {
        std::lock_guard<std::mutex> g(mutex_);
        if (!levels_.empty() && static_cast<int>(levels_.size()) - 1 == level) {
            levels_.pop_back();
        }
    }

    int current_level() const {
        return static_cast<int>(levels_.size()) - 1;
    }

    bool has_active_level() const { return !levels_.empty(); }

    void set(c10::TensorImpl* impl, const Tensor& tangent) {
        if (!impl) return;
        std::lock_guard<std::mutex> g(mutex_);
        if (levels_.empty()) return;
        levels_.back()[impl] = tangent;
    }

    Tensor get(c10::TensorImpl* impl) {
        if (!impl) return Tensor();
        std::lock_guard<std::mutex> g(mutex_);
        if (levels_.empty()) return Tensor();
        auto& top = levels_.back();
        auto it = top.find(impl);
        return it == top.end() ? Tensor() : it->second;
    }

    bool has(c10::TensorImpl* impl) {
        if (!impl) return false;
        std::lock_guard<std::mutex> g(mutex_);
        if (levels_.empty()) return false;
        return levels_.back().count(impl) > 0;
    }
};

// Build a zero tangent matching the shape/dtype of t.
inline Tensor zero_tangent_like(const Tensor& t) {
    return at::zeros(t.sizes(), at::TensorOptions().dtype(t.dtype()).device(t.device()));
}

// Lookup a tangent or return a zero tensor of the right shape.
inline Tensor get_or_zero_tangent(const Tensor& t) {
    Tensor tg = TangentMap::instance().get(t.unsafeGetTensorImpl());
    if (tg.defined()) return tg;
    return zero_tangent_like(t);
}

// True iff at least one of the inputs has a tangent attached.
inline bool any_has_tangent(std::initializer_list<const Tensor*> ts) {
    auto& tm = TangentMap::instance();
    if (!tm.has_active_level()) return false;
    for (const Tensor* t : ts) {
        if (t && t->defined() && tm.has(t->unsafeGetTensorImpl())) return true;
    }
    return false;
}

// Register a tangent on out (only if a level is active and any input had one).
inline Tensor& attach_tangent(Tensor& out, const Tensor& tangent) {
    TangentMap::instance().set(out.unsafeGetTensorImpl(), tangent);
    return out;
}

} // namespace detail

// ============================================================================
// DualLevel — RAII scope for a forward-AD context
// ============================================================================
class DualLevel {
public:
    DualLevel() { level_ = detail::TangentMap::instance().push(); }
    ~DualLevel() { detail::TangentMap::instance().pop(level_); }
    int level() const { return level_; }

    DualLevel(const DualLevel&) = delete;
    DualLevel& operator=(const DualLevel&) = delete;
private:
    int level_;
};

// ============================================================================
// make_dual / unpack_dual
// ============================================================================
inline Tensor make_dual(const Tensor& primal, const Tensor& tangent) {
    PT_CHECK_MSG(primal.defined() && tangent.defined(),
                 "make_dual: primal and tangent must be defined");
    PT_CHECK_MSG(primal.sizes() == tangent.sizes(),
                 "make_dual: primal and tangent must have matching shapes");
    PT_CHECK_MSG(detail::TangentMap::instance().has_active_level(),
                 "make_dual: must be called inside a DualLevel scope");
    detail::TangentMap::instance().set(primal.unsafeGetTensorImpl(), tangent);
    return primal;
}

struct Unpacked {
    Tensor primal;
    Tensor tangent;
};

inline Unpacked unpack_dual(const Tensor& dual_tensor) {
    Tensor tg = detail::TangentMap::instance().get(dual_tensor.unsafeGetTensorImpl());
    if (!tg.defined()) tg = detail::zero_tangent_like(dual_tensor);
    return {dual_tensor, tg};
}

// ============================================================================
// JVP rules — wrapped ops
// ============================================================================
// Each op:
//   1. computes primal via at::native::*
//   2. if any input has a tangent (and a level is active), computes the
//      output tangent via the analytic JVP rule and attaches it.
//
// Rules:
//   add(x,y)     : t = tx + ty
//   sub(x,y)     : t = tx - ty
//   mul(x,y)     : t = tx*y + x*ty
//   div(x,y)     : t = (tx*y - x*ty) / (y*y)
//   mm(x,y)      : t = tx@y + x@ty
//   matmul(x,y)  : t = tx@y + x@ty
//   relu(x)      : t = tx * (x > 0)
//   sigmoid(x)   : s = sigmoid(x); t = tx * s * (1 - s)
//   tanh(x)      : th = tanh(x); t = tx * (1 - th*th)
//   exp(x)       : t = exp(x) * tx
//   log(x)       : t = tx / x
//   softmax(x,d) : s = softmax(x,d); t = s * (tx - sum(tx*s, dim=d, keepdim))
//   linear(x,W,b): primal = x@W^T (+ b); tangent = tx@W^T + x@tW^T
// ============================================================================

inline Tensor add(const Tensor& x, const Tensor& y) {
    Tensor out = at::native::add(x, y, /*alpha=*/1);
    if (detail::any_has_tangent({&x, &y})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor ty = detail::get_or_zero_tangent(y);
        Tensor tg = at::native::add(tx, ty, 1);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor sub(const Tensor& x, const Tensor& y) {
    Tensor out = at::native::sub(x, y, /*alpha=*/1);
    if (detail::any_has_tangent({&x, &y})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor ty = detail::get_or_zero_tangent(y);
        Tensor tg = at::native::sub(tx, ty, 1);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor mul(const Tensor& x, const Tensor& y) {
    Tensor out = at::native::mul(x, y);
    if (detail::any_has_tangent({&x, &y})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor ty = detail::get_or_zero_tangent(y);
        // d(x*y) = tx*y + x*ty
        Tensor tg = at::native::add(at::native::mul(tx, y),
                                    at::native::mul(x,  ty), 1);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor div(const Tensor& x, const Tensor& y) {
    Tensor out = at::native::div(x, y);
    if (detail::any_has_tangent({&x, &y})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor ty = detail::get_or_zero_tangent(y);
        // d(x/y) = (tx - out*ty) / y     (numerically equivalent, one fewer mul)
        Tensor num = at::native::sub(tx, at::native::mul(out, ty), 1);
        Tensor tg  = at::native::div(num, y);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor mm(const Tensor& x, const Tensor& y) {
    Tensor out = at::native::mm(x, y);
    if (detail::any_has_tangent({&x, &y})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor ty = detail::get_or_zero_tangent(y);
        Tensor tg = at::native::add(at::native::mm(tx, y),
                                    at::native::mm(x,  ty), 1);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor matmul(const Tensor& x, const Tensor& y) {
    Tensor out = at::native::matmul(x, y);
    if (detail::any_has_tangent({&x, &y})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor ty = detail::get_or_zero_tangent(y);
        Tensor tg = at::native::add(at::native::matmul(tx, y),
                                    at::native::matmul(x,  ty), 1);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor relu(const Tensor& x) {
    Tensor out = at::native::relu(x);
    if (detail::any_has_tangent({&x})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        // mask = (x > 0) as float
        Tensor zero = at::zeros(x.sizes(),
            at::TensorOptions().dtype(x.dtype()).device(x.device()));
        Tensor cond = at::native::gt(x, at::Scalar(0.0f));
        Tensor one  = at::ones(x.sizes(),
            at::TensorOptions().dtype(x.dtype()).device(x.device()));
        Tensor mask = at::native::where(cond, one, zero);
        Tensor tg = at::native::mul(tx, mask);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor sigmoid(const Tensor& x) {
    Tensor out = at::native::sigmoid(x);
    if (detail::any_has_tangent({&x})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        // d(sigmoid) = s*(1-s)
        Tensor one_minus = at::native::sub(
            at::ones(x.sizes(), at::TensorOptions().dtype(x.dtype()).device(x.device())),
            out, 1);
        Tensor deriv = at::native::mul(out, one_minus);
        Tensor tg    = at::native::mul(tx, deriv);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor tanh(const Tensor& x) {
    Tensor out = at::native::tanh(x);
    if (detail::any_has_tangent({&x})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        // d(tanh) = 1 - tanh^2
        Tensor sq = at::native::mul(out, out);
        Tensor deriv = at::native::sub(
            at::ones(x.sizes(), at::TensorOptions().dtype(x.dtype()).device(x.device())),
            sq, 1);
        Tensor tg = at::native::mul(tx, deriv);
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor exp(const Tensor& x) {
    Tensor out = at::native::exp(x);
    if (detail::any_has_tangent({&x})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor tg = at::native::mul(out, tx);  // d(exp(x)) = exp(x)*tx
        detail::attach_tangent(out, tg);
    }
    return out;
}

inline Tensor log(const Tensor& x) {
    Tensor out = at::native::log(x);
    if (detail::any_has_tangent({&x})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor tg = at::native::div(tx, x);    // d(log(x)) = tx / x
        detail::attach_tangent(out, tg);
    }
    return out;
}

// linear: y = x @ W^T (+ b).  W is [out, in], x is [..., in], b is [out].
// JVP: ty = tx @ W^T + x @ tW^T (tangent of bias adds directly).
inline Tensor linear(const Tensor& x, const Tensor& W, const Tensor& b = Tensor()) {
    Tensor Wt = at::native::transpose(W, 0, 1);
    Tensor out = at::native::matmul(x, Wt);
    if (b.defined()) out = at::native::add(out, b, 1);

    if (detail::any_has_tangent({&x, &W, &b})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor tW = detail::get_or_zero_tangent(W);
        Tensor tWt = at::native::transpose(tW, 0, 1);
        Tensor tg = at::native::add(at::native::matmul(tx, Wt),
                                    at::native::matmul(x,  tWt), 1);
        if (b.defined()) {
            Tensor tb = detail::get_or_zero_tangent(b);
            tg = at::native::add(tg, tb, 1);
        }
        detail::attach_tangent(out, tg);
    }
    return out;
}

// softmax along `dim`. y_i = exp(x_i) / sum(exp(x)). JVP:
//   ty = y * (tx - sum(tx * y, dim, keepdim))
inline Tensor softmax(const Tensor& x, int64_t dim) {
    // primal — manual implementation (CPU, float32). Avoids dependency on
    // a non-existent at::native::softmax wrapper.
    auto compute_softmax = [](const Tensor& in, int64_t d) -> Tensor {
        Tensor xc = in.is_contiguous() ? in : in.contiguous();
        // shift for numerical stability: x - max(x, dim, keepdim)
        // at::native::max(t, dim, keepdim) returns (values, indices) tuple.
        auto mxr = at::native::max(xc, d, /*keepdim=*/true);
        Tensor mx = std::get<0>(mxr);
        Tensor shifted = at::native::sub(xc, mx, 1);
        Tensor ex = at::native::exp(shifted);
        Tensor sm = at::native::sum(ex, d, /*keepdim=*/true);
        return at::native::div(ex, sm);
    };
    Tensor out = compute_softmax(x, dim);
    if (detail::any_has_tangent({&x})) {
        Tensor tx = detail::get_or_zero_tangent(x);
        Tensor ty_dot = at::native::sum(at::native::mul(tx, out), dim, /*keepdim=*/true);
        Tensor tg = at::native::mul(out, at::native::sub(tx, ty_dot, 1));
        detail::attach_tangent(out, tg);
    }
    return out;
}

// ============================================================================
// jvp — top-level Jacobian-vector product
// ============================================================================
//
// Runs f(primal) inside a fresh DualLevel with the supplied tangent attached
// to the primal input. f MUST use forward_ad::* wrapped ops to propagate the
// tangent (see file-top LIMITATION note). Returns the output tangent (df/dx . v).
// ============================================================================
inline Tensor jvp(std::function<Tensor(Tensor)> f,
                  const Tensor& primal,
                  const Tensor& tangent) {
    PT_CHECK_MSG(primal.defined() && tangent.defined(),
                 "jvp: primal and tangent must be defined");
    PT_CHECK_MSG(primal.sizes() == tangent.sizes(),
                 "jvp: primal and tangent must have matching shapes");

    // Disable reverse-mode recording while running forward-AD — we only want
    // primal values for the JVP rules, no reverse graph.
    NoGradGuard ng;

    DualLevel lvl;
    Tensor x = make_dual(primal, tangent);
    Tensor y = f(x);
    auto unpacked = unpack_dual(y);
    return unpacked.tangent;  // == df/dx . v
}

// ============================================================================
// Self-test  (compiled out unless PT_FORWARD_AD_SELFTEST is defined)
// ============================================================================
#ifdef PT_FORWARD_AD_SELFTEST
#include <cstdio>
#include <cstdlib>
inline int forward_ad_selftest() {
    using namespace at;
    // f(x) = x * x = x^2 ;  df/dx = 2x ; at x=3 -> 6
    float xv = 3.0f, vv = 1.0f;
    Tensor x = at::full({1}, Scalar(xv), TensorOptions().dtype(c10::ScalarType::Float));
    Tensor v = at::full({1}, Scalar(vv), TensorOptions().dtype(c10::ScalarType::Float));

    Tensor j = jvp([](Tensor in) { return forward_ad::mul(in, in); }, x, v);
    float got = j.data_ptr<float>()[0];
    float want = 2.0f * xv;
    std::printf("forward_ad selftest: d(x^2)/dx at x=%.1f -> got=%.4f want=%.4f\n",
                xv, got, want);
    if (std::fabs(got - want) > 1e-4f) {
        std::fprintf(stderr, "forward_ad selftest FAILED\n");
        std::abort();
    }

    // Bonus: chain rule check —  f(x) = exp(2x);  df/dx = 2*exp(2x) ; v=1
    Tensor x2 = at::full({1}, Scalar(0.5f), TensorOptions().dtype(c10::ScalarType::Float));
    Tensor v2 = at::full({1}, Scalar(1.0f), TensorOptions().dtype(c10::ScalarType::Float));
    Tensor two = at::full({1}, Scalar(2.0f), TensorOptions().dtype(c10::ScalarType::Float));
    Tensor j2 = jvp([&two](Tensor in){ return forward_ad::exp(forward_ad::mul(in, two)); },
                    x2, v2);
    float got2  = j2.data_ptr<float>()[0];
    float want2 = 2.0f * std::exp(2.0f * 0.5f);
    std::printf("forward_ad selftest: d(exp(2x))/dx at x=0.5 -> got=%.4f want=%.4f\n",
                got2, want2);
    if (std::fabs(got2 - want2) > 1e-3f) {
        std::fprintf(stderr, "forward_ad selftest exp(2x) FAILED\n");
        std::abort();
    }
    return 0;
}
#endif

}}}  // namespace torch::autograd::forward_ad
