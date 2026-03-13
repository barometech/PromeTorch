#pragma once

// ============================================================================
// Autograd - Automatic Differentiation System
// ============================================================================
// This is the main header for the autograd system. It provides:
// - Computational graph (Node, Edge)
// - Gradient metadata (AutogradMeta)
// - Backward engine (Engine)
// - Backward functions for all differentiable operations

// Core components
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "torch/csrc/autograd/engine.h"

// For graph cleanup
#include <queue>
#include <unordered_set>

// Backward functions
#include "torch/csrc/autograd/functions/MathBackward.h"
#include "torch/csrc/autograd/functions/ReduceBackward.h"
#include "torch/csrc/autograd/functions/LinearAlgebraBackward.h"
#include "torch/csrc/autograd/functions/ShapeBackward.h"
#include "torch/csrc/autograd/functions/IndexBackward.h"

namespace torch {
namespace autograd {

using at::Tensor;
using at::Scalar;

// GradMode, NoGradGuard, EnableGradGuard are defined in grad_mode.h
// and included via node.h

// Helper function to check if grad tracking should be enabled for an operation
// (convenience wrappers that mirror compute_requires_grad)
inline bool should_compute_grad(const Tensor& t) {
    return compute_requires_grad(t);
}

inline bool should_compute_grad(const Tensor& a, const Tensor& b) {
    return compute_requires_grad(a, b);
}

// ============================================================================
// Implementations deferred from headers (to avoid circular dependencies)
// ============================================================================

// Implementation of Node::add_input_metadata
inline void Node::add_input_metadata(const Tensor& tensor) {
    if (tensor.defined() && tensor.requires_grad()) {
        add_next_edge(gradient_edge(tensor));
    } else {
        add_next_edge(Edge());
    }
}

// Implementation of AccumulateGrad::apply
inline variable_list AccumulateGrad::apply(variable_list&& grads) {
    auto impl_ptr = weak_impl_.lock();
    if (!impl_ptr) {
        return {};
    }

    auto& grad = grads[0];
    if (!grad.defined()) {
        return {};
    }

    // Get autograd meta (base class c10::AutogradMeta)
    auto* raw_meta = impl_ptr->autograd_meta();
    if (!raw_meta) {
        return {};
    }

    // CRITICAL: Make gradient contiguous before storing!
    // Backward ops like TBackward return transposed views (non-contiguous).
    // CUDA element-wise ops (add, sub, mul_scalar) read data_ptr sequentially
    // and ignore strides, so non-contiguous gradients produce wrong results
    // in gradient accumulation and SGD updates.
    Tensor grad_contig = grad.is_contiguous() ? grad : grad.contiguous();

    // Accumulate gradient (using base class grad_ field)
    if (!raw_meta->grad_) {
        // First gradient - just copy (contiguous)
        raw_meta->grad_ = grad_contig.getIntrusivePtr();
    } else {
        // Accumulate: grad_ = grad_ + grad
        Tensor existing_grad(raw_meta->grad_);
        Tensor accumulated = existing_grad.add(grad_contig);
        raw_meta->grad_ = accumulated.getIntrusivePtr();
    }

    // Call hooks only if we have AutogradMetaImpl (hooks_ is not in base class)
    auto* meta_impl = dynamic_cast<AutogradMetaImpl*>(raw_meta);
    if (meta_impl && !meta_impl->hooks_.empty()) {
        Tensor grad_tensor(raw_meta->grad_);
        for (auto& hook : meta_impl->hooks_) {
            hook(grad_tensor);
        }
    }

    return {};
}

// Debug counter for weak_ptr failures
inline std::atomic<int64_t> g_weak_ptr_hit{0};
inline std::atomic<int64_t> g_weak_ptr_miss{0};
inline std::atomic<int64_t> g_weak_ptr_upgrade{0};

inline void print_weak_ptr_stats() {
    std::cout << "[WEAK PTR] hit=" << g_weak_ptr_hit.load()
              << " miss=" << g_weak_ptr_miss.load()
              << " upgrade=" << g_weak_ptr_upgrade.load()
              << std::endl;
}

// Implementation of get_grad_accumulator
inline std::shared_ptr<Node> get_grad_accumulator(const Tensor& tensor) {
    if (!tensor.requires_grad() || !tensor.is_leaf()) {
        return nullptr;
    }

    // We need AutogradMetaImpl to store the grad_accumulator
    // Use const_cast since we're only upgrading internal metadata
    Tensor& mutable_tensor = const_cast<Tensor&>(tensor);

    // Check if we need to upgrade metadata
    auto* raw_meta = tensor.autograd_meta();
    bool was_base = raw_meta && !dynamic_cast<AutogradMetaImpl*>(raw_meta);

    auto* meta = ensure_autograd_meta_impl(mutable_tensor);

    if (was_base) {
        g_weak_ptr_upgrade++;
    }

    // Check if we already have an accumulator
    auto accumulator = meta->grad_accumulator_.lock();
    if (accumulator) {
        g_weak_ptr_hit++;
        return accumulator;
    }

    g_weak_ptr_miss++;

    // Create a new accumulator
    accumulator = std::make_shared<AccumulateGrad>(tensor);
    meta->grad_accumulator_ = accumulator;
    return accumulator;
}

// ============================================================================
// Autograd Context for Forward Operations
// ============================================================================
// This macro makes it easy to create autograd-aware forward operations

#define AUTOGRAD_UNARY_OP(name, backward_class, save_input, save_result)      \
    inline Tensor name##_autograd(const Tensor& self) {                       \
        Tensor result = at::native::name(self);                               \
        if (compute_requires_grad(self)) {                                    \
            auto grad_fn = std::make_shared<backward_class>(                  \
                save_input ? self : Tensor(),                                 \
                save_result ? result : Tensor()                               \
            );                                                                \
            grad_fn->add_input_metadata(self);                                \
            set_grad_fn(result, grad_fn);                                     \
            result.set_requires_grad(true);                                   \
        }                                                                     \
        return result;                                                        \
    }

// ============================================================================
// Autograd-Aware Operations
// ============================================================================
// These wrap the native operations and set up the computational graph

namespace {

// Helper to create result with grad_fn
template<typename BackwardT, typename... SavedArgs>
Tensor make_result_with_grad(
    Tensor result,
    const Tensor& input,
    SavedArgs&&... saved
) {
    if (compute_requires_grad(input)) {
        auto grad_fn = std::make_shared<BackwardT>(std::forward<SavedArgs>(saved)...);
        grad_fn->add_input_metadata(input);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

template<typename BackwardT, typename... SavedArgs>
Tensor make_result_with_grad2(
    Tensor result,
    const Tensor& input1,
    const Tensor& input2,
    SavedArgs&&... saved
) {
    if (compute_requires_grad(input1, input2)) {
        auto grad_fn = std::make_shared<BackwardT>(std::forward<SavedArgs>(saved)...);
        grad_fn->add_input_metadata(input1);
        grad_fn->add_input_metadata(input2);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

} // anonymous namespace

// ============================================================================
// Unary Operations with Autograd
// ============================================================================

// Unary ops use Tensor methods which have CUDA dispatch
inline Tensor neg_autograd(const Tensor& self) {
    Tensor result = self.neg();  // Has CUDA dispatch
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<NegBackward>();
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor abs_autograd(const Tensor& self) {
    Tensor result = self.abs();  // Has CUDA dispatch
    return make_result_with_grad<AbsBackward>(result, self, self);
}

inline Tensor sqrt_autograd(const Tensor& self) {
    Tensor result = self.sqrt();  // Has CUDA dispatch
    return make_result_with_grad<SqrtBackward>(result, self, result);
}

inline Tensor exp_autograd(const Tensor& self) {
    Tensor result = self.exp();  // Has CUDA dispatch
    return make_result_with_grad<ExpBackward>(result, self, result);
}

inline Tensor log_autograd(const Tensor& self) {
    Tensor result = self.log();  // Has CUDA dispatch
    return make_result_with_grad<LogBackward>(result, self, self);
}

inline Tensor sin_autograd(const Tensor& self) {
    Tensor result = at::native::sin(self);  // No CUDA dispatch yet
    return make_result_with_grad<SinBackward>(result, self, self);
}

inline Tensor cos_autograd(const Tensor& self) {
    Tensor result = at::native::cos(self);  // No CUDA dispatch yet
    return make_result_with_grad<CosBackward>(result, self, self);
}

inline Tensor tanh_autograd(const Tensor& self) {
    Tensor result = self.tanh();  // Has CUDA dispatch
    return make_result_with_grad<TanhBackward>(result, self, result);
}

inline Tensor sigmoid_autograd(const Tensor& self) {
    Tensor result = self.sigmoid();  // Has CUDA dispatch
    return make_result_with_grad<SigmoidBackward>(result, self, result);
}

inline Tensor relu_autograd(const Tensor& self) {
    Tensor result = self.relu();  // Has CUDA dispatch
    return make_result_with_grad<ReluBackward>(result, self, self);
}

inline Tensor tan_autograd(const Tensor& self) {
    Tensor result = self.tan();
    return make_result_with_grad<TanBackward>(result, self, result);
}

inline Tensor rsqrt_autograd(const Tensor& self) {
    Tensor result = self.rsqrt();  // Has CUDA dispatch
    return make_result_with_grad<RsqrtBackward>(result, self, result);
}

inline Tensor square_autograd(const Tensor& self) {
    Tensor result = self.square();  // Has CUDA dispatch
    return make_result_with_grad<SquareBackward>(result, self, self);
}

inline Tensor reciprocal_autograd(const Tensor& self) {
    Tensor result = self.reciprocal();
    return make_result_with_grad<ReciprocalBackward>(result, self, result);
}

inline Tensor log2_autograd(const Tensor& self) {
    Tensor result = self.log2();
    return make_result_with_grad<Log2Backward>(result, self, self);
}

inline Tensor log10_autograd(const Tensor& self) {
    Tensor result = self.log10();
    return make_result_with_grad<Log10Backward>(result, self, self);
}

// ============================================================================
// Activation Operations with Autograd
// ============================================================================

inline Tensor leaky_relu_autograd(const Tensor& self, double negative_slope = 0.01) {
    // Forward: compute leaky_relu
    Tensor result;
#ifdef PT_USE_CUDA
    if (self.is_cuda()) {
        result = at::cuda_ops::leaky_relu(self, static_cast<float>(negative_slope));
    } else
#endif
    {
        Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
        result = at::empty(self_contig.sizes(), self_contig.dtype());
        const float* in_data = self_contig.data_ptr<float>();
        float* out_data = result.mutable_data_ptr<float>();
        float slope = static_cast<float>(negative_slope);
        for (int64_t i = 0; i < self_contig.numel(); ++i) {
            out_data[i] = in_data[i] > 0.0f ? in_data[i] : in_data[i] * slope;
        }
    }

    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<LeakyReluBackward>(self, negative_slope);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor elu_autograd(const Tensor& self, double alpha = 1.0) {
    // Forward: compute ELU
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor result = at::empty(self_contig.sizes(), self_contig.dtype());
    const float* in_data = self_contig.data_ptr<float>();
    float* out_data = result.mutable_data_ptr<float>();
    float a = static_cast<float>(alpha);

    for (int64_t i = 0; i < self_contig.numel(); ++i) {
        out_data[i] = in_data[i] > 0.0f ? in_data[i] : a * (std::exp(in_data[i]) - 1.0f);
    }

    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<ELUBackward>(self, alpha);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor selu_autograd(const Tensor& self) {
    // Forward: compute SELU
    constexpr float alpha = 1.6732632423543772848170429916717f;
    constexpr float scale = 1.0507009873554804934193349852946f;

    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor result = at::empty(self_contig.sizes(), self_contig.dtype());
    const float* in_data = self_contig.data_ptr<float>();
    float* out_data = result.mutable_data_ptr<float>();

    for (int64_t i = 0; i < self_contig.numel(); ++i) {
        if (in_data[i] > 0.0f) {
            out_data[i] = scale * in_data[i];
        } else {
            out_data[i] = scale * alpha * (std::exp(in_data[i]) - 1.0f);
        }
    }

    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<SELUBackward>(self);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor mish_autograd(const Tensor& self) {
    // Forward: compute Mish = x * tanh(softplus(x))
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor result = at::empty(self_contig.sizes(), self_contig.dtype());
    const float* in_data = self_contig.data_ptr<float>();
    float* out_data = result.mutable_data_ptr<float>();

    for (int64_t i = 0; i < self_contig.numel(); ++i) {
        float x = in_data[i];
        float sp = std::log(1.0f + std::exp(x));  // softplus
        out_data[i] = x * std::tanh(sp);
    }

    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<MishBackward>(self);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor hardtanh_autograd(const Tensor& self, double min_val = -1.0, double max_val = 1.0) {
    // Forward: hardtanh = clamp(x, min_val, max_val)
    Tensor result = self.clamp(Scalar(min_val), Scalar(max_val));

    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<HardtanhBackward>(self, min_val, max_val);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor hardsigmoid_autograd(const Tensor& self) {
    // Forward: hardsigmoid(x) = 0 if x <= -3, 1 if x >= 3, x/6 + 0.5 otherwise
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor result = at::empty(self_contig.sizes(), self_contig.dtype());
    const float* in_data = self_contig.data_ptr<float>();
    float* out_data = result.mutable_data_ptr<float>();

    for (int64_t i = 0; i < self_contig.numel(); ++i) {
        float x = in_data[i];
        if (x <= -3.0f) {
            out_data[i] = 0.0f;
        } else if (x >= 3.0f) {
            out_data[i] = 1.0f;
        } else {
            out_data[i] = x / 6.0f + 0.5f;
        }
    }

    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<HardsigmoidBackward>(self);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor hardswish_autograd(const Tensor& self) {
    // Forward: hardswish(x) = x * hardsigmoid(x)
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor result = at::empty(self_contig.sizes(), self_contig.dtype());
    const float* in_data = self_contig.data_ptr<float>();
    float* out_data = result.mutable_data_ptr<float>();

    for (int64_t i = 0; i < self_contig.numel(); ++i) {
        float x = in_data[i];
        float hs;
        if (x <= -3.0f) {
            hs = 0.0f;
        } else if (x >= 3.0f) {
            hs = 1.0f;
        } else {
            hs = x / 6.0f + 0.5f;
        }
        out_data[i] = x * hs;
    }

    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<HardswishBackward>(self);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

// ============================================================================
// Binary Operations with Autograd
// ============================================================================

// Binary ops use Tensor methods which have CUDA dispatch
inline Tensor add_autograd(const Tensor& self, const Tensor& other, Scalar alpha = 1) {
    Tensor result = self.add(other, alpha);  // Has CUDA dispatch
    if (compute_requires_grad(self, other)) {
        auto grad_fn = std::make_shared<AddBackward>(alpha, self.sizes(), other.sizes());
        grad_fn->add_input_metadata(self);
        grad_fn->add_input_metadata(other);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor sub_autograd(const Tensor& self, const Tensor& other, Scalar alpha = 1) {
    Tensor result = self.sub(other, alpha);  // Has CUDA dispatch
    if (compute_requires_grad(self, other)) {
        auto grad_fn = std::make_shared<SubBackward>(alpha, self.sizes(), other.sizes());
        grad_fn->add_input_metadata(self);
        grad_fn->add_input_metadata(other);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor mul_autograd(const Tensor& self, const Tensor& other) {
    Tensor result = self.mul(other);  // Has CUDA dispatch
    return make_result_with_grad2<MulBackward>(result, self, other, self, other);
}

inline Tensor div_autograd(const Tensor& self, const Tensor& other) {
    Tensor result = self.div(other);  // Has CUDA dispatch
    return make_result_with_grad2<DivBackward>(result, self, other, self, other);
}

inline Tensor pow_autograd(const Tensor& self, const Tensor& exponent) {
    Tensor result = self.pow(exponent);  // No CUDA dispatch yet
    return make_result_with_grad2<PowBackward>(result, self, exponent, self, exponent, result);
}

inline Tensor pow_autograd(const Tensor& self, Scalar exponent) {
    Tensor result = self.pow(exponent);  // No CUDA dispatch yet
    return make_result_with_grad<PowScalarBackward>(result, self, self, exponent);
}

// ============================================================================
// Reduction Operations with Autograd
// ============================================================================

// Reduction ops use Tensor methods which have CUDA dispatch
inline Tensor sum_autograd(const Tensor& self) {
    Tensor result = self.sum();  // Has CUDA dispatch
    return make_result_with_grad<SumBackward>(result, self, self.sizes());
}

inline Tensor sum_autograd(const Tensor& self, int64_t dim, bool keepdim = false) {
    Tensor result = self.sum(dim, keepdim);  // Has CUDA dispatch
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<SumDimBackward>(self.sizes(), dim, keepdim);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor mean_autograd(const Tensor& self) {
    Tensor result = self.mean();  // No CUDA dispatch yet
    return make_result_with_grad<MeanBackward>(result, self, self.sizes());
}

inline Tensor mean_autograd(const Tensor& self, int64_t dim, bool keepdim = false) {
    Tensor result = self.mean(dim, keepdim);  // No CUDA dispatch yet
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<MeanDimBackward>(self.sizes(), dim, keepdim);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

// ============================================================================
// New Operations with Autograd
// ============================================================================

inline Tensor clamp_autograd(const Tensor& self, Scalar min_val, Scalar max_val) {
    Tensor result = self.clamp(min_val, max_val);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<ClampBackward>(self, min_val.toDouble(), max_val.toDouble());
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor triu_autograd(const Tensor& self, int64_t diagonal = 0) {
    Tensor result = at::native::triu(self, diagonal);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<TriuBackward>(diagonal);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor tril_autograd(const Tensor& self, int64_t diagonal = 0) {
    Tensor result = at::native::tril(self, diagonal);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<TrilBackward>(diagonal);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor diag_autograd(const Tensor& self, int64_t diagonal = 0) {
    Tensor result = at::native::diag(self, diagonal);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<DiagBackward>(diagonal, self.sizes(), self.dim());
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor cumsum_autograd(const Tensor& self, int64_t dim) {
    Tensor result = at::native::cumsum(self, dim);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<CumsumBackward>(dim, self.sizes());
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor cumprod_autograd(const Tensor& self, int64_t dim) {
    Tensor result = at::native::cumprod(self, dim);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<CumprodBackward>(dim, self, result);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline std::tuple<Tensor, Tensor> sort_autograd(const Tensor& self, int64_t dim = -1, bool descending = false) {
    auto [values, indices] = at::native::sort(self, dim, descending);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<SortBackward>(indices, dim, self.sizes());
        grad_fn->add_input_metadata(self);
        set_grad_fn(values, grad_fn);
        values.set_requires_grad(true);
    }
    return std::make_tuple(values, indices);
}

inline std::tuple<Tensor, Tensor> topk_autograd(const Tensor& self, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true) {
    auto [values, indices] = at::native::topk(self, k, dim, largest, sorted);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<TopkBackward>(indices, dim, self.sizes());
        grad_fn->add_input_metadata(self);
        set_grad_fn(values, grad_fn);
        values.set_requires_grad(true);
    }
    return std::make_tuple(values, indices);
}

// ============================================================================
// Linear Algebra Operations with Autograd
// ============================================================================

// Linear algebra ops - use CUDA dispatched versions from torch:: namespace
inline Tensor mm_autograd(const Tensor& self, const Tensor& other) {
#ifdef PT_USE_CUDA
    if (self.is_cuda()) {
        Tensor result = at::cuda_ops::mm(self, other);
        return make_result_with_grad2<MmBackward>(result, self, other, self, other);
    }
#endif
    Tensor result = at::native::mm(self, other);
    return make_result_with_grad2<MmBackward>(result, self, other, self, other);
}

inline Tensor mv_autograd(const Tensor& self, const Tensor& vec) {
    // mv: no CUDA dispatch yet, fallback to CPU
    Tensor result = at::native::mv(self, vec);
    return make_result_with_grad2<MvBackward>(result, self, vec, self, vec);
}

inline Tensor bmm_autograd(const Tensor& self, const Tensor& other) {
#ifdef PT_USE_CUDA
    if (self.is_cuda()) {
        Tensor result = at::cuda_ops::bmm(self, other);
        return make_result_with_grad2<BmmBackward>(result, self, other, self, other);
    }
#endif
    Tensor result = at::native::bmm(self, other);
    return make_result_with_grad2<BmmBackward>(result, self, other, self, other);
}

inline Tensor matmul_autograd(const Tensor& self, const Tensor& other) {
#ifdef PT_USE_CUDA
    if (self.is_cuda()) {
        // matmul dispatches to mm for 2D, bmm for 3D
        if (self.dim() == 2 && other.dim() == 2) {
            Tensor result = at::cuda_ops::mm(self, other);
            return make_result_with_grad2<MatmulBackward>(result, self, other, self, other);
        } else if (self.dim() == 3 && other.dim() == 3) {
            Tensor result = at::cuda_ops::bmm(self, other);
            return make_result_with_grad2<MatmulBackward>(result, self, other, self, other);
        }
    }
#endif
    Tensor result = at::native::matmul(self, other);
    return make_result_with_grad2<MatmulBackward>(result, self, other, self, other);
}

inline Tensor dot_autograd(const Tensor& self, const Tensor& other) {
    // dot: no CUDA dispatch yet, fallback to CPU
    Tensor result = at::native::dot(self, other);
    return make_result_with_grad2<DotBackward>(result, self, other, self, other);
}

inline Tensor einsum_autograd(const std::string& equation, const std::vector<Tensor>& tensors) {
    Tensor result = at::native::einsum(equation, tensors);

    bool any_requires_grad = false;
    for (const auto& t : tensors) {
        if (compute_requires_grad(t)) {
            any_requires_grad = true;
            break;
        }
    }

    if (any_requires_grad) {
        auto grad_fn = std::make_shared<EinsumBackward>(equation, tensors);
        for (const auto& t : tensors) {
            grad_fn->add_input_metadata(t);
        }
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }

    return result;
}

// ============================================================================
// Shape Operations with Autograd
// ============================================================================

inline Tensor view_autograd(const Tensor& self, c10::IntArrayRef sizes) {
    Tensor result = at::native::view(self, sizes);
    return make_result_with_grad<ViewBackward>(result, self, self.sizes());
}

inline Tensor reshape_autograd(const Tensor& self, c10::IntArrayRef sizes) {
    Tensor result = at::native::reshape(self, sizes);
    return make_result_with_grad<ReshapeBackward>(result, self, self.sizes());
}

inline Tensor squeeze_autograd(const Tensor& self) {
    Tensor result = at::native::squeeze(self);
    return make_result_with_grad<SqueezeBackward>(result, self, self.sizes());
}

inline Tensor squeeze_autograd(const Tensor& self, int64_t dim) {
    Tensor result = at::native::squeeze(self, dim);
    return make_result_with_grad<SqueezeDimBackward>(result, self, dim);
}

inline Tensor unsqueeze_autograd(const Tensor& self, int64_t dim) {
    Tensor result = at::native::unsqueeze(self, dim);
    return make_result_with_grad<UnsqueezeBackward>(result, self, dim);
}

inline Tensor transpose_autograd(const Tensor& self, int64_t dim0, int64_t dim1) {
    Tensor result = at::native::transpose(self, dim0, dim1);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<TransposeBackward>(dim0, dim1);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor t_autograd(const Tensor& self) {
    Tensor result = at::native::t(self);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<TBackward>();
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor narrow_autograd(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    Tensor result = at::native::narrow(self, dim, start, length);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<NarrowBackward>(self.sizes(), dim, start);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor select_autograd(const Tensor& self, int64_t dim, int64_t index) {
    Tensor result = at::native::select(self, dim, index);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<SelectBackward>(self.sizes(), dim, index);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

// ============================================================================
// Index Operations with Autograd
// ============================================================================

inline Tensor index_with_tensor_autograd(const Tensor& self, int64_t dim, const Tensor& index) {
    Tensor result = at::native::index_with_tensor(self, dim, index);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<IndexWithTensorBackward>(
            self.sizes().vec(), index, dim, self.dtype());
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor boolean_index_autograd(const Tensor& self, const Tensor& mask) {
    Tensor result = at::native::boolean_index(self, mask);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<BooleanIndexBackward>(
            self.sizes().vec(), mask, self.dtype());
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

// ============================================================================
// Tensor.backward() implementation
// ============================================================================

inline void tensor_backward(const Tensor& self, const Tensor& gradient = Tensor(),
                             bool retain_graph = false, bool create_graph = false) {
    Tensor grad = gradient;
    if (!grad.defined()) {
        // Default gradient is ones for scalar, error for non-scalar
        if (self.numel() == 1) {
            grad = at::ones({});
        } else {
            throw std::runtime_error(
                "grad can be implicitly created only for scalar outputs"
            );
        }
    }

    backward({self}, {grad}, retain_graph, create_graph);
}

// ============================================================================
// Graph Cleanup Functions
// ============================================================================
// These are critical for preventing memory leaks after backward()

// Clear the entire autograd graph starting from a tensor's grad_fn
// This recursively releases all nodes and breaks reference cycles
inline void clear_autograd_graph(Tensor& tensor) {
    auto* meta = get_autograd_meta(tensor);
    if (!meta || !meta->grad_fn) {
        return;
    }

    // Use a queue to traverse the graph (BFS)
    std::queue<std::shared_ptr<Node>> to_release;
    std::unordered_set<Node*> visited;

    to_release.push(meta->grad_fn);
    visited.insert(meta->grad_fn.get());

    while (!to_release.empty()) {
        auto node = to_release.front();
        to_release.pop();

        // Add children to queue before releasing
        for (const auto& edge : node->next_edges()) {
            if (edge.function && visited.find(edge.function.get()) == visited.end()) {
                visited.insert(edge.function.get());
                to_release.push(edge.function);
            }
        }

        // Release this node (clears next_edges and saved tensors)
        node->release();
    }

    // Finally clear the root grad_fn
    meta->grad_fn.reset();
    meta->is_leaf_ = true;
}

// Clear grad_fn from all tensors in a list
inline void clear_autograd_graph(std::vector<Tensor>& tensors) {
    for (auto& t : tensors) {
        clear_autograd_graph(t);
    }
}

// Clear the grad_fn from a gradient tensor (param.grad_)
// This is critical to break reference cycles from old backward passes
inline void clear_gradient_graph(Tensor& grad) {
    if (!grad.defined()) return;

    auto* meta = get_autograd_meta(grad);
    if (meta && meta->grad_fn) {
        // Just reset, don't traverse - gradients shouldn't have deep graphs
        meta->grad_fn.reset();
        meta->is_leaf_ = true;
    }
}

// Note: clear_param_gradients moved to nn.h to avoid circular dependency
// Use torch::nn::clear_param_gradients() instead

// ============================================================================
// Linalg Operations with Autograd
// ============================================================================

inline Tensor inverse_autograd(const Tensor& self) {
    Tensor result = at::native::inverse(self);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<InverseBackward>(result);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor det_autograd(const Tensor& self) {
    Tensor result = at::native::det(self);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<DetBackward>(self, result);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor cholesky_autograd(const Tensor& self, bool upper = false) {
    Tensor result = at::native::cholesky(self, upper);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<CholeskyBackward>(result);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor trace_autograd(const Tensor& self) {
    Tensor result = at::native::trace(self);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<TraceBackward>(self.size(0), self.size(1));
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

// ============================================================================
// Flip/Roll/RepeatInterleave with Autograd
// ============================================================================

inline Tensor flip_autograd(const Tensor& self, c10::IntArrayRef dims) {
    Tensor result = at::native::flip(self, dims);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<FlipBackward>(dims);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor roll_autograd(const Tensor& self, c10::IntArrayRef shifts, c10::IntArrayRef dims) {
    Tensor result = at::native::roll(self, shifts, dims);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<RollBackward>(shifts, dims);
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

inline Tensor repeat_interleave_autograd(const Tensor& self, int64_t repeats, int64_t dim = 0) {
    Tensor result = at::native::repeat_interleave(self, repeats, dim);
    if (compute_requires_grad(self)) {
        auto grad_fn = std::make_shared<RepeatInterleaveBackward>(repeats, dim, self.sizes());
        grad_fn->add_input_metadata(self);
        set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

} // namespace autograd
} // namespace torch
