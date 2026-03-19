#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/tuda/TudaMath.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "c10/core/ScalarType.h"
#include <cmath>
#include <algorithm>
#include <optional>
#include <limits>

// Threading via c10::ThreadPool (hot_loops.cpp handles parallelism for float)
// Non-float fallback paths run serially — they are rare and small.

namespace at {
namespace native {

// ============================================================================
// Broadcasting Helper
// ============================================================================

namespace detail {

// Check if shapes are broadcastable and compute result shape
inline std::vector<int64_t> broadcast_shapes(
    c10::IntArrayRef a,
    c10::IntArrayRef b
) {
    int64_t ndim_a = static_cast<int64_t>(a.size());
    int64_t ndim_b = static_cast<int64_t>(b.size());
    int64_t ndim = std::max(ndim_a, ndim_b);

    std::vector<int64_t> result(ndim);

    for (int64_t i = ndim - 1; i >= 0; --i) {
        int64_t idx_a = i - (ndim - ndim_a);
        int64_t idx_b = i - (ndim - ndim_b);

        int64_t dim_a = (idx_a >= 0) ? a[idx_a] : 1;
        int64_t dim_b = (idx_b >= 0) ? b[idx_b] : 1;

        PT_CHECK_MSG(dim_a == dim_b || dim_a == 1 || dim_b == 1,
            "Shapes are not broadcastable: ", dim_a, " vs ", dim_b);

        result[i] = std::max(dim_a, dim_b);
    }

    return result;
}

// Compute linear index for broadcasting
inline int64_t broadcast_index(
    int64_t linear_idx,
    c10::IntArrayRef result_shape,
    c10::IntArrayRef tensor_shape,
    c10::IntArrayRef tensor_strides
) {
    int64_t ndim_result = static_cast<int64_t>(result_shape.size());
    int64_t ndim_tensor = static_cast<int64_t>(tensor_shape.size());

    int64_t tensor_idx = 0;
    int64_t remaining = linear_idx;

    for (int64_t i = 0; i < ndim_result; ++i) {
        int64_t idx_in_dim = remaining / 1;  // Will be computed properly

        // Compute index in this dimension
        int64_t stride = 1;
        for (int64_t j = i + 1; j < ndim_result; ++j) {
            stride *= result_shape[j];
        }
        idx_in_dim = (remaining / stride) % result_shape[i];

        // Map to tensor dimension
        int64_t tensor_dim = i - (ndim_result - ndim_tensor);
        if (tensor_dim >= 0) {
            int64_t tensor_dim_size = tensor_shape[tensor_dim];
            // If dimension is 1, broadcast (use index 0)
            int64_t tensor_dim_idx = (tensor_dim_size == 1) ? 0 : idx_in_dim;
            tensor_idx += tensor_dim_idx * tensor_strides[tensor_dim];
        }
    }

    return tensor_idx;
}

} // namespace detail

// ============================================================================
// Unary Operations — AVX2 SIMD + scalar fallback
// ============================================================================

// Scalar helper functions (templated for float/double dispatch)
namespace unary_ops {
    template<typename T> static inline T neg_val(T x) { return -x; }
    template<typename T> static inline T abs_val(T x) { return std::abs(x); }
    template<typename T> static inline T sqrt_val(T x) { return std::sqrt(x); }
    template<typename T> static inline T rsqrt_val(T x) { return T(1) / std::sqrt(x); }
    template<typename T> static inline T square_val(T x) { return x * x; }
    template<typename T> static inline T exp_val(T x) { return std::exp(x); }
    template<typename T> static inline T log_val(T x) { return std::log(x); }
    template<typename T> static inline T log2_val(T x) { return std::log2(x); }
    template<typename T> static inline T log10_val(T x) { return std::log10(x); }
    template<typename T> static inline T sin_val(T x) { return std::sin(x); }
    template<typename T> static inline T cos_val(T x) { return std::cos(x); }
    template<typename T> static inline T tan_val(T x) { return std::tan(x); }
    template<typename T> static inline T tanh_val(T x) { return std::tanh(x); }
    template<typename T> static inline T sigmoid_val(T x) { return T(1) / (T(1) + std::exp(-x)); }
    template<typename T> static inline T relu_val(T x) { return x > T(0) ? x : T(0); }
    template<typename T> static inline T ceil_val(T x) { return std::ceil(x); }
    template<typename T> static inline T floor_val(T x) { return std::floor(x); }
    template<typename T> static inline T round_val(T x) { return std::round(x); }
    template<typename T> static inline T sign_val(T x) { return (x > T(0)) ? T(1) : ((x < T(0)) ? T(-1) : T(0)); }
    template<typename T> static inline T reciprocal_val(T x) { return T(1) / x; }
} // namespace unary_ops

// Unary op: delegates to hot::name_loop() (compiled in hot_loops.cpp) for float32.
// On Elbrus, this allows LCC VLIW scheduling across the full loop body.
// On x86, reduces binary size by eliminating per-TU inline copies.
// vec_body / scalar_fn kept for non-float dispatch and in-place ops.
// FAST PATH: trusted tensors skip contiguous check & dtype dispatch entirely.
#define DEFINE_UNARY_OP(name, vec_body, scalar_fn) \
inline Tensor name(const Tensor& self) { \
    if (self.is_trusted()) { \
        Tensor result = at::empty(self.sizes()); \
        hot::name##_loop(self.data_ptr<float>(), \
                         result.mutable_data_ptr<float>(), self.numel()); \
        return result; \
    } \
    Tensor input = self.is_contiguous() ? self : self.contiguous(); \
    Tensor result = empty_like(input); \
    if (input.dtype() == c10::ScalarType::Float) { \
        hot::name##_loop(input.data_ptr<float>(), \
                         result.mutable_data_ptr<float>(), input.numel()); \
        return result; \
    } \
    PT_DISPATCH_FLOATING_TYPES(input.dtype(), #name, [&] { \
        const scalar_t* in = input.data_ptr<scalar_t>(); \
        scalar_t* out = result.mutable_data_ptr<scalar_t>(); \
        int64_t n = input.numel(); \
        for (int64_t i = 0; i < n; ++i) out[i] = scalar_fn<scalar_t>(in[i]); \
    }); \
    return result; \
}

#define DEFINE_UNARY_OP_INPLACE(name, vec_body, scalar_fn) \
inline Tensor& name##_(Tensor& self) { \
    if (!self.is_contiguous()) { \
        Tensor tmp = name(self); \
        self.copy_(tmp); \
        return self; \
    } \
    if (self.dtype() == c10::ScalarType::Float) { \
        float* data = self.mutable_data_ptr<float>(); \
        int64_t n = self.numel(); \
        constexpr int W = tuda::VecF::width; \
        for (int64_t i = 0; i < n - (n % W); i += W) { \
            tuda::VecF v = tuda::VecF::load(data + i); \
            (vec_body).store(data + i); \
        } \
        for (int64_t i = n - (n % W); i < n; ++i) data[i] = scalar_fn<float>(data[i]); \
        return self; \
    } \
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), #name "_", [&] { \
        scalar_t* data = self.mutable_data_ptr<scalar_t>(); \
        int64_t n = self.numel(); \
        for (int64_t i = 0; i < n; ++i) data[i] = scalar_fn<scalar_t>(data[i]); \
    }); \
    return self; \
}

// Unary ops with TUDA VecF (portable across AVX2/NEON/E2K/Scalar)
DEFINE_UNARY_OP(neg,        tuda::neg_vec(v),        unary_ops::neg_val)
DEFINE_UNARY_OP(abs,        tuda::abs_vec(v),        unary_ops::abs_val)
DEFINE_UNARY_OP(sqrt,       tuda::sqrt_vec(v),       unary_ops::sqrt_val)
DEFINE_UNARY_OP(rsqrt,      tuda::rsqrt_vec(v),      unary_ops::rsqrt_val)
DEFINE_UNARY_OP(square,     tuda::square_vec(v),     unary_ops::square_val)
DEFINE_UNARY_OP(exp,        tuda::exp_vec(v),        unary_ops::exp_val)
DEFINE_UNARY_OP(log,        tuda::log_vec(v),        unary_ops::log_val)
DEFINE_UNARY_OP(log2,       tuda::log2_vec(v),       unary_ops::log2_val)
DEFINE_UNARY_OP(log10,      tuda::log10_vec(v),      unary_ops::log10_val)
DEFINE_UNARY_OP(sin,        tuda::sin_vec(v),        unary_ops::sin_val)
DEFINE_UNARY_OP(cos,        tuda::cos_vec(v),        unary_ops::cos_val)
DEFINE_UNARY_OP(tan,        tuda::tan_vec(v),        unary_ops::tan_val)
DEFINE_UNARY_OP(tanh,       tuda::tanh_vec(v),       unary_ops::tanh_val)
DEFINE_UNARY_OP(sigmoid,    tuda::sigmoid_vec(v),    unary_ops::sigmoid_val)
DEFINE_UNARY_OP(relu,       tuda::relu_vec(v),       unary_ops::relu_val)
DEFINE_UNARY_OP(ceil,       tuda::ceil_vec(v),       unary_ops::ceil_val)
DEFINE_UNARY_OP(floor,      tuda::floor_vec(v),      unary_ops::floor_val)
DEFINE_UNARY_OP(round,      tuda::round_vec(v),      unary_ops::round_val)
DEFINE_UNARY_OP(sign,       tuda::sign_vec(v),       unary_ops::sign_val)
DEFINE_UNARY_OP(reciprocal, tuda::reciprocal_vec(v), unary_ops::reciprocal_val)

// In-place variants
DEFINE_UNARY_OP_INPLACE(neg,     tuda::neg_vec(v),        unary_ops::neg_val)
DEFINE_UNARY_OP_INPLACE(abs,     tuda::abs_vec(v),        unary_ops::abs_val)
DEFINE_UNARY_OP_INPLACE(sqrt,    tuda::sqrt_vec(v),       unary_ops::sqrt_val)
DEFINE_UNARY_OP_INPLACE(exp,     tuda::exp_vec(v),        unary_ops::exp_val)
DEFINE_UNARY_OP_INPLACE(log,     tuda::log_vec(v),        unary_ops::log_val)
DEFINE_UNARY_OP_INPLACE(sin,     tuda::sin_vec(v),        unary_ops::sin_val)
DEFINE_UNARY_OP_INPLACE(cos,     tuda::cos_vec(v),        unary_ops::cos_val)
DEFINE_UNARY_OP_INPLACE(tanh,    tuda::tanh_vec(v),       unary_ops::tanh_val)
DEFINE_UNARY_OP_INPLACE(sigmoid, tuda::sigmoid_vec(v),    unary_ops::sigmoid_val)
DEFINE_UNARY_OP_INPLACE(relu,    tuda::relu_vec(v),       unary_ops::relu_val)
DEFINE_UNARY_OP_INPLACE(ceil,    tuda::ceil_vec(v),       unary_ops::ceil_val)
DEFINE_UNARY_OP_INPLACE(floor,   tuda::floor_vec(v),      unary_ops::floor_val)
DEFINE_UNARY_OP_INPLACE(round,   tuda::round_vec(v),      unary_ops::round_val)

#undef DEFINE_UNARY_OP
#undef DEFINE_UNARY_OP_INPLACE

// Zero and fill
inline Tensor& zero_(Tensor& self) {
    if (self.is_contiguous()) {
        // For large tensors, parallelize via typed zero-fill
        int64_t n = self.numel();
        {
            std::memset(self.data_ptr(), 0, self.nbytes());
        }
    } else {
        PT_DISPATCH_ALL_TYPES(self.dtype(), "zero_", [&] {
            scalar_t* base = self.mutable_data_ptr<scalar_t>();
            int64_t n = self.numel();
            int64_t ndim = self.dim();
            auto sz = self.sizes();
            auto st = self.strides();
            for (int64_t flat = 0; flat < n; ++flat) {
                int64_t offset = 0;
                int64_t rem = flat;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    int64_t idx = rem % sz[d];
                    rem /= sz[d];
                    offset += idx * st[d];
                }
                base[offset] = 0;
            }
        });
    }
    return self;
}

inline Tensor& fill_(Tensor& self, Scalar value) {
    // SIMD fast path for contiguous float
    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous()) {
        float* data = self.mutable_data_ptr<float>();
        float fval = static_cast<float>(value.toDouble());
        int64_t n = self.numel();
        tuda::vec_fill(data, fval, n);
        return self;
    }

    PT_DISPATCH_ALL_TYPES(self.dtype(), "fill_", [&] {
        scalar_t val = value.to<scalar_t>();
        if (self.is_contiguous()) {
            scalar_t* data = self.mutable_data_ptr<scalar_t>();
            int64_t numel = self.numel();
            for (int64_t i = 0; i < numel; ++i) data[i] = val;
        } else {
            scalar_t* base = self.mutable_data_ptr<scalar_t>();
            int64_t n = self.numel();
            int64_t ndim = self.dim();
            auto sz = self.sizes();
            auto st = self.strides();
            for (int64_t flat = 0; flat < n; ++flat) {
                int64_t offset = 0;
                int64_t rem = flat;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    int64_t idx = rem % sz[d];
                    rem /= sz[d];
                    offset += idx * st[d];
                }
                base[offset] = val;
            }
        }
    });
    return self;
}

// ============================================================================
// Clamp Operations
// ============================================================================

// clamp(self, min_val, max_val) — element-wise clamp
inline Tensor clamp(const Tensor& self, Scalar min_val, Scalar max_val) {
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);
    PT_DISPATCH_ALL_TYPES(input.dtype(), "clamp", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        scalar_t lo = min_val.to<scalar_t>();
        scalar_t hi = max_val.to<scalar_t>();
        int64_t n = input.numel();

        for (int64_t i = 0; i < n; ++i) {
            out[i] = std::min(std::max(in[i], lo), hi);
        }
    });
    return result;
}

// clamp with optional min/max
inline Tensor clamp(const Tensor& self, std::optional<Scalar> min_opt, std::optional<Scalar> max_opt) {
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);
    PT_DISPATCH_ALL_TYPES(input.dtype(), "clamp_opt", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t n = input.numel();
        bool has_min = min_opt.has_value();
        bool has_max = max_opt.has_value();
        scalar_t lo = has_min ? min_opt->to<scalar_t>() : std::numeric_limits<scalar_t>::lowest();
        scalar_t hi = has_max ? max_opt->to<scalar_t>() : std::numeric_limits<scalar_t>::max();
        for (int64_t i = 0; i < n; ++i) {
            out[i] = std::min(std::max(in[i], lo), hi);
        }
    });
    return result;
}

// In-place clamp
inline Tensor& clamp_(Tensor& self, Scalar min_val, Scalar max_val) {
    if (!self.is_contiguous()) {
        Tensor tmp = clamp(self, min_val, max_val);
        self.copy_(tmp);
        return self;
    }
    PT_DISPATCH_ALL_TYPES(self.dtype(), "clamp_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        scalar_t lo = min_val.to<scalar_t>();
        scalar_t hi = max_val.to<scalar_t>();
        int64_t n = self.numel();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = std::min(std::max(data[i], lo), hi);
        }
    });
    return self;
}

inline Tensor clamp_min(const Tensor& self, Scalar min_val) {
    return clamp(self, min_val, Scalar(std::numeric_limits<double>::max()));
}

inline Tensor clamp_max(const Tensor& self, Scalar max_val) {
    return clamp(self, Scalar(std::numeric_limits<double>::lowest()), max_val);
}

// ============================================================================
// Matrix Triangle and Diagonal Operations
// ============================================================================

// Upper triangular
inline Tensor triu(const Tensor& self, int64_t diagonal = 0) {
    PT_CHECK(self.dim() == 2);
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = zeros_like(input);
    int64_t rows = input.size(0);
    int64_t cols = input.size(1);
    PT_DISPATCH_ALL_TYPES(input.dtype(), "triu", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = std::max((int64_t)0, i + diagonal); j < cols; ++j) {
                out[i * cols + j] = in[i * cols + j];
            }
        }
    });
    return result;
}

// Lower triangular
inline Tensor tril(const Tensor& self, int64_t diagonal = 0) {
    PT_CHECK(self.dim() == 2);
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = zeros_like(input);
    int64_t rows = input.size(0);
    int64_t cols = input.size(1);
    PT_DISPATCH_ALL_TYPES(input.dtype(), "tril", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j <= std::min(i + diagonal, cols - 1); ++j) {
                if (j >= 0) out[i * cols + j] = in[i * cols + j];
            }
        }
    });
    return result;
}

// Diagonal extraction (2D->1D) or diagonal matrix creation (1D->2D)
inline Tensor diag(const Tensor& self, int64_t diagonal = 0) {
    if (self.dim() == 1) {
        // Create 2D diagonal matrix from 1D
        int64_t n = self.size(0) + std::abs(diagonal);
        Tensor result = zeros({n, n}, TensorOptions().dtype(self.dtype()).device(self.device()));
        Tensor input = self.is_contiguous() ? self : self.contiguous();
        PT_DISPATCH_ALL_TYPES(self.dtype(), "diag_1d", [&] {
            const scalar_t* in = input.data_ptr<scalar_t>();
            scalar_t* out = result.mutable_data_ptr<scalar_t>();
            for (int64_t i = 0; i < self.size(0); ++i) {
                int64_t r = (diagonal >= 0) ? i : i - diagonal;
                int64_t c = (diagonal >= 0) ? i + diagonal : i;
                out[r * n + c] = in[i];
            }
        });
        return result;
    } else if (self.dim() == 2) {
        // Extract diagonal from 2D
        int64_t rows = self.size(0);
        int64_t cols = self.size(1);
        int64_t diag_size;
        if (diagonal >= 0) {
            diag_size = std::min(rows, cols - diagonal);
        } else {
            diag_size = std::min(rows + diagonal, cols);
        }
        if (diag_size <= 0) return zeros({0}, TensorOptions().dtype(self.dtype()).device(self.device()));
        Tensor input = self.is_contiguous() ? self : self.contiguous();
        Tensor result = empty({diag_size}, TensorOptions().dtype(self.dtype()).device(self.device()));
        PT_DISPATCH_ALL_TYPES(self.dtype(), "diag_2d", [&] {
            const scalar_t* in = input.data_ptr<scalar_t>();
            scalar_t* out = result.mutable_data_ptr<scalar_t>();
            for (int64_t i = 0; i < diag_size; ++i) {
                int64_t r = (diagonal >= 0) ? i : i - diagonal;
                int64_t c = (diagonal >= 0) ? i + diagonal : i;
                out[i] = in[r * cols + c];
            }
        });
        return result;
    }
    PT_CHECK_MSG(false, "diag: expected 1D or 2D tensor");
    return Tensor();
}

// ============================================================================
// Binary Operations Implementation
// ============================================================================

// Tensor + Tensor with broadcasting
inline Tensor add(const Tensor& self, const Tensor& other, Scalar alpha = 1) {
    // FAST PATH: trusted tensors skip all dtype/contiguous/device checks
    if (self.is_trusted() && other.is_trusted() && self.sizes() == other.sizes()) {
        Tensor result = at::empty(self.sizes());
        hot::add_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      result.mutable_data_ptr<float>(), result.numel(),
                      static_cast<float>(alpha.toDouble()));
        return result;
    }

    // Ultra-fast path: float, same shape, contiguous — delegate to hot_loops.cpp
    if (self.dtype() == c10::ScalarType::Float && other.dtype() == c10::ScalarType::Float &&
        self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
        Tensor result = empty_like(self);
        hot::add_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      result.mutable_data_ptr<float>(), result.numel(),
                      static_cast<float>(alpha.toDouble()));
        return result;
    }

    // SIMD broadcast fast path: [*, N] + [N] (bias addition pattern)
    if (self.dtype() == c10::ScalarType::Float && other.dtype() == c10::ScalarType::Float &&
        self.is_contiguous() && other.is_contiguous() && other.dim() == 1 &&
        self.dim() >= 2 && self.size(-1) == other.size(0)) {
        Tensor result = empty_like(self);
        int64_t inner = other.size(0);
        int64_t outer = self.numel() / inner;
        hot::add_broadcast_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                                result.mutable_data_ptr<float>(), outer, inner,
                                static_cast<float>(alpha.toDouble()));
        return result;
    }

    // General path with broadcasting and type promotion
    auto result_shape = detail::broadcast_shapes(self.sizes(), other.sizes());
    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());
    Tensor result = empty(result_shape, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_ALL_TYPES(result_dtype, "add", [&] {
        scalar_t alpha_val = alpha.to<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t n = result.numel();

        if (self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
            const scalar_t* a = self.data_ptr<scalar_t>();
            const scalar_t* b = other.data_ptr<scalar_t>();
    
            for (int64_t i = 0; i < n; ++i) out[i] = a[i] + alpha_val * b[i];
        } else {
            const scalar_t* a = self.data_ptr<scalar_t>();
            const scalar_t* b = other.data_ptr<scalar_t>();
    
            for (int64_t i = 0; i < n; ++i) {
                int64_t idx_a = detail::broadcast_index(i, result.sizes(), self.sizes(), self.strides());
                int64_t idx_b = detail::broadcast_index(i, result.sizes(), other.sizes(), other.strides());
                out[i] = a[idx_a] + alpha_val * b[idx_b];
            }
        }
    });

    return result;
}

inline Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha = 1) {
    // FAST PATH: trusted tensors skip all checks
    if (self.is_trusted() && other.is_trusted() && self.sizes() == other.sizes() && alpha.toDouble() == 1.0) {
        Tensor result = at::empty(self.sizes());
        hot::sub_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      result.mutable_data_ptr<float>(), result.numel());
        return result;
    }

    // Fast path: float, same shape, contiguous — delegate to hot_loops.cpp
    if (alpha.toDouble() == 1.0 &&
        self.dtype() == c10::ScalarType::Float && other.dtype() == c10::ScalarType::Float &&
        self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
        Tensor result = empty_like(self);
        hot::sub_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      result.mutable_data_ptr<float>(), result.numel());
        return result;
    }
    return add(self, other, Scalar(-alpha.toDouble()));
}

inline Tensor mul(const Tensor& self, const Tensor& other) {
    // FAST PATH: trusted tensors skip all checks
    if (self.is_trusted() && other.is_trusted() && self.sizes() == other.sizes()) {
        Tensor result = at::empty(self.sizes());
        hot::mul_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      result.mutable_data_ptr<float>(), result.numel());
        return result;
    }

    // Ultra-fast path: float, same shape, contiguous — delegate to hot_loops.cpp
    if (self.dtype() == c10::ScalarType::Float && other.dtype() == c10::ScalarType::Float &&
        self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
        Tensor result = empty_like(self);
        hot::mul_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      result.mutable_data_ptr<float>(), result.numel());
        return result;
    }

    auto result_shape = detail::broadcast_shapes(self.sizes(), other.sizes());
    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());
    Tensor result = empty(result_shape, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_ALL_TYPES(result_dtype, "mul", [&] {
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t n = result.numel();

        if (self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
            const scalar_t* a = self.data_ptr<scalar_t>();
            const scalar_t* b = other.data_ptr<scalar_t>();
    
            for (int64_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
        } else {
            const scalar_t* a = self.data_ptr<scalar_t>();
            const scalar_t* b = other.data_ptr<scalar_t>();
    
            for (int64_t i = 0; i < n; ++i) {
                int64_t idx_a = detail::broadcast_index(i, result.sizes(), self.sizes(), self.strides());
                int64_t idx_b = detail::broadcast_index(i, result.sizes(), other.sizes(), other.strides());
                out[i] = a[idx_a] * b[idx_b];
            }
        }
    });

    return result;
}

inline Tensor div(const Tensor& self, const Tensor& other) {
    // FAST PATH: trusted tensors skip all checks
    if (self.is_trusted() && other.is_trusted() && self.sizes() == other.sizes()) {
        Tensor result = at::empty(self.sizes());
        hot::div_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      result.mutable_data_ptr<float>(), result.numel());
        return result;
    }

    // Ultra-fast path: float, same shape, contiguous — delegate to hot_loops.cpp
    if (self.dtype() == c10::ScalarType::Float && other.dtype() == c10::ScalarType::Float &&
        self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
        Tensor result = empty_like(self);
        hot::div_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      result.mutable_data_ptr<float>(), result.numel());
        return result;
    }

    auto result_shape = detail::broadcast_shapes(self.sizes(), other.sizes());
    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());

    if (c10::isIntegralType(result_dtype, true)) {
        result_dtype = c10::ScalarType::Float;
    }

    Tensor result = empty(result_shape, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_FLOATING_TYPES(result_dtype, "div", [&] {
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t n = result.numel();

        if (self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
            const scalar_t* a = self.data_ptr<scalar_t>();
            const scalar_t* b = other.data_ptr<scalar_t>();
    
            for (int64_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
        } else {
            const scalar_t* a = self.data_ptr<scalar_t>();
            const scalar_t* b = other.data_ptr<scalar_t>();
    
            for (int64_t i = 0; i < n; ++i) {
                int64_t idx_a = detail::broadcast_index(i, result.sizes(), self.sizes(), self.strides());
                int64_t idx_b = detail::broadcast_index(i, result.sizes(), other.sizes(), other.strides());
                out[i] = a[idx_a] / b[idx_b];
            }
        }
    });

    return result;
}

// Tensor + Scalar
inline Tensor add(const Tensor& self, Scalar other, Scalar alpha = 1) {
    // FAST PATH: trusted tensors skip contiguous check & dtype dispatch
    if (self.is_trusted()) {
        Tensor result = at::empty(self.sizes());
        const float* in = self.data_ptr<float>();
        float* out = result.mutable_data_ptr<float>();
        float fval = static_cast<float>(other.toDouble() * alpha.toDouble());
        tuda::VecF vval = tuda::VecF::broadcast(fval);
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        for (; i + W <= n; i += W)
            (tuda::VecF::load(in+i) + vval).store(out+i);
        for (; i < n; ++i) out[i] = in[i] + fval;
        return result;
    }

    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);
    double val = other.toDouble() * alpha.toDouble();

    if (input.dtype() == c10::ScalarType::Float) {
        const float* in = input.data_ptr<float>();
        float* out = result.mutable_data_ptr<float>();
        float fval = static_cast<float>(val);
        tuda::VecF vval = tuda::VecF::broadcast(fval);
        constexpr int W = tuda::VecF::width;
        int64_t n = input.numel(), i = 0;
        for (; i + W <= n; i += W)
            (tuda::VecF::load(in+i) + vval).store(out+i);
        for (; i < n; ++i) out[i] = in[i] + fval;
        return result;
    }

    PT_DISPATCH_ALL_TYPES(input.dtype(), "add_scalar", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        scalar_t scalar_val = static_cast<scalar_t>(val);
        int64_t n = input.numel();

        for (int64_t i = 0; i < n; ++i) out[i] = in[i] + scalar_val;
    });

    return result;
}

inline Tensor sub(const Tensor& self, Scalar other, Scalar alpha = 1) {
    return add(self, Scalar(-other.toDouble()), alpha);
}

inline Tensor mul(const Tensor& self, Scalar other) {
    // FAST PATH: trusted tensors skip contiguous check & dtype dispatch
    if (self.is_trusted()) {
        Tensor result = at::empty(self.sizes());
        const float* in = self.data_ptr<float>();
        float* out = result.mutable_data_ptr<float>();
        float fval = static_cast<float>(other.toDouble());
        tuda::VecF vval = tuda::VecF::broadcast(fval);
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        for (; i + W <= n; i += W)
            (tuda::VecF::load(in+i) * vval).store(out+i);
        for (; i < n; ++i) out[i] = in[i] * fval;
        return result;
    }

    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);

    if (input.dtype() == c10::ScalarType::Float) {
        const float* in = input.data_ptr<float>();
        float* out = result.mutable_data_ptr<float>();
        float fval = static_cast<float>(other.toDouble());
        tuda::VecF vval = tuda::VecF::broadcast(fval);
        constexpr int W = tuda::VecF::width;
        int64_t n = input.numel(), i = 0;
        for (; i + W <= n; i += W)
            (tuda::VecF::load(in+i) * vval).store(out+i);
        for (; i < n; ++i) out[i] = in[i] * fval;
        return result;
    }

    PT_DISPATCH_ALL_TYPES(input.dtype(), "mul_scalar", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        scalar_t scalar_val = other.to<scalar_t>();
        int64_t n = input.numel();

        for (int64_t i = 0; i < n; ++i) out[i] = in[i] * scalar_val;
    });

    return result;
}

inline Tensor div(const Tensor& self, Scalar other) {
    return mul(self, Scalar(1.0 / other.toDouble()));
}

inline Tensor pow(const Tensor& self, Scalar exponent) {
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);

    PT_DISPATCH_FLOATING_TYPES(input.dtype(), "pow", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        double exp_val = exponent.toDouble();
        int64_t n = input.numel();

        for (int64_t i = 0; i < n; ++i) {
            out[i] = static_cast<scalar_t>(std::pow(static_cast<double>(in[i]), exp_val));
        }
    });

    return result;
}

inline Tensor pow(const Tensor& self, const Tensor& exponent) {
    auto result_shape = detail::broadcast_shapes(self.sizes(), exponent.sizes());
    Tensor result = empty(result_shape, TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "pow_tensor", [&] {
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        const scalar_t* base = self.data_ptr<scalar_t>();
        const scalar_t* exp = exponent.data_ptr<scalar_t>();
        int64_t n = result.numel();

        if (self.sizes() == exponent.sizes()) {
            for (int64_t i = 0; i < n; ++i) {
                out[i] = static_cast<scalar_t>(std::pow(
                    static_cast<double>(base[i]),
                    static_cast<double>(exp[i])
                ));
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                int64_t idx_a = detail::broadcast_index(i, result.sizes(), self.sizes(), self.strides());
                int64_t idx_b = detail::broadcast_index(i, result.sizes(), exponent.sizes(), exponent.strides());
                out[i] = static_cast<scalar_t>(std::pow(
                    static_cast<double>(base[idx_a]),
                    static_cast<double>(exp[idx_b])
                ));
            }
        }
    });

    return result;
}

// In-place operations
inline Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha = 1) {
    // FAST PATH: trusted tensors skip all checks
    if (self.is_trusted() && other.is_trusted()) {
        hot::axpy_inplace(self.mutable_data_ptr<float>(),
                          static_cast<float>(alpha.toDouble()),
                          other.data_ptr<float>(), self.numel());
        return self;
    }

    PT_CHECK_MSG(self.sizes() == other.sizes(), "In-place operation requires same shapes");

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous() && other.is_contiguous()) {
        float* data = self.mutable_data_ptr<float>();
        const float* other_data = other.data_ptr<float>();
        float alpha_val = static_cast<float>(alpha.toDouble());
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        if (alpha_val == 1.0f) {
            for (; i + W <= n; i += W)
                (tuda::VecF::load(data+i) + tuda::VecF::load(other_data+i)).store(data+i);
        } else {
            tuda::VecF va = tuda::VecF::broadcast(alpha_val);
            for (; i + W <= n; i += W)
                tuda::VecF::fmadd(va, tuda::VecF::load(other_data+i), tuda::VecF::load(data+i)).store(data+i);
        }
        for (; i < n; ++i) data[i] += alpha_val * other_data[i];
        return self;
    }

    // Generic path: ensure contiguous for sequential data[i] access
    Tensor self_c = self.is_contiguous() ? self : self.contiguous();
    Tensor other_c = other.is_contiguous() ? other : other.contiguous();
    PT_DISPATCH_ALL_TYPES(self_c.dtype(), "add_", [&] {
        scalar_t* data = self_c.mutable_data_ptr<scalar_t>();
        const scalar_t* other_data = other_c.data_ptr<scalar_t>();
        scalar_t alpha_val = alpha.to<scalar_t>();
        int64_t n = self_c.numel();

        for (int64_t i = 0; i < n; ++i) data[i] += alpha_val * other_data[i];
    });
    if (!self.is_contiguous()) {
        self.copy_(self_c);
    }

    return self;
}

inline Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha = 1) {
    return add_(self, other, Scalar(-alpha.toDouble()));
}

inline Tensor& mul_(Tensor& self, const Tensor& other) {
    // FAST PATH: trusted tensors skip all checks
    if (self.is_trusted() && other.is_trusted()) {
        hot::mul_loop(self.data_ptr<float>(), other.data_ptr<float>(),
                      self.mutable_data_ptr<float>(), self.numel());
        return self;
    }

    PT_CHECK_MSG(self.sizes() == other.sizes(), "In-place operation requires same shapes");

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous() && other.is_contiguous()) {
        float* data = self.mutable_data_ptr<float>();
        const float* other_data = other.data_ptr<float>();
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        for (; i + W <= n; i += W)
            (tuda::VecF::load(data+i) * tuda::VecF::load(other_data+i)).store(data+i);
        for (; i < n; ++i) data[i] *= other_data[i];
        return self;
    }

    // Generic path: ensure contiguous for sequential data[i] access
    Tensor self_c = self.is_contiguous() ? self : self.contiguous();
    Tensor other_c = other.is_contiguous() ? other : other.contiguous();
    PT_DISPATCH_ALL_TYPES(self_c.dtype(), "mul_", [&] {
        scalar_t* data = self_c.mutable_data_ptr<scalar_t>();
        const scalar_t* other_data = other_c.data_ptr<scalar_t>();
        int64_t n = self_c.numel();

        for (int64_t i = 0; i < n; ++i) data[i] *= other_data[i];
    });
    if (!self.is_contiguous()) {
        self.copy_(self_c);
    }

    return self;
}

inline Tensor& div_(Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.sizes() == other.sizes(), "In-place operation requires same shapes");

    // Generic path: ensure contiguous for sequential data[i] access
    Tensor self_c = self.is_contiguous() ? self : self.contiguous();
    Tensor other_c = other.is_contiguous() ? other : other.contiguous();
    PT_DISPATCH_FLOATING_TYPES(self_c.dtype(), "div_", [&] {
        scalar_t* data = self_c.mutable_data_ptr<scalar_t>();
        const scalar_t* other_data = other_c.data_ptr<scalar_t>();
        int64_t n = self_c.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] /= other_data[i];
        }
    });
    if (!self.is_contiguous()) {
        self.copy_(self_c);
    }

    return self;
}

inline Tensor& add_(Tensor& self, Scalar other, Scalar alpha = 1) {
    // FAST PATH: trusted tensor — skip dtype/contiguous checks
    if (self.is_trusted()) {
        float* data = self.mutable_data_ptr<float>();
        float val = static_cast<float>(other.toDouble() * alpha.toDouble());
        tuda::VecF vval = tuda::VecF::broadcast(val);
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        for (; i + W <= n; i += W)
            (tuda::VecF::load(data+i) + vval).store(data+i);
        for (; i < n; ++i) data[i] += val;
        return self;
    }

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous()) {
        float* data = self.mutable_data_ptr<float>();
        float val = static_cast<float>(other.toDouble() * alpha.toDouble());
        tuda::VecF vval = tuda::VecF::broadcast(val);
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        for (; i + W <= n; i += W)
            (tuda::VecF::load(data+i) + vval).store(data+i);
        for (; i < n; ++i) data[i] += val;
        return self;
    }

    PT_DISPATCH_ALL_TYPES(self.dtype(), "add_scalar_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        scalar_t val = static_cast<scalar_t>(other.toDouble() * alpha.toDouble());
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) data[i] += val;
    });

    return self;
}

inline Tensor& sub_(Tensor& self, Scalar other, Scalar alpha = 1) {
    return add_(self, Scalar(-other.toDouble()), alpha);
}

inline Tensor& mul_(Tensor& self, Scalar other) {
    // FAST PATH: trusted tensor — skip dtype/contiguous checks
    if (self.is_trusted()) {
        hot::mul_scalar_inplace(self.mutable_data_ptr<float>(),
                                static_cast<float>(other.toDouble()), self.numel());
        return self;
    }

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous()) {
        float* data = self.mutable_data_ptr<float>();
        float val = static_cast<float>(other.toDouble());
        tuda::VecF vval = tuda::VecF::broadcast(val);
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        for (; i + W <= n; i += W)
            (tuda::VecF::load(data+i) * vval).store(data+i);
        for (; i < n; ++i) data[i] *= val;
        return self;
    }

    PT_DISPATCH_ALL_TYPES(self.dtype(), "mul_scalar_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        scalar_t val = other.to<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) data[i] *= val;
    });

    return self;
}

inline Tensor& div_(Tensor& self, Scalar other) {
    return mul_(self, Scalar(1.0 / other.toDouble()));
}

// ============================================================================
// Fused operations (for optimizer efficiency)
// ============================================================================

// addcmul_: self += value * tensor1 * tensor2
inline Tensor& addcmul_(Tensor& self, const Tensor& tensor1, const Tensor& tensor2, Scalar value) {
    PT_CHECK_MSG(self.sizes() == tensor1.sizes() && self.sizes() == tensor2.sizes(),
                 "addcmul_ requires all tensors to have the same shape");

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous() &&
        tensor1.is_contiguous() && tensor2.is_contiguous()) {
        float* data = self.mutable_data_ptr<float>();
        const float* t1 = tensor1.data_ptr<float>();
        const float* t2 = tensor2.data_ptr<float>();
        float val = static_cast<float>(value.toDouble());
        tuda::VecF vval = tuda::VecF::broadcast(val);
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        for (; i + W <= n; i += W)
            tuda::VecF::fmadd(vval, tuda::VecF::load(t1+i) * tuda::VecF::load(t2+i),
                              tuda::VecF::load(data+i)).store(data+i);
        for (; i < n; ++i) data[i] += val * t1[i] * t2[i];
        return self;
    }

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "addcmul_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        const scalar_t* t1 = tensor1.data_ptr<scalar_t>();
        const scalar_t* t2 = tensor2.data_ptr<scalar_t>();
        scalar_t val = value.to<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) data[i] += val * t1[i] * t2[i];
    });

    return self;
}

// addcmul: self + value * tensor1 * tensor2 (non-inplace)
inline Tensor addcmul(const Tensor& self, const Tensor& tensor1, const Tensor& tensor2, Scalar value = 1) {
    Tensor result = self.clone();
    return addcmul_(result, tensor1, tensor2, value);
}

// addcdiv_: self += value * tensor1 / tensor2 (inplace)
inline Tensor& addcdiv_(Tensor& self, const Tensor& tensor1, const Tensor& tensor2, Scalar value) {
    PT_CHECK_MSG(self.sizes() == tensor1.sizes() && self.sizes() == tensor2.sizes(),
                 "addcdiv_ requires all tensors to have the same shape");

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous() &&
        tensor1.is_contiguous() && tensor2.is_contiguous()) {
        float* data = self.mutable_data_ptr<float>();
        const float* t1 = tensor1.data_ptr<float>();
        const float* t2 = tensor2.data_ptr<float>();
        float val = static_cast<float>(value.toDouble());
        tuda::VecF vval = tuda::VecF::broadcast(val);
        constexpr int W = tuda::VecF::width;
        int64_t n = self.numel(), i = 0;
        for (; i + W <= n; i += W)
            tuda::VecF::fmadd(vval, tuda::VecF::load(t1+i) / tuda::VecF::load(t2+i),
                              tuda::VecF::load(data+i)).store(data+i);
        for (; i < n; ++i) data[i] += val * t1[i] / t2[i];
        return self;
    }

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "addcdiv_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        const scalar_t* t1 = tensor1.data_ptr<scalar_t>();
        const scalar_t* t2 = tensor2.data_ptr<scalar_t>();
        scalar_t val = value.to<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) data[i] += val * t1[i] / t2[i];
    });

    return self;
}

// addcdiv: self + value * tensor1 / tensor2 (non-inplace)
inline Tensor addcdiv(const Tensor& self, const Tensor& tensor1, const Tensor& tensor2, Scalar value = 1) {
    Tensor result = self.clone();
    return addcdiv_(result, tensor1, tensor2, value);
}

// Element-wise maximum
inline Tensor maximum(const Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.sizes() == other.sizes(), "maximum requires same shapes");
    Tensor result = empty_like(self);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "maximum", [&] {
        const scalar_t* a = self.data_ptr<scalar_t>();
        const scalar_t* b = other.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            out[i] = a[i] > b[i] ? a[i] : b[i];
        }
    });

    return result;
}

// Element-wise minimum
inline Tensor minimum(const Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.sizes() == other.sizes(), "minimum requires same shapes");
    Tensor result = empty_like(self);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "minimum", [&] {
        const scalar_t* a = self.data_ptr<scalar_t>();
        const scalar_t* b = other.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            out[i] = a[i] < b[i] ? a[i] : b[i];
        }
    });

    return result;
}

// ============================================================================
// Comparison Operations
// ============================================================================

#define DEFINE_COMPARISON_OP(name, op) \
inline Tensor name(const Tensor& self, const Tensor& other) { \
    auto result_shape = detail::broadcast_shapes(self.sizes(), other.sizes()); \
    Tensor result = empty(result_shape, TensorOptions().dtype(c10::ScalarType::Bool).device(self.device())); \
    \
    /* Ensure contiguous for sequential data[i] access */ \
    Tensor self_c = self.is_contiguous() ? self : self.contiguous(); \
    Tensor other_c = other.is_contiguous() ? other : other.contiguous(); \
    PT_DISPATCH_ALL_TYPES(self_c.dtype(), #name, [&] { \
        bool* out = result.mutable_data_ptr<bool>(); \
        const scalar_t* a = self_c.data_ptr<scalar_t>(); \
        const scalar_t* b = other_c.data_ptr<scalar_t>(); \
        int64_t n = result.numel(); \
        \
        if (self_c.sizes() == other_c.sizes()) { \
            for (int64_t i = 0; i < n; ++i) { \
                out[i] = a[i] op b[i]; \
            } \
        } else { \
            for (int64_t i = 0; i < n; ++i) { \
                int64_t idx_a = detail::broadcast_index(i, result.sizes(), self_c.sizes(), self_c.strides()); \
                int64_t idx_b = detail::broadcast_index(i, result.sizes(), other_c.sizes(), other_c.strides()); \
                out[i] = a[idx_a] op b[idx_b]; \
            } \
        } \
    }); \
    return result; \
} \
\
inline Tensor name(const Tensor& self, Scalar other) { \
    Tensor result = empty(self.sizes(), TensorOptions().dtype(c10::ScalarType::Bool).device(self.device())); \
    \
    /* Ensure contiguous for sequential data[i] access */ \
    Tensor self_c = self.is_contiguous() ? self : self.contiguous(); \
    PT_DISPATCH_ALL_TYPES(self_c.dtype(), #name "_scalar", [&] { \
        bool* out = result.mutable_data_ptr<bool>(); \
        const scalar_t* a = self_c.data_ptr<scalar_t>(); \
        scalar_t val = other.to<scalar_t>(); \
        int64_t n = self_c.numel(); \
        \
        for (int64_t i = 0; i < n; ++i) { \
            out[i] = a[i] op val; \
        } \
    }); \
    return result; \
}

DEFINE_COMPARISON_OP(eq, ==)
DEFINE_COMPARISON_OP(ne, !=)
DEFINE_COMPARISON_OP(lt, <)
DEFINE_COMPARISON_OP(le, <=)
DEFINE_COMPARISON_OP(gt, >)
DEFINE_COMPARISON_OP(ge, >=)

#undef DEFINE_COMPARISON_OP

// ============================================================================
// Modular Arithmetic Operations
// ============================================================================

inline Tensor fmod(const Tensor& self, const Tensor& other) {
    Tensor a = self.contiguous();
    Tensor b = other.contiguous();
    Tensor result = empty(a.sizes(), TensorOptions().dtype(a.dtype()).device(a.device()));
    PT_DISPATCH_FLOATING_TYPES(a.dtype(), "fmod", [&] {
        const scalar_t* ap = a.data_ptr<scalar_t>();
        const scalar_t* bp = b.data_ptr<scalar_t>();
        scalar_t* rp = result.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < a.numel(); ++i) {
            rp[i] = std::fmod(ap[i], bp[i]);
        }
    });
    return result;
}

inline Tensor remainder(const Tensor& self, const Tensor& other) {
    Tensor a = self.contiguous();
    Tensor b = other.contiguous();
    Tensor result = empty(a.sizes(), TensorOptions().dtype(a.dtype()).device(a.device()));
    PT_DISPATCH_FLOATING_TYPES(a.dtype(), "remainder", [&] {
        const scalar_t* ap = a.data_ptr<scalar_t>();
        const scalar_t* bp = b.data_ptr<scalar_t>();
        scalar_t* rp = result.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < a.numel(); ++i) {
            rp[i] = ap[i] - std::floor(ap[i] / bp[i]) * bp[i];
        }
    });
    return result;
}

} // namespace native
} // namespace at
