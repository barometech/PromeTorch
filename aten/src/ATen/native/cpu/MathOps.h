#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "c10/core/ScalarType.h"
#include <cmath>
#include <algorithm>

// OpenMP conditional support
#ifdef _OPENMP
#include <omp.h>
#define PT_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define PT_OMP_PARALLEL_FOR_IF(cond) _Pragma("omp parallel for if(" #cond ")")
#else
#define PT_OMP_PARALLEL_FOR
#define PT_OMP_PARALLEL_FOR_IF(cond)
#endif

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
// Unary Operations Implementation
// ============================================================================

#define DEFINE_UNARY_OP(name, op) \
inline Tensor name(const Tensor& self) { \
    Tensor input = self.is_contiguous() ? self : self.contiguous(); \
    Tensor result = empty_like(input); \
    PT_DISPATCH_FLOATING_TYPES(input.dtype(), #name, [&] { \
        const scalar_t* in = input.data_ptr<scalar_t>(); \
        scalar_t* out = result.mutable_data_ptr<scalar_t>(); \
        int64_t n = input.numel(); \
        _Pragma("omp parallel for if(n > 10000)") \
        for (int64_t i = 0; i < n; ++i) { \
            out[i] = op; \
        } \
    }); \
    return result; \
}

#define DEFINE_UNARY_OP_INPLACE(name, op) \
inline Tensor& name##_(Tensor& self) { \
    if (!self.is_contiguous()) { \
        Tensor tmp = name(self); \
        self.copy_(tmp); \
        return self; \
    } \
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), #name "_", [&] { \
        scalar_t* data = self.mutable_data_ptr<scalar_t>(); \
        int64_t n = self.numel(); \
        _Pragma("omp parallel for if(n > 10000)") \
        for (int64_t i = 0; i < n; ++i) { \
            data[i] = op; \
        } \
    }); \
    return self; \
}

DEFINE_UNARY_OP(neg, -in[i])
DEFINE_UNARY_OP(abs, std::abs(in[i]))
DEFINE_UNARY_OP(sqrt, std::sqrt(in[i]))
DEFINE_UNARY_OP(rsqrt, static_cast<scalar_t>(1) / std::sqrt(in[i]))
DEFINE_UNARY_OP(square, in[i] * in[i])
DEFINE_UNARY_OP(exp, std::exp(in[i]))
DEFINE_UNARY_OP(log, std::log(in[i]))
DEFINE_UNARY_OP(log2, std::log2(in[i]))
DEFINE_UNARY_OP(log10, std::log10(in[i]))
DEFINE_UNARY_OP(sin, std::sin(in[i]))
DEFINE_UNARY_OP(cos, std::cos(in[i]))
DEFINE_UNARY_OP(tan, std::tan(in[i]))
DEFINE_UNARY_OP(tanh, std::tanh(in[i]))
DEFINE_UNARY_OP(sigmoid, static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp(-in[i])))
DEFINE_UNARY_OP(relu, in[i] > 0 ? in[i] : static_cast<scalar_t>(0))
DEFINE_UNARY_OP(ceil, std::ceil(in[i]))
DEFINE_UNARY_OP(floor, std::floor(in[i]))
DEFINE_UNARY_OP(round, std::round(in[i]))
DEFINE_UNARY_OP(sign, (in[i] > 0) ? static_cast<scalar_t>(1) : ((in[i] < 0) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(0)))
DEFINE_UNARY_OP(reciprocal, static_cast<scalar_t>(1) / in[i])

DEFINE_UNARY_OP_INPLACE(neg, -data[i])
DEFINE_UNARY_OP_INPLACE(abs, std::abs(data[i]))
DEFINE_UNARY_OP_INPLACE(sqrt, std::sqrt(data[i]))
DEFINE_UNARY_OP_INPLACE(exp, std::exp(data[i]))
DEFINE_UNARY_OP_INPLACE(log, std::log(data[i]))
DEFINE_UNARY_OP_INPLACE(sin, std::sin(data[i]))
DEFINE_UNARY_OP_INPLACE(cos, std::cos(data[i]))
DEFINE_UNARY_OP_INPLACE(tanh, std::tanh(data[i]))
DEFINE_UNARY_OP_INPLACE(sigmoid, static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp(-data[i])))
DEFINE_UNARY_OP_INPLACE(relu, data[i] > 0 ? data[i] : static_cast<scalar_t>(0))
DEFINE_UNARY_OP_INPLACE(ceil, std::ceil(data[i]))
DEFINE_UNARY_OP_INPLACE(floor, std::floor(data[i]))
DEFINE_UNARY_OP_INPLACE(round, std::round(data[i]))

#undef DEFINE_UNARY_OP
#undef DEFINE_UNARY_OP_INPLACE

// Zero and fill
inline Tensor& zero_(Tensor& self) {
    if (self.is_contiguous()) {
        std::memset(self.data_ptr(), 0, self.nbytes());
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
    PT_DISPATCH_ALL_TYPES(self.dtype(), "fill_", [&] {
        scalar_t val = value.to<scalar_t>();
        if (self.is_contiguous()) {
            scalar_t* data = self.mutable_data_ptr<scalar_t>();
            int64_t numel = self.numel();
            for (int64_t i = 0; i < numel; ++i) {
                data[i] = val;
            }
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
// Binary Operations Implementation
// ============================================================================

// Tensor + Tensor with broadcasting
inline Tensor add(const Tensor& self, const Tensor& other, Scalar alpha = 1) {
    auto result_shape = detail::broadcast_shapes(self.sizes(), other.sizes());
    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());

    Tensor result = empty(result_shape, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_ALL_TYPES(result_dtype, "add", [&] {
        scalar_t alpha_val = alpha.to<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t n = result.numel();

        // Simple case: same shape, contiguous
        if (self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
            const scalar_t* a = self.data_ptr<scalar_t>();
            const scalar_t* b = other.data_ptr<scalar_t>();

            for (int64_t i = 0; i < n; ++i) {
                out[i] = a[i] + alpha_val * b[i];
            }
        } else {
            // Broadcasting case - simplified implementation
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
    return add(self, other, Scalar(-alpha.toDouble()));
}

inline Tensor mul(const Tensor& self, const Tensor& other) {
    auto result_shape = detail::broadcast_shapes(self.sizes(), other.sizes());
    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());

    Tensor result = empty(result_shape, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_ALL_TYPES(result_dtype, "mul", [&] {
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t n = result.numel();

        if (self.sizes() == other.sizes() && self.is_contiguous() && other.is_contiguous()) {
            const scalar_t* a = self.data_ptr<scalar_t>();
            const scalar_t* b = other.data_ptr<scalar_t>();

            for (int64_t i = 0; i < n; ++i) {
                out[i] = a[i] * b[i];
            }
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
    auto result_shape = detail::broadcast_shapes(self.sizes(), other.sizes());
    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());

    // Division always promotes to float
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

            for (int64_t i = 0; i < n; ++i) {
                out[i] = a[i] / b[i];
            }
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
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);
    double val = other.toDouble() * alpha.toDouble();

    PT_DISPATCH_ALL_TYPES(input.dtype(), "add_scalar", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        scalar_t scalar_val = static_cast<scalar_t>(val);
        int64_t n = input.numel();

        for (int64_t i = 0; i < n; ++i) {
            out[i] = in[i] + scalar_val;
        }
    });

    return result;
}

inline Tensor sub(const Tensor& self, Scalar other, Scalar alpha = 1) {
    return add(self, Scalar(-other.toDouble()), alpha);
}

inline Tensor mul(const Tensor& self, Scalar other) {
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);

    PT_DISPATCH_ALL_TYPES(input.dtype(), "mul_scalar", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        scalar_t scalar_val = other.to<scalar_t>();
        int64_t n = input.numel();

        for (int64_t i = 0; i < n; ++i) {
            out[i] = in[i] * scalar_val;
        }
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
    PT_CHECK_MSG(self.sizes() == other.sizes(), "In-place operation requires same shapes");

    PT_DISPATCH_ALL_TYPES(self.dtype(), "add_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        const scalar_t* other_data = other.data_ptr<scalar_t>();
        scalar_t alpha_val = alpha.to<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] += alpha_val * other_data[i];
        }
    });

    return self;
}

inline Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha = 1) {
    return add_(self, other, Scalar(-alpha.toDouble()));
}

inline Tensor& mul_(Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.sizes() == other.sizes(), "In-place operation requires same shapes");

    PT_DISPATCH_ALL_TYPES(self.dtype(), "mul_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        const scalar_t* other_data = other.data_ptr<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] *= other_data[i];
        }
    });

    return self;
}

inline Tensor& div_(Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.sizes() == other.sizes(), "In-place operation requires same shapes");

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "div_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        const scalar_t* other_data = other.data_ptr<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] /= other_data[i];
        }
    });

    return self;
}

inline Tensor& add_(Tensor& self, Scalar other, Scalar alpha = 1) {
    PT_DISPATCH_ALL_TYPES(self.dtype(), "add_scalar_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        scalar_t val = static_cast<scalar_t>(other.toDouble() * alpha.toDouble());
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] += val;
        }
    });

    return self;
}

inline Tensor& sub_(Tensor& self, Scalar other, Scalar alpha = 1) {
    return add_(self, Scalar(-other.toDouble()), alpha);
}

inline Tensor& mul_(Tensor& self, Scalar other) {
    PT_DISPATCH_ALL_TYPES(self.dtype(), "mul_scalar_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        scalar_t val = other.to<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] *= val;
        }
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

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "addcmul_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        const scalar_t* t1 = tensor1.data_ptr<scalar_t>();
        const scalar_t* t2 = tensor2.data_ptr<scalar_t>();
        scalar_t val = value.to<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] += val * t1[i] * t2[i];
        }
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

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "addcdiv_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        const scalar_t* t1 = tensor1.data_ptr<scalar_t>();
        const scalar_t* t2 = tensor2.data_ptr<scalar_t>();
        scalar_t val = value.to<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] += val * t1[i] / t2[i];
        }
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
    PT_DISPATCH_ALL_TYPES(self.dtype(), #name, [&] { \
        bool* out = result.mutable_data_ptr<bool>(); \
        const scalar_t* a = self.data_ptr<scalar_t>(); \
        const scalar_t* b = other.data_ptr<scalar_t>(); \
        int64_t n = result.numel(); \
        \
        if (self.sizes() == other.sizes()) { \
            for (int64_t i = 0; i < n; ++i) { \
                out[i] = a[i] op b[i]; \
            } \
        } else { \
            for (int64_t i = 0; i < n; ++i) { \
                int64_t idx_a = detail::broadcast_index(i, result.sizes(), self.sizes(), self.strides()); \
                int64_t idx_b = detail::broadcast_index(i, result.sizes(), other.sizes(), other.strides()); \
                out[i] = a[idx_a] op b[idx_b]; \
            } \
        } \
    }); \
    return result; \
} \
\
inline Tensor name(const Tensor& self, Scalar other) { \
    Tensor result = empty_like(self); \
    result = empty(self.sizes(), TensorOptions().dtype(c10::ScalarType::Bool).device(self.device())); \
    \
    PT_DISPATCH_ALL_TYPES(self.dtype(), #name "_scalar", [&] { \
        bool* out = result.mutable_data_ptr<bool>(); \
        const scalar_t* a = self.data_ptr<scalar_t>(); \
        scalar_t val = other.to<scalar_t>(); \
        int64_t n = self.numel(); \
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

} // namespace native
} // namespace at
