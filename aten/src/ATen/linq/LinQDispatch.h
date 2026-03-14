#pragma once
// ============================================================================
// LinQDispatch.h — Tensor-level dispatch for LinQ H1M accelerator
// ============================================================================
// Provides: empty_linq(), to_linq(), linq_to_cpu()
// Plus tensor-level operation wrappers for the dispatch layer
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "c10/linq/LinQAllocator.h"
#include "aten/src/ATen/linq/LinQOps.h"

#include <cstring>

namespace at {

// ============================================================================
// Device transfer operations
// ============================================================================

inline Tensor empty_linq(c10::IntArrayRef sizes,
                         c10::ScalarType dtype = c10::ScalarType::Float,
                         int device = 0) {
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;
    size_t nbytes = numel * c10::elementSize(dtype);

    auto& allocator = c10::linq::LinQAllocator::get();
    c10::DataPtr data_ptr = allocator.allocate(nbytes);

    auto* storage_impl = new c10::StorageImpl(
        nbytes, std::move(data_ptr), &allocator, false);
    c10::Storage storage(storage_impl);

    std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());
    auto impl = std::make_shared<c10::TensorImpl>(storage, dtype, sizes_vec);

    return Tensor(impl);
}

// Copy CPU tensor to LinQ device
inline Tensor to_linq(const Tensor& src, int device = 0) {
    if (src.device().is_linq()) return src;

    Tensor contig = src.is_contiguous() ? src : src.contiguous();
    auto dst = empty_linq(contig.sizes().vec(), contig.dtype(), device);

    std::memcpy(dst.data_ptr(), contig.data_ptr(), contig.nbytes());

    // Preserve autograd
    auto* src_meta = contig.autograd_meta();
    if (src_meta && src_meta->requires_grad_) {
        dst.set_requires_grad(true);
    }

    return dst;
}

// Copy LinQ tensor to CPU
inline Tensor linq_to_cpu(const Tensor& src) {
    if (src.device().is_cpu()) return src;

    Tensor contig = src.is_contiguous() ? src : src.contiguous();
    auto dst = empty(contig.sizes().vec(),
                     TensorOptions().dtype(contig.dtype()));

    std::memcpy(dst.data_ptr(), contig.data_ptr(), contig.nbytes());

    return dst;
}

// ============================================================================
// LinQ Tensor operations namespace
// ============================================================================

namespace linq_dispatch {

// Helper: ensure contiguous on LinQ device
inline Tensor ensure_contiguous(const Tensor& t) {
    if (t.is_contiguous()) return t;

    auto result = empty_linq(t.sizes().vec(), t.dtype());
    const float* src = t.data_ptr<float>();
    float* dst = result.mutable_data_ptr<float>();
    int64_t n = t.numel();
    int64_t ndim = t.dim();
    auto sz = t.sizes();
    auto st = t.strides();

    for (int64_t flat = 0; flat < n; ++flat) {
        int64_t src_offset = 0;
        int64_t remainder = flat;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t idx = remainder % sz[d];
            remainder /= sz[d];
            src_offset += idx * st[d];
        }
        dst[flat] = src[src_offset];
    }
    return result;
}

// ============================================================================
// Unary operations
// ============================================================================

#define LINQ_UNARY_OP(name, launch_fn) \
inline Tensor name(const Tensor& input) { \
    Tensor ic = ensure_contiguous(input); \
    auto output = empty_linq(ic.sizes().vec(), ic.dtype()); \
    launch_fn(ic.data_ptr<float>(), \
              output.mutable_data_ptr<float>(), ic.numel()); \
    return output; \
}

LINQ_UNARY_OP(neg, linq_ops::launch_neg)
LINQ_UNARY_OP(abs, linq_ops::launch_abs)
LINQ_UNARY_OP(sqrt, linq_ops::launch_sqrt)
LINQ_UNARY_OP(exp, linq_ops::launch_exp)
LINQ_UNARY_OP(log, linq_ops::launch_log)
LINQ_UNARY_OP(relu, linq_ops::launch_relu)
LINQ_UNARY_OP(sigmoid, linq_ops::launch_sigmoid)
LINQ_UNARY_OP(tanh, linq_ops::launch_tanh)
LINQ_UNARY_OP(silu, linq_ops::launch_silu)
LINQ_UNARY_OP(gelu, linq_ops::launch_gelu)
LINQ_UNARY_OP(sin, linq_ops::launch_sin)
LINQ_UNARY_OP(cos, linq_ops::launch_cos)
LINQ_UNARY_OP(tan, linq_ops::launch_tan)
LINQ_UNARY_OP(log2, linq_ops::launch_log2)
LINQ_UNARY_OP(log10, linq_ops::launch_log10)
LINQ_UNARY_OP(rsqrt, linq_ops::launch_rsqrt)
LINQ_UNARY_OP(square, linq_ops::launch_square)
LINQ_UNARY_OP(reciprocal, linq_ops::launch_reciprocal)
LINQ_UNARY_OP(ceil, linq_ops::launch_ceil)
LINQ_UNARY_OP(floor, linq_ops::launch_floor)
LINQ_UNARY_OP(round, linq_ops::launch_round)
LINQ_UNARY_OP(sign, linq_ops::launch_sign)

#undef LINQ_UNARY_OP

inline Tensor leaky_relu(const Tensor& input, float alpha = 0.01f) {
    Tensor ic = ensure_contiguous(input);
    auto output = empty_linq(ic.sizes().vec(), ic.dtype());
    linq_ops::launch_leaky_relu(ic.data_ptr<float>(),
                                output.mutable_data_ptr<float>(),
                                ic.numel(), alpha);
    return output;
}

// ============================================================================
// Binary operations
// ============================================================================

#define LINQ_BINARY_OP(name, launch_fn) \
inline Tensor name(const Tensor& a, const Tensor& b) { \
    Tensor ac = ensure_contiguous(a); \
    Tensor bc = ensure_contiguous(b); \
    auto output = empty_linq(ac.sizes().vec(), ac.dtype()); \
    launch_fn(ac.data_ptr<float>(), bc.data_ptr<float>(), \
              output.mutable_data_ptr<float>(), ac.numel()); \
    return output; \
}

LINQ_BINARY_OP(add, linq_ops::launch_add)
LINQ_BINARY_OP(sub, linq_ops::launch_sub)
LINQ_BINARY_OP(mul, linq_ops::launch_mul)
LINQ_BINARY_OP(div, linq_ops::launch_div)
LINQ_BINARY_OP(maximum, linq_ops::launch_maximum)
LINQ_BINARY_OP(minimum, linq_ops::launch_minimum)

#undef LINQ_BINARY_OP

// ============================================================================
// Comparison operations
// ============================================================================

#define LINQ_CMP_OP(name, launch_fn) \
inline Tensor name(const Tensor& a, const Tensor& b) { \
    Tensor ac = ensure_contiguous(a); \
    Tensor bc = ensure_contiguous(b); \
    auto output = empty_linq(ac.sizes().vec(), ac.dtype()); \
    launch_fn(ac.data_ptr<float>(), bc.data_ptr<float>(), \
              output.mutable_data_ptr<float>(), ac.numel()); \
    return output; \
}

LINQ_CMP_OP(eq, linq_ops::launch_eq)
LINQ_CMP_OP(ne, linq_ops::launch_ne)
LINQ_CMP_OP(lt, linq_ops::launch_lt)
LINQ_CMP_OP(le, linq_ops::launch_le)
LINQ_CMP_OP(gt, linq_ops::launch_gt)
LINQ_CMP_OP(ge, linq_ops::launch_ge)

#undef LINQ_CMP_OP

#define LINQ_CMP_SCALAR(name, launch_fn) \
inline Tensor name##_scalar(const Tensor& a, float scalar) { \
    Tensor ac = ensure_contiguous(a); \
    auto output = empty_linq(ac.sizes().vec(), ac.dtype()); \
    launch_fn(ac.data_ptr<float>(), \
              output.mutable_data_ptr<float>(), ac.numel(), scalar); \
    return output; \
}

LINQ_CMP_SCALAR(eq, linq_ops::launch_eq_scalar)
LINQ_CMP_SCALAR(ne, linq_ops::launch_ne_scalar)
LINQ_CMP_SCALAR(lt, linq_ops::launch_lt_scalar)
LINQ_CMP_SCALAR(le, linq_ops::launch_le_scalar)
LINQ_CMP_SCALAR(gt, linq_ops::launch_gt_scalar)
LINQ_CMP_SCALAR(ge, linq_ops::launch_ge_scalar)

#undef LINQ_CMP_SCALAR

// ============================================================================
// Scalar binary operations
// ============================================================================

inline Tensor add_scalar(const Tensor& a, float scalar) {
    Tensor ac = ensure_contiguous(a);
    auto output = empty_linq(ac.sizes().vec(), ac.dtype());
    linq_ops::launch_add_scalar(ac.data_ptr<float>(),
                                output.mutable_data_ptr<float>(),
                                ac.numel(), scalar);
    return output;
}

inline Tensor mul_scalar(const Tensor& a, float scalar) {
    Tensor ac = ensure_contiguous(a);
    auto output = empty_linq(ac.sizes().vec(), ac.dtype());
    linq_ops::launch_mul_scalar(ac.data_ptr<float>(),
                                output.mutable_data_ptr<float>(),
                                ac.numel(), scalar);
    return output;
}

inline Tensor pow_scalar(const Tensor& a, float p) {
    Tensor ac = ensure_contiguous(a);
    auto output = empty_linq(ac.sizes().vec(), ac.dtype());
    linq_ops::launch_pow_scalar(ac.data_ptr<float>(),
                                output.mutable_data_ptr<float>(),
                                ac.numel(), p);
    return output;
}

// ============================================================================
// Clamp
// ============================================================================

inline Tensor clamp(const Tensor& input, float lo, float hi) {
    Tensor ic = ensure_contiguous(input);
    auto output = empty_linq(ic.sizes().vec(), ic.dtype());
    linq_ops::launch_clamp(ic.data_ptr<float>(),
                           output.mutable_data_ptr<float>(),
                           ic.numel(), lo, hi);
    return output;
}

// ============================================================================
// Matrix multiplication
// ============================================================================

inline Tensor mm(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous(a);
    Tensor bc = ensure_contiguous(b);
    int64_t M = ac.size(0), K = ac.size(1), N = bc.size(1);
    auto output = empty_linq({M, N}, ac.dtype());
    linq_ops::launch_matmul(ac.data_ptr<float>(), bc.data_ptr<float>(),
                            output.mutable_data_ptr<float>(), M, K, N);
    return output;
}

inline Tensor mv(const Tensor& mat, const Tensor& vec) {
    Tensor mc = ensure_contiguous(mat);
    Tensor vc = ensure_contiguous(vec);
    int64_t M = mc.size(0), N = mc.size(1);
    auto output = empty_linq({M}, mc.dtype());
    linq_ops::launch_matvec(mc.data_ptr<float>(), vc.data_ptr<float>(),
                            output.mutable_data_ptr<float>(), M, N);
    return output;
}

inline Tensor dot(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous(a);
    Tensor bc = ensure_contiguous(b);
    float result = linq_ops::launch_dot(ac.data_ptr<float>(),
                                        bc.data_ptr<float>(), ac.numel());
    auto output = empty_linq({}, ac.dtype()); // scalar
    *output.mutable_data_ptr<float>() = result;
    return output;
}

// ============================================================================
// Reductions
// ============================================================================

inline Tensor sum(const Tensor& input) {
    Tensor ic = ensure_contiguous(input);
    float result = linq_ops::launch_sum(ic.data_ptr<float>(), ic.numel());
    auto output = empty_linq({}, ic.dtype());
    *output.mutable_data_ptr<float>() = result;
    return output;
}

inline Tensor mean(const Tensor& input) {
    Tensor ic = ensure_contiguous(input);
    float s = linq_ops::launch_sum(ic.data_ptr<float>(), ic.numel());
    auto output = empty_linq({}, ic.dtype());
    *output.mutable_data_ptr<float>() = s / static_cast<float>(ic.numel());
    return output;
}

inline Tensor argmax(const Tensor& input) {
    Tensor ic = ensure_contiguous(input);
    int64_t idx = linq_ops::launch_argmax(ic.data_ptr<float>(), ic.numel());
    auto output = empty_linq({}, c10::ScalarType::Float);
    *output.mutable_data_ptr<float>() = static_cast<float>(idx);
    return output;
}

inline Tensor argmin(const Tensor& input) {
    Tensor ic = ensure_contiguous(input);
    int64_t idx = linq_ops::launch_argmin(ic.data_ptr<float>(), ic.numel());
    auto output = empty_linq({}, c10::ScalarType::Float);
    *output.mutable_data_ptr<float>() = static_cast<float>(idx);
    return output;
}

// ============================================================================
// Softmax
// ============================================================================

inline Tensor softmax(const Tensor& input, int64_t dim) {
    Tensor ic = ensure_contiguous(input);
    auto output = empty_linq(ic.sizes().vec(), ic.dtype());

    int64_t outer = 1, inner = ic.size(dim), trailing = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= ic.size(d);
    for (int64_t d = dim + 1; d < ic.dim(); ++d) trailing *= ic.size(d);

    const float* in = ic.data_ptr<float>();
    float* out = output.mutable_data_ptr<float>();

    if (trailing == 1) {
        for (int64_t o = 0; o < outer; ++o) {
            linq_ops::launch_softmax(in + o * inner, out + o * inner, inner);
        }
    } else {
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t t = 0; t < trailing; ++t) {
                std::vector<float> slice(inner);
                for (int64_t i = 0; i < inner; ++i)
                    slice[i] = in[(o * inner + i) * trailing + t];
                std::vector<float> result(inner);
                linq_ops::launch_softmax(slice.data(), result.data(), inner);
                for (int64_t i = 0; i < inner; ++i)
                    out[(o * inner + i) * trailing + t] = result[i];
            }
        }
    }

    return output;
}

// ============================================================================
// In-place / memory operations
// ============================================================================

inline void fill_(Tensor& t, float val) {
    Tensor tc = ensure_contiguous(t);
    linq_ops::launch_fill(tc.mutable_data_ptr<float>(), tc.numel(), val);
}

inline void zero_(Tensor& t) {
    fill_(t, 0.0f);
}

inline void copy_(Tensor& dst, const Tensor& src) {
    Tensor sc = ensure_contiguous(src);
    linq_ops::launch_copy(sc.data_ptr<float>(), dst.mutable_data_ptr<float>(),
                          sc.numel());
}

// ============================================================================
// Fused ops
// ============================================================================

inline Tensor addcmul(const Tensor& self, const Tensor& t1, const Tensor& t2, float value) {
    Tensor sc = ensure_contiguous(self);
    Tensor t1c = ensure_contiguous(t1);
    Tensor t2c = ensure_contiguous(t2);
    auto output = empty_linq(sc.sizes().vec(), sc.dtype());
    linq_ops::launch_addcmul(sc.data_ptr<float>(), t1c.data_ptr<float>(),
                             t2c.data_ptr<float>(), output.mutable_data_ptr<float>(),
                             sc.numel(), value);
    return output;
}

inline Tensor addcdiv(const Tensor& self, const Tensor& t1, const Tensor& t2, float value) {
    Tensor sc = ensure_contiguous(self);
    Tensor t1c = ensure_contiguous(t1);
    Tensor t2c = ensure_contiguous(t2);
    auto output = empty_linq(sc.sizes().vec(), sc.dtype());
    linq_ops::launch_addcdiv(sc.data_ptr<float>(), t1c.data_ptr<float>(),
                             t2c.data_ptr<float>(), output.mutable_data_ptr<float>(),
                             sc.numel(), value);
    return output;
}

} // namespace linq_dispatch

} // namespace at
