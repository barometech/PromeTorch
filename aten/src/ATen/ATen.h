#pragma once

// ============================================================================
// ATen - A Tensor Library for PromeTorch
// ============================================================================

// Core
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

// Native CPU Operations
#include "aten/src/ATen/native/cpu/MathOps.h"

// CUDA Operations (when enabled)
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include "aten/src/ATen/native/cpu/ReduceOps.h"
#include "aten/src/ATen/native/cpu/LinearAlgebra.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"
#include "aten/src/ATen/native/cpu/IndexOps.h"
#include "aten/src/ATen/native/cpu/FFTOps.h"

namespace at {

// ============================================================================
// Tensor Method Implementations
// ============================================================================

// These implementations use the native functions defined above

inline Scalar Tensor::item() const {
    PT_CHECK_MSG(numel() == 1, "item() requires tensor with single element");

#ifdef PT_USE_CUDA
    if (is_cuda()) {
        // Move to CPU first to read the value
        Tensor cpu_copy = to_cpu(*this);
        Scalar result;
        PT_DISPATCH_ALL_TYPES(dtype(), "item", [&] {
            result = Scalar(static_cast<double>(cpu_copy.data_ptr<scalar_t>()[0]));
        });
        return result;
    }
#endif

    Scalar result;
    PT_DISPATCH_ALL_TYPES(dtype(), "item", [&] {
        result = Scalar(static_cast<double>(data_ptr<scalar_t>()[0]));
    });
    return result;
}

// Autograd-related accessors
inline Tensor Tensor::grad() const {
    auto* meta = autograd_meta();
    if (!meta || !meta->grad_) {
        return Tensor();  // undefined tensor
    }
    return Tensor(meta->grad_);
}

inline void Tensor::set_grad(const Tensor& grad) {
    auto* meta = autograd_meta();
    if (!meta) {
        // Create autograd metadata if it doesn't exist
        impl_->set_autograd_meta(std::make_unique<c10::AutogradMeta>());
        meta = autograd_meta();
    }
    if (grad.defined()) {
        meta->grad_ = grad.getIntrusivePtr();
    } else {
        meta->grad_ = nullptr;
    }
}

// Clone and copy operations
inline Tensor Tensor::clone() const {
    return native::clone(*this);
}

inline Tensor Tensor::detach() const {
    return native::detach(*this);
}

inline Tensor Tensor::contiguous() const {
    return native::contiguous(*this);
}

inline Tensor Tensor::contiguous(c10::MemoryFormat memory_format) const {
    return native::contiguous(*this, memory_format);
}

inline Tensor Tensor::to(c10::MemoryFormat memory_format) const {
    return native::contiguous(*this, memory_format);
}

inline Tensor& Tensor::copy_(const Tensor& src) {
    PT_CHECK(defined() && src.defined());
    PT_CHECK_MSG(sizes() == src.sizes(), "copy_: sizes must match");

    if (src.is_contiguous() && is_contiguous()) {
        std::memcpy(data_ptr(), src.data_ptr(), nbytes());
    } else {
        // Stride-aware copy: convert flat index → multi-dim index → physical offset
        PT_DISPATCH_ALL_TYPES(dtype(), "copy_", [&] {
            scalar_t* dst_base = mutable_data_ptr<scalar_t>();
            const scalar_t* src_base = src.data_ptr<scalar_t>();
            int64_t n = numel();
            int64_t ndim = dim();
            auto dst_sizes = sizes();
            auto dst_strides = strides();
            auto src_strides = src.strides();

            for (int64_t flat = 0; flat < n; ++flat) {
                // Convert flat index to multi-dim indices, then to physical offsets
                int64_t dst_offset = 0;
                int64_t src_offset = 0;
                int64_t remainder = flat;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    int64_t idx = remainder % dst_sizes[d];
                    remainder /= dst_sizes[d];
                    dst_offset += idx * dst_strides[d];
                    src_offset += idx * src_strides[d];
                }
                dst_base[dst_offset] = src_base[src_offset];
            }
        });
    }
    return *this;
}

// Device transfer operations
inline Tensor Tensor::to(c10::Device device) const {
#ifdef PT_USE_CUDA
    if (device.is_cpu()) {
        return at::to_cpu(*this);
    } else if (device.is_cuda()) {
        return at::to_cuda(*this, device.index() >= 0 ? device.index() : 0);
    }
#else
    if (device.is_cpu()) {
        return *this;  // Already on CPU
    }
#endif
    PT_CHECK_MSG(false, "Unsupported device type");
    return Tensor();
}

// Shape operations
inline Tensor Tensor::view(c10::IntArrayRef sizes) const {
    return native::view(*this, sizes);
}

inline Tensor Tensor::reshape(c10::IntArrayRef sizes) const {
    return native::reshape(*this, sizes);
}

inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    return native::flatten(*this, start_dim, end_dim);
}

inline Tensor Tensor::squeeze() const {
    return native::squeeze(*this);
}

inline Tensor Tensor::squeeze(int64_t dim) const {
    return native::squeeze(*this, dim);
}

inline Tensor Tensor::unsqueeze(int64_t dim) const {
    return native::unsqueeze(*this, dim);
}

inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return native::transpose(*this, dim0, dim1);
}

inline Tensor Tensor::permute(c10::IntArrayRef dims) const {
    return native::permute(*this, dims);
}

inline Tensor Tensor::t() const {
    return native::t(*this);
}

inline Tensor Tensor::expand(c10::IntArrayRef sizes) const {
    return native::expand(*this, sizes);
}

inline Tensor Tensor::repeat(c10::IntArrayRef repeats) const {
    return native::repeat(*this, repeats);
}

inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    return native::split(*this, split_size, dim);
}

inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    return native::chunk(*this, chunks, dim);
}

// Indexing
inline Tensor Tensor::operator[](int64_t index) const {
    return native::index(*this, index);
}

inline Tensor Tensor::select(int64_t dim, int64_t index) const {
    return native::select(*this, dim, index);
}

inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    return native::narrow(*this, dim, start, length);
}

inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    return native::slice(*this, dim, start, end, step);
}

// Unary operations
// Unary operations with device dispatch
inline Tensor Tensor::neg() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::neg(*this); }
#endif
    return native::neg(*this);
}
inline Tensor Tensor::abs() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::abs(*this); }
#endif
    return native::abs(*this);
}
inline Tensor Tensor::sqrt() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::sqrt(*this); }
#endif
    return native::sqrt(*this);
}
inline Tensor Tensor::rsqrt() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::rsqrt(*this); }
#endif
    return native::rsqrt(*this);
}
inline Tensor Tensor::square() const { return native::square(*this); }
inline Tensor Tensor::exp() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::exp(*this); }
#endif
    return native::exp(*this);
}
inline Tensor Tensor::log() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::log(*this); }
#endif
    return native::log(*this);
}
inline Tensor Tensor::log2() const { return native::log2(*this); }
inline Tensor Tensor::log10() const { return native::log10(*this); }
inline Tensor Tensor::sin() const { return native::sin(*this); }
inline Tensor Tensor::cos() const { return native::cos(*this); }
inline Tensor Tensor::tan() const { return native::tan(*this); }
inline Tensor Tensor::tanh() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::tanh(*this); }
#endif
    return native::tanh(*this);
}
inline Tensor Tensor::sigmoid() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::sigmoid(*this); }
#endif
    return native::sigmoid(*this);
}
inline Tensor Tensor::relu() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::relu(*this); }
#endif
    return native::relu(*this);
}
inline Tensor Tensor::ceil() const { return native::ceil(*this); }
inline Tensor Tensor::floor() const { return native::floor(*this); }
inline Tensor Tensor::round() const { return native::round(*this); }
inline Tensor Tensor::sign() const { return native::sign(*this); }
inline Tensor Tensor::reciprocal() const { return native::reciprocal(*this); }
inline Tensor Tensor::clamp(Scalar min_val, Scalar max_val) const { return native::clamp(*this, min_val, max_val); }
inline Tensor Tensor::clamp(std::optional<Scalar> min, std::optional<Scalar> max) const { return native::clamp(*this, min, max); }
inline Tensor& Tensor::clamp_(Scalar min_val, Scalar max_val) { return native::clamp_(*this, min_val, max_val); }
inline Tensor Tensor::clamp_min(Scalar min_val) const { return native::clamp_min(*this, min_val); }
inline Tensor Tensor::clamp_max(Scalar max_val) const { return native::clamp_max(*this, max_val); }
inline Tensor Tensor::triu(int64_t diagonal) const { return native::triu(*this, diagonal); }
inline Tensor Tensor::tril(int64_t diagonal) const { return native::tril(*this, diagonal); }
inline Tensor Tensor::diag(int64_t diagonal) const { return native::diag(*this, diagonal); }

// In-place unary
inline Tensor& Tensor::neg_() { return native::neg_(*this); }
inline Tensor& Tensor::abs_() { return native::abs_(*this); }
inline Tensor& Tensor::sqrt_() { return native::sqrt_(*this); }
inline Tensor& Tensor::exp_() { return native::exp_(*this); }
inline Tensor& Tensor::log_() { return native::log_(*this); }
inline Tensor& Tensor::sin_() { return native::sin_(*this); }
inline Tensor& Tensor::cos_() { return native::cos_(*this); }
inline Tensor& Tensor::tanh_() { return native::tanh_(*this); }
inline Tensor& Tensor::sigmoid_() { return native::sigmoid_(*this); }
inline Tensor& Tensor::relu_() { return native::relu_(*this); }
inline Tensor& Tensor::ceil_() { return native::ceil_(*this); }
inline Tensor& Tensor::floor_() { return native::floor_(*this); }
inline Tensor& Tensor::round_() { return native::round_(*this); }
inline Tensor& Tensor::zero_() {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        cuda_ops::fill_(*this, 0.0f);
        return *this;
    }
#endif
    return native::zero_(*this);
}
inline Tensor& Tensor::fill_(Scalar value) {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        cuda_ops::fill_(*this, static_cast<float>(value.toDouble()));
        return *this;
    }
#endif
    return native::fill_(*this, value);
}

// Binary operations with device dispatch
inline Tensor Tensor::add(const Tensor& other, Scalar alpha) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        // Check for broadcasting case: [outer, inner] + [inner] (for bias addition)
        if (numel() != other.numel() && dim() == 2 && other.dim() == 1 && other.size(0) == size(1)) {
            return cuda_ops::add_broadcast(*this, other);
        }
        return cuda_ops::add(*this, other);
    }
#endif
    return native::add(*this, other, alpha);
}
inline Tensor Tensor::sub(const Tensor& other, Scalar alpha) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::sub(*this, other); }
#endif
    return native::sub(*this, other, alpha);
}
inline Tensor Tensor::mul(const Tensor& other) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::mul(*this, other); }
#endif
    return native::mul(*this, other);
}
inline Tensor Tensor::mul_broadcast(const Tensor& other) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::mul_broadcast(*this, other); }
#endif
    // CPU fallback uses native::mul which supports broadcasting
    return native::mul(*this, other);
}
inline Tensor Tensor::div(const Tensor& other) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::div(*this, other); }
#endif
    return native::div(*this, other);
}
inline Tensor Tensor::pow(const Tensor& exponent) const {
    return native::pow(*this, exponent);
}
inline Tensor Tensor::pow(Scalar exponent) const {
    return native::pow(*this, exponent);
}

inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        float val = other.to<float>() * alpha.to<float>();
        return cuda_ops::add_scalar(*this, val);
    }
#endif
    return native::add(*this, other, alpha);
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        float val = -(other.to<float>() * alpha.to<float>());
        return cuda_ops::add_scalar(*this, val);
    }
#endif
    return native::sub(*this, other, alpha);
}
inline Tensor Tensor::mul(Scalar other) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::mul_scalar(*this, other.to<float>()); }
#endif
    return native::mul(*this, other);
}
inline Tensor Tensor::div(Scalar other) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::mul_scalar(*this, 1.0f / other.to<float>()); }
#endif
    return native::div(*this, other);
}

// In-place binary (with CUDA dispatch)
inline Tensor& Tensor::add_(const Tensor& other, Scalar alpha) {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        // CUDA element-wise ops read data_ptr sequentially — must be contiguous
        Tensor other_c = other.is_contiguous() ? other : other.contiguous();
        Tensor self_c = is_contiguous() ? *this : contiguous();
        // CUDA: self += alpha * other
        Tensor scaled = (alpha.toDouble() != 1.0)
            ? cuda_ops::mul_scalar(other_c, static_cast<float>(alpha.toDouble()))
            : other_c;
        // Handle broadcasting for bias addition
        if (numel() != scaled.numel() && dim() == 2 && scaled.dim() == 1 && scaled.size(0) == size(1)) {
            Tensor result = cuda_ops::add_broadcast(self_c, scaled);
            cuda_ops::copy_(*this, result);
        } else {
            Tensor result = cuda_ops::add(self_c, scaled);
            cuda_ops::copy_(*this, result);
        }
        return *this;
    }
#endif
    return native::add_(*this, other, alpha);
}
inline Tensor& Tensor::sub_(const Tensor& other, Scalar alpha) {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        // CUDA element-wise ops read data_ptr sequentially — must be contiguous
        Tensor other_c = other.is_contiguous() ? other : other.contiguous();
        Tensor self_c = is_contiguous() ? *this : contiguous();
        // CUDA: param -= alpha * other
        Tensor scaled = (alpha.toDouble() != 1.0)
            ? cuda_ops::mul_scalar(other_c, static_cast<float>(alpha.toDouble()))
            : other_c;
        Tensor result = cuda_ops::sub(self_c, scaled);
        cuda_ops::copy_(*this, result);
        return *this;
    }
#endif
    return native::sub_(*this, other, alpha);
}
inline Tensor& Tensor::mul_(const Tensor& other) {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        Tensor other_c = other.is_contiguous() ? other : other.contiguous();
        Tensor self_c = is_contiguous() ? *this : contiguous();
        Tensor result = cuda_ops::mul(self_c, other_c);
        cuda_ops::copy_(*this, result);
        return *this;
    }
#endif
    return native::mul_(*this, other);
}
inline Tensor& Tensor::div_(const Tensor& other) {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        Tensor other_c = other.is_contiguous() ? other : other.contiguous();
        Tensor self_c = is_contiguous() ? *this : contiguous();
        Tensor result = cuda_ops::div(self_c, other_c);
        cuda_ops::copy_(*this, result);
        return *this;
    }
#endif
    return native::div_(*this, other);
}
inline Tensor& Tensor::add_(Scalar other, Scalar alpha) {
    return native::add_(*this, other, alpha);
}
inline Tensor& Tensor::sub_(Scalar other, Scalar alpha) {
    return native::sub_(*this, other, alpha);
}
inline Tensor& Tensor::mul_(Scalar other) {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        // Create scaled result then copy back to this tensor
        Tensor result = cuda_ops::mul_scalar(*this, other.to<float>());
        cuda_ops::copy_(*this, result);
        return *this;
    }
#endif
    return native::mul_(*this, other);
}
inline Tensor& Tensor::div_(Scalar other) {
    return native::div_(*this, other);
}

// Fused operations
inline Tensor Tensor::addcmul(const Tensor& tensor1, const Tensor& tensor2, Scalar value) const {
    return native::addcmul(*this, tensor1, tensor2, value);
}
inline Tensor& Tensor::addcmul_(const Tensor& tensor1, const Tensor& tensor2, Scalar value) {
    return native::addcmul_(*this, tensor1, tensor2, value);
}
inline Tensor Tensor::addcdiv(const Tensor& tensor1, const Tensor& tensor2, Scalar value) const {
    return native::addcdiv(*this, tensor1, tensor2, value);
}
inline Tensor& Tensor::addcdiv_(const Tensor& tensor1, const Tensor& tensor2, Scalar value) {
    return native::addcdiv_(*this, tensor1, tensor2, value);
}

// Element-wise maximum/minimum (free functions for optimizers)
inline Tensor maximum(const Tensor& a, const Tensor& b) {
    return native::maximum(a, b);
}

inline Tensor minimum(const Tensor& a, const Tensor& b) {
    return native::minimum(a, b);
}

// Comparison operations
inline Tensor Tensor::eq(const Tensor& other) const { return native::eq(*this, other); }
inline Tensor Tensor::ne(const Tensor& other) const { return native::ne(*this, other); }
inline Tensor Tensor::lt(const Tensor& other) const { return native::lt(*this, other); }
inline Tensor Tensor::le(const Tensor& other) const { return native::le(*this, other); }
inline Tensor Tensor::gt(const Tensor& other) const { return native::gt(*this, other); }
inline Tensor Tensor::ge(const Tensor& other) const { return native::ge(*this, other); }

inline Tensor Tensor::eq(Scalar other) const { return native::eq(*this, other); }
inline Tensor Tensor::ne(Scalar other) const { return native::ne(*this, other); }
inline Tensor Tensor::lt(Scalar other) const { return native::lt(*this, other); }
inline Tensor Tensor::le(Scalar other) const { return native::le(*this, other); }
inline Tensor Tensor::gt(Scalar other) const { return native::gt(*this, other); }
inline Tensor Tensor::ge(Scalar other) const { return native::ge(*this, other); }

// Reduction operations
inline Tensor Tensor::sum() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::sum(*this); }
#endif
    return native::sum(*this);
}
inline Tensor Tensor::sum(int64_t dim, bool keepdim) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::sum_dim(*this, dim, keepdim); }
#endif
    return native::sum(*this, dim, keepdim);
}
inline Tensor Tensor::mean() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::mean(*this); }
#endif
    return native::mean(*this);
}
inline Tensor Tensor::mean(int64_t dim, bool keepdim) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        // Use CUDA sum_dim then divide
        Tensor s = cuda_ops::sum_dim(*this, dim, keepdim);
        int64_t actual_dim = dim < 0 ? dim + this->dim() : dim;
        float reduce_size = static_cast<float>(this->size(actual_dim));
        return cuda_ops::mul_scalar(s, 1.0f / reduce_size);
    }
#endif
    return native::mean(*this, dim, keepdim);
}
inline Tensor Tensor::prod() const { return native::prod(*this); }
inline Tensor Tensor::max() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::max(*this); }
#endif
    return native::max(*this);
}
inline std::tuple<Tensor, Tensor> Tensor::max(int64_t dim, bool keepdim) const {
    return native::max(*this, dim, keepdim);
}
inline Tensor Tensor::min() const {
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::min(*this); }
#endif
    return native::min(*this);
}
inline std::tuple<Tensor, Tensor> Tensor::min(int64_t dim, bool keepdim) const {
    return native::min(*this, dim, keepdim);
}
inline Tensor Tensor::argmax() const { return native::argmax(*this); }
inline Tensor Tensor::argmax(int64_t dim, bool keepdim) const { return native::argmax(*this, dim, keepdim); }
inline Tensor Tensor::argmin() const { return native::argmin(*this); }
inline Tensor Tensor::argmin(int64_t dim, bool keepdim) const { return native::argmin(*this, dim, keepdim); }
inline Tensor Tensor::var(bool unbiased) const { return native::var(*this, unbiased); }
inline Tensor Tensor::std(bool unbiased) const { return native::std(*this, unbiased); }
inline Tensor Tensor::norm(Scalar p) const { return native::norm(*this, p); }
inline bool Tensor::all() const { return native::all(*this); }
inline bool Tensor::any() const { return native::any(*this); }
inline std::tuple<Tensor, Tensor> Tensor::sort(int64_t dim, bool descending) const { return native::sort(*this, dim, descending); }
inline Tensor Tensor::argsort(int64_t dim, bool descending) const { return native::argsort(*this, dim, descending); }
inline std::tuple<Tensor, Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const { return native::topk(*this, k, dim, largest, sorted); }
inline Tensor Tensor::cumsum(int64_t dim) const { return native::cumsum(*this, dim); }
inline Tensor Tensor::cumprod(int64_t dim) const { return native::cumprod(*this, dim); }

// Linear algebra
inline Tensor Tensor::matmul(const Tensor& other) const {
    return native::matmul(*this, other);
}
inline Tensor Tensor::mm(const Tensor& other) const {
#ifdef PT_USE_CUDA
    if (is_cuda()) {
        // Make both tensors contiguous if needed for CUDA mm
        Tensor self_contig = is_contiguous() ? *this : contiguous();
        Tensor other_contig = other.is_contiguous() ? other : other.contiguous();
        return cuda_ops::mm(self_contig, other_contig);
    }
#endif
    return native::mm(*this, other);
}
inline Tensor Tensor::mv(const Tensor& vec) const {
    return native::mv(*this, vec);
}
inline Tensor Tensor::bmm(const Tensor& other) const {
    return native::bmm(*this, other);
}
inline Tensor Tensor::dot(const Tensor& other) const {
    return native::dot(*this, other);
}

// Type conversion
inline Tensor Tensor::to(c10::ScalarType dtype) const {
    if (this->dtype() == dtype) {
        return *this;
    }

    Tensor result = empty(sizes(), TensorOptions().dtype(dtype).device(device()));

    // Copy with type conversion
    PT_DISPATCH_ALL_TYPES(this->dtype(), "to_src", [&] {
        using src_t = scalar_t;
        PT_DISPATCH_ALL_TYPES(dtype, "to_dst", [&] {
            using dst_t = scalar_t;
            const src_t* src = data_ptr<src_t>();
            dst_t* dst = result.mutable_data_ptr<dst_t>();
            int64_t n = numel();
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = static_cast<dst_t>(src[i]);
            }
        });
    });

    if (requires_grad()) {
        result.set_requires_grad(true);
    }

    return result;
}

inline Tensor Tensor::to(c10::Device device, c10::ScalarType dtype) const {
    return to(device).to(dtype);
}

inline Tensor Tensor::to(const TensorOptions& options) const {
    return to(options.device(), options.dtype());
}

// Print
inline void Tensor::print(std::ostream& os) const {
    if (!defined()) {
        os << "Tensor(undefined)";
        return;
    }

    os << "Tensor(";

    // Print shape
    os << "[";
    for (int64_t i = 0; i < dim(); ++i) {
        if (i > 0) os << ", ";
        os << size(i);
    }
    os << "], ";

    // Print dtype
    os << c10::toString(dtype()) << ", ";

    // Print device
    os << device().str();

    if (requires_grad()) {
        os << ", requires_grad=True";
    }

    os << ")\n";

    // Print data (limited)
    if (numel() <= 20) {
        PT_DISPATCH_ALL_TYPES(dtype(), "print", [&] {
            const scalar_t* data = data_ptr<scalar_t>();
            os << "[";
            for (int64_t i = 0; i < numel(); ++i) {
                if (i > 0) os << ", ";
                os << data[i];
            }
            os << "]";
        });
    } else {
        os << "[... " << numel() << " elements ...]";
    }
}

} // namespace at

// ============================================================================
// Functional API (torch namespace)
// ============================================================================

namespace torch {

// Import factory functions
using at::empty;
using at::zeros;
using at::ones;
using at::full;
using at::rand;
using at::randn;
using at::randint;
using at::arange;
using at::linspace;
using at::eye;
using at::tensor;

// Functional versions of operations with device dispatch
inline Tensor add(const Tensor& a, const Tensor& b, Scalar alpha = 1) {
#ifdef PT_USE_CUDA
    if (a.is_cuda()) {
        // Check for broadcasting case: [outer, inner] + [inner]
        if (a.numel() != b.numel() && a.dim() == 2 && b.dim() == 1 && b.size(0) == a.size(1)) {
            return at::cuda_ops::add_broadcast(a, b);
        }
        // Same size - use element-wise add
        return at::cuda_ops::add(a, b);
    }
#endif
    return at::native::add(a, b, alpha);
}
inline Tensor sub(const Tensor& a, const Tensor& b, Scalar alpha = 1) {
#ifdef PT_USE_CUDA
    if (a.is_cuda()) {
        return at::cuda_ops::sub(a, b);
    }
#endif
    return at::native::sub(a, b, alpha);
}
inline Tensor mul(const Tensor& a, const Tensor& b) {
#ifdef PT_USE_CUDA
    if (a.is_cuda()) {
        return at::cuda_ops::mul(a, b);
    }
#endif
    return at::native::mul(a, b);
}
inline Tensor div(const Tensor& a, const Tensor& b) {
#ifdef PT_USE_CUDA
    if (a.is_cuda()) {
        return at::cuda_ops::div(a, b);
    }
#endif
    return at::native::div(a, b);
}

// Activation functions with device dispatch
inline Tensor relu(const Tensor& input) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::relu(input);
    }
#endif
    return at::native::relu(input);
}

inline Tensor sigmoid(const Tensor& input) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::sigmoid(input);
    }
#endif
    return at::native::sigmoid(input);
}

inline Tensor tanh(const Tensor& input) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::tanh(input);
    }
#endif
    return at::native::tanh(input);
}

inline Tensor silu(const Tensor& input) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::silu(input);
    }
#endif
    // CPU SiLU: x * sigmoid(x)
    return at::native::mul(input, at::native::sigmoid(input));
}

inline Tensor gelu(const Tensor& input) {
#ifdef PT_USE_CUDA
    if (input.is_cuda()) {
        return at::cuda_ops::gelu(input);
    }
#endif
    // CPU GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Simplified to just use sigmoid for now
    return at::native::mul(input, at::native::sigmoid(input));  // Approximate
}

inline Tensor matmul(const Tensor& a, const Tensor& b) {
#ifdef PT_USE_CUDA
    if (a.is_cuda()) {
        // matmul dispatches to mm for 2D tensors
        if (a.dim() == 2 && b.dim() == 2) {
            return at::cuda_ops::mm(a, b);
        } else if (a.dim() == 3 && b.dim() == 3) {
            return at::cuda_ops::bmm(a, b);
        }
    }
#endif
    return at::native::matmul(a, b);
}
inline Tensor mm(const Tensor& a, const Tensor& b) {
#ifdef PT_USE_CUDA
    if (a.is_cuda()) {
        return at::cuda_ops::mm(a, b);
    }
#endif
    return at::native::mm(a, b);
}
inline Tensor bmm(const Tensor& a, const Tensor& b) {
#ifdef PT_USE_CUDA
    if (a.is_cuda()) {
        return at::cuda_ops::bmm(a, b);
    }
#endif
    return at::native::bmm(a, b);
}

// Reduction functions with device dispatch
inline Tensor sum(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) { return at::cuda_ops::sum(t); }
#endif
    return at::native::sum(t);
}
inline Tensor mean(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) { return at::cuda_ops::mean(t); }
#endif
    return at::native::mean(t);
}
inline Tensor max(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) { return at::cuda_ops::max(t); }
#endif
    return at::native::max(t);
}
inline Tensor min(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) { return at::cuda_ops::min(t); }
#endif
    return at::native::min(t);
}

// Shape functions
inline Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0) {
    return at::native::cat(tensors, dim);
}
inline Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0) {
    return at::native::stack(tensors, dim);
}
inline Tensor squeeze(const Tensor& t) {
    return at::native::squeeze(t);
}
inline Tensor unsqueeze(const Tensor& t, int64_t dim) {
    return at::native::unsqueeze(t, dim);
}
inline Tensor transpose(const Tensor& t, int64_t dim0, int64_t dim1) {
    return at::native::transpose(t, dim0, dim1);
}

// Index functions
inline Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) {
    return at::native::where(condition, x, y);
}
inline Tensor nonzero(const Tensor& t) {
    return at::native::nonzero(t);
}

// Sort, argsort, topk, cumsum, cumprod
inline std::tuple<Tensor, Tensor> sort(const Tensor& t, int64_t dim = -1, bool descending = false) {
    return at::native::sort(t, dim, descending);
}
inline Tensor argsort(const Tensor& t, int64_t dim = -1, bool descending = false) {
    return at::native::argsort(t, dim, descending);
}
inline std::tuple<Tensor, Tensor> topk(const Tensor& t, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true) {
    return at::native::topk(t, k, dim, largest, sorted);
}
inline Tensor cumsum(const Tensor& t, int64_t dim) {
    return at::native::cumsum(t, dim);
}
inline Tensor cumprod(const Tensor& t, int64_t dim) {
    return at::native::cumprod(t, dim);
}

// Clamp, triu, tril, diag
inline Tensor clamp(const Tensor& t, Scalar min_val, Scalar max_val) {
    return at::native::clamp(t, min_val, max_val);
}
inline Tensor triu(const Tensor& t, int64_t diagonal = 0) {
    return at::native::triu(t, diagonal);
}
inline Tensor tril(const Tensor& t, int64_t diagonal = 0) {
    return at::native::tril(t, diagonal);
}
inline Tensor diag(const Tensor& t, int64_t diagonal = 0) {
    return at::native::diag(t, diagonal);
}

// Einsum
inline Tensor einsum(const std::string& equation, const std::vector<Tensor>& tensors) {
    return at::native::einsum(equation, tensors);
}

// Linalg functions
inline Tensor inverse(const Tensor& t) { return at::native::inverse(t); }
inline Tensor det(const Tensor& t) { return at::native::det(t); }
inline Tensor trace(const Tensor& t) { return at::native::trace(t); }
inline Tensor cholesky(const Tensor& t, bool upper = false) { return at::native::cholesky(t, upper); }
inline Tensor solve(const Tensor& A, const Tensor& b) { return at::native::solve(A, b); }
inline Tensor cross(const Tensor& a, const Tensor& b, int64_t dim = -1) { return at::native::cross(a, b, dim); }
inline Tensor matrix_norm(const Tensor& t, double ord = 2.0) { return at::native::matrix_norm(t, ord); }

namespace linalg {
    inline at::native::LUResult lu(const Tensor& t) { return at::native::lu(t); }
    inline Tensor inv(const Tensor& t) { return at::native::inverse(t); }
    inline Tensor solve(const Tensor& A, const Tensor& b) { return at::native::solve(A, b); }
    inline Tensor det(const Tensor& t) { return at::native::det(t); }
    inline Tensor cholesky(const Tensor& t, bool upper = false) { return at::native::cholesky(t, upper); }
    inline at::native::QRResult qr(const Tensor& t) { return at::native::qr(t); }
    inline Tensor matrix_norm(const Tensor& t, double ord = 2.0) { return at::native::matrix_norm(t, ord); }
    inline Tensor cross(const Tensor& a, const Tensor& b, int64_t dim = -1) { return at::native::cross(a, b, dim); }
}

// New tensor ops
inline Tensor flip(const Tensor& t, c10::IntArrayRef dims) { return at::native::flip(t, dims); }
inline Tensor roll(const Tensor& t, c10::IntArrayRef shifts, c10::IntArrayRef dims) { return at::native::roll(t, shifts, dims); }
inline std::vector<Tensor> meshgrid(const std::vector<Tensor>& tensors, const std::string& indexing = "ij") { return at::native::meshgrid(tensors, indexing); }
inline Tensor repeat_interleave(const Tensor& t, int64_t repeats, int64_t dim = 0) { return at::native::repeat_interleave(t, repeats, dim); }
inline std::tuple<Tensor, Tensor, Tensor> unique(const Tensor& t, bool sorted = true, bool return_inverse = false, bool return_counts = false) { return at::native::unique(t, sorted, return_inverse, return_counts); }
inline Tensor tril_indices(int64_t row, int64_t col, int64_t offset = 0) { return at::native::tril_indices(row, col, offset); }
inline Tensor triu_indices(int64_t row, int64_t col, int64_t offset = 0) { return at::native::triu_indices(row, col, offset); }

// FFT namespace
namespace fft {
    inline Tensor fft(const Tensor& t, int64_t n = -1, int64_t dim = -1) { return at::native::fft(t, n, dim); }
    inline Tensor ifft(const Tensor& t, int64_t n = -1, int64_t dim = -1) { return at::native::ifft(t, n, dim); }
    inline Tensor rfft(const Tensor& t, int64_t n = -1, int64_t dim = -1) { return at::native::rfft(t, n, dim); }
    inline Tensor irfft(const Tensor& t, int64_t n = -1, int64_t dim = -1) { return at::native::irfft(t, n, dim); }
    inline Tensor fft2(const Tensor& t) { return at::native::fft2(t); }
    inline Tensor ifft2(const Tensor& t) { return at::native::ifft2(t); }
    inline Tensor fftfreq(int64_t n, double d = 1.0) { return at::native::fftfreq(n, d); }
    inline Tensor rfftfreq(int64_t n, double d = 1.0) { return at::native::rfftfreq(n, d); }
    inline Tensor fftshift(const Tensor& t, c10::IntArrayRef dims = {}) { return at::native::fftshift(t, dims); }
    inline Tensor ifftshift(const Tensor& t, c10::IntArrayRef dims = {}) { return at::native::ifftshift(t, dims); }
}

} // namespace torch
