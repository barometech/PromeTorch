#pragma once

#include <memory>
#include <vector>
#include <iostream>
#include <initializer_list>
#include <functional>
#include <optional>
#include <tuple>
#include "c10/core/TensorImpl.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Device.h"
#include "c10/core/Storage.h"
#include "c10/util/Exception.h"

// Forward declarations for autograd
namespace torch {
namespace autograd {
struct AutogradMetaImpl;
class Node;
template<typename T1, typename T2, typename T3>
void set_grad_fn(T1&, T2, T3);
} // namespace autograd
} // namespace torch

namespace at {

// Forward declarations
class Tensor;

// ============================================================================
// TensorOptions - Configuration for tensor creation
// ============================================================================

struct PT_API TensorOptions {
    c10::ScalarType dtype_ = c10::ScalarType::Float;
    c10::Device device_ = c10::kCPU;
    bool requires_grad_ = false;

    TensorOptions() = default;

    TensorOptions& dtype(c10::ScalarType dtype) {
        dtype_ = dtype;
        return *this;
    }

    TensorOptions& device(c10::Device device) {
        device_ = device;
        return *this;
    }

    TensorOptions& device(c10::DeviceType type, c10::DeviceIndex index = -1) {
        device_ = c10::Device(type, index);
        return *this;
    }

    TensorOptions& requires_grad(bool requires_grad) {
        requires_grad_ = requires_grad;
        return *this;
    }

    c10::ScalarType dtype() const { return dtype_; }
    c10::Device device() const { return device_; }
    bool requires_grad() const { return requires_grad_; }
};

// Convenience functions for TensorOptions
inline TensorOptions dtype(c10::ScalarType dtype) {
    return TensorOptions().dtype(dtype);
}

inline TensorOptions device(c10::Device device) {
    return TensorOptions().device(device);
}

inline TensorOptions requires_grad(bool requires_grad = true) {
    return TensorOptions().requires_grad(requires_grad);
}

// ============================================================================
// Scalar - Wrapper for scalar values
// ============================================================================

class PT_API Scalar {
public:
    Scalar() : type_(c10::ScalarType::Float) { v_.d = 0.0; }

    Scalar(double v) : type_(c10::ScalarType::Double) { v_.d = v; }
    Scalar(float v) : type_(c10::ScalarType::Float) { v_.f = v; }
    Scalar(int64_t v) : type_(c10::ScalarType::Long) { v_.i = v; }
    Scalar(int32_t v) : type_(c10::ScalarType::Int) { v_.i = v; }
    Scalar(bool v) : type_(c10::ScalarType::Bool) { v_.i = v ? 1 : 0; }

    template<typename T>
    T to() const {
        switch (type_) {
            case c10::ScalarType::Double: return static_cast<T>(v_.d);
            case c10::ScalarType::Float: return static_cast<T>(v_.f);
            case c10::ScalarType::Long:
            case c10::ScalarType::Int:
            case c10::ScalarType::Bool: return static_cast<T>(v_.i);
            default: return static_cast<T>(v_.d);
        }
    }

    double toDouble() const { return to<double>(); }
    float toFloat() const { return to<float>(); }
    int64_t toLong() const { return to<int64_t>(); }
    int32_t toInt() const { return to<int32_t>(); }
    bool toBool() const { return to<bool>(); }

    c10::ScalarType type() const { return type_; }

    bool isFloatingPoint() const {
        return c10::isFloatingType(type_);
    }

    bool isIntegral(bool include_bool = false) const {
        return c10::isIntegralType(type_, include_bool);
    }

private:
    c10::ScalarType type_;
    union {
        double d;
        float f;
        int64_t i;
    } v_;
};

// ============================================================================
// Tensor Class - Main tensor interface
// ============================================================================

class PT_API Tensor {
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    // Default constructor - undefined tensor
    Tensor() : impl_(nullptr) {}

    // Constructor from TensorImpl
    explicit Tensor(std::shared_ptr<c10::TensorImpl> impl)
        : impl_(std::move(impl)) {}

    // Copy and move
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // ========================================================================
    // Basic properties
    // ========================================================================

    bool defined() const { return impl_ != nullptr; }
    explicit operator bool() const { return defined(); }

    c10::TensorImpl* unsafeGetTensorImpl() const { return impl_.get(); }

    // Dimensions
    int64_t dim() const {
        PT_CHECK(defined());
        return impl_->dim();
    }

    int64_t size(int64_t dim) const {
        PT_CHECK(defined());
        return impl_->size(dim);
    }

    c10::IntArrayRef sizes() const {
        PT_CHECK(defined());
        return impl_->sizes();
    }

    int64_t stride(int64_t dim) const {
        PT_CHECK(defined());
        return impl_->stride(dim);
    }

    c10::IntArrayRef strides() const {
        PT_CHECK(defined());
        return impl_->strides();
    }

    int64_t numel() const {
        PT_CHECK(defined());
        return impl_->numel();
    }

    size_t nbytes() const {
        PT_CHECK(defined());
        return impl_->nbytes();
    }

    size_t itemsize() const {
        PT_CHECK(defined());
        return impl_->itemsize();
    }

    // Type and device
    c10::ScalarType dtype() const {
        PT_CHECK(defined());
        return impl_->dtype();
    }

    c10::Device device() const {
        PT_CHECK(defined());
        return impl_->device();
    }

    bool is_cpu() const { return defined() && impl_->is_cpu(); }
    bool is_cuda() const { return defined() && impl_->is_cuda(); }
    bool is_nmcard() const { return defined() && impl_->is_nmcard(); }
    bool is_nmquad() const { return defined() && impl_->is_nmquad(); }

    // Contiguity
    bool is_contiguous() const {
        PT_CHECK(defined());
        return impl_->is_contiguous();
    }

    bool is_contiguous(c10::MemoryFormat format) const {
        PT_CHECK(defined());
        return impl_->is_contiguous(format);
    }

    c10::MemoryFormat suggest_memory_format() const {
        PT_CHECK(defined());
        return impl_->suggest_memory_format();
    }

    // Trusted flag: skip dtype/contiguous/device checks in hot paths.
    // Trusted tensors are guaranteed float32, contiguous, CPU by construction.
    bool is_trusted() const { return impl_ && impl_->is_trusted(); }
    void set_trusted(bool t) { if (impl_) impl_->set_trusted(t); }

    // ========================================================================
    // Data access
    // ========================================================================

    template<typename T>
    T* data_ptr() const {
        PT_CHECK(defined());
        return impl_->data<T>();
    }

    void* data_ptr() const {
        PT_CHECK(defined());
        return impl_->data();
    }

    template<typename T>
    T* mutable_data_ptr() {
        PT_CHECK(defined());
        return impl_->mutable_data<T>();
    }

    // Get single item (for scalar tensors)
    template<typename T>
    T item() const {
        PT_CHECK(defined());
        PT_CHECK_MSG(numel() == 1, "item() requires tensor with single element");
        return data_ptr<T>()[0];
    }

    Scalar item() const;

    // ========================================================================
    // Autograd
    // ========================================================================

    bool requires_grad() const {
        return defined() && impl_->requires_grad();
    }

    Tensor& set_requires_grad(bool requires_grad) {
        PT_CHECK(defined());
        impl_->set_requires_grad(requires_grad);
        return *this;
    }

    bool is_leaf() const {
        return !defined() || impl_->is_leaf();
    }

    // Gradient
    Tensor grad() const;
    void set_grad(const Tensor& grad);

    // ========================================================================
    // Storage
    // ========================================================================

    const c10::Storage& storage() const {
        PT_CHECK(defined());
        return impl_->storage();
    }

    int64_t storage_offset() const {
        PT_CHECK(defined());
        return impl_->storage_offset();
    }

    // ========================================================================
    // Copy operations
    // ========================================================================

    Tensor clone() const;
    Tensor detach() const;
    Tensor contiguous() const;
    Tensor contiguous(c10::MemoryFormat memory_format) const;
    Tensor to(c10::MemoryFormat memory_format) const;

    // Copy data from another tensor
    Tensor& copy_(const Tensor& src);

    // ========================================================================
    // Shape operations (declarations - implementations in TensorShapeOps.h)
    // ========================================================================

    Tensor view(c10::IntArrayRef sizes) const;
    Tensor reshape(c10::IntArrayRef sizes) const;
    Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;

    Tensor squeeze() const;
    Tensor squeeze(int64_t dim) const;
    Tensor unsqueeze(int64_t dim) const;

    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(c10::IntArrayRef dims) const;
    Tensor t() const;  // 2D transpose

    Tensor expand(c10::IntArrayRef sizes) const;
    Tensor repeat(c10::IntArrayRef repeats) const;

    std::vector<Tensor> split(int64_t split_size, int64_t dim = 0) const;
    std::vector<Tensor> chunk(int64_t chunks, int64_t dim = 0) const;

    // ========================================================================
    // Indexing operations (declarations)
    // ========================================================================

    Tensor operator[](int64_t index) const;
    Tensor select(int64_t dim, int64_t index) const;
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;

    // ========================================================================
    // Unary operations (declarations - implementations in TensorOps.h)
    // ========================================================================

    Tensor neg() const;
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor rsqrt() const;
    Tensor square() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor log2() const;
    Tensor log10() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor tan() const;
    Tensor tanh() const;
    Tensor sigmoid() const;
    Tensor relu() const;
    Tensor ceil() const;
    Tensor floor() const;
    Tensor round() const;
    Tensor sign() const;
    Tensor reciprocal() const;
    Tensor clamp(Scalar min_val, Scalar max_val) const;
    Tensor clamp(std::optional<Scalar> min, std::optional<Scalar> max) const;
    Tensor& clamp_(Scalar min_val, Scalar max_val);
    Tensor clamp_min(Scalar min_val) const;
    Tensor clamp_max(Scalar max_val) const;
    Tensor triu(int64_t diagonal = 0) const;
    Tensor tril(int64_t diagonal = 0) const;
    Tensor diag(int64_t diagonal = 0) const;

    // In-place versions
    Tensor& neg_();
    Tensor& abs_();
    Tensor& sqrt_();
    Tensor& exp_();
    Tensor& log_();
    Tensor& sin_();
    Tensor& cos_();
    Tensor& tanh_();
    Tensor& sigmoid_();
    Tensor& relu_();
    Tensor& ceil_();
    Tensor& floor_();
    Tensor& round_();
    Tensor& zero_();
    Tensor& fill_(Scalar value);

    // ========================================================================
    // Binary operations (declarations)
    // ========================================================================

    Tensor add(const Tensor& other, Scalar alpha = 1) const;
    Tensor sub(const Tensor& other, Scalar alpha = 1) const;
    Tensor mul(const Tensor& other) const;
    Tensor mul_broadcast(const Tensor& other) const;  // Broadcasting mul for CUDA
    Tensor div(const Tensor& other) const;
    Tensor pow(const Tensor& exponent) const;
    Tensor pow(Scalar exponent) const;
    Tensor fmod(const Tensor& other) const;
    Tensor remainder(const Tensor& other) const;

    Tensor add(Scalar other, Scalar alpha = 1) const;
    Tensor sub(Scalar other, Scalar alpha = 1) const;
    Tensor mul(Scalar other) const;
    Tensor div(Scalar other) const;

    // In-place versions
    Tensor& add_(const Tensor& other, Scalar alpha = 1);
    Tensor& sub_(const Tensor& other, Scalar alpha = 1);
    Tensor& mul_(const Tensor& other);
    Tensor& div_(const Tensor& other);
    Tensor& add_(Scalar other, Scalar alpha = 1);
    Tensor& sub_(Scalar other, Scalar alpha = 1);
    Tensor& mul_(Scalar other);
    Tensor& div_(Scalar other);

    // Fused operations (for optimizer efficiency)
    // addcmul: self + value * tensor1 * tensor2
    Tensor addcmul(const Tensor& tensor1, const Tensor& tensor2, Scalar value = 1) const;
    Tensor& addcmul_(const Tensor& tensor1, const Tensor& tensor2, Scalar value = 1);

    // addcdiv: self + value * tensor1 / tensor2
    Tensor addcdiv(const Tensor& tensor1, const Tensor& tensor2, Scalar value = 1) const;
    Tensor& addcdiv_(const Tensor& tensor1, const Tensor& tensor2, Scalar value = 1);

    // ========================================================================
    // Comparison operations
    // ========================================================================

    Tensor eq(const Tensor& other) const;
    Tensor ne(const Tensor& other) const;
    Tensor lt(const Tensor& other) const;
    Tensor le(const Tensor& other) const;
    Tensor gt(const Tensor& other) const;
    Tensor ge(const Tensor& other) const;

    Tensor eq(Scalar other) const;
    Tensor ne(Scalar other) const;
    Tensor lt(Scalar other) const;
    Tensor le(Scalar other) const;
    Tensor gt(Scalar other) const;
    Tensor ge(Scalar other) const;

    // ========================================================================
    // Reduction operations
    // ========================================================================

    Tensor sum() const;
    Tensor sum(int64_t dim, bool keepdim = false) const;
    Tensor sum(c10::IntArrayRef dims, bool keepdim = false) const;

    Tensor mean() const;
    Tensor mean(int64_t dim, bool keepdim = false) const;
    Tensor mean(c10::IntArrayRef dims, bool keepdim = false) const;

    Tensor prod() const;
    Tensor prod(int64_t dim, bool keepdim = false) const;

    Tensor max() const;
    std::tuple<Tensor, Tensor> max(int64_t dim, bool keepdim = false) const;

    Tensor min() const;
    std::tuple<Tensor, Tensor> min(int64_t dim, bool keepdim = false) const;

    Tensor argmax() const;
    Tensor argmax(int64_t dim, bool keepdim = false) const;

    Tensor argmin() const;
    Tensor argmin(int64_t dim, bool keepdim = false) const;

    Tensor var(bool unbiased = true) const;
    Tensor var(int64_t dim, bool unbiased = true, bool keepdim = false) const;

    Tensor std(bool unbiased = true) const;
    Tensor std(int64_t dim, bool unbiased = true, bool keepdim = false) const;

    Tensor norm(Scalar p = 2) const;
    Tensor norm(Scalar p, int64_t dim, bool keepdim = false) const;

    bool all() const;
    Tensor all(int64_t dim, bool keepdim = false) const;

    bool any() const;
    Tensor any(int64_t dim, bool keepdim = false) const;

    std::tuple<Tensor, Tensor> sort(int64_t dim = -1, bool descending = false) const;
    Tensor argsort(int64_t dim = -1, bool descending = false) const;
    std::tuple<Tensor, Tensor> topk(int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true) const;
    Tensor cumsum(int64_t dim) const;
    Tensor cumprod(int64_t dim) const;

    // ========================================================================
    // Linear algebra operations
    // ========================================================================

    Tensor matmul(const Tensor& other) const;
    Tensor mm(const Tensor& other) const;
    Tensor mv(const Tensor& vec) const;
    Tensor bmm(const Tensor& other) const;
    Tensor dot(const Tensor& other) const;
    Tensor outer(const Tensor& other) const;
    Tensor addmm(const Tensor& mat1, const Tensor& mat2, Scalar beta = 1, Scalar alpha = 1) const;

    // ========================================================================
    // Type conversion
    // ========================================================================

    Tensor to(c10::ScalarType dtype) const;
    Tensor to(c10::Device device) const;
    Tensor to(c10::Device device, c10::ScalarType dtype) const;
    Tensor to(const TensorOptions& options) const;

    Tensor toType(c10::ScalarType dtype) const { return to(dtype); }
    Tensor cuda() const { return to(c10::Device(c10::DeviceType::CUDA, 0)); }
    Tensor cpu() const { return to(c10::kCPU); }

    Tensor float_() const { return to(c10::ScalarType::Float); }
    Tensor double_() const { return to(c10::ScalarType::Double); }
    Tensor half() const { return to(c10::ScalarType::Half); }
    Tensor int_() const { return to(c10::ScalarType::Int); }
    Tensor long_() const { return to(c10::ScalarType::Long); }
    Tensor bool_() const { return to(c10::ScalarType::Bool); }

    // ========================================================================
    // Operators
    // ========================================================================

    Tensor operator-() const { return neg(); }
    Tensor operator+(const Tensor& other) const { return add(other); }
    Tensor operator-(const Tensor& other) const { return sub(other); }
    Tensor operator*(const Tensor& other) const { return mul(other); }
    Tensor operator/(const Tensor& other) const { return div(other); }

    Tensor operator+(Scalar other) const { return add(other); }
    Tensor operator-(Scalar other) const { return sub(other); }
    Tensor operator*(Scalar other) const { return mul(other); }
    Tensor operator/(Scalar other) const { return div(other); }

    Tensor& operator+=(const Tensor& other) { return add_(other); }
    Tensor& operator-=(const Tensor& other) { return sub_(other); }
    Tensor& operator*=(const Tensor& other) { return mul_(other); }
    Tensor& operator/=(const Tensor& other) { return div_(other); }

    Tensor& operator+=(Scalar other) { return add_(other); }
    Tensor& operator-=(Scalar other) { return sub_(other); }
    Tensor& operator*=(Scalar other) { return mul_(other); }
    Tensor& operator/=(Scalar other) { return div_(other); }

    // Comparison operators return bool tensors
    Tensor operator==(const Tensor& other) const { return eq(other); }
    Tensor operator!=(const Tensor& other) const { return ne(other); }
    Tensor operator<(const Tensor& other) const { return lt(other); }
    Tensor operator<=(const Tensor& other) const { return le(other); }
    Tensor operator>(const Tensor& other) const { return gt(other); }
    Tensor operator>=(const Tensor& other) const { return ge(other); }

    // ========================================================================
    // Print/Debug
    // ========================================================================

    void print(std::ostream& os = std::cout) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        t.print(os);
        return os;
    }

    // ========================================================================
    // Implementation Access (for autograd)
    // ========================================================================

    const std::shared_ptr<c10::TensorImpl>& getIntrusivePtr() const {
        return impl_;
    }

    std::shared_ptr<c10::TensorImpl>& getIntrusivePtr() {
        return impl_;
    }

    c10::AutogradMeta* autograd_meta() const {
        return defined() ? impl_->autograd_meta() : nullptr;
    }

    // Friend declarations for autograd
    friend struct torch::autograd::AutogradMetaImpl;
    friend class torch::autograd::Node;

    template<typename T1, typename T2, typename T3>
    friend void torch::autograd::set_grad_fn(T1&, T2, T3);

private:
    std::shared_ptr<c10::TensorImpl> impl_;
};

// ============================================================================
// Scalar operators with Tensor
// ============================================================================

inline Tensor operator+(Scalar lhs, const Tensor& rhs) { return rhs.add(lhs); }
inline Tensor operator-(Scalar lhs, const Tensor& rhs) { return rhs.neg().add(lhs); }
inline Tensor operator*(Scalar lhs, const Tensor& rhs) { return rhs.mul(lhs); }
inline Tensor operator/(Scalar lhs, const Tensor& rhs) { return rhs.reciprocal().mul(lhs); }

} // namespace at

// ============================================================================
// Namespace alias
// ============================================================================

namespace torch {
    using at::Tensor;
    using at::TensorOptions;
    using at::Scalar;
    using at::dtype;
    using at::device;
    using at::requires_grad;
}
