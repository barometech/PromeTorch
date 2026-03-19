#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include "c10/macros/Macros.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Device.h"
#include "c10/core/Storage.h"
#include "c10/core/Allocator.h"
#include "c10/util/Exception.h"

namespace c10 {

// ============================================================================
// MemoryFormat - Describes memory layout of tensors
// ============================================================================

enum class MemoryFormat : int8_t {
    Contiguous = 0,      // NCHW (default row-major)
    ChannelsLast = 1,    // NHWC (channel dim is innermost)
    ChannelsLast3d = 2,  // NDHWC (3D channels-last)
    Preserve = 3         // Keep existing format
};

// Forward declarations
class TensorImpl;
struct AutogradMeta;

// ============================================================================
// SmallVector - Optimized vector for small sizes (avoids heap allocation)
// ============================================================================

template<typename T, size_t N>
class SmallVector {
public:
    SmallVector() : size_(0), heap_data_(nullptr) {}

    SmallVector(std::initializer_list<T> init) : size_(0), heap_data_(nullptr) {
        reserve(init.size());
        for (const auto& val : init) {
            push_back(val);
        }
    }

    SmallVector(size_t count, const T& value = T()) : size_(0), heap_data_(nullptr) {
        resize(count, value);
    }

    SmallVector(const SmallVector& other) : size_(0), heap_data_(nullptr) {
        *this = other;
    }

    SmallVector(SmallVector&& other) noexcept : size_(0), heap_data_(nullptr) {
        *this = std::move(other);
    }

    ~SmallVector() {
        delete[] heap_data_;
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            clear();
            reserve(other.size_);
            size_ = other.size_;
            T* dst = data();
            const T* src = other.data();
            for (size_t i = 0; i < size_; ++i) {
                dst[i] = src[i];
            }
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            delete[] heap_data_;
            if (other.is_small()) {
                heap_data_ = nullptr;
                size_ = other.size_;
                for (size_t i = 0; i < size_; ++i) {
                    stack_data_[i] = std::move(other.stack_data_[i]);
                }
            } else {
                heap_data_ = other.heap_data_;
                heap_capacity_ = other.heap_capacity_;
                size_ = other.size_;
                other.heap_data_ = nullptr;
            }
            other.size_ = 0;
        }
        return *this;
    }

    // Element access
    T& operator[](size_t index) { return data()[index]; }
    const T& operator[](size_t index) const { return data()[index]; }

    T& at(size_t index) {
        PT_CHECK(index < size_);
        return data()[index];
    }

    const T& at(size_t index) const {
        PT_CHECK(index < size_);
        return data()[index];
    }

    T& front() { return data()[0]; }
    const T& front() const { return data()[0]; }

    T& back() { return data()[size_ - 1]; }
    const T& back() const { return data()[size_ - 1]; }

    T* data() { return is_small() ? stack_data_ : heap_data_; }
    const T* data() const { return is_small() ? stack_data_ : heap_data_; }

    // Iterators
    T* begin() { return data(); }
    const T* begin() const { return data(); }
    T* end() { return data() + size_; }
    const T* end() const { return data() + size_; }

    // Capacity
    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }
    size_t capacity() const { return is_small() ? N : heap_capacity_; }

    void reserve(size_t new_cap) {
        if (new_cap <= capacity()) return;

        T* new_data = new T[new_cap];
        T* old_data = data();
        for (size_t i = 0; i < size_; ++i) {
            new_data[i] = std::move(old_data[i]);
        }

        delete[] heap_data_;
        heap_data_ = new_data;
        heap_capacity_ = new_cap;
    }

    // Modifiers
    void clear() { size_ = 0; }

    void push_back(const T& value) {
        if (size_ >= capacity()) {
            reserve(std::max(capacity() * 2, size_t(4)));
        }
        data()[size_++] = value;
    }

    void push_back(T&& value) {
        if (size_ >= capacity()) {
            reserve(std::max(capacity() * 2, size_t(4)));
        }
        data()[size_++] = std::move(value);
    }

    void pop_back() {
        PT_CHECK(size_ > 0);
        --size_;
    }

    void resize(size_t new_size, const T& value = T()) {
        if (new_size > capacity()) {
            reserve(new_size);
        }
        if (new_size > size_) {
            T* d = data();
            for (size_t i = size_; i < new_size; ++i) {
                d[i] = value;
            }
        }
        size_ = new_size;
    }

private:
    bool is_small() const { return heap_data_ == nullptr; }

    T stack_data_[N];
    size_t size_;
    T* heap_data_;
    size_t heap_capacity_ = 0;
};

// ============================================================================
// IntArrayRef - Non-owning reference to array of integers
// ============================================================================

class IntArrayRef {
public:
    IntArrayRef() : data_(nullptr), size_(0) {}

    IntArrayRef(const int64_t* data, size_t size) : data_(data), size_(size) {}

    IntArrayRef(const std::vector<int64_t>& vec)
        : data_(vec.data()), size_(vec.size()) {}

    IntArrayRef(std::initializer_list<int64_t> list)
        : data_(list.begin()), size_(list.size()) {}

    template<size_t N>
    IntArrayRef(const SmallVector<int64_t, N>& vec)
        : data_(vec.data()), size_(vec.size()) {}

    // Element access
    const int64_t& operator[](size_t index) const { return data_[index]; }

    const int64_t& at(size_t index) const {
        PT_CHECK(index < size_);
        return data_[index];
    }

    const int64_t& front() const { return data_[0]; }
    const int64_t& back() const { return data_[size_ - 1]; }
    const int64_t* data() const { return data_; }

    // Iterators
    const int64_t* begin() const { return data_; }
    const int64_t* end() const { return data_ + size_; }

    // Capacity
    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }

    // Convert to vector
    std::vector<int64_t> vec() const {
        return std::vector<int64_t>(data_, data_ + size_);
    }

    // Comparison
    bool equals(IntArrayRef other) const {
        if (size_ != other.size_) return false;
        for (size_t i = 0; i < size_; ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }

private:
    const int64_t* data_;
    size_t size_;
};

inline bool operator==(IntArrayRef a, IntArrayRef b) {
    return a.equals(b);
}

inline bool operator!=(IntArrayRef a, IntArrayRef b) {
    return !a.equals(b);
}

// ============================================================================
// AutogradMeta - Metadata for automatic differentiation
// ============================================================================

struct PT_API AutogradMetaInterface {
    virtual ~AutogradMetaInterface() = default;
};

// Forward declaration - will be fully defined in autograd module
struct PT_API AutogradMeta : public AutogradMetaInterface {
    // Gradient tensor
    std::shared_ptr<TensorImpl> grad_;

    // Gradient function (backward node)
    // Will be defined later: std::shared_ptr<Node> grad_fn_;

    // Output number in grad_fn_
    uint32_t output_nr_ = 0;

    // Flags
    bool requires_grad_ = false;
    bool retains_grad_ = false;
    bool is_leaf_ = true;

    // Gradient accumulator (for leaf variables)
    // Will be defined later: std::weak_ptr<Node> grad_accumulator_;

    // Hooks
    // std::vector<std::function<Tensor(const Tensor&)>> hooks_;

    virtual ~AutogradMeta() = default;
};

// ============================================================================
// AutogradMeta Factory - allows torch to register custom AutogradMeta creation
// ============================================================================
// This allows the autograd module to create AutogradMetaImpl directly
// instead of creating base AutogradMeta and then upgrading it.
//
// IMPORTANT: These functions are implemented in TensorImpl.cpp to avoid
// DLL boundary issues with static variables. Do NOT use inline functions
// with static variables for factories on Windows!

using AutogradMetaFactory = std::unique_ptr<AutogradMeta>(*)();

// Declarations - implementations in TensorImpl.cpp (exported from c10.dll)
PT_API AutogradMetaFactory& get_autograd_meta_factory_impl();
PT_API void set_autograd_meta_factory_impl(AutogradMetaFactory factory);
PT_API std::unique_ptr<AutogradMeta> create_autograd_meta_impl();

// Inline wrappers that call the exported functions
inline AutogradMetaFactory& get_autograd_meta_factory() {
    return get_autograd_meta_factory_impl();
}

inline void set_autograd_meta_factory(AutogradMetaFactory factory) {
    set_autograd_meta_factory_impl(factory);
}

inline std::unique_ptr<AutogradMeta> create_autograd_meta() {
    return create_autograd_meta_impl();
}

// ============================================================================
// TensorImpl - Core Tensor Implementation
// ============================================================================

class PT_API TensorImpl {
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    // Empty tensor
    TensorImpl()
        : storage_()
        , storage_offset_(0)
        , numel_(0)
        , dtype_(ScalarType::Float)
        , is_contiguous_(true)
        , is_wrapped_number_(false)
        , allow_tensor_metadata_change_(true)
        , autograd_meta_(nullptr)
        , ref_count_(1)
    {}

    // Tensor with storage
    TensorImpl(
        Storage storage,
        ScalarType dtype
    )
        : storage_(std::move(storage))
        , storage_offset_(0)
        , numel_(0)
        , dtype_(dtype)
        , is_contiguous_(true)
        , is_wrapped_number_(false)
        , allow_tensor_metadata_change_(true)
        , autograd_meta_(nullptr)
        , ref_count_(1)
    {}

    // Tensor with sizes
    TensorImpl(
        Storage storage,
        ScalarType dtype,
        IntArrayRef sizes
    )
        : storage_(std::move(storage))
        , storage_offset_(0)
        , dtype_(dtype)
        , is_wrapped_number_(false)
        , allow_tensor_metadata_change_(true)
        , autograd_meta_(nullptr)
        , ref_count_(1)
    {
        set_sizes_contiguous(sizes);
    }

    // Tensor with sizes and strides
    TensorImpl(
        Storage storage,
        ScalarType dtype,
        IntArrayRef sizes,
        IntArrayRef strides,
        int64_t storage_offset = 0
    )
        : storage_(std::move(storage))
        , storage_offset_(storage_offset)
        , dtype_(dtype)
        , is_wrapped_number_(false)
        , allow_tensor_metadata_change_(true)
        , autograd_meta_(nullptr)
        , ref_count_(1)
    {
        set_sizes_and_strides(sizes, strides);
    }

    // No copy
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;

    // Virtual destructor for inheritance
    virtual ~TensorImpl() = default;

    // ========================================================================
    // Reference counting
    // ========================================================================

    void retain() {
        ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void release() {
        if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

    int64_t use_count() const {
        return ref_count_.load(std::memory_order_relaxed);
    }

    bool unique() const {
        return use_count() == 1;
    }

    // ========================================================================
    // Data access
    // ========================================================================

    template<typename T>
    T* data() const {
        return storage_.data<T>() + storage_offset_;
    }

    void* data() const {
        char* ptr = static_cast<char*>(storage_.data());
        return ptr + storage_offset_ * elementSize(dtype_);
    }

    template<typename T>
    T* mutable_data() {
        return storage_.mutable_data<T>() + storage_offset_;
    }

    void* mutable_data() {
        char* ptr = static_cast<char*>(storage_.unsafeGetStorageImpl()->mutable_data());
        return ptr + storage_offset_ * elementSize(dtype_);
    }

    // ========================================================================
    // Storage
    // ========================================================================

    const Storage& storage() const { return storage_; }
    Storage& storage() { return storage_; }

    void set_storage(Storage storage) {
        storage_ = std::move(storage);
    }

    bool has_storage() const {
        return storage_.defined();
    }

    int64_t storage_offset() const { return storage_offset_; }

    void set_storage_offset(int64_t offset) {
        storage_offset_ = offset;
    }

    // ========================================================================
    // Shape and strides
    // ========================================================================

    int64_t dim() const { return static_cast<int64_t>(sizes_.size()); }

    int64_t size(int64_t d) const {
        d = maybe_wrap_dim(d);
        return sizes_[d];
    }

    IntArrayRef sizes() const {
        return IntArrayRef(sizes_.data(), sizes_.size());
    }

    int64_t stride(int64_t d) const {
        d = maybe_wrap_dim(d);
        return strides_[d];
    }

    IntArrayRef strides() const {
        return IntArrayRef(strides_.data(), strides_.size());
    }

    int64_t numel() const { return numel_; }

    bool is_contiguous() const { return is_contiguous_; }

    // Check contiguity with specific memory format
    bool is_contiguous(MemoryFormat format) const {
        switch (format) {
            case MemoryFormat::Contiguous:
                return is_contiguous_;
            case MemoryFormat::ChannelsLast:
                return is_channels_last_contiguous();
            case MemoryFormat::ChannelsLast3d:
                return is_channels_last_3d_contiguous();
            case MemoryFormat::Preserve:
                return is_contiguous_;
            default:
                return is_contiguous_;
        }
    }

    // Check if tensor is in NHWC layout (4D only)
    bool is_channels_last_contiguous() const {
        if (dim() != 4) return false;
        // NHWC strides: {C*H*W, 1, W*C, C}
        // Channel stride = 1 (innermost)
        int64_t N = sizes_[0], C = sizes_[1], H = sizes_[2], W = sizes_[3];
        return strides_[0] == C * H * W &&
               strides_[1] == 1 &&
               strides_[2] == W * C &&
               strides_[3] == C;
    }

    // Check if tensor is in NDHWC layout (5D only)
    bool is_channels_last_3d_contiguous() const {
        if (dim() != 5) return false;
        int64_t N = sizes_[0], C = sizes_[1], D = sizes_[2], H = sizes_[3], W = sizes_[4];
        return strides_[0] == C * D * H * W &&
               strides_[1] == 1 &&
               strides_[2] == H * W * C &&
               strides_[3] == W * C &&
               strides_[4] == C;
    }

    // Suggest the memory format of this tensor
    MemoryFormat suggest_memory_format() const {
        if (is_channels_last_contiguous()) return MemoryFormat::ChannelsLast;
        if (is_channels_last_3d_contiguous()) return MemoryFormat::ChannelsLast3d;
        return MemoryFormat::Contiguous;
    }

    // Set sizes (computes contiguous strides)
    void set_sizes_contiguous(IntArrayRef sizes) {
        sizes_.resize(sizes.size());
        strides_.resize(sizes.size());

        for (size_t i = 0; i < sizes.size(); ++i) {
            sizes_[i] = sizes[i];
        }

        refresh_numel();
        compute_contiguous_strides();
        is_contiguous_ = true;
    }

    // Set sizes and strides explicitly
    void set_sizes_and_strides(IntArrayRef sizes, IntArrayRef strides) {
        PT_CHECK(sizes.size() == strides.size());

        sizes_.resize(sizes.size());
        strides_.resize(strides.size());

        for (size_t i = 0; i < sizes.size(); ++i) {
            sizes_[i] = sizes[i];
            strides_[i] = strides[i];
        }

        refresh_numel();
        refresh_contiguous();
    }

    void resize(IntArrayRef sizes) {
        set_sizes_contiguous(sizes);
    }

    // ========================================================================
    // Data type
    // ========================================================================

    ScalarType dtype() const { return dtype_; }

    void set_dtype(ScalarType dtype) {
        dtype_ = dtype;
    }

    size_t itemsize() const {
        return elementSize(dtype_);
    }

    // ========================================================================
    // Device
    // ========================================================================

    Device device() const {
        return storage_.device();
    }

    DeviceType device_type() const {
        return storage_.device_type();
    }

    bool is_cpu() const { return device_type() == DeviceType::CPU; }
    bool is_cuda() const { return device_type() == DeviceType::CUDA; }
    bool is_meta() const { return device_type() == DeviceType::Meta; }
    bool is_nmcard() const { return device_type() == DeviceType::PrivateUse1; }

    // ========================================================================
    // Autograd
    // ========================================================================

    bool requires_grad() const {
        return autograd_meta_ && autograd_meta_->requires_grad_;
    }

    void set_requires_grad(bool requires_grad) {
        if (requires_grad && !autograd_meta_) {
            // Use factory to create correct type (AutogradMetaImpl if registered)
            autograd_meta_ = create_autograd_meta();
        }
        if (autograd_meta_) {
            autograd_meta_->requires_grad_ = requires_grad;
        }
    }

    AutogradMeta* autograd_meta() const {
        return autograd_meta_.get();
    }

    void set_autograd_meta(std::unique_ptr<AutogradMeta> meta) {
        autograd_meta_ = std::move(meta);
    }

    bool is_leaf() const {
        return !autograd_meta_ || autograd_meta_->is_leaf_;
    }

    // ========================================================================
    // Flags
    // ========================================================================

    bool is_wrapped_number() const { return is_wrapped_number_; }

    void set_wrapped_number(bool value) {
        is_wrapped_number_ = value;
    }

    bool allow_tensor_metadata_change() const {
        return allow_tensor_metadata_change_;
    }

    void set_allow_tensor_metadata_change(bool value) {
        allow_tensor_metadata_change_ = value;
    }

    // Trusted flag: when true, ops skip dtype/contiguous/device checks.
    // Set for tensors known to be float32, contiguous, CPU by construction.
    bool is_trusted() const { return trusted_; }
    void set_trusted(bool t) { trusted_ = t; }

    // ========================================================================
    // Memory
    // ========================================================================

    size_t nbytes() const {
        return numel_ * itemsize();
    }

    // Make contiguous copy
    void make_contiguous() {
        if (is_contiguous_) return;

        size_t new_nbytes = numel_ * itemsize();
        Storage new_storage = Storage::create(new_nbytes, device());

        // Copy with stride-based indexing
        char* dst = static_cast<char*>(new_storage.data());
        const char* src = static_cast<const char*>(data());
        size_t elem_size = itemsize();

        for (int64_t i = 0; i < numel_; ++i) {
            int64_t src_offset = 0;
            int64_t remaining = i;
            for (int64_t d = sizes_.size() - 1; d >= 0; --d) {
                int64_t idx = remaining % sizes_[d];
                remaining /= sizes_[d];
                src_offset += idx * strides_[d];
            }
            std::memcpy(dst + i * elem_size, src + src_offset * elem_size, elem_size);
        }

        storage_ = std::move(new_storage);
        storage_offset_ = 0;
        compute_contiguous_strides();
        is_contiguous_ = true;
    }

    // ========================================================================
    // Copy
    // ========================================================================

    std::shared_ptr<TensorImpl> clone() const {
        Storage new_storage = Storage::create(nbytes(), device());

        auto impl = std::make_shared<TensorImpl>(
            std::move(new_storage),
            dtype_,
            sizes()
        );

        // Copy data
        if (is_contiguous_) {
            std::memcpy(impl->mutable_data(), data(), nbytes());
        } else {
            // Stride-based copy for non-contiguous source
            char* dst = static_cast<char*>(impl->mutable_data());
            const char* src = static_cast<const char*>(data());
            size_t elem_size = itemsize();

            for (int64_t i = 0; i < numel_; ++i) {
                int64_t src_offset = 0;
                int64_t remaining = i;
                for (int64_t d = sizes_.size() - 1; d >= 0; --d) {
                    int64_t idx = remaining % sizes_[d];
                    remaining /= sizes_[d];
                    src_offset += idx * strides_[d];
                }
                std::memcpy(dst + i * elem_size, src + src_offset * elem_size, elem_size);
            }
        }

        return impl;
    }

    // ========================================================================
    // Shallow copy (share storage)
    // ========================================================================

    std::shared_ptr<TensorImpl> shallow_copy() const {
        auto impl = std::make_shared<TensorImpl>(
            storage_,  // Share storage
            dtype_,
            sizes(),
            strides(),
            storage_offset_
        );

        if (autograd_meta_) {
            impl->set_requires_grad(requires_grad());
        }

        return impl;
    }

protected:
    // ========================================================================
    // Helper methods
    // ========================================================================

    int64_t maybe_wrap_dim(int64_t dim) const {
        int64_t ndim = this->dim();
        if (dim < 0) {
            dim = dim + ndim;
        }
        PT_CHECK(dim >= 0 && dim < ndim);
        return dim;
    }

    void refresh_numel() {
        numel_ = 1;
        for (size_t i = 0; i < sizes_.size(); ++i) {
            numel_ *= sizes_[i];
        }
    }

    // Compute strides for channels-last (NHWC) format
    void compute_channels_last_strides() {
        if (sizes_.size() != 4) {
            compute_contiguous_strides();
            return;
        }
        // NHWC: strides for [N, C, H, W] = {C*H*W, 1, W*C, C}
        strides_.resize(4);
        strides_[1] = 1;            // C is innermost
        strides_[3] = sizes_[1];    // W stride = C
        strides_[2] = strides_[3] * sizes_[3]; // H stride = W*C
        strides_[0] = strides_[2] * sizes_[2]; // N stride = H*W*C
    }

    // Compute strides for channels-last 3D (NDHWC) format
    void compute_channels_last_3d_strides() {
        if (sizes_.size() != 5) {
            compute_contiguous_strides();
            return;
        }
        strides_.resize(5);
        strides_[1] = 1;                       // C
        strides_[4] = sizes_[1];               // W stride = C
        strides_[3] = strides_[4] * sizes_[4]; // H stride = W*C
        strides_[2] = strides_[3] * sizes_[3]; // D stride = H*W*C
        strides_[0] = strides_[2] * sizes_[2]; // N stride = D*H*W*C
    }

    void compute_contiguous_strides() {
        if (sizes_.empty()) {
            strides_.clear();
            return;
        }

        strides_.resize(sizes_.size());
        int64_t stride = 1;
        for (int64_t i = static_cast<int64_t>(sizes_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= sizes_[i];
        }
    }

    void refresh_contiguous() {
        is_contiguous_ = compute_is_contiguous();
    }

    bool compute_is_contiguous() const {
        if (numel_ == 0) return true;
        if (dim() == 0) return true;

        int64_t expected_stride = 1;
        for (int64_t i = dim() - 1; i >= 0; --i) {
            if (sizes_[i] != 1) {
                if (strides_[i] != expected_stride) {
                    return false;
                }
                expected_stride *= sizes_[i];
            }
        }
        return true;
    }

protected:
    // Storage
    Storage storage_;
    int64_t storage_offset_;

    // Shape
    SmallVector<int64_t, 5> sizes_;
    SmallVector<int64_t, 5> strides_;
    int64_t numel_;

    // Type
    ScalarType dtype_;

    // Flags
    bool is_contiguous_;
    bool is_wrapped_number_;
    bool allow_tensor_metadata_change_;
    bool trusted_ = false;  // When true, ops skip type/contiguous/device checks

    // Autograd metadata
    std::unique_ptr<AutogradMeta> autograd_meta_;

    // Reference count
    std::atomic<int64_t> ref_count_;
};

// ============================================================================
// Factory Functions for TensorImpl
// ============================================================================

inline std::shared_ptr<TensorImpl> make_tensor_impl(
    IntArrayRef sizes,
    ScalarType dtype = ScalarType::Float,
    Device device = kCPU
) {
    // Calculate total bytes
    int64_t numel = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        numel *= sizes[i];
    }
    size_t nbytes = numel * elementSize(dtype);

    // Create storage
    Storage storage = Storage::create(nbytes, device);

    // Create tensor impl
    auto impl = std::make_shared<TensorImpl>(
        std::move(storage),
        dtype,
        sizes
    );

    return impl;
}

inline std::shared_ptr<TensorImpl> make_empty_tensor_impl(
    ScalarType dtype = ScalarType::Float,
    Device device = kCPU
) {
    Storage storage = Storage::create(0, device);
    return std::make_shared<TensorImpl>(std::move(storage), dtype);
}

} // namespace c10
