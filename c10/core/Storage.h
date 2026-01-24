#pragma once

#include <atomic>
#include <memory>
#include <cstddef>
#include <cstring>
#include "c10/macros/Macros.h"
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/util/Exception.h"

namespace c10 {

// ============================================================================
// StorageImpl - Implementation of Storage
// ============================================================================

class PT_API StorageImpl {
public:
    // Constructor with allocator
    StorageImpl(
        size_t nbytes,
        DataPtr data_ptr,
        Allocator* allocator,
        bool resizable = false
    )
        : data_ptr_(std::move(data_ptr))
        , nbytes_(nbytes)
        , allocator_(allocator)
        , resizable_(resizable)
        , received_cuda_(false)
        , ref_count_(1)
    {}

    // Constructor that allocates memory
    StorageImpl(
        size_t nbytes,
        Allocator* allocator,
        bool resizable = false
    )
        : data_ptr_(allocator->allocate(nbytes))
        , nbytes_(nbytes)
        , allocator_(allocator)
        , resizable_(resizable)
        , received_cuda_(false)
        , ref_count_(1)
    {}

    // No copy
    StorageImpl(const StorageImpl&) = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;

    // Move constructor
    StorageImpl(StorageImpl&& other) noexcept
        : data_ptr_(std::move(other.data_ptr_))
        , nbytes_(other.nbytes_)
        , allocator_(other.allocator_)
        , resizable_(other.resizable_)
        , received_cuda_(other.received_cuda_)
        , ref_count_(1)  // New storage, new ref count
    {
        other.nbytes_ = 0;
        other.allocator_ = nullptr;
    }

    // Destructor
    ~StorageImpl() = default;

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
        return data_ptr_.cast<T>();
    }

    void* data() const {
        return data_ptr_.get();
    }

    template<typename T>
    T* mutable_data() {
        return static_cast<T*>(data_ptr_.mutable_get());
    }

    void* mutable_data() {
        return data_ptr_.mutable_get();
    }

    const DataPtr& data_ptr() const {
        return data_ptr_;
    }

    DataPtr& data_ptr() {
        return data_ptr_;
    }

    // Set data pointer (takes ownership)
    void set_data_ptr(DataPtr data_ptr) {
        data_ptr_ = std::move(data_ptr);
    }

    void set_data_ptr_noswap(DataPtr data_ptr) {
        data_ptr_ = std::move(data_ptr);
    }

    // ========================================================================
    // Size and capacity
    // ========================================================================

    size_t nbytes() const {
        return nbytes_;
    }

    void set_nbytes(size_t nbytes) {
        nbytes_ = nbytes;
    }

    // ========================================================================
    // Allocator
    // ========================================================================

    Allocator* allocator() const {
        return allocator_;
    }

    void set_allocator(Allocator* allocator) {
        allocator_ = allocator;
    }

    // ========================================================================
    // Device
    // ========================================================================

    Device device() const {
        return data_ptr_.device();
    }

    DeviceType device_type() const {
        return data_ptr_.device().type();
    }

    // ========================================================================
    // Resizable flag
    // ========================================================================

    bool resizable() const {
        return resizable_;
    }

    void set_resizable(bool resizable) {
        resizable_ = resizable;
    }

    // ========================================================================
    // CUDA received flag (for async operations)
    // ========================================================================

    bool received_cuda() const {
        return received_cuda_;
    }

    void set_received_cuda(bool received) {
        received_cuda_ = received;
    }

    // ========================================================================
    // Resize
    // ========================================================================

    void resize(size_t new_nbytes) {
        PT_CHECK_MSG(resizable_,
            "Cannot resize non-resizable storage");
        PT_CHECK_MSG(allocator_ != nullptr,
            "Cannot resize storage without allocator");

        if (new_nbytes == nbytes_) {
            return;
        }

        DataPtr new_data = allocator_->allocate(new_nbytes);

        // Copy existing data
        if (data_ptr_.get() != nullptr && new_data.get() != nullptr) {
            size_t copy_size = std::min(nbytes_, new_nbytes);
            std::memcpy(new_data.mutable_get(), data_ptr_.get(), copy_size);
        }

        data_ptr_ = std::move(new_data);
        nbytes_ = new_nbytes;
    }

private:
    DataPtr data_ptr_;
    size_t nbytes_;
    Allocator* allocator_;
    bool resizable_;
    bool received_cuda_;
    std::atomic<int64_t> ref_count_;
};

// ============================================================================
// Storage - Reference-counted handle to StorageImpl
// ============================================================================

class PT_API Storage {
public:
    // Default constructor - null storage
    Storage() : impl_(nullptr) {}

    // Constructor from StorageImpl pointer (takes ownership)
    explicit Storage(StorageImpl* impl) : impl_(impl) {}

    // Constructor that creates new storage
    Storage(size_t nbytes, Allocator* allocator, bool resizable = false)
        : impl_(new StorageImpl(nbytes, allocator, resizable)) {}

    // Constructor with pre-allocated data
    Storage(size_t nbytes, DataPtr data_ptr, Allocator* allocator, bool resizable = false)
        : impl_(new StorageImpl(nbytes, std::move(data_ptr), allocator, resizable)) {}

    // Copy constructor (increases ref count)
    Storage(const Storage& other) : impl_(other.impl_) {
        if (impl_) {
            impl_->retain();
        }
    }

    // Move constructor
    Storage(Storage&& other) noexcept : impl_(other.impl_) {
        other.impl_ = nullptr;
    }

    // Copy assignment
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            if (impl_) {
                impl_->release();
            }
            impl_ = other.impl_;
            if (impl_) {
                impl_->retain();
            }
        }
        return *this;
    }

    // Move assignment
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            if (impl_) {
                impl_->release();
            }
            impl_ = other.impl_;
            other.impl_ = nullptr;
        }
        return *this;
    }

    // Destructor
    ~Storage() {
        if (impl_) {
            impl_->release();
        }
    }

    // ========================================================================
    // Access to implementation
    // ========================================================================

    StorageImpl* unsafeGetStorageImpl() const {
        return impl_;
    }

    StorageImpl* unsafe_release() {
        StorageImpl* result = impl_;
        impl_ = nullptr;
        return result;
    }

    bool defined() const {
        return impl_ != nullptr;
    }

    explicit operator bool() const {
        return defined();
    }

    // ========================================================================
    // Forwarding methods
    // ========================================================================

    template<typename T>
    T* data() const {
        PT_CHECK(impl_ != nullptr);
        return impl_->data<T>();
    }

    void* data() const {
        PT_CHECK(impl_ != nullptr);
        return impl_->data();
    }

    template<typename T>
    T* mutable_data() {
        PT_CHECK(impl_ != nullptr);
        return impl_->mutable_data<T>();
    }

    const DataPtr& data_ptr() const {
        PT_CHECK(impl_ != nullptr);
        return impl_->data_ptr();
    }

    size_t nbytes() const {
        PT_CHECK(impl_ != nullptr);
        return impl_->nbytes();
    }

    Allocator* allocator() const {
        PT_CHECK(impl_ != nullptr);
        return impl_->allocator();
    }

    Device device() const {
        PT_CHECK(impl_ != nullptr);
        return impl_->device();
    }

    DeviceType device_type() const {
        PT_CHECK(impl_ != nullptr);
        return impl_->device_type();
    }

    bool resizable() const {
        PT_CHECK(impl_ != nullptr);
        return impl_->resizable();
    }

    int64_t use_count() const {
        if (impl_ == nullptr) return 0;
        return impl_->use_count();
    }

    bool unique() const {
        if (impl_ == nullptr) return false;
        return impl_->unique();
    }

    // ========================================================================
    // Modifications
    // ========================================================================

    void set_data_ptr(DataPtr data_ptr) {
        PT_CHECK(impl_ != nullptr);
        impl_->set_data_ptr(std::move(data_ptr));
    }

    void set_nbytes(size_t nbytes) {
        PT_CHECK(impl_ != nullptr);
        impl_->set_nbytes(nbytes);
    }

    void resize(size_t new_nbytes) {
        PT_CHECK(impl_ != nullptr);
        impl_->resize(new_nbytes);
    }

    // ========================================================================
    // Static factory methods
    // ========================================================================

    static Storage create(size_t nbytes, Device device = kCPU, bool resizable = false) {
        Allocator* allocator = GetAllocator(device);
        return Storage(nbytes, allocator, resizable);
    }

    static Storage create_with_data(
        void* data,
        size_t nbytes,
        DeleterFn deleter,
        Device device = kCPU,
        bool resizable = false
    ) {
        DataPtr data_ptr(data, nullptr, deleter, device);
        Allocator* allocator = GetAllocator(device);
        return Storage(nbytes, std::move(data_ptr), allocator, resizable);
    }

private:
    StorageImpl* impl_;
};

// ============================================================================
// Comparison operators
// ============================================================================

inline bool operator==(const Storage& lhs, const Storage& rhs) {
    return lhs.unsafeGetStorageImpl() == rhs.unsafeGetStorageImpl();
}

inline bool operator!=(const Storage& lhs, const Storage& rhs) {
    return !(lhs == rhs);
}

} // namespace c10
