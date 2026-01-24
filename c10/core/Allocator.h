#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <functional>
#include <atomic>
#include "c10/macros/Macros.h"
#include "c10/core/Device.h"
#include "c10/util/Exception.h"

namespace c10 {

// ============================================================================
// DataPtr - Smart Pointer for Allocated Data
// ============================================================================

// Forward declaration
class Allocator;

// Deleter type for DataPtr
using DeleterFn = void (*)(void* data, void* context);

class PT_API DataPtr {
public:
    // Default constructor - null pointer
    DataPtr() : ptr_(nullptr), ctx_(nullptr), deleter_(nullptr), device_(kCPU) {}

    // Constructor with data, context, deleter and device
    DataPtr(void* data, void* ctx, DeleterFn deleter, Device device)
        : ptr_(data), ctx_(ctx), deleter_(deleter), device_(device) {}

    // Move-only semantics
    DataPtr(DataPtr&& other) noexcept
        : ptr_(other.ptr_)
        , ctx_(other.ctx_)
        , deleter_(other.deleter_)
        , device_(other.device_) {
        other.ptr_ = nullptr;
        other.ctx_ = nullptr;
        other.deleter_ = nullptr;
    }

    DataPtr& operator=(DataPtr&& other) noexcept {
        if (this != &other) {
            clear();
            ptr_ = other.ptr_;
            ctx_ = other.ctx_;
            deleter_ = other.deleter_;
            device_ = other.device_;
            other.ptr_ = nullptr;
            other.ctx_ = nullptr;
            other.deleter_ = nullptr;
        }
        return *this;
    }

    // No copy
    DataPtr(const DataPtr&) = delete;
    DataPtr& operator=(const DataPtr&) = delete;

    // Destructor - calls deleter
    ~DataPtr() {
        clear();
    }

    // Clear/release the pointer
    void clear() {
        if (deleter_ && ptr_) {
            deleter_(ptr_, ctx_);
        }
        ptr_ = nullptr;
        ctx_ = nullptr;
        deleter_ = nullptr;
    }

    // Release ownership without calling deleter
    void* release_context() {
        void* result = ctx_;
        ptr_ = nullptr;
        ctx_ = nullptr;
        deleter_ = nullptr;
        return result;
    }

    // Accessors
    void* get() const noexcept { return ptr_; }
    void* get_context() const noexcept { return ctx_; }
    DeleterFn get_deleter() const noexcept { return deleter_; }
    Device device() const noexcept { return device_; }

    // Mutable access (be careful!)
    void* mutable_get() noexcept { return ptr_; }

    // Boolean conversion
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    // Pointer-like access
    template<typename T>
    T* cast() const {
        return static_cast<T*>(ptr_);
    }

    // Compare with raw pointer
    bool operator==(const void* other) const noexcept { return ptr_ == other; }
    bool operator!=(const void* other) const noexcept { return ptr_ != other; }

private:
    void* ptr_;           // Pointer to data
    void* ctx_;           // Context for deleter (e.g., Allocator pointer)
    DeleterFn deleter_;   // Function to delete data
    Device device_;       // Device where data resides
};

// ============================================================================
// Allocator - Abstract Base Class
// ============================================================================

class PT_API Allocator {
public:
    virtual ~Allocator() = default;

    // Allocate nbytes of memory
    virtual DataPtr allocate(size_t nbytes) = 0;

    // Raw allocation (without DataPtr wrapper)
    virtual void* raw_allocate(size_t nbytes) {
        DataPtr ptr = allocate(nbytes);
        // Note: This leaks the context! Use with caution.
        void* data = ptr.get();
        ptr.release_context();
        return data;
    }

    // Raw deallocation
    virtual void raw_deallocate(void* ptr) = 0;

    // Delete function for DataPtr
    virtual DeleterFn raw_deleter() const = 0;
};

// ============================================================================
// CPU Allocator Implementation
// ============================================================================

class PT_API CPUAllocator : public Allocator {
public:
    static CPUAllocator& get() {
        static CPUAllocator instance;
        return instance;
    }

    DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return DataPtr(nullptr, nullptr, nullptr, kCPU);
        }

        void* data = nullptr;

        #if defined(_MSC_VER)
            // Windows aligned allocation
            data = _aligned_malloc(nbytes, kAlignment);
        #else
            // POSIX aligned allocation
            int ret = posix_memalign(&data, kAlignment, nbytes);
            if (ret != 0) {
                data = nullptr;
            }
        #endif

        if (data == nullptr) {
            PT_OOM_ERROR(
                "Failed to allocate ", nbytes, " bytes on CPU. "
                "Out of memory?"
            );
        }

        return DataPtr(data, nullptr, &CPUAllocator::Delete, kCPU);
    }

    void* raw_allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return nullptr;
        }

        void* data = nullptr;

        #if defined(_MSC_VER)
            data = _aligned_malloc(nbytes, kAlignment);
        #else
            int ret = posix_memalign(&data, kAlignment, kAlignment > nbytes ? kAlignment : nbytes);
            if (ret != 0) {
                data = nullptr;
            }
        #endif

        if (data == nullptr) {
            PT_OOM_ERROR("Failed to allocate ", nbytes, " bytes on CPU");
        }

        return data;
    }

    void raw_deallocate(void* ptr) override {
        if (ptr == nullptr) return;

        #if defined(_MSC_VER)
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    DeleterFn raw_deleter() const override {
        return &CPUAllocator::Delete;
    }

private:
    CPUAllocator() = default;

    // Alignment for SIMD operations (AVX-512 requires 64-byte alignment)
    static constexpr size_t kAlignment = 64;

    static void Delete(void* data, void* /*context*/) {
        if (data == nullptr) return;

        #if defined(_MSC_VER)
            _aligned_free(data);
        #else
            free(data);
        #endif
    }
};

// ============================================================================
// Pinned Memory Allocator (for fast CPU-GPU transfers)
// ============================================================================

class PT_API PinnedMemoryAllocator : public Allocator {
public:
    static PinnedMemoryAllocator& get() {
        static PinnedMemoryAllocator instance;
        return instance;
    }

    DataPtr allocate(size_t nbytes) override;
    void raw_deallocate(void* ptr) override;
    DeleterFn raw_deleter() const override;

private:
    PinnedMemoryAllocator() = default;
    static void Delete(void* data, void* context);
};

// ============================================================================
// Allocator Registry
// ============================================================================

class PT_API AllocatorRegistry {
public:
    static AllocatorRegistry& get() {
        static AllocatorRegistry instance;
        return instance;
    }

    // Register allocator for a device type
    void registerAllocator(DeviceType type, Allocator* allocator) {
        PT_CHECK(static_cast<int>(type) < kNumDeviceTypes);
        allocators_[static_cast<int>(type)] = allocator;
    }

    // Get allocator for a device type
    Allocator* getAllocator(DeviceType type) const {
        PT_CHECK(static_cast<int>(type) < kNumDeviceTypes);
        Allocator* alloc = allocators_[static_cast<int>(type)];
        PT_CHECK_MSG(alloc != nullptr,
            std::string("No allocator registered for device type: ") + DeviceTypeName(type));
        return alloc;
    }

    // Check if allocator is registered
    bool hasAllocator(DeviceType type) const {
        if (static_cast<int>(type) >= kNumDeviceTypes) return false;
        return allocators_[static_cast<int>(type)] != nullptr;
    }

private:
    AllocatorRegistry() {
        // Initialize all to nullptr
        for (int i = 0; i < kNumDeviceTypes; ++i) {
            allocators_[i] = nullptr;
        }
        // Register CPU allocator by default
        allocators_[static_cast<int>(DeviceType::CPU)] = &CPUAllocator::get();
    }

    Allocator* allocators_[kNumDeviceTypes];
};

// ============================================================================
// Helper Functions
// ============================================================================

// Get the default allocator for a device
inline Allocator* GetAllocator(DeviceType type) {
    return AllocatorRegistry::get().getAllocator(type);
}

inline Allocator* GetAllocator(const Device& device) {
    return GetAllocator(device.type());
}

// Allocate memory on a device
inline DataPtr AllocateMemory(size_t nbytes, Device device) {
    return GetAllocator(device)->allocate(nbytes);
}

// ============================================================================
// InefficientStdFunctionContext - for lambdas as deleters
// ============================================================================

// This is less efficient but allows arbitrary deleters
class InefficientStdFunctionContext {
public:
    using Deleter = std::function<void(void*)>;

    InefficientStdFunctionContext(void* data, Deleter deleter)
        : data_(data), deleter_(std::move(deleter)) {}

    ~InefficientStdFunctionContext() {
        if (deleter_) {
            deleter_(data_);
        }
    }

    static DataPtr makeDataPtr(void* data, Deleter deleter, Device device) {
        auto* ctx = new InefficientStdFunctionContext(data, std::move(deleter));
        return DataPtr(data, ctx, &InefficientStdFunctionContext::Delete, device);
    }

private:
    void* data_;
    Deleter deleter_;

    static void Delete(void* /*data*/, void* context) {
        delete static_cast<InefficientStdFunctionContext*>(context);
    }
};

} // namespace c10
