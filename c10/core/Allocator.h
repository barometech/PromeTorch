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

        // Round up to bucket size for better cache hit rate
        size_t alloc_size = round_to_bucket(nbytes);

        // Try cache first
        void* data = cache_pop(alloc_size);

        if (data == nullptr) {
            // Cache miss: real allocation
            #if defined(_MSC_VER)
                data = _aligned_malloc(alloc_size, kAlignment);
            #else
                int ret = posix_memalign(&data, kAlignment, alloc_size);
                if (ret != 0) data = nullptr;
            #endif

            if (data == nullptr) {
                PT_OOM_ERROR(
                    "Failed to allocate ", nbytes, " bytes on CPU. "
                    "Out of memory?"
                );
            }
        }

        // Store alloc_size in context for Delete to know the bucket
        return DataPtr(data, reinterpret_cast<void*>(alloc_size),
                       &CPUAllocator::CachedDelete, kCPU);
    }

    void* raw_allocate(size_t nbytes) override {
        if (nbytes == 0) return nullptr;

        void* data = nullptr;
        #if defined(_MSC_VER)
            data = _aligned_malloc(nbytes, kAlignment);
        #else
            int ret = posix_memalign(&data, kAlignment, kAlignment > nbytes ? kAlignment : nbytes);
            if (ret != 0) data = nullptr;
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
        return &CPUAllocator::CachedDelete;
    }

    // Empty the cache (e.g., to reclaim memory)
    void empty_cache() {
        for (int b = 0; b < kNumBuckets; ++b) {
            for (int i = 0; i < bucket_count_[b]; ++i) {
                if (buckets_[b][i]) {
                    #if defined(_MSC_VER)
                        _aligned_free(buckets_[b][i]);
                    #else
                        free(buckets_[b][i]);
                    #endif
                    buckets_[b][i] = nullptr;
                }
            }
            bucket_count_[b] = 0;
        }
    }

private:
    CPUAllocator() {
        for (int b = 0; b < kNumBuckets; ++b) {
            bucket_count_[b] = 0;
            for (int i = 0; i < kMaxPerBucket; ++i)
                buckets_[b][i] = nullptr;
        }
    }

    ~CPUAllocator() {
        empty_cache();
    }

    static constexpr size_t kAlignment = 64;

    // Cache: up to kMaxPerBucket blocks per size bucket
    // Buckets: powers of 2 from 2^10 (1KB) to 2^26 (64MB) = 17 buckets
    static constexpr int kMinBucketLog2 = 10;   // 1 KB
    static constexpr int kMaxBucketLog2 = 26;   // 64 MB
    static constexpr int kNumBuckets = kMaxBucketLog2 - kMinBucketLog2 + 1; // 17
    static constexpr int kMaxPerBucket = 16;
    static constexpr size_t kMaxCacheableSize = (1ULL << kMaxBucketLog2); // 64 MB

    void* buckets_[kNumBuckets][kMaxPerBucket];
    int bucket_count_[kNumBuckets];

    // Round up to next power of 2 (for bucket matching)
    static size_t round_to_bucket(size_t n) {
        if (n <= (1ULL << kMinBucketLog2)) return (1ULL << kMinBucketLog2);
        if (n > kMaxCacheableSize) return n; // Don't bucket huge allocs
        // Next power of 2
        n--;
        n |= n >> 1;  n |= n >> 2;  n |= n >> 4;
        n |= n >> 8;  n |= n >> 16; n |= n >> 32;
        return n + 1;
    }

    // Bucket index for a power-of-2 size
    static int bucket_index(size_t alloc_size) {
        if (alloc_size > kMaxCacheableSize) return -1;
        // log2 via bit scan
        int idx = 0;
        size_t s = alloc_size;
        while (s > 1) { s >>= 1; idx++; }
        return idx - kMinBucketLog2;
    }

    // Pop a cached block of given size
    void* cache_pop(size_t alloc_size) {
        int idx = bucket_index(alloc_size);
        if (idx < 0 || idx >= kNumBuckets) return nullptr;
        if (bucket_count_[idx] > 0) {
            bucket_count_[idx]--;
            void* ptr = buckets_[idx][bucket_count_[idx]];
            buckets_[idx][bucket_count_[idx]] = nullptr;
            return ptr;
        }
        return nullptr;
    }

    // Push a block to cache
    bool cache_push(void* ptr, size_t alloc_size) {
        int idx = bucket_index(alloc_size);
        if (idx < 0 || idx >= kNumBuckets) return false;
        if (bucket_count_[idx] < kMaxPerBucket) {
            buckets_[idx][bucket_count_[idx]] = ptr;
            bucket_count_[idx]++;
            return true;
        }
        return false; // Bucket full
    }

    static void CachedDelete(void* data, void* context) {
        if (data == nullptr) return;
        size_t alloc_size = reinterpret_cast<size_t>(context);
        // Try to cache instead of freeing
        if (alloc_size > 0 && CPUAllocator::get().cache_push(data, alloc_size)) {
            return; // Cached successfully
        }
        // Cache full or uncacheable: actually free
        #if defined(_MSC_VER)
            _aligned_free(data);
        #else
            free(data);
        #endif
    }

    // Legacy non-cached delete for backward compat
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
