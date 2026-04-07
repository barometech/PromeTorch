#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
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
    // ---- Allocation statistics ----
    std::atomic<int64_t> total_allocs_{0};
    std::atomic<int64_t> cache_hits_{0};
    std::atomic<int64_t> arena_hits_{0};
    std::atomic<int64_t> tl_cache_hits_{0};

    static CPUAllocator& get() {
        static CPUAllocator instance;
        return instance;
    }

    DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return DataPtr(nullptr, nullptr, nullptr, kCPU);
        }

        total_allocs_.fetch_add(1, std::memory_order_relaxed);

        // ---- Fast path: small arena allocation (<= kArenaMaxAlloc) ----
        if (nbytes <= kArenaMaxAlloc) {
            void* data = arena_alloc(nbytes);
            if (data) {
                arena_hits_.fetch_add(1, std::memory_order_relaxed);
                // Context encodes: high bit set = arena marker, low bits = size
                // We use kArenaMarker | nbytes as context so CachedDelete knows
                // this came from the arena and should NOT be freed/cached
                size_t ctx = kArenaMarker | nbytes;
                return DataPtr(data, reinterpret_cast<void*>(ctx),
                               &CPUAllocator::CachedDelete, kCPU);
            }
            // Arena full/exhausted: fall through to normal path
        }

        // Round up to bucket size for better cache hit rate
        size_t alloc_size = round_to_bucket(nbytes);

        // ---- Fast path: thread-local cache (no mutex) ----
        void* data = tl_cache_pop(alloc_size);
        if (data) {
            tl_cache_hits_.fetch_add(1, std::memory_order_relaxed);
            return DataPtr(data, reinterpret_cast<void*>(alloc_size),
                           &CPUAllocator::CachedDelete, kCPU);
        }

        // ---- Medium path: global bucket cache (with mutex) ----
        data = cache_pop(alloc_size);
        if (data) {
            cache_hits_.fetch_add(1, std::memory_order_relaxed);
            return DataPtr(data, reinterpret_cast<void*>(alloc_size),
                           &CPUAllocator::CachedDelete, kCPU);
        }

        // ---- Slow path: OS allocation ----
        data = aligned_alloc_impl(alloc_size);
        if (data == nullptr) {
            PT_OOM_ERROR(
                "Failed to allocate ", nbytes, " bytes on CPU. "
                "Out of memory?"
            );
        }

        // Store alloc_size in context for Delete to know the bucket
        return DataPtr(data, reinterpret_cast<void*>(alloc_size),
                       &CPUAllocator::CachedDelete, kCPU);
    }

    void* raw_allocate(size_t nbytes) override {
        if (nbytes == 0) return nullptr;

        void* data = aligned_alloc_impl(nbytes);
        if (data == nullptr) {
            PT_OOM_ERROR("Failed to allocate ", nbytes, " bytes on CPU");
        }
        return data;
    }

    void raw_deallocate(void* ptr) override {
        if (ptr == nullptr) return;
        aligned_free_impl(ptr);
    }

    DeleterFn raw_deleter() const override {
        return &CPUAllocator::CachedDelete;
    }

    // Empty the cache (e.g., to reclaim memory)
    void empty_cache() {
        // Empty global bucket cache
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (int b = 0; b < kNumBuckets; ++b) {
            for (int i = 0; i < bucket_count_[b]; ++i) {
                if (buckets_[b][i]) {
                    aligned_free_impl(buckets_[b][i]);
                    buckets_[b][i] = nullptr;
                }
            }
            bucket_count_[b] = 0;
        }
    }

    // Print allocation statistics
    void print_stats() const {
        int64_t total = total_allocs_.load(std::memory_order_relaxed);
        int64_t hits = cache_hits_.load(std::memory_order_relaxed);
        int64_t arena = arena_hits_.load(std::memory_order_relaxed);
        int64_t tl = tl_cache_hits_.load(std::memory_order_relaxed);
        int64_t mallocs = total - hits - arena - tl;
        fprintf(stderr, "[CPUAllocator] total=%lld tl_cache=%lld global_cache=%lld arena=%lld malloc=%lld (%.1f%% hit rate)\n",
                (long long)total, (long long)tl, (long long)hits, (long long)arena, (long long)mallocs,
                total > 0 ? 100.0 * (hits + arena + tl) / total : 0.0);
    }

    void reset_stats() {
        total_allocs_.store(0, std::memory_order_relaxed);
        cache_hits_.store(0, std::memory_order_relaxed);
        arena_hits_.store(0, std::memory_order_relaxed);
        tl_cache_hits_.store(0, std::memory_order_relaxed);
    }

private:
    CPUAllocator() {
        for (int b = 0; b < kNumBuckets; ++b) {
            bucket_count_[b] = 0;
            for (int i = 0; i < kMaxPerBucket; ++i)
                buckets_[b][i] = nullptr;
        }
        // Pre-allocate the small-tensor arena
        arena_ = static_cast<char*>(aligned_alloc_impl(kArenaSize));
        arena_offset_ = 0;
    }

    ~CPUAllocator() {
        empty_cache();
        // Free arena (but not individual arena allocations - they're sub-regions)
        if (arena_) {
            aligned_free_impl(arena_);
            arena_ = nullptr;
        }
    }

    // ---- Platform-specific aligned alloc/free ----
    static void* aligned_alloc_impl(size_t nbytes) {
        void* data = nullptr;
        #if defined(_MSC_VER)
            data = _aligned_malloc(nbytes, kAlignment);
        #else
            int ret = posix_memalign(&data, kAlignment, nbytes < kAlignment ? kAlignment : nbytes);
            if (ret != 0) data = nullptr;
        #endif
        return data;
    }

    static void aligned_free_impl(void* ptr) {
        #if defined(_MSC_VER)
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    static constexpr size_t kAlignment = 64;

    // ========================================================================
    // Small-tensor arena: one big malloc, carve out pieces
    // Eliminates syscall for small tensors (grad buffers, scalars, small intermediates)
    // Arena allocations are bump-pointer, never individually freed.
    // When arena is exhausted, falls through to bucket cache / malloc.
    // ========================================================================
    static constexpr size_t kArenaSize = 16 * 1024 * 1024;     // 16MB
    static constexpr size_t kArenaMaxAlloc = 4096;              // Max 4KB per arena alloc
    static constexpr size_t kArenaMarker = size_t(1) << (sizeof(size_t) * 8 - 1); // High bit

    char* arena_ = nullptr;
    std::atomic<size_t> arena_offset_{0};

    // Lock-free bump-pointer arena allocation (aligned to kAlignment)
    void* arena_alloc(size_t nbytes) {
        if (!arena_) return nullptr;
        // FIX 5.1: overflow check before alignment
        if (nbytes > SIZE_MAX - kAlignment) return nullptr;
        size_t aligned_nbytes = (nbytes + kAlignment - 1) & ~(kAlignment - 1);
        // Atomic bump
        size_t old_offset = arena_offset_.load(std::memory_order_relaxed);
        while (true) {
            if (old_offset + aligned_nbytes > kArenaSize) return nullptr; // Exhausted
            if (arena_offset_.compare_exchange_weak(old_offset, old_offset + aligned_nbytes,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                return arena_ + old_offset;
            }
        }
    }

    // Check if pointer is inside the arena
    bool is_arena_ptr(void* ptr) const {
        if (!arena_) return false;
        char* p = static_cast<char*>(ptr);
        return p >= arena_ && p < arena_ + kArenaSize;
    }

    // ========================================================================
    // Thread-local free list for hot sizes (no mutex needed)
    // Avoids mutex contention entirely for repeated alloc/free of same size
    // ========================================================================
    static constexpr int kTLCacheSlots = 64;

    struct TLCacheEntry {
        void* ptr;
        size_t size;
    };

    struct TLCache {
        TLCacheEntry entries[kTLCacheSlots];
        int count = 0;

        TLCache() {
            for (int i = 0; i < kTLCacheSlots; ++i) {
                entries[i].ptr = nullptr;
                entries[i].size = 0;
            }
        }

        ~TLCache() {
            // On thread exit, return all cached blocks to global cache or free them
            for (int i = 0; i < count; ++i) {
                if (entries[i].ptr) {
                    if (!CPUAllocator::get().cache_push(entries[i].ptr, entries[i].size)) {
                        aligned_free_impl(entries[i].ptr);
                    }
                    entries[i].ptr = nullptr;
                }
            }
            count = 0;
        }
    };

    static TLCache& get_tl_cache() {
        static thread_local TLCache tl;
        return tl;
    }

    // Pop from thread-local cache (exact size match, LIFO for temporal locality)
    static void* tl_cache_pop(size_t alloc_size) {
        TLCache& tl = get_tl_cache();
        // Search from end (LIFO) for matching size
        for (int i = tl.count - 1; i >= 0; --i) {
            if (tl.entries[i].size == alloc_size) {
                void* ptr = tl.entries[i].ptr;
                // Move last entry to fill the gap
                tl.count--;
                if (i < tl.count) {
                    tl.entries[i] = tl.entries[tl.count];
                }
                tl.entries[tl.count].ptr = nullptr;
                return ptr;
            }
        }
        return nullptr;
    }

    // Push to thread-local cache
    static bool tl_cache_push(void* ptr, size_t alloc_size) {
        TLCache& tl = get_tl_cache();
        if (tl.count < kTLCacheSlots) {
            tl.entries[tl.count].ptr = ptr;
            tl.entries[tl.count].size = alloc_size;
            tl.count++;
            return true;
        }
        return false; // TL cache full
    }

    // ========================================================================
    // Global bucket cache (with mutex)
    // Buckets: powers of 2 from 2^5 (32B) to 2^36 (64GB) = 32 buckets
    // ========================================================================
    static constexpr int kMinBucketLog2 = 5;    // 32 B
    static constexpr int kMaxBucketLog2 = 36;   // 64 GB
    static constexpr int kNumBuckets = kMaxBucketLog2 - kMinBucketLog2 + 1; // 32
    static constexpr int kMaxPerBucket = 256;
    static constexpr size_t kMaxCacheableSize = (1ULL << kMaxBucketLog2);

    void* buckets_[kNumBuckets][kMaxPerBucket];
    int bucket_count_[kNumBuckets];
    std::mutex cache_mutex_;

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

    // Pop a cached block of given size (global, mutex-protected)
    void* cache_pop(size_t alloc_size) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
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

    // Push a block to global cache (mutex-protected)
    bool cache_push(void* ptr, size_t alloc_size) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
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
        size_t ctx_val = reinterpret_cast<size_t>(context);

        // Check if this is an arena allocation (high bit set)
        if (ctx_val & kArenaMarker) {
            // Arena memory: never freed individually, the arena block is freed on shutdown.
            // Just let it leak back into the arena (bump allocator, no reuse of individual slots).
            return;
        }

        size_t alloc_size = ctx_val;

        // Fast path: try thread-local cache first (no mutex)
        if (alloc_size > 0 && tl_cache_push(data, alloc_size)) {
            return;
        }

        // Medium path: try global bucket cache (with mutex)
        if (alloc_size > 0 && CPUAllocator::get().cache_push(data, alloc_size)) {
            return;
        }

        // Slow path: cache full or uncacheable, actually free
        aligned_free_impl(data);
    }

    // Legacy non-cached delete for backward compat
    static void Delete(void* data, void* /*context*/) {
        if (data == nullptr) return;
        aligned_free_impl(data);
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
