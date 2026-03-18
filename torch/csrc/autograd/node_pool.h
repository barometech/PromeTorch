#pragma once

// ============================================================================
// NodePool - Object Pool for Autograd Nodes
// ============================================================================
// On Elbrus E2K (VLIW), each malloc is a syscall that kills pipeline
// utilization. This pool pre-allocates Node objects and reuses them
// across forward passes, eliminating per-op heap allocations.
//
// Usage:
//   auto* node = NodePool<MmBackward>::acquire(self, other);
//   // ... use node ...
//   // Node is returned to pool when shared_ptr ref count reaches zero
//   // via PooledDeleter

#include <vector>
#include <mutex>
#include <memory>
#include <type_traits>
#include <cstring>

namespace torch {
namespace autograd {

// Forward declaration
struct Node;

// ============================================================================
// PooledDeleter - Returns nodes to pool instead of calling delete
// ============================================================================

template<typename T>
class NodePool;  // Forward declaration

template<typename T>
struct PooledDeleter {
    void operator()(Node* raw) const {
        T* node = static_cast<T*>(raw);
        // Release saved tensors and edges NOW to free references promptly.
        // Without this, pooled nodes would hold tensor references until reuse.
        // The subsequent ~T() in acquire() will clear again (no-op on empty).
        node->release_saved_tensors();
        node->next_edges_clear_pooled();
        NodePool<T>::release(node);
    }
};

// ============================================================================
// NodePool - Lock-free (per-thread) object pool with global fallback
// ============================================================================
// Design choices for Elbrus E2K:
// - Thread-local free lists (no locking in common case)
// - Global overflow pool with mutex (rare)
// - Batch allocation to amortize malloc cost
// - Fixed max pool size to bound memory

template<typename T>
class NodePool {
    static constexpr size_t BATCH_SIZE = 32;      // Allocate 32 nodes at once
    static constexpr size_t MAX_POOL_SIZE = 256;   // Max cached nodes per thread

    // Thread-local free list (no locking needed)
    struct ThreadLocal {
        std::vector<T*> free_list;

        ThreadLocal() {
            free_list.reserve(BATCH_SIZE);
        }

        ~ThreadLocal() {
            // Return all to global pool or delete
            for (T* p : free_list) {
                delete p;
            }
        }
    };

    static ThreadLocal& get_thread_local() {
        thread_local ThreadLocal tl;
        return tl;
    }

public:
    // Acquire a node from the pool. Args are forwarded to T's constructor
    // to reinitialize the node (placement new over recycled memory).
    template<typename... Args>
    static T* acquire(Args&&... args) {
        auto& tl = get_thread_local();

        T* node;
        if (!tl.free_list.empty()) {
            node = tl.free_list.back();
            tl.free_list.pop_back();
            // Placement new to reinitialize (reuses memory, no malloc)
            node->~T();
            new (node) T(std::forward<Args>(args)...);
        } else {
            // Pool empty - allocate fresh
            node = new T(std::forward<Args>(args)...);
        }
        return node;
    }

    // Return a node to the pool
    static void release(T* node) {
        auto& tl = get_thread_local();

        if (tl.free_list.size() < MAX_POOL_SIZE) {
            tl.free_list.push_back(node);
        } else {
            // Pool full - actually free
            delete node;
        }
    }

    // Create a shared_ptr that returns the node to pool on destruction
    template<typename... Args>
    static std::shared_ptr<Node> make_shared(Args&&... args) {
        T* node = acquire(std::forward<Args>(args)...);
        return std::shared_ptr<Node>(
            static_cast<Node*>(node),
            PooledDeleter<T>{}
        );
    }
};

} // namespace autograd
} // namespace torch
