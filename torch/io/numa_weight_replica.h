// NUMA weight replication for single-process CPU inference.
//
// Keeps N copies of a read-only weight buffer, one per NUMA node, so that
// thread-pool workers pinned to a node read from their node's local DDR at
// full per-channel bandwidth instead of contending for a single remote
// controller through the inter-chip interconnect.
//
// Enabled via env `PT_NUMA_REPLICATE=1`. When disabled or libnuma missing,
// `replicate()` just records a single pointer — all threads share it.
//
// Memory budget: model_size × num_nodes. For qwen3:4b Q4_K_M on 4-node
// Elbrus 8C2 this is 2.4 GB × 4 = 9.6 GB (fits easily in 125 GB).
//
// Combines with `thread_numa_node()` from ThreadPool.h — each parallel_for
// worker asks which node it's on and picks the right replica.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <array>

#if defined(__linux__) && !defined(_WIN32)
#  include <numa.h>
#  include <numaif.h>
#  include <sys/mman.h>
#  define PT_HAS_LIBNUMA 1
#else
#  define PT_HAS_LIBNUMA 0
#endif

namespace torch {
namespace io {

// Compile-time cap. Elbrus 8C2 has 4 nodes; other NUMA boxes up to 8.
inline constexpr int kMaxNumaNodes = 8;

// Global query: how many replicas should we allocate?
// 0 or 1 → no replication (single shared copy). N → per-node copies up to cap.
inline int numa_replicate_count() {
    const char* env = std::getenv("PT_NUMA_REPLICATE");
    if (!env || !env[0] || env[0] == '0') return 0;
#if PT_HAS_LIBNUMA
    if (numa_available() < 0) return 0;
    int nodes = numa_num_configured_nodes();
    if (nodes <= 1) return 0;
    if (nodes > kMaxNumaNodes) nodes = kMaxNumaNodes;
    return nodes;
#else
    return 0;
#endif
}

// Holds per-node copies of a raw weight buffer. `get(node)` returns the local
// copy; if replication is disabled or node is out of range, returns the
// fallback (original pointer). Non-owning if not replicated.
struct ReplicatedWeight {
    std::array<void*, kMaxNumaNodes> replicas{};  // null for unused slots
    void* fallback = nullptr;    // original pointer, used when no replication
    size_t size_bytes = 0;
    int num_replicas = 0;        // 0 = not replicated, fallback only
    bool owns_memory = false;    // true if we malloced replicas

    // Build from existing buffer. If replication disabled, stores fallback
    // only (no copy, no alloc). Otherwise allocates one copy per NUMA node
    // via numa_alloc_onnode + madvise(MADV_HUGEPAGE) and memcpy's the data.
    // Returns true on success; on any failure falls back to single pointer.
    bool replicate(const void* src, size_t nbytes) {
        fallback = const_cast<void*>(src);
        size_bytes = nbytes;
        num_replicas = 0;
        owns_memory = false;

        int n = numa_replicate_count();
        if (n <= 1) return true;  // no replication, but not an error

#if PT_HAS_LIBNUMA
        for (int i = 0; i < n; ++i) {
            void* p = numa_alloc_onnode(nbytes, i);
            if (!p) {
                std::fprintf(stderr,
                    "[NUMA] numa_alloc_onnode(%zu, node=%d) failed; rolling back\n",
                    nbytes, i);
                for (int j = 0; j < i; ++j) {
                    if (replicas[j]) numa_free(replicas[j], nbytes);
                    replicas[j] = nullptr;
                }
                return true;  // still ok — callers use fallback
            }
            // Request transparent huge pages (2 MB) for the region. Without
            // this, 2.4 GB of weights would need 600K 4-KB TLB entries per
            // rank, thrashing TLB on every inner loop iteration.
            (void)madvise(p, nbytes, MADV_HUGEPAGE);
            std::memcpy(p, src, nbytes);
            replicas[i] = p;
        }
        num_replicas = n;
        owns_memory = true;
#endif
        return true;
    }

    // Fast path: used inside parallel_for. If not replicated, returns fallback.
    inline const void* get(int node) const {
        if (num_replicas == 0 || node < 0 || node >= num_replicas) {
            return fallback;
        }
        void* r = replicas[node];
        return r ? r : fallback;
    }

    // Release all allocated replicas.
    void free() {
#if PT_HAS_LIBNUMA
        if (owns_memory) {
            for (int i = 0; i < num_replicas; ++i) {
                if (replicas[i]) {
                    numa_free(replicas[i], size_bytes);
                    replicas[i] = nullptr;
                }
            }
        }
#endif
        num_replicas = 0;
        owns_memory = false;
        fallback = nullptr;
        size_bytes = 0;
    }

    ~ReplicatedWeight() { free(); }

    ReplicatedWeight() = default;
    ReplicatedWeight(const ReplicatedWeight&) = delete;
    ReplicatedWeight& operator=(const ReplicatedWeight&) = delete;
    ReplicatedWeight(ReplicatedWeight&& o) noexcept { *this = std::move(o); }
    ReplicatedWeight& operator=(ReplicatedWeight&& o) noexcept {
        if (this != &o) {
            free();
            replicas = o.replicas;
            fallback = o.fallback;
            size_bytes = o.size_bytes;
            num_replicas = o.num_replicas;
            owns_memory = o.owns_memory;
            o.replicas.fill(nullptr);
            o.fallback = nullptr;
            o.size_bytes = 0;
            o.num_replicas = 0;
            o.owns_memory = false;
        }
        return *this;
    }
};

}  // namespace io
}  // namespace torch
