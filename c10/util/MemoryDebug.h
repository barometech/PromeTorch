#pragma once

// ============================================================================
// Memory Debug Utilities for PromeTorch
// ============================================================================
// Enable with -DPT_MEMORY_DEBUG=1 at compile time
// Usage:
//   PT_MEM_CHECKPOINT("after forward");
//   PT_MEM_LOG_ALLOC(size, "tensor allocation");

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <atomic>
#include <mutex>

#ifdef PT_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace c10 {
namespace debug {

// Global memory tracking stats
struct MemoryStats {
    std::atomic<int64_t> cpu_allocs{0};
    std::atomic<int64_t> cpu_frees{0};
    std::atomic<int64_t> cpu_bytes_allocated{0};
    std::atomic<int64_t> cpu_bytes_freed{0};

    std::atomic<int64_t> cuda_allocs{0};
    std::atomic<int64_t> cuda_frees{0};
    std::atomic<int64_t> cuda_bytes_allocated{0};
    std::atomic<int64_t> cuda_bytes_freed{0};

    std::atomic<int64_t> tensor_count{0};
    std::atomic<int64_t> autograd_node_count{0};

    static MemoryStats& instance() {
        static MemoryStats stats;
        return stats;
    }

    void reset() {
        cpu_allocs = cpu_frees = cpu_bytes_allocated = cpu_bytes_freed = 0;
        cuda_allocs = cuda_frees = cuda_bytes_allocated = cuda_bytes_freed = 0;
        tensor_count = autograd_node_count = 0;
    }

    void print(const std::string& label = "") const {
        std::cout << "\n=== Memory Stats";
        if (!label.empty()) std::cout << " [" << label << "]";
        std::cout << " ===" << std::endl;
        std::cout << "CPU:  allocs=" << cpu_allocs << " frees=" << cpu_frees
                  << " net=" << (cpu_allocs - cpu_frees)
                  << " bytes=" << (cpu_bytes_allocated - cpu_bytes_freed) / 1048576.0 << " MB" << std::endl;
        std::cout << "CUDA: allocs=" << cuda_allocs << " frees=" << cuda_frees
                  << " net=" << (cuda_allocs - cuda_frees)
                  << " bytes=" << (cuda_bytes_allocated - cuda_bytes_freed) / 1048576.0 << " MB" << std::endl;
        std::cout << "Tensors: " << tensor_count << " AutogradNodes: " << autograd_node_count << std::endl;
    }
};

// CUDA memory query
inline void get_cuda_memory_info(size_t& used, size_t& total) {
#ifdef PT_USE_CUDA
    size_t free_mem = 0;
    cudaMemGetInfo(&free_mem, &total);
    used = total - free_mem;
#else
    used = total = 0;
#endif
}

// Memory checkpoint - prints current state
inline void memory_checkpoint(const std::string& label) {
    auto& stats = MemoryStats::instance();
    size_t cuda_used = 0, cuda_total = 0;
    get_cuda_memory_info(cuda_used, cuda_total);

    std::cout << "[MEM] " << label
              << " | CUDA: " << (cuda_used / 1048576.0) << "/" << (cuda_total / 1048576.0) << " MB"
              << " | Tensors: " << stats.tensor_count
              << " | Nodes: " << stats.autograd_node_count
              << std::endl;
}

// Log allocation
inline void log_alloc(int64_t bytes, const std::string& source, bool is_cuda) {
    auto& stats = MemoryStats::instance();
    if (is_cuda) {
        stats.cuda_allocs++;
        stats.cuda_bytes_allocated += bytes;
    } else {
        stats.cpu_allocs++;
        stats.cpu_bytes_allocated += bytes;
    }
}

// Log free
inline void log_free(int64_t bytes, bool is_cuda) {
    auto& stats = MemoryStats::instance();
    if (is_cuda) {
        stats.cuda_frees++;
        stats.cuda_bytes_freed += bytes;
    } else {
        stats.cpu_frees++;
        stats.cpu_bytes_freed += bytes;
    }
}

// Increment/decrement counters
inline void tensor_created() { MemoryStats::instance().tensor_count++; }
inline void tensor_destroyed() { MemoryStats::instance().tensor_count--; }
inline void autograd_node_created() { MemoryStats::instance().autograd_node_count++; }
inline void autograd_node_destroyed() { MemoryStats::instance().autograd_node_count--; }

} // namespace debug
} // namespace c10

// Convenient macros
#ifdef PT_MEMORY_DEBUG
    #define PT_MEM_CHECKPOINT(label) c10::debug::memory_checkpoint(label)
    #define PT_MEM_LOG_ALLOC(bytes, source, is_cuda) c10::debug::log_alloc(bytes, source, is_cuda)
    #define PT_MEM_LOG_FREE(bytes, is_cuda) c10::debug::log_free(bytes, is_cuda)
    #define PT_TENSOR_CREATED() c10::debug::tensor_created()
    #define PT_TENSOR_DESTROYED() c10::debug::tensor_destroyed()
    #define PT_NODE_CREATED() c10::debug::autograd_node_created()
    #define PT_NODE_DESTROYED() c10::debug::autograd_node_destroyed()
    #define PT_MEM_PRINT_STATS(label) c10::debug::MemoryStats::instance().print(label)
#else
    #define PT_MEM_CHECKPOINT(label) ((void)0)
    #define PT_MEM_LOG_ALLOC(bytes, source, is_cuda) ((void)0)
    #define PT_MEM_LOG_FREE(bytes, is_cuda) ((void)0)
    #define PT_TENSOR_CREATED() ((void)0)
    #define PT_TENSOR_DESTROYED() ((void)0)
    #define PT_NODE_CREATED() ((void)0)
    #define PT_NODE_DESTROYED() ((void)0)
    #define PT_MEM_PRINT_STATS(label) ((void)0)
#endif
