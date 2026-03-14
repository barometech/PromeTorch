#pragma once
// ============================================================================
// distributed.h — Multi-device distributed training for PromeTorch
// ============================================================================
// Emulates distributed training across multiple virtual devices.
// Supports: AllReduce (sum/avg), Broadcast, Barrier
// Backend: shared memory (intra-process) — works with all backends
//
// Usage:
//   dist::init(world_size=4);
//   for (int rank = 0; rank < 4; ++rank) {
//       // Each rank trains on its shard
//       auto grads = model.parameters_grad();
//       dist::all_reduce(grads, dist::ReduceOp::AVG);
//       optimizer.step();
//   }
//   dist::finalize();
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <cstring>

namespace torch {
namespace distributed {

// ============================================================================
// Reduce operations
// ============================================================================

enum class ReduceOp {
    SUM,
    AVG,
    MAX,
    MIN
};

// ============================================================================
// Distributed context (shared-memory backend)
// ============================================================================

class DistributedContext {
public:
    static DistributedContext& get() {
        static DistributedContext instance;
        return instance;
    }

    // Initialize distributed context
    void init(int world_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        world_size_ = world_size;
        initialized_ = true;
        barrier_count_ = 0;
        barrier_generation_ = 0;
        buffers_.clear();
    }

    void finalize() {
        std::lock_guard<std::mutex> lock(mutex_);
        initialized_ = false;
        world_size_ = 1;
        buffers_.clear();
    }

    bool is_initialized() const { return initialized_; }
    int world_size() const { return world_size_; }

    // ================================================================
    // AllReduce: all ranks contribute a tensor, all get the reduced result
    // ================================================================

    // Synchronous AllReduce (called from each rank's thread)
    void all_reduce(at::Tensor& tensor, int rank, ReduceOp op = ReduceOp::SUM) {
        int64_t numel = tensor.numel();
        size_t nbytes = numel * sizeof(float);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Initialize buffer on first call
            if (buffers_.size() != static_cast<size_t>(world_size_)) {
                buffers_.resize(world_size_);
                result_buffer_.resize(numel);
            }
            // Copy this rank's data into shared buffer
            buffers_[rank].resize(numel);
            std::memcpy(buffers_[rank].data(), tensor.data_ptr<float>(), nbytes);
        }

        // Barrier — wait for all ranks
        barrier();

        // Rank 0 computes the reduction
        if (rank == 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            std::fill(result_buffer_.begin(), result_buffer_.end(), 0.0f);

            switch (op) {
                case ReduceOp::SUM:
                case ReduceOp::AVG:
                    for (int r = 0; r < world_size_; ++r)
                        for (int64_t i = 0; i < numel; ++i)
                            result_buffer_[i] += buffers_[r][i];
                    if (op == ReduceOp::AVG) {
                        float inv = 1.0f / static_cast<float>(world_size_);
                        for (int64_t i = 0; i < numel; ++i)
                            result_buffer_[i] *= inv;
                    }
                    break;
                case ReduceOp::MAX:
                    result_buffer_ = buffers_[0];
                    for (int r = 1; r < world_size_; ++r)
                        for (int64_t i = 0; i < numel; ++i)
                            result_buffer_[i] = std::max(result_buffer_[i], buffers_[r][i]);
                    break;
                case ReduceOp::MIN:
                    result_buffer_ = buffers_[0];
                    for (int r = 1; r < world_size_; ++r)
                        for (int64_t i = 0; i < numel; ++i)
                            result_buffer_[i] = std::min(result_buffer_[i], buffers_[r][i]);
                    break;
            }
        }

        // Barrier — wait for reduction to complete
        barrier();

        // All ranks copy result back
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::memcpy(tensor.mutable_data_ptr<float>(), result_buffer_.data(), nbytes);
        }

        // Final barrier to ensure no one modifies result_buffer_ before copy
        barrier();
    }

    // ================================================================
    // Broadcast: rank src sends tensor to all other ranks
    // ================================================================

    void broadcast(at::Tensor& tensor, int rank, int src = 0) {
        if (rank == src) {
            std::lock_guard<std::mutex> lock(mutex_);
            result_buffer_.resize(tensor.numel());
            std::memcpy(result_buffer_.data(), tensor.data_ptr<float>(),
                       tensor.numel() * sizeof(float));
        }

        barrier();

        if (rank != src) {
            std::lock_guard<std::mutex> lock(mutex_);
            std::memcpy(tensor.mutable_data_ptr<float>(), result_buffer_.data(),
                       tensor.numel() * sizeof(float));
        }

        barrier();
    }

    // ================================================================
    // Barrier: wait for all ranks
    // ================================================================

    void barrier() {
        std::unique_lock<std::mutex> lock(barrier_mutex_);
        int gen = barrier_generation_;
        barrier_count_++;
        if (barrier_count_ >= world_size_) {
            barrier_count_ = 0;
            barrier_generation_++;
            barrier_cv_.notify_all();
        } else {
            barrier_cv_.wait(lock, [&] { return barrier_generation_ > gen; });
        }
    }

    // ================================================================
    // Scatter data across ranks (for DataParallel)
    // ================================================================

    // Split a batch tensor along dim 0 for given rank
    at::Tensor scatter(const at::Tensor& tensor, int rank) {
        int64_t batch = tensor.size(0);
        int64_t chunk = batch / world_size_;
        int64_t start = rank * chunk;
        int64_t end = (rank == world_size_ - 1) ? batch : start + chunk;
        return tensor.slice(0, start, end);
    }

private:
    DistributedContext() = default;

    bool initialized_ = false;
    int world_size_ = 1;

    std::mutex mutex_;
    std::vector<std::vector<float>> buffers_;
    std::vector<float> result_buffer_;

    // Barrier state
    std::mutex barrier_mutex_;
    std::condition_variable barrier_cv_;
    int barrier_count_ = 0;
    int barrier_generation_ = 0;
};

// ============================================================================
// Convenience free functions
// ============================================================================

namespace dist {

inline void init(int world_size) {
    DistributedContext::get().init(world_size);
}

inline void finalize() {
    DistributedContext::get().finalize();
}

inline bool is_initialized() {
    return DistributedContext::get().is_initialized();
}

inline int world_size() {
    return DistributedContext::get().world_size();
}

inline void all_reduce(at::Tensor& tensor, int rank, ReduceOp op = ReduceOp::SUM) {
    DistributedContext::get().all_reduce(tensor, rank, op);
}

inline void all_reduce(std::vector<at::Tensor>& tensors, int rank, ReduceOp op = ReduceOp::SUM) {
    for (auto& t : tensors)
        DistributedContext::get().all_reduce(t, rank, op);
}

inline void broadcast(at::Tensor& tensor, int rank, int src = 0) {
    DistributedContext::get().broadcast(tensor, rank, src);
}

inline void barrier() {
    // Can't use standalone barrier without rank — noop
}

inline at::Tensor scatter(const at::Tensor& tensor, int rank) {
    return DistributedContext::get().scatter(tensor, rank);
}

// ============================================================================
// DataParallel: run forward+backward on multiple virtual devices
// ============================================================================

// Simple data-parallel training step
// fn(rank, data_shard) -> loss_scalar
// Gradients are averaged across all ranks
template<typename ForwardFn>
float data_parallel_step(
    ForwardFn fn,
    const at::Tensor& data,
    const at::Tensor& labels,
    std::vector<at::Tensor>& params,
    int world_size
) {
    auto& ctx = DistributedContext::get();
    if (!ctx.is_initialized()) ctx.init(world_size);

    float total_loss = 0.0f;

    // Launch threads for each rank
    std::vector<std::thread> threads;
    std::vector<float> losses(world_size, 0.0f);

    for (int rank = 0; rank < world_size; ++rank) {
        threads.emplace_back([&, rank]() {
            // Scatter data
            at::Tensor data_shard = ctx.scatter(data, rank);
            at::Tensor label_shard = ctx.scatter(labels, rank);

            // Forward + backward (user function)
            losses[rank] = fn(rank, data_shard, label_shard);

            // AllReduce gradients (average)
            for (auto& p : params) {
                if (p.requires_grad()) {
                    auto* meta = p.autograd_meta();
                    if (meta && meta->grad_.defined()) {
                        ctx.all_reduce(meta->grad_, rank, ReduceOp::AVG);
                    }
                }
            }
        });
    }

    for (auto& t : threads) t.join();

    for (float l : losses) total_loss += l;
    return total_loss / static_cast<float>(world_size);
}

} // namespace dist

} // namespace distributed
} // namespace torch
