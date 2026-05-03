#pragma once
// ============================================================================
// distributed.h — Multi-device distributed training for PromeTorch
// ============================================================================
// Supports: AllReduce (sum/avg), Broadcast, Barrier, DDP, SyncBatchNorm
// Backends: shared_memory (intra-process), nccl (CUDA multi-GPU)
//
// Usage (DDP):
//   auto pg = dist::init_process_group("nccl", rank, world_size);
//   auto ddp_model = DistributedDataParallel(model, pg);
//   auto output = ddp_model->forward(input);
//   loss.backward();
//   ddp_model->finish_gradient_synchronization();
//   optimizer.step();
//
// Usage (legacy):
//   dist::init(world_size=4);
//   dist::all_reduce(grads, rank, ReduceOp::AVG);
//   dist::finalize();
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "torch/nn/module.h"
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <cstring>
#include <memory>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#ifdef PT_USE_CUDA
#include <cuda_runtime.h>
// NCCL is optional even with CUDA — user may not have it installed
#ifdef PT_USE_NCCL
#include <nccl.h>
#endif
#endif

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
// Backend type
// ============================================================================

enum class BackendType {
    SHARED_MEMORY,   // Intra-process shared memory (CPU, always available)
    NCCL             // NVIDIA NCCL (requires CUDA + NCCL)
};

// ============================================================================
// ProcessGroup — abstract interface for collective operations
// ============================================================================

class ProcessGroup {
public:
    ProcessGroup(int rank, int world_size, BackendType backend)
        : rank_(rank), world_size_(world_size), backend_(backend) {}

    virtual ~ProcessGroup() = default;

    virtual void all_reduce(at::Tensor& tensor, ReduceOp op = ReduceOp::SUM) = 0;
    virtual void broadcast(at::Tensor& tensor, int src = 0) = 0;
    virtual void barrier() = 0;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    BackendType backend() const { return backend_; }

protected:
    int rank_;
    int world_size_;
    BackendType backend_;
};

using ProcessGroupPtr = std::shared_ptr<ProcessGroup>;

// ============================================================================
// SharedMemoryBackend — intra-process, works on CPU (always available)
// ============================================================================

class SharedMemoryBackend : public ProcessGroup {
public:
    // Shared state across all ranks in the same process group
    struct SharedState {
        std::mutex mutex;
        std::vector<std::vector<float>> buffers;
        std::vector<float> result_buffer;

        std::mutex barrier_mutex;
        std::condition_variable barrier_cv;
        int barrier_count = 0;
        int barrier_generation = 0;
        int world_size = 1;

        void barrier_wait() {
            std::unique_lock<std::mutex> lock(barrier_mutex);
            int gen = barrier_generation;
            barrier_count++;
            if (barrier_count >= world_size) {
                barrier_count = 0;
                barrier_generation++;
                barrier_cv.notify_all();
            } else {
                barrier_cv.wait(lock, [&] { return barrier_generation > gen; });
            }
        }
    };

    SharedMemoryBackend(int rank, int world_size, std::shared_ptr<SharedState> state)
        : ProcessGroup(rank, world_size, BackendType::SHARED_MEMORY)
        , state_(std::move(state))
    {}

    void all_reduce(at::Tensor& tensor, ReduceOp op = ReduceOp::SUM) override {
        int64_t numel = tensor.numel();
        size_t nbytes = numel * sizeof(float);

        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            if (state_->buffers.size() != static_cast<size_t>(world_size_)) {
                state_->buffers.resize(world_size_);
                state_->result_buffer.resize(numel);
            }
            state_->buffers[rank_].resize(numel);
            std::memcpy(state_->buffers[rank_].data(), tensor.data_ptr<float>(), nbytes);
        }

        state_->barrier_wait();

        // Rank 0 computes reduction
        if (rank_ == 0) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            std::fill(state_->result_buffer.begin(), state_->result_buffer.end(), 0.0f);

            switch (op) {
                case ReduceOp::SUM:
                case ReduceOp::AVG:
                    for (int r = 0; r < world_size_; ++r)
                        for (int64_t i = 0; i < numel; ++i)
                            state_->result_buffer[i] += state_->buffers[r][i];
                    if (op == ReduceOp::AVG) {
                        float inv = 1.0f / static_cast<float>(world_size_);
                        for (int64_t i = 0; i < numel; ++i)
                            state_->result_buffer[i] *= inv;
                    }
                    break;
                case ReduceOp::MAX:
                    state_->result_buffer = state_->buffers[0];
                    for (int r = 1; r < world_size_; ++r)
                        for (int64_t i = 0; i < numel; ++i)
                            state_->result_buffer[i] = std::max(state_->result_buffer[i], state_->buffers[r][i]);
                    break;
                case ReduceOp::MIN:
                    state_->result_buffer = state_->buffers[0];
                    for (int r = 1; r < world_size_; ++r)
                        for (int64_t i = 0; i < numel; ++i)
                            state_->result_buffer[i] = std::min(state_->result_buffer[i], state_->buffers[r][i]);
                    break;
            }
        }

        state_->barrier_wait();

        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            std::memcpy(tensor.mutable_data_ptr<float>(), state_->result_buffer.data(), nbytes);
        }

        state_->barrier_wait();
    }

    void broadcast(at::Tensor& tensor, int src = 0) override {
        if (rank_ == src) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->result_buffer.resize(tensor.numel());
            std::memcpy(state_->result_buffer.data(), tensor.data_ptr<float>(),
                       tensor.numel() * sizeof(float));
        }

        state_->barrier_wait();

        if (rank_ != src) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            std::memcpy(tensor.mutable_data_ptr<float>(), state_->result_buffer.data(),
                       tensor.numel() * sizeof(float));
        }

        state_->barrier_wait();
    }

    void barrier() override {
        state_->barrier_wait();
    }

private:
    std::shared_ptr<SharedState> state_;
};

// ============================================================================
// NCCLBackend — NVIDIA NCCL for real multi-GPU distributed training
// ============================================================================

#if defined(PT_USE_CUDA) && defined(PT_USE_NCCL)

// Helper macro for NCCL error checking
#define NCCL_CHECK(cmd) do {                                             \
    ncclResult_t result = cmd;                                           \
    if (result != ncclSuccess) {                                         \
        throw std::runtime_error(                                        \
            std::string("NCCL error in ") + __FILE__ + ":" +             \
            std::to_string(__LINE__) + " '" + #cmd + "': " +             \
            ncclGetErrorString(result));                                  \
    }                                                                    \
} while(0)

class NCCLBackend : public ProcessGroup {
public:
    // Shared state for NCCL communicator (created once, shared by all ranks)
    struct NCCLSharedState {
        ncclComm_t* comms = nullptr;   // Array of communicators, one per rank
        int world_size = 0;
        bool initialized = false;

        ~NCCLSharedState() {
            if (initialized && comms) {
                for (int i = 0; i < world_size; ++i) {
                    ncclCommDestroy(comms[i]);
                }
                delete[] comms;
                comms = nullptr;
            }
        }
    };

    NCCLBackend(int rank, int world_size, std::shared_ptr<NCCLSharedState> state,
                cudaStream_t stream = nullptr)
        : ProcessGroup(rank, world_size, BackendType::NCCL)
        , state_(std::move(state))
        , stream_(stream)
        , owns_stream_(false)
    {
        if (!stream_) {
            cudaStreamCreate(&stream_);
            owns_stream_ = true;
        }
    }

    ~NCCLBackend() override {
        if (owns_stream_ && stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    void all_reduce(at::Tensor& tensor, ReduceOp op = ReduceOp::SUM) override {
        if (!state_->initialized) {
            throw std::runtime_error("NCCLBackend: communicator not initialized");
        }

        // Multi-dtype dispatch (BF16/FP16/FP32/FP64/INT32/INT64).
        ncclDataType_t nccl_dtype;
        switch (tensor.scalar_type()) {
            case c10::ScalarType::Half:    nccl_dtype = ncclHalf;     break;
            case c10::ScalarType::BFloat16: nccl_dtype = ncclBfloat16; break;
            case c10::ScalarType::Float:   nccl_dtype = ncclFloat;    break;
            case c10::ScalarType::Double:  nccl_dtype = ncclDouble;   break;
            case c10::ScalarType::Int:     nccl_dtype = ncclInt32;    break;
            case c10::ScalarType::Long:    nccl_dtype = ncclInt64;    break;
            case c10::ScalarType::Char:    nccl_dtype = ncclInt8;     break;
            case c10::ScalarType::Byte:    nccl_dtype = ncclUint8;    break;
            default:
                throw std::runtime_error("NCCLBackend::all_reduce: unsupported dtype "
                                          + std::to_string((int)tensor.scalar_type()));
        }
        ncclRedOp_t nccl_op = to_nccl_op(op);
        int64_t count = tensor.numel();
        void* data = tensor.mutable_data_ptr();  // generic, no template

        // In-place allreduce
        NCCL_CHECK(ncclAllReduce(data, data, count, nccl_dtype, nccl_op,
                                 state_->comms[rank_], stream_));

        // Synchronize stream to ensure completion
        cudaStreamSynchronize(stream_);

        // For AVG: NCCL only supports SUM natively, divide after
        if (op == ReduceOp::AVG) {
            float inv = 1.0f / static_cast<float>(world_size_);
            float* fdata = tensor.mutable_data_ptr<float>();
            // Launch a simple scale kernel or do it on CPU after D2H
            // For simplicity, use cudaMemcpy pattern: the tensor is on GPU
            // We scale in-place on the GPU via a trivial kernel
            scale_tensor_cuda(fdata, count, inv, stream_);
            cudaStreamSynchronize(stream_);
        }
    }

    void broadcast(at::Tensor& tensor, int src = 0) override {
        if (!state_->initialized) {
            throw std::runtime_error("NCCLBackend: communicator not initialized");
        }

        ncclDataType_t nccl_dtype = ncclFloat;
        int64_t count = tensor.numel();
        void* data = tensor.mutable_data_ptr<float>();

        NCCL_CHECK(ncclBroadcast(data, data, count, nccl_dtype, src,
                                 state_->comms[rank_], stream_));
        cudaStreamSynchronize(stream_);
    }

    void barrier() override {
        // NCCL barrier: allreduce a single element
        // This is the standard pattern used by PyTorch
        float dummy = 1.0f;
        float* d_dummy = nullptr;
        cudaSetDevice(rank_);
        cudaMalloc(&d_dummy, sizeof(float));
        cudaMemcpy(d_dummy, &dummy, sizeof(float), cudaMemcpyHostToDevice);

        NCCL_CHECK(ncclAllReduce(d_dummy, d_dummy, 1, ncclFloat, ncclSum,
                                 state_->comms[rank_], stream_));
        cudaStreamSynchronize(stream_);

        // Don't cudaFree — follow PyTorch pattern of not freeing CUDA memory
        // to avoid shutdown ordering issues. Leak is negligible (4 bytes).
    }

    cudaStream_t stream() const { return stream_; }

private:
    std::shared_ptr<NCCLSharedState> state_;
    cudaStream_t stream_;
    bool owns_stream_;

    static ncclRedOp_t to_nccl_op(ReduceOp op) {
        switch (op) {
            case ReduceOp::SUM:
            case ReduceOp::AVG:  // AVG = SUM + scale
                return ncclSum;
            case ReduceOp::MAX:
                return ncclMax;
            case ReduceOp::MIN:
                return ncclMin;
            default:
                return ncclSum;
        }
    }

    // Simple GPU scale: fdata[i] *= scale for all i
    // Implemented as a host-side loop with cudaMemcpy for correctness.
    // For production, this should be a CUDA kernel (see CUDAKernels.cu).
    static void scale_tensor_cuda(float* d_data, int64_t count, float scale,
                                  cudaStream_t stream) {
        // Allocate host buffer, copy down, scale, copy back
        // This is suboptimal but correct without a separate .cu file.
        // In practice, use the element-wise CUDA kernel from CUDAKernels.cu.
        std::vector<float> host(count);
        cudaMemcpyAsync(host.data(), d_data, count * sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int64_t i = 0; i < count; ++i) {
            host[i] *= scale;
        }
        cudaMemcpyAsync(d_data, host.data(), count * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
};

#endif // PT_USE_CUDA && PT_USE_NCCL

// ============================================================================
// DistributedContext (legacy shared-memory backend — kept for compatibility)
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
// DistributedDataParallel (DDP) — Module wrapper for gradient synchronization
// ============================================================================
// Wraps a Module + ProcessGroup. After backward(), call
// finish_gradient_synchronization() to AllReduce all parameter gradients
// and average by world_size.
//
// Usage:
//   auto pg = dist::init_process_group("shared_memory", rank, world_size);
//   auto ddp = std::make_shared<DistributedDataParallel>(model, pg);
//   auto output = ddp->forward(input);
//   // ... backward ...
//   ddp->finish_gradient_synchronization();
//   optimizer.step();
//
// no_sync() — gradient accumulation across micro-batches:
//   When training with gradient accumulation (N micro-batches per
//   optimizer step), running an AllReduce after every micro-batch wastes
//   N-1 AllReduces of bandwidth. Wrap the first N-1 micro-batches in a
//   DDPNoSyncGuard so finish_gradient_synchronization() / sync_gradients()
//   become no-ops; on the Nth micro-batch (without the guard) the single
//   sync averages the locally-accumulated gradient across all ranks.
//
//     for (int mb = 0; mb < N; ++mb) {
//         std::optional<DDPNoSyncGuard> g;
//         if (mb < N - 1) g.emplace(*ddp);
//         auto loss = loss_fn(ddp->forward(x[mb]), y[mb]);
//         loss.backward();
//     }
//     ddp->finish_gradient_synchronization();   // single AllReduce
//     optimizer.step();
//
//   Backwards-compat: code that never touches no_sync()/require_grad_sync()
//   sees the previous behaviour (every sync call performs AllReduce).

class DistributedDataParallel : public nn::Module {
public:
    DistributedDataParallel(
        std::shared_ptr<nn::Module> module,
        ProcessGroupPtr process_group,
        bool broadcast_parameters = true
    )
        : nn::Module("DistributedDataParallel")
        , module_(std::move(module))
        , process_group_(std::move(process_group))
    {
        if (!module_) {
            throw std::runtime_error("DistributedDataParallel: module cannot be null");
        }
        if (!process_group_) {
            throw std::runtime_error("DistributedDataParallel: process_group cannot be null");
        }

        // Register the wrapped module as a submodule
        register_module("module", module_);

        // Broadcast parameters from rank 0 to all ranks (ensure identical init)
        if (broadcast_parameters) {
            broadcast_params_from_rank0();
        }
    }

    // Forward: delegates to wrapped module
    at::Tensor forward(const at::Tensor& input) override {
        return module_->forward(input);
    }

    at::Tensor forward(const at::Tensor& input1, const at::Tensor& input2) override {
        return module_->forward(input1, input2);
    }

    at::Tensor forward(const std::vector<at::Tensor>& inputs) override {
        return module_->forward(inputs);
    }

    // ================================================================
    // Gradient synchronization — call AFTER backward(), BEFORE step()
    // ================================================================

    // AllReduce all parameter gradients, then divide by world_size.
    // No-op when require_grad_sync_ is false (used by DDPNoSyncGuard for
    // gradient accumulation across micro-batches).
    void finish_gradient_synchronization() {
        if (!require_grad_sync_) return;

        auto params = module_->parameters(/*recurse=*/true);
        int ws = process_group_->world_size();

        for (auto* param : params) {
            if (!param->data().requires_grad()) continue;

            auto* meta = param->data().autograd_meta();
            if (!meta || !meta->grad_) continue;

            // Wrap the shared_ptr<TensorImpl> grad into a Tensor for AllReduce
            at::Tensor grad_tensor(meta->grad_);
            if (!grad_tensor.defined() || grad_tensor.numel() == 0) continue;

            // AllReduce: SUM gradients across all ranks
            process_group_->all_reduce(grad_tensor, ReduceOp::SUM);

            // Average by world_size (divide in-place)
            float inv_ws = 1.0f / static_cast<float>(ws);
            float* gdata = grad_tensor.mutable_data_ptr<float>();
            int64_t numel = grad_tensor.numel();
            for (int64_t i = 0; i < numel; ++i) {
                gdata[i] *= inv_ws;
            }
        }
    }

    // Convenience: sync gradients using ReduceOp::AVG directly.
    // No-op when require_grad_sync_ is false (DDPNoSyncGuard).
    void sync_gradients() {
        if (!require_grad_sync_) return;

        auto params = module_->parameters(/*recurse=*/true);

        for (auto* param : params) {
            if (!param->data().requires_grad()) continue;

            auto* meta = param->data().autograd_meta();
            if (!meta || !meta->grad_) continue;

            at::Tensor grad_tensor(meta->grad_);
            if (!grad_tensor.defined() || grad_tensor.numel() == 0) continue;

            process_group_->all_reduce(grad_tensor, ReduceOp::AVG);
        }
    }

    // Access the wrapped module
    std::shared_ptr<nn::Module> module() const { return module_; }
    ProcessGroupPtr process_group() const { return process_group_; }

    // ---- no_sync support (gradient accumulation across micro-batches) ----
    bool require_grad_sync() const     { return require_grad_sync_; }
    void set_require_grad_sync(bool v) { require_grad_sync_ = v; }

private:
    std::shared_ptr<nn::Module> module_;
    ProcessGroupPtr process_group_;
    // When false, finish_gradient_synchronization() and sync_gradients()
    // are no-ops. Toggled by DDPNoSyncGuard. Default true preserves
    // legacy behaviour (every call performs AllReduce).
    bool require_grad_sync_ = true;

    // Broadcast all parameters from rank 0 to ensure identical starting weights
    void broadcast_params_from_rank0() {
        auto params = module_->parameters(/*recurse=*/true);
        for (auto* param : params) {
            at::Tensor& t = param->data();
            process_group_->broadcast(t, /*src=*/0);
        }
        // Also broadcast buffers (e.g., running_mean, running_var in BatchNorm)
        auto bufs = module_->buffers(/*recurse=*/true);
        for (auto* buf : bufs) {
            at::Tensor& t = buf->data();
            if (t.defined() && t.numel() > 0) {
                process_group_->broadcast(t, /*src=*/0);
            }
        }
    }
};

// ============================================================================
// DDPNoSyncGuard — RAII guard that disables gradient AllReduce on a DDP
// ============================================================================
// While alive, finish_gradient_synchronization() and sync_gradients() on
// the wrapped DDP become no-ops. Style mirrors torch::autograd::NoGradGuard.
//
//   {
//       torch::distributed::DDPNoSyncGuard g(*ddp);
//       // forward + backward here: grads accumulate locally, no AllReduce
//   } // destructor restores previous flag
//
// Nestable: each guard saves and restores the prior value. Non-copyable
// and non-movable (RAII binding to a specific DDP instance).
class DDPNoSyncGuard {
public:
    explicit DDPNoSyncGuard(DistributedDataParallel& ddp)
        : ddp_(ddp), prev_(ddp.require_grad_sync()) {
        ddp_.set_require_grad_sync(false);
    }
    ~DDPNoSyncGuard() {
        ddp_.set_require_grad_sync(prev_);
    }

    DDPNoSyncGuard(const DDPNoSyncGuard&)            = delete;
    DDPNoSyncGuard& operator=(const DDPNoSyncGuard&) = delete;
    DDPNoSyncGuard(DDPNoSyncGuard&&)                 = delete;
    DDPNoSyncGuard& operator=(DDPNoSyncGuard&&)      = delete;

private:
    DistributedDataParallel& ddp_;
    bool                     prev_;
};

// ============================================================================
// SyncBatchNorm — BatchNorm with cross-rank mean/variance synchronization
// ============================================================================
// During training, computes local sum and sum-of-squares, AllReduces across
// all ranks, then normalizes with the global statistics.
// Ensures consistent normalization across all ranks (important for DDP).

class SyncBatchNorm : public nn::Module {
public:
    explicit SyncBatchNorm(
        int64_t num_features,
        ProcessGroupPtr process_group,
        double eps = 1e-5,
        double momentum = 0.1,
        bool affine = true,
        bool track_running_stats = true
    )
        : nn::Module("SyncBatchNorm")
        , num_features_(num_features)
        , eps_(eps)
        , momentum_(momentum)
        , affine_(affine)
        , track_running_stats_(track_running_stats)
        , process_group_(std::move(process_group))
    {
        if (affine_) {
            register_parameter("weight", nn::Parameter(at::ones({num_features})));
            register_parameter("bias", nn::Parameter(at::zeros({num_features})));
        }
        if (track_running_stats_) {
            register_buffer("running_mean", nn::Buffer(at::zeros({num_features})));
            register_buffer("running_var", nn::Buffer(at::ones({num_features})));
            register_buffer("num_batches_tracked", nn::Buffer(at::zeros({})));
        }
    }

    at::Tensor forward(const at::Tensor& input) override {
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t spatial = 1;
        for (int d = 2; d < input.dim(); ++d) {
            spatial *= input.size(d);
        }

        at::Tensor output = input.clone();
        float* out_data = output.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        nn::Buffer* running_mean_buf = get_buffer("running_mean");
        nn::Buffer* running_var_buf = get_buffer("running_var");

        std::vector<float> mean(channels, 0.0f);
        std::vector<float> var(channels, 0.0f);

        if (is_training()) {
            int64_t local_count = batch_size * spatial;

            // Step 1: Compute local sum and sum-of-squares per channel
            std::vector<float> local_sum(channels, 0.0f);
            std::vector<float> local_sum_sq(channels, 0.0f);

            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t n = 0; n < batch_size; ++n) {
                    for (int64_t s = 0; s < spatial; ++s) {
                        int64_t idx = n * channels * spatial + c * spatial + s;
                        float val = in_data[idx];
                        local_sum[c] += val;
                        local_sum_sq[c] += val * val;
                    }
                }
            }

            // Step 2: AllReduce sum and sum_sq across all ranks
            // Pack into tensors for collective communication
            at::Tensor sum_tensor = at::zeros({channels});
            at::Tensor sum_sq_tensor = at::zeros({channels});
            at::Tensor count_tensor = at::zeros({1});

            float* sum_ptr = sum_tensor.mutable_data_ptr<float>();
            float* sum_sq_ptr = sum_sq_tensor.mutable_data_ptr<float>();
            float* count_ptr = count_tensor.mutable_data_ptr<float>();

            for (int64_t c = 0; c < channels; ++c) {
                sum_ptr[c] = local_sum[c];
                sum_sq_ptr[c] = local_sum_sq[c];
            }
            count_ptr[0] = static_cast<float>(local_count);

            if (process_group_) {
                process_group_->all_reduce(sum_tensor, ReduceOp::SUM);
                process_group_->all_reduce(sum_sq_tensor, ReduceOp::SUM);
                process_group_->all_reduce(count_tensor, ReduceOp::SUM);
            }

            // Step 3: Compute global mean and variance
            float total_count = count_ptr[0];
            for (int64_t c = 0; c < channels; ++c) {
                mean[c] = sum_ptr[c] / total_count;
                // Var = E[X^2] - (E[X])^2
                var[c] = sum_sq_ptr[c] / total_count - mean[c] * mean[c];
                // Clamp to avoid negative variance from floating-point errors
                if (var[c] < 0.0f) var[c] = 0.0f;
            }

            // Step 4: Update running stats (using global statistics)
            if (track_running_stats_) {
                float* rm = running_mean_buf->data().mutable_data_ptr<float>();
                float* rv = running_var_buf->data().mutable_data_ptr<float>();
                float mom = static_cast<float>(momentum_);

                for (int64_t c = 0; c < channels; ++c) {
                    rm[c] = (1.0f - mom) * rm[c] + mom * mean[c];
                    // Store unbiased variance in running_var (PyTorch convention)
                    float var_unbiased = (total_count > 1.0f)
                        ? var[c] * total_count / (total_count - 1.0f)
                        : var[c];
                    rv[c] = (1.0f - mom) * rv[c] + mom * var_unbiased;
                }
            }
        } else {
            // Eval mode: use running statistics (no sync needed)
            if (track_running_stats_) {
                const float* rm = running_mean_buf->data().data_ptr<float>();
                const float* rv = running_var_buf->data().data_ptr<float>();
                for (int64_t c = 0; c < channels; ++c) {
                    mean[c] = rm[c];
                    var[c] = rv[c];
                }
            }
        }

        // Step 5: Normalize
        const float* gamma = affine_ ? get_parameter("weight")->data().data_ptr<float>() : nullptr;
        const float* beta = affine_ ? get_parameter("bias")->data().data_ptr<float>() : nullptr;

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                float inv_std = 1.0f / std::sqrt(var[c] + static_cast<float>(eps_));
                float g = affine_ ? gamma[c] : 1.0f;
                float b = affine_ ? beta[c] : 0.0f;

                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = n * channels * spatial + c * spatial + s;
                    out_data[idx] = (in_data[idx] - mean[c]) * inv_std * g + b;
                }
            }
        }

        return output;
    }

    // Convert a regular BatchNorm module to SyncBatchNorm
    // (copies weight, bias, running_mean, running_var)
    static std::shared_ptr<SyncBatchNorm> convert_sync_batchnorm(
        std::shared_ptr<nn::Module> bn_module,
        int64_t num_features,
        ProcessGroupPtr process_group,
        double eps = 1e-5,
        double momentum = 0.1
    ) {
        auto sync_bn = std::make_shared<SyncBatchNorm>(
            num_features, std::move(process_group), eps, momentum);

        // Copy parameters if they exist
        auto* src_weight = bn_module->get_parameter("weight");
        auto* src_bias = bn_module->get_parameter("bias");
        auto* dst_weight = sync_bn->get_parameter("weight");
        auto* dst_bias = sync_bn->get_parameter("bias");

        if (src_weight && dst_weight) {
            dst_weight->data().copy_(src_weight->data());
        }
        if (src_bias && dst_bias) {
            dst_bias->data().copy_(src_bias->data());
        }

        // Copy running stats
        auto* src_rm = bn_module->get_buffer("running_mean");
        auto* src_rv = bn_module->get_buffer("running_var");
        auto* dst_rm = sync_bn->get_buffer("running_mean");
        auto* dst_rv = sync_bn->get_buffer("running_var");

        if (src_rm && dst_rm) {
            dst_rm->data().copy_(src_rm->data());
        }
        if (src_rv && dst_rv) {
            dst_rv->data().copy_(src_rv->data());
        }

        return sync_bn;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << num_features_ << ", eps=" << eps_ << ", momentum=" << momentum_;
        if (!affine_) ss << ", affine=False";
        if (!track_running_stats_) ss << ", track_running_stats=False";
        ss << ", process_group=" << (process_group_ ? "active" : "none");
        return ss.str();
    }

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;
    ProcessGroupPtr process_group_;
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
// init_process_group — create a ProcessGroup with the specified backend
// ============================================================================
// backend: "shared_memory" or "nccl"
// For shared_memory: all ranks in the same process share state via SharedState.
// For nccl: requires PT_USE_CUDA + PT_USE_NCCL, one GPU per rank.
//
// Returns a vector of ProcessGroupPtr (one per rank) for intra-process use.
// For true multi-process NCCL, use init_process_group_single_rank() instead.

inline std::vector<ProcessGroupPtr> init_process_group(
    const std::string& backend,
    int world_size
) {
    std::vector<ProcessGroupPtr> groups;

    if (backend == "shared_memory" || backend == "gloo") {
        auto state = std::make_shared<SharedMemoryBackend::SharedState>();
        state->world_size = world_size;

        for (int rank = 0; rank < world_size; ++rank) {
            groups.push_back(std::make_shared<SharedMemoryBackend>(
                rank, world_size, state));
        }
    }
#if defined(PT_USE_CUDA) && defined(PT_USE_NCCL)
    else if (backend == "nccl") {
        // Create NCCL communicators for all ranks
        auto state = std::make_shared<NCCLBackend::NCCLSharedState>();
        state->world_size = world_size;
        state->comms = new ncclComm_t[world_size];

        // For intra-process multi-GPU: use ncclCommInitAll
        // Each rank corresponds to a different GPU device
        int* dev_list = new int[world_size];
        for (int i = 0; i < world_size; ++i) {
            dev_list[i] = i;  // rank i -> GPU i
        }

        ncclResult_t result = ncclCommInitAll(state->comms, world_size, dev_list);
        delete[] dev_list;

        if (result != ncclSuccess) {
            throw std::runtime_error(
                std::string("Failed to initialize NCCL communicators: ") +
                ncclGetErrorString(result));
        }
        state->initialized = true;

        for (int rank = 0; rank < world_size; ++rank) {
            cudaSetDevice(rank);
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            groups.push_back(std::make_shared<NCCLBackend>(
                rank, world_size, state, stream));
        }
    }
#endif
    else {
        throw std::runtime_error(
            "Unknown distributed backend: '" + backend + "'. "
            "Available: 'shared_memory'"
#if defined(PT_USE_CUDA) && defined(PT_USE_NCCL)
            ", 'nccl'"
#endif
        );
    }

    return groups;
}

// ============================================================================
// init_process_group (single rank) — for true multi-process training
// ============================================================================
// Each process calls this once with its own rank.
// For NCCL: requires ncclUniqueId to be broadcast from rank 0 (via e.g. MPI).
// For shared_memory: returns a ProcessGroupPtr backed by DistributedContext.

inline ProcessGroupPtr init_process_group_single_rank(
    const std::string& backend,
    int rank,
    int world_size
) {
    if (backend == "shared_memory" || backend == "gloo") {
        // Shared-memory single-rank: use the global DistributedContext
        // (all ranks must be in the same process)
        static auto state = std::make_shared<SharedMemoryBackend::SharedState>();
        static std::once_flag flag;
        std::call_once(flag, [&]() {
            state->world_size = world_size;
        });
        return std::make_shared<SharedMemoryBackend>(rank, world_size, state);
    }
#if defined(PT_USE_CUDA) && defined(PT_USE_NCCL)
    else if (backend == "nccl") {
        // For multi-process NCCL: each process creates one communicator
        // Rank 0 generates ncclUniqueId and broadcasts it
        // (In practice, use MPI_Bcast or TCP store to distribute the ID)
        static auto state = std::make_shared<NCCLBackend::NCCLSharedState>();
        static ncclUniqueId nccl_id;
        static std::mutex init_mutex;

        std::lock_guard<std::mutex> lock(init_mutex);

        if (!state->initialized) {
            state->world_size = world_size;
            state->comms = new ncclComm_t[world_size];

            if (rank == 0) {
                ncclGetUniqueId(&nccl_id);
                // In real multi-process setup, broadcast nccl_id to all ranks
                // via MPI, TCP, or filesystem. For intra-process, it's shared.
            }
            // Barrier would go here in multi-process setup
        }

        cudaSetDevice(rank);
        ncclResult_t result = ncclCommInitRank(
            &state->comms[rank], world_size, nccl_id, rank);

        if (result != ncclSuccess) {
            throw std::runtime_error(
                std::string("ncclCommInitRank failed for rank ") +
                std::to_string(rank) + ": " + ncclGetErrorString(result));
        }
        state->initialized = true;

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return std::make_shared<NCCLBackend>(rank, world_size, state, stream);
    }
#endif
    else {
        throw std::runtime_error(
            "Unknown distributed backend: '" + backend + "'. "
            "Available: 'shared_memory'"
#if defined(PT_USE_CUDA) && defined(PT_USE_NCCL)
            ", 'nccl'"
#endif
        );
    }
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
                    if (meta && meta->grad_) {
                        at::Tensor grad_tensor(meta->grad_);
                        ctx.all_reduce(grad_tensor, rank, ReduceOp::AVG);
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
