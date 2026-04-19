// ============================================================================
// ddp.h — Real DistributedDataParallel via TCP sockets (no MPI/NCCL/files)
// ============================================================================
// Per-step gradient AllReduce using a star topology over POSIX TCP sockets.
// Rank 0 is the hub: collects tensors from all workers, sums them, broadcasts
// the result back. Works on CPU-only Elbrus with LCC.
//
// Usage:
//   torch::distributed::DDPConfig cfg{rank, nprocs, "127.0.0.1", 29500};
//   torch::distributed::init_process_group(cfg);
//   auto ddp = std::make_shared<torch::distributed::DistributedDataParallel>(model, cfg);
//   auto out = ddp->forward(x);
//   loss.backward();
//   ddp->allreduce_grads();   // sums grads across ranks; caller divides by world_size
//   optimizer.step();
//   torch::distributed::destroy_process_group();
//
// Wire protocol (every message): 8 bytes big-endian uint64 size + payload.
// Tensor payload = contiguous float32 little-endian (host endianness assumed
// uniform across ranks in this LAN/loopback use case).
// ============================================================================
#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "torch/nn/module.h"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace torch {
namespace distributed {

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------
struct DDPConfig {
    int rank = 0;            // 0..world_size-1
    int world_size = 1;      // total processes
    std::string master_addr; // e.g. "127.0.0.1"
    int master_port = 29500;
    int timeout_sec = 300;   // socket recv/connect timeout
};

// ----------------------------------------------------------------------------
// Process-group lifecycle (singleton; one group per process)
// ----------------------------------------------------------------------------
// init_process_group: rank 0 listens on master_addr:master_port; ranks
// 1..N-1 connect. Blocks until all peers handshake. Throws std::runtime_error
// on socket / timeout failures.
void init_process_group(const DDPConfig& cfg);

// destroy_process_group: closes all sockets and resets the singleton state.
// Safe to call even if init failed; idempotent.
void destroy_process_group();

bool is_initialized();
int  get_rank();
int  get_world_size();

// ----------------------------------------------------------------------------
// Collective operations (float32 contiguous tensors only)
// ----------------------------------------------------------------------------
// all_reduce: in-place SUM. After return, every rank's tensor holds the
// element-wise sum across all ranks. Caller divides by world_size for AVG.
void all_reduce(at::Tensor& tensor);

// broadcast: src_rank sends tensor data to every other rank in-place.
void broadcast(at::Tensor& tensor, int src_rank = 0);

// barrier: all ranks block until everyone has called barrier().
void barrier();

// ----------------------------------------------------------------------------
// DistributedDataParallel — wraps a module, syncs grads each backward
// ----------------------------------------------------------------------------
//
// Gradient-accumulation tip — use no_sync():
//   For every micro-batch except the last, wrap the forward+backward in a
//   NoSyncGuard so that allreduce_grads() becomes a no-op. On the FINAL
//   micro-batch, do NOT use the guard: that one call to allreduce_grads()
//   then averages the locally-accumulated gradient across all ranks.
//   This saves N-1 of every N AllReduces during gradient accumulation.
//
//   Example (N=4 micro-batches, 1 optimizer step):
//     for (int mb = 0; mb < 4; ++mb) {
//         bool last = (mb == 3);
//         std::optional<torch::distributed::DDPNoSyncGuard> guard;
//         if (!last) guard.emplace(*ddp);
//         auto out  = ddp->forward(batches[mb]);
//         auto loss = loss_fn(out, labels[mb]);
//         loss.backward();          // accumulates into .grad locally
//     }
//     ddp->allreduce_grads();       // single AllReduce of the summed grad
//     optimizer.step();
//
// Backwards-compat: if you never touch require_grad_sync()/no_sync(),
// behaviour is identical to the previous version (always sync).
class DistributedDataParallel : public nn::Module {
public:
    DistributedDataParallel(std::shared_ptr<nn::Module> module,
                            const DDPConfig& cfg,
                            bool broadcast_init_params = true);

    // Forward delegates to wrapped module.
    at::Tensor forward(const at::Tensor& input) override;
    at::Tensor forward(const at::Tensor& a, const at::Tensor& b) override;
    at::Tensor forward(const std::vector<at::Tensor>& inputs) override;

    // Manual gradient sync. Call after loss.backward(), before optimizer.step().
    // Sums all parameter gradients across ranks AND divides by world_size
    // (so that the resulting gradient is the average across ranks — the standard
    // DDP semantic).
    //
    // When require_grad_sync() is false (set via NoSyncGuard / no_sync()) this
    // call is a no-op — grads stay rank-local until sync is re-enabled.
    void allreduce_grads();

    // ---- no_sync support (gradient accumulation across micro-batches) ----
    bool require_grad_sync() const  { return require_grad_sync_; }
    void set_require_grad_sync(bool v) { require_grad_sync_ = v; }

    std::shared_ptr<nn::Module> module() const { return module_; }
    int rank() const { return cfg_.rank; }
    int world_size() const { return cfg_.world_size; }

private:
    std::shared_ptr<nn::Module> module_;
    DDPConfig cfg_;
    // When false, allreduce_grads() returns immediately. Toggled by
    // DDPNoSyncGuard. Default true preserves the legacy behaviour.
    bool require_grad_sync_ = true;
};

// RAII guard: while alive, suppresses allreduce_grads() on the wrapped DDP.
// Mirrors the style of torch::autograd::NoGradGuard.
//
//   {
//       torch::distributed::DDPNoSyncGuard g(*ddp);
//       loss.backward();      // grads accumulate locally; no AllReduce
//   } // guard destructor restores previous state
//
// Nestable: each guard saves and restores the prior flag value, so
// nested guards behave correctly.
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

}  // namespace distributed
}  // namespace torch
