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
    void allreduce_grads();

    std::shared_ptr<nn::Module> module() const { return module_; }
    int rank() const { return cfg_.rank; }
    int world_size() const { return cfg_.world_size; }

private:
    std::shared_ptr<nn::Module> module_;
    DDPConfig cfg_;
};

}  // namespace distributed
}  // namespace torch
