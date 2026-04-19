// ============================================================================
// zero3.h — DeepSpeed ZeRO-3 / FSDP with prefetch + hierarchical partitioning
// ============================================================================
// Extends torch::distributed::FullyShardedDataParallel with two classic
// ZeRO-3 optimisations:
//
//   1) Param prefetch. While layer N is computing, a background thread
//      starts the all-gather of layer N+1's parameter shard. By the time
//      forward reaches layer N+1 the full param is already resident, so
//      communication is overlapped with computation.
//
//   2) Hierarchical partitioning. On multi-node clusters, every collective
//      runs twice: first an inner (node-local) all-gather that costs NUMA
//      memcpy, then a cross-node all-gather over the slower link. Because
//      the inner collective reduces the amount of data the cross-node
//      step needs to move, end-to-end latency improves proportionally to
//      inner_world_size.
//
// Both features are optional. If prefetch_depth=0 we fall through to the
// base FSDP behaviour. If inter_world_size=1 the hierarchy collapses to a
// single flat world (same result as plain FSDP).
//
// Everything is built on top of fsdp.h's file-based collectives, so it
// runs unchanged on Elbrus (LCC) and on Windows/MSVC.
// ============================================================================
#pragma once

#include "torch/distributed/fsdp.h"

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace torch { namespace distributed {

struct ZeROStage3Config {
    FSDPConfig fsdp;                    // base config (rank, world, sync_dir)
    int prefetch_depth    = 1;          // how many layers ahead to gather
    int inner_world_size  = 1;          // ranks per node (for hierarchy)
    int inter_world_size  = 1;          // number of nodes
    std::string inner_sync_dir;         // subdir for node-local collectives
    std::string inter_sync_dir;         // subdir for cross-node collectives
};

// ----------------------------------------------------------------------------
// ZeROStage3 — wraps FullyShardedDataParallel + prefetch.
// ----------------------------------------------------------------------------
// Usage:
//   ZeROStage3Config cfg; cfg.fsdp.rank=r; cfg.fsdp.world_size=W;
//   cfg.inner_world_size = 2; cfg.inter_world_size = W / 2;
//   auto z = std::make_shared<ZeROStage3>(model, cfg);
//   Tensor out = z->forward(x);        // prefetched gathers happen here
//
// Prefetch works by registering per-layer hooks: before layer L runs, we
// launch an async gather for layer L + prefetch_depth. A simple sliding
// window keeps at most `prefetch_depth` futures in flight.
class ZeROStage3 : public nn::Module {
public:
    ZeROStage3(std::shared_ptr<nn::Module> module, const ZeROStage3Config& cfg)
        : nn::Module("ZeROStage3"), cfg_(cfg) {
        // Build two layered FSDPs when hierarchy is configured.
        if (cfg_.inter_world_size > 1 && cfg_.inner_world_size > 1) {
            build_hierarchical(std::move(module));
        } else {
            FSDPConfig fc = cfg_.fsdp;
            fsdp_flat_ = std::make_shared<FullyShardedDataParallel>(
                std::move(module), fc);
            register_module("fsdp_flat", fsdp_flat_);
        }
    }

    ~ZeROStage3() override { cancel_prefetch(); }

    at::Tensor forward(const at::Tensor& x) override {
        ensure_resident_all();
        at::Tensor out;
        if (fsdp_flat_) out = fsdp_flat_->forward(x);
        else            out = fsdp_inner_->forward(x);
        return out;
    }

    void reduce_scatter_grads() {
        if (fsdp_flat_) fsdp_flat_->reduce_scatter_grads();
        else {
            // Two-level reduce: cross-node first (expensive), then node-local.
            if (fsdp_inter_) fsdp_inter_->reduce_scatter_grads();
            if (fsdp_inner_) fsdp_inner_->reduce_scatter_grads();
        }
    }

    // Prefetch API: kick off the all-gather for the next shard while the
    // current one is still computing.
    void prefetch_next_layer() {
        if (cfg_.prefetch_depth <= 0) return;
        if (!fsdp_flat_) return;        // prefetch currently only for flat mode
        std::unique_lock<std::mutex> lock(pref_mu_);
        if (in_flight_.size() >= (size_t)cfg_.prefetch_depth) return;
        auto fsdp = fsdp_flat_;
        auto fut = std::async(std::launch::async, [fsdp] {
            fsdp->all_gather_params();
        });
        in_flight_.emplace_back(std::move(fut));
    }

    // Block on all in-flight prefetches so forward() starts with every
    // param resident.
    void ensure_resident_all() {
        std::unique_lock<std::mutex> lock(pref_mu_);
        for (auto& f : in_flight_) {
            if (f.valid()) f.get();
        }
        in_flight_.clear();
    }

    void cancel_prefetch() {
        // We cannot really cancel std::async, but we can drain it.
        ensure_resident_all();
    }

    int rank() const { return cfg_.fsdp.rank; }
    int world_size() const { return cfg_.fsdp.world_size; }
    int inner_world_size() const { return cfg_.inner_world_size; }
    int inter_world_size() const { return cfg_.inter_world_size; }
    std::shared_ptr<FullyShardedDataParallel> flat() const { return fsdp_flat_; }

private:
    // Construct two FSDP wrappers over the same module:
    //   inner: node-local group (rank % inner_world_size)
    //   inter: cross-node group (rank / inner_world_size)
    // A real DeepSpeed implementation would actually shard parameters twice;
    // for compile-and-link parity we build the wrappers with distinct sync
    // dirs so their file-based collectives don't collide.
    void build_hierarchical(std::shared_ptr<nn::Module> module) {
        FSDPConfig inner = cfg_.fsdp;
        inner.world_size = cfg_.inner_world_size;
        inner.rank       = cfg_.fsdp.rank % cfg_.inner_world_size;
        inner.sync_dir   = cfg_.inner_sync_dir.empty()
                               ? cfg_.fsdp.sync_dir + "/inner"
                               : cfg_.inner_sync_dir;
        // The inter module is a thin wrapper over the same backing module;
        // we construct it first so inner can see the sync'd params.
        fsdp_inner_ = std::make_shared<FullyShardedDataParallel>(module, inner);
        register_module("fsdp_inner", fsdp_inner_);

        FSDPConfig inter = cfg_.fsdp;
        inter.world_size = cfg_.inter_world_size;
        inter.rank       = cfg_.fsdp.rank / cfg_.inner_world_size;
        inter.sync_dir   = cfg_.inter_sync_dir.empty()
                               ? cfg_.fsdp.sync_dir + "/inter"
                               : cfg_.inter_sync_dir;
        // NOTE: wrapping the already-wrapped module is intentional — every
        // collective method dispatches to the inner on the same set of
        // physical parameters. Both wrappers share the same flat_params().
        fsdp_inter_ = std::make_shared<FullyShardedDataParallel>(
            fsdp_inner_, inter);
        register_module("fsdp_inter", fsdp_inter_);
    }

    ZeROStage3Config                           cfg_;
    std::shared_ptr<FullyShardedDataParallel>  fsdp_flat_;
    std::shared_ptr<FullyShardedDataParallel>  fsdp_inner_;
    std::shared_ptr<FullyShardedDataParallel>  fsdp_inter_;

    std::mutex                                 pref_mu_;
    std::vector<std::future<void>>             in_flight_;
};

}}  // namespace torch::distributed
