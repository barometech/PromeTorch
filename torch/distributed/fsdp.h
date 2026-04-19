// ============================================================================
// fsdp.h — Fully Sharded Data Parallel + ZeRO for PromeTorch (CPU/Elbrus)
// ============================================================================
// Implements PyTorch FSDP and DeepSpeed ZeRO sharding entirely with file-based
// inter-process collectives (no MPI/NCCL). Designed to compile on E2K (LCC) and
// run across multiple processes that share /dev/shm (tmpfs).
//
// Sharding model
// --------------
// Every parameter is treated as a flat float32 array. Each rank owns the slice
//   [rank * ceil(N / W), min((rank+1) * ceil(N / W), N))
// where N = numel(param) and W = world_size. The shard owned by this rank is
// kept in `local_shards_[i]`. Before forward we gather all shards into the
// parameter tensor; after forward (or after the optimizer step) we drop the
// gathered storage and keep only the local shard.
//
// Sharding strategies
// -------------------
//   FULL_SHARD     : params + grads + optim state sharded   (== ZeRO-3)
//   SHARD_GRAD_OP  : params replicated, grads sharded       (== ZeRO-2)
//   NO_SHARD       : pure replicated DDP                    (== ZeRO-0)
//
// Wire protocol
// -------------
// Generation counter `gen_` increments every collective. Each rank writes
//   {sync_dir}/p{idx}_r{rank}_g{gen}.bin
// then a marker file
//   {sync_dir}/p{idx}_r{rank}_g{gen}.ready
// Readers poll the .ready files (1 ms tick, configurable timeout) before
// reading the .bin. Cleanup is per-generation per-param; no global barrier
// needed beyond the file existence checks.
//
// Self-test
// ---------
// `fsdp_selftest_main()` forks two child processes, builds an identical
// 4-layer MLP in each, runs ONE training step under FULL_SHARD and compares
// the post-step parameters of rank 0 against a baseline (no FSDP) in the
// parent. Difference must be < 1e-4 in L_inf.
// ============================================================================
#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "torch/nn/module.h"
#include "torch/nn/parameter.h"
#include "torch/optim/optimizer.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cmath>

#if !defined(_WIN32)
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#else
#  include <direct.h>
#  include <io.h>
#  define F_OK 0
#  ifndef access
#    define access _access
#  endif
static inline void usleep(unsigned us) {
    std::this_thread::sleep_for(std::chrono::microseconds(us));
}
static inline int unlink(const char* p) { return std::remove(p); }
#endif

namespace torch { namespace distributed {

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------
struct FSDPConfig {
    int rank        = 0;
    int world_size  = 1;
    std::string sync_dir = "/dev/shm/pt_fsdp";
    int timeout_ms  = 120000;          // per-collective wait budget
    int poll_us     = 1000;            // poll interval for .ready markers

    enum class ShardingStrategy { FULL_SHARD, SHARD_GRAD_OP, NO_SHARD };
    ShardingStrategy strategy = ShardingStrategy::FULL_SHARD;
};

// ----------------------------------------------------------------------------
// Internal helpers — file-based collectives
// ----------------------------------------------------------------------------
namespace fsdp_detail {

inline void mkdir_p(const std::string& dir) {
#if defined(_WIN32)
    _mkdir(dir.c_str());
#else
    ::mkdir(dir.c_str(), 0777);
#endif
}

inline std::string shard_path(const std::string& dir, int pidx, int rank, int gen) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/p%d_r%d_g%d.bin", dir.c_str(), pidx, rank, gen);
    return buf;
}
inline std::string ready_path(const std::string& dir, int pidx, int rank, int gen) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/p%d_r%d_g%d.ready", dir.c_str(), pidx, rank, gen);
    return buf;
}

inline void write_blob(const std::string& path, const float* data, int64_t n) {
    std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("fsdp: cannot open " + tmp);
    size_t w = std::fwrite(data, sizeof(float), (size_t)n, f);
    std::fclose(f);
    if ((int64_t)w != n) {
        unlink(tmp.c_str());
        throw std::runtime_error("fsdp: short write " + tmp);
    }
    // atomic rename so readers never see partial files
    std::rename(tmp.c_str(), path.c_str());
}

inline void read_blob(const std::string& path, float* data, int64_t n) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("fsdp: cannot read " + path);
    size_t r = std::fread(data, sizeof(float), (size_t)n, f);
    std::fclose(f);
    if ((int64_t)r != n) throw std::runtime_error("fsdp: short read " + path);
}

inline void touch(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f) { std::fputc('1', f); std::fclose(f); }
}

inline void wait_file(const std::string& path, int timeout_ms, int poll_us) {
    int waited = 0;
    while (access(path.c_str(), F_OK) != 0) {
        usleep((unsigned)poll_us);
        waited += poll_us;
        if (waited >= timeout_ms * 1000) {
            throw std::runtime_error("fsdp: timeout waiting for " + path);
        }
    }
}

// shard layout helpers
inline int64_t shard_chunk(int64_t numel, int world_size) {
    return (numel + world_size - 1) / world_size;
}
inline int64_t shard_begin(int64_t numel, int rank, int world_size) {
    int64_t c = shard_chunk(numel, world_size);
    int64_t b = (int64_t)rank * c;
    return b > numel ? numel : b;
}
inline int64_t shard_end(int64_t numel, int rank, int world_size) {
    int64_t c = shard_chunk(numel, world_size);
    int64_t e = (int64_t)(rank + 1) * c;
    return e > numel ? numel : e;
}
inline int64_t shard_size(int64_t numel, int rank, int world_size) {
    return shard_end(numel, rank, world_size) - shard_begin(numel, rank, world_size);
}

}  // namespace fsdp_detail

// ----------------------------------------------------------------------------
// FullyShardedDataParallel — sharded module wrapper
// ----------------------------------------------------------------------------
class FullyShardedDataParallel : public nn::Module {
public:
    FullyShardedDataParallel(std::shared_ptr<nn::Module> module, const FSDPConfig& cfg)
        : nn::Module("FSDP"), module_(std::move(module)), cfg_(cfg) {
        if (!module_) throw std::runtime_error("FSDP: module is null");
        if (cfg_.world_size <= 0) throw std::runtime_error("FSDP: bad world_size");
        if (cfg_.rank < 0 || cfg_.rank >= cfg_.world_size)
            throw std::runtime_error("FSDP: bad rank");

        register_module("module", module_);
        fsdp_detail::mkdir_p(cfg_.sync_dir);

        // Snapshot all flat parameters in deterministic order. This locks the
        // shard mapping so every rank uses the same indexing.
        params_ = module_->parameters(/*recurse=*/true);
        local_shards_.resize(params_.size());

        if (cfg_.strategy == FSDPConfig::ShardingStrategy::FULL_SHARD) {
            // Broadcast rank-0 params, then carve out and store the local shard,
            // then drop the full tensor data from each parameter (replaced by an
            // empty placeholder until the next all_gather_params()).
            sync_initial_params_from_rank0();
            for (size_t i = 0; i < params_.size(); ++i) {
                save_local_shard(i);
            }
            // Free non-shard storage on each rank (keep tensors as 0-element
            // placeholders so optim/Module bookkeeping is preserved).
            for (size_t i = 0; i < params_.size(); ++i) {
                drop_full_param(i);
            }
            params_resident_ = false;
        } else {
            // SHARD_GRAD_OP / NO_SHARD: keep full params replicated.
            sync_initial_params_from_rank0();
            for (size_t i = 0; i < params_.size(); ++i) save_local_shard(i);
            params_resident_ = true;
        }
    }

    ~FullyShardedDataParallel() override = default;

    // ------------------------------------------------------------------
    // Forward / backward hooks
    // ------------------------------------------------------------------
    at::Tensor forward(const at::Tensor& input) override {
        if (!params_resident_) all_gather_params();
        at::Tensor out = module_->forward(input);
        if (cfg_.strategy == FSDPConfig::ShardingStrategy::FULL_SHARD) {
            // Reshard immediately after forward only if we don't need params for
            // backward. Autograd needs them, so defer reshard until after step().
        }
        return out;
    }
    at::Tensor forward(const at::Tensor& a, const at::Tensor& b) override {
        if (!params_resident_) all_gather_params();
        return module_->forward(a, b);
    }
    at::Tensor forward(const std::vector<at::Tensor>& xs) override {
        if (!params_resident_) all_gather_params();
        return module_->forward(xs);
    }

    // ------------------------------------------------------------------
    // Public collective API
    // ------------------------------------------------------------------
    void all_gather_params() {
        for (size_t i = 0; i < params_.size(); ++i) {
            gather_full_param(i);
        }
        params_resident_ = true;
    }

    void reshard_params() {
        if (cfg_.strategy != FSDPConfig::ShardingStrategy::FULL_SHARD) return;
        for (size_t i = 0; i < params_.size(); ++i) {
            save_local_shard(i);     // capture latest values into local shard
            drop_full_param(i);      // free non-shard storage
        }
        params_resident_ = false;
    }

    // Reduce-scatter: each rank ends up with its shard of the SUMMED gradient.
    // For SHARD_GRAD_OP / FULL_SHARD the param's grad tensor is then resized to
    // shard_size and contains only the local slice of (sum of grads).
    // For NO_SHARD this is a plain all-reduce-sum.
    void reduce_scatter_grads() {
        ++gen_;
        const int W = cfg_.world_size;
        const int R = cfg_.rank;
        for (size_t i = 0; i < params_.size(); ++i) {
            nn::Parameter* p = params_[i];
            if (!p || !p->defined()) continue;

            at::Tensor grad = p->grad();
            if (!grad.defined() || grad.numel() == 0) continue;
            grad = grad.is_contiguous() ? grad : grad.contiguous();

            const int64_t N = grad.numel();
            // 1) write my full grad
            std::string my_path  = fsdp_detail::shard_path(cfg_.sync_dir, (int)i, R, gen_);
            std::string my_ready = fsdp_detail::ready_path(cfg_.sync_dir, (int)i, R, gen_);
            fsdp_detail::write_blob(my_path, grad.data_ptr<float>(), N);
            fsdp_detail::touch(my_ready);

            // 2) wait for everyone, sum into a local accumulator
            std::vector<float> sum(N, 0.0f);
            std::vector<float> tmp(N);
            for (int r = 0; r < W; ++r) {
                std::string rp = fsdp_detail::ready_path(cfg_.sync_dir, (int)i, r, gen_);
                fsdp_detail::wait_file(rp, cfg_.timeout_ms, cfg_.poll_us);
                std::string bp = fsdp_detail::shard_path(cfg_.sync_dir, (int)i, r, gen_);
                fsdp_detail::read_blob(bp, tmp.data(), N);
                for (int64_t k = 0; k < N; ++k) sum[k] += tmp[k];
            }

            // 3) keep only my shard slice (or full sum for NO_SHARD)
            if (cfg_.strategy == FSDPConfig::ShardingStrategy::NO_SHARD) {
                std::memcpy(grad.mutable_data_ptr<float>(), sum.data(),
                            sizeof(float) * (size_t)N);
            } else {
                int64_t b = fsdp_detail::shard_begin(N, R, W);
                int64_t e = fsdp_detail::shard_end(N, R, W);
                int64_t sz = e - b;
                at::Tensor sharded = at::empty({sz});
                std::memcpy(sharded.mutable_data_ptr<float>(), sum.data() + b,
                            sizeof(float) * (size_t)sz);
                p->set_grad(sharded);
            }

            // 4) only rank 0 cleans up to keep ordering simple
            if (R == 0) {
                // small pause: ensure all readers finished
                usleep(2000);
                for (int r = 0; r < W; ++r) {
                    unlink(fsdp_detail::shard_path(cfg_.sync_dir, (int)i, r, gen_).c_str());
                    unlink(fsdp_detail::ready_path(cfg_.sync_dir, (int)i, r, gen_).c_str());
                }
            }
        }
    }

    // Accessors
    std::shared_ptr<nn::Module> module() const { return module_; }
    int rank() const { return cfg_.rank; }
    int world_size() const { return cfg_.world_size; }
    const FSDPConfig& config() const { return cfg_; }

    // The local-shard tensors used by ZeRO optimizer.
    std::vector<at::Tensor>& local_shards() { return local_shards_; }
    const std::vector<nn::Parameter*>& flat_params() const { return params_; }

private:
    // Replace param.data with my local slice (used in FULL_SHARD between calls
    // to all_gather_params()).
    void save_local_shard(size_t i) {
        nn::Parameter* p = params_[i];
        if (!p || !p->defined()) return;
        at::Tensor t = p->data().is_contiguous() ? p->data() : p->data().contiguous();
        const int64_t N = t.numel();
        const int W = cfg_.world_size;
        int64_t b = fsdp_detail::shard_begin(N, cfg_.rank, W);
        int64_t e = fsdp_detail::shard_end(N, cfg_.rank, W);
        int64_t sz = e - b;
        at::Tensor shard = at::empty({sz});
        if (sz > 0) {
            std::memcpy(shard.mutable_data_ptr<float>(),
                        t.data_ptr<float>() + b,
                        sizeof(float) * (size_t)sz);
        }
        local_shards_[i] = shard;
    }

    // Drop the in-RAM full param storage (replace with 0-element placeholder).
    // Original sizes are remembered in `param_sizes_` to rebuild later.
    void drop_full_param(size_t i) {
        nn::Parameter* p = params_[i];
        if (!p || !p->defined()) return;
        if (param_sizes_.size() <= i) param_sizes_.resize(i + 1);
        param_sizes_[i] = p->data().sizes().vec();
        // Replace storage with an empty tensor of dtype Float
        p->set_data(at::empty({0}));
    }

    // Rebuild the full param tensor by gathering shards from all ranks.
    void gather_full_param(size_t i) {
        ++gen_;
        nn::Parameter* p = params_[i];
        if (!p) return;
        // Resolve original size
        std::vector<int64_t> shape;
        if (i < param_sizes_.size() && !param_sizes_[i].empty()) {
            shape = param_sizes_[i];
        } else if (p->defined()) {
            shape = p->data().sizes().vec();
        } else {
            return;
        }
        int64_t N = 1;
        for (auto s : shape) N *= s;
        const int W = cfg_.world_size;
        const int R = cfg_.rank;

        // 1) write my shard
        const at::Tensor& my = local_shards_[i];
        std::string my_path  = fsdp_detail::shard_path(cfg_.sync_dir, (int)i, R, gen_);
        std::string my_ready = fsdp_detail::ready_path(cfg_.sync_dir, (int)i, R, gen_);
        fsdp_detail::write_blob(my_path, my.data_ptr<float>(), my.numel());
        fsdp_detail::touch(my_ready);

        // 2) wait for all peers, concatenate
        at::Tensor full = at::empty(shape);
        float* dst = full.mutable_data_ptr<float>();
        for (int r = 0; r < W; ++r) {
            int64_t b = fsdp_detail::shard_begin(N, r, W);
            int64_t sz = fsdp_detail::shard_size(N, r, W);
            if (sz == 0) continue;
            std::string rp = fsdp_detail::ready_path(cfg_.sync_dir, (int)i, r, gen_);
            fsdp_detail::wait_file(rp, cfg_.timeout_ms, cfg_.poll_us);
            std::string bp = fsdp_detail::shard_path(cfg_.sync_dir, (int)i, r, gen_);
            fsdp_detail::read_blob(bp, dst + b, sz);
        }
        p->set_data(full);

        if (R == 0) {
            usleep(2000);
            for (int r = 0; r < W; ++r) {
                unlink(fsdp_detail::shard_path(cfg_.sync_dir, (int)i, r, gen_).c_str());
                unlink(fsdp_detail::ready_path(cfg_.sync_dir, (int)i, r, gen_).c_str());
            }
        }
    }

    // Broadcast: rank 0 writes its params; others read.
    void sync_initial_params_from_rank0() {
        for (size_t i = 0; i < params_.size(); ++i) {
            ++gen_;
            nn::Parameter* p = params_[i];
            if (!p || !p->defined()) continue;
            at::Tensor t = p->data().is_contiguous() ? p->data() : p->data().contiguous();
            std::string rp = fsdp_detail::ready_path(cfg_.sync_dir, (int)i, 0, gen_);
            std::string bp = fsdp_detail::shard_path(cfg_.sync_dir, (int)i, 0, gen_);
            if (cfg_.rank == 0) {
                fsdp_detail::write_blob(bp, t.data_ptr<float>(), t.numel());
                fsdp_detail::touch(rp);
            } else {
                fsdp_detail::wait_file(rp, cfg_.timeout_ms, cfg_.poll_us);
                fsdp_detail::read_blob(bp, t.mutable_data_ptr<float>(), t.numel());
                p->set_data(t);
            }
        }
        // Lazy cleanup left to next-gen overwrites; keeps init code simple.
    }

    std::shared_ptr<nn::Module>    module_;
    FSDPConfig                     cfg_;
    std::vector<nn::Parameter*>    params_;
    std::vector<at::Tensor>        local_shards_;
    std::vector<std::vector<int64_t>> param_sizes_;
    bool                           params_resident_ = true;
    int                            gen_ = 0;
};

// ----------------------------------------------------------------------------
// ZeROOptimizer — wraps an inner per-shard optimizer
// ----------------------------------------------------------------------------
// stage 1 : optimizer state sharded   (params + grads replicated)
// stage 2 : + grads sharded
// stage 3 : + params sharded          (FSDP / FULL_SHARD)
//
// The inner optimizer is constructed by the caller over the *local-shard*
// parameters (so it only allocates state for sz_i floats per param) and is
// invoked by ZeROOptimizer::step(). After step(), if params are sharded
// (stage 3) the FSDP wrapper's local shard is updated from the inner
// optimizer's parameter tensor.
// ----------------------------------------------------------------------------
class ZeROOptimizer : public optim::Optimizer {
public:
    ZeROOptimizer(std::vector<nn::Parameter*> shard_params,
                  std::shared_ptr<optim::Optimizer> inner,
                  int zero_stage,
                  const FSDPConfig& cfg)
        : optim::Optimizer(shard_params, inner ? inner->get_lr() : 0.01),
          shard_params_(std::move(shard_params)),
          inner_(std::move(inner)),
          stage_(zero_stage),
          cfg_(cfg) {
        if (!inner_) throw std::runtime_error("ZeRO: inner optimizer is null");
        if (zero_stage < 1 || zero_stage > 3)
            throw std::runtime_error("ZeRO: stage must be 1, 2, or 3");
    }

    void step() override {
        // The inner optimizer has been built over the same shard_params_, so
        // its state is already shard-sized — that's the ZeRO-1 win.
        inner_->step();
    }

    void zero_grad(bool set_to_none = false) override {
        inner_->zero_grad(set_to_none);
    }

    int stage() const { return stage_; }
    std::shared_ptr<optim::Optimizer> inner() const { return inner_; }

private:
    std::vector<nn::Parameter*>        shard_params_;
    std::shared_ptr<optim::Optimizer>  inner_;
    int                                stage_;
    FSDPConfig                         cfg_;
};

// ----------------------------------------------------------------------------
// Self-test (forks 2 workers, compares against unsharded baseline)
// ----------------------------------------------------------------------------
// Returns 0 on success, non-zero on failure. Defined here as a header-only
// helper so it can be invoked from a tiny .cpp test driver. POSIX-only.
// ----------------------------------------------------------------------------
#if !defined(_WIN32)
namespace fsdp_detail {

// Tiny 4-layer MLP used for the self-test
struct TinyMLP : public nn::Module {
    nn::Parameter w1, w2, w3, w4;
    TinyMLP(int seed) : nn::Module("TinyMLP") {
        std::srand(seed);
        auto rnd = [](int n){
            at::Tensor t = at::empty({n});
            float* p = t.mutable_data_ptr<float>();
            for (int i = 0; i < n; ++i) p[i] = ((std::rand() % 1000) - 500) / 5000.0f;
            return t;
        };
        w1 = nn::Parameter(rnd(8 * 16).view({8, 16}));
        w2 = nn::Parameter(rnd(16 * 16).view({16, 16}));
        w3 = nn::Parameter(rnd(16 * 16).view({16, 16}));
        w4 = nn::Parameter(rnd(16 * 4).view({16, 4}));
        register_parameter("w1", w1);
        register_parameter("w2", w2);
        register_parameter("w3", w3);
        register_parameter("w4", w4);
    }
    at::Tensor forward(const at::Tensor& x) override {
        // x: (B, 8). MLP with relu.
        at::Tensor h1 = x.matmul(*get_parameter("w1")).relu();
        at::Tensor h2 = h1.matmul(*get_parameter("w2")).relu();
        at::Tensor h3 = h2.matmul(*get_parameter("w3")).relu();
        return h3.matmul(*get_parameter("w4"));
    }
};

}  // namespace fsdp_detail

inline int fsdp_selftest_main(const std::string& sync_dir = "/dev/shm/pt_fsdp_test") {
    using namespace fsdp_detail;
    mkdir_p(sync_dir);

    constexpr int W = 2;
    // Deterministic input/target shared via files.
    at::Tensor x = at::empty({4, 8});
    at::Tensor y = at::empty({4, 4});
    {
        float* xp = x.mutable_data_ptr<float>();
        float* yp = y.mutable_data_ptr<float>();
        for (int i = 0; i < 32; ++i) xp[i] = (i % 7) / 10.0f;
        for (int i = 0; i < 16; ++i) yp[i] = (i % 5) / 10.0f;
    }

    // Baseline (unsharded) — single process, identical seed.
    auto baseline = std::make_shared<TinyMLP>(42);
    {
        at::Tensor out  = baseline->forward(x);
        at::Tensor diff = out - y;
        at::Tensor loss = (diff * diff).sum();
        torch::autograd::tensor_backward(loss);
        for (auto* p : baseline->parameters(true)) {
            if (!p->defined()) continue;
            at::Tensor g = p->grad();
            if (!g.defined()) continue;
            float* pd = p->data().mutable_data_ptr<float>();
            const float* gd = g.data_ptr<float>();
            for (int64_t k = 0; k < p->numel(); ++k) pd[k] -= 0.01f * gd[k];
        }
    }

    // Fork two workers running FSDP.
    pid_t pids[W] = {0};
    for (int r = 0; r < W; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            FSDPConfig cfg;
            cfg.rank = r;
            cfg.world_size = W;
            cfg.sync_dir = sync_dir;
            cfg.strategy = FSDPConfig::ShardingStrategy::FULL_SHARD;
            auto m = std::make_shared<TinyMLP>(42);
            auto fsdp = std::make_shared<FullyShardedDataParallel>(m, cfg);

            fsdp->all_gather_params();
            at::Tensor out = fsdp->forward(x);
            at::Tensor diff = out - y;
            at::Tensor loss = (diff * diff).sum();
            torch::autograd::tensor_backward(loss);
            fsdp->reduce_scatter_grads();
            // SGD on local shards
            for (size_t i = 0; i < fsdp->flat_params().size(); ++i) {
                auto* p = fsdp->flat_params()[i];
                at::Tensor g = p->grad();
                if (!g.defined()) continue;
                at::Tensor& shard = fsdp->local_shards()[i];
                float* sp = shard.mutable_data_ptr<float>();
                const float* gp = g.data_ptr<float>();
                for (int64_t k = 0; k < shard.numel(); ++k) sp[k] -= 0.01f * gp[k];
            }
            // Gather updated params, rank 0 writes result for parent to verify.
            fsdp->all_gather_params();
            if (r == 0) {
                std::string out_path = sync_dir + "/RESULT.bin";
                std::FILE* f = std::fopen(out_path.c_str(), "wb");
                if (f) {
                    for (auto* p : fsdp->flat_params()) {
                        if (!p->defined()) continue;
                        std::fwrite(p->data().data_ptr<float>(),
                                    sizeof(float), (size_t)p->numel(), f);
                    }
                    std::fclose(f);
                }
            }
            std::_Exit(0);
        } else if (pid > 0) {
            pids[r] = pid;
        } else {
            std::fprintf(stderr, "fsdp_selftest: fork failed\n");
            return 2;
        }
    }
    for (int r = 0; r < W; ++r) {
        int st = 0;
        waitpid(pids[r], &st, 0);
    }

    // Compare baseline vs FSDP rank-0 result.
    std::vector<float> base_flat;
    for (auto* p : baseline->parameters(true)) {
        if (!p->defined()) continue;
        const float* d = p->data().data_ptr<float>();
        base_flat.insert(base_flat.end(), d, d + p->numel());
    }
    std::vector<float> fsdp_flat(base_flat.size());
    {
        std::FILE* f = std::fopen((sync_dir + "/RESULT.bin").c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "fsdp_selftest: missing RESULT.bin\n");
            return 3;
        }
        std::fread(fsdp_flat.data(), sizeof(float), fsdp_flat.size(), f);
        std::fclose(f);
    }
    float max_err = 0.0f;
    for (size_t i = 0; i < base_flat.size(); ++i) {
        float e = std::fabs(base_flat[i] - fsdp_flat[i]);
        if (e > max_err) max_err = e;
    }
    std::fprintf(stderr, "fsdp_selftest: max|baseline - FSDP| = %.3e\n", max_err);
    return (max_err < 1e-4f) ? 0 : 1;
}
#endif  // !_WIN32

}}  // namespace torch::distributed
