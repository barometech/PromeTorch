// ============================================================================
// tensor_parallel.h — Megatron-style Tensor Parallelism for PromeTorch
// ============================================================================
// Header-only. CPU-only. No CUDA, no NCCL, no MPI.
// Cross-process collectives via /dev/shm tmpfs files (rename-atomic publish,
// generation counter to avoid stale reads, file-based barrier rendezvous).
//
//   ColumnParallelLinear : weight sharded along OUT (rows of W in PyTorch
//                          [out, in] storage). Optional all_gather of output.
//   RowParallelLinear    : weight sharded along IN (cols of W). Forward
//                          requires input to be already split along last dim
//                          (one shard per rank); partial output is all-reduced
//                          (sum) to produce the full result.
//
//   tp_all_gather(local, dim, tp)   — concat all ranks' shards along `dim`.
//   tp_all_reduce_sum(local, tp)    — element-wise sum across all ranks.
//   tp_barrier(tp)                  — file-based rendezvous.
//
// All collectives are blocking and run on the calling thread. They serialize
// tensors as raw float32 (currently the only TP-supported dtype) and use a
// monotonic generation counter held by TPConfig to disambiguate calls.
// ============================================================================
#pragma once

#include "torch/nn/module.h"
#include "torch/nn/parameter.h"
#include "torch/nn/init.h"
#include "torch/nn/modules/linear.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "aten/src/ATen/ATen.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

#if defined(_WIN32)
  #include <direct.h>
  #include <io.h>
  #define PT_TP_MKDIR(p) _mkdir(p)
  #define PT_TP_ACCESS(p) _access((p), 0)
  #define PT_TP_UNLINK(p) _unlink(p)
  #include <windows.h>
  static inline void pt_tp_usleep(unsigned us) { ::Sleep((us + 999) / 1000); }
#else
  #include <unistd.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #define PT_TP_MKDIR(p) ::mkdir((p), 0777)
  #define PT_TP_ACCESS(p) ::access((p), 0)
  #define PT_TP_UNLINK(p) ::unlink(p)
  static inline void pt_tp_usleep(unsigned us) { ::usleep(us); }
#endif

namespace torch {
namespace nn {
namespace parallel {

using at::Tensor;

// ============================================================================
// TPConfig — process-group description.
// ============================================================================
// gen_ is the monotonic generation counter; each collective bumps it so that
// stale files from prior iterations cannot be misread.

struct TPConfig {
    int rank = 0;
    int world_size = 1;
    std::string sync_dir = "/dev/shm/pt_tp";
    long timeout_us = 60L * 1000000L;     // 60s default
    mutable int64_t gen_ = 0;             // mutated by collectives
    mutable bool dir_ready_ = false;
};

// ----------------------------------------------------------------------------
// Filesystem helpers (private).
// ----------------------------------------------------------------------------

namespace detail {

inline void ensure_dir(const TPConfig& tp) {
    if (tp.dir_ready_) return;
    PT_TP_MKDIR(tp.sync_dir.c_str());  // ignore EEXIST
    tp.dir_ready_ = true;
}

inline std::string make_path(const TPConfig& tp, const char* tag,
                             int64_t gen, int rank) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/%s_g%lld_r%d.bin",
                  tp.sync_dir.c_str(), tag, (long long)gen, rank);
    return std::string(buf);
}

inline std::string make_ready(const TPConfig& tp, const char* tag,
                              int64_t gen, int rank) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/%s_g%lld_r%d.ready",
                  tp.sync_dir.c_str(), tag, (long long)gen, rank);
    return std::string(buf);
}

// Atomic publish: write full payload to a tmp file, rename into place, then
// touch a sibling .ready marker. Readers wait for the marker before opening
// the data file, which guarantees they never see a partial write.
inline void atomic_write_blob(const std::string& final_path,
                              const std::string& ready_path,
                              const void* data, size_t bytes) {
    std::string tmp = final_path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("tp: fopen tmp failed: " + tmp);
    }
    size_t wrote = std::fwrite(data, 1, bytes, f);
    std::fflush(f);
    std::fclose(f);
    if (wrote != bytes) {
        PT_TP_UNLINK(tmp.c_str());
        throw std::runtime_error("tp: short write " + tmp);
    }
    // rename is atomic on POSIX/tmpfs. On Windows we settle for unlink+rename.
#if defined(_WIN32)
    PT_TP_UNLINK(final_path.c_str());
    if (std::rename(tmp.c_str(), final_path.c_str()) != 0) {
        throw std::runtime_error("tp: rename failed: " + final_path);
    }
#else
    if (std::rename(tmp.c_str(), final_path.c_str()) != 0) {
        throw std::runtime_error("tp: rename failed: " + final_path);
    }
#endif
    FILE* mf = std::fopen(ready_path.c_str(), "w");
    if (mf) { std::fputc('1', mf); std::fclose(mf); }
}

inline void wait_for(const std::string& path, long timeout_us) {
    long waited = 0;
    while (PT_TP_ACCESS(path.c_str()) != 0) {
        pt_tp_usleep(1000);
        waited += 1000;
        if (waited > timeout_us) {
            throw std::runtime_error("tp: timeout waiting for " + path);
        }
    }
}

inline void read_blob(const std::string& path, void* dst, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("tp: fopen failed: " + path);
    size_t got = std::fread(dst, 1, bytes, f);
    std::fclose(f);
    if (got != bytes) throw std::runtime_error("tp: short read " + path);
}

inline void cleanup_gen(const TPConfig& tp, const char* tag, int64_t gen) {
    for (int r = 0; r < tp.world_size; ++r) {
        PT_TP_UNLINK(make_path(tp, tag, gen, r).c_str());
        PT_TP_UNLINK(make_ready(tp, tag, gen, r).c_str());
    }
}

}  // namespace detail

// ----------------------------------------------------------------------------
// Barrier — every rank publishes a marker, waits for everyone else's marker.
// ----------------------------------------------------------------------------

inline void tp_barrier(const TPConfig& tp) {
    if (tp.world_size <= 1) return;
    detail::ensure_dir(tp);
    int64_t gen = ++tp.gen_;
    std::string my_ready = detail::make_ready(tp, "bar", gen, tp.rank);
    FILE* f = std::fopen(my_ready.c_str(), "w");
    if (f) { std::fputc('1', f); std::fclose(f); }
    for (int r = 0; r < tp.world_size; ++r) {
        detail::wait_for(detail::make_ready(tp, "bar", gen, r), tp.timeout_us);
    }
    if (tp.rank == 0) {
        // Lazy cleanup of *previous* barrier generation only — current one is
        // still being read by stragglers.
        if (gen > 1) detail::cleanup_gen(tp, "bar", gen - 1);
    }
}

// ----------------------------------------------------------------------------
// All-gather along arbitrary dim. Each rank contributes a contiguous shard;
// shards must already share all dims except `gather_dim`.
// ----------------------------------------------------------------------------

inline Tensor tp_all_gather(const Tensor& local, int gather_dim,
                            const TPConfig& tp) {
    if (tp.world_size <= 1) return local;
    if (local.dtype() != c10::ScalarType::Float) {
        throw std::runtime_error("tp_all_gather: only float32 supported");
    }
    detail::ensure_dir(tp);
    int64_t gen = ++tp.gen_;

    Tensor local_c = local.contiguous();
    size_t local_bytes = local_c.numel() * sizeof(float);

    // 1. Publish my shard.
    detail::atomic_write_blob(
        detail::make_path(tp, "ag", gen, tp.rank),
        detail::make_ready(tp, "ag", gen, tp.rank),
        local_c.data_ptr<float>(), local_bytes);

    // 2. Collect every rank's shard (assumes all shards have identical shape;
    //    for unequal splits the caller is responsible for padding upstream).
    std::vector<Tensor> shards;
    shards.reserve(tp.world_size);
    for (int r = 0; r < tp.world_size; ++r) {
        if (r == tp.rank) {
            shards.push_back(local_c);
            continue;
        }
        detail::wait_for(detail::make_ready(tp, "ag", gen, r), tp.timeout_us);
        Tensor t = at::empty(local_c.sizes());
        detail::read_blob(detail::make_path(tp, "ag", gen, r),
                          t.mutable_data_ptr<float>(), local_bytes);
        shards.push_back(t);
    }

    // 3. Concatenate into the full tensor.
    Tensor full = at::native::cat(shards, gather_dim);

    // 4. Synchronize so no rank deletes files another is still reading, then
    //    rank 0 reaps the generation.
    tp_barrier(tp);
    if (tp.rank == 0) detail::cleanup_gen(tp, "ag", gen);

    return full;
}

// ----------------------------------------------------------------------------
// All-reduce sum. Every rank ends up with the same elementwise sum.
// ----------------------------------------------------------------------------

inline Tensor tp_all_reduce_sum(const Tensor& local, const TPConfig& tp) {
    if (tp.world_size <= 1) return local;
    if (local.dtype() != c10::ScalarType::Float) {
        throw std::runtime_error("tp_all_reduce_sum: only float32 supported");
    }
    detail::ensure_dir(tp);
    int64_t gen = ++tp.gen_;

    Tensor local_c = local.contiguous();
    int64_t n = local_c.numel();
    size_t bytes = n * sizeof(float);

    detail::atomic_write_blob(
        detail::make_path(tp, "ar", gen, tp.rank),
        detail::make_ready(tp, "ar", gen, tp.rank),
        local_c.data_ptr<float>(), bytes);

    Tensor result = at::empty(local_c.sizes());
    float* out = result.mutable_data_ptr<float>();
    std::memcpy(out, local_c.data_ptr<float>(), bytes);

    std::vector<float> tmp(n);
    for (int r = 0; r < tp.world_size; ++r) {
        if (r == tp.rank) continue;
        detail::wait_for(detail::make_ready(tp, "ar", gen, r), tp.timeout_us);
        detail::read_blob(detail::make_path(tp, "ar", gen, r),
                          tmp.data(), bytes);
        for (int64_t i = 0; i < n; ++i) out[i] += tmp[i];
    }

    tp_barrier(tp);
    if (tp.rank == 0) detail::cleanup_gen(tp, "ar", gen);
    return result;
}

// ============================================================================
// ColumnParallelLinear
// ============================================================================
// Weight stored in PyTorch convention W[out, in]; we shard the OUT dimension
// across ranks. Each rank owns rows [rank*shard, (rank+1)*shard) of W and the
// matching slice of bias. The forward computes the local slice y_local =
// x @ W_local^T + b_local; if `gather_output` is true we all_gather along the
// last dim to reproduce the full output.
//
// Init contract: every rank produces the *same* full weight (seeded RNG) and
// then keeps only its own slice. This makes weight-loading deterministic and
// matches how Megatron initializes column-parallel layers.
// ============================================================================

class ColumnParallelLinear : public Module {
public:
    ColumnParallelLinear(int64_t in_features, int64_t out_features,
                         TPConfig tp, bool gather_output = true,
                         bool bias = true, uint64_t init_seed = 1234567ULL)
        : Module("ColumnParallelLinear")
        , in_features_(in_features)
        , out_features_(out_features)
        , tp_(tp)
        , gather_output_(gather_output)
        , has_bias_(bias)
    {
        if (out_features % tp_.world_size != 0) {
            throw std::runtime_error(
                "ColumnParallelLinear: out_features must be divisible by world_size");
        }
        shard_ = out_features / tp_.world_size;

        // Allocate sharded parameters [shard_, in_features].
        register_parameter("weight",
            Parameter(at::empty({shard_, in_features})));
        if (has_bias_) {
            register_parameter("bias", Parameter(at::empty({shard_})));
        }
        seed_ = init_seed;
        reset_parameters();
    }

    void reset_parameters() override {
        // Identical RNG state on every rank → identical full weight; we then
        // keep only our slice. This guarantees that ColumnParallel + gather
        // is mathematically identical to a plain Linear with the same seed.
        double bound = 1.0 / std::sqrt(static_cast<double>(in_features_));
        Tensor W = get_parameter("weight")->data();
        float* wd = W.mutable_data_ptr<float>();
        uint64_t s = seed_;
        // Skip rows belonging to lower ranks.
        for (int64_t i = 0; i < tp_.rank * shard_ * in_features_; ++i) {
            (void)next_uniform_(s, bound);
        }
        for (int64_t i = 0; i < shard_ * in_features_; ++i) {
            wd[i] = next_uniform_(s, bound);
        }
        if (has_bias_) {
            // Continue RNG stream past all weight elements, then skip lower
            // ranks' bias slice.
            for (int64_t i = (tp_.rank + 1) * shard_ * in_features_;
                 i < out_features_ * in_features_; ++i) {
                (void)next_uniform_(s, bound);
            }
            for (int64_t i = 0; i < tp_.rank * shard_; ++i) {
                (void)next_uniform_(s, bound);
            }
            Tensor b = get_parameter("bias")->data();
            float* bd = b.mutable_data_ptr<float>();
            for (int64_t i = 0; i < shard_; ++i) bd[i] = next_uniform_(s, bound);
        }
    }

    Tensor forward(const Tensor& input) override {
        Tensor W = get_parameter("weight")->data();   // [shard, in]
        // y_local = input @ W^T   -> last dim becomes `shard`.
        Tensor Wt = torch::autograd::t_autograd(W);
        Tensor flat = input;
        std::vector<int64_t> out_shape;
        if (input.dim() != 2) {
            int64_t batch = 1;
            for (int64_t d = 0; d < input.dim() - 1; ++d) batch *= input.size(d);
            flat = torch::autograd::reshape_autograd(input, {batch, in_features_});
            out_shape.assign(input.sizes().begin(), input.sizes().end() - 1);
        }
        Tensor y = torch::autograd::mm_autograd(flat, Wt);  // [batch, shard]
        if (has_bias_) {
            y = torch::autograd::add_autograd(y, get_parameter("bias")->data());
        }
        if (input.dim() != 2) {
            std::vector<int64_t> shape = out_shape;
            shape.push_back(shard_);
            y = torch::autograd::reshape_autograd(y, shape);
        }
        if (!gather_output_ || tp_.world_size == 1) return y;

        // Gather along the last (feature) dim. all_gather is currently a
        // forward-only collective; gradients of the gathered output are
        // assumed to flow only through this rank's shard (which is the
        // standard Megatron convention when followed by a RowParallelLinear).
        return tp_all_gather(y, /*gather_dim=*/y.dim() - 1, tp_);
    }

    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    int64_t shard_size() const { return shard_; }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "in_features=" << in_features_
           << ", out_features=" << out_features_
           << " (shard=" << shard_ << "/" << tp_.world_size << ")"
           << ", gather_output=" << (gather_output_ ? "True" : "False")
           << ", bias=" << (has_bias_ ? "True" : "False");
        return ss.str();
    }

private:
    static float next_uniform_(uint64_t& state, double bound) {
        // splitmix64 → uniform in [-bound, bound). Same on every rank.
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        double u = (double)(z >> 11) / (double)(1ULL << 53);  // [0,1)
        return (float)((2.0 * u - 1.0) * bound);
    }

    int64_t in_features_;
    int64_t out_features_;
    int64_t shard_ = 0;
    TPConfig tp_;
    bool gather_output_;
    bool has_bias_;
    uint64_t seed_ = 1234567ULL;
};

// ============================================================================
// RowParallelLinear
// ============================================================================
// Weight stored as W[out, in]; we shard the IN dimension. Each rank owns
// columns [rank*shard, (rank+1)*shard) of W. Forward expects either:
//   * input_is_parallel = true  : input already contains this rank's slice
//                                 along the last dim (shape [..., in/world])
//   * input_is_parallel = false : full input [..., in]; this rank slices it
// The local matmul produces a partial sum over the in dim; we all_reduce_sum
// to combine partials. Bias (full-width) is added once after the reduce.
// ============================================================================

class RowParallelLinear : public Module {
public:
    RowParallelLinear(int64_t in_features, int64_t out_features,
                      TPConfig tp, bool input_is_parallel = true,
                      bool bias = true, uint64_t init_seed = 7654321ULL)
        : Module("RowParallelLinear")
        , in_features_(in_features)
        , out_features_(out_features)
        , tp_(tp)
        , input_is_parallel_(input_is_parallel)
        , has_bias_(bias)
    {
        if (in_features % tp_.world_size != 0) {
            throw std::runtime_error(
                "RowParallelLinear: in_features must be divisible by world_size");
        }
        shard_ = in_features / tp_.world_size;

        register_parameter("weight",
            Parameter(at::empty({out_features, shard_})));
        if (has_bias_) {
            register_parameter("bias", Parameter(at::empty({out_features})));
        }
        seed_ = init_seed;
        reset_parameters();
    }

    void reset_parameters() override {
        // Same trick as ColumnParallel: identical RNG → keep our column slice.
        double bound = 1.0 / std::sqrt(static_cast<double>(in_features_));
        Tensor W = get_parameter("weight")->data();   // [out, shard]
        float* wd = W.mutable_data_ptr<float>();
        uint64_t s = seed_;
        // The full virtual weight is W_full[out, in]; we walk it row by row.
        for (int64_t r = 0; r < out_features_; ++r) {
            for (int64_t c = 0; c < in_features_; ++c) {
                float v = next_uniform_(s, bound);
                if (c >= tp_.rank * shard_ && c < (tp_.rank + 1) * shard_) {
                    wd[r * shard_ + (c - tp_.rank * shard_)] = v;
                }
            }
        }
        if (has_bias_) {
            // Bias is replicated, not sharded — same value on every rank.
            Tensor b = get_parameter("bias")->data();
            float* bd = b.mutable_data_ptr<float>();
            for (int64_t i = 0; i < out_features_; ++i) bd[i] = next_uniform_(s, bound);
        }
    }

    Tensor forward(const Tensor& input) override {
        Tensor W = get_parameter("weight")->data();   // [out, shard]
        Tensor x = input;

        if (!input_is_parallel_) {
            // Slice the last dim down to our shard. We keep this outside of
            // autograd: the upstream layer is expected to produce a sharded
            // tensor; non-parallel input is a convenience for testing only.
            int64_t last = x.dim() - 1;
            x = x.narrow(last, tp_.rank * shard_, shard_).contiguous();
        }

        // Flatten leading dims for mm.
        std::vector<int64_t> out_shape;
        Tensor flat = x;
        if (x.dim() != 2) {
            int64_t batch = 1;
            for (int64_t d = 0; d < x.dim() - 1; ++d) batch *= x.size(d);
            flat = torch::autograd::reshape_autograd(x, {batch, shard_});
            out_shape.assign(x.sizes().begin(), x.sizes().end() - 1);
        }

        Tensor Wt = torch::autograd::t_autograd(W);     // [shard, out]
        Tensor partial = torch::autograd::mm_autograd(flat, Wt);   // [batch, out]

        if (x.dim() != 2) {
            std::vector<int64_t> shape = out_shape;
            shape.push_back(out_features_);
            partial = torch::autograd::reshape_autograd(partial, shape);
        }

        // All-reduce SUM across the in-dim shards. Bias is added *after*
        // the reduce so it isn't summed `world_size` times.
        Tensor reduced = (tp_.world_size > 1) ? tp_all_reduce_sum(partial, tp_)
                                              : partial;
        if (has_bias_) {
            reduced = torch::autograd::add_autograd(
                reduced, get_parameter("bias")->data());
        }
        return reduced;
    }

    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    int64_t shard_size() const { return shard_; }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "in_features=" << in_features_
           << " (shard=" << shard_ << "/" << tp_.world_size << ")"
           << ", out_features=" << out_features_
           << ", input_is_parallel=" << (input_is_parallel_ ? "True" : "False")
           << ", bias=" << (has_bias_ ? "True" : "False");
        return ss.str();
    }

private:
    static float next_uniform_(uint64_t& state, double bound) {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        double u = (double)(z >> 11) / (double)(1ULL << 53);
        return (float)((2.0 * u - 1.0) * bound);
    }

    int64_t in_features_;
    int64_t out_features_;
    int64_t shard_ = 0;
    TPConfig tp_;
    bool input_is_parallel_;
    bool has_bias_;
    uint64_t seed_ = 7654321ULL;
};

}  // namespace parallel
}  // namespace nn
}  // namespace torch
