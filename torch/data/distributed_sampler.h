#pragma once
// ============================================================================
// distributed_sampler.h — DistributedSampler for DDP training
// ============================================================================
// Splits dataset indices across `num_replicas` (ranks) so each rank gets a
// unique, non-overlapping subset per epoch. Mirrors PyTorch's
// torch.utils.data.distributed.DistributedSampler semantics.
//
// Algorithm (per epoch):
//   1. Build [0, 1, ..., dataset_size-1].
//   2. If shuffle: deterministic shuffle with seed = (seed + epoch).
//      Same seed across ranks => identical permutation => no overlap.
//   3. If drop_last: truncate to a multiple of num_replicas.
//      Else: pad with wrap-around indices to next multiple of num_replicas.
//   4. Slice with stride: indices[rank :: num_replicas]
//      (same scheme as PyTorch).
//   5. Yield in `batch_size` chunks via sample().
//
// Usage:
//   DistributedSampler s(dataset.size(), world_size, rank, /*shuffle=*/true);
//   for (uint64_t epoch = 0; epoch < n_epochs; ++epoch) {
//       s.set_epoch(epoch);
//       while (auto batch = s.sample(batch_size); !batch.empty()) {
//           train_step(batch);
//       }
//   }
// ============================================================================

#include "torch/data/sampler.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace torch {
namespace data {

class DistributedSampler : public Sampler {
public:
    DistributedSampler(size_t dataset_size,
                       int num_replicas,
                       int rank,
                       bool shuffle = true,
                       uint64_t seed = 0,
                       bool drop_last = false)
        : dataset_size_(dataset_size)
        , num_replicas_(num_replicas)
        , rank_(rank)
        , shuffle_(shuffle)
        , seed_(seed)
        , drop_last_(drop_last)
        , epoch_(0)
        , cursor_(0)
    {
        if (num_replicas_ <= 0) {
            throw std::invalid_argument(
                "DistributedSampler: num_replicas must be > 0");
        }
        if (rank_ < 0 || rank_ >= num_replicas_) {
            throw std::invalid_argument(
                "DistributedSampler: rank must be in [0, num_replicas)");
        }

        // Compute num_samples_ and total_size_ deterministically.
        if (drop_last_) {
            // Drop tail so size divides evenly.
            num_samples_ = dataset_size_ / static_cast<size_t>(num_replicas_);
            total_size_  = num_samples_ * static_cast<size_t>(num_replicas_);
        } else {
            // Pad up to next multiple of num_replicas.
            num_samples_ = (dataset_size_ + static_cast<size_t>(num_replicas_) - 1)
                         / static_cast<size_t>(num_replicas_);
            total_size_  = num_samples_ * static_cast<size_t>(num_replicas_);
        }

        rebuild_indices_();
    }

    // ------------------------------------------------------------------
    // Sampler interface
    // ------------------------------------------------------------------
    void reset() override {
        cursor_ = 0;
        rebuild_indices_();
    }

    std::optional<size_t> next() override {
        if (cursor_ >= local_indices_.size()) return std::nullopt;
        return local_indices_[cursor_++];
    }

    size_t size() const override { return num_samples_; }

    // ------------------------------------------------------------------
    // Batched sampling
    // ------------------------------------------------------------------
    // Returns up to `batch_size` indices for this rank, advancing the cursor.
    // Returns an empty vector when the epoch is exhausted.
    std::vector<size_t> sample(size_t batch_size) {
        std::vector<size_t> batch;
        if (cursor_ >= local_indices_.size() || batch_size == 0) return batch;
        const size_t take = std::min(batch_size, local_indices_.size() - cursor_);
        batch.reserve(take);
        for (size_t i = 0; i < take; ++i) {
            batch.push_back(local_indices_[cursor_++]);
        }
        return batch;
    }

    // ------------------------------------------------------------------
    // Epoch control — call BEFORE iterating each epoch so the shuffle
    // permutation is unique per epoch but identical across ranks.
    // ------------------------------------------------------------------
    void set_epoch(uint64_t epoch) {
        epoch_  = epoch;
        cursor_ = 0;
        rebuild_indices_();
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------
    size_t num_samples() const { return num_samples_; }
    size_t total_size()  const { return total_size_; }
    int    num_replicas() const { return num_replicas_; }
    int    rank() const { return rank_; }
    uint64_t epoch() const { return epoch_; }
    uint64_t seed()  const { return seed_; }
    bool   shuffle() const { return shuffle_; }
    bool   drop_last() const { return drop_last_; }

private:
    // Rebuild local_indices_ for the current (seed, epoch, rank).
    void rebuild_indices_() {
        // 1. Full permutation (or identity).
        std::vector<size_t> all(dataset_size_);
        for (size_t i = 0; i < dataset_size_; ++i) all[i] = i;

        if (shuffle_) {
            // Same RNG state on every rank ⇒ identical permutation.
            std::mt19937_64 gen(seed_ + epoch_);
            std::shuffle(all.begin(), all.end(), gen);
        }

        // 2. drop_last: truncate, else pad with wrap-around.
        if (drop_last_) {
            if (all.size() > total_size_) all.resize(total_size_);
        } else if (all.size() < total_size_) {
            const size_t pad = total_size_ - all.size();
            if (dataset_size_ == 0) {
                // Edge: empty dataset — nothing to pad with. Leave empty.
                all.clear();
            } else {
                all.reserve(total_size_);
                for (size_t i = 0; i < pad; ++i) {
                    all.push_back(all[i % dataset_size_]);
                }
            }
        }

        // 3. Strided slice: indices[rank :: num_replicas]  (PyTorch semantics).
        local_indices_.clear();
        local_indices_.reserve(num_samples_);
        for (size_t i = static_cast<size_t>(rank_); i < all.size();
             i += static_cast<size_t>(num_replicas_)) {
            local_indices_.push_back(all[i]);
        }
    }

    // Public-spec fields (per task contract).
    size_t   dataset_size_;
    size_t   total_size_;
    size_t   num_samples_;
    int      num_replicas_;
    int      rank_;
    bool     shuffle_;
    bool     drop_last_;
    uint64_t seed_;
    uint64_t epoch_;

    // Internal state.
    std::vector<size_t> local_indices_;
    size_t              cursor_;
};

} // namespace data
} // namespace torch
