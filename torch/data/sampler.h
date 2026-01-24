#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <optional>
#include <cstdint>

namespace torch {
namespace data {

// ============================================================================
// Sampler - Base class for all samplers
// ============================================================================
// Samplers define the order in which dataset indices are accessed.
// They provide an iterable over indices.
//
// Usage:
//   RandomSampler sampler(dataset.size());
//   for (size_t idx : sampler) {
//       auto example = dataset.get(idx);
//   }

class Sampler {
public:
    virtual ~Sampler() = default;

    // Reset sampler for new epoch
    virtual void reset() = 0;

    // Get next index, returns nullopt when exhausted
    virtual std::optional<size_t> next() = 0;

    // Total number of indices this sampler will produce
    virtual size_t size() const = 0;

    // Iterator support
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const size_t*;
        using reference = const size_t&;

        Iterator() : sampler_(nullptr), current_(), at_end_(true) {}

        Iterator(Sampler* sampler) : sampler_(sampler), at_end_(false) {
            advance();
        }

        size_t operator*() const { return current_.value(); }

        Iterator& operator++() {
            advance();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            advance();
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return at_end_ == other.at_end_;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        void advance() {
            if (sampler_) {
                current_ = sampler_->next();
                at_end_ = !current_.has_value();
            }
        }

        Sampler* sampler_;
        std::optional<size_t> current_;
        bool at_end_;
    };

    Iterator begin() {
        reset();
        return Iterator(this);
    }

    Iterator end() {
        return Iterator();
    }
};

// ============================================================================
// SequentialSampler - Sample elements sequentially
// ============================================================================
// Always produces indices 0, 1, 2, ..., n-1 in order.

class SequentialSampler : public Sampler {
public:
    explicit SequentialSampler(size_t size)
        : size_(size), current_(0) {}

    void reset() override {
        current_ = 0;
    }

    std::optional<size_t> next() override {
        if (current_ >= size_) {
            return std::nullopt;
        }
        return current_++;
    }

    size_t size() const override {
        return size_;
    }

private:
    size_t size_;
    size_t current_;
};

// ============================================================================
// RandomSampler - Sample elements randomly
// ============================================================================
// Produces all indices in random order (without replacement by default).
//
// With replacement=true, samples indices randomly with replacement,
// potentially sampling the same index multiple times.

class RandomSampler : public Sampler {
public:
    explicit RandomSampler(
        size_t size,
        bool replacement = false,
        std::optional<size_t> num_samples = std::nullopt,
        std::optional<uint64_t> seed = std::nullopt
    )
        : size_(size)
        , replacement_(replacement)
        , num_samples_(num_samples.value_or(size))
        , gen_(seed.value_or(std::random_device{}()))
        , current_(0)
    {
        if (!replacement) {
            shuffle_indices();
        }
    }

    void reset() override {
        current_ = 0;
        if (!replacement_) {
            shuffle_indices();
        }
    }

    std::optional<size_t> next() override {
        if (current_ >= num_samples_) {
            return std::nullopt;
        }

        if (replacement_) {
            std::uniform_int_distribution<size_t> dist(0, size_ - 1);
            ++current_;
            return dist(gen_);
        } else {
            return indices_[current_++];
        }
    }

    size_t size() const override {
        return num_samples_;
    }

    // Set seed for reproducibility
    void set_seed(uint64_t seed) {
        gen_.seed(seed);
    }

private:
    void shuffle_indices() {
        indices_.resize(size_);
        for (size_t i = 0; i < size_; ++i) {
            indices_[i] = i;
        }
        std::shuffle(indices_.begin(), indices_.end(), gen_);
    }

    size_t size_;
    bool replacement_;
    size_t num_samples_;
    std::mt19937_64 gen_;
    size_t current_;
    std::vector<size_t> indices_;
};

// ============================================================================
// SubsetRandomSampler - Sample randomly from a subset of indices
// ============================================================================
// Useful for creating train/val splits.

class SubsetRandomSampler : public Sampler {
public:
    explicit SubsetRandomSampler(
        std::vector<size_t> indices,
        std::optional<uint64_t> seed = std::nullopt
    )
        : indices_(std::move(indices))
        , gen_(seed.value_or(std::random_device{}()))
        , current_(0)
    {
        shuffle_indices();
    }

    void reset() override {
        current_ = 0;
        shuffle_indices();
    }

    std::optional<size_t> next() override {
        if (current_ >= indices_.size()) {
            return std::nullopt;
        }
        return shuffled_[current_++];
    }

    size_t size() const override {
        return indices_.size();
    }

private:
    void shuffle_indices() {
        shuffled_ = indices_;
        std::shuffle(shuffled_.begin(), shuffled_.end(), gen_);
    }

    std::vector<size_t> indices_;
    std::vector<size_t> shuffled_;
    std::mt19937_64 gen_;
    size_t current_;
};

// ============================================================================
// WeightedRandomSampler - Sample with weights
// ============================================================================
// Samples indices according to specified weights (with replacement).
// Higher weight = higher probability of being sampled.

class WeightedRandomSampler : public Sampler {
public:
    WeightedRandomSampler(
        std::vector<double> weights,
        size_t num_samples,
        bool replacement = true,
        std::optional<uint64_t> seed = std::nullopt
    )
        : weights_(std::move(weights))
        , num_samples_(num_samples)
        , replacement_(replacement)
        , gen_(seed.value_or(std::random_device{}()))
        , dist_(weights_.begin(), weights_.end())
        , current_(0)
    {}

    void reset() override {
        current_ = 0;
        if (!replacement_) {
            // For non-replacement, we need to pre-generate all samples
            sampled_indices_.clear();
            sampled_indices_.reserve(num_samples_);
            std::vector<bool> selected(weights_.size(), false);

            while (sampled_indices_.size() < num_samples_ &&
                   sampled_indices_.size() < weights_.size()) {
                size_t idx = dist_(gen_);
                if (!selected[idx]) {
                    selected[idx] = true;
                    sampled_indices_.push_back(idx);
                }
            }
        }
    }

    std::optional<size_t> next() override {
        if (current_ >= num_samples_) {
            return std::nullopt;
        }

        if (replacement_) {
            ++current_;
            return dist_(gen_);
        } else {
            if (current_ >= sampled_indices_.size()) {
                return std::nullopt;
            }
            return sampled_indices_[current_++];
        }
    }

    size_t size() const override {
        return num_samples_;
    }

private:
    std::vector<double> weights_;
    size_t num_samples_;
    bool replacement_;
    std::mt19937_64 gen_;
    std::discrete_distribution<size_t> dist_;
    size_t current_;
    std::vector<size_t> sampled_indices_;  // For non-replacement
};

// ============================================================================
// BatchSampler - Wrap sampler to yield batches of indices
// ============================================================================
// Takes a base sampler and groups its outputs into batches.

class BatchSampler {
public:
    BatchSampler(
        std::unique_ptr<Sampler> sampler,
        size_t batch_size,
        bool drop_last = false
    )
        : sampler_(std::move(sampler))
        , batch_size_(batch_size)
        , drop_last_(drop_last)
    {}

    // Reset for new epoch
    void reset() {
        sampler_->reset();
    }

    // Get next batch of indices
    std::optional<std::vector<size_t>> next_batch() {
        std::vector<size_t> batch;
        batch.reserve(batch_size_);

        while (batch.size() < batch_size_) {
            auto idx = sampler_->next();
            if (!idx.has_value()) {
                break;
            }
            batch.push_back(idx.value());
        }

        if (batch.empty()) {
            return std::nullopt;
        }

        if (drop_last_ && batch.size() < batch_size_) {
            return std::nullopt;
        }

        return batch;
    }

    // Approximate number of batches
    size_t size() const {
        size_t n = sampler_->size();
        if (drop_last_) {
            return n / batch_size_;
        }
        return (n + batch_size_ - 1) / batch_size_;
    }

    // Iterator support
    class Iterator {
    public:
        Iterator() : sampler_(nullptr), at_end_(true) {}

        Iterator(BatchSampler* sampler) : sampler_(sampler), at_end_(false) {
            advance();
        }

        const std::vector<size_t>& operator*() const { return current_.value(); }

        Iterator& operator++() {
            advance();
            return *this;
        }

        bool operator==(const Iterator& other) const {
            return at_end_ == other.at_end_;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        void advance() {
            if (sampler_) {
                current_ = sampler_->next_batch();
                at_end_ = !current_.has_value();
            }
        }

        BatchSampler* sampler_;
        std::optional<std::vector<size_t>> current_;
        bool at_end_;
    };

    Iterator begin() {
        reset();
        return Iterator(this);
    }

    Iterator end() {
        return Iterator();
    }

private:
    std::unique_ptr<Sampler> sampler_;
    size_t batch_size_;
    bool drop_last_;
};

// ============================================================================
// Factory functions
// ============================================================================

inline std::unique_ptr<SequentialSampler> make_sequential_sampler(size_t size) {
    return std::make_unique<SequentialSampler>(size);
}

inline std::unique_ptr<RandomSampler> make_random_sampler(
    size_t size,
    bool replacement = false,
    std::optional<uint64_t> seed = std::nullopt
) {
    return std::make_unique<RandomSampler>(size, replacement, std::nullopt, seed);
}

inline std::unique_ptr<BatchSampler> make_batch_sampler(
    std::unique_ptr<Sampler> sampler,
    size_t batch_size,
    bool drop_last = false
) {
    return std::make_unique<BatchSampler>(std::move(sampler), batch_size, drop_last);
}

} // namespace data
} // namespace torch
