#pragma once

#include "torch/data/dataset.h"
#include "torch/data/sampler.h"
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

namespace torch {
namespace data {

// ============================================================================
// Batch - A batch of examples
// ============================================================================
// Contains stacked data and target tensors for a batch of examples.

template<typename Data = Tensor, typename Target = Tensor>
struct Batch {
    Data data;
    Target target;
    size_t size;  // Number of examples in batch

    Batch() : size(0) {}

    Batch(Data data_, Target target_, size_t size_)
        : data(std::move(data_)), target(std::move(target_)), size(size_) {}
};

// ============================================================================
// Collate Functions - Stack examples into batches
// ============================================================================

// Default collate: stack tensors along new dimension 0
template<typename Data = Tensor, typename Target = Tensor>
struct DefaultCollate {
    Batch<Data, Target> operator()(
        const std::vector<Example<Data, Target>>& examples
    ) const {
        if (examples.empty()) {
            return Batch<Data, Target>();
        }

        // Stack data tensors
        std::vector<Tensor> data_tensors;
        std::vector<Tensor> target_tensors;
        data_tensors.reserve(examples.size());

        bool has_targets = examples[0].target.defined();
        if (has_targets) {
            target_tensors.reserve(examples.size());
        }

        for (const auto& ex : examples) {
            data_tensors.push_back(ex.data);
            if (has_targets) {
                target_tensors.push_back(ex.target);
            }
        }

        // Stack into batch
        Tensor batch_data = at::native::stack(data_tensors, 0);
        Tensor batch_target;
        if (has_targets) {
            batch_target = at::native::stack(target_tensors, 0);
        }

        return Batch<Data, Target>(
            std::move(batch_data),
            std::move(batch_target),
            examples.size()
        );
    }
};

// ============================================================================
// DataLoaderOptions - Configuration for DataLoader
// ============================================================================

struct DataLoaderOptions {
    size_t batch_size = 1;
    bool shuffle = false;
    size_t num_workers = 0;      // 0 = main thread only
    bool drop_last = false;      // Drop incomplete last batch
    bool pin_memory = false;     // Pin memory for faster GPU transfer (future)
    std::optional<uint64_t> seed = std::nullopt;  // Random seed

    DataLoaderOptions& batch_size_(size_t value) { batch_size = value; return *this; }
    DataLoaderOptions& shuffle_(bool value) { shuffle = value; return *this; }
    DataLoaderOptions& num_workers_(size_t value) { num_workers = value; return *this; }
    DataLoaderOptions& drop_last_(bool value) { drop_last = value; return *this; }
    DataLoaderOptions& pin_memory_(bool value) { pin_memory = value; return *this; }
    DataLoaderOptions& seed_(uint64_t value) { seed = value; return *this; }
};

// ============================================================================
// DataLoader - Iterate over dataset in batches
// ============================================================================
// Combines a dataset with a sampler and provides an iterable over batches.
//
// Usage:
//   TensorDataset dataset(inputs, labels);
//   DataLoader loader(dataset, DataLoaderOptions().batch_size_(32).shuffle_(true));
//
//   for (auto& batch : loader) {
//       auto output = model.forward(batch.data);
//       auto loss = criterion(output, batch.target);
//       loss.backward();
//       optimizer.step();
//   }

template<
    typename Dataset,
    typename Collate = DefaultCollate<typename Dataset::DataType, typename Dataset::TargetType>
>
class DataLoader {
public:
    using DataType = typename Dataset::DataType;
    using TargetType = typename Dataset::TargetType;
    using ExampleType = typename Dataset::ExampleType;
    using BatchType = Batch<DataType, TargetType>;

    DataLoader(
        Dataset dataset,
        DataLoaderOptions options = DataLoaderOptions(),
        Collate collate_fn = Collate()
    )
        : dataset_(std::move(dataset))
        , options_(options)
        , collate_fn_(std::move(collate_fn))
        , batch_sampler_(nullptr)
    {
        // Create appropriate sampler
        std::unique_ptr<Sampler> sampler;
        if (options_.shuffle) {
            sampler = std::make_unique<RandomSampler>(
                dataset_.size(), false, std::nullopt, options_.seed
            );
        } else {
            sampler = std::make_unique<SequentialSampler>(dataset_.size());
        }

        // Wrap in batch sampler
        batch_sampler_ = std::make_unique<BatchSampler>(
            std::move(sampler),
            options_.batch_size,
            options_.drop_last
        );
    }

    // Constructor with custom sampler
    DataLoader(
        Dataset dataset,
        std::unique_ptr<Sampler> sampler,
        DataLoaderOptions options = DataLoaderOptions(),
        Collate collate_fn = Collate()
    )
        : dataset_(std::move(dataset))
        , options_(options)
        , collate_fn_(std::move(collate_fn))
    {
        batch_sampler_ = std::make_unique<BatchSampler>(
            std::move(sampler),
            options_.batch_size,
            options_.drop_last
        );
    }

    // Number of batches
    size_t size() const {
        return batch_sampler_->size();
    }

    // Dataset size
    size_t dataset_size() const {
        return dataset_.size();
    }

    // Batch size
    size_t batch_size() const {
        return options_.batch_size;
    }

    // Iterator class
    class Iterator {
    public:
        Iterator() : loader_(nullptr), at_end_(true) {}

        Iterator(DataLoader* loader) : loader_(loader), at_end_(false) {
            advance();
        }

        const BatchType& operator*() const { return current_batch_; }
        const BatchType* operator->() const { return &current_batch_; }

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
            if (!loader_) {
                at_end_ = true;
                return;
            }

            auto indices = loader_->batch_sampler_->next_batch();
            if (!indices.has_value()) {
                at_end_ = true;
                return;
            }

            // Fetch examples
            std::vector<ExampleType> examples;
            examples.reserve(indices->size());

            for (size_t idx : *indices) {
                examples.push_back(loader_->dataset_.get(idx));
            }

            // Collate into batch
            current_batch_ = loader_->collate_fn_(examples);
        }

        DataLoader* loader_;
        BatchType current_batch_;
        bool at_end_;
    };

    Iterator begin() {
        batch_sampler_->reset();
        return Iterator(this);
    }

    Iterator end() {
        return Iterator();
    }

    // Access dataset
    Dataset& dataset() { return dataset_; }
    const Dataset& dataset() const { return dataset_; }

private:
    Dataset dataset_;
    DataLoaderOptions options_;
    Collate collate_fn_;
    std::unique_ptr<BatchSampler> batch_sampler_;
};

// ============================================================================
// Factory function
// ============================================================================

template<typename Dataset>
DataLoader<Dataset> make_data_loader(
    Dataset dataset,
    DataLoaderOptions options = DataLoaderOptions()
) {
    return DataLoader<Dataset>(std::move(dataset), options);
}

template<typename Dataset>
DataLoader<Dataset> make_data_loader(
    Dataset dataset,
    size_t batch_size,
    bool shuffle = false
) {
    return DataLoader<Dataset>(
        std::move(dataset),
        DataLoaderOptions().batch_size_(batch_size).shuffle_(shuffle)
    );
}

// ============================================================================
// Utility: count batches
// ============================================================================

template<typename Dataset>
size_t count_batches(const Dataset& dataset, size_t batch_size, bool drop_last = false) {
    size_t n = dataset.size();
    if (drop_last) {
        return n / batch_size;
    }
    return (n + batch_size - 1) / batch_size;
}

} // namespace data
} // namespace torch
