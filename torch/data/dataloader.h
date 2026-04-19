#pragma once

#include "torch/data/dataset.h"
#include "torch/data/sampler.h"
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <utility>
#include <stdexcept>

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
    size_t num_workers = 0;      // 0 = main thread only (synchronous)
    size_t prefetch_factor = 2;  // Per-worker queue capacity multiplier (only when num_workers>0)
    bool drop_last = false;      // Drop incomplete last batch
    bool pin_memory = false;     // CPU noop; accepted for PyTorch API compat
    std::optional<uint64_t> seed = std::nullopt;  // Random seed

    DataLoaderOptions& batch_size_(size_t value) { batch_size = value; return *this; }
    DataLoaderOptions& shuffle_(bool value) { shuffle = value; return *this; }
    DataLoaderOptions& num_workers_(size_t value) { num_workers = value; return *this; }
    DataLoaderOptions& prefetch_factor_(size_t value) { prefetch_factor = value; return *this; }
    DataLoaderOptions& drop_last_(bool value) { drop_last = value; return *this; }
    DataLoaderOptions& pin_memory_(bool value) { pin_memory = value; return *this; }
    DataLoaderOptions& seed_(uint64_t value) { seed = value; return *this; }
};

// ============================================================================
// ThreadSafeQueue - bounded blocking queue used for the worker prefetch path
// ============================================================================
// Only header-only STL primitives are used so this compiles cleanly on the
// Elbrus LCC toolchain (no Boost / no platform-specific synchronization).
//
// close() is a one-shot signal: producers stop pushing, consumers drain the
// queue then receive nullopt from pop().

template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() : capacity_(1), closed_(false) {}
    explicit ThreadSafeQueue(size_t capacity)
        : capacity_(capacity == 0 ? 1 : capacity), closed_(false) {}

    // Returns false if the queue was closed before the push completed.
    bool push(T value) {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [this] { return closed_ || queue_.size() < capacity_; });
        if (closed_) {
            return false;
        }
        queue_.push_back(std::move(value));
        not_empty_.notify_one();
        return true;
    }

    // Blocks until an element is available or the queue has been closed AND drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mu_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return std::nullopt;  // closed and drained
        }
        T value = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return value;
    }

    // Wakes up all waiters. Subsequent push() calls fail; pop() drains and
    // then returns nullopt.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mu_);
        return closed_;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return queue_.size();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> queue_;
    size_t capacity_;
    bool closed_;
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

    // ========================================================================
    // PrefetchContext - per-iteration shared state used when num_workers > 0
    // ========================================================================
    // Lifetime: created by begin() if num_workers > 0; held by every Iterator
    // returned for that epoch (via shared_ptr) so that copy/move of Iterator
    // does not race the workers. shutdown() is idempotent and safe to call
    // from the destructor.
    struct PrefetchContext {
        ThreadSafeQueue<std::vector<size_t>> index_queue;
        ThreadSafeQueue<BatchType> batch_queue;
        std::vector<std::thread> workers;
        std::thread feeder;
        std::atomic<bool> stop;
        std::exception_ptr first_error;
        std::mutex error_mu;

        PrefetchContext(size_t idx_cap, size_t batch_cap)
            : index_queue(idx_cap), batch_queue(batch_cap), stop(false) {}

        ~PrefetchContext() { shutdown(); }

        void record_error(std::exception_ptr ep) {
            std::lock_guard<std::mutex> lock(error_mu);
            if (!first_error) {
                first_error = ep;
            }
        }

        void shutdown() {
            stop.store(true, std::memory_order_release);
            index_queue.close();
            batch_queue.close();
            if (feeder.joinable()) feeder.join();
            for (auto& w : workers) {
                if (w.joinable()) w.join();
            }
        }
    };

    // Iterator class
    class Iterator {
    public:
        Iterator() : loader_(nullptr), at_end_(true) {}

        Iterator(DataLoader* loader) : loader_(loader), at_end_(false) {
            advance();
        }

        Iterator(DataLoader* loader, std::shared_ptr<PrefetchContext> ctx)
            : loader_(loader), ctx_(std::move(ctx)), at_end_(false) {
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

            if (ctx_) {
                // Multi-worker prefetch path: pull a finished batch off the
                // ready queue. If a worker raised, propagate it on the main
                // thread.
                auto maybe_batch = ctx_->batch_queue.pop();
                if (!maybe_batch.has_value()) {
                    if (ctx_->first_error) {
                        std::exception_ptr ep = ctx_->first_error;
                        ctx_->shutdown();
                        std::rethrow_exception(ep);
                    }
                    at_end_ = true;
                    return;
                }
                current_batch_ = std::move(maybe_batch.value());
                return;
            }

            // Synchronous path (num_workers == 0) - identical to legacy behavior.
            auto indices = loader_->batch_sampler_->next_batch();
            if (!indices.has_value()) {
                at_end_ = true;
                return;
            }

            std::vector<ExampleType> examples;
            examples.reserve(indices->size());

            for (size_t idx : *indices) {
                examples.push_back(loader_->dataset_.get(idx));
            }

            current_batch_ = loader_->collate_fn_(examples);
        }

        DataLoader* loader_;
        std::shared_ptr<PrefetchContext> ctx_;
        BatchType current_batch_;
        bool at_end_;
    };

    Iterator begin() {
        batch_sampler_->reset();

        if (options_.num_workers == 0) {
            return Iterator(this);
        }

        // ====================================================================
        // Multi-worker prefetch
        // ====================================================================
        // Threading model:
        //   * Feeder thread drains the batch_sampler_ (which is NOT thread
        //     safe) on a single thread and pushes batch index lists into
        //     index_queue.
        //   * N worker threads pop index lists, fetch each example via
        //     dataset_.get(idx), collate via collate_fn_, push into batch_queue.
        //   * Main thread pops finished batches from batch_queue.
        //
        // Capacities:
        //   * index_queue : 2 * num_workers (small backlog so workers never starve)
        //   * batch_queue : num_workers * prefetch_factor (PyTorch convention)
        //
        // The dataset is accessed concurrently by N workers. The user is
        // expected to provide a thread-safe `get(size_t)` (the built-in
        // TensorDataset is read-only after construction and therefore safe).
        const size_t num_workers = options_.num_workers;
        const size_t pf = options_.prefetch_factor == 0 ? 1 : options_.prefetch_factor;
        const size_t idx_cap = 2 * num_workers;
        const size_t batch_cap = num_workers * pf;

        auto ctx = std::make_shared<PrefetchContext>(idx_cap, batch_cap);

        DataLoader* self = this;

        // ---- feeder ----
        ctx->feeder = std::thread([self, ctx]() {
            try {
                while (!ctx->stop.load(std::memory_order_acquire)) {
                    auto indices = self->batch_sampler_->next_batch();
                    if (!indices.has_value()) break;
                    if (!ctx->index_queue.push(std::move(indices.value()))) {
                        break;
                    }
                }
            } catch (...) {
                ctx->record_error(std::current_exception());
            }
            ctx->index_queue.close();  // signal workers: no more work
        });

        // ---- workers ----
        // Track in-flight workers so the LAST one to exit can close the
        // batch_queue, signaling end-of-epoch to the main thread.
        auto active = std::make_shared<std::atomic<size_t>>(num_workers);
        ctx->workers.reserve(num_workers);
        for (size_t w = 0; w < num_workers; ++w) {
            ctx->workers.emplace_back([self, ctx, active]() {
                try {
                    while (!ctx->stop.load(std::memory_order_acquire)) {
                        auto maybe_indices = ctx->index_queue.pop();
                        if (!maybe_indices.has_value()) break;
                        const auto& indices = maybe_indices.value();

                        std::vector<ExampleType> examples;
                        examples.reserve(indices.size());
                        for (size_t idx : indices) {
                            examples.push_back(self->dataset_.get(idx));
                        }
                        BatchType batch = self->collate_fn_(examples);

                        if (!ctx->batch_queue.push(std::move(batch))) {
                            break;
                        }
                    }
                } catch (...) {
                    ctx->record_error(std::current_exception());
                }
                if (active->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    ctx->batch_queue.close();
                }
            });
        }

        return Iterator(this, std::move(ctx));
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
