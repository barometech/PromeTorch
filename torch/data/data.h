#pragma once

// ============================================================================
// PromeTorch Data Module
// ============================================================================
// This is the main header for the torch::data namespace.
// It provides utilities for loading and iterating over datasets.
//
// Components:
//   - Dataset: Abstract base class for datasets
//   - TensorDataset: Dataset from tensors
//   - MapDataset: Apply transforms to datasets
//   - Sampler: Control iteration order
//   - DataLoader: Iterate over batches
//
// Usage:
//   #include "torch/data/data.h"
//
//   // Create dataset
//   auto inputs = torch::randn({1000, 784});
//   auto labels = torch::randint(0, 10, {1000});
//   TensorDataset dataset(inputs, labels);
//
//   // Create data loader
//   auto loader = make_data_loader(dataset, DataLoaderOptions()
//       .batch_size_(32)
//       .shuffle_(true));
//
//   // Training loop
//   for (auto& batch : loader) {
//       auto output = model.forward(batch.data);
//       auto loss = criterion(output, batch.target);
//       loss.backward();
//       optimizer.step();
//       optimizer.zero_grad();
//   }
// ============================================================================

#include "torch/data/dataset.h"
#include "torch/data/sampler.h"
#include "torch/data/dataloader.h"
#include "torch/data/transforms.h"

namespace torch {
namespace data {

// ============================================================================
// Convenience type aliases
// ============================================================================

using TensorExample = Example<Tensor, Tensor>;
using TensorBatch = Batch<Tensor, Tensor>;

// ============================================================================
// Common dataset transforms
// ============================================================================

// Normalize transform: (x - mean) / std
struct Normalize {
    double mean;
    double std;

    Normalize(double mean_ = 0.0, double std_ = 1.0)
        : mean(mean_), std(std_) {}

    Example<Tensor, Tensor> operator()(Example<Tensor, Tensor> example) const {
        example.data = (example.data - mean) / std;
        return example;
    }
};

// Flatten transform: flatten tensor to 1D
struct Flatten {
    Example<Tensor, Tensor> operator()(Example<Tensor, Tensor> example) const {
        example.data = example.data.view({-1});
        return example;
    }
};

// Lambda transform: apply custom function
template<typename Func>
struct Lambda {
    Func func;

    explicit Lambda(Func f) : func(std::move(f)) {}

    Example<Tensor, Tensor> operator()(Example<Tensor, Tensor> example) const {
        return func(std::move(example));
    }
};

template<typename Func>
Lambda<Func> make_lambda(Func f) {
    return Lambda<Func>(std::move(f));
}

// ============================================================================
// Compose multiple transforms
// ============================================================================

template<typename... Transforms>
class Compose {
public:
    explicit Compose(Transforms... transforms)
        : transforms_(std::make_tuple(std::move(transforms)...)) {}

    Example<Tensor, Tensor> operator()(Example<Tensor, Tensor> example) const {
        return apply_impl(std::move(example), std::index_sequence_for<Transforms...>{});
    }

private:
    template<size_t... Is>
    Example<Tensor, Tensor> apply_impl(
        Example<Tensor, Tensor> example,
        std::index_sequence<Is...>
    ) const {
        // Apply transforms in order
        ((example = std::get<Is>(transforms_)(std::move(example))), ...);
        return example;
    }

    std::tuple<Transforms...> transforms_;
};

template<typename... Transforms>
Compose<Transforms...> compose(Transforms... transforms) {
    return Compose<Transforms...>(std::move(transforms)...);
}

} // namespace data
} // namespace torch
