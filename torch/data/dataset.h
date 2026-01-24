#pragma once

#include "aten/src/ATen/ATen.h"
#include <vector>
#include <memory>
#include <optional>
#include <functional>

namespace torch {
namespace data {

using at::Tensor;

// ============================================================================
// Example - A single data sample with optional target
// ============================================================================
// Represents one item from a dataset, typically an input tensor and its label.

template<typename Data = Tensor, typename Target = Tensor>
struct Example {
    Data data;
    Target target;

    Example() = default;

    Example(Data data_, Target target_)
        : data(std::move(data_)), target(std::move(target_)) {}

    // For datasets without targets
    explicit Example(Data data_)
        : data(std::move(data_)), target() {}
};

// Specialization for datasets without targets
template<typename Data>
struct Example<Data, void> {
    Data data;

    Example() = default;
    explicit Example(Data data_) : data(std::move(data_)) {}
};

// ============================================================================
// Dataset - Abstract base class for all datasets
// ============================================================================
// Datasets must implement get() and size() methods.
//
// Usage:
//   class MyDataset : public Dataset<Tensor, Tensor> {
//   public:
//       Example<Tensor, Tensor> get(size_t index) override {
//           return {inputs[index], labels[index]};
//       }
//       size_t size() const override { return inputs.size(); }
//   };

template<typename Data = Tensor, typename Target = Tensor>
class Dataset {
public:
    using ExampleType = Example<Data, Target>;
    using DataType = Data;
    using TargetType = Target;

    virtual ~Dataset() = default;

    // Get a single example at the given index
    virtual ExampleType get(size_t index) = 0;

    // Return the total number of examples
    virtual size_t size() const = 0;

    // Optional: Check if dataset is empty
    bool empty() const { return size() == 0; }
};

// ============================================================================
// TensorDataset - Dataset from tensors
// ============================================================================
// Creates a dataset from input and target tensors, where the first dimension
// represents the number of samples.
//
// Usage:
//   auto inputs = torch::randn({100, 784});  // 100 samples, 784 features
//   auto labels = torch::randint(0, 10, {100});  // 100 labels
//   TensorDataset dataset(inputs, labels);

class TensorDataset : public Dataset<Tensor, Tensor> {
public:
    TensorDataset(Tensor data, Tensor targets)
        : data_(std::move(data)), targets_(std::move(targets)) {
        PT_CHECK_MSG(data_.size(0) == targets_.size(0),
            "Data and targets must have same number of samples");
    }

    // Single tensor dataset (no targets)
    explicit TensorDataset(Tensor data)
        : data_(std::move(data)), targets_() {}

    ExampleType get(size_t index) override {
        PT_CHECK_MSG(index < size(), "Index out of bounds");

        // Select along first dimension
        Tensor x = data_.select(0, static_cast<int64_t>(index));

        if (targets_.defined()) {
            Tensor y = targets_.select(0, static_cast<int64_t>(index));
            return ExampleType(x, y);
        }
        return ExampleType(x, Tensor());
    }

    size_t size() const override {
        return static_cast<size_t>(data_.size(0));
    }

    // Access underlying tensors
    const Tensor& data() const { return data_; }
    const Tensor& targets() const { return targets_; }

private:
    Tensor data_;
    Tensor targets_;
};

// ============================================================================
// MapDataset - Apply transform to another dataset
// ============================================================================
// Wraps another dataset and applies a transform function to each example.
//
// Usage:
//   auto normalized = dataset.map([](Example<Tensor, Tensor> e) {
//       e.data = (e.data - 0.5) / 0.5;  // Normalize to [-1, 1]
//       return e;
//   });

template<typename SourceDataset, typename Transform>
class MapDataset : public Dataset<
    typename std::invoke_result_t<Transform, typename SourceDataset::ExampleType>::DataType,
    typename std::invoke_result_t<Transform, typename SourceDataset::ExampleType>::TargetType
> {
public:
    using InputExample = typename SourceDataset::ExampleType;
    using OutputExample = std::invoke_result_t<Transform, InputExample>;
    using ExampleType = OutputExample;

    MapDataset(SourceDataset source, Transform transform)
        : source_(std::move(source)), transform_(std::move(transform)) {}

    ExampleType get(size_t index) override {
        return transform_(source_.get(index));
    }

    size_t size() const override {
        return source_.size();
    }

private:
    SourceDataset source_;
    Transform transform_;
};

// Helper function to create MapDataset
template<typename SourceDataset, typename Transform>
MapDataset<SourceDataset, Transform> map_dataset(
    SourceDataset source,
    Transform transform
) {
    return MapDataset<SourceDataset, Transform>(
        std::move(source), std::move(transform)
    );
}

// ============================================================================
// ConcatDataset - Concatenate multiple datasets
// ============================================================================
// Concatenates multiple datasets of the same type into one.

template<typename Data = Tensor, typename Target = Tensor>
class ConcatDataset : public Dataset<Data, Target> {
public:
    using ExampleType = Example<Data, Target>;
    using DatasetPtr = std::shared_ptr<Dataset<Data, Target>>;

    ConcatDataset(std::vector<DatasetPtr> datasets)
        : datasets_(std::move(datasets)) {
        cumulative_sizes_.reserve(datasets_.size());
        size_t cumulative = 0;
        for (const auto& ds : datasets_) {
            cumulative += ds->size();
            cumulative_sizes_.push_back(cumulative);
        }
    }

    ExampleType get(size_t index) override {
        PT_CHECK_MSG(index < size(), "Index out of bounds");

        // Find which dataset this index belongs to
        size_t dataset_idx = 0;
        size_t offset = 0;
        for (size_t i = 0; i < cumulative_sizes_.size(); ++i) {
            if (index < cumulative_sizes_[i]) {
                dataset_idx = i;
                offset = (i == 0) ? 0 : cumulative_sizes_[i - 1];
                break;
            }
        }

        return datasets_[dataset_idx]->get(index - offset);
    }

    size_t size() const override {
        return cumulative_sizes_.empty() ? 0 : cumulative_sizes_.back();
    }

private:
    std::vector<DatasetPtr> datasets_;
    std::vector<size_t> cumulative_sizes_;
};

// ============================================================================
// SubsetDataset - Subset of another dataset
// ============================================================================
// Creates a subset of a dataset given a list of indices.

template<typename Data = Tensor, typename Target = Tensor>
class SubsetDataset : public Dataset<Data, Target> {
public:
    using ExampleType = Example<Data, Target>;
    using DatasetPtr = std::shared_ptr<Dataset<Data, Target>>;

    SubsetDataset(DatasetPtr source, std::vector<size_t> indices)
        : source_(std::move(source)), indices_(std::move(indices)) {}

    ExampleType get(size_t index) override {
        PT_CHECK_MSG(index < size(), "Index out of bounds");
        return source_->get(indices_[index]);
    }

    size_t size() const override {
        return indices_.size();
    }

private:
    DatasetPtr source_;
    std::vector<size_t> indices_;
};

// ============================================================================
// Utility functions
// ============================================================================

// Split dataset into train/val subsets
template<typename Data = Tensor, typename Target = Tensor>
std::pair<SubsetDataset<Data, Target>, SubsetDataset<Data, Target>>
random_split(
    std::shared_ptr<Dataset<Data, Target>> dataset,
    double train_ratio,
    bool shuffle = true
) {
    size_t total = dataset->size();
    size_t train_size = static_cast<size_t>(total * train_ratio);

    std::vector<size_t> indices(total);
    for (size_t i = 0; i < total; ++i) {
        indices[i] = i;
    }

    if (shuffle) {
        // Simple Fisher-Yates shuffle
        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i = total - 1; i > 0; --i) {
            std::uniform_int_distribution<size_t> dist(0, i);
            std::swap(indices[i], indices[dist(gen)]);
        }
    }

    std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> val_indices(indices.begin() + train_size, indices.end());

    return {
        SubsetDataset<Data, Target>(dataset, std::move(train_indices)),
        SubsetDataset<Data, Target>(dataset, std::move(val_indices))
    };
}

} // namespace data
} // namespace torch
