#pragma once
#include "aten/src/ATen/ATen.h"
#include <optional>

namespace torch {
namespace data {

template<typename Data = at::Tensor, typename Target = at::Tensor>
class IterableDataset {
public:
    using ExampleType = std::pair<Data, Target>;

    virtual ~IterableDataset() = default;

    // Return next example, or nullopt if exhausted
    virtual std::optional<ExampleType> next() = 0;

    // Reset to beginning
    virtual void reset() = 0;

    // Optional: estimated size (-1 if unknown)
    virtual int64_t size_hint() const { return -1; }
};

} // namespace data
} // namespace torch
