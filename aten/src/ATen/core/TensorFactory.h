#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/ScalarType.h"
#include "c10/core/Device.h"
#include <random>
#include <chrono>
#include <cstring>

namespace at {

// ============================================================================
// Random Number Generator
// ============================================================================

class PT_API Generator {
public:
    static Generator& getDefault() {
        static Generator instance;
        return instance;
    }

    Generator() : gen_(static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    )) {}

    explicit Generator(uint64_t seed) : gen_(seed) {}

    void manual_seed(uint64_t seed) {
        gen_.seed(seed);
    }

    uint64_t seed() {
        std::uniform_int_distribution<uint64_t> dist;
        return dist(gen_);
    }

    std::mt19937_64& engine() { return gen_; }

private:
    std::mt19937_64 gen_;
};

// ============================================================================
// Factory Functions Implementation
// ============================================================================

namespace detail {

// Helper to create a tensor with given shape and options
inline Tensor make_tensor(
    c10::IntArrayRef sizes,
    const TensorOptions& options = TensorOptions()
) {
    int64_t numel = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        numel *= sizes[i];
    }

    size_t nbytes = numel * c10::elementSize(options.dtype());
    c10::Storage storage = c10::Storage::create(nbytes, options.device());

    auto impl = std::make_shared<c10::TensorImpl>(
        std::move(storage),
        options.dtype(),
        sizes
    );

    if (options.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// Fill tensor with value
template<typename T>
void fill_tensor(Tensor& tensor, T value) {
    T* data = tensor.mutable_data_ptr<T>();
    int64_t n = tensor.numel();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        data[i] = value;
    }
}

// Fill tensor with uniform random values
template<typename T>
void fill_uniform(Tensor& tensor, T low, T high, Generator& gen) {
    T* data = tensor.mutable_data_ptr<T>();
    int64_t n = tensor.numel();

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(low, high);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen.engine());
        }
    } else {
        std::uniform_int_distribution<T> dist(low, high);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen.engine());
        }
    }
}

// Fill tensor with normal random values
template<typename T>
void fill_normal(Tensor& tensor, T mean, T std, Generator& gen) {
    T* data = tensor.mutable_data_ptr<T>();
    int64_t n = tensor.numel();

    std::normal_distribution<double> dist(
        static_cast<double>(mean),
        static_cast<double>(std)
    );

    for (int64_t i = 0; i < n; ++i) {
        data[i] = static_cast<T>(dist(gen.engine()));
    }
}

} // namespace detail

// ============================================================================
// Empty Tensor
// ============================================================================

inline Tensor empty(
    c10::IntArrayRef sizes,
    const TensorOptions& options = TensorOptions()
) {
    return detail::make_tensor(sizes, options);
}

inline Tensor empty(
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    c10::Device device = c10::kCPU
) {
    return empty(sizes, TensorOptions().dtype(dtype).device(device));
}

inline Tensor empty_like(const Tensor& other) {
    return empty(other.sizes(), TensorOptions().dtype(other.dtype()).device(other.device()));
}

// ============================================================================
// Zeros
// ============================================================================

inline Tensor zeros(
    c10::IntArrayRef sizes,
    const TensorOptions& options = TensorOptions()
) {
    Tensor result = empty(sizes, options);

    // Zero out memory
    std::memset(result.data_ptr(), 0, result.nbytes());

    return result;
}

inline Tensor zeros(
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    c10::Device device = c10::kCPU
) {
    return zeros(sizes, TensorOptions().dtype(dtype).device(device));
}

inline Tensor zeros_like(const Tensor& other) {
    return zeros(other.sizes(), TensorOptions().dtype(other.dtype()).device(other.device()));
}

// ============================================================================
// Ones
// ============================================================================

inline Tensor ones(
    c10::IntArrayRef sizes,
    const TensorOptions& options = TensorOptions()
) {
    Tensor result = empty(sizes, options);

    PT_DISPATCH_ALL_TYPES(options.dtype(), "ones", [&] {
        detail::fill_tensor<scalar_t>(result, static_cast<scalar_t>(1));
    });

    return result;
}

inline Tensor ones(
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    c10::Device device = c10::kCPU
) {
    return ones(sizes, TensorOptions().dtype(dtype).device(device));
}

inline Tensor ones_like(const Tensor& other) {
    return ones(other.sizes(), TensorOptions().dtype(other.dtype()).device(other.device()));
}

// ============================================================================
// Full (filled with specific value)
// ============================================================================

inline Tensor full(
    c10::IntArrayRef sizes,
    Scalar fill_value,
    const TensorOptions& options = TensorOptions()
) {
    Tensor result = empty(sizes, options);

    PT_DISPATCH_ALL_TYPES(options.dtype(), "full", [&] {
        detail::fill_tensor<scalar_t>(result, fill_value.to<scalar_t>());
    });

    return result;
}

inline Tensor full_like(const Tensor& other, Scalar fill_value) {
    return full(other.sizes(), fill_value,
                TensorOptions().dtype(other.dtype()).device(other.device()));
}

// ============================================================================
// Arange
// ============================================================================

inline Tensor arange(
    Scalar start,
    Scalar end,
    Scalar step = 1,
    const TensorOptions& options = TensorOptions()
) {
    double start_d = start.toDouble();
    double end_d = end.toDouble();
    double step_d = step.toDouble();

    PT_CHECK_MSG(step_d != 0, "step cannot be zero");
    PT_CHECK_MSG((step_d > 0 && start_d < end_d) || (step_d < 0 && start_d > end_d),
        "invalid range for arange");

    int64_t size = static_cast<int64_t>(std::ceil((end_d - start_d) / step_d));
    Tensor result = empty({size}, options);

    PT_DISPATCH_ALL_TYPES(options.dtype(), "arange", [&] {
        scalar_t* data = result.mutable_data_ptr<scalar_t>();
        scalar_t val = static_cast<scalar_t>(start_d);
        scalar_t step_val = static_cast<scalar_t>(step_d);

        for (int64_t i = 0; i < size; ++i) {
            data[i] = val;
            val += step_val;
        }
    });

    return result;
}

inline Tensor arange(
    Scalar end,
    const TensorOptions& options = TensorOptions()
) {
    return arange(0, end, 1, options);
}

// ============================================================================
// Linspace
// ============================================================================

inline Tensor linspace(
    Scalar start,
    Scalar end,
    int64_t steps,
    const TensorOptions& options = TensorOptions()
) {
    PT_CHECK_MSG(steps >= 0, "steps must be non-negative");

    Tensor result = empty({steps}, options);

    if (steps == 0) {
        return result;
    }

    if (steps == 1) {
        PT_DISPATCH_FLOATING_TYPES(options.dtype(), "linspace", [&] {
            result.mutable_data_ptr<scalar_t>()[0] = start.to<scalar_t>();
        });
        return result;
    }

    double start_d = start.toDouble();
    double end_d = end.toDouble();
    double step = (end_d - start_d) / (steps - 1);

    PT_DISPATCH_FLOATING_TYPES(options.dtype(), "linspace", [&] {
        scalar_t* data = result.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < steps; ++i) {
            data[i] = static_cast<scalar_t>(start_d + i * step);
        }
        // Ensure exact end value
        data[steps - 1] = static_cast<scalar_t>(end_d);
    });

    return result;
}

// ============================================================================
// Logspace
// ============================================================================

inline Tensor logspace(
    Scalar start,
    Scalar end,
    int64_t steps,
    double base = 10.0,
    const TensorOptions& options = TensorOptions()
) {
    Tensor lin = linspace(start, end, steps, options);

    PT_DISPATCH_FLOATING_TYPES(options.dtype(), "logspace", [&] {
        scalar_t* data = lin.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < steps; ++i) {
            data[i] = static_cast<scalar_t>(std::pow(base, static_cast<double>(data[i])));
        }
    });

    return lin;
}

// ============================================================================
// Eye (Identity matrix)
// ============================================================================

inline Tensor eye(
    int64_t n,
    int64_t m,
    const TensorOptions& options = TensorOptions()
) {
    Tensor result = zeros({n, m}, options);

    PT_DISPATCH_ALL_TYPES(options.dtype(), "eye", [&] {
        scalar_t* data = result.mutable_data_ptr<scalar_t>();
        int64_t min_dim = std::min(n, m);
        for (int64_t i = 0; i < min_dim; ++i) {
            data[i * m + i] = static_cast<scalar_t>(1);
        }
    });

    return result;
}

inline Tensor eye(
    int64_t n,
    const TensorOptions& options = TensorOptions()
) {
    return eye(n, n, options);
}

// ============================================================================
// Random Tensors
// ============================================================================

// Uniform random [0, 1)
inline Tensor rand(
    c10::IntArrayRef sizes,
    const TensorOptions& options = TensorOptions()
) {
    Tensor result = empty(sizes, options);

    PT_DISPATCH_FLOATING_TYPES(options.dtype(), "rand", [&] {
        detail::fill_uniform<scalar_t>(
            result,
            static_cast<scalar_t>(0),
            static_cast<scalar_t>(1),
            Generator::getDefault()
        );
    });

    return result;
}

inline Tensor rand_like(const Tensor& other) {
    return rand(other.sizes(), TensorOptions().dtype(other.dtype()).device(other.device()));
}

// Standard normal distribution
inline Tensor randn(
    c10::IntArrayRef sizes,
    const TensorOptions& options = TensorOptions()
) {
    Tensor result = empty(sizes, options);

    PT_DISPATCH_FLOATING_TYPES(options.dtype(), "randn", [&] {
        detail::fill_normal<scalar_t>(
            result,
            static_cast<scalar_t>(0),
            static_cast<scalar_t>(1),
            Generator::getDefault()
        );
    });

    return result;
}

inline Tensor randn_like(const Tensor& other) {
    return randn(other.sizes(), TensorOptions().dtype(other.dtype()).device(other.device()));
}

// Random integers [low, high)
inline Tensor randint(
    int64_t low,
    int64_t high,
    c10::IntArrayRef sizes,
    const TensorOptions& options = TensorOptions().dtype(c10::ScalarType::Long)
) {
    PT_CHECK_MSG(low < high, "low must be less than high");

    Tensor result = empty(sizes, options);

    std::uniform_int_distribution<int64_t> dist(low, high - 1);
    auto& gen = Generator::getDefault();

    int64_t* data = result.mutable_data_ptr<int64_t>();
    int64_t n = result.numel();

    for (int64_t i = 0; i < n; ++i) {
        data[i] = dist(gen.engine());
    }

    return result;
}

inline Tensor randint(
    int64_t high,
    c10::IntArrayRef sizes,
    const TensorOptions& options = TensorOptions().dtype(c10::ScalarType::Long)
) {
    return randint(0, high, sizes, options);
}

// Random permutation
inline Tensor randperm(
    int64_t n,
    const TensorOptions& options = TensorOptions().dtype(c10::ScalarType::Long)
) {
    Tensor result = arange(0, n, 1, options);

    int64_t* data = result.mutable_data_ptr<int64_t>();
    auto& gen = Generator::getDefault();

    // Fisher-Yates shuffle
    for (int64_t i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<int64_t> dist(0, i);
        int64_t j = dist(gen.engine());
        std::swap(data[i], data[j]);
    }

    return result;
}

// ============================================================================
// From Data
// ============================================================================

template<typename T>
Tensor tensor(
    std::initializer_list<T> data,
    const TensorOptions& options = TensorOptions()
) {
    int64_t size = static_cast<int64_t>(data.size());
    Tensor result = empty({size}, options);

    PT_DISPATCH_ALL_TYPES(options.dtype(), "tensor", [&] {
        scalar_t* ptr = result.mutable_data_ptr<scalar_t>();
        int64_t i = 0;
        for (const auto& val : data) {
            ptr[i++] = static_cast<scalar_t>(val);
        }
    });

    return result;
}

template<typename T>
Tensor tensor(
    const std::vector<T>& data,
    const TensorOptions& options = TensorOptions()
) {
    int64_t size = static_cast<int64_t>(data.size());
    Tensor result = empty({size}, options);

    PT_DISPATCH_ALL_TYPES(options.dtype(), "tensor", [&] {
        scalar_t* ptr = result.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < size; ++i) {
            ptr[i] = static_cast<scalar_t>(data[i]);
        }
    });

    return result;
}

// 2D tensor from nested initializer list
template<typename T>
Tensor tensor(
    std::initializer_list<std::initializer_list<T>> data,
    const TensorOptions& options = TensorOptions()
) {
    int64_t rows = static_cast<int64_t>(data.size());
    int64_t cols = rows > 0 ? static_cast<int64_t>(data.begin()->size()) : 0;

    Tensor result = empty({rows, cols}, options);

    PT_DISPATCH_ALL_TYPES(options.dtype(), "tensor_2d", [&] {
        scalar_t* ptr = result.mutable_data_ptr<scalar_t>();
        int64_t idx = 0;
        for (const auto& row : data) {
            PT_CHECK_MSG(static_cast<int64_t>(row.size()) == cols,
                "All rows must have the same size");
            for (const auto& val : row) {
                ptr[idx++] = static_cast<scalar_t>(val);
            }
        }
    });

    return result;
}

// Scalar tensor
inline Tensor scalar_tensor(
    Scalar value,
    const TensorOptions& options = TensorOptions()
) {
    Tensor result = empty({}, options);

    PT_DISPATCH_ALL_TYPES(options.dtype(), "scalar_tensor", [&] {
        result.mutable_data_ptr<scalar_t>()[0] = value.to<scalar_t>();
    });

    return result;
}

// ============================================================================
// Manual Seed
// ============================================================================

inline void manual_seed(uint64_t seed) {
    Generator::getDefault().manual_seed(seed);
}

// ============================================================================
// Multinomial — sample from probability distribution
// ============================================================================

inline Tensor multinomial(const Tensor& probs, int64_t num_samples, bool replacement = false) {
    PT_CHECK_MSG(probs.dim() == 1 || probs.dim() == 2,
        "multinomial: probs must be 1D or 2D");

    bool is_batched = (probs.dim() == 2);
    int64_t batch = is_batched ? probs.size(0) : 1;
    int64_t n_categories = is_batched ? probs.size(1) : probs.size(0);

    PT_CHECK_MSG(replacement || num_samples <= n_categories,
        "multinomial: cannot sample more than categories without replacement");

    std::vector<int64_t> out_shape;
    if (is_batched) {
        out_shape = {batch, num_samples};
    } else {
        out_shape = {num_samples};
    }

    Tensor result = empty(out_shape, TensorOptions().dtype(c10::ScalarType::Long));
    Tensor p = probs.contiguous();

    PT_DISPATCH_FLOATING_TYPES(probs.dtype(), "multinomial", [&] {
        const scalar_t* p_data = p.data_ptr<scalar_t>();
        int64_t* out_data = result.mutable_data_ptr<int64_t>();
        auto& gen = Generator::getDefault();

        for (int64_t b = 0; b < batch; ++b) {
            const scalar_t* row = p_data + b * n_categories;

            // Build CDF
            std::vector<double> cdf(n_categories);
            cdf[0] = static_cast<double>(row[0]);
            for (int64_t j = 1; j < n_categories; ++j) {
                cdf[j] = cdf[j - 1] + static_cast<double>(row[j]);
            }
            // Normalize
            double total = cdf[n_categories - 1];
            if (total > 0) {
                for (auto& c : cdf) c /= total;
            }

            std::vector<bool> used(n_categories, false);

            for (int64_t s = 0; s < num_samples; ++s) {
                double u = gen.uniform();

                // Binary search in CDF
                int64_t lo = 0, hi = n_categories;
                while (lo < hi) {
                    int64_t mid = lo + (hi - lo) / 2;
                    double cdf_val = cdf[mid];
                    // For no-replacement, skip used entries
                    if (!replacement && used[mid]) {
                        lo = mid + 1;
                        continue;
                    }
                    if (cdf_val < u) lo = mid + 1;
                    else hi = mid;
                }

                if (!replacement) {
                    // Simple approach: sample with replacement then reject
                    int64_t idx = lo < n_categories ? lo : n_categories - 1;
                    while (used[idx] && idx < n_categories - 1) idx++;
                    while (used[idx] && idx > 0) idx--;
                    used[idx] = true;
                    out_data[b * num_samples + s] = idx;
                } else {
                    out_data[b * num_samples + s] = lo < n_categories ? lo : n_categories - 1;
                }
            }
        }
    });

    return result;
}

} // namespace at

// ============================================================================
// Torch namespace aliases
// ============================================================================

namespace torch {
    using at::empty;
    using at::empty_like;
    using at::zeros;
    using at::zeros_like;
    using at::ones;
    using at::ones_like;
    using at::full;
    using at::full_like;
    using at::arange;
    using at::linspace;
    using at::logspace;
    using at::eye;
    using at::rand;
    using at::rand_like;
    using at::randn;
    using at::randn_like;
    using at::randint;
    using at::randperm;
    using at::tensor;
    using at::scalar_tensor;
    using at::manual_seed;
    using at::multinomial;
    using at::Generator;
}
