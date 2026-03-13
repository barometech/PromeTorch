#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

namespace at {
namespace native {

// ============================================================================
// Select - Select a single index along a dimension
// ============================================================================

inline Tensor select(const Tensor& self, int64_t dim, int64_t index) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    int64_t dim_size = self.size(dim);
    if (index < 0) index += dim_size;
    PT_CHECK_MSG(index >= 0 && index < dim_size,
        "select: index ", index, " out of range for dimension ", dim, " with size ", dim_size);

    // Result has one fewer dimension
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;

    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }

    // Compute new storage offset
    int64_t new_offset = self.storage_offset() + index * self.stride(dim);

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        new_offset
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// ============================================================================
// Narrow - Select a range along a dimension
// ============================================================================

inline Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    int64_t dim_size = self.size(dim);
    if (start < 0) start += dim_size;
    PT_CHECK_MSG(start >= 0 && start < dim_size,
        "narrow: start ", start, " out of range");
    PT_CHECK_MSG(length >= 0 && start + length <= dim_size,
        "narrow: length ", length, " out of range");

    // Same dimensions, but size along dim is reduced
    std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
    new_sizes[dim] = length;

    // Strides remain the same
    std::vector<int64_t> new_strides(self.strides().begin(), self.strides().end());

    // Compute new storage offset
    int64_t new_offset = self.storage_offset() + start * self.stride(dim);

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        new_offset
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// ============================================================================
// Slice - General slicing with step
// ============================================================================

inline Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step = 1) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(step > 0, "slice step must be positive");

    int64_t dim_size = self.size(dim);

    // Handle negative indices and clamp
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;

    start = std::max(int64_t(0), std::min(start, dim_size));
    end = std::max(int64_t(0), std::min(end, dim_size));

    if (start >= end) {
        // Empty tensor
        std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
        new_sizes[dim] = 0;
        return empty(new_sizes, TensorOptions().dtype(self.dtype()).device(self.device()));
    }

    int64_t new_dim_size = (end - start + step - 1) / step;

    std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
    new_sizes[dim] = new_dim_size;

    std::vector<int64_t> new_strides(self.strides().begin(), self.strides().end());
    new_strides[dim] = self.stride(dim) * step;

    int64_t new_offset = self.storage_offset() + start * self.stride(dim);

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        new_offset
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// ============================================================================
// Index operator [] - Select along first dimension
// ============================================================================

inline Tensor index(const Tensor& self, int64_t index) {
    return select(self, 0, index);
}

// ============================================================================
// Index Select - Select specific indices along a dimension
// ============================================================================

inline Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dim() == 1, "index_select: index must be 1D");
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
        "index_select: index must be LongTensor");

    int64_t num_indices = index.size(0);

    // Result shape: same as self but with dim size replaced by num_indices
    std::vector<int64_t> result_shape(self.sizes().begin(), self.sizes().end());
    result_shape[dim] = num_indices;

    Tensor result = empty(result_shape, TensorOptions().dtype(self.dtype()).device(self.device()));

    const int64_t* idx_data = index.data_ptr<int64_t>();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "index_select", [&] {
        const scalar_t* src = self.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        // Compute sizes for iteration
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= self.size(i);
        }

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            inner_size *= self.size(i);
        }

        int64_t src_dim_stride = self.stride(dim);
        int64_t dst_dim_stride = result.stride(dim);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t idx = 0; idx < num_indices; ++idx) {
                int64_t src_idx = idx_data[idx];
                PT_CHECK_MSG(src_idx >= 0 && src_idx < self.size(dim),
                    "index_select: index out of bounds");

                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    int64_t src_offset = outer * self.stride(0) + src_idx * src_dim_stride + inner;
                    int64_t dst_offset = outer * result.stride(0) + idx * dst_dim_stride + inner;

                    // Simplified - assumes contiguous in inner dimensions
                    dst[dst_offset] = src[src_offset];
                }
            }
        }
    });

    return result;
}

// ============================================================================
// Masked Select - Select elements where mask is true
// ============================================================================

inline Tensor masked_select(const Tensor& self, const Tensor& mask) {
    PT_CHECK_MSG(mask.dtype() == c10::ScalarType::Bool,
        "masked_select: mask must be BoolTensor");

    // First count true elements
    int64_t count = 0;
    const bool* mask_data = mask.data_ptr<bool>();
    int64_t n = mask.numel();

    for (int64_t i = 0; i < n; ++i) {
        if (mask_data[i]) ++count;
    }

    Tensor result = empty({count}, TensorOptions().dtype(self.dtype()).device(self.device()));

    if (count == 0) {
        return result;
    }

    PT_DISPATCH_ALL_TYPES(self.dtype(), "masked_select", [&] {
        const scalar_t* src = self.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        int64_t dst_idx = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (mask_data[i]) {
                dst[dst_idx++] = src[i];
            }
        }
    });

    return result;
}

// ============================================================================
// Masked Fill - Fill elements where mask is true
// ============================================================================

inline Tensor& masked_fill_(Tensor& self, const Tensor& mask, Scalar value) {
    PT_CHECK_MSG(mask.dtype() == c10::ScalarType::Bool,
        "masked_fill_: mask must be BoolTensor");
    PT_CHECK_MSG(self.sizes() == mask.sizes(),
        "masked_fill_: mask and tensor must have same shape");

    const bool* mask_data = mask.data_ptr<bool>();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "masked_fill_", [&] {
        scalar_t* data = self.mutable_data_ptr<scalar_t>();
        scalar_t fill_val = value.to<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            if (mask_data[i]) {
                data[i] = fill_val;
            }
        }
    });

    return self;
}

inline Tensor masked_fill(const Tensor& self, const Tensor& mask, Scalar value) {
    Tensor result = self.clone();
    masked_fill_(result, mask, value);
    return result;
}

// ============================================================================
// Where - Conditional selection
// ============================================================================

inline Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(condition.dtype() == c10::ScalarType::Bool,
        "where: condition must be BoolTensor");

    // Broadcasting
    auto result_shape = detail::broadcast_shapes(
        detail::broadcast_shapes(condition.sizes(), self.sizes()),
        other.sizes()
    );

    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());
    Tensor result = empty(result_shape, TensorOptions().dtype(result_dtype).device(self.device()));

    const bool* cond_data = condition.data_ptr<bool>();

    PT_DISPATCH_ALL_TYPES(result_dtype, "where", [&] {
        const scalar_t* self_data = self.data_ptr<scalar_t>();
        const scalar_t* other_data = other.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();

        int64_t n = result.numel();

        // Simple case: all same shape
        if (condition.sizes() == self.sizes() && self.sizes() == other.sizes()) {
            for (int64_t i = 0; i < n; ++i) {
                out[i] = cond_data[i] ? self_data[i] : other_data[i];
            }
        } else {
            // Broadcasting case
            for (int64_t i = 0; i < n; ++i) {
                int64_t cond_idx = detail::broadcast_index(i, result.sizes(), condition.sizes(), condition.strides());
                int64_t self_idx = detail::broadcast_index(i, result.sizes(), self.sizes(), self.strides());
                int64_t other_idx = detail::broadcast_index(i, result.sizes(), other.sizes(), other.strides());

                out[i] = cond_data[cond_idx] ? self_data[self_idx] : other_data[other_idx];
            }
        }
    });

    return result;
}

// ============================================================================
// Nonzero - Return indices of nonzero elements
// ============================================================================

inline Tensor nonzero(const Tensor& self) {
    // First count nonzero elements
    int64_t count = 0;

    PT_DISPATCH_ALL_TYPES(self.dtype(), "nonzero_count", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            if (data[i] != static_cast<scalar_t>(0)) {
                ++count;
            }
        }
    });

    int64_t ndim = self.dim();
    Tensor result = empty({count, ndim}, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));

    if (count == 0) {
        return result;
    }

    int64_t* out = result.mutable_data_ptr<int64_t>();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "nonzero", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        int64_t n = self.numel();
        int64_t out_idx = 0;

        for (int64_t i = 0; i < n; ++i) {
            if (data[i] != static_cast<scalar_t>(0)) {
                // Convert linear index to multi-dimensional
                int64_t remaining = i;
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    out[out_idx * ndim + d] = remaining % self.size(d);
                    remaining /= self.size(d);
                }
                ++out_idx;
            }
        }
    });

    return result;
}

// ============================================================================
// Gather - Gather values along an axis
// ============================================================================

inline Tensor gather(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
        "gather: index must be LongTensor");
    PT_CHECK_MSG(index.dim() == ndim,
        "gather: index must have same number of dimensions as input");

    Tensor result = empty(index.sizes(), TensorOptions().dtype(self.dtype()).device(self.device()));

    const int64_t* idx_data = index.data_ptr<int64_t>();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "gather", [&] {
        const scalar_t* src = self.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        int64_t n = index.numel();

        for (int64_t i = 0; i < n; ++i) {
            // Convert linear index to multi-dimensional for index tensor
            int64_t remaining = i;
            std::vector<int64_t> idx_coords(ndim);

            for (int64_t d = ndim - 1; d >= 0; --d) {
                idx_coords[d] = remaining % index.size(d);
                remaining /= index.size(d);
            }

            // Get the index value for gather dimension
            int64_t gather_idx = idx_data[i];
            PT_CHECK_MSG(gather_idx >= 0 && gather_idx < self.size(dim),
                "gather: index out of bounds");

            // Compute source index
            int64_t src_idx = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = (d == dim) ? gather_idx : idx_coords[d];
                src_idx += coord * self.stride(d);
            }

            dst[i] = src[src_idx];
        }
    });

    return result;
}

// ============================================================================
// Scatter - Scatter values along an axis
// ============================================================================

inline Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
        "scatter_: index must be LongTensor");

    const int64_t* idx_data = index.data_ptr<int64_t>();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "scatter_", [&] {
        scalar_t* dst = self.mutable_data_ptr<scalar_t>();
        const scalar_t* src_data = src.data_ptr<scalar_t>();

        int64_t n = index.numel();

        for (int64_t i = 0; i < n; ++i) {
            int64_t remaining = i;
            std::vector<int64_t> idx_coords(ndim);

            for (int64_t d = ndim - 1; d >= 0; --d) {
                idx_coords[d] = remaining % index.size(d);
                remaining /= index.size(d);
            }

            int64_t scatter_idx = idx_data[i];
            PT_CHECK_MSG(scatter_idx >= 0 && scatter_idx < self.size(dim),
                "scatter_: index out of bounds");

            int64_t dst_idx = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = (d == dim) ? scatter_idx : idx_coords[d];
                dst_idx += coord * self.stride(d);
            }

            dst[dst_idx] = src_data[i];
        }
    });

    return self;
}

inline Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    Tensor result = self.clone();
    scatter_(result, dim, index, src);
    return result;
}

// ============================================================================
// Scatter Add - Scatter values with addition (needed for index backward)
// ============================================================================

inline Tensor& scatter_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
        "scatter_add_: index must be LongTensor");

    const int64_t* idx_data = index.data_ptr<int64_t>();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "scatter_add_", [&] {
        scalar_t* dst = self.mutable_data_ptr<scalar_t>();
        const scalar_t* src_data = src.data_ptr<scalar_t>();

        int64_t n = index.numel();

        for (int64_t i = 0; i < n; ++i) {
            int64_t remaining = i;
            std::vector<int64_t> idx_coords(ndim);

            for (int64_t d = ndim - 1; d >= 0; --d) {
                idx_coords[d] = remaining % index.size(d);
                remaining /= index.size(d);
            }

            int64_t scatter_idx = idx_data[i];
            PT_CHECK_MSG(scatter_idx >= 0 && scatter_idx < self.size(dim),
                "scatter_add_: index out of bounds");

            int64_t dst_idx = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = (d == dim) ? scatter_idx : idx_coords[d];
                dst_idx += coord * self.stride(d);
            }

            dst[dst_idx] += src_data[i];
        }
    });

    return self;
}

// ============================================================================
// Index with Tensor (fancy indexing) — self[LongTensor] along dim 0
// ============================================================================

inline Tensor index_with_tensor(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
        "index_with_tensor: index must be LongTensor");

    // index can be any shape — output replaces dim with index.sizes()
    auto idx_sizes = index.sizes().vec();
    int64_t idx_numel = index.numel();
    const int64_t* idx_data = index.data_ptr<int64_t>();

    // Build result shape: self.sizes[:dim] + index.sizes + self.sizes[dim+1:]
    std::vector<int64_t> result_sizes;
    for (int64_t d = 0; d < dim; ++d) result_sizes.push_back(self.size(d));
    for (auto s : idx_sizes) result_sizes.push_back(s);
    for (int64_t d = dim + 1; d < ndim; ++d) result_sizes.push_back(self.size(d));

    Tensor result = empty(result_sizes, TensorOptions().dtype(self.dtype()).device(self.device()));

    // Inner size = product of dims after dim
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= self.size(d);

    // Outer size = product of dims before dim
    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= self.size(d);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "index_with_tensor", [&] {
        Tensor self_c = self.contiguous();
        const scalar_t* src = self_c.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        int64_t src_dim_size = self.size(dim);
        int64_t src_dim_stride = inner_size; // contiguous stride along dim

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t ii = 0; ii < idx_numel; ++ii) {
                int64_t src_idx = idx_data[ii];
                if (src_idx < 0) src_idx += src_dim_size;
                PT_CHECK_MSG(src_idx >= 0 && src_idx < src_dim_size,
                    "index_with_tensor: index out of bounds");

                const scalar_t* src_slice = src + outer * src_dim_size * inner_size + src_idx * inner_size;
                scalar_t* dst_slice = dst + outer * idx_numel * inner_size + ii * inner_size;

                std::memcpy(dst_slice, src_slice, inner_size * sizeof(scalar_t));
            }
        }
    });

    return result;
}

// ============================================================================
// Index Put — self[LongTensor] = values along dim 0
// ============================================================================

inline Tensor& index_put_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& values) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
        "index_put_: index must be LongTensor");

    int64_t idx_numel = index.numel();
    const int64_t* idx_data = index.data_ptr<int64_t>();

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= self.size(d);

    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= self.size(d);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "index_put_", [&] {
        scalar_t* dst = self.mutable_data_ptr<scalar_t>();
        Tensor vals_c = values.contiguous();
        const scalar_t* src = vals_c.data_ptr<scalar_t>();

        int64_t src_dim_size = self.size(dim);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t ii = 0; ii < idx_numel; ++ii) {
                int64_t dst_idx = idx_data[ii];
                if (dst_idx < 0) dst_idx += src_dim_size;
                PT_CHECK_MSG(dst_idx >= 0 && dst_idx < src_dim_size,
                    "index_put_: index out of bounds");

                scalar_t* dst_slice = dst + outer * src_dim_size * inner_size + dst_idx * inner_size;
                const scalar_t* src_slice = src + outer * idx_numel * inner_size + ii * inner_size;

                std::memcpy(dst_slice, src_slice, inner_size * sizeof(scalar_t));
            }
        }
    });

    return self;
}

// ============================================================================
// Boolean Index — self[BoolTensor] -> 1D result
// ============================================================================

inline Tensor boolean_index(const Tensor& self, const Tensor& mask) {
    PT_CHECK_MSG(mask.dtype() == c10::ScalarType::Bool,
        "boolean_index: mask must be BoolTensor");
    // Flatten self and mask if needed, then masked_select
    return masked_select(self.contiguous(), mask.contiguous());
}

// ============================================================================
// Boolean Index Put — self[BoolTensor] = values
// ============================================================================

inline Tensor& boolean_index_put_(Tensor& self, const Tensor& mask, const Tensor& values) {
    PT_CHECK_MSG(mask.dtype() == c10::ScalarType::Bool,
        "boolean_index_put_: mask must be BoolTensor");
    PT_CHECK_MSG(self.sizes() == mask.sizes(),
        "boolean_index_put_: mask and tensor must have same shape");

    const bool* mask_data = mask.data_ptr<bool>();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "boolean_index_put_", [&] {
        scalar_t* dst = self.mutable_data_ptr<scalar_t>();
        const scalar_t* src = values.data_ptr<scalar_t>();
        int64_t n = self.numel();
        int64_t src_idx = 0;

        for (int64_t i = 0; i < n; ++i) {
            if (mask_data[i]) {
                dst[i] = src[src_idx++];
            }
        }
    });

    return self;
}

// ============================================================================
// Scatter Reduce — scatter with reduction (sum, mean, amax, amin, prod)
// ============================================================================

inline Tensor& scatter_reduce_(Tensor& self, int64_t dim, const Tensor& index,
                                const Tensor& src, const std::string& reduce) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
        "scatter_reduce_: index must be LongTensor");

    const int64_t* idx_data = index.data_ptr<int64_t>();

    // For "mean" we need to track counts
    Tensor counts;
    if (reduce == "mean") {
        counts = at::zeros(self.sizes(), TensorOptions().dtype(c10::ScalarType::Long));
    }

    PT_DISPATCH_ALL_TYPES(self.dtype(), "scatter_reduce_", [&] {
        scalar_t* dst = self.mutable_data_ptr<scalar_t>();
        const scalar_t* src_data = src.data_ptr<scalar_t>();
        int64_t* cnt_data = (reduce == "mean") ? counts.mutable_data_ptr<int64_t>() : nullptr;

        int64_t n = index.numel();

        for (int64_t i = 0; i < n; ++i) {
            int64_t remaining = i;
            std::vector<int64_t> idx_coords(ndim);

            for (int64_t d = ndim - 1; d >= 0; --d) {
                idx_coords[d] = remaining % index.size(d);
                remaining /= index.size(d);
            }

            int64_t scatter_idx = idx_data[i];
            PT_CHECK_MSG(scatter_idx >= 0 && scatter_idx < self.size(dim),
                "scatter_reduce_: index out of bounds");

            int64_t dst_idx = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = (d == dim) ? scatter_idx : idx_coords[d];
                dst_idx += coord * self.stride(d);
            }

            scalar_t val = src_data[i];

            if (reduce == "sum" || reduce == "add") {
                dst[dst_idx] += val;
            } else if (reduce == "prod") {
                dst[dst_idx] *= val;
            } else if (reduce == "amax") {
                dst[dst_idx] = std::max(dst[dst_idx], val);
            } else if (reduce == "amin") {
                dst[dst_idx] = std::min(dst[dst_idx], val);
            } else if (reduce == "mean") {
                dst[dst_idx] += val;
                cnt_data[dst_idx] += 1;
            }
        }

        // For mean, divide by counts
        if (reduce == "mean") {
            int64_t total = self.numel();
            for (int64_t j = 0; j < total; ++j) {
                if (cnt_data[j] > 0) {
                    dst[j] /= static_cast<scalar_t>(cnt_data[j]);
                }
            }
        }
    });

    return self;
}

// ============================================================================
// Search Sorted — binary search on sorted sequence
// ============================================================================

inline Tensor searchsorted(const Tensor& sorted_sequence, const Tensor& values, bool right = false) {
    PT_CHECK_MSG(sorted_sequence.dim() == 1, "searchsorted: sorted_sequence must be 1D");

    Tensor result = at::empty(values.sizes(), TensorOptions().dtype(c10::ScalarType::Long));

    Tensor seq = sorted_sequence.contiguous();
    Tensor vals = values.contiguous();

    PT_DISPATCH_FLOATING_TYPES(sorted_sequence.dtype(), "searchsorted", [&] {
        const scalar_t* seq_data = seq.data_ptr<scalar_t>();
        const scalar_t* val_data = vals.data_ptr<scalar_t>();
        int64_t* out_data = result.mutable_data_ptr<int64_t>();
        int64_t seq_len = sorted_sequence.size(0);
        int64_t n = values.numel();

        for (int64_t i = 0; i < n; ++i) {
            scalar_t v = val_data[i];
            int64_t lo = 0, hi = seq_len;

            while (lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                if (right ? (seq_data[mid] <= v) : (seq_data[mid] < v)) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            out_data[i] = lo;
        }
    });

    return result;
}

} // namespace native
} // namespace at
