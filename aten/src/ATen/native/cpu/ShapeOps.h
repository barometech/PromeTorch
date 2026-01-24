#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <numeric>

namespace at {
namespace native {

// ============================================================================
// View - Returns a tensor with a new shape (shares storage)
// ============================================================================

inline Tensor view(const Tensor& self, c10::IntArrayRef sizes) {
    PT_CHECK_MSG(self.is_contiguous(),
        "view is only supported for contiguous tensors");

    // Compute numel of new shape, handling -1
    int64_t new_numel = 1;
    int64_t infer_dim = -1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == -1) {
            PT_CHECK_MSG(infer_dim == -1, "only one dimension can be inferred");
            infer_dim = static_cast<int64_t>(i);
        } else {
            PT_CHECK_MSG(sizes[i] >= 0, "invalid shape dimension");
            new_numel *= sizes[i];
        }
    }

    std::vector<int64_t> new_sizes(sizes.begin(), sizes.end());
    if (infer_dim >= 0) {
        PT_CHECK_MSG(new_numel > 0 && self.numel() % new_numel == 0,
            "cannot infer dimension");
        new_sizes[infer_dim] = self.numel() / new_numel;
    } else {
        PT_CHECK_MSG(new_numel == self.numel(),
            "view size is not compatible with tensor size");
    }

    // Create new TensorImpl sharing storage
    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes
    );
    impl->set_storage_offset(self.storage_offset());

    // IMPORTANT: Properly propagate autograd metadata for views
    // Views should maintain the autograd properties of the source tensor
    auto* src_meta = self.autograd_meta();
    if (src_meta && src_meta->requires_grad_) {
        impl->set_requires_grad(true);
        auto* new_meta = impl->autograd_meta();
        if (new_meta) {
            // Copy is_leaf status - views of leaves are also leaves
            // Views of non-leaves should also be non-leaves
            new_meta->is_leaf_ = src_meta->is_leaf_;
            new_meta->output_nr_ = src_meta->output_nr_;
        }
    }

    return Tensor(std::move(impl));
}

// ============================================================================
// Reshape - Like view but can handle non-contiguous tensors
// ============================================================================

inline Tensor reshape(const Tensor& self, c10::IntArrayRef sizes) {
    // If contiguous, use view
    if (self.is_contiguous()) {
        return view(self, sizes);
    }

    // Otherwise, make contiguous copy and view
    return self.contiguous().view(sizes);
}

// ============================================================================
// Flatten
// ============================================================================

inline Tensor flatten(const Tensor& self, int64_t start_dim = 0, int64_t end_dim = -1) {
    int64_t ndim = self.dim();

    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    PT_CHECK(start_dim >= 0 && start_dim < ndim);
    PT_CHECK(end_dim >= 0 && end_dim < ndim);
    PT_CHECK(start_dim <= end_dim);

    if (start_dim == end_dim) {
        return self;
    }

    // Compute new shape
    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < start_dim; ++i) {
        new_shape.push_back(self.size(i));
    }

    int64_t flat_size = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i) {
        flat_size *= self.size(i);
    }
    new_shape.push_back(flat_size);

    for (int64_t i = end_dim + 1; i < ndim; ++i) {
        new_shape.push_back(self.size(i));
    }

    return reshape(self, new_shape);
}

// ============================================================================
// Squeeze - Remove dimensions of size 1
// ============================================================================

inline Tensor squeeze(const Tensor& self) {
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;

    for (int64_t i = 0; i < self.dim(); ++i) {
        if (self.size(i) != 1) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }

    if (new_sizes.empty()) {
        // Scalar tensor
        new_sizes = {};
        new_strides = {};
    }

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        self.storage_offset()
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

inline Tensor squeeze(const Tensor& self, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    if (self.size(dim) != 1) {
        return self;  // No change if dimension is not 1
    }

    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;

    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        self.storage_offset()
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// ============================================================================
// Unsqueeze - Add a dimension of size 1
// ============================================================================

inline Tensor unsqueeze(const Tensor& self, int64_t dim) {
    int64_t ndim = self.dim();
    // dim can be in range [0, ndim]
    if (dim < 0) dim += ndim + 1;
    PT_CHECK(dim >= 0 && dim <= ndim);

    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;

    for (int64_t i = 0; i < ndim + 1; ++i) {
        if (i == dim) {
            new_sizes.push_back(1);
            // Stride for new dimension
            if (i < ndim) {
                new_strides.push_back(self.stride(i) * self.size(i));
            } else {
                new_strides.push_back(1);
            }
        }
        if (i < ndim) {
            if (i >= dim) {
                new_sizes.push_back(self.size(i));
                new_strides.push_back(self.stride(i));
            } else {
                new_sizes.push_back(self.size(i));
                new_strides.push_back(self.stride(i));
            }
        }
    }

    // Fix: rebuild properly
    new_sizes.clear();
    new_strides.clear();

    for (int64_t i = 0; i <= ndim; ++i) {
        if (i == dim) {
            new_sizes.push_back(1);
            // Calculate stride: product of sizes after this dim
            int64_t stride = 1;
            for (int64_t j = i; j < ndim; ++j) {
                stride *= self.size(j);
            }
            // For unsqueezed dim at the end, stride is 1
            if (i == ndim) stride = 1;
            else if (i < ndim) stride = self.stride(i) * self.size(i);

            new_strides.push_back(stride > 0 ? stride : 1);
        }
        if (i < ndim) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        self.storage_offset()
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// ============================================================================
// Transpose - Swap two dimensions
// ============================================================================

inline Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
    int64_t ndim = self.dim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    PT_CHECK(dim0 >= 0 && dim0 < ndim);
    PT_CHECK(dim1 >= 0 && dim1 < ndim);

    if (dim0 == dim1) {
        return self;
    }

    std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
    std::vector<int64_t> new_strides(self.strides().begin(), self.strides().end());

    std::swap(new_sizes[dim0], new_sizes[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        self.storage_offset()
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// 2D transpose (matrix transpose)
inline Tensor t(const Tensor& self) {
    PT_CHECK_MSG(self.dim() <= 2,
        "t() expects a tensor with <= 2 dimensions, but self is ", self.dim(), "D");

    if (self.dim() < 2) {
        return self;
    }

    return transpose(self, 0, 1);
}

// ============================================================================
// Permute - Reorder dimensions
// ============================================================================

inline Tensor permute(const Tensor& self, c10::IntArrayRef dims) {
    int64_t ndim = self.dim();
    PT_CHECK_MSG(static_cast<int64_t>(dims.size()) == ndim,
        "permute: number of dims doesn't match");

    // Check that dims is a valid permutation
    std::vector<bool> seen(ndim, false);
    for (int64_t d : dims) {
        int64_t dim = d < 0 ? d + ndim : d;
        PT_CHECK(dim >= 0 && dim < ndim);
        PT_CHECK_MSG(!seen[dim], "permute: repeated dim");
        seen[dim] = true;
    }

    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i) {
        int64_t dim = dims[i] < 0 ? dims[i] + ndim : dims[i];
        new_sizes[i] = self.size(dim);
        new_strides[i] = self.stride(dim);
    }

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        self.storage_offset()
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// ============================================================================
// Expand - Broadcast to larger shape (no copy)
// ============================================================================

inline Tensor expand(const Tensor& self, c10::IntArrayRef sizes) {
    int64_t ndim_self = self.dim();
    int64_t ndim_new = static_cast<int64_t>(sizes.size());

    PT_CHECK_MSG(ndim_new >= ndim_self,
        "expand: number of dimensions cannot decrease");

    std::vector<int64_t> new_sizes(ndim_new);
    std::vector<int64_t> new_strides(ndim_new);

    int64_t offset = ndim_new - ndim_self;

    for (int64_t i = 0; i < ndim_new; ++i) {
        int64_t self_dim = i - offset;

        if (self_dim < 0) {
            // New dimension
            PT_CHECK_MSG(sizes[i] >= 0, "expand: invalid size");
            new_sizes[i] = sizes[i];
            new_strides[i] = 0;  // Broadcasting stride
        } else {
            int64_t self_size = self.size(self_dim);
            int64_t new_size = sizes[i];

            if (new_size == -1) {
                new_size = self_size;
            }

            if (self_size == 1) {
                // Broadcast this dimension
                new_sizes[i] = new_size;
                new_strides[i] = 0;
            } else {
                PT_CHECK_MSG(self_size == new_size,
                    "expand: size mismatch at dimension ", i);
                new_sizes[i] = self_size;
                new_strides[i] = self.stride(self_dim);
            }
        }
    }

    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        new_sizes,
        new_strides,
        self.storage_offset()
    );

    if (self.requires_grad()) {
        impl->set_requires_grad(true);
    }

    return Tensor(std::move(impl));
}

// ============================================================================
// Repeat - Tile tensor (creates copy)
// ============================================================================

inline Tensor repeat(const Tensor& self, c10::IntArrayRef repeats) {
    int64_t ndim = self.dim();
    int64_t ndim_repeats = static_cast<int64_t>(repeats.size());

    PT_CHECK_MSG(ndim_repeats >= ndim,
        "repeat: number of dimensions cannot decrease");

    // Compute result shape
    std::vector<int64_t> result_shape(ndim_repeats);
    int64_t offset = ndim_repeats - ndim;

    for (int64_t i = 0; i < ndim_repeats; ++i) {
        int64_t self_dim = i - offset;
        int64_t self_size = (self_dim >= 0) ? self.size(self_dim) : 1;
        result_shape[i] = self_size * repeats[i];
    }

    Tensor result = empty(result_shape, TensorOptions().dtype(self.dtype()).device(self.device()));

    // Copy data with tiling
    PT_DISPATCH_ALL_TYPES(self.dtype(), "repeat", [&] {
        const scalar_t* src = self.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        // Simple implementation: iterate over result and compute source index
        int64_t total = result.numel();
        for (int64_t i = 0; i < total; ++i) {
            // Convert linear index to multi-dimensional
            int64_t remaining = i;
            int64_t src_idx = 0;
            int64_t src_stride = 1;

            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t result_dim = d + offset;
                int64_t idx_in_dim = remaining % result_shape[result_dim];
                remaining /= result_shape[result_dim];

                int64_t src_idx_in_dim = idx_in_dim % self.size(d);
                src_idx += src_idx_in_dim * self.stride(d);
            }

            dst[i] = src[src_idx];
        }
    });

    return result;
}

// ============================================================================
// Contiguous - Return contiguous tensor (copy if needed)
// ============================================================================

inline Tensor contiguous(const Tensor& self) {
    if (self.is_contiguous()) {
        return self;
    }

    // For CUDA tensors, use CPU fallback (until CUDA strided copy kernel)
    bool is_cuda = self.is_cuda();
    Tensor self_cpu = self;
#ifdef PT_USE_CUDA
    if (is_cuda) {
        // Move to CPU for strided copy operation
        self_cpu = at::to_cpu(self);
    }
#endif

    Tensor result = empty(self_cpu.sizes(), TensorOptions().dtype(self_cpu.dtype()));

    // Copy data in contiguous order (CPU computation)
    PT_DISPATCH_ALL_TYPES(self_cpu.dtype(), "contiguous", [&] {
        const scalar_t* src = self_cpu.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        int64_t ndim = self_cpu.dim();
        int64_t total = self_cpu.numel();

        // Use strided copy
        for (int64_t i = 0; i < total; ++i) {
            // Convert contiguous index to strided index
            int64_t remaining = i;
            int64_t src_idx = 0;

            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t idx_in_dim = remaining % self_cpu.size(d);
                remaining /= self_cpu.size(d);
                src_idx += idx_in_dim * self_cpu.stride(d);
            }

            dst[i] = src[src_idx];
        }
    });

    // Move result back to GPU if original was on GPU
#ifdef PT_USE_CUDA
    if (is_cuda) {
        result = at::to_cuda(result);
    }
#endif

    if (self.requires_grad()) {
        result.set_requires_grad(true);
    }

    return result;
}

// ============================================================================
// Clone - Deep copy
// ============================================================================

inline Tensor clone(const Tensor& self) {
    // For CUDA tensors, use CUDA copy
#ifdef PT_USE_CUDA
    if (self.is_cuda()) {
        if (self.is_contiguous()) {
            Tensor result = at::empty_cuda(self.sizes().vec(), self.dtype(), self.device().index());
            // Use CUDA copy
            at::cuda::launch_copy(self.data_ptr<float>(), result.mutable_data_ptr<float>(), self.numel(), nullptr);
            if (self.requires_grad()) {
                result.set_requires_grad(true);
            }
            return result;
        } else {
            // Non-contiguous CUDA tensor: use contiguous() which handles CUDA
            return contiguous(self);
        }
    }
#endif

    Tensor result = empty(self.sizes(), TensorOptions().dtype(self.dtype()).device(self.device()));

    if (self.is_contiguous()) {
        std::memcpy(result.data_ptr(), self.data_ptr(), self.nbytes());
    } else {
        result = contiguous(self);
    }

    if (self.requires_grad()) {
        result.set_requires_grad(true);
    }

    return result;
}

// ============================================================================
// Detach - Create tensor without autograd history
// ============================================================================

inline Tensor detach(const Tensor& self) {
    auto impl = std::make_shared<c10::TensorImpl>(
        self.storage(),
        self.dtype(),
        self.sizes(),
        self.strides(),
        self.storage_offset()
    );

    // Don't copy requires_grad - detached tensor doesn't require grad
    return Tensor(std::move(impl));
}

// ============================================================================
// Split and Chunk
// ============================================================================

inline std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim = 0) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK(split_size > 0);

    int64_t dim_size = self.size(dim);
    int64_t num_splits = (dim_size + split_size - 1) / split_size;

    std::vector<Tensor> result;
    result.reserve(num_splits);

    int64_t start = 0;
    for (int64_t i = 0; i < num_splits; ++i) {
        int64_t length = std::min(split_size, dim_size - start);
        result.push_back(self.narrow(dim, start, length));
        start += length;
    }

    return result;
}

inline std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim = 0) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK(chunks > 0);

    int64_t dim_size = self.size(dim);
    int64_t split_size = (dim_size + chunks - 1) / chunks;

    return split(self, split_size, dim);
}

// ============================================================================
// Stack and Cat
// ============================================================================

inline Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0) {
    PT_CHECK_MSG(!tensors.empty(), "cat: empty tensor list");

    const Tensor& first = tensors[0];
    int64_t ndim = first.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    // Compute result shape
    std::vector<int64_t> result_shape(first.sizes().begin(), first.sizes().end());
    int64_t total_dim_size = first.size(dim);

    for (size_t i = 1; i < tensors.size(); ++i) {
        const Tensor& t = tensors[i];
        PT_CHECK_MSG(t.dim() == ndim, "cat: all tensors must have same number of dimensions");

        for (int64_t d = 0; d < ndim; ++d) {
            if (d == dim) {
                total_dim_size += t.size(d);
            } else {
                PT_CHECK_MSG(t.size(d) == first.size(d),
                    "cat: sizes must match except in concatenation dimension");
            }
        }
    }

    result_shape[dim] = total_dim_size;

    Tensor result = empty(result_shape, TensorOptions().dtype(first.dtype()).device(first.device()));

    // Copy data
    int64_t offset = 0;
    for (const Tensor& t : tensors) {
        // Use narrow to get view into result, then copy
        Tensor dst = result.narrow(dim, offset, t.size(dim));

        PT_DISPATCH_ALL_TYPES(result.dtype(), "cat_copy", [&] {
            const scalar_t* src_data = t.data_ptr<scalar_t>();
            scalar_t* dst_data = dst.mutable_data_ptr<scalar_t>();

            // Simple copy for contiguous case
            if (t.is_contiguous() && dst.is_contiguous()) {
                std::memcpy(dst_data, src_data, t.nbytes());
            } else {
                // Strided copy
                int64_t n = t.numel();
                for (int64_t i = 0; i < n; ++i) {
                    dst_data[i] = src_data[i];
                }
            }
        });

        offset += t.size(dim);
    }

    return result;
}

inline Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0) {
    PT_CHECK_MSG(!tensors.empty(), "stack: empty tensor list");

    // Unsqueeze all tensors at dim, then cat
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());

    for (const Tensor& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }

    return cat(unsqueezed, dim);
}

} // namespace native
} // namespace at
