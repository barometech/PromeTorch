#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <numeric>
#include <algorithm>
#include <array>
#include <string>

// Threading via c10::ThreadPool — parallelism in hot_loops.cpp

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDAOps.h"
#include "c10/cuda/CUDAAllocator.h"
#endif

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

    bool is_cuda = self.is_cuda();

#ifdef PT_USE_CUDA
    // Fast path for CUDA transposed 2D tensors - use GPU transpose kernel
    if (is_cuda && self.dim() == 2 &&
        self.stride(0) == 1 && self.stride(1) == self.size(0)) {
        // This is a simple 2D transpose: physical [N, K] -> logical [K, N]
        // Use CUDA transpose kernel (no CPU roundtrip!)
        int64_t logical_rows = self.size(0);  // K (rows in logical view)
        int64_t logical_cols = self.size(1);  // N (cols in logical view)
        // Physical layout is [N, K], so transpose from [N, K] to [K, N]
        int64_t phys_rows = logical_cols;  // N
        int64_t phys_cols = logical_rows;  // K

        // Allocate output tensor on CUDA
        size_t nbytes = static_cast<size_t>(self.numel() * c10::elementSize(self.dtype()));
        c10::DataPtr data_ptr = c10::cuda::CUDACachingAllocator::get().allocate(nbytes);
        c10::Allocator* cuda_alloc = &c10::cuda::CUDACachingAllocator::get();
        c10::Storage storage(nbytes, std::move(data_ptr), cuda_alloc, false);

        // Compute contiguous strides for [K, N] output
        std::vector<int64_t> out_strides = {logical_cols, 1};  // row-major: [N, 1]

        auto impl = std::make_shared<c10::TensorImpl>(
            std::move(storage),
            self.dtype(),
            self.sizes(),
            out_strides,
            0  // storage_offset
        );
        Tensor result(impl);

        at::cuda::launch_transpose(
            self.data_ptr<float>(), result.mutable_data_ptr<float>(),
            static_cast<int>(phys_rows), static_cast<int>(phys_cols), nullptr);

        // FIX 3.1: removed cuda_synchronize() — CUDA kernels on same stream
        // are implicitly ordered. Sync here killed async pipeline.

        if (self.requires_grad()) {
            result.set_requires_grad(true);
        }
        return result;
    }
#endif

    // CPU fallback for non-transposed or non-CUDA tensors
    // IMPORTANT: Save original strides BEFORE to_cpu (to_cpu creates contiguous strides!)
    auto original_strides = self.strides().vec();
    auto original_sizes = self.sizes().vec();

    Tensor self_cpu = self;
#ifdef PT_USE_CUDA
    if (is_cuda) {
        // Move raw data to CPU (preserving physical layout)
        self_cpu = at::to_cpu(self);
    }
#endif

    Tensor result = empty(original_sizes, TensorOptions().dtype(self_cpu.dtype()));

    // Copy data in contiguous order using ORIGINAL strides (not self_cpu strides!)
    PT_DISPATCH_ALL_TYPES(self_cpu.dtype(), "contiguous", [&] {
        const scalar_t* src = self_cpu.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        int64_t ndim = static_cast<int64_t>(original_sizes.size());
        int64_t total = self_cpu.numel();

        // Use strided copy with ORIGINAL strides

        for (int64_t i = 0; i < total; ++i) {
            // Convert contiguous index to strided index using original strides
            int64_t remaining = i;
            int64_t src_idx = 0;

            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t idx_in_dim = remaining % original_sizes[d];
                remaining /= original_sizes[d];
                src_idx += idx_in_dim * original_strides[d];
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
// Contiguous with MemoryFormat
// ============================================================================

inline Tensor contiguous(const Tensor& self, c10::MemoryFormat memory_format) {
    if (memory_format == c10::MemoryFormat::Contiguous || memory_format == c10::MemoryFormat::Preserve) {
        return contiguous(self);
    }

    // Check if already in desired format
    if (self.is_contiguous(memory_format)) {
        return self;
    }

    // Convert to channels-last (NHWC) or channels-last-3d (NDHWC)
    int64_t ndim = self.dim();

    if (memory_format == c10::MemoryFormat::ChannelsLast) {
        PT_CHECK_MSG(ndim == 4, "ChannelsLast requires 4D tensor, got ", ndim, "D");
    } else if (memory_format == c10::MemoryFormat::ChannelsLast3d) {
        PT_CHECK_MSG(ndim == 5, "ChannelsLast3d requires 5D tensor, got ", ndim, "D");
    }

    // Compute new strides for desired layout
    std::vector<int64_t> new_strides(ndim);

    if (memory_format == c10::MemoryFormat::ChannelsLast) {
        // NHWC strides for [N, C, H, W]: {C*H*W, 1, W*C, C}
        int64_t C = self.size(1), H = self.size(2), W = self.size(3);
        new_strides[0] = C * H * W;  // N
        new_strides[1] = 1;          // C (innermost)
        new_strides[2] = W * C;      // H
        new_strides[3] = C;          // W
    } else {
        // NDHWC: {C*D*H*W, 1, H*W*C, W*C, C}
        int64_t C = self.size(1), D = self.size(2), H = self.size(3), W = self.size(4);
        new_strides[0] = C * D * H * W;
        new_strides[1] = 1;
        new_strides[2] = H * W * C;
        new_strides[3] = W * C;
        new_strides[4] = C;
    }

    // Create result tensor with new strides and copy data
    Tensor result = empty(self.sizes(), TensorOptions().dtype(self.dtype()).device(self.device()));

    // Set channels-last strides
    result.unsafeGetTensorImpl()->set_sizes_and_strides(self.sizes(), new_strides);

    // Copy data from source using strided access
    PT_DISPATCH_ALL_TYPES(self.dtype(), "contiguous_channels_last", [&] {
        const scalar_t* src = self.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        int64_t total = self.numel();

        for (int64_t i = 0; i < total; ++i) {
            // Convert from dst (channels-last) linear index to multi-dim coords
            int64_t remaining = i;
            int64_t src_idx = 0;
            int64_t dst_idx = 0;

            // Compute multi-dim coords from dst's linear traversal order
            // For channels-last, the traversal goes N, H, W, C
            // We need to convert this to source strides
            std::vector<int64_t> coords(ndim);
            for (int64_t d = ndim - 1; d >= 0; --d) {
                // Traverse in order of decreasing dst stride
                coords[d] = remaining % self.size(d);
                remaining /= self.size(d);
            }

            for (int64_t d = 0; d < ndim; ++d) {
                src_idx += coords[d] * self.stride(d);
                dst_idx += coords[d] * new_strides[d];
            }

            dst[dst_idx] = src[src_idx];
        }
    });

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
            // Simple copy for contiguous case
            if (t.is_contiguous() && dst.is_contiguous()) {
                const scalar_t* src_data = t.data_ptr<scalar_t>();
                scalar_t* dst_data = dst.mutable_data_ptr<scalar_t>();
                std::memcpy(dst_data, src_data, t.nbytes());
            } else {
                // Non-contiguous: make source contiguous, then copy
                Tensor t_c = t.contiguous();
                Tensor dst_c = dst.contiguous();
                const scalar_t* src_data = t_c.data_ptr<scalar_t>();
                scalar_t* dst_data = dst_c.mutable_data_ptr<scalar_t>();
                int64_t n = t_c.numel();
                for (int64_t i = 0; i < n; ++i) {
                    dst_data[i] = src_data[i];
                }
                // Copy back to the (possibly non-contiguous) dst view
                dst.copy_(dst_c);
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

// ============================================================================
// Flip — reverse elements along given dimensions
// ============================================================================

inline Tensor flip(const Tensor& self, c10::IntArrayRef dims) {
    Tensor input = self.contiguous();
    Tensor result = empty(input.sizes(), TensorOptions().dtype(input.dtype()).device(input.device()));

    int64_t ndim = input.dim();
    int64_t total = input.numel();
    auto sizes = input.sizes();
    auto strides = input.strides();

    // Build a set of dims to flip
    std::vector<bool> flip_dim(ndim, false);
    for (auto d : dims) {
        int64_t dim = d < 0 ? d + ndim : d;
        PT_CHECK(dim >= 0 && dim < ndim);
        flip_dim[dim] = true;
    }

    PT_DISPATCH_ALL_TYPES(input.dtype(), "flip", [&] {
        const scalar_t* src = input.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        for (int64_t i = 0; i < total; ++i) {
            // Convert flat index to multi-dim index
            int64_t remaining = i;
            int64_t src_offset = 0;

            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t idx = remaining % sizes[d];
                remaining /= sizes[d];

                if (flip_dim[d]) {
                    src_offset += (sizes[d] - 1 - idx) * strides[d];
                } else {
                    src_offset += idx * strides[d];
                }
            }

            dst[i] = src[src_offset];
        }
    });

    return result;
}

// ============================================================================
// Roll — cyclically shift elements along given dimensions
// ============================================================================

inline Tensor roll(const Tensor& self, c10::IntArrayRef shifts, c10::IntArrayRef dims) {
    PT_CHECK_MSG(shifts.size() == dims.size(),
        "roll: shifts and dims must have the same size");

    Tensor result = self.contiguous().clone();

    for (size_t i = 0; i < shifts.size(); ++i) {
        int64_t shift = shifts[i];
        int64_t dim = dims[i] < 0 ? dims[i] + result.dim() : dims[i];
        PT_CHECK(dim >= 0 && dim < result.dim());

        int64_t dim_size = result.size(dim);
        shift = ((shift % dim_size) + dim_size) % dim_size;  // normalize
        if (shift == 0) continue;

        Tensor tmp = result.clone();
        int64_t total = result.numel();
        auto sizes = result.sizes();

        PT_DISPATCH_ALL_TYPES(result.dtype(), "roll", [&] {
            const scalar_t* src = tmp.data_ptr<scalar_t>();
            scalar_t* dst = result.mutable_data_ptr<scalar_t>();

            for (int64_t idx = 0; idx < total; ++idx) {
                // Convert flat index to multi-dim coords
                int64_t remaining = idx;
                std::vector<int64_t> coords(result.dim());
                for (int64_t d = result.dim() - 1; d >= 0; --d) {
                    coords[d] = remaining % sizes[d];
                    remaining /= sizes[d];
                }

                // Compute source coord (shift backwards)
                int64_t src_coord = ((coords[dim] - shift) % dim_size + dim_size) % dim_size;
                coords[dim] = src_coord;

                // Convert back to flat index
                int64_t src_idx = 0;
                int64_t stride = 1;
                for (int64_t d = result.dim() - 1; d >= 0; --d) {
                    src_idx += coords[d] * stride;
                    stride *= sizes[d];
                }

                dst[idx] = src[src_idx];
            }
        });
    }

    return result;
}

// ============================================================================
// Meshgrid — create coordinate grids
// ============================================================================

inline std::vector<Tensor> meshgrid(const std::vector<Tensor>& tensors, const std::string& indexing = "ij") {
    PT_CHECK_MSG(!tensors.empty(), "meshgrid: empty tensor list");
    for (const auto& t : tensors) {
        PT_CHECK_MSG(t.dim() == 1, "meshgrid: all tensors must be 1D");
    }

    size_t N = tensors.size();
    std::vector<int64_t> shape;

    if (indexing == "xy" && N >= 2) {
        shape.push_back(tensors[1].size(0));
        shape.push_back(tensors[0].size(0));
        for (size_t i = 2; i < N; ++i) shape.push_back(tensors[i].size(0));
    } else {
        for (const auto& t : tensors) shape.push_back(t.size(0));
    }

    std::vector<Tensor> result;

    for (size_t i = 0; i < N; ++i) {
        // Determine which dimension this tensor maps to
        size_t dim;
        if (indexing == "xy" && N >= 2) {
            if (i == 0) dim = 1;
            else if (i == 1) dim = 0;
            else dim = i;
        } else {
            dim = i;
        }

        // Reshape to broadcast shape
        std::vector<int64_t> view_shape(N, 1);
        view_shape[dim] = tensors[i].size(0);

        Tensor expanded = tensors[i].reshape(view_shape).expand(shape);
        result.push_back(expanded.contiguous());
    }

    return result;
}

// ============================================================================
// repeat_interleave — repeat each element
// ============================================================================

inline Tensor repeat_interleave(const Tensor& self, int64_t repeats, int64_t dim = 0) {
    if (dim < 0) dim += self.dim();
    PT_CHECK(dim >= 0 && dim < self.dim());

    int64_t dim_size = self.size(dim);
    int64_t new_dim_size = dim_size * repeats;

    std::vector<int64_t> new_shape = self.sizes().vec();
    new_shape[dim] = new_dim_size;

    Tensor input = self.contiguous();
    Tensor result = empty(new_shape, TensorOptions().dtype(input.dtype()).device(input.device()));

    int64_t total = result.numel();
    auto sizes = result.sizes();
    auto in_sizes = input.sizes();

    PT_DISPATCH_ALL_TYPES(input.dtype(), "repeat_interleave", [&] {
        const scalar_t* src = input.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();

        for (int64_t idx = 0; idx < total; ++idx) {
            // Convert flat index to multi-dim coords
            int64_t remaining = idx;
            std::vector<int64_t> coords(self.dim());
            for (int64_t d = self.dim() - 1; d >= 0; --d) {
                coords[d] = remaining % sizes[d];
                remaining /= sizes[d];
            }

            // Map back to source: divide dim coordinate by repeats
            coords[dim] = coords[dim] / repeats;

            // Convert to source flat index
            int64_t src_idx = 0;
            int64_t stride = 1;
            for (int64_t d = self.dim() - 1; d >= 0; --d) {
                src_idx += coords[d] * stride;
                stride *= in_sizes[d];
            }

            dst[idx] = src[src_idx];
        }
    });

    return result;
}

// ============================================================================
// unique — return unique elements
// ============================================================================

inline std::tuple<Tensor, Tensor, Tensor> unique(
    const Tensor& self,
    bool sorted = true,
    bool return_inverse = false,
    bool return_counts = false
) {
    Tensor input = self.contiguous();
    int64_t n = input.numel();

    Tensor result_unique, result_inverse, result_counts;

    PT_DISPATCH_ALL_TYPES(input.dtype(), "unique", [&] {
        const scalar_t* data = input.data_ptr<scalar_t>();

        // Collect unique values
        std::vector<scalar_t> vals(data, data + n);
        if (sorted) std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

        int64_t num_unique = static_cast<int64_t>(vals.size());
        result_unique = empty({num_unique}, TensorOptions().dtype(input.dtype()));
        scalar_t* u_data = result_unique.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < num_unique; ++i) u_data[i] = vals[i];

        if (return_inverse) {
            result_inverse = empty({n}, TensorOptions().dtype(c10::ScalarType::Long));
            int64_t* inv = result_inverse.mutable_data_ptr<int64_t>();
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < num_unique; ++j) {
                    if (data[i] == vals[j]) { inv[i] = j; break; }
                }
            }
        }

        if (return_counts) {
            result_counts = zeros({num_unique}, TensorOptions().dtype(c10::ScalarType::Long));
            int64_t* cnt = result_counts.mutable_data_ptr<int64_t>();
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < num_unique; ++j) {
                    if (data[i] == vals[j]) { cnt[j]++; break; }
                }
            }
        }
    });

    return {result_unique, result_inverse, result_counts};
}

// ============================================================================
// tril_indices / triu_indices — indices of triangular parts
// ============================================================================

inline Tensor tril_indices(int64_t row, int64_t col, int64_t offset = 0) {
    std::vector<int64_t> rows_vec, cols_vec;
    for (int64_t i = 0; i < row; ++i) {
        for (int64_t j = 0; j <= std::min(i + offset, col - 1); ++j) {
            if (j >= 0) {
                rows_vec.push_back(i);
                cols_vec.push_back(j);
            }
        }
    }

    int64_t n = static_cast<int64_t>(rows_vec.size());
    Tensor result = empty({2, n}, TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* data = result.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = rows_vec[i];
        data[n + i] = cols_vec[i];
    }
    return result;
}

inline Tensor triu_indices(int64_t row, int64_t col, int64_t offset = 0) {
    std::vector<int64_t> rows_vec, cols_vec;
    for (int64_t i = 0; i < row; ++i) {
        for (int64_t j = std::max(i + offset, (int64_t)0); j < col; ++j) {
            rows_vec.push_back(i);
            cols_vec.push_back(j);
        }
    }

    int64_t n = static_cast<int64_t>(rows_vec.size());
    Tensor result = empty({2, n}, TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* data = result.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = rows_vec[i];
        data[n + i] = cols_vec[i];
    }
    return result;
}

// ============================================================================
// Unfold (im2col) — extract sliding local blocks
// Input: (N, C, H, W) → Output: (N, C*kH*kW, L)
// where L = number of valid positions
// ============================================================================

inline Tensor unfold_im2col(const Tensor& input,
                            std::array<int64_t, 2> kernel_size,
                            std::array<int64_t, 2> dilation = {1, 1},
                            std::array<int64_t, 2> padding = {0, 0},
                            std::array<int64_t, 2> stride = {1, 1}) {
    PT_CHECK_MSG(input.dim() == 4, "unfold requires 4D input (N, C, H, W)");

    Tensor inp = input.contiguous();
    int64_t N = inp.size(0);
    int64_t C = inp.size(1);
    int64_t H = inp.size(2);
    int64_t W = inp.size(3);
    int64_t kH = kernel_size[0], kW = kernel_size[1];
    int64_t dH = dilation[0], dW = dilation[1];
    int64_t pH = padding[0], pW = padding[1];
    int64_t sH = stride[0], sW = stride[1];

    int64_t out_H = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t out_W = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    int64_t L = out_H * out_W;
    int64_t col_channels = C * kH * kW;

    Tensor output = at::zeros({N, col_channels, L});

    const float* in_data = inp.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t col_idx = (c * kH + kh) * kW + kw;
                    for (int64_t oh = 0; oh < out_H; ++oh) {
                        for (int64_t ow = 0; ow < out_W; ++ow) {
                            int64_t ih = oh * sH - pH + kh * dH;
                            int64_t iw = ow * sW - pW + kw * dW;
                            int64_t l_idx = oh * out_W + ow;

                            float val = 0.0f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                val = in_data[((n * C + c) * H + ih) * W + iw];
                            }
                            out_data[(n * col_channels + col_idx) * L + l_idx] = val;
                        }
                    }
                }
            }
        }
    }

    return output;
}

// ============================================================================
// Fold (col2im) — combine sliding local blocks into image
// Input: (N, C*kH*kW, L) → Output: (N, C, H, W)
// ============================================================================

inline Tensor fold_col2im(const Tensor& input,
                          std::array<int64_t, 2> output_size,
                          std::array<int64_t, 2> kernel_size,
                          std::array<int64_t, 2> dilation = {1, 1},
                          std::array<int64_t, 2> padding = {0, 0},
                          std::array<int64_t, 2> stride = {1, 1}) {
    PT_CHECK_MSG(input.dim() == 3, "fold requires 3D input (N, C*kH*kW, L)");

    Tensor inp = input.contiguous();
    int64_t N = inp.size(0);
    int64_t col_channels = inp.size(1);
    int64_t L = inp.size(2);

    int64_t oH = output_size[0], oW = output_size[1];
    int64_t kH = kernel_size[0], kW = kernel_size[1];
    int64_t dH = dilation[0], dW = dilation[1];
    int64_t pH = padding[0], pW = padding[1];
    int64_t sH = stride[0], sW = stride[1];

    int64_t C = col_channels / (kH * kW);
    PT_CHECK_MSG(C * kH * kW == col_channels, "fold: col_channels must be divisible by kH*kW");

    int64_t out_H_calc = (oH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t out_W_calc = (oW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    PT_CHECK_MSG(out_H_calc * out_W_calc == L, "fold: L doesn't match output_size");

    Tensor output = at::zeros({N, C, oH, oW});

    const float* in_data = inp.data_ptr<float>();
    float* out_data = output.mutable_data_ptr<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t col_idx = (c * kH + kh) * kW + kw;
                    for (int64_t oh = 0; oh < out_H_calc; ++oh) {
                        for (int64_t ow = 0; ow < out_W_calc; ++ow) {
                            int64_t ih = oh * sH - pH + kh * dH;
                            int64_t iw = ow * sW - pW + kw * dW;
                            int64_t l_idx = oh * out_W_calc + ow;

                            if (ih >= 0 && ih < oH && iw >= 0 && iw < oW) {
                                out_data[((n * C + c) * oH + ih) * oW + iw] +=
                                    in_data[(n * col_channels + col_idx) * L + l_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

} // namespace native
} // namespace at
