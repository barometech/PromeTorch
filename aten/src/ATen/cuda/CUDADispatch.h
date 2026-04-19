#pragma once

// ============================================================================
// CUDA Dispatch Layer for PromeTorch
// ============================================================================
// This file provides device dispatch - operations automatically route to
// CPU or CUDA implementations based on tensor device

#include "aten/src/ATen/core/Tensor.h"
#include "c10/core/Device.h"

#ifdef PT_USE_CUDA
#include "c10/cuda/CUDAAllocator.h"
#include "aten/src/ATen/cuda/CUDAOps.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace at {

// ============================================================================
// Device Transfer Functions
// ============================================================================

#ifdef PT_USE_CUDA

// Create a tensor on CUDA device
inline Tensor empty_cuda(c10::IntArrayRef sizes, c10::ScalarType dtype = c10::ScalarType::Float, int device = 0) {
    // Calculate total size
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;

    size_t nbytes = numel * c10::elementSize(dtype);

    // Allocate on CUDA
    auto& allocator = c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr data_ptr = allocator.allocate(nbytes, device, nullptr);

    // Create storage
    auto* storage_impl = new c10::StorageImpl(nbytes, std::move(data_ptr), &allocator, false);
    c10::Storage storage(storage_impl);

    // Create TensorImpl
    std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());
    auto impl = std::make_shared<c10::TensorImpl>(
        storage,
        dtype,
        sizes_vec
    );

    return Tensor(impl);
}

// Copy tensor to CUDA
inline Tensor to_cuda(const Tensor& src, int device = 0) {
    if (src.is_cuda() && src.device().index() == device) {
        return src;  // Already on the right device
    }

    // Create destination tensor on CUDA
    auto dst = empty_cuda(src.sizes().vec(), src.dtype(), device);

    // Copy data
    c10::cuda::cuda_memcpy_h2d(
        dst.data_ptr(),
        src.data_ptr(),
        src.nbytes(),
        nullptr
    );

    // Sync to ensure copy is complete
    c10::cuda::cuda_synchronize();

    // IMPORTANT: Preserve autograd metadata from source tensor
    // This ensures proper gradient tracking across device transfers
    auto* src_meta = src.autograd_meta();
    if (src_meta && src_meta->requires_grad_) {
        dst.set_requires_grad(true);
        auto* dst_meta = dst.autograd_meta();
        if (dst_meta) {
            // Copy key metadata fields
            dst_meta->is_leaf_ = src_meta->is_leaf_;
            dst_meta->output_nr_ = src_meta->output_nr_;
            dst_meta->retains_grad_ = src_meta->retains_grad_;
        }
    }

    return dst;
}

// Copy tensor to CPU
inline Tensor to_cpu(const Tensor& src) {
    if (src.is_cpu()) {
        return src;  // Already on CPU
    }

    // Create destination tensor on CPU
    auto dst = empty(src.sizes().vec(), TensorOptions().dtype(src.dtype()));

    // Copy data
    c10::cuda::cuda_memcpy_d2h(
        dst.data_ptr(),
        src.data_ptr(),
        src.nbytes(),
        nullptr
    );

    // Sync to ensure copy is complete
    c10::cuda::cuda_synchronize();

    return dst;
}

// Copy tensor to specified device
inline Tensor to_device(const Tensor& src, c10::Device device) {
    if (device.is_cpu()) {
        return to_cpu(src);
    } else if (device.is_cuda()) {
        return to_cuda(src, device.index() >= 0 ? device.index() : 0);
    }
    PT_CHECK_MSG(false, "Unsupported device type");
    return Tensor();
}

#else

// Stub implementations when CUDA is not enabled
inline Tensor to_cuda(const Tensor& src, int device = 0) {
    PT_CHECK_MSG(false, "CUDA not enabled. Build with -DPT_USE_CUDA=ON");
    return Tensor();
}

inline Tensor to_cpu(const Tensor& src) {
    return src;  // Already on CPU
}

inline Tensor to_device(const Tensor& src, c10::Device device) {
    if (device.is_cpu()) {
        return src;
    }
    PT_CHECK_MSG(false, "CUDA not enabled. Build with -DPT_USE_CUDA=ON");
    return Tensor();
}

#endif

// ============================================================================
// CUDA Operation Dispatch
// ============================================================================

#ifdef PT_USE_CUDA

// Unary operation dispatch
template<typename CPUOp, typename CUDALaunch>
inline Tensor unary_dispatch(const Tensor& input, CPUOp cpu_op, CUDALaunch cuda_launch) {
    if (input.is_cuda()) {
        auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
        cuda_launch(
            input.data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            input.numel(),
            nullptr
        );
        return output;
    } else {
        return cpu_op(input);
    }
}

// Binary operation dispatch (same shape)
template<typename CPUOp, typename CUDALaunch>
inline Tensor binary_dispatch(const Tensor& a, const Tensor& b, CPUOp cpu_op, CUDALaunch cuda_launch) {
    PT_CHECK_MSG(a.device() == b.device(), "Tensors must be on the same device");

    if (a.is_cuda()) {
        auto output = empty_cuda(a.sizes().vec(), a.dtype(), a.device().index());
        cuda_launch(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            a.numel(),
            nullptr
        );
        return output;
    } else {
        return cpu_op(a, b);
    }
}

// Reduction dispatch
template<typename CPUOp, typename CUDALaunch>
inline Tensor reduce_dispatch(const Tensor& input, CPUOp cpu_op, CUDALaunch cuda_launch) {
    if (input.is_cuda()) {
        auto output = empty_cuda({}, input.dtype(), input.device().index());
        cuda_launch(
            input.data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            input.numel(),
            nullptr
        );
        return output;
    } else {
        return cpu_op(input);
    }
}

#endif

// ============================================================================
// High-Level CUDA Operations
// ============================================================================

namespace cuda_ops {

#ifdef PT_USE_CUDA

// Element-wise operations
inline Tensor neg(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    at::cuda::launch_neg(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    return output;
}

inline Tensor abs(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    at::cuda::launch_abs(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    return output;
}

inline Tensor sqrt(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    at::cuda::launch_sqrt(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    return output;
}

inline Tensor rsqrt(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    at::cuda::launch_rsqrt(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    return output;
}

inline Tensor exp(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    at::cuda::launch_exp(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    return output;
}

inline Tensor log(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    at::cuda::launch_log(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    return output;
}

inline Tensor tanh(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    if (input.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_tanh_fp16(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            input.numel(), nullptr);
    } else {
        at::cuda::launch_tanh(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    }
    return output;
}

inline Tensor sigmoid(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    if (input.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_sigmoid_fp16(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            input.numel(), nullptr);
    } else {
        at::cuda::launch_sigmoid(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    }
    return output;
}

inline Tensor relu(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    if (input.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_relu_fp16(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            input.numel(), nullptr);
    } else {
        at::cuda::launch_relu(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    }
    return output;
}

inline Tensor silu(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    if (input.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_silu_fp16(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            input.numel(), nullptr);
    } else {
        at::cuda::launch_silu(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    }
    return output;
}

inline Tensor gelu(const Tensor& input) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    if (input.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_gelu_fp16(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            input.numel(), nullptr);
    } else {
        at::cuda::launch_gelu(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    }
    return output;
}

// Additional unary operations
inline Tensor log2(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_log2(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor log10(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_log10(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor tan(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_tan(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor ceil(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_ceil(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor floor(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_floor(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor round(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_round(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor sign(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_sign(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor reciprocal(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_reciprocal(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

// Binary operations
// IMPORTANT: All element-wise CUDA ops read data_ptr sequentially, so inputs
// MUST be contiguous. Non-contiguous tensors (e.g., transposed views from t())
// have different physical vs logical layout and would produce wrong results.
inline Tensor add(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.numel() == b.numel(), "Tensors must have same number of elements");
    PT_CHECK_MSG(a.dtype() == b.dtype(), "add: dtype mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    if (ac.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_add_fp16(
            reinterpret_cast<const __half*>(ac.data_ptr()),
            reinterpret_cast<const __half*>(bc.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            ac.numel(), nullptr);
    } else {
        at::cuda::launch_add(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    }
    return output;
}

inline Tensor sub(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.numel() == b.numel(), "Tensors must have same number of elements");
    PT_CHECK_MSG(a.dtype() == b.dtype(), "sub: dtype mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    if (ac.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_sub_fp16(
            reinterpret_cast<const __half*>(ac.data_ptr()),
            reinterpret_cast<const __half*>(bc.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            ac.numel(), nullptr);
    } else {
        at::cuda::launch_sub(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    }
    return output;
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.numel() == b.numel(), "Tensors must have same number of elements");
    PT_CHECK_MSG(a.dtype() == b.dtype(), "mul: dtype mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    if (ac.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_mul_fp16(
            reinterpret_cast<const __half*>(ac.data_ptr()),
            reinterpret_cast<const __half*>(bc.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            ac.numel(), nullptr);
    } else {
        at::cuda::launch_mul(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    }
    return output;
}

inline Tensor div(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.numel() == b.numel(), "Tensors must have same number of elements");
    PT_CHECK_MSG(a.dtype() == b.dtype(), "div: dtype mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    if (ac.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_div_fp16(
            reinterpret_cast<const __half*>(ac.data_ptr()),
            reinterpret_cast<const __half*>(bc.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            ac.numel(), nullptr);
    } else {
        at::cuda::launch_div(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    }
    return output;
}

inline Tensor mul_scalar(const Tensor& a, float scalar) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_mul_scalar(ac.data_ptr<float>(), scalar, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor add_scalar(const Tensor& a, float scalar) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_add_scalar(ac.data_ptr<float>(), scalar, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

// Broadcasting mul: [outer, inner] * [outer, 1] or [outer, inner] * [inner]
inline Tensor mul_broadcast(const Tensor& a, const Tensor& b) {
    // a is [outer, inner], b can be [outer, 1] or [inner]
    PT_CHECK_MSG(a.dim() == 2, "mul_broadcast: first tensor must be 2D");

    int64_t outer = a.size(0);
    int64_t inner = a.size(1);

    auto output = empty_cuda(a.sizes().vec(), a.dtype(), a.device().index());

    if (b.dim() == 2 && b.size(0) == outer && b.size(1) == 1) {
        // [outer, inner] * [outer, 1] -> broadcast row
        at::cuda::launch_mul_broadcast_row(
            a.data_ptr<float>(), b.data_ptr<float>(), output.mutable_data_ptr<float>(),
            outer, inner, nullptr);
    } else if (b.dim() == 1 && b.size(0) == inner) {
        // [outer, inner] * [inner] -> broadcast col
        at::cuda::launch_mul_broadcast_col(
            a.data_ptr<float>(), b.data_ptr<float>(), output.mutable_data_ptr<float>(),
            outer, inner, nullptr);
    } else {
        PT_CHECK_MSG(false, "mul_broadcast: unsupported shapes");
    }

    return output;
}

// Broadcasting add: [outer, inner] + [inner] (for bias addition)
inline Tensor add_broadcast(const Tensor& a, const Tensor& b) {
    // a is [outer, inner], b is [inner] (bias vector)
    PT_CHECK_MSG(a.dim() == 2, "add_broadcast: first tensor must be 2D");
    PT_CHECK_MSG(b.dim() == 1 && b.size(0) == a.size(1), "add_broadcast: second tensor must be 1D with size matching last dim");

    int64_t outer = a.size(0);
    int64_t inner = a.size(1);

    auto output = empty_cuda(a.sizes().vec(), a.dtype(), a.device().index());
    if (a.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_add_broadcast_fp16(
            reinterpret_cast<const __half*>(a.data_ptr()),
            reinterpret_cast<const __half*>(b.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            outer, inner, nullptr);
    } else {
        at::cuda::launch_add_broadcast_col(
            a.data_ptr<float>(), b.data_ptr<float>(), output.mutable_data_ptr<float>(),
            outer, inner, nullptr);
    }

    return output;
}

// Reductions
inline Tensor sum(const Tensor& input) {
    auto output = empty_cuda({1}, input.dtype(), input.device().index());
    at::cuda::launch_sum(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    c10::cuda::cuda_synchronize();
    return output;
}

inline Tensor mean(const Tensor& input) {
    auto output = empty_cuda({1}, input.dtype(), input.device().index());
    at::cuda::launch_mean(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    c10::cuda::cuda_synchronize();
    return output;
}

inline Tensor max(const Tensor& input) {
    auto output = empty_cuda({1}, input.dtype(), input.device().index());
    at::cuda::launch_max(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    c10::cuda::cuda_synchronize();
    return output;
}

inline Tensor min(const Tensor& input) {
    auto output = empty_cuda({1}, input.dtype(), input.device().index());
    at::cuda::launch_min(input.data_ptr<float>(), output.mutable_data_ptr<float>(), input.numel(), nullptr);
    c10::cuda::cuda_synchronize();
    return output;
}

// Dimensional reduction: sum over one dimension
inline Tensor sum_dim(const Tensor& input, int64_t dim, bool keepdim = false) {
    auto sizes = input.sizes().vec();
    int64_t ndim = static_cast<int64_t>(sizes.size());
    if (dim < 0) dim += ndim;

    // Compute outer_size, reduce_size, inner_size
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= sizes[i];
    int64_t reduce_size = sizes[dim];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= sizes[i];

    // Output shape
    std::vector<int64_t> output_sizes;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) output_sizes.push_back(1);
        } else {
            output_sizes.push_back(sizes[i]);
        }
    }
    if (output_sizes.empty()) output_sizes.push_back(1);

    auto output = empty_cuda(output_sizes, input.dtype(), input.device().index());
    at::cuda::launch_sum_dim(input.data_ptr<float>(), output.mutable_data_ptr<float>(),
                             outer_size, reduce_size, inner_size, nullptr);
    c10::cuda::cuda_synchronize();
    return output;
}

// Helper to check if tensor is transposed (row-major vs col-major)
inline bool is_transposed_2d(const Tensor& t) {
    if (t.dim() != 2) return false;
    // Transposed: stride[0] == 1, stride[1] == size[0]
    // Normal:     stride[0] == size[1], stride[1] == 1
    return t.stride(0) == 1 && t.stride(1) == t.size(0);
}

// Matrix multiplication - use contiguous for now
// TODO: Add GEMM transpose detection to avoid CPU roundtrip
inline Tensor mm(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.dim() == 2 && b.dim() == 2, "mm expects 2D tensors");
    PT_CHECK_MSG(a.size(1) == b.size(0), "Matrix dimensions incompatible for multiplication");

    // Make contiguous if needed (this is slow for transposed views)
    Tensor a_contig = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contig = b.is_contiguous() ? b : b.contiguous();

    int M = a_contig.size(0);
    int K = a_contig.size(1);
    int N = b_contig.size(1);

    auto output = empty_cuda({M, N}, a.dtype(), a.device().index());
    at::cuda::launch_gemm(
        a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), output.mutable_data_ptr<float>(),
        M, N, K, 1.0f, 0.0f, false, false, nullptr
    );
    // No sync needed: all ops on default stream, CUDA guarantees ordered execution.
    // Sync happens implicitly in to_cpu() when results are needed on host.
    return output;
}

// Batched matrix multiplication
inline Tensor bmm(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.dim() == 3 && b.dim() == 3, "bmm expects 3D tensors");
    PT_CHECK_MSG(a.size(0) == b.size(0), "Batch sizes must match");
    PT_CHECK_MSG(a.size(2) == b.size(1), "Matrix dimensions incompatible");

    // For bmm, require contiguous for now (batched transpose detection is complex)
    // TODO: Add batched GEMM with transpose support
    Tensor a_contig = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contig = b.is_contiguous() ? b : b.contiguous();

    int batch = a_contig.size(0);
    int M = a_contig.size(1);
    int K = a_contig.size(2);
    int N = b_contig.size(2);

    auto output = empty_cuda({batch, M, N}, a_contig.dtype(), a_contig.device().index());
    at::cuda::launch_bmm(a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), output.mutable_data_ptr<float>(), batch, M, N, K, nullptr);
    return output;
}

// Matrix-vector multiplication
inline Tensor mv(const Tensor& mat, const Tensor& vec) {
    PT_CHECK_MSG(mat.dim() == 2 && vec.dim() == 1, "mv expects 2D matrix and 1D vector");
    PT_CHECK_MSG(mat.size(1) == vec.size(0), "Matrix and vector dimensions incompatible");

    // IMPORTANT: CUDA kernels require contiguous memory layout!
    Tensor mat_contig = mat.is_contiguous() ? mat : mat.contiguous();
    Tensor vec_contig = vec.is_contiguous() ? vec : vec.contiguous();

    int M = mat_contig.size(0);
    int N = mat_contig.size(1);

    auto output = empty_cuda({M}, mat_contig.dtype(), mat_contig.device().index());
    at::cuda::launch_gemv(mat_contig.data_ptr<float>(), vec_contig.data_ptr<float>(), output.mutable_data_ptr<float>(), M, N, nullptr);
    return output;
}

// Fill tensor
inline void fill_(Tensor& tensor, float value) {
    at::cuda::launch_fill(tensor.mutable_data_ptr<float>(), value, tensor.numel(), nullptr);
}

// Copy data
inline void copy_(Tensor& dst, const Tensor& src) {
    PT_CHECK_MSG(dst.numel() == src.numel(), "Tensors must have same number of elements");
    at::cuda::launch_copy(src.data_ptr<float>(), dst.mutable_data_ptr<float>(), src.numel(), nullptr);
}

// Softmax
inline Tensor softmax(const Tensor& input, int dim) {
    auto sizes = input.sizes().vec();
    int64_t ndim = input.dim();
    if (dim < 0) dim += ndim;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= sizes[i];

    int64_t dim_size = sizes[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= sizes[i];

    auto output = empty_cuda(sizes, input.dtype(), input.device().index());
    if (input.dtype() == c10::ScalarType::Half) {
        at::cuda::launch_softmax_fp16(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.mutable_data_ptr()),
            outer_size, dim_size, inner_size, nullptr);
    } else {
        at::cuda::launch_softmax(input.data_ptr<float>(), output.mutable_data_ptr<float>(),
                                 outer_size, dim_size, inner_size, nullptr);
    }
    return output;
}

// Conv2d forward
inline Tensor conv2d_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,  // can be empty
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);

    int64_t out_channels = weight.size(0);
    int64_t kernel_h = weight.size(2);
    int64_t kernel_w = weight.size(3);

    int64_t out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = empty_cuda({batch_size, out_channels, out_height, out_width}, input.dtype(), input.device().index());

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    at::cuda::launch_conv2d_forward(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.mutable_data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(in_channels),
        static_cast<int>(in_height),
        static_cast<int>(in_width),
        static_cast<int>(out_channels),
        static_cast<int>(out_height),
        static_cast<int>(out_width),
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        nullptr
    );

    return output;
}

// MaxPool2d forward
inline Tensor max_pool2d_forward(
    const Tensor& input,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);

    int64_t out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    auto output = empty_cuda({batch_size, channels, out_height, out_width}, input.dtype(), input.device().index());

    at::cuda::launch_max_pool2d_forward(
        input.data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(channels),
        static_cast<int>(in_height),
        static_cast<int>(in_width),
        static_cast<int>(out_height),
        static_cast<int>(out_width),
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        nullptr
    );

    return output;
}

// AvgPool2d forward
inline Tensor avg_pool2d_forward(
    const Tensor& input,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    bool count_include_pad = true
) {
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);

    int64_t out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    auto output = empty_cuda({batch_size, channels, out_height, out_width}, input.dtype(), input.device().index());

    at::cuda::launch_avg_pool2d_forward(
        input.data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(channels),
        static_cast<int>(in_height),
        static_cast<int>(in_width),
        static_cast<int>(out_height),
        static_cast<int>(out_width),
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        count_include_pad,
        nullptr
    );

    return output;
}

// AdaptiveAvgPool2d forward
inline Tensor adaptive_avg_pool2d_forward(
    const Tensor& input,
    int out_height, int out_width
) {
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);

    auto output = empty_cuda({batch_size, channels, out_height, out_width}, input.dtype(), input.device().index());

    at::cuda::launch_adaptive_avg_pool2d_forward(
        input.data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(channels),
        static_cast<int>(in_height),
        static_cast<int>(in_width),
        out_height, out_width,
        nullptr
    );

    return output;
}

// BatchNorm2d forward (inference mode)
inline Tensor batch_norm2d_forward(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    const Tensor& running_mean,
    const Tensor& running_var,
    float eps
) {
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    auto output = empty_cuda({batch_size, channels, height, width}, input.dtype(), input.device().index());

    at::cuda::launch_batch_norm2d_forward(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(channels),
        static_cast<int>(height),
        static_cast<int>(width),
        eps,
        nullptr
    );

    return output;
}

// LeakyReLU
inline Tensor leaky_relu(const Tensor& input, float alpha) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    at::cuda::launch_leaky_relu(input.data_ptr<float>(), output.mutable_data_ptr<float>(), alpha, input.numel(), nullptr);
    return output;
}

// Clamp (for ReLU6 etc)
inline Tensor clamp(const Tensor& input, float min_val, float max_val) {
    auto output = empty_cuda(input.sizes().vec(), input.dtype(), input.device().index());
    at::cuda::launch_clamp(input.data_ptr<float>(), output.mutable_data_ptr<float>(), min_val, max_val, input.numel(), nullptr);
    return output;
}

// Cross Entropy Loss
// Input: logits (batch_size, num_classes)
// Target: (batch_size,) class indices as float
// reduction: 0=None, 1=Mean, 2=Sum
inline Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets, int reduction = 1) {
    PT_CHECK_MSG(logits.dim() == 2, "cross_entropy_loss: logits must be 2D (batch_size, num_classes)");
    PT_CHECK_MSG(targets.dim() == 1, "cross_entropy_loss: targets must be 1D (batch_size,)");

    int batch_size = logits.size(0);
    int num_classes = logits.size(1);

    // Output shape depends on reduction
    std::vector<int64_t> output_shape;
    if (reduction == 0) {
        output_shape = {batch_size};  // None: per-sample loss
    } else {
        output_shape = {1};  // Mean or Sum: scalar
    }

    auto output = empty_cuda(output_shape, logits.dtype(), logits.device().index());

    at::cuda::launch_cross_entropy_loss(
        logits.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        batch_size,
        num_classes,
        reduction,
        nullptr
    );

    c10::cuda::cuda_synchronize();
    return output;
}

// NLL Loss
inline Tensor nll_loss(const Tensor& log_probs, const Tensor& targets, int reduction = 1) {
    PT_CHECK_MSG(log_probs.dim() == 2, "nll_loss: log_probs must be 2D (batch_size, num_classes)");
    PT_CHECK_MSG(targets.dim() == 1, "nll_loss: targets must be 1D (batch_size,)");

    int batch_size = log_probs.size(0);
    int num_classes = log_probs.size(1);

    std::vector<int64_t> output_shape;
    if (reduction == 0) {
        output_shape = {batch_size};
    } else {
        output_shape = {1};
    }

    auto output = empty_cuda(output_shape, log_probs.dtype(), log_probs.device().index());

    at::cuda::launch_nll_loss(
        log_probs.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        batch_size,
        num_classes,
        reduction,
        nullptr
    );

    c10::cuda::cuda_synchronize();
    return output;
}

// Unary: sin, cos, square
inline Tensor sin(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_sin(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor cos(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_cos(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

inline Tensor square(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::cuda::launch_square(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel(), nullptr);
    return output;
}

// Binary: pow, pow_scalar, maximum, minimum
inline Tensor pow(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_pow(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor pow_scalar(const Tensor& a, float exp) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_pow_scalar(ac.data_ptr<float>(), exp, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor maximum(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_maximum(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor minimum(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_minimum(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

// Reduction: dot, argmax, argmin
inline Tensor dot(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda({}, ac.dtype(), ac.device().index());
    at::cuda::launch_dot(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor argmax(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda({}, c10::ScalarType::Long, ic.device().index());
    at::cuda::launch_argmax(ic.data_ptr<float>(), output.mutable_data_ptr<int64_t>(), ic.numel(), nullptr);
    return output;
}

inline Tensor argmin(const Tensor& input) {
    Tensor ic = input.is_contiguous() ? input : input.contiguous();
    auto output = empty_cuda({}, c10::ScalarType::Long, ic.device().index());
    at::cuda::launch_argmin(ic.data_ptr<float>(), output.mutable_data_ptr<int64_t>(), ic.numel(), nullptr);
    return output;
}

// Comparison operations (tensor vs tensor, float-returning)
inline Tensor eq(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_eq(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor ne(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_ne(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor lt(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_lt(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor le(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_le(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor gt(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_gt(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor ge(const Tensor& a, const Tensor& b) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_ge(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

// Scalar comparison operations
inline Tensor eq_scalar(const Tensor& a, float val) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_eq_scalar(ac.data_ptr<float>(), val, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor ne_scalar(const Tensor& a, float val) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_ne_scalar(ac.data_ptr<float>(), val, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor lt_scalar(const Tensor& a, float val) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_lt_scalar(ac.data_ptr<float>(), val, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor le_scalar(const Tensor& a, float val) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_le_scalar(ac.data_ptr<float>(), val, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor gt_scalar(const Tensor& a, float val) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_gt_scalar(ac.data_ptr<float>(), val, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

inline Tensor ge_scalar(const Tensor& a, float val) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    auto output = empty_cuda(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::cuda::launch_ge_scalar(ac.data_ptr<float>(), val, output.mutable_data_ptr<float>(), ac.numel(), nullptr);
    return output;
}

// Fused operations
inline Tensor addcmul(const Tensor& self, const Tensor& t1, const Tensor& t2, float value) {
    Tensor sc = self.is_contiguous() ? self : self.contiguous();
    Tensor t1c = t1.is_contiguous() ? t1 : t1.contiguous();
    Tensor t2c = t2.is_contiguous() ? t2 : t2.contiguous();
    auto output = empty_cuda(sc.sizes().vec(), sc.dtype(), sc.device().index());
    at::cuda::launch_addcmul(sc.data_ptr<float>(), t1c.data_ptr<float>(), t2c.data_ptr<float>(), value, output.mutable_data_ptr<float>(), sc.numel(), nullptr);
    return output;
}

inline Tensor addcdiv(const Tensor& self, const Tensor& t1, const Tensor& t2, float value) {
    Tensor sc = self.is_contiguous() ? self : self.contiguous();
    Tensor t1c = t1.is_contiguous() ? t1 : t1.contiguous();
    Tensor t2c = t2.is_contiguous() ? t2 : t2.contiguous();
    auto output = empty_cuda(sc.sizes().vec(), sc.dtype(), sc.device().index());
    at::cuda::launch_addcdiv(sc.data_ptr<float>(), t1c.data_ptr<float>(), t2c.data_ptr<float>(), value, output.mutable_data_ptr<float>(), sc.numel(), nullptr);
    return output;
}

#else

// Stub implementations when CUDA is disabled
inline Tensor neg(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor abs(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor sqrt(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor rsqrt(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor exp(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor log(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor tanh(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor sigmoid(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor relu(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor silu(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor gelu(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor add(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor add_broadcast(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor sub(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor mul(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor div(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor mul_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor add_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor sum(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor sum_dim(const Tensor&, int64_t, bool = false) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor mean(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor max(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor min(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor mm(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor bmm(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor mv(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline void fill_(Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); }
inline void copy_(Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); }
inline Tensor softmax(const Tensor&, int) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor conv2d_forward(const Tensor&, const Tensor&, const Tensor&, int, int, int, int, int, int, int) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor max_pool2d_forward(const Tensor&, int, int, int, int, int, int) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor avg_pool2d_forward(const Tensor&, int, int, int, int, int, int, bool = true) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor adaptive_avg_pool2d_forward(const Tensor&, int, int) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor batch_norm2d_forward(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor leaky_relu(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor clamp(const Tensor&, float, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor cross_entropy_loss(const Tensor&, const Tensor&, int = 1) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor nll_loss(const Tensor&, const Tensor&, int = 1) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor sin(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor cos(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor square(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor pow(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor pow_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor maximum(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor minimum(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor dot(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor argmax(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor argmin(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
// Additional unary stubs
inline Tensor log2(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor log10(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor tan(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor ceil(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor floor(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor round(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor sign(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor reciprocal(const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
// Comparison stubs (tensor vs tensor)
inline Tensor eq(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor ne(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor lt(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor le(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor gt(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor ge(const Tensor&, const Tensor&) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
// Scalar comparison stubs
inline Tensor eq_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor ne_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor lt_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor le_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor gt_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor ge_scalar(const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
// Fused ops stubs
inline Tensor addcmul(const Tensor&, const Tensor&, const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }
inline Tensor addcdiv(const Tensor&, const Tensor&, const Tensor&, float) { PT_CHECK_MSG(false, "CUDA not enabled"); return Tensor(); }

#endif

} // namespace cuda_ops

} // namespace at
