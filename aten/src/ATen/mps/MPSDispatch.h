#pragma once
// ============================================================================
// MPSDispatch.h — MPS-side analogue of CUDADispatch.h
// ============================================================================
// Routes operations to Metal Performance Shaders when a Tensor sits on
// c10::DeviceType::MPS. When PT_USE_MPS is OFF (or we're not on Apple), the
// dispatch throws a clear error.
//
// Conventions mirror CUDADispatch.h so higher-level code can switch devices
// with no surprises.
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "c10/core/Device.h"

#if defined(__APPLE__) && defined(PT_USE_MPS)
#  include "c10/mps/MPSAllocator.h"
#  include "aten/src/ATen/mps/MPSDevice.h"
#  include "aten/src/ATen/mps/MPSKernels.h"
#endif

namespace at {

// ============================================================================
// Error helper — used by stubs on non-Apple builds.
// ============================================================================

[[noreturn]] inline void mps_unavailable() {
    PT_CHECK_MSG(false,
        "MPS backend only available on Apple platforms with PT_USE_MPS=ON");
    std::abort();  // unreachable
}

// ============================================================================
// Tensor factory / transfer
// ============================================================================

#if defined(__APPLE__) && defined(PT_USE_MPS)

inline Tensor empty_mps(c10::IntArrayRef sizes,
                        c10::ScalarType dtype = c10::ScalarType::Float,
                        int device = 0) {
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;
    size_t nbytes = static_cast<size_t>(numel) * c10::elementSize(dtype);

    auto& allocator = c10::mps::MPSAllocator::get();
    c10::DataPtr data_ptr = allocator.allocate(nbytes);

    auto* storage_impl = new c10::StorageImpl(
        nbytes, std::move(data_ptr), &allocator, /*resizable=*/false);
    c10::Storage storage(storage_impl);

    std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());
    auto impl = std::make_shared<c10::TensorImpl>(storage, dtype, sizes_vec);
    (void)device;  // single-GPU for now
    return Tensor(impl);
}

// CPU → MPS copy. Unified memory makes this a memcpy on Apple Silicon.
inline Tensor to_mps(const Tensor& src, int device = 0) {
    if (src.device().is_mps() && src.device().index() == device) return src;
    auto dst = empty_mps(src.sizes().vec(), src.dtype(), device);
    std::memcpy(dst.mutable_data_ptr<float>(),
                src.data_ptr<float>(),
                src.nbytes());
    // Propagate autograd flag (details follow CUDA path).
    auto* src_meta = src.autograd_meta();
    if (src_meta && src_meta->requires_grad_) {
        dst.set_requires_grad(true);
    }
    return dst;
}

inline Tensor to_cpu_from_mps(const Tensor& src) {
    if (src.is_cpu()) return src;
    // Make sure all in-flight MPS work is done before we read contents.
    at::mps::MPSDevice::get().synchronize();
    auto dst = empty(src.sizes().vec(),
                     TensorOptions().dtype(src.dtype()));
    std::memcpy(dst.mutable_data_ptr<float>(),
                src.data_ptr<float>(),
                src.nbytes());
    return dst;
}

#else // stubs

inline Tensor empty_mps(c10::IntArrayRef, c10::ScalarType = c10::ScalarType::Float,
                        int = 0)              { mps_unavailable(); }
inline Tensor to_mps(const Tensor&, int = 0)  { mps_unavailable(); }
inline Tensor to_cpu_from_mps(const Tensor&)  { mps_unavailable(); }

#endif

// ============================================================================
// High-level op dispatch. When `is_mps()` — run on Metal; otherwise fall
// back to the caller-provided CPU op.
// ============================================================================

namespace mps_ops {

#if defined(__APPLE__) && defined(PT_USE_MPS)

inline Tensor add(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.numel() == b.numel(),
                 "mps_ops::add: shape mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto out = empty_mps(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::mps::launch_add_mps(ac.data_ptr<float>(),
                            bc.data_ptr<float>(),
                            out.mutable_data_ptr<float>(),
                            static_cast<std::size_t>(ac.numel()));
    return out;
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.numel() == b.numel(),
                 "mps_ops::mul: shape mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    auto out = empty_mps(ac.sizes().vec(), ac.dtype(), ac.device().index());
    at::mps::launch_mul_mps(ac.data_ptr<float>(),
                            bc.data_ptr<float>(),
                            out.mutable_data_ptr<float>(),
                            static_cast<std::size_t>(ac.numel()));
    return out;
}

inline Tensor relu(const Tensor& in) {
    Tensor ic = in.is_contiguous() ? in : in.contiguous();
    auto out = empty_mps(ic.sizes().vec(), ic.dtype(), ic.device().index());
    at::mps::launch_relu_mps(ic.data_ptr<float>(),
                             out.mutable_data_ptr<float>(),
                             static_cast<std::size_t>(ic.numel()));
    return out;
}

inline Tensor mm(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.dim() == 2 && b.dim() == 2,
                 "mps_ops::mm: expects 2D tensors");
    PT_CHECK_MSG(a.size(1) == b.size(0),
                 "mps_ops::mm: inner dims mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    int M = static_cast<int>(ac.size(0));
    int K = static_cast<int>(ac.size(1));
    int N = static_cast<int>(bc.size(1));
    auto out = empty_mps({M, N}, ac.dtype(), ac.device().index());
    at::mps::launch_matmul_mps(ac.data_ptr<float>(),
                               bc.data_ptr<float>(),
                               out.mutable_data_ptr<float>(),
                               M, N, K);
    return out;
}

#else

inline Tensor add (const Tensor&, const Tensor&)  { mps_unavailable(); }
inline Tensor mul (const Tensor&, const Tensor&)  { mps_unavailable(); }
inline Tensor relu(const Tensor&)                 { mps_unavailable(); }
inline Tensor mm  (const Tensor&, const Tensor&)  { mps_unavailable(); }

#endif

// ---- Generic dispatch helpers (mirror CUDADispatch.h idioms) ---------------

template <typename CPUOp>
inline Tensor unary_dispatch(const Tensor& in, CPUOp cpu_op) {
#if defined(__APPLE__) && defined(PT_USE_MPS)
    if (in.is_mps()) {
        // Only relu is hooked up in this first drop; add more as needed.
        PT_CHECK_MSG(false, "mps_ops::unary_dispatch: generic path not "
                            "implemented — call specific op directly");
    }
#endif
    return cpu_op(in);
}

template <typename CPUOp>
inline Tensor binary_dispatch(const Tensor& a, const Tensor& b, CPUOp cpu_op) {
    PT_CHECK_MSG(a.device() == b.device(),
                 "binary_dispatch: tensors on different devices");
#if defined(__APPLE__) && defined(PT_USE_MPS)
    if (a.is_mps()) {
        // Specialise per-op via the direct functions above; this helper is
        // a convenience for call sites that want one entry point.
        PT_CHECK_MSG(false, "mps_ops::binary_dispatch: call add/mul directly");
    }
#endif
    return cpu_op(a, b);
}

} // namespace mps_ops

// ============================================================================
// Uniform device-transfer entry point. Extends the CUDA/CPU pair in
// CUDADispatch.h with the MPS leg. Call this from higher-level `.to()` impls.
// ============================================================================

inline Tensor to_device_mps_aware(const Tensor& src, c10::Device device) {
    if (device.is_cpu()) {
#if defined(__APPLE__) && defined(PT_USE_MPS)
        if (src.is_mps()) return to_cpu_from_mps(src);
#endif
        return src;  // assume already on CPU
    }
    if (device.is_mps()) {
#if defined(__APPLE__) && defined(PT_USE_MPS)
        return to_mps(src, device.index() >= 0 ? device.index() : 0);
#else
        mps_unavailable();
#endif
    }
    PT_CHECK_MSG(false, "to_device_mps_aware: unsupported target device");
    return Tensor();
}

} // namespace at
