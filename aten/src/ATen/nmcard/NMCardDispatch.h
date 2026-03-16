#pragma once
// ============================================================================
// NMCard Dispatch Layer for PromeTorch
// ============================================================================
// Device transfer: empty_nmcard, to_nmcard, nmcard_to_cpu
// Pattern follows CUDADispatch.h

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "c10/core/Device.h"
#include "c10/nmcard/NMCardAllocator.h"
#include "aten/src/ATen/nmcard/NMCardOps.h"

namespace at {

#ifdef PT_USE_NMCARD

// Create an empty tensor on NMCard device (emulator)
inline Tensor empty_nmcard(c10::IntArrayRef sizes,
                           c10::ScalarType dtype = c10::ScalarType::Float,
                           int device = 0) {
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;

    size_t nbytes = numel * c10::elementSize(dtype);

    auto& allocator = c10::nmcard::NMCardAllocator::get();
    c10::DataPtr data_ptr = allocator.allocate(nbytes);

    auto* storage_impl = new c10::StorageImpl(nbytes, std::move(data_ptr), &allocator, false);
    c10::Storage storage(storage_impl);

    std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());
    auto impl = std::make_shared<c10::TensorImpl>(
        storage, dtype, sizes_vec
    );

    return Tensor(impl);
}

// Auto-initialize hardware on first use
inline void ensure_hardware_init() {
    static bool tried = false;
    if (!tried) {
        tried = true;
        auto& hw = nmcard::NMCardHardware::get();
        if (!hw.is_available()) {
            // Try default dispatcher locations
            const char* paths[] = {
                "aten/src/ATen/nmcard/nmc_programs/dispatcher.abs",
                "dispatcher.abs",
                nullptr
            };
            for (int i = 0; paths[i]; ++i) {
                if (hw.init(paths[i])) {
                    std::cout << "NMCard hardware auto-initialized" << std::endl;
                    break;
                }
            }
        }
    }
}

// Copy tensor to NMCard device
inline Tensor to_nmcard(const Tensor& src, int device = 0) {
    if (src.is_nmcard()) {
        return src; // Already on NMCard
    }

    // Auto-init hardware
    ensure_hardware_init();

    auto dst = empty_nmcard(src.sizes().vec(), src.dtype(), device);

    // In emulator mode: both are host RAM, just memcpy
    std::memcpy(dst.data_ptr(), src.data_ptr(), src.nbytes());

    // Preserve autograd metadata
    auto* src_meta = src.autograd_meta();
    if (src_meta && src_meta->requires_grad_) {
        dst.set_requires_grad(true);
        auto* dst_meta = dst.autograd_meta();
        if (dst_meta) {
            dst_meta->is_leaf_ = src_meta->is_leaf_;
            dst_meta->output_nr_ = src_meta->output_nr_;
            dst_meta->retains_grad_ = src_meta->retains_grad_;
        }
    }

    return dst;
}

// Copy tensor from NMCard to CPU
inline Tensor nmcard_to_cpu(const Tensor& src) {
    if (src.is_cpu()) {
        return src;
    }

    auto dst = empty(src.sizes().vec(), TensorOptions().dtype(src.dtype()));

    // Emulator: both host RAM, just memcpy
    std::memcpy(dst.data_ptr(), src.data_ptr(), src.nbytes());

    return dst;
}

// ============================================================================
// High-Level NMCard Operations (Tensor-level wrappers)
// ============================================================================

namespace nmc_ops {

// Helper: make contiguous on nmcard
inline Tensor ensure_contiguous_nmcard(const Tensor& t) {
    if (t.is_contiguous()) return t;
    // For emulator: data is host RAM, contiguous() creates CPU tensor
    // We need to keep it on nmcard
    auto result = empty_nmcard(t.sizes().vec(), t.dtype());
    // Manual strided copy
    int64_t n = t.numel();
    const float* src = t.data_ptr<float>();
    float* dst = result.mutable_data_ptr<float>();
    auto sizes_v = t.sizes();
    auto strides_v = t.strides();
    int64_t ndim = t.dim();
    for (int64_t flat = 0; flat < n; ++flat) {
        int64_t src_offset = 0;
        int64_t remainder = flat;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t idx = remainder % sizes_v[d];
            remainder /= sizes_v[d];
            src_offset += idx * strides_v[d];
        }
        dst[flat] = src[src_offset];
    }
    return result;
}

// ---- Unary ops ----

inline Tensor neg(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_neg(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor abs(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_abs(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor sqrt(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_sqrt(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor rsqrt(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_rsqrt(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor square(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_square(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor exp(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_exp(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor log(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_log(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor log2(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_log2(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor log10(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_log10(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor sin(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_sin(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor cos(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_cos(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor tan(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_tan(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor tanh(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_tanh(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor sigmoid(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_sigmoid(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor relu(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_relu(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor silu(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_silu(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor gelu(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_gelu(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor ceil(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_ceil(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor floor(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_floor(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor round(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_round(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor sign(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_sign(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor reciprocal(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_reciprocal(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor clamp(const Tensor& input, float min_val, float max_val) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_clamp(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), min_val, max_val, ic.numel());
    return output;
}

// ---- Binary ops ----

inline Tensor add(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_add(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor add_scalar(const Tensor& a, float val) {
    Tensor ac = ensure_contiguous_nmcard(a);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_add_scalar(ac.data_ptr<float>(), val, output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor add_broadcast(const Tensor& a, const Tensor& b) {
    // [outer, inner] + [inner]
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_add_broadcast_col(ac.data_ptr<float>(), bc.data_ptr<float>(),
        output.mutable_data_ptr<float>(), ac.size(0), ac.size(1));
    return output;
}

inline Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_sub(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_mul(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor mul_scalar(const Tensor& a, float val) {
    Tensor ac = ensure_contiguous_nmcard(a);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_mul_scalar(ac.data_ptr<float>(), val, output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor mul_broadcast(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    // Determine broadcast direction
    if (b.dim() == 1 && b.size(0) == a.size(a.dim() - 1)) {
        at::nmcard_ops::launch_mul_broadcast_row(ac.data_ptr<float>(), bc.data_ptr<float>(),
            output.mutable_data_ptr<float>(), ac.numel() / bc.numel(), bc.numel());
    } else {
        at::nmcard_ops::launch_mul_broadcast_col(ac.data_ptr<float>(), bc.data_ptr<float>(),
            output.mutable_data_ptr<float>(), ac.size(0), ac.numel() / ac.size(0));
    }
    return output;
}

inline Tensor div(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_div(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor pow(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_pow(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor pow_scalar(const Tensor& a, float exp_val) {
    Tensor ac = ensure_contiguous_nmcard(a);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_pow_scalar(ac.data_ptr<float>(), exp_val, output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor maximum(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_maximum(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor minimum(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_minimum(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

// ---- Comparison ----

inline Tensor eq(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_eq(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor ne(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_ne(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor lt(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_lt(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor le(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_le(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor gt(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_gt(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

inline Tensor ge(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard(ac.sizes().vec(), ac.dtype());
    at::nmcard_ops::launch_ge(ac.data_ptr<float>(), bc.data_ptr<float>(), output.mutable_data_ptr<float>(), ac.numel());
    return output;
}

// Scalar comparisons (create temporary scalar tensor)
inline Tensor eq_scalar(const Tensor& a, float val) {
    auto b = empty_nmcard(a.sizes().vec(), a.dtype());
    at::nmcard_ops::launch_fill(b.mutable_data_ptr<float>(), val, b.numel());
    return eq(a, b);
}

inline Tensor ne_scalar(const Tensor& a, float val) {
    auto b = empty_nmcard(a.sizes().vec(), a.dtype());
    at::nmcard_ops::launch_fill(b.mutable_data_ptr<float>(), val, b.numel());
    return ne(a, b);
}

inline Tensor lt_scalar(const Tensor& a, float val) {
    auto b = empty_nmcard(a.sizes().vec(), a.dtype());
    at::nmcard_ops::launch_fill(b.mutable_data_ptr<float>(), val, b.numel());
    return lt(a, b);
}

inline Tensor le_scalar(const Tensor& a, float val) {
    auto b = empty_nmcard(a.sizes().vec(), a.dtype());
    at::nmcard_ops::launch_fill(b.mutable_data_ptr<float>(), val, b.numel());
    return le(a, b);
}

inline Tensor gt_scalar(const Tensor& a, float val) {
    auto b = empty_nmcard(a.sizes().vec(), a.dtype());
    at::nmcard_ops::launch_fill(b.mutable_data_ptr<float>(), val, b.numel());
    return gt(a, b);
}

inline Tensor ge_scalar(const Tensor& a, float val) {
    auto b = empty_nmcard(a.sizes().vec(), a.dtype());
    at::nmcard_ops::launch_fill(b.mutable_data_ptr<float>(), val, b.numel());
    return ge(a, b);
}

// ---- Reduce ----

inline Tensor sum(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard({}, ic.dtype());
    at::nmcard_ops::launch_sum(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor sum_dim(const Tensor& input, int64_t dim, bool keepdim) {
    Tensor ic = ensure_contiguous_nmcard(input);
    int64_t actual_dim = dim < 0 ? dim + ic.dim() : dim;
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < actual_dim; d++) outer *= ic.size(d);
    int64_t reduce = ic.size(actual_dim);
    for (int64_t d = actual_dim + 1; d < ic.dim(); d++) inner *= ic.size(d);

    std::vector<int64_t> out_sizes;
    for (int64_t d = 0; d < ic.dim(); d++) {
        if (d == actual_dim) {
            if (keepdim) out_sizes.push_back(1);
        } else {
            out_sizes.push_back(ic.size(d));
        }
    }
    auto output = empty_nmcard(out_sizes, ic.dtype());
    at::nmcard_ops::launch_sum_dim(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), outer, reduce, inner);
    return output;
}

inline Tensor mean(const Tensor& input) {
    Tensor s = sum(input);
    float n = static_cast<float>(input.numel());
    return mul_scalar(s, 1.0f / n);
}

inline Tensor max(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard({}, ic.dtype());
    at::nmcard_ops::launch_max(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), ic.numel());
    return output;
}

inline Tensor min(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard({}, ic.dtype());
    float min_val = ic.data_ptr<float>()[0];
    for (int64_t i = 1; i < ic.numel(); i++) {
        if (ic.data_ptr<float>()[i] < min_val) min_val = ic.data_ptr<float>()[i];
    }
    output.mutable_data_ptr<float>()[0] = min_val;
    return output;
}

inline Tensor argmax(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    // Return CPU tensor (index)
    auto output = at::empty({}, TensorOptions().dtype(c10::ScalarType::Long));
    float max_val = ic.data_ptr<float>()[0];
    int64_t max_idx = 0;
    for (int64_t i = 1; i < ic.numel(); i++) {
        if (ic.data_ptr<float>()[i] > max_val) {
            max_val = ic.data_ptr<float>()[i];
            max_idx = i;
        }
    }
    output.mutable_data_ptr<int64_t>()[0] = max_idx;
    return output;
}

inline Tensor argmin(const Tensor& input) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = at::empty({}, TensorOptions().dtype(c10::ScalarType::Long));
    float min_val = ic.data_ptr<float>()[0];
    int64_t min_idx = 0;
    for (int64_t i = 1; i < ic.numel(); i++) {
        if (ic.data_ptr<float>()[i] < min_val) {
            min_val = ic.data_ptr<float>()[i];
            min_idx = i;
        }
    }
    output.mutable_data_ptr<int64_t>()[0] = min_idx;
    return output;
}

// ---- MatMul ----

inline Tensor mm(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    int64_t M = ac.size(0), K = ac.size(1), N = bc.size(1);
    auto output = empty_nmcard({M, N}, ac.dtype());
    at::nmcard_ops::launch_matmul(ac.data_ptr<float>(), bc.data_ptr<float>(),
        output.mutable_data_ptr<float>(), M, K, N);
    return output;
}

inline Tensor mv(const Tensor& mat, const Tensor& vec) {
    Tensor mc = ensure_contiguous_nmcard(mat);
    Tensor vc = ensure_contiguous_nmcard(vec);
    int64_t M = mc.size(0), K = mc.size(1);
    auto output = empty_nmcard({M}, mc.dtype());
    // mv is matmul with N=1
    at::nmcard_ops::launch_matmul(mc.data_ptr<float>(), vc.data_ptr<float>(),
        output.mutable_data_ptr<float>(), M, K, 1);
    return output;
}

inline Tensor bmm(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    int64_t batch = ac.size(0), M = ac.size(1), K = ac.size(2), N = bc.size(2);
    auto output = empty_nmcard({batch, M, N}, ac.dtype());
    for (int64_t i = 0; i < batch; i++) {
        at::nmcard_ops::launch_matmul(
            ac.data_ptr<float>() + i * M * K,
            bc.data_ptr<float>() + i * K * N,
            output.mutable_data_ptr<float>() + i * M * N,
            M, K, N);
    }
    return output;
}

inline Tensor dot(const Tensor& a, const Tensor& b) {
    Tensor ac = ensure_contiguous_nmcard(a);
    Tensor bc = ensure_contiguous_nmcard(b);
    auto output = empty_nmcard({}, ac.dtype());
    float sum = 0.0f;
    for (int64_t i = 0; i < ac.numel(); i++) {
        sum += ac.data_ptr<float>()[i] * bc.data_ptr<float>()[i];
    }
    output.mutable_data_ptr<float>()[0] = sum;
    return output;
}

// ---- Fill/Copy ----

inline void fill_(Tensor& t, float val) {
    at::nmcard_ops::launch_fill(t.mutable_data_ptr<float>(), val, t.numel());
}

inline void copy_(Tensor& dst, const Tensor& src) {
    std::memcpy(dst.mutable_data_ptr<float>(), src.data_ptr<float>(),
        src.numel() * sizeof(float));
}

// ---- Fused ops (for optimizers) ----

inline Tensor addcmul(const Tensor& self, const Tensor& t1, const Tensor& t2, float value) {
    Tensor sc = ensure_contiguous_nmcard(self);
    Tensor t1c = ensure_contiguous_nmcard(t1);
    Tensor t2c = ensure_contiguous_nmcard(t2);
    auto output = empty_nmcard(sc.sizes().vec(), sc.dtype());
    int64_t n = sc.numel();
    const float* s = sc.data_ptr<float>();
    const float* a = t1c.data_ptr<float>();
    const float* b = t2c.data_ptr<float>();
    float* o = output.mutable_data_ptr<float>();
    for (int64_t i = 0; i < n; i++) {
        o[i] = s[i] + value * a[i] * b[i];
    }
    return output;
}

inline Tensor addcdiv(const Tensor& self, const Tensor& t1, const Tensor& t2, float value) {
    Tensor sc = ensure_contiguous_nmcard(self);
    Tensor t1c = ensure_contiguous_nmcard(t1);
    Tensor t2c = ensure_contiguous_nmcard(t2);
    auto output = empty_nmcard(sc.sizes().vec(), sc.dtype());
    int64_t n = sc.numel();
    const float* s = sc.data_ptr<float>();
    const float* a = t1c.data_ptr<float>();
    const float* b = t2c.data_ptr<float>();
    float* o = output.mutable_data_ptr<float>();
    for (int64_t i = 0; i < n; i++) {
        o[i] = s[i] + value * a[i] / b[i];
    }
    return output;
}

// ---- Leaky ReLU ----

inline Tensor leaky_relu(const Tensor& input, float alpha) {
    Tensor ic = ensure_contiguous_nmcard(input);
    auto output = empty_nmcard(ic.sizes().vec(), ic.dtype());
    at::nmcard_ops::launch_leaky_relu(ic.data_ptr<float>(), output.mutable_data_ptr<float>(), alpha, ic.numel());
    return output;
}

} // namespace nmc_ops

#endif // PT_USE_NMCARD

} // namespace at
