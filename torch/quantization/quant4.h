#pragma once

// ============================================================================
// torch/quantization/quant4.h
// 4-bit quantization for PromeTorch (INT4 and NF4).
// ----------------------------------------------------------------------------
// Block-wise: each block of `block_size` values (default 64) shares one
// scale (INT4) / absmax (NF4).  Values are packed two-per-byte (low nibble
// first).  NF4 levels are taken from the QLoRA paper (Dettmers et al, 2023)
// and chosen to minimise reconstruction error on a unit-normal source.
//
// Public surface:
//   Int4Block  quantize_int4(x, block_size);
//   Tensor     dequantize_int4(q);
//   NF4Block   quantize_nf4(x, block_size);
//   Tensor     dequantize_nf4(q);
//   FakeQuantize4bit  — QAT module (Q->DQ with STE gradient pass-through).
//   Linear4bit        — QLoRA-style linear (weight frozen in NF4/INT4).
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/autograd/function.h"
#include "torch/quantization/qat.h"  // FakeQuantizeFn, STE primitive reuse
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace torch {
namespace quantization {

using at::Tensor;
using at::TensorOptions;

// ============================================================================
// NF4 levels — 16 code points fitted to a standard normal distribution.
// Taken verbatim from QLoRA (Dettmers et al., 2023).
// ============================================================================
inline const std::array<float, 16>& nf4_levels() {
    static const std::array<float, 16> L = {
        -1.0f,
        -0.6961928009986877f,
        -0.5250730514526367f,
        -0.39491748809814453f,
        -0.28444138169288635f,
        -0.18477343022823334f,
        -0.09105003625154495f,
         0.0f,
         0.07958029955625534f,
         0.16093020141124725f,
         0.24611230194568634f,
         0.33791524171829224f,
         0.44070982933044434f,
         0.5626170039176941f,
         0.7229568362236023f,
         1.0f
    };
    return L;
}

// Nearest-code lookup via linear scan (16 entries — faster than binary search
// in practice because of branch prediction on short tables).
inline uint8_t nearest_nf4(float v) {
    const auto& L = nf4_levels();
    int best = 0;
    float bestd = std::fabs(v - L[0]);
    for (int i = 1; i < 16; ++i) {
        const float d = std::fabs(v - L[i]);
        if (d < bestd) { bestd = d; best = i; }
    }
    return static_cast<uint8_t>(best);
}

// ============================================================================
// Block structs.  `packed` holds ceil(n/2) bytes; two nibbles per byte, low
// nibble stores the even-indexed value.  `scales` / `absmax` has one float
// per block.
// ============================================================================
struct Int4Block {
    std::vector<uint8_t> packed;
    std::vector<float>   scales;
    std::vector<int64_t> shape;
    int64_t              block_size = 64;
    int64_t              numel      = 0;
};

struct NF4Block {
    std::vector<uint8_t> packed;
    std::vector<float>   absmax;
    std::vector<int64_t> shape;
    int64_t              block_size = 64;
    int64_t              numel      = 0;
};

// ---------- packing helpers (low nibble = even index) -----------------------
inline void pack_nibble(std::vector<uint8_t>& out, int64_t i, uint8_t nib) {
    const int64_t byte = i >> 1;
    if ((i & 1) == 0) {
        out[byte] = (out[byte] & 0xF0) | (nib & 0x0F);
    } else {
        out[byte] = (out[byte] & 0x0F) | static_cast<uint8_t>((nib & 0x0F) << 4);
    }
}
inline uint8_t unpack_nibble(const std::vector<uint8_t>& in, int64_t i) {
    const int64_t byte = i >> 1;
    return (i & 1) == 0 ? (in[byte] & 0x0F)
                        : static_cast<uint8_t>((in[byte] >> 4) & 0x0F);
}

// ============================================================================
// INT4 quantize / dequantize.  Symmetric, per-block, range [-8..+7] (we clamp
// to [-7..+7] so scale = absmax/7 stays symmetric — 8 encodes to -8 but is
// never produced by round(x/scale)).
// ============================================================================
inline Int4Block quantize_int4(const Tensor& x, int64_t block_size = 64) {
    Tensor xc = x.contiguous();
    const int64_t n = xc.numel();
    const float* xd = xc.data_ptr<float>();

    Int4Block q;
    q.block_size = block_size;
    q.numel      = n;
    q.shape.assign(xc.sizes().begin(), xc.sizes().end());
    q.packed.assign((n + 1) / 2, 0);

    const int64_t nblocks = (n + block_size - 1) / block_size;
    q.scales.assign(nblocks, 0.0f);

    for (int64_t b = 0; b < nblocks; ++b) {
        const int64_t i0 = b * block_size;
        const int64_t i1 = std::min(i0 + block_size, n);
        float absmax = 0.f;
        for (int64_t i = i0; i < i1; ++i) {
            const float a = std::fabs(xd[i]);
            if (a > absmax) absmax = a;
        }
        const float scale = absmax > 1e-12f ? absmax / 7.0f : 1.0f;
        q.scales[b] = scale;
        const float inv = 1.0f / scale;
        for (int64_t i = i0; i < i1; ++i) {
            int32_t qi = static_cast<int32_t>(std::lrintf(xd[i] * inv));
            if (qi < -7) qi = -7;
            if (qi >  7) qi =  7;
            // two's-complement 4-bit: store qi + 8 in unsigned form? No —
            // we use signed encoding directly: nibble = qi & 0x0F.
            pack_nibble(q.packed, i, static_cast<uint8_t>(qi & 0x0F));
        }
    }
    return q;
}

inline Tensor dequantize_int4(const Int4Block& q) {
    std::vector<int64_t> shape = q.shape;
    Tensor out = at::empty(shape, TensorOptions().dtype(c10::ScalarType::Float));
    float* od = out.mutable_data_ptr<float>();
    const int64_t n  = q.numel;
    const int64_t bs = q.block_size;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t b = i / bs;
        uint8_t nib = unpack_nibble(q.packed, i);
        int32_t s = (nib & 0x08) ? (static_cast<int32_t>(nib) | ~0x0F)   // sign-extend
                                 :  static_cast<int32_t>(nib);
        od[i] = static_cast<float>(s) * q.scales[b];
    }
    return out;
}

// ============================================================================
// NF4 quantize / dequantize.  Per block: divide by absmax, find nearest level.
// ============================================================================
inline NF4Block quantize_nf4(const Tensor& x, int64_t block_size = 64) {
    Tensor xc = x.contiguous();
    const int64_t n = xc.numel();
    const float* xd = xc.data_ptr<float>();

    NF4Block q;
    q.block_size = block_size;
    q.numel      = n;
    q.shape.assign(xc.sizes().begin(), xc.sizes().end());
    q.packed.assign((n + 1) / 2, 0);

    const int64_t nblocks = (n + block_size - 1) / block_size;
    q.absmax.assign(nblocks, 0.0f);

    for (int64_t b = 0; b < nblocks; ++b) {
        const int64_t i0 = b * block_size;
        const int64_t i1 = std::min(i0 + block_size, n);
        float absmax = 0.f;
        for (int64_t i = i0; i < i1; ++i) {
            const float a = std::fabs(xd[i]);
            if (a > absmax) absmax = a;
        }
        q.absmax[b] = absmax > 1e-12f ? absmax : 1.0f;
        const float inv = 1.0f / q.absmax[b];
        for (int64_t i = i0; i < i1; ++i) {
            const uint8_t code = nearest_nf4(xd[i] * inv);
            pack_nibble(q.packed, i, code);
        }
    }
    return q;
}

inline Tensor dequantize_nf4(const NF4Block& q) {
    std::vector<int64_t> shape = q.shape;
    Tensor out = at::empty(shape, TensorOptions().dtype(c10::ScalarType::Float));
    float* od = out.mutable_data_ptr<float>();
    const int64_t n  = q.numel;
    const int64_t bs = q.block_size;
    const auto& L = nf4_levels();
    for (int64_t i = 0; i < n; ++i) {
        const int64_t b = i / bs;
        const uint8_t code = unpack_nibble(q.packed, i);
        od[i] = L[code] * q.absmax[b];
    }
    return out;
}

// ============================================================================
// QAT: FakeQuantize4bit.  Forward = Q then DQ; backward = identity (STE).
// Re-uses FakeQuantizeFn from qat.h to splice into the autograd graph: we
// perform the Q->DQ out-of-graph, then treat the result as "quantized" input
// to a fresh STE node that passes gradients straight through to the original.
// ============================================================================
class FakeQuantize4bit : public nn::Module {
public:
    enum class Scheme { INT4, NF4 };

    FakeQuantize4bit(int64_t block_size = 64, Scheme scheme = Scheme::NF4)
        : nn::Module("FakeQuantize4bit")
        , block_size_(block_size)
        , scheme_(scheme) {}

    Scheme  scheme()     const { return scheme_; }
    int64_t block_size() const { return block_size_; }

    Tensor forward(const Tensor& x) override {
        // Compute Q->DQ reference.
        Tensor dq;
        if (scheme_ == Scheme::INT4) {
            dq = dequantize_int4(quantize_int4(x, block_size_));
        } else {
            dq = dequantize_nf4(quantize_nf4(x, block_size_));
        }
        // STE trick: y = x + (dq - x).detach()  — forward value equals dq,
        // backward gradient flows unchanged into x.
        Tensor delta = (dq - x.detach()).detach();
        return x + delta;
    }

private:
    int64_t block_size_;
    Scheme  scheme_;
};

// ============================================================================
// Linear4bit — weights stored 4-bit, dequantized on each forward.  Bias and
// gradients (through activations) remain float32.  Weight is NOT trainable;
// adapter modules (e.g. LoRA) should be layered on top (QLoRA pattern).
// ============================================================================
class Linear4bit : public nn::Module {
public:
    using Scheme = FakeQuantize4bit::Scheme;

    Linear4bit(int64_t in_features, int64_t out_features,
               int64_t block_size = 64,
               Scheme  scheme     = Scheme::NF4,
               bool    bias       = true)
        : nn::Module("Linear4bit")
        , in_features_(in_features)
        , out_features_(out_features)
        , block_size_(block_size)
        , scheme_(scheme)
        , has_bias_(bias)
    {
        // Build with a temporary Linear to get identical init, then quantize.
        nn::Linear tmp(in_features, out_features, bias, /*fused_relu=*/false);
        Tensor W = tmp.get_parameter("weight")->data().clone();
        load_weight(W);
        if (has_bias_) {
            Tensor b = tmp.get_parameter("bias")->data().clone();
            register_parameter("bias", nn::Parameter(b));
        }
    }

    // Replace the quantized weight from a float tensor (e.g. after loading
    // a pretrained checkpoint and converting in-place).
    void load_weight(const Tensor& W_float) {
        PT_CHECK_MSG(W_float.dim() == 2 &&
                     W_float.size(0) == out_features_ &&
                     W_float.size(1) == in_features_,
                     "Linear4bit::load_weight shape mismatch");
        if (scheme_ == Scheme::INT4) {
            qw_int4_ = quantize_int4(W_float, block_size_);
            has_int4_ = true; has_nf4_ = false;
        } else {
            qw_nf4_ = quantize_nf4(W_float, block_size_);
            has_nf4_ = true; has_int4_ = false;
        }
    }

    Tensor dequantized_weight() const {
        return scheme_ == Scheme::INT4 ? dequantize_int4(qw_int4_)
                                       : dequantize_nf4(qw_nf4_);
    }

    Tensor forward(const Tensor& input) override {
        Tensor W = dequantized_weight();                    // [out, in]
        Tensor y = at::native::matmul(input, W.t()).contiguous();  // (*, out)
        if (has_bias_) {
            Tensor b = get_parameter("bias")->data();
            const int64_t out = out_features_;
            const int64_t rows = y.numel() / out;
            float*       yd = y.mutable_data_ptr<float>();
            const float* bd = b.data_ptr<float>();
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t c = 0; c < out; ++c) yd[r*out + c] += bd[c];
            }
        }
        return y;
    }

    int64_t in_features()  const { return in_features_;  }
    int64_t out_features() const { return out_features_; }
    Scheme  scheme()       const { return scheme_;       }

private:
    int64_t in_features_, out_features_, block_size_;
    Scheme  scheme_;
    bool    has_bias_;
    Int4Block qw_int4_;
    NF4Block  qw_nf4_;
    bool      has_int4_ = false;
    bool      has_nf4_  = false;
};

// ============================================================================
// Self-test helper.  Returns pair<int4_rms, nf4_rms> reconstruction errors on
// a caller-supplied tensor (expected to be roughly unit-variance for a fair
// NF4 comparison).
// ============================================================================
inline std::pair<float, float> quant4_reconstruction_rms(const Tensor& x,
                                                         int64_t block_size = 64) {
    Tensor xc = x.contiguous();
    Tensor dq_int4 = dequantize_int4(quantize_int4(xc, block_size));
    Tensor dq_nf4  = dequantize_nf4 (quantize_nf4 (xc, block_size));
    const int64_t n = xc.numel();
    const float* a = xc.data_ptr<float>();
    const float* bi = dq_int4.data_ptr<float>();
    const float* bn = dq_nf4.data_ptr<float>();
    double si = 0.0, sn = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double di = static_cast<double>(a[i]) - bi[i];
        const double dn = static_cast<double>(a[i]) - bn[i];
        si += di*di; sn += dn*dn;
    }
    return { static_cast<float>(std::sqrt(si / n)),
             static_cast<float>(std::sqrt(sn / n)) };
}

}}  // namespace torch::quantization
