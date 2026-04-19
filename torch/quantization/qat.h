#pragma once

// ============================================================================
// torch/quantization/qat.h
// Quantization-Aware Training (QAT) for PromeTorch
// ----------------------------------------------------------------------------
// FakeQuantize    : simulates INT8 rounding in forward, identity gradient (STE)
// QuantizedLinear : nn::Linear wrapped with input/weight/output FakeQuantize
// prepare_qat()   : replaces nn::Linear submodules in-place with QuantizedLinear
// convert()       : freezes observers (no further scale updates)
// int8_matmul()   : real INT8 GEMM with float-scaled accumulation (inference)
//
// Quantization scheme: per-tensor symmetric INT8.
//     scale       = max(|min|, |max|) / 127
//     zero_point  = 0
//     q  = clamp(round(x/scale), -128, 127)
//     dq = q * scale
// Backward propagates the upstream gradient unchanged (Straight-Through
// Estimator), which is the standard QAT trick first described in Bengio 2013.
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/autograd/function.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

namespace torch {
namespace quantization {

using at::Tensor;
using at::TensorOptions;
using torch::autograd::variable_list;
using torch::autograd::FunctionCtx;
using torch::autograd::Function;

// ============================================================================
// Pure helper: per-tensor symmetric Q -> DQ on float32 buffers.
// ============================================================================
inline Tensor fake_quantize_qdq(const Tensor& input, float scale, int zp,
                                int qmin, int qmax) {
    Tensor x   = input.contiguous();
    Tensor out = at::empty(x.sizes(), TensorOptions().dtype(x.dtype()));
    const float* xs = x.data_ptr<float>();
    float*       os = out.mutable_data_ptr<float>();
    const int64_t n = x.numel();
    const float inv = scale > 0.f ? (1.0f / scale) : 0.f;
    for (int64_t i = 0; i < n; ++i) {
        int32_t q = static_cast<int32_t>(std::lrintf(xs[i] * inv)) + zp;
        if (q < qmin) q = qmin;
        if (q > qmax) q = qmax;
        os[i] = static_cast<float>(q - zp) * scale;
    }
    return out;
}

// ============================================================================
// FakeQuantizeFn — autograd op implementing Q->DQ with straight-through grad.
// ============================================================================
struct FakeQuantizeFn : Function<FakeQuantizeFn> {
    static variable_list forward(FunctionCtx& ctx, variable_list&& inputs) {
        // Inputs: [x, scale_t (1-elem), zp_t (1-elem), qmin_t (1-elem), qmax_t (1-elem)]
        const Tensor& x = inputs[0];
        const float   s = inputs[1].data_ptr<float>()[0];
        const int    zp = static_cast<int>(inputs[2].data_ptr<float>()[0]);
        const int  qmin = static_cast<int>(inputs[3].data_ptr<float>()[0]);
        const int  qmax = static_cast<int>(inputs[4].data_ptr<float>()[0]);
        Tensor y = fake_quantize_qdq(x, s, zp, qmin, qmax);
        // No saved tensors needed — STE gradient is identity.
        (void)ctx;
        return {y};
    }
    static variable_list backward(FunctionCtx& ctx, variable_list&& grads) {
        // Pass gradient straight through; non-x inputs are scalar config.
        (void)ctx;
        Tensor g = grads.empty() ? Tensor{} : grads[0];
        return {g, Tensor{}, Tensor{}, Tensor{}, Tensor{}};
    }
};

// Convenience wrapper.
inline Tensor fake_quantize_ste(const Tensor& x, float scale, int zp,
                                int qmin, int qmax) {
    auto mk = [](float v) {
        Tensor t = at::empty({1}, TensorOptions().dtype(c10::ScalarType::Float));
        t.mutable_data_ptr<float>()[0] = v;
        return t;
    };
    variable_list ins = { x, mk(scale), mk(static_cast<float>(zp)),
                          mk(static_cast<float>(qmin)), mk(static_cast<float>(qmax)) };
    auto outs = FakeQuantizeFn::apply(std::move(ins));
    return outs[0];
}

// ============================================================================
// FakeQuantize — module wrapper around the fake_quantize_ste op.
// ============================================================================
class FakeQuantize : public nn::Module {
public:
    explicit FakeQuantize(float scale = 1.0f / 127.0f,
                          int   zero_point = 0,
                          int   quant_min  = -128,
                          int   quant_max  =  127)
        : nn::Module("FakeQuantize")
        , scale_(scale > 0.f ? scale : 1.0f / 127.0f)
        , zero_point_(zero_point)
        , quant_min_(quant_min)
        , quant_max_(quant_max)
        , running_min_(0.0f)
        , running_max_(0.0f)
        , observed_(false)
        , frozen_(false) {}

    // --- accessors -----------------------------------------------------------
    float scale()      const { return scale_;      }
    int   zero_point() const { return zero_point_; }
    int   quant_min()  const { return quant_min_;  }
    int   quant_max()  const { return quant_max_;  }
    bool  is_frozen()  const { return frozen_;     }
    void  freeze()           { frozen_ = true;     }

    // Update running min/max + recompute symmetric scale.
    void observe(const Tensor& input) {
        if (frozen_) return;
        Tensor inp = input.contiguous();
        const float* data = inp.data_ptr<float>();
        const int64_t n   = inp.numel();
        float lo = std::numeric_limits<float>::max();
        float hi = std::numeric_limits<float>::lowest();
        for (int64_t i = 0; i < n; ++i) {
            const float v = data[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        if (observed_) {
            running_min_ = 0.9f * running_min_ + 0.1f * lo;
            running_max_ = 0.9f * running_max_ + 0.1f * hi;
        } else {
            running_min_ = lo;
            running_max_ = hi;
            observed_    = true;
        }
        const float absmax = std::max(std::fabs(running_min_),
                                      std::fabs(running_max_));
        const float new_scale = absmax > 1e-8f
            ? absmax / static_cast<float>(quant_max_)
            : 1.0f / 127.0f;
        scale_      = new_scale;
        zero_point_ = 0;  // symmetric
    }

    Tensor forward(const Tensor& input) override {
        if (this->is_training() && !frozen_) {
            observe(input);
        }
        return fake_quantize_ste(input, scale_, zero_point_,
                                 quant_min_, quant_max_);
    }

private:
    float scale_;
    int   zero_point_;
    int   quant_min_, quant_max_;
    float running_min_, running_max_;
    bool  observed_;
    bool  frozen_;
};

// ============================================================================
// QuantizedLinear — Linear layer with input/weight/output FakeQuantize.
// ============================================================================
class QuantizedLinear : public nn::Module {
public:
    QuantizedLinear(int64_t in_features, int64_t out_features, bool bias = true)
        : nn::Module("QuantizedLinear")
        , in_features_(in_features)
        , out_features_(out_features)
        , has_bias_(bias)
    {
        linear_    = std::make_shared<nn::Linear>(in_features, out_features,
                                                  bias, /*fused_relu=*/false);
        input_fq_  = std::make_shared<FakeQuantize>();
        weight_fq_ = std::make_shared<FakeQuantize>();
        output_fq_ = std::make_shared<FakeQuantize>();
        register_module("linear",    linear_);
        register_module("input_fq",  input_fq_);
        register_module("weight_fq", weight_fq_);
        register_module("output_fq", output_fq_);
    }

    static std::shared_ptr<QuantizedLinear> from_linear(
        const std::shared_ptr<nn::Linear>& src)
    {
        auto* W = src->get_parameter("weight");
        const int64_t out_f = W->data().size(0);
        const int64_t in_f  = W->data().size(1);
        const bool has_b    = (src->get_parameter("bias") != nullptr);
        auto q = std::make_shared<QuantizedLinear>(in_f, out_f, has_b);
        std::memcpy(q->linear_->get_parameter("weight")->data().mutable_data_ptr<float>(),
                    W->data().data_ptr<float>(),
                    W->data().numel() * sizeof(float));
        if (has_b) {
            auto* B = src->get_parameter("bias");
            std::memcpy(q->linear_->get_parameter("bias")->data().mutable_data_ptr<float>(),
                        B->data().data_ptr<float>(),
                        B->data().numel() * sizeof(float));
        }
        return q;
    }

    Tensor forward(const Tensor& input) override {
        // 1) Fake-quantize activations.
        Tensor x_q = input_fq_->forward(input);

        // 2) Fake-quantize the weight.  We update statistics on the live
        //    weight tensor, then forward through the underlying Linear with
        //    the dequantized weight by temporarily swapping it in.  The
        //    original is restored immediately so the optimizer always sees
        //    the unmodified parameter (gradients accumulate on it normally).
        Tensor& W      = linear_->get_parameter("weight")->data();
        Tensor  W_orig = W.clone();
        if (this->is_training()) weight_fq_->observe(W_orig);
        Tensor W_q = fake_quantize_qdq(W_orig,
                                       weight_fq_->scale(),
                                       weight_fq_->zero_point(),
                                       weight_fq_->quant_min(),
                                       weight_fq_->quant_max());
        std::memcpy(W.mutable_data_ptr<float>(), W_q.data_ptr<float>(),
                    W.numel() * sizeof(float));
        Tensor y = linear_->forward(x_q);
        std::memcpy(W.mutable_data_ptr<float>(), W_orig.data_ptr<float>(),
                    W.numel() * sizeof(float));

        // 3) Fake-quantize outputs (STE).
        return output_fq_->forward(y);
    }

    std::shared_ptr<nn::Linear>   linear()    const { return linear_;    }
    std::shared_ptr<FakeQuantize> input_fq()  const { return input_fq_;  }
    std::shared_ptr<FakeQuantize> weight_fq() const { return weight_fq_; }
    std::shared_ptr<FakeQuantize> output_fq() const { return output_fq_; }

private:
    int64_t in_features_, out_features_;
    bool    has_bias_;
    std::shared_ptr<nn::Linear>   linear_;
    std::shared_ptr<FakeQuantize> input_fq_;
    std::shared_ptr<FakeQuantize> weight_fq_;
    std::shared_ptr<FakeQuantize> output_fq_;
};

// ============================================================================
// prepare_qat — replace every nn::Linear child with QuantizedLinear, recursively.
// ============================================================================
inline void prepare_qat(nn::Module& model) {
    auto children = model.named_children();
    for (auto& [name, child] : children) {
        if (!child) continue;
        auto lin = std::dynamic_pointer_cast<nn::Linear>(child);
        if (lin) {
            auto qlin = QuantizedLinear::from_linear(lin);
            model.replace_module(name, qlin);
            continue;
        }
        prepare_qat(*child);
    }
}

// ============================================================================
// convert — freeze every FakeQuantize observer in the tree (post-training).
// ============================================================================
inline void convert(nn::Module& model) {
    for (auto m : model.modules(/*recurse=*/true)) {
        auto fq = std::dynamic_pointer_cast<FakeQuantize>(m);
        if (fq) fq->freeze();
    }
}

// ============================================================================
// int8_matmul — INT8 GEMM with int32 accumulation, scaled to float.
// ----------------------------------------------------------------------------
// a_int8: [M, K] int8     (row-major)
// b_int8: [K, N] int8     (row-major)
// returns: [M, N] float ≈ (a_scale * b_scale) * (A_i32 @ B_i32)
// ============================================================================
inline Tensor int8_matmul(const Tensor& a_int8, float a_scale,
                          const Tensor& b_int8, float b_scale) {
    PT_CHECK_MSG(a_int8.dim() == 2 && b_int8.dim() == 2, "int8_matmul: 2D only");
    PT_CHECK_MSG(a_int8.size(1) == b_int8.size(0), "int8_matmul: shape mismatch");
    const int64_t M = a_int8.size(0);
    const int64_t K = a_int8.size(1);
    const int64_t N = b_int8.size(1);

    Tensor a = a_int8.contiguous();
    Tensor b = b_int8.contiguous();
    Tensor out = at::empty({M, N}, TensorOptions().dtype(c10::ScalarType::Float));

    const int8_t* A = a.data_ptr<int8_t>();
    const int8_t* B = b.data_ptr<int8_t>();
    float*        C = out.mutable_data_ptr<float>();
    const float   s = a_scale * b_scale;

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[i * K + k])
                     * static_cast<int32_t>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<float>(acc) * s;
        }
    }
    return out;
}

}}  // namespace torch::quantization
