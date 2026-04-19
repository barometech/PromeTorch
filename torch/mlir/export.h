#pragma once
// MLIR text export for PromeTorch. Emits func/arith/linalg/tosa/math dialect
// text — pure string output, no MLIR C++ API, CPU-only, Elbrus-clean.
// Supported: Linear, ReLU, Sigmoid, Tanh, GELU, Identity, Conv2d, MaxPool2d,
// AvgPool2d, BatchNorm2d, Flatten. Small constants are inlined as
// `arith.constant dense<...>`; tensors above kInlineLimit emit a dense<0.0>
// placeholder with a comment naming an external sidecar path.

#include "torch/nn/modules/container.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/modules/conv.h"
#include "torch/nn/modules/pooling.h"
#include "torch/nn/modules/normalization.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

namespace torch {
namespace mlir {

namespace detail {

constexpr int64_t kInlineLimit = 4096;  // emit dense<...> for <= 4096 floats

inline std::string shape_str(const std::vector<int64_t>& shape, const char* dtype = "f32") {
    std::ostringstream ss;
    ss << "tensor<";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) ss << "x";
        ss << shape[i];
    }
    if (shape.empty()) ss << "f32";  // tensor<f32> for scalar
    else ss << "x" << dtype;
    ss << ">";
    return ss.str();
}

inline std::string format_float(float v) {
    std::ostringstream ss;
    ss.precision(8);
    if (v != v) { ss << "0x7FC00000"; return ss.str(); }  // NaN
    ss << v;
    std::string s = ss.str();
    // Ensure the literal contains a '.' or exponent so MLIR parses as f32.
    if (s.find('.') == std::string::npos &&
        s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos &&
        s.find("0x") == std::string::npos) {
        s += ".0";
    }
    return s;
}

// Emit `arith.constant dense<[...]> : tensor<...xf32>` for a flat float buffer.
// For multi-dim shapes we still use the flat form `dense<[v0, v1, ...]>` since
// MLIR accepts a 1-D initializer for any shape with matching numel.
inline std::string emit_dense_constant(
    const std::string& ssa_name,
    const float* data,
    int64_t numel,
    const std::vector<int64_t>& shape,
    const std::string& sidecar_hint = "")
{
    std::ostringstream ss;
    if (numel <= kInlineLimit && data != nullptr) {
        ss << "    " << ssa_name << " = arith.constant dense<[";
        for (int64_t i = 0; i < numel; ++i) {
            if (i) ss << ", ";
            ss << format_float(data[i]);
        }
        ss << "]> : " << shape_str(shape) << "\n";
    } else {
        // Large tensor — emit splat zero placeholder and document sidecar.
        ss << "    // weights stored externally: " << sidecar_hint
           << " (numel=" << numel << ")\n";
        ss << "    " << ssa_name << " = arith.constant dense<0.0> : "
           << shape_str(shape) << "\n";
    }
    return ss.str();
}

// State carried through emission: rolling SSA index + current value/shape.
struct EmitCtx {
    std::ostringstream body;
    int next_ssa = 0;
    std::string cur_val;             // %arg0 / %3 / etc.
    std::vector<int64_t> cur_shape;  // current tensor shape
    int const_id = 0;
    std::string sidecar_base;        // for external weights filename hint

    std::string fresh() {
        std::ostringstream ss;
        ss << "%" << next_ssa++;
        return ss.str();
    }
    std::string fresh_const(const std::string& tag) {
        std::ostringstream ss;
        ss << "%cst_" << tag << "_" << (const_id++);
        return ss.str();
    }
};

// ---- Op emitters ----

inline void emit_linear(EmitCtx& ctx, nn::Linear* lin) {
    int64_t in_f = lin->in_features();
    int64_t out_f = lin->out_features();
    int64_t M = 1;
    for (size_t i = 0; i + 1 < ctx.cur_shape.size(); ++i) M *= ctx.cur_shape[i];
    // Linear weight is [out_f, in_f]; linalg.matmul wants (K,N) = [in_f, out_f].
    auto* W = lin->get_parameter("weight");
    std::vector<float> W_t(static_cast<size_t>(in_f * out_f), 0.0f);
    if (W && W->defined()) {
        const float* w = W->data().template data_ptr<float>();
        for (int64_t i = 0; i < out_f; ++i)
            for (int64_t k = 0; k < in_f; ++k)
                W_t[k * out_f + i] = w[i * in_f + k];
    }
    std::string w_ssa = ctx.fresh_const("W");
    ctx.body << emit_dense_constant(
        w_ssa, W_t.data(), in_f * out_f, {in_f, out_f},
        ctx.sidecar_base + ".weight");
    std::string init_ssa = ctx.fresh();
    ctx.body << "    " << init_ssa << " = tensor.empty() : "
             << shape_str({M, out_f}) << "\n";
    std::string zero_ssa = ctx.fresh();
    ctx.body << "    " << zero_ssa << " = arith.constant 0.0 : f32\n";
    std::string fill_ssa = ctx.fresh();
    ctx.body << "    " << fill_ssa << " = linalg.fill ins(" << zero_ssa
             << " : f32) outs(" << init_ssa << " : "
             << shape_str({M, out_f}) << ") -> " << shape_str({M, out_f}) << "\n";
    std::string mm_ssa = ctx.fresh();
    ctx.body << "    " << mm_ssa << " = linalg.matmul ins("
             << ctx.cur_val << ", " << w_ssa << " : "
             << shape_str({M, in_f}) << ", " << shape_str({in_f, out_f})
             << ") outs(" << fill_ssa << " : " << shape_str({M, out_f})
             << ") -> " << shape_str({M, out_f}) << "\n";
    std::string out_ssa = mm_ssa;
    auto* B = lin->get_parameter("bias");
    if (B && B->defined()) {
        std::string b_ssa = ctx.fresh_const("B");
        ctx.body << emit_dense_constant(
            b_ssa, B->data().template data_ptr<float>(), out_f, {out_f},
            ctx.sidecar_base + ".bias");
        std::string add_ssa = ctx.fresh();
        ctx.body
            << "    " << add_ssa
            << " = linalg.generic {indexing_maps = ["
            << "affine_map<(d0, d1) -> (d0, d1)>, "
            << "affine_map<(d0, d1) -> (d1)>, "
            << "affine_map<(d0, d1) -> (d0, d1)>], "
            << "iterator_types = [\"parallel\", \"parallel\"]} "
            << "ins(" << out_ssa << ", " << b_ssa << " : "
            << shape_str({M, out_f}) << ", " << shape_str({out_f}) << ") "
            << "outs(" << out_ssa << " : " << shape_str({M, out_f}) << ") {\n"
            << "    ^bb0(%a: f32, %b: f32, %c: f32):\n"
            << "      %s = arith.addf %a, %b : f32\n"
            << "      linalg.yield %s : f32\n"
            << "    } -> " << shape_str({M, out_f}) << "\n";
        out_ssa = add_ssa;
    }

    ctx.cur_val = out_ssa;
    ctx.cur_shape = {M, out_f};
}

// Generic 1-input element-wise via linalg.generic — caller supplies body fragment.
inline void emit_elementwise(EmitCtx& ctx, const std::string& body_expr) {
    std::string out_ssa = ctx.fresh();
    std::ostringstream maps;
    std::ostringstream iters;
    for (size_t i = 0; i < ctx.cur_shape.size(); ++i) {
        if (i) { maps << ", "; iters << ", "; }
        maps << "d" << i;
        iters << "\"parallel\"";
    }
    std::string idx_map = "affine_map<(" + maps.str() + ") -> (" + maps.str() + ")>";
    ctx.body << "    " << out_ssa
             << " = linalg.generic {indexing_maps = [" << idx_map << ", " << idx_map
             << "], iterator_types = [" << iters.str() << "]} ins("
             << ctx.cur_val << " : " << shape_str(ctx.cur_shape) << ") outs("
             << ctx.cur_val << " : " << shape_str(ctx.cur_shape) << ") {\n"
             << "    ^bb0(%x: f32, %y: f32):\n"
             << "      " << body_expr << "\n"
             << "      linalg.yield %r : f32\n"
             << "    } -> " << shape_str(ctx.cur_shape) << "\n";
    ctx.cur_val = out_ssa;
}

inline void emit_relu(EmitCtx& ctx) {
    emit_elementwise(ctx,
        "%z = arith.constant 0.0 : f32\n      "
        "%r = arith.maximumf %x, %z : f32");
}
inline void emit_sigmoid(EmitCtx& ctx) {
    emit_elementwise(ctx,
        "%n = arith.negf %x : f32\n      "
        "%e = math.exp %n : f32\n      "
        "%o = arith.constant 1.0 : f32\n      "
        "%d = arith.addf %o, %e : f32\n      "
        "%r = arith.divf %o, %d : f32");
}
inline void emit_tanh(EmitCtx& ctx) {
    emit_elementwise(ctx, "%r = math.tanh %x : f32");
}
inline void emit_gelu(EmitCtx& ctx) {
    // Approximate: x * sigmoid(1.702 * x)
    emit_elementwise(ctx,
        "%k = arith.constant 1.702 : f32\n      "
        "%t = arith.mulf %x, %k : f32\n      "
        "%n = arith.negf %t : f32\n      "
        "%e = math.exp %n : f32\n      "
        "%o = arith.constant 1.0 : f32\n      "
        "%d = arith.addf %o, %e : f32\n      "
        "%s = arith.divf %o, %d : f32\n      "
        "%r = arith.mulf %x, %s : f32");
}

inline void emit_conv2d(EmitCtx& ctx, nn::Conv2d* c) {
    if (ctx.cur_shape.size() != 4) {
        ctx.body << "    // skipping Conv2d: input is not 4D\n";
        return;
    }
    int64_t N = ctx.cur_shape[0], Cin = ctx.cur_shape[1];
    int64_t H = ctx.cur_shape[2], W = ctx.cur_shape[3];
    int64_t Cout = c->out_channels();
    auto ks = c->kernel_size(); auto st = c->stride();
    auto pd = c->padding();     auto dl = c->dilation();
    int64_t Hout = (H + 2 * pd[0] - dl[0] * (ks[0] - 1) - 1) / st[0] + 1;
    int64_t Wout = (W + 2 * pd[1] - dl[1] * (ks[1] - 1) - 1) / st[1] + 1;
    auto* Wp = c->get_parameter("weight");  // [Cout, Cin, kH, kW]
    int64_t wnumel = Cout * Cin * ks[0] * ks[1];
    std::string w_ssa = ctx.fresh_const("CW");
    ctx.body << emit_dense_constant(
        w_ssa, Wp ? Wp->data().template data_ptr<float>() : nullptr, wnumel,
        {Cout, Cin, ks[0], ks[1]}, ctx.sidecar_base + ".conv_weight");
    std::string init_ssa = ctx.fresh();
    ctx.body << "    " << init_ssa << " = tensor.empty() : "
             << shape_str({N, Cout, Hout, Wout}) << "\n";
    std::string zero_ssa = ctx.fresh();
    ctx.body << "    " << zero_ssa << " = arith.constant 0.0 : f32\n";
    std::string fill_ssa = ctx.fresh();
    ctx.body << "    " << fill_ssa << " = linalg.fill ins(" << zero_ssa
             << " : f32) outs(" << init_ssa << " : "
             << shape_str({N, Cout, Hout, Wout}) << ") -> "
             << shape_str({N, Cout, Hout, Wout}) << "\n";
    std::string conv_ssa = ctx.fresh();
    ctx.body << "    " << conv_ssa << " = linalg.conv_2d_nchw_fchw "
             << "{strides = dense<[" << st[0] << ", " << st[1] << "]> : tensor<2xi64>, "
             << "dilations = dense<[" << dl[0] << ", " << dl[1] << "]> : tensor<2xi64>} "
             << "ins(" << ctx.cur_val << ", " << w_ssa << " : "
             << shape_str({N, Cin, H, W}) << ", " << shape_str({Cout, Cin, ks[0], ks[1]})
             << ") outs(" << fill_ssa << " : " << shape_str({N, Cout, Hout, Wout})
             << ") -> " << shape_str({N, Cout, Hout, Wout}) << "\n";
    std::string out_ssa = conv_ssa;
    auto* Bp = c->get_parameter("bias");
    if (Bp && Bp->defined()) {
        std::string b_ssa = ctx.fresh_const("CB");
        ctx.body << emit_dense_constant(
            b_ssa, Bp->data().template data_ptr<float>(), Cout, {Cout},
            ctx.sidecar_base + ".conv_bias");
        std::string add_ssa = ctx.fresh();
        ctx.body
            << "    " << add_ssa
            << " = linalg.generic {indexing_maps = ["
            << "affine_map<(n, c, h, w) -> (n, c, h, w)>, "
            << "affine_map<(n, c, h, w) -> (c)>, "
            << "affine_map<(n, c, h, w) -> (n, c, h, w)>], "
            << "iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]} "
            << "ins(" << out_ssa << ", " << b_ssa << " : "
            << shape_str({N, Cout, Hout, Wout}) << ", " << shape_str({Cout}) << ") "
            << "outs(" << out_ssa << " : " << shape_str({N, Cout, Hout, Wout}) << ") {\n"
            << "    ^bb0(%a: f32, %b: f32, %c: f32):\n"
            << "      %s = arith.addf %a, %b : f32\n"
            << "      linalg.yield %s : f32\n"
            << "    } -> " << shape_str({N, Cout, Hout, Wout}) << "\n";
        out_ssa = add_ssa;
    }
    ctx.cur_val = out_ssa;
    ctx.cur_shape = {N, Cout, Hout, Wout};
}

template <bool IsMax>
inline void emit_pool2d(EmitCtx& ctx,
                        const std::array<int64_t, 2>& ks,
                        const std::array<int64_t, 2>& st)
{
    if (ctx.cur_shape.size() != 4) {
        ctx.body << "    // skipping Pool2d: input not 4D\n";
        return;
    }
    int64_t N = ctx.cur_shape[0], C = ctx.cur_shape[1];
    int64_t H = ctx.cur_shape[2], W = ctx.cur_shape[3];
    int64_t Hout = (H - ks[0]) / st[0] + 1;
    int64_t Wout = (W - ks[1]) / st[1] + 1;
    std::string ks_ssa = ctx.fresh();
    ctx.body << "    " << ks_ssa << " = tensor.empty() : "
             << shape_str({ks[0], ks[1]}) << "\n";
    std::string init_ssa = ctx.fresh();
    ctx.body << "    " << init_ssa << " = tensor.empty() : "
             << shape_str({N, C, Hout, Wout}) << "\n";
    std::string seed_ssa = ctx.fresh();
    if (IsMax) {
        ctx.body << "    " << seed_ssa
                 << " = arith.constant 0xFF800000 : f32  // -inf\n";
    } else {
        ctx.body << "    " << seed_ssa << " = arith.constant 0.0 : f32\n";
    }
    std::string fill_ssa = ctx.fresh();
    ctx.body << "    " << fill_ssa << " = linalg.fill ins(" << seed_ssa
             << " : f32) outs(" << init_ssa << " : "
             << shape_str({N, C, Hout, Wout}) << ") -> "
             << shape_str({N, C, Hout, Wout}) << "\n";
    std::string pool_ssa = ctx.fresh();
    const char* op = IsMax ? "linalg.pooling_nchw_max" : "linalg.pooling_nchw_sum";
    ctx.body << "    " << pool_ssa << " = " << op
             << " {strides = dense<[" << st[0] << ", " << st[1] << "]> : tensor<2xi64>, "
             << "dilations = dense<[1, 1]> : tensor<2xi64>} ins("
             << ctx.cur_val << ", " << ks_ssa << " : "
             << shape_str({N, C, H, W}) << ", " << shape_str({ks[0], ks[1]})
             << ") outs(" << fill_ssa << " : " << shape_str({N, C, Hout, Wout})
             << ") -> " << shape_str({N, C, Hout, Wout}) << "\n";
    std::string out_ssa = pool_ssa;
    if (!IsMax) {
        // Convert sum -> average: divide by ks[0]*ks[1].
        float divisor = static_cast<float>(ks[0] * ks[1]);
        std::string div_ssa = ctx.fresh();
        ctx.body << "    " << div_ssa
                 << " = arith.constant " << format_float(divisor) << " : f32\n";
        std::string avg_ssa = ctx.fresh();
        std::ostringstream maps, iters;
        for (int i = 0; i < 4; ++i) {
            if (i) { maps << ", "; iters << ", "; }
            maps << "d" << i;
            iters << "\"parallel\"";
        }
        std::string idx_map = "affine_map<(" + maps.str() + ") -> (" + maps.str() + ")>";
        ctx.body
            << "    " << avg_ssa
            << " = linalg.generic {indexing_maps = [" << idx_map << ", " << idx_map
            << "], iterator_types = [" << iters.str() << "]} ins("
            << out_ssa << " : " << shape_str({N, C, Hout, Wout}) << ") outs("
            << out_ssa << " : " << shape_str({N, C, Hout, Wout}) << ") {\n"
            << "    ^bb0(%x: f32, %y: f32):\n"
            << "      %r = arith.divf %x, " << div_ssa << " : f32\n"
            << "      linalg.yield %r : f32\n"
            << "    } -> " << shape_str({N, C, Hout, Wout}) << "\n";
        out_ssa = avg_ssa;
    }
    ctx.cur_val = out_ssa;
    ctx.cur_shape = {N, C, Hout, Wout};
}

inline void emit_batchnorm2d(EmitCtx& ctx, nn::BatchNorm2d* bn) {
    if (ctx.cur_shape.size() != 4) {
        ctx.body << "    // skipping BatchNorm2d: input not 4D\n";
        return;
    }
    int64_t N = ctx.cur_shape[0], C = ctx.cur_shape[1];
    int64_t H = ctx.cur_shape[2], W = ctx.cur_shape[3];
    auto emit_param = [&](const char* name, const char* tag, bool ones_default) {
        std::vector<float> tmp(static_cast<size_t>(C),
                               ones_default ? 1.0f : 0.0f);
        const float* src = nullptr;
        auto* p = bn->get_parameter(name);
        if (p && p->defined()) src = p->data().template data_ptr<float>();
        else {
            auto* b = bn->get_buffer(name);
            if (b && b->data().defined()) src = b->data().template data_ptr<float>();
        }
        std::string ssa = ctx.fresh_const(tag);
        ctx.body << emit_dense_constant(
            ssa, src ? src : tmp.data(), C, {C},
            ctx.sidecar_base + "." + name);
        return ssa;
    };
    std::string g = emit_param("weight", "Bg", true);
    std::string b = emit_param("bias",   "Bb", false);
    std::string m = emit_param("running_mean", "Bm", false);
    std::string v = emit_param("running_var",  "Bv", true);
    float eps = static_cast<float>(bn->eps());

    std::string out_ssa = ctx.fresh();
    ctx.body
        << "    " << out_ssa
        << " = linalg.generic {indexing_maps = ["
        << "affine_map<(n, c, h, w) -> (n, c, h, w)>, "
        << "affine_map<(n, c, h, w) -> (c)>, "
        << "affine_map<(n, c, h, w) -> (c)>, "
        << "affine_map<(n, c, h, w) -> (c)>, "
        << "affine_map<(n, c, h, w) -> (c)>, "
        << "affine_map<(n, c, h, w) -> (n, c, h, w)>], "
        << "iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]} "
        << "ins(" << ctx.cur_val << ", " << g << ", " << b << ", " << m << ", " << v
        << " : " << shape_str({N, C, H, W}) << ", " << shape_str({C}) << ", "
        << shape_str({C}) << ", " << shape_str({C}) << ", " << shape_str({C}) << ") "
        << "outs(" << ctx.cur_val << " : " << shape_str({N, C, H, W}) << ") {\n"
        << "    ^bb0(%x: f32, %gv: f32, %bv: f32, %mv: f32, %vv: f32, %z: f32):\n"
        << "      %eps = arith.constant " << format_float(eps) << " : f32\n"
        << "      %ve = arith.addf %vv, %eps : f32\n"
        << "      %sd = math.sqrt %ve : f32\n"
        << "      %d  = arith.subf %x, %mv : f32\n"
        << "      %nm = arith.divf %d, %sd : f32\n"
        << "      %sc = arith.mulf %nm, %gv : f32\n"
        << "      %r  = arith.addf %sc, %bv : f32\n"
        << "      linalg.yield %r : f32\n"
        << "    } -> " << shape_str({N, C, H, W}) << "\n";
    ctx.cur_val = out_ssa;
}

inline void emit_flatten(EmitCtx& ctx, nn::Flatten* /*f*/) {
    if (ctx.cur_shape.size() < 2) return;
    int64_t M = ctx.cur_shape[0];
    int64_t K = 1;
    for (size_t i = 1; i < ctx.cur_shape.size(); ++i) K *= ctx.cur_shape[i];
    std::string out_ssa = ctx.fresh();
    ctx.body << "    " << out_ssa << " = tensor.collapse_shape " << ctx.cur_val
             << " [[0], [";
    for (size_t i = 1; i < ctx.cur_shape.size(); ++i) {
        if (i > 1) ctx.body << ", ";
        ctx.body << i;
    }
    ctx.body << "]] : " << shape_str(ctx.cur_shape)
             << " into " << shape_str({static_cast<int64_t>(M), static_cast<int64_t>(K)}) << "\n";
    ctx.cur_val = out_ssa;
    ctx.cur_shape = {M, K};
}

// Dispatch one module by RTTI.
inline void emit_module(EmitCtx& ctx, nn::Module* m) {
    if (auto* x = dynamic_cast<nn::Linear*>(m))      { emit_linear(ctx, x); return; }
    if (dynamic_cast<nn::ReLU*>(m))                  { emit_relu(ctx); return; }
    if (dynamic_cast<nn::Sigmoid*>(m))               { emit_sigmoid(ctx); return; }
    if (dynamic_cast<nn::Tanh*>(m))                  { emit_tanh(ctx); return; }
    if (dynamic_cast<nn::GELU*>(m))                  { emit_gelu(ctx); return; }
    if (dynamic_cast<nn::Identity*>(m))              { return; }
    if (auto* x = dynamic_cast<nn::Conv2d*>(m))      { emit_conv2d(ctx, x); return; }
    if (auto* x = dynamic_cast<nn::MaxPool2d*>(m))   {
        emit_pool2d<true>(ctx, x->kernel_size(), x->stride()); return;
    }
    if (auto* x = dynamic_cast<nn::AvgPool2d*>(m))   {
        emit_pool2d<false>(ctx, x->kernel_size(), x->stride()); return;
    }
    if (auto* x = dynamic_cast<nn::BatchNorm2d*>(m)) { emit_batchnorm2d(ctx, x); return; }
    if (auto* x = dynamic_cast<nn::Flatten*>(m))     { emit_flatten(ctx, x); return; }

    ctx.body << "    // unsupported op: " << m->name() << "\n";
}

} // namespace detail

// =============================================================================
// Public API
// =============================================================================
//
// Serialize a Sequential to MLIR text. Returns the file content.
// If `path` is non-empty, also writes the content to that path.
//
inline std::string export_mlir(const nn::Sequential& model_const,
                               const std::vector<int64_t>& input_shape,
                               const std::string& path = "")
{
    auto& model = const_cast<nn::Sequential&>(model_const);

    detail::EmitCtx ctx;
    ctx.cur_val = "%arg0";
    ctx.cur_shape = input_shape;
    ctx.next_ssa = 1;  // %0 reserved (unused), start fresh at %1
    ctx.sidecar_base = path.empty() ? "model" : path + ".weights";
    auto children = model.named_children();
    for (auto& [child_name, child] : children) {
        ctx.body << "    // " << child_name << ": " << child->name() << "\n";
        detail::emit_module(ctx, child.get());
    }
    std::ostringstream out;
    out << "// MLIR generated by PromeTorch torch::mlir::export_mlir\n";
    out << "module {\n";
    out << "  func.func @forward(%arg0: " << detail::shape_str(input_shape)
        << ") -> " << detail::shape_str(ctx.cur_shape) << " {\n";
    out << ctx.body.str();
    out << "    return " << ctx.cur_val << " : "
        << detail::shape_str(ctx.cur_shape) << "\n";
    out << "  }\n";
    out << "}\n";
    std::string text = out.str();
    if (!path.empty()) {
        std::ofstream f(path);
        f << text;
    }
    return text;
}

}} // namespace torch::mlir
