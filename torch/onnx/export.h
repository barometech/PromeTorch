#pragma once

// ============================================================================
// PromeTorch ONNX Export (header-only, zero deps, CPU/Elbrus-safe)
// ----------------------------------------------------------------------------
// Manually emits ONNX protobuf wire-format bytes — no protobuf library used.
//
// Spec references:
//   https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
//   https://github.com/onnx/onnx/blob/main/docs/IR.md
//   https://protobuf.dev/programming-guides/encoding/
//
// Protobuf wire encoding cheat-sheet:
//   tag         = (field_number << 3) | wire_type
//   wire types  : 0 = varint   (int32/int64/uint64/bool/enum)
//                 1 = fixed64  (double)
//                 2 = length-delimited (string/bytes/sub-message/packed)
//                 5 = fixed32  (float)
//   varint      : little-endian base-128, MSB=1 means "more bytes follow"
//   sub-message : write message body to a temp buffer, prepend its length as varint
//   packed repeated : same as sub-message but body is concatenated raw values
//
// ONNX message field-number cheat-sheet (subset we care about):
//
//   ModelProto {
//     1  int64       ir_version
//     2  string      producer_name
//     3  string      producer_version
//     4  string      domain
//     5  int64       model_version
//     6  string      doc_string
//     7  GraphProto  graph
//     8  OperatorSetIdProto[]  opset_import
//   }
//   OperatorSetIdProto { 1 string domain;  2 int64 version }
//   GraphProto {
//     1  NodeProto[]    node
//     2  string         name
//     5  TensorProto[]  initializer
//     10 string         doc_string
//     11 ValueInfoProto[] input
//     12 ValueInfoProto[] output
//   }
//   NodeProto {
//     1 string[] input
//     2 string[] output
//     3 string   name
//     4 string   op_type
//     5 AttributeProto[] attribute
//     7 string   domain
//   }
//   AttributeProto {
//     1  string name
//     20 AttributeType type   (1=FLOAT 2=INT 3=STRING 6=FLOATS 7=INTS 8=STRINGS)
//     2  float  f
//     3  int64  i
//     4  bytes  s
//     7  float[] floats   (packed)
//     8  int64[] ints     (packed)
//     9  bytes[]  strings
//   }
//   ValueInfoProto { 1 string name;  2 TypeProto type;  3 string doc_string }
//   TypeProto      { 1 Tensor tensor_type }
//   Tensor (TypeProto.Tensor) { 1 int32 elem_type;  2 TensorShapeProto shape }
//   TensorShapeProto { 1 Dimension[] dim }
//   Dimension      { 1 int64 dim_value;  2 string dim_param }
//   TensorProto    {
//     1  int64[] dims        (packed)
//     2  int32   data_type   (1=FLOAT 7=INT64)
//     8  string  name
//     9  bytes   raw_data
//   }
//
// Supported PromeTorch nn modules:
//   Linear         -> Gemm                (preferred — handles 2D in & bias)
//   Conv2d         -> Conv
//   ReLU           -> Relu
//   Sigmoid        -> Sigmoid
//   Tanh           -> Tanh
//   Softmax        -> Softmax
//   BatchNorm2d    -> BatchNormalization
//   MaxPool2d      -> MaxPool
//   AvgPool2d      -> AveragePool
//   Flatten        -> Flatten
//   Identity       -> Identity
//   Sequential     -> walked recursively
// ============================================================================

#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/modules/conv.h"
#include "torch/nn/modules/pooling.h"
#include "torch/nn/modules/normalization.h"
#include "torch/nn/modules/container.h"
#include "aten/src/ATen/core/Tensor.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <typeinfo>

namespace torch {
namespace onnx {

using nn::Module;
using nn::ModulePtr;
using at::Tensor;

// ============================================================================
// Wire-format primitives
// ============================================================================
namespace detail {

inline void write_varint(std::string& buf, uint64_t v) {
    while (v >= 0x80) {
        buf.push_back(static_cast<char>((v & 0x7f) | 0x80));
        v >>= 7;
    }
    buf.push_back(static_cast<char>(v));
}

inline void write_tag(std::string& buf, uint32_t field_number, uint32_t wire_type) {
    write_varint(buf, (uint64_t(field_number) << 3) | wire_type);
}

// Wire types
constexpr uint32_t WT_VARINT = 0;
constexpr uint32_t WT_FIXED64 = 1;
constexpr uint32_t WT_LENGTH = 2;
constexpr uint32_t WT_FIXED32 = 5;

inline void write_int64(std::string& buf, uint32_t fn, int64_t v) {
    write_tag(buf, fn, WT_VARINT);
    // protobuf encodes signed int64 with two-complement -> reinterpret cast.
    write_varint(buf, static_cast<uint64_t>(v));
}

inline void write_int32(std::string& buf, uint32_t fn, int32_t v) {
    write_tag(buf, fn, WT_VARINT);
    write_varint(buf, static_cast<uint64_t>(static_cast<int64_t>(v)));
}

inline void write_float(std::string& buf, uint32_t fn, float v) {
    write_tag(buf, fn, WT_FIXED32);
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(float));
    for (int i = 0; i < 4; ++i) {
        buf.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
    }
}

inline void write_string(std::string& buf, uint32_t fn, const std::string& s) {
    write_tag(buf, fn, WT_LENGTH);
    write_varint(buf, s.size());
    buf.append(s);
}

inline void write_bytes(std::string& buf, uint32_t fn, const void* data, size_t n) {
    write_tag(buf, fn, WT_LENGTH);
    write_varint(buf, n);
    buf.append(static_cast<const char*>(data), n);
}

// Write a length-delimited submessage by prepending its varint length.
inline void write_submsg(std::string& buf, uint32_t fn, const std::string& msg) {
    write_tag(buf, fn, WT_LENGTH);
    write_varint(buf, msg.size());
    buf.append(msg);
}

// Packed repeated int64 (TensorProto.dims, AttributeProto.ints, etc).
inline void write_packed_int64(std::string& buf, uint32_t fn, const std::vector<int64_t>& vs) {
    std::string body;
    for (int64_t v : vs) write_varint(body, static_cast<uint64_t>(v));
    write_submsg(buf, fn, body);
}

// ============================================================================
// AttributeProto helpers
// ============================================================================
//   AttributeType enum: UNDEFINED=0, FLOAT=1, INT=2, STRING=3, TENSOR=4,
//                       GRAPH=5, FLOATS=6, INTS=7, STRINGS=8

inline std::string make_attr_int(const std::string& name, int64_t v) {
    std::string a;
    write_string(a, 1, name);          // name
    write_int32(a, 20, 2);             // type = INT
    write_int64(a, 3, v);              // i
    return a;
}

inline std::string make_attr_ints(const std::string& name, const std::vector<int64_t>& vs) {
    std::string a;
    write_string(a, 1, name);
    write_int32(a, 20, 7);             // type = INTS
    write_packed_int64(a, 8, vs);
    return a;
}

inline std::string make_attr_float(const std::string& name, float v) {
    std::string a;
    write_string(a, 1, name);
    write_int32(a, 20, 1);             // type = FLOAT
    write_float(a, 2, v);
    return a;
}

inline std::string make_attr_string(const std::string& name, const std::string& v) {
    std::string a;
    write_string(a, 1, name);
    write_int32(a, 20, 3);             // type = STRING
    write_bytes(a, 4, v.data(), v.size());
    return a;
}

// ============================================================================
// TensorProto for an initializer (FLOAT, raw_data little-endian)
// ============================================================================
inline std::string make_tensor_proto(const std::string& name, const Tensor& t) {
    std::string body;
    Tensor c = t.is_contiguous() ? t : t.contiguous();
    std::vector<int64_t> dims;
    for (int64_t d = 0; d < c.dim(); ++d) dims.push_back(c.size(d));
    write_packed_int64(body, 1, dims);              // dims (packed)
    write_int32(body, 2, 1);                         // data_type = FLOAT
    write_string(body, 8, name);                     // name
    const float* data = c.data_ptr<float>();
    write_bytes(body, 9, data,
                static_cast<size_t>(c.numel()) * sizeof(float));   // raw_data
    return body;
}

// ============================================================================
// ValueInfoProto with explicit shape (FLOAT)
// ============================================================================
inline std::string make_value_info_proto(const std::string& name,
                                          const std::vector<int64_t>& shape) {
    // TensorShapeProto { Dimension[] dim }
    std::string shape_msg;
    for (int64_t d : shape) {
        std::string dim;
        write_int64(dim, 1, d);                      // dim_value
        write_submsg(shape_msg, 1, dim);             // dim
    }
    // Tensor (TypeProto.Tensor) { 1 elem_type, 2 shape }
    std::string tensor_type;
    write_int32(tensor_type, 1, 1);                  // elem_type = FLOAT
    write_submsg(tensor_type, 2, shape_msg);         // shape
    // TypeProto { 1 tensor_type }
    std::string type_proto;
    write_submsg(type_proto, 1, tensor_type);
    // ValueInfoProto { 1 name, 2 type }
    std::string vi;
    write_string(vi, 1, name);
    write_submsg(vi, 2, type_proto);
    return vi;
}

// ============================================================================
// NodeProto
// ============================================================================
inline std::string make_node(const std::vector<std::string>& inputs,
                              const std::vector<std::string>& outputs,
                              const std::string& op_type,
                              const std::string& name,
                              const std::vector<std::string>& attributes = {}) {
    std::string n;
    for (auto& s : inputs)  write_string(n, 1, s);
    for (auto& s : outputs) write_string(n, 2, s);
    write_string(n, 3, name);
    write_string(n, 4, op_type);
    for (auto& a : attributes) write_submsg(n, 5, a);
    return n;
}

// ============================================================================
// Dynamic-cast helper: returns nullptr if module is not the requested type.
// ============================================================================
template <typename T>
inline T* as(Module* m) { return dynamic_cast<T*>(m); }

// ============================================================================
// Compute output shape of standard 2D conv/pool (PyTorch formula).
// ============================================================================
inline std::vector<int64_t> conv2d_out_shape(const std::vector<int64_t>& in,
                                              int64_t out_channels,
                                              std::array<int64_t, 2> k,
                                              std::array<int64_t, 2> s,
                                              std::array<int64_t, 2> p,
                                              std::array<int64_t, 2> d) {
    int64_t H = in[2], W = in[3];
    int64_t Ho = (H + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1;
    int64_t Wo = (W + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1;
    return {in[0], out_channels, Ho, Wo};
}

inline std::vector<int64_t> pool2d_out_shape(const std::vector<int64_t>& in,
                                              std::array<int64_t, 2> k,
                                              std::array<int64_t, 2> s,
                                              std::array<int64_t, 2> p,
                                              std::array<int64_t, 2> d = {1,1}) {
    int64_t H = in[2], W = in[3];
    int64_t Ho = (H + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1;
    int64_t Wo = (W + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1;
    return {in[0], in[1], Ho, Wo};
}

// ============================================================================
// Tracer state — mutated as we walk the module graph.
// ============================================================================
struct ExportContext {
    std::string current_value;            // name of tensor currently flowing
    std::vector<int64_t> current_shape;   // its shape
    std::vector<std::string> initializers;  // serialized TensorProto bodies
    std::vector<std::string> nodes;         // serialized NodeProto bodies
    int unique = 0;

    std::string fresh(const std::string& hint) {
        return hint + "_" + std::to_string(unique++);
    }
};

// ============================================================================
// Add an initializer (parameter / buffer) to the graph.
// ============================================================================
inline std::string add_initializer(ExportContext& ctx,
                                    const std::string& base,
                                    const Tensor& t) {
    std::string name = ctx.fresh(base);
    ctx.initializers.push_back(make_tensor_proto(name, t));
    return name;
}

// ============================================================================
// Module dispatch: emit nodes for a single leaf module.
// Returns false if the module type is unsupported.
// ============================================================================
inline bool emit_module(ExportContext& ctx, Module* m) {
    // ---------------- Identity ----------------
    if (as<nn::Identity>(m)) {
        std::string out = ctx.fresh("identity");
        ctx.nodes.push_back(make_node({ctx.current_value}, {out}, "Identity",
                                       ctx.fresh("Identity")));
        ctx.current_value = out;
        return true;
    }

    // ---------------- Linear -> Gemm ----------------
    if (auto* lin = as<nn::Linear>(m)) {
        // ONNX Gemm: Y = alpha * (A or A^T) * (B or B^T) + beta * C
        // Linear stores W as [out, in]; PromeTorch computes y = x @ W^T + b
        // We pass W with transB=1 so ONNX computes x @ W^T as well.
        std::string W_name = add_initializer(ctx, "linear_W",
                                              lin->get_parameter("weight")->data());
        std::vector<std::string> ins = {ctx.current_value, W_name};
        std::vector<std::string> attrs = {
            detail::make_attr_float("alpha", 1.0f),
            detail::make_attr_float("beta", 1.0f),
            detail::make_attr_int("transA", 0),
            detail::make_attr_int("transB", 1),
        };
        if (lin->get_parameter("bias")) {
            std::string B_name = add_initializer(ctx, "linear_b",
                                                  lin->get_parameter("bias")->data());
            ins.push_back(B_name);
        } else {
            // Gemm requires C; if no bias, omit transA/transB attrs that need it.
            // Actually Gemm makes C optional in opset 11+. Leave inputs as-is.
        }
        std::string out = ctx.fresh("gemm");
        ctx.nodes.push_back(make_node(ins, {out}, "Gemm",
                                       ctx.fresh("Gemm"), attrs));
        // shape: flatten leading dims, then [..., out_features]
        std::vector<int64_t> outs = ctx.current_shape;
        if (!outs.empty()) outs.back() = lin->out_features();
        else outs = {lin->out_features()};
        ctx.current_value = out;
        ctx.current_shape = outs;
        return true;
    }

    // ---------------- Conv2d -> Conv ----------------
    if (auto* c = as<nn::Conv2d>(m)) {
        std::string W_name = add_initializer(ctx, "conv_W",
                                              c->get_parameter("weight")->data());
        std::vector<std::string> ins = {ctx.current_value, W_name};
        if (c->has_bias() && c->get_parameter("bias")) {
            std::string B_name = add_initializer(ctx, "conv_b",
                                                  c->get_parameter("bias")->data());
            ins.push_back(B_name);
        }
        auto k = c->kernel_size();
        auto s = c->stride();
        auto p = c->padding();
        auto d = c->dilation();
        std::vector<std::string> attrs = {
            detail::make_attr_ints("kernel_shape", {k[0], k[1]}),
            detail::make_attr_ints("strides",     {s[0], s[1]}),
            // ONNX `pads` = [pad_top, pad_left, pad_bottom, pad_right]
            detail::make_attr_ints("pads",        {p[0], p[1], p[0], p[1]}),
            detail::make_attr_ints("dilations",   {d[0], d[1]}),
            detail::make_attr_int("group",         c->groups()),
        };
        std::string out = ctx.fresh("conv");
        ctx.nodes.push_back(make_node(ins, {out}, "Conv",
                                       ctx.fresh("Conv"), attrs));
        ctx.current_shape = conv2d_out_shape(ctx.current_shape,
                                              c->out_channels(), k, s, p, d);
        ctx.current_value = out;
        return true;
    }

    // ---------------- BatchNorm2d -> BatchNormalization ----------------
    if (auto* bn = as<nn::BatchNorm2d>(m)) {
        // BatchNormalization inputs: X, scale, B, mean, var
        Tensor scale, bias, mean, var;
        if (bn->affine() && bn->get_parameter("weight")) {
            scale = bn->get_parameter("weight")->data();
            bias  = bn->get_parameter("bias")->data();
        } else {
            scale = at::ones({bn->num_features()});
            bias  = at::zeros({bn->num_features()});
        }
        if (bn->track_running_stats() && bn->get_buffer("running_mean")) {
            mean = bn->get_buffer("running_mean")->data();
            var  = bn->get_buffer("running_var")->data();
        } else {
            mean = at::zeros({bn->num_features()});
            var  = at::ones({bn->num_features()});
        }
        std::string s_name = add_initializer(ctx, "bn_scale", scale);
        std::string b_name = add_initializer(ctx, "bn_bias",  bias);
        std::string m_name = add_initializer(ctx, "bn_mean",  mean);
        std::string v_name = add_initializer(ctx, "bn_var",   var);
        std::vector<std::string> attrs = {
            detail::make_attr_float("epsilon", static_cast<float>(bn->eps())),
            detail::make_attr_float("momentum", static_cast<float>(1.0 - bn->momentum())),
        };
        std::string out = ctx.fresh("bn");
        ctx.nodes.push_back(make_node(
            {ctx.current_value, s_name, b_name, m_name, v_name},
            {out}, "BatchNormalization",
            ctx.fresh("BatchNormalization"), attrs));
        ctx.current_value = out;
        // shape unchanged
        return true;
    }

    // ---------------- MaxPool2d -> MaxPool ----------------
    if (auto* mp = as<nn::MaxPool2d>(m)) {
        auto k = mp->kernel_size();
        auto s = mp->stride();
        auto p = mp->padding();
        auto d = mp->dilation();
        std::vector<std::string> attrs = {
            detail::make_attr_ints("kernel_shape", {k[0], k[1]}),
            detail::make_attr_ints("strides",     {s[0], s[1]}),
            detail::make_attr_ints("pads",        {p[0], p[1], p[0], p[1]}),
            detail::make_attr_ints("dilations",   {d[0], d[1]}),
        };
        std::string out = ctx.fresh("maxpool");
        ctx.nodes.push_back(make_node({ctx.current_value}, {out}, "MaxPool",
                                       ctx.fresh("MaxPool"), attrs));
        ctx.current_shape = pool2d_out_shape(ctx.current_shape, k, s, p, d);
        ctx.current_value = out;
        return true;
    }

    // ---------------- AvgPool2d -> AveragePool ----------------
    if (auto* ap = as<nn::AvgPool2d>(m)) {
        auto k = ap->kernel_size();
        auto s = ap->stride();
        auto p = ap->padding();
        std::vector<std::string> attrs = {
            detail::make_attr_ints("kernel_shape", {k[0], k[1]}),
            detail::make_attr_ints("strides",     {s[0], s[1]}),
            detail::make_attr_ints("pads",        {p[0], p[1], p[0], p[1]}),
            detail::make_attr_int("count_include_pad", ap->count_include_pad() ? 1 : 0),
        };
        std::string out = ctx.fresh("avgpool");
        ctx.nodes.push_back(make_node({ctx.current_value}, {out}, "AveragePool",
                                       ctx.fresh("AveragePool"), attrs));
        ctx.current_shape = pool2d_out_shape(ctx.current_shape, k, s, p);
        ctx.current_value = out;
        return true;
    }

    // ---------------- Activations: ReLU / Sigmoid / Tanh ----------------
    auto emit_unary = [&](const std::string& op_type) {
        std::string out = ctx.fresh(op_type);
        ctx.nodes.push_back(make_node({ctx.current_value}, {out}, op_type,
                                       ctx.fresh(op_type)));
        ctx.current_value = out;
    };
    if (as<nn::ReLU>(m))     { emit_unary("Relu");    return true; }
    if (as<nn::Sigmoid>(m))  { emit_unary("Sigmoid"); return true; }
    if (as<nn::Tanh>(m))     { emit_unary("Tanh");    return true; }

    // ---------------- Softmax ----------------
    if (auto* sm = as<nn::Softmax>(m)) {
        int64_t dim = sm->dim();
        if (dim < 0) dim += static_cast<int64_t>(ctx.current_shape.size());
        std::vector<std::string> attrs = {
            detail::make_attr_int("axis", dim),
        };
        std::string out = ctx.fresh("softmax");
        ctx.nodes.push_back(make_node({ctx.current_value}, {out}, "Softmax",
                                       ctx.fresh("Softmax"), attrs));
        ctx.current_value = out;
        return true;
    }

    // ---------------- Flatten ----------------
    if (as<nn::Flatten>(m)) {
        std::vector<std::string> attrs = { detail::make_attr_int("axis", 1) };
        std::string out = ctx.fresh("flatten");
        ctx.nodes.push_back(make_node({ctx.current_value}, {out}, "Flatten",
                                       ctx.fresh("Flatten"), attrs));
        // Update shape: flatten dims [1..end]
        int64_t prod = 1;
        for (size_t i = 1; i < ctx.current_shape.size(); ++i) prod *= ctx.current_shape[i];
        std::vector<int64_t> new_shape = {ctx.current_shape.empty() ? 1 : ctx.current_shape[0], prod};
        ctx.current_shape = new_shape;
        ctx.current_value = out;
        return true;
    }

    return false;
}

// ============================================================================
// Walk module graph (Sequential -> recurse, leaf -> emit).
// ============================================================================
inline bool walk(ExportContext& ctx, Module* m) {
    // Sequential / ModuleList / ModuleDict — recurse over children
    if (auto* seq = as<nn::Sequential>(m)) {
        for (size_t i = 0; i < seq->size(); ++i) {
            if (!walk(ctx, (*seq)[i].get())) return false;
        }
        return true;
    }
    // Single leaf
    return emit_module(ctx, m);
}

// ============================================================================
// Build the GraphProto + ModelProto and write to disk.
// ============================================================================
inline std::string build_graph(const ExportContext& ctx,
                                const std::string& input_name,
                                const std::vector<int64_t>& input_shape,
                                const std::string& output_name,
                                const std::vector<int64_t>& output_shape) {
    std::string graph;
    write_string(graph, 2, "promethorch_graph");          // GraphProto.name
    for (auto& n : ctx.nodes)         write_submsg(graph, 1, n);
    for (auto& init : ctx.initializers) write_submsg(graph, 5, init);
    write_submsg(graph, 11, make_value_info_proto(input_name,  input_shape));
    write_submsg(graph, 12, make_value_info_proto(output_name, output_shape));
    return graph;
}

}  // namespace detail

// ============================================================================
// Public API
// ============================================================================

// Export a model to ONNX. Returns true on success.
//
// Supported modules: Linear, Conv2d, ReLU, Sigmoid, Tanh, Softmax,
// BatchNorm2d, MaxPool2d, AvgPool2d, Flatten, Identity, Sequential.
//
// `example_input` is required to record the input tensor's shape into the
// ONNX ValueInfoProto so downstream tools (Netron, ORT) get static shapes.
// We also use it to forward-trace shape transforms for Linear/Conv/Pool.
inline bool export_model(Module& model,
                          const Tensor& example_input,
                          const std::string& path,
                          const std::string& input_name = "input",
                          const std::string& output_name = "output") {
    detail::ExportContext ctx;
    ctx.current_value = input_name;
    for (int64_t d = 0; d < example_input.dim(); ++d)
        ctx.current_shape.push_back(example_input.size(d));
    std::vector<int64_t> in_shape = ctx.current_shape;

    if (!detail::walk(ctx, &model)) return false;

    // Insert an Identity at the tail so the graph output name matches user spec.
    ctx.nodes.push_back(detail::make_node({ctx.current_value}, {output_name},
                                           "Identity",
                                           ctx.fresh("output_identity")));
    std::vector<int64_t> out_shape = ctx.current_shape;

    // Build GraphProto
    std::string graph = detail::build_graph(ctx, input_name, in_shape,
                                             output_name, out_shape);

    // Build ModelProto.
    //   1  int64       ir_version
    //   2  string      producer_name
    //   3  string      producer_version
    //   4  string      domain
    //   5  int64       model_version
    //   6  string      doc_string
    //   7  GraphProto  graph
    //   8  OperatorSetIdProto[] opset_import
    std::string model_msg;
    detail::write_int64 (model_msg, 1, 7);                   // ir_version = 7
    detail::write_string(model_msg, 2, "promethorch");       // producer_name
    detail::write_string(model_msg, 3, "0.1");                // producer_version
    detail::write_string(model_msg, 4, "");                   // domain
    detail::write_int64 (model_msg, 5, 1);                    // model_version
    detail::write_string(model_msg, 6, "");                   // doc_string
    detail::write_submsg(model_msg, 7, graph);                // graph
    {
        // OperatorSetIdProto { 1 string domain, 2 int64 version }
        std::string opset;
        detail::write_string(opset, 1, "");                   // ai.onnx default
        detail::write_int64 (opset, 2, 13);
        detail::write_submsg(model_msg, 8, opset);
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(model_msg.data(), static_cast<std::streamsize>(model_msg.size()));
    return static_cast<bool>(f);
}

// ============================================================================
// Self-test: build a tiny MLP, export it, return true if file > 100 bytes.
// ============================================================================
inline bool onnx_self_test(const std::string& tmp_path = "/tmp/test.onnx") {
    auto net = std::make_shared<nn::Sequential>();
    net->push_back("fc1",  std::make_shared<nn::Linear>(8, 16));
    net->push_back("relu", std::make_shared<nn::ReLU>());
    net->push_back("fc2",  std::make_shared<nn::Linear>(16, 4));

    Tensor x = at::zeros({1, 8});
    if (!export_model(*net, x, tmp_path)) return false;

    std::ifstream f(tmp_path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto size = f.tellg();
    return size > std::streamoff(100);
}

}  // namespace onnx
}  // namespace torch

// ============================================================================
// Known limitations:
//   * Inference-only (no autograd state in graph).
//   * Sequential/leaf only — no skip connections, no attention, no
//     RNN/LSTM/Transformer/PIR.
//   * float32 dtype only; no INT8/FP16.
//   * No GroupConv beyond what Conv2d's `group` attribute can express.
//   * Linear with >2D input is exported as Gemm; ONNX Gemm requires 2D
//     inputs — caller should insert a Flatten before Linear when feeding
//     non-2D activations.
//   * Padding is symmetric (top==bottom, left==right) — asymmetric pads
//     would require splitting into Pad + Conv.
//   * Custom ops (RMSNorm, RoPE, FlashAttention, fused_linear_relu) are
//     unsupported and will cause export_model() to return false.
// ============================================================================
