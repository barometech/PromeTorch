#pragma once
// ============================================================================
// torch::jit — torch.compile-style tracing JIT for PromeTorch (CPU prototype)
// ============================================================================
//
// Approach (Option A: IR-less replay):
//   1. trace: A function f: Tensor -> Tensor is invoked exactly once with an
//      example input. Inside f the user must call jit::traced_* variants of
//      ops (e.g. jit::traced_add, jit::traced_mul). Those record an OpRecord
//      onto a thread-local Recorder while still computing the eager result so
//      the trace produces a valid output.
//   2. fuse: After tracing we scan for chains of element-wise ops (add/sub/
//      mul/div/relu/sigmoid/tanh/exp/log) and merge them into a single
//      FUSED_EWISE record that carries a list of micro-ops. Fused records
//      avoid intermediate buffer allocations and execute as one OMP loop.
//   3. execute: On replay we walk the (post-fusion) trace, re-using
//      preallocated buffers for non-fused ops and a single output buffer for
//      fused chains. No autograd hookup, no Tensor wrapping per op, no
//      dispatch overhead.
//
// Supported ops (traced_*): add, sub, mul, div, mm, matmul, relu, sigmoid,
// tanh, exp, log, softmax, linear (= mm + bias add).
//
// Limitations:
//   * f must be straight-line (single Tensor -> Tensor). No control flow.
//   * Shapes must be stable across calls (replay assumes example shapes).
//   * CPU only. Float32 only. Contiguous tensors only.
//   * mm/matmul/softmax are NOT fused with element-wise neighbours.
//   * The user must call jit::traced_* helpers; raw Tensor operators are not
//     hooked. This keeps the prototype small and avoids modifying Tensor.h.
// ============================================================================

// NOTE: Include "aten/src/ATen/ATen.h" (or torch/nn/nn.h) BEFORE this header.
// We rely on at::Tensor methods (add/mul/relu/...) being declared/defined.
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace torch {
namespace jit {

// ============================================================================
// Op tag
// ============================================================================
enum class Op : uint8_t {
    // Element-wise (fusable)
    ADD_T,    // out = a + b   (tensor, tensor)
    SUB_T,    // out = a - b
    MUL_T,    // out = a * b
    DIV_T,    // out = a / b
    ADD_S,    // out = a + s   (tensor, scalar)
    SUB_S,    // out = a - s
    MUL_S,    // out = a * s
    DIV_S,    // out = a / s
    RELU,
    SIGMOID,
    TANH,
    EXP,
    LOG,

    // Non-fusable
    MM,        // a @ b  (2D)
    MATMUL,    // alias of MM for 2D in this prototype
    SOFTMAX,   // softmax along last dim
    LINEAR,    // x @ W^T + b

    // Synthetic
    FUSED_EWISE
};

inline bool is_unary_ewise(Op o) {
    return o == Op::RELU || o == Op::SIGMOID || o == Op::TANH ||
           o == Op::EXP  || o == Op::LOG;
}
inline bool is_scalar_ewise(Op o) {
    return o == Op::ADD_S || o == Op::SUB_S || o == Op::MUL_S || o == Op::DIV_S;
}
inline bool is_binary_tensor_ewise(Op o) {
    return o == Op::ADD_T || o == Op::SUB_T || o == Op::MUL_T || o == Op::DIV_T;
}
inline bool is_fusable(Op o) {
    return is_unary_ewise(o) || is_scalar_ewise(o) || is_binary_tensor_ewise(o);
}

// ============================================================================
// Op record
// ============================================================================
struct MicroOp {
    Op op;
    float scalar = 0.0f;     // for *_S ops
    int64_t rhs_value_id = -1; // for *_T ops, id of the second tensor input
};

struct OpRecord {
    Op op;
    std::vector<int64_t> input_ids;  // value-ids of inputs
    int64_t output_id;               // value-id of output
    float scalar = 0.0f;             // for *_S ops
    std::vector<int64_t> out_shape;  // shape of the output

    // For FUSED_EWISE: chain of micro-ops applied left-to-right.
    // The base input (input_ids[0]) flows through each micro-op; for *_T
    // micro-ops, the second operand is identified via micro.rhs_value_id and
    // also appears in input_ids (so the executor can resolve it).
    std::vector<MicroOp> chain;
};

// ============================================================================
// Recorder (thread-local active tracer)
// ============================================================================
class Recorder {
public:
    int64_t next_id = 0;
    std::vector<OpRecord> records;

    // Map from value-id to its captured shape during tracing.
    std::vector<std::vector<int64_t>> shape_of;

    // For inputs that are NOT produced by ops (e.g. captured constants like
    // weight matrices), we keep a direct Tensor reference so replay can use
    // them.
    std::vector<at::Tensor> external_tensors;
    std::vector<bool> is_external;  // parallel to value-ids

    int64_t input_id = -1;   // value-id of the traced input
    int64_t output_id = -1;  // value-id of the traced output

    int64_t new_id(const std::vector<int64_t>& shape, bool external,
                   const at::Tensor& t = at::Tensor()) {
        int64_t id = next_id++;
        shape_of.push_back(shape);
        is_external.push_back(external);
        external_tensors.push_back(external ? t : at::Tensor());
        return id;
    }
};

inline Recorder*& current_recorder() {
    thread_local Recorder* rec = nullptr;
    return rec;
}

inline bool tracing_active() { return current_recorder() != nullptr; }

// ============================================================================
// TracedTensor — thin wrapper that carries a value-id alongside an eager
// Tensor. Returned by traced_* helpers. Convertible to at::Tensor for
// passing into eager ops or for the user's final return.
// ============================================================================
struct TracedTensor {
    at::Tensor value;     // eager-computed value (used during trace)
    int64_t id = -1;      // value-id in the recorder

    TracedTensor() = default;
    TracedTensor(at::Tensor v, int64_t i) : value(std::move(v)), id(i) {}

    // Implicit conversion so `return x;` from f works when f returns Tensor.
    operator at::Tensor() const { return value; }
};

// Wrap a raw tensor as a "constant" tracer value (for closed-over weights).
inline TracedTensor traced_const(const at::Tensor& t) {
    auto* rec = current_recorder();
    if (!rec) return TracedTensor(t, -1);
    int64_t id = rec->new_id(std::vector<int64_t>(t.sizes().begin(),
                                                  t.sizes().end()),
                             /*external=*/true, t);
    return TracedTensor(t, id);
}

// Mark the input value. Called by compile() before invoking f.
inline TracedTensor traced_input(const at::Tensor& t) {
    auto* rec = current_recorder();
    int64_t id = rec->new_id(std::vector<int64_t>(t.sizes().begin(),
                                                  t.sizes().end()),
                             /*external=*/false);
    rec->input_id = id;
    return TracedTensor(t, id);
}

// ============================================================================
// Helpers to record an op with one or two TracedTensor inputs.
// ============================================================================
namespace detail {

inline int64_t resolve_id(const TracedTensor& t) {
    auto* rec = current_recorder();
    if (t.id >= 0) return t.id;
    // Tensor came from outside the trace — register as a constant.
    return traced_const(t.value).id;
}

inline TracedTensor record(Op op, at::Tensor result,
                           std::vector<int64_t> in_ids,
                           float scalar = 0.0f) {
    auto* rec = current_recorder();
    std::vector<int64_t> out_shape(result.sizes().begin(), result.sizes().end());
    int64_t out_id = rec->new_id(out_shape, /*external=*/false);
    OpRecord r;
    r.op = op;
    r.input_ids = std::move(in_ids);
    r.output_id = out_id;
    r.scalar = scalar;
    r.out_shape = std::move(out_shape);
    rec->records.push_back(std::move(r));
    return TracedTensor(std::move(result), out_id);
}

} // namespace detail

// ============================================================================
// traced_* op API
// ============================================================================
inline TracedTensor traced_add(const TracedTensor& a, const TracedTensor& b) {
    if (!tracing_active()) return TracedTensor(a.value.add(b.value), -1);
    int64_t aid = detail::resolve_id(a), bid = detail::resolve_id(b);
    return detail::record(Op::ADD_T, a.value.add(b.value), {aid, bid});
}
inline TracedTensor traced_sub(const TracedTensor& a, const TracedTensor& b) {
    if (!tracing_active()) return TracedTensor(a.value.sub(b.value), -1);
    int64_t aid = detail::resolve_id(a), bid = detail::resolve_id(b);
    return detail::record(Op::SUB_T, a.value.sub(b.value), {aid, bid});
}
inline TracedTensor traced_mul(const TracedTensor& a, const TracedTensor& b) {
    if (!tracing_active()) return TracedTensor(a.value.mul(b.value), -1);
    int64_t aid = detail::resolve_id(a), bid = detail::resolve_id(b);
    return detail::record(Op::MUL_T, a.value.mul(b.value), {aid, bid});
}
inline TracedTensor traced_div(const TracedTensor& a, const TracedTensor& b) {
    if (!tracing_active()) return TracedTensor(a.value.div(b.value), -1);
    int64_t aid = detail::resolve_id(a), bid = detail::resolve_id(b);
    return detail::record(Op::DIV_T, a.value.div(b.value), {aid, bid});
}

inline TracedTensor traced_add(const TracedTensor& a, float s) {
    if (!tracing_active()) return TracedTensor(a.value.add(at::Scalar(s)), -1);
    return detail::record(Op::ADD_S, a.value.add(at::Scalar(s)),
                          {detail::resolve_id(a)}, s);
}
inline TracedTensor traced_sub(const TracedTensor& a, float s) {
    if (!tracing_active()) return TracedTensor(a.value.sub(at::Scalar(s)), -1);
    return detail::record(Op::SUB_S, a.value.sub(at::Scalar(s)),
                          {detail::resolve_id(a)}, s);
}
inline TracedTensor traced_mul(const TracedTensor& a, float s) {
    if (!tracing_active()) return TracedTensor(a.value.mul(at::Scalar(s)), -1);
    return detail::record(Op::MUL_S, a.value.mul(at::Scalar(s)),
                          {detail::resolve_id(a)}, s);
}
inline TracedTensor traced_div(const TracedTensor& a, float s) {
    if (!tracing_active()) return TracedTensor(a.value.div(at::Scalar(s)), -1);
    return detail::record(Op::DIV_S, a.value.div(at::Scalar(s)),
                          {detail::resolve_id(a)}, s);
}

inline TracedTensor traced_relu(const TracedTensor& a) {
    if (!tracing_active()) return TracedTensor(a.value.relu(), -1);
    return detail::record(Op::RELU, a.value.relu(), {detail::resolve_id(a)});
}
inline TracedTensor traced_sigmoid(const TracedTensor& a) {
    if (!tracing_active()) return TracedTensor(a.value.sigmoid(), -1);
    return detail::record(Op::SIGMOID, a.value.sigmoid(),
                          {detail::resolve_id(a)});
}
inline TracedTensor traced_tanh(const TracedTensor& a) {
    if (!tracing_active()) return TracedTensor(a.value.tanh(), -1);
    return detail::record(Op::TANH, a.value.tanh(), {detail::resolve_id(a)});
}
inline TracedTensor traced_exp(const TracedTensor& a) {
    if (!tracing_active()) return TracedTensor(a.value.exp(), -1);
    return detail::record(Op::EXP, a.value.exp(), {detail::resolve_id(a)});
}
inline TracedTensor traced_log(const TracedTensor& a) {
    if (!tracing_active()) return TracedTensor(a.value.log(), -1);
    return detail::record(Op::LOG, a.value.log(), {detail::resolve_id(a)});
}

inline TracedTensor traced_mm(const TracedTensor& a, const TracedTensor& b) {
    if (!tracing_active()) return TracedTensor(a.value.mm(b.value), -1);
    return detail::record(Op::MM, a.value.mm(b.value),
                          {detail::resolve_id(a), detail::resolve_id(b)});
}
inline TracedTensor traced_matmul(const TracedTensor& a, const TracedTensor& b) {
    if (!tracing_active()) return TracedTensor(a.value.matmul(b.value), -1);
    return detail::record(Op::MATMUL, a.value.matmul(b.value),
                          {detail::resolve_id(a), detail::resolve_id(b)});
}

// Softmax along last dim.
inline TracedTensor traced_softmax(const TracedTensor& a) {
    // Eager fallback compute via exp/sum/div on last dim.
    auto x = a.value.contiguous();
    auto m = std::get<0>(x.max(x.dim() - 1, /*keepdim=*/true));
    auto e = (x.sub(m)).exp();
    auto s = e.sum(x.dim() - 1, /*keepdim=*/true);
    auto y = e.div(s);
    if (!tracing_active()) return TracedTensor(y, -1);
    return detail::record(Op::SOFTMAX, y, {detail::resolve_id(a)});
}

// linear(x, W, b): x @ W^T + b
inline TracedTensor traced_linear(const TracedTensor& x, const TracedTensor& W,
                                  const TracedTensor& b) {
    auto y = x.value.mm(W.value.t().contiguous()).add(b.value);
    if (!tracing_active()) return TracedTensor(y, -1);
    return detail::record(Op::LINEAR, y,
                          {detail::resolve_id(x), detail::resolve_id(W),
                           detail::resolve_id(b)});
}

// ============================================================================
// Fusion pass: collapse maximal runs of fusable ops that form a linear chain
// (output of op[i] is the sole consumer's input on op[i+1]).
// ============================================================================
inline std::vector<OpRecord> fuse(const std::vector<OpRecord>& in,
                                  int64_t output_id) {
    // Count consumer references for each value-id.
    int64_t max_id = 0;
    for (const auto& r : in) {
        for (auto i : r.input_ids) max_id = std::max(max_id, i);
        max_id = std::max(max_id, r.output_id);
    }
    std::vector<int> use_count(max_id + 1, 0);
    for (const auto& r : in) for (auto i : r.input_ids) use_count[i]++;
    if (output_id >= 0) use_count[output_id]++; // pin the final output

    std::vector<OpRecord> out;
    out.reserve(in.size());

    size_t i = 0;
    while (i < in.size()) {
        const auto& r0 = in[i];
        if (!is_fusable(r0.op)) { out.push_back(r0); ++i; continue; }

        // Start a fused chain seeded by r0. base value-id = r0.input_ids[0].
        OpRecord fused;
        fused.op = Op::FUSED_EWISE;
        fused.input_ids.push_back(r0.input_ids[0]);
        if (is_binary_tensor_ewise(r0.op)) {
            fused.input_ids.push_back(r0.input_ids[1]);
        }
        MicroOp m0; m0.op = r0.op; m0.scalar = r0.scalar;
        if (is_binary_tensor_ewise(r0.op)) m0.rhs_value_id = r0.input_ids[1];
        fused.chain.push_back(m0);
        fused.output_id = r0.output_id;
        fused.out_shape = r0.out_shape;

        // Try to extend.
        size_t j = i + 1;
        while (j < in.size()) {
            const auto& rj = in[j];
            if (!is_fusable(rj.op)) break;
            // The chain-head value (current output) must feed rj as its
            // FIRST input, and must have only one consumer (this op), AND
            // shape must be unchanged.
            int64_t head = fused.output_id;
            if (rj.input_ids.empty() || rj.input_ids[0] != head) break;
            if (use_count[head] != 1) break;
            if (rj.out_shape != fused.out_shape) break;

            MicroOp mj; mj.op = rj.op; mj.scalar = rj.scalar;
            if (is_binary_tensor_ewise(rj.op)) {
                mj.rhs_value_id = rj.input_ids[1];
                fused.input_ids.push_back(rj.input_ids[1]);
            }
            fused.chain.push_back(mj);
            fused.output_id = rj.output_id;
            ++j;
        }

        out.push_back(std::move(fused));
        i = j;
    }
    return out;
}

// ============================================================================
// Run a whole fused chain in a single OMP parallel region. Each thread walks
// its slice of the output through every micro-op, so intermediates stay in
// L1/registers — no per-op kernel launch, no per-op memory round-trip.
// ============================================================================
// Encode each micro-op as a small POD so the inner loop reads just 16 bytes
// per op and the compiler has a chance to predict / unroll.
struct CompiledMicro {
    uint8_t op;            // matches Op enum (cast)
    float scalar;          // scalar payload (also used as zero for unaries)
    const float* rhs;      // null for non-binary ops
};

inline void run_fused_chain(const std::vector<CompiledMicro>& chain,
                            const float* __restrict__ base,
                            float* __restrict__ out, int64_t n) {
    const size_t L = chain.size();
    const CompiledMicro* __restrict__ ops = chain.data();

    auto kernel = [&](int64_t start, int64_t end) {
        for (int64_t k = start; k < end; ++k) {
            float v = base[k];
            for (size_t i = 0; i < L; ++i) {
                const CompiledMicro& m = ops[i];
                switch (static_cast<Op>(m.op)) {
                    case Op::ADD_T: v = v + m.rhs[k]; break;
                    case Op::SUB_T: v = v - m.rhs[k]; break;
                    case Op::MUL_T: v = v * m.rhs[k]; break;
                    case Op::DIV_T: v = v / m.rhs[k]; break;
                    case Op::ADD_S: v = v + m.scalar; break;
                    case Op::SUB_S: v = v - m.scalar; break;
                    case Op::MUL_S: v = v * m.scalar; break;
                    case Op::DIV_S: v = v / m.scalar; break;
                    case Op::RELU: v = v > 0.0f ? v : 0.0f; break;
                    case Op::SIGMOID: v = 1.0f / (1.0f + std::exp(-v)); break;
                    case Op::TANH: v = std::tanh(v); break;
                    case Op::EXP: v = std::exp(v); break;
                    case Op::LOG: v = std::log(v); break;
                    default: break;
                }
            }
            out[k] = v;
        }
    };

    // OMP only pays off for large n; serial for small.
#ifdef _OPENMP
    if (n >= 4096) {
        #pragma omp parallel
        {
            int nth = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int64_t chunk = (n + nth - 1) / nth;
            int64_t s = tid * chunk;
            int64_t e = std::min<int64_t>(s + chunk, n);
            if (s < e) kernel(s, e);
        }
        return;
    }
#endif
    kernel(0, n);
}

// ============================================================================
// CompiledFn — owns the (post-fusion) trace and replays it.
// ============================================================================
class CompiledFn {
public:
    CompiledFn() = default;

    // Internal state populated by compile().
    std::vector<OpRecord> program;
    std::vector<at::Tensor> externals;        // by value-id
    std::vector<bool> is_external;            // by value-id
    std::vector<std::vector<int64_t>> shapes; // by value-id
    int64_t input_id = -1;
    int64_t output_id = -1;
    int64_t value_count = 0;
    size_t original_trace_len = 0;

    // Per-record output cache: we allocate a Tensor once on the first call
    // (or when shapes change) and reuse the storage on every subsequent call.
    // mutable so operator() can be const and still update caches.
    mutable std::vector<at::Tensor> output_cache;

    size_t trace_len() const { return program.size(); }
    size_t raw_trace_len() const { return original_trace_len; }

    at::Tensor operator()(const at::Tensor& input) const {
        if (output_cache.size() != program.size())
            output_cache.assign(program.size(), at::Tensor());

        // value-id -> Tensor. nullopt-equivalent = undefined Tensor.
        std::vector<at::Tensor> values(value_count);
        for (int64_t i = 0; i < value_count; ++i) {
            if (is_external[i]) values[i] = externals[i];
        }
        values[input_id] = input.contiguous();

        for (size_t pi = 0; pi < program.size(); ++pi) {
            const auto& r = program[pi];
            switch (r.op) {
                case Op::FUSED_EWISE: {
                    // Allocate output buffer once and cache it.
                    auto base = values[r.input_ids[0]].contiguous();
                    at::Tensor& out = output_cache[pi];
                    bool need_alloc = !out.defined() ||
                        out.numel() != int64_t(1) * [&]() {
                            int64_t n = 1;
                            for (auto s : r.out_shape) n *= s;
                            return n;
                        }();
                    if (need_alloc) {
                        out = at::detail::make_tensor(
                            c10::IntArrayRef(r.out_shape.data(),
                                             r.out_shape.size()),
                            at::TensorOptions()
                                .dtype(c10::ScalarType::Float)
                                .device(c10::kCPU));
                    }
                    int64_t n = out.numel();

                    // Build CompiledMicro array on the stack-ish (small).
                    std::vector<CompiledMicro> chain(r.chain.size());
                    for (size_t i = 0; i < r.chain.size(); ++i) {
                        const auto& m = r.chain[i];
                        chain[i].op = static_cast<uint8_t>(m.op);
                        chain[i].scalar = m.scalar;
                        chain[i].rhs = nullptr;
                        if (is_binary_tensor_ewise(m.op)) {
                            auto& rt = values[m.rhs_value_id];
                            if (!rt.is_contiguous()) rt = rt.contiguous();
                            chain[i].rhs = rt.data_ptr<float>();
                        }
                    }
                    run_fused_chain(chain, base.data_ptr<float>(),
                                    out.mutable_data_ptr<float>(), n);
                    values[r.output_id] = out;
                    break;
                }
                case Op::MM: {
                    values[r.output_id] = values[r.input_ids[0]]
                        .mm(values[r.input_ids[1]]);
                    break;
                }
                case Op::MATMUL: {
                    values[r.output_id] = values[r.input_ids[0]]
                        .matmul(values[r.input_ids[1]]);
                    break;
                }
                case Op::SOFTMAX: {
                    auto x = values[r.input_ids[0]].contiguous();
                    auto m = std::get<0>(x.max(x.dim() - 1, true));
                    auto e = x.sub(m).exp();
                    auto s = e.sum(x.dim() - 1, true);
                    values[r.output_id] = e.div(s);
                    break;
                }
                case Op::LINEAR: {
                    auto& x = values[r.input_ids[0]];
                    auto& W = values[r.input_ids[1]];
                    auto& b = values[r.input_ids[2]];
                    values[r.output_id] = x.mm(W.t().contiguous()).add(b);
                    break;
                }
                default:
                    // All fusable ewise ops are converted to FUSED_EWISE by
                    // the fusion pass, so any remaining ewise op here is a
                    // bug — but we fall back gracefully via traced eager.
                    PT_ERROR("jit: unsupported op in replay");
            }
        }
        return values[output_id];
    }
};

// ============================================================================
// Top-level compile() entry point.
// ============================================================================
inline CompiledFn compile(std::function<TracedTensor(TracedTensor)> f,
                          const at::Tensor& example_input) {
    Recorder rec;
    Recorder* prev = current_recorder();
    current_recorder() = &rec;

    TracedTensor x_in = traced_input(example_input.contiguous());
    TracedTensor y = f(x_in);
    rec.output_id = (y.id >= 0) ? y.id : detail::resolve_id(y);

    current_recorder() = prev;

    CompiledFn cf;
    cf.original_trace_len = rec.records.size();
    cf.program = fuse(rec.records, rec.output_id);
    cf.externals = std::move(rec.external_tensors);
    cf.is_external = std::move(rec.is_external);
    cf.shapes = std::move(rec.shape_of);
    cf.input_id = rec.input_id;
    cf.output_id = rec.output_id;
    cf.value_count = rec.next_id;
    return cf;
}

} // namespace jit
} // namespace torch
