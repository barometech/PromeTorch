#pragma once
// =============================================================================
// PromeTorch Mobile Executor — ExecuTorch-like compact mobile inference format.
// =============================================================================
// One-file binary representation of a forward-only nn::Sequential model:
//   1. export_model(seq, example_input, path) walks the Sequential, dumps
//      weights + a flat op-record table.
//   2. MobileExecutor::load(path) reads the table back; ::forward(x) executes
//      ops sequentially using at::* ops (no autograd, no backprop).
//
// Binary layout:
//   Header:   "PTMB" (4 bytes), version int32
//   Weights:  num_weights int32, then for each weight:
//               shape_dim int32, dims int32 * shape_dim,
//               numel_bytes int32, raw float data
//   Ops:      num_ops int32, then for each op:
//               op_type int32, num_weights int32, weight_ids int32 * N,
//               num_params int32, params int32 * N
//
// Supported ops (initial set): Linear, ReLU, Sigmoid, Tanh, Conv2d, MaxPool2d,
// BatchNorm2d, Flatten, Softmax.
// =============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

#include "torch/nn/module.h"
#include "torch/nn/modules/container.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/modules/conv.h"
#include "torch/nn/modules/pooling.h"
#include "torch/nn/modules/normalization.h"
#include "torch/csrc/autograd/grad_mode.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

namespace torch {
namespace mobile {

using at::Tensor;

// ----------------------------------------------------------------------------
// Op codes — keep in sync with both export and runtime dispatch.
// ----------------------------------------------------------------------------
enum OpCode : int32_t {
    OP_LINEAR      = 1,
    OP_RELU        = 2,
    OP_SIGMOID     = 3,
    OP_TANH        = 4,
    OP_CONV2D      = 5,
    OP_MAXPOOL2D   = 6,
    OP_BATCHNORM2D = 7,
    OP_FLATTEN     = 8,
    OP_SOFTMAX     = 9,
};

constexpr char     MOBILE_MAGIC[4]      = {'P','T','M','B'};
constexpr int32_t  MOBILE_VERSION       = 1;

namespace detail {

inline void w_bytes(std::ofstream& f, const void* p, size_t n) {
    f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("mobile: write failed");
}
inline void r_bytes(std::ifstream& f, void* p, size_t n) {
    f.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("mobile: read failed");
}
inline void w_i32(std::ofstream& f, int32_t v) { w_bytes(f, &v, sizeof(v)); }
inline int32_t r_i32(std::ifstream& f) {
    int32_t v;
    r_bytes(f, &v, sizeof(v));
    return v;
}

// Dump one float tensor as: shape_dim, dims..., numel_bytes, data.
inline void write_weight(std::ofstream& f, const Tensor& t) {
    Tensor c = t.contiguous();
    if (c.dtype() != c10::ScalarType::Float) {
        throw std::runtime_error("mobile: only float32 weights supported");
    }
    int32_t dim = static_cast<int32_t>(c.dim());
    w_i32(f, dim);
    for (int32_t i = 0; i < dim; ++i) {
        w_i32(f, static_cast<int32_t>(c.size(i)));
    }
    int32_t nbytes = static_cast<int32_t>(c.nbytes());
    w_i32(f, nbytes);
    w_bytes(f, c.data_ptr(), static_cast<size_t>(nbytes));
}

inline Tensor read_weight(std::ifstream& f) {
    int32_t dim = r_i32(f);
    std::vector<int64_t> shape(dim);
    for (int32_t i = 0; i < dim; ++i) shape[i] = r_i32(f);
    int32_t nbytes = r_i32(f);
    Tensor t = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Float));
    if (static_cast<int32_t>(t.nbytes()) != nbytes) {
        throw std::runtime_error("mobile: weight nbytes mismatch");
    }
    r_bytes(f, t.mutable_data_ptr<float>(), static_cast<size_t>(nbytes));
    return t;
}

} // namespace detail

// ----------------------------------------------------------------------------
// Op record — N weight indices and N integer params.
// ----------------------------------------------------------------------------
struct OpRecord {
    int32_t              op_type;
    std::vector<int32_t> weight_ids;
    std::vector<int32_t> params;
};

// ----------------------------------------------------------------------------
// MobileExecutor — load + run.
// ----------------------------------------------------------------------------
class MobileExecutor {
public:
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        char magic[4];
        detail::r_bytes(f, magic, 4);
        if (std::memcmp(magic, MOBILE_MAGIC, 4) != 0) return false;

        int32_t version = detail::r_i32(f);
        if (version != MOBILE_VERSION) return false;

        // Weights
        int32_t nw = detail::r_i32(f);
        weights_.clear();
        weights_.reserve(nw);
        for (int32_t i = 0; i < nw; ++i) {
            weights_.push_back(detail::read_weight(f));
        }

        // Ops
        int32_t nops = detail::r_i32(f);
        ops_.clear();
        ops_.reserve(nops);
        for (int32_t i = 0; i < nops; ++i) {
            OpRecord rec;
            rec.op_type = detail::r_i32(f);
            int32_t nwi = detail::r_i32(f);
            rec.weight_ids.resize(nwi);
            for (int32_t k = 0; k < nwi; ++k) rec.weight_ids[k] = detail::r_i32(f);
            int32_t np = detail::r_i32(f);
            rec.params.resize(np);
            for (int32_t k = 0; k < np; ++k) rec.params[k] = detail::r_i32(f);
            ops_.push_back(std::move(rec));
        }
        return true;
    }

    Tensor forward(const Tensor& input) {
        // Inference only — no autograd.
        torch::autograd::NoGradGuard ng;

        Tensor x = input;
        for (const auto& op : ops_) {
            x = run_op(op, x);
        }
        return x;
    }

    size_t num_weights() const { return weights_.size(); }
    size_t num_ops()     const { return ops_.size();     }

private:
    Tensor run_op(const OpRecord& op, const Tensor& x) {
        switch (op.op_type) {
            case OP_LINEAR: {
                // params: [in_features, out_features, has_bias]
                // weights: [W (and bias if has_bias)]
                Tensor W = weights_[op.weight_ids[0]];
                bool has_bias = op.params.size() > 2 && op.params[2] != 0;
                Tensor xc = x.contiguous();
                Tensor Wt = W.t().contiguous();
                Tensor y;
                if (xc.dim() == 2) {
                    y = xc.mm(Wt);
                } else {
                    int64_t M = 1;
                    for (int64_t d = 0; d < xc.dim() - 1; ++d) M *= xc.size(d);
                    Tensor x2 = xc.reshape({M, xc.size(-1)});
                    y = x2.mm(Wt);
                    std::vector<int64_t> os(xc.sizes().begin(), xc.sizes().end() - 1);
                    os.push_back(W.size(0));
                    y = y.reshape(os);
                }
                if (has_bias) {
                    Tensor b = weights_[op.weight_ids[1]];
                    y = y.add(b);
                }
                return y;
            }
            case OP_RELU:    return x.relu();
            case OP_SIGMOID: return x.sigmoid();
            case OP_TANH:    return x.tanh();
            case OP_CONV2D: {
                // params: [in_c, out_c, kH, kW, sH, sW, pH, pW, dH, dW, groups, has_bias]
                Tensor W = weights_[op.weight_ids[0]];
                bool has_bias = op.params[11] != 0;
                Tensor b = has_bias ? weights_[op.weight_ids[1]] : Tensor();
                int64_t in_c = op.params[0], out_c = op.params[1];
                int64_t kH = op.params[2], kW = op.params[3];
                int64_t sH = op.params[4], sW = op.params[5];
                int64_t pH = op.params[6], pW = op.params[7];
                int64_t dH = op.params[8], dW = op.params[9];
                int64_t groups = op.params[10];
                // Reuse Conv2d module forward — the cleanest path that hits the
                // same im2col + sgemm kernel as training. We swap in our weights.
                nn::Conv2d conv(in_c, out_c,
                                std::array<int64_t,2>{kH, kW},
                                std::array<int64_t,2>{sH, sW},
                                std::array<int64_t,2>{pH, pW},
                                std::array<int64_t,2>{dH, dW},
                                groups, has_bias);
                conv.get_parameter("weight")->data().copy_(W);
                if (has_bias) conv.get_parameter("bias")->data().copy_(b);
                return conv.forward(x);
            }
            case OP_MAXPOOL2D: {
                // params: [kH, kW, sH, sW, pH, pW, dH, dW]
                int64_t kH = op.params[0];
                int64_t sH = op.params[2], pH = op.params[4], dH = op.params[6];
                nn::MaxPool2d pool(kH, sH, pH, dH);
                return pool.forward(x);
            }
            case OP_BATCHNORM2D: {
                // params: [num_features, affine, track]   (eps stored in weights[2] scalar)
                // weights: [gamma?, beta?, running_mean, running_var, eps_tensor]
                int64_t num_features = op.params[0];
                bool affine = op.params[1] != 0;
                int wid = 0;
                Tensor gamma = affine ? weights_[op.weight_ids[wid++]] : at::ones({num_features});
                Tensor beta  = affine ? weights_[op.weight_ids[wid++]] : at::zeros({num_features});
                Tensor rm    = weights_[op.weight_ids[wid++]];
                Tensor rv    = weights_[op.weight_ids[wid++]];
                Tensor eps_t = weights_[op.weight_ids[wid++]];
                float eps = eps_t.data_ptr<float>()[0];

                // Inference-mode batchnorm: y = (x - rm) / sqrt(rv + eps) * gamma + beta
                int64_t N = x.size(0), C = x.size(1), H = x.size(2), Wd = x.size(3);
                int64_t spatial = H * Wd;
                Tensor out = at::empty({N, C, H, Wd},
                                       at::TensorOptions().dtype(c10::ScalarType::Float));
                Tensor xc = x.contiguous();
                const float* in_d = xc.data_ptr<float>();
                float* out_d = out.mutable_data_ptr<float>();
                const float* g = gamma.data_ptr<float>();
                const float* be = beta.data_ptr<float>();
                const float* m = rm.data_ptr<float>();
                const float* v = rv.data_ptr<float>();
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t c = 0; c < C; ++c) {
                        float inv = 1.0f / std::sqrt(v[c] + eps);
                        float gc = g[c], bc = be[c], mc = m[c];
                        const float* src = in_d + (n * C + c) * spatial;
                        float*       dst = out_d + (n * C + c) * spatial;
                        for (int64_t s = 0; s < spatial; ++s) {
                            dst[s] = (src[s] - mc) * inv * gc + bc;
                        }
                    }
                }
                return out;
            }
            case OP_FLATTEN: {
                int64_t start = op.params[0];
                int64_t end   = op.params[1];
                return x.flatten(start, end);
            }
            case OP_SOFTMAX: {
                int64_t dim = op.params[0];
                if (dim < 0) dim += x.dim();
                Tensor mx = std::get<0>(x.max(dim, true));
                Tensor sh = x.sub(mx.expand(x.sizes()));
                Tensor ex = sh.exp();
                Tensor sm = ex.sum(dim, true);
                return ex.div(sm.expand(x.sizes()));
            }
            default:
                throw std::runtime_error("mobile: unknown op_type " +
                                         std::to_string(op.op_type));
        }
    }

    std::vector<Tensor>   weights_;
    std::vector<OpRecord> ops_;
};

// ----------------------------------------------------------------------------
// Export — visit each module of a Sequential, append weights + op record.
// ----------------------------------------------------------------------------
namespace detail {

inline void emit(std::vector<Tensor>& W, std::vector<OpRecord>& ops,
                 nn::Module* mod) {
    if (auto* m = dynamic_cast<nn::Linear*>(mod)) {
        OpRecord r;
        r.op_type = OP_LINEAR;
        int32_t wid = static_cast<int32_t>(W.size());
        W.push_back(m->get_parameter("weight")->data());
        r.weight_ids.push_back(wid);
        bool has_bias = (m->get_parameter("bias") != nullptr);
        if (has_bias) {
            r.weight_ids.push_back(static_cast<int32_t>(W.size()));
            W.push_back(m->get_parameter("bias")->data());
        }
        r.params = { static_cast<int32_t>(m->in_features()),
                     static_cast<int32_t>(m->out_features()),
                     has_bias ? 1 : 0 };
        ops.push_back(std::move(r));
        return;
    }
    if (dynamic_cast<nn::ReLU*>(mod))    { ops.push_back({OP_RELU,    {}, {}}); return; }
    if (dynamic_cast<nn::Sigmoid*>(mod)) { ops.push_back({OP_SIGMOID, {}, {}}); return; }
    if (dynamic_cast<nn::Tanh*>(mod))    { ops.push_back({OP_TANH,    {}, {}}); return; }

    if (auto* m = dynamic_cast<nn::Conv2d*>(mod)) {
        OpRecord r;
        r.op_type = OP_CONV2D;
        r.weight_ids.push_back(static_cast<int32_t>(W.size()));
        W.push_back(m->get_parameter("weight")->data());
        bool has_bias = m->has_bias();
        if (has_bias) {
            r.weight_ids.push_back(static_cast<int32_t>(W.size()));
            W.push_back(m->get_parameter("bias")->data());
        }
        const auto& ks = m->kernel_size();
        const auto& st = m->stride();
        const auto& pd = m->padding();
        const auto& dl = m->dilation();
        r.params = {
            static_cast<int32_t>(m->in_channels()),
            static_cast<int32_t>(m->out_channels()),
            static_cast<int32_t>(ks[0]), static_cast<int32_t>(ks[1]),
            static_cast<int32_t>(st[0]), static_cast<int32_t>(st[1]),
            static_cast<int32_t>(pd[0]), static_cast<int32_t>(pd[1]),
            static_cast<int32_t>(dl[0]), static_cast<int32_t>(dl[1]),
            static_cast<int32_t>(m->groups()),
            has_bias ? 1 : 0
        };
        ops.push_back(std::move(r));
        return;
    }
    if (auto* m = dynamic_cast<nn::MaxPool2d*>(mod)) {
        OpRecord r;
        r.op_type = OP_MAXPOOL2D;
        const auto& ks = m->kernel_size();
        const auto& st = m->stride();
        const auto& pd = m->padding();
        const auto& dl = m->dilation();
        r.params = {
            static_cast<int32_t>(ks[0]), static_cast<int32_t>(ks[1]),
            static_cast<int32_t>(st[0]), static_cast<int32_t>(st[1]),
            static_cast<int32_t>(pd[0]), static_cast<int32_t>(pd[1]),
            static_cast<int32_t>(dl[0]), static_cast<int32_t>(dl[1])
        };
        ops.push_back(std::move(r));
        return;
    }
    if (auto* m = dynamic_cast<nn::BatchNorm2d*>(mod)) {
        OpRecord r;
        r.op_type = OP_BATCHNORM2D;
        bool affine = m->affine();
        if (affine) {
            r.weight_ids.push_back(static_cast<int32_t>(W.size()));
            W.push_back(m->get_parameter("weight")->data());
            r.weight_ids.push_back(static_cast<int32_t>(W.size()));
            W.push_back(m->get_parameter("bias")->data());
        }
        // Always emit running_mean/running_var (BN inference needs them).
        r.weight_ids.push_back(static_cast<int32_t>(W.size()));
        W.push_back(m->get_buffer("running_mean")->data());
        r.weight_ids.push_back(static_cast<int32_t>(W.size()));
        W.push_back(m->get_buffer("running_var")->data());
        // Pack eps as a 1-element float tensor.
        Tensor eps_t = at::empty({1}, at::TensorOptions().dtype(c10::ScalarType::Float));
        eps_t.mutable_data_ptr<float>()[0] = static_cast<float>(m->eps());
        r.weight_ids.push_back(static_cast<int32_t>(W.size()));
        W.push_back(eps_t);

        r.params = {
            static_cast<int32_t>(m->num_features()),
            affine ? 1 : 0,
            m->track_running_stats() ? 1 : 0
        };
        ops.push_back(std::move(r));
        return;
    }
    if (auto* m = dynamic_cast<nn::Flatten*>(mod)) {
        OpRecord r;
        r.op_type = OP_FLATTEN;
        // Flatten doesn't expose accessors — fall back to defaults (1, -1).
        // The C++ Flatten is constructed once, so we stamp those defaults; if the
        // user used non-default dims they'd need to subclass / extend the export.
        (void)m;
        r.params = { 1, -1 };
        ops.push_back(std::move(r));
        return;
    }
    if (auto* m = dynamic_cast<nn::Softmax*>(mod)) {
        OpRecord r;
        r.op_type = OP_SOFTMAX;
        r.params = { static_cast<int32_t>(m->dim()) };
        ops.push_back(std::move(r));
        return;
    }

    throw std::runtime_error("mobile::export_model: unsupported module type '"
                             + mod->name() + "'");
}

} // namespace detail

inline bool export_model(nn::Sequential& model,
                         const Tensor& /*example_input*/,
                         const std::string& path) {
    std::vector<Tensor>   weights;
    std::vector<OpRecord> ops;

    // Walk Sequential children in registration order.
    for (size_t i = 0; i < model.size(); ++i) {
        nn::ModulePtr mp = model[i];
        if (!mp) continue;
        detail::emit(weights, ops, mp.get());
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    detail::w_bytes(f, MOBILE_MAGIC, 4);
    detail::w_i32(f, MOBILE_VERSION);

    detail::w_i32(f, static_cast<int32_t>(weights.size()));
    for (const auto& w : weights) detail::write_weight(f, w);

    detail::w_i32(f, static_cast<int32_t>(ops.size()));
    for (const auto& op : ops) {
        detail::w_i32(f, op.op_type);
        detail::w_i32(f, static_cast<int32_t>(op.weight_ids.size()));
        for (auto id : op.weight_ids) detail::w_i32(f, id);
        detail::w_i32(f, static_cast<int32_t>(op.params.size()));
        for (auto p  : op.params)     detail::w_i32(f, p);
    }
    return static_cast<bool>(f);
}

} // namespace mobile
} // namespace torch
