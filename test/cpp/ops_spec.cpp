// ops_spec.cpp - Concrete registry of OpSpec entries.
// Reference implementations use at::native::op on a contiguous copy,
// which exercises the strided/non-contiguous code path in spec.impl.

#include "ops_spec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ops_spec {

using at::Tensor;

// ---------------------------------------------------------------------------
// Helper: run method-style unary op on contiguous copy -> reference
// ---------------------------------------------------------------------------
namespace {

using UnaryFn = Tensor (Tensor::*)() const;
using UnaryNativeFn = Tensor (*)(const Tensor&);

// For unary ops, the reference calls the same method on a contiguous tensor,
// which guarantees a known-good path (ops work on contiguous F32 by spec).
OpSpec U(const std::string& name,
         std::function<Tensor(const Tensor&)> op,
         float tol = 1e-4f,
         bool positive_only = false,
         bool avoid_zero = false) {
    OpSpec s;
    s.name = name;
    s.category = Category::kUnary;
    s.tolerance = tol;
    s.positive_only = positive_only;
    s.avoid_zero = avoid_zero;
    s.impl = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0]);
    };
    s.reference = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0].contiguous());
    };
    return s;
}

OpSpec US(const std::string& name,
          std::function<Tensor(const Tensor&, Scalar)> op,
          Scalar arg,
          float tol = 1e-4f,
          bool positive_only = false,
          bool avoid_zero = false) {
    OpSpec s;
    s.name = name;
    s.category = Category::kUnaryScalar;
    s.tolerance = tol;
    s.scalar_arg = arg;
    s.positive_only = positive_only;
    s.avoid_zero = avoid_zero;
    s.impl = [op](const std::vector<Tensor>& in, const OpSpec& sp) -> Tensor {
        return op(in[0], sp.scalar_arg);
    };
    s.reference = [op](const std::vector<Tensor>& in, const OpSpec& sp) -> Tensor {
        return op(in[0].contiguous(), sp.scalar_arg);
    };
    return s;
}

OpSpec B(const std::string& name,
         std::function<Tensor(const Tensor&, const Tensor&)> op,
         float tol = 1e-4f,
         bool positive_only = false,
         bool avoid_zero = false) {
    OpSpec s;
    s.name = name;
    s.category = Category::kBinary;
    s.tolerance = tol;
    s.positive_only = positive_only;
    s.avoid_zero = avoid_zero;
    s.impl = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0], in[1]);
    };
    s.reference = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0].contiguous(), in[1].contiguous());
    };
    return s;
}

OpSpec RA(const std::string& name,
          std::function<Tensor(const Tensor&)> op,
          float tol = 1e-4f,
          bool positive_only = false) {
    OpSpec s;
    s.name = name;
    s.category = Category::kReduceAll;
    s.tolerance = tol;
    s.positive_only = positive_only;
    s.impl = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0]);
    };
    s.reference = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0].contiguous());
    };
    return s;
}

OpSpec RD(const std::string& name,
          std::function<Tensor(const Tensor&, int64_t)> op,
          int64_t dim,
          float tol = 1e-4f,
          bool positive_only = false) {
    OpSpec s;
    s.name = name + "_dim" + std::to_string(dim);
    s.category = Category::kReduceDim;
    s.tolerance = tol;
    s.dim = dim;
    s.positive_only = positive_only;
    s.impl = [op](const std::vector<Tensor>& in, const OpSpec& sp) -> Tensor {
        return op(in[0], sp.dim);
    };
    s.reference = [op](const std::vector<Tensor>& in, const OpSpec& sp) -> Tensor {
        return op(in[0].contiguous(), sp.dim);
    };
    return s;
}

OpSpec M(const std::string& name,
         std::function<Tensor(const Tensor&, const Tensor&)> op,
         float tol = 1e-3f) {
    OpSpec s;
    s.name = name;
    s.category = Category::kMatmul;
    s.tolerance = tol;
    s.impl = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0], in[1]);
    };
    s.reference = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0].contiguous(), in[1].contiguous());
    };
    return s;
}

OpSpec SH(const std::string& name,
          std::function<Tensor(const Tensor&)> op,
          float tol = 0.0f) {
    OpSpec s;
    s.name = name;
    s.category = Category::kShape;
    s.tolerance = tol;
    s.impl = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0]);
    };
    s.reference = [op](const std::vector<Tensor>& in, const OpSpec&) -> Tensor {
        return op(in[0].contiguous());
    };
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Unary specs (methods that take (Tensor) -> Tensor, no params)
// ---------------------------------------------------------------------------
std::vector<OpSpec> unary_specs() {
    std::vector<OpSpec> v;
    v.push_back(U("neg",        [](const Tensor& t){ return t.neg(); }));
    v.push_back(U("abs",        [](const Tensor& t){ return t.abs(); }));
    v.push_back(U("square",     [](const Tensor& t){ return t.square(); }));
    v.push_back(U("sqrt",       [](const Tensor& t){ return t.sqrt();       }, 1e-4f, true));
    v.push_back(U("rsqrt",      [](const Tensor& t){ return t.rsqrt();      }, 1e-3f, true, true));
    v.push_back(U("exp",        [](const Tensor& t){ return t.exp();        }, 1e-3f));
    v.push_back(U("log",        [](const Tensor& t){ return t.log();        }, 1e-4f, true, true));
    v.push_back(U("log2",       [](const Tensor& t){ return t.log2();       }, 1e-4f, true, true));
    v.push_back(U("log10",      [](const Tensor& t){ return t.log10();      }, 1e-4f, true, true));
    v.push_back(U("sin",        [](const Tensor& t){ return t.sin();        }, 1e-4f));
    v.push_back(U("cos",        [](const Tensor& t){ return t.cos();        }, 1e-4f));
    v.push_back(U("tan",        [](const Tensor& t){ return t.tan();        }, 1e-2f));
    v.push_back(U("tanh",       [](const Tensor& t){ return t.tanh();       }, 1e-4f));
    v.push_back(U("sigmoid",    [](const Tensor& t){ return t.sigmoid();    }, 1e-4f));
    v.push_back(U("relu",       [](const Tensor& t){ return t.relu();       }, 0.0f));
    v.push_back(U("ceil",       [](const Tensor& t){ return t.ceil();       }, 0.0f));
    v.push_back(U("floor",      [](const Tensor& t){ return t.floor();      }, 0.0f));
    v.push_back(U("round",      [](const Tensor& t){ return t.round();      }, 0.0f));
    v.push_back(U("sign",       [](const Tensor& t){ return t.sign();       }, 0.0f));
    v.push_back(U("reciprocal", [](const Tensor& t){ return t.reciprocal(); }, 1e-3f, false, true));
    return v;
}

// ---------------------------------------------------------------------------
// Unary + scalar
// ---------------------------------------------------------------------------
std::vector<OpSpec> unary_scalar_specs() {
    std::vector<OpSpec> v;
    v.push_back(US("add_scalar",   [](const Tensor& t, Scalar s){ return t.add(s); },   Scalar(1.5)));
    v.push_back(US("sub_scalar",   [](const Tensor& t, Scalar s){ return t.sub(s); },   Scalar(0.25)));
    v.push_back(US("mul_scalar",   [](const Tensor& t, Scalar s){ return t.mul(s); },   Scalar(2.5)));
    v.push_back(US("div_scalar",   [](const Tensor& t, Scalar s){ return t.div(s); },   Scalar(3.0)));
    v.push_back(US("pow_scalar",   [](const Tensor& t, Scalar s){ return t.pow(s); },   Scalar(2.0), 1e-3f));
    v.push_back(US("pow_scalar3",  [](const Tensor& t, Scalar s){ return t.pow(s); },   Scalar(3.0), 1e-2f));
    v.push_back(US("clamp_min",    [](const Tensor& t, Scalar s){ return t.clamp_min(s); }, Scalar(-0.5)));
    v.push_back(US("clamp_max",    [](const Tensor& t, Scalar s){ return t.clamp_max(s); }, Scalar(0.5)));
    v.push_back(US("eq_scalar",    [](const Tensor& t, Scalar s){ return t.eq(s); },    Scalar(0.0)));
    v.push_back(US("ne_scalar",    [](const Tensor& t, Scalar s){ return t.ne(s); },    Scalar(0.0)));
    v.push_back(US("lt_scalar",    [](const Tensor& t, Scalar s){ return t.lt(s); },    Scalar(0.0)));
    v.push_back(US("le_scalar",    [](const Tensor& t, Scalar s){ return t.le(s); },    Scalar(0.0)));
    v.push_back(US("gt_scalar",    [](const Tensor& t, Scalar s){ return t.gt(s); },    Scalar(0.0)));
    v.push_back(US("ge_scalar",    [](const Tensor& t, Scalar s){ return t.ge(s); },    Scalar(0.0)));
    return v;
}

// ---------------------------------------------------------------------------
// Binary specs (Tensor, Tensor) -> Tensor  (same shape)
// ---------------------------------------------------------------------------
std::vector<OpSpec> binary_specs() {
    std::vector<OpSpec> v;
    v.push_back(B("add",    [](const Tensor& a, const Tensor& b){ return a.add(b); }));
    v.push_back(B("sub",    [](const Tensor& a, const Tensor& b){ return a.sub(b); }));
    v.push_back(B("mul",    [](const Tensor& a, const Tensor& b){ return a.mul(b); }));
    v.push_back(B("div",    [](const Tensor& a, const Tensor& b){ return a.div(b); }, 1e-3f, false, true));
    v.push_back(B("pow",    [](const Tensor& a, const Tensor& b){ return a.pow(b); }, 1e-2f, true));
    v.push_back(B("maximum",[](const Tensor& a, const Tensor& b){ return at::maximum(a,b); }));
    v.push_back(B("minimum",[](const Tensor& a, const Tensor& b){ return at::minimum(a,b); }));
    v.push_back(B("eq",     [](const Tensor& a, const Tensor& b){ return a.eq(b); }));
    v.push_back(B("ne",     [](const Tensor& a, const Tensor& b){ return a.ne(b); }));
    v.push_back(B("lt",     [](const Tensor& a, const Tensor& b){ return a.lt(b); }));
    v.push_back(B("le",     [](const Tensor& a, const Tensor& b){ return a.le(b); }));
    v.push_back(B("gt",     [](const Tensor& a, const Tensor& b){ return a.gt(b); }));
    v.push_back(B("ge",     [](const Tensor& a, const Tensor& b){ return a.ge(b); }));
    v.push_back(B("fmod",   [](const Tensor& a, const Tensor& b){ return a.fmod(b); }, 1e-3f, false, true));
    v.push_back(B("remainder",[](const Tensor& a, const Tensor& b){ return a.remainder(b); }, 1e-3f, false, true));
    return v;
}

// ---------------------------------------------------------------------------
// Reduce-all specs: reduce entire tensor to a scalar
// ---------------------------------------------------------------------------
std::vector<OpSpec> reduce_all_specs() {
    std::vector<OpSpec> v;
    v.push_back(RA("sum",    [](const Tensor& t){ return t.sum();  }, 1e-3f));
    v.push_back(RA("mean",   [](const Tensor& t){ return t.mean(); }, 1e-4f));
    v.push_back(RA("max",    [](const Tensor& t){ return t.max();  }, 1e-5f));
    v.push_back(RA("min",    [](const Tensor& t){ return t.min();  }, 1e-5f));
    v.push_back(RA("argmax", [](const Tensor& t){ return t.argmax(); }, 0.0f));
    v.push_back(RA("argmin", [](const Tensor& t){ return t.argmin(); }, 0.0f));
    v.push_back(RA("var",    [](const Tensor& t){ return t.var(true); },  1e-3f));
    v.push_back(RA("std",    [](const Tensor& t){ return t.std(true); },  1e-3f));
    v.push_back(RA("norm",   [](const Tensor& t){ return t.norm(Scalar(2.0)); }, 1e-3f, true));
    v.push_back(RA("prod",   [](const Tensor& t){ return t.prod(); }, 1e-2f));
    return v;
}

// ---------------------------------------------------------------------------
// Reduce along dim specs (multiple ops × multiple dim choices)
// ---------------------------------------------------------------------------
std::vector<OpSpec> reduce_dim_specs() {
    std::vector<OpSpec> v;
    auto sum_dim  = [](const Tensor& t, int64_t d){ return t.sum(d, false); };
    auto mean_dim = [](const Tensor& t, int64_t d){ return t.mean(d, false); };
    auto amax_dim = [](const Tensor& t, int64_t d){ return t.argmax(d, false); };
    auto amin_dim = [](const Tensor& t, int64_t d){ return t.argmin(d, false); };
    auto var_dim  = [](const Tensor& t, int64_t d){ return t.var(d, true, false); };
    auto std_dim  = [](const Tensor& t, int64_t d){ return t.std(d, true, false); };
    auto prod_dim = [](const Tensor& t, int64_t d){ return t.prod(d, false); };
    auto csum_dim = [](const Tensor& t, int64_t d){ return t.cumsum(d); };
    auto cprod_dim= [](const Tensor& t, int64_t d){ return t.cumprod(d); };

    // dim 0 and dim 1 variants for several ops
    for (int64_t d = 0; d <= 2; ++d) {
        v.push_back(RD("sum",    sum_dim,  d, 1e-3f));
        v.push_back(RD("mean",   mean_dim, d, 1e-4f));
        v.push_back(RD("argmax", amax_dim, d, 0.0f));
        v.push_back(RD("argmin", amin_dim, d, 0.0f));
        v.push_back(RD("var",    var_dim,  d, 1e-3f));
        v.push_back(RD("std",    std_dim,  d, 1e-3f));
        v.push_back(RD("prod",   prod_dim, d, 1e-2f));
        v.push_back(RD("cumsum", csum_dim, d, 1e-3f));
        v.push_back(RD("cumprod",cprod_dim,d, 1e-2f));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Shape ops: contiguous(), clone(), flatten(), squeeze(), unsqueeze(),
//            transpose(), permute(). We re-normalize via .contiguous() in
//            tensors_near so the important thing is "same values, same shape".
// ---------------------------------------------------------------------------
std::vector<OpSpec> shape_specs() {
    std::vector<OpSpec> v;
    v.push_back(SH("clone",       [](const Tensor& t){ return t.clone(); }));
    v.push_back(SH("contiguous",  [](const Tensor& t){ return t.contiguous(); }));
    v.push_back(SH("detach",      [](const Tensor& t){ return t.detach(); }));
    v.push_back(SH("flatten",     [](const Tensor& t){ return t.flatten(); }));
    v.push_back(SH("unsqueeze0",  [](const Tensor& t){ return t.unsqueeze(0); }));
    v.push_back(SH("unsqueeze_1", [](const Tensor& t){ return t.unsqueeze(-1); }));
    v.push_back(SH("neg_neg",     [](const Tensor& t){ return t.neg().neg(); }));
    v.push_back(SH("abs_square",  [](const Tensor& t){ return t.abs().square(); }));
    return v;
}

// ---------------------------------------------------------------------------
// Matmul specs (2D)
// ---------------------------------------------------------------------------
std::vector<OpSpec> matmul_specs() {
    std::vector<OpSpec> v;
    v.push_back(M("mm",     [](const Tensor& a, const Tensor& b){ return a.mm(b); }, 2e-3f));
    v.push_back(M("matmul", [](const Tensor& a, const Tensor& b){ return a.matmul(b); }, 2e-3f));
    return v;
}

}  // namespace ops_spec
