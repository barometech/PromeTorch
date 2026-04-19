#pragma once

// ============================================================================
// OpsExpansion.h — Batch 2026-04-18
//
// Adds 50+ missing tensor ops to PromeTorch.
// Everything lives in at::native and is header-only.
// Focus: composite/derived ops that wrap existing primitives (safe & small),
// plus a handful of hot loops for stuff we couldn't route through existing
// native::* helpers.
//
// Naming convention: where a name would collide with an existing overload,
// we add `_dim` / `_along_dim` / `_reduce` suffixes. `_dim` aliases also let
// the Python bindings disambiguate.
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/MathOps.h"
#include "aten/src/ATen/native/cpu/ReduceOps.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"
#include "aten/src/ATen/native/cpu/IndexOps.h"
#include "aten/src/ATen/native/cpu/LinearAlgebra.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <tuple>
#include <vector>

namespace at {
namespace native {

// ============================================================================
// Shape / view helpers
// ============================================================================

// unsqueeze / squeeze_dim are just aliases for existing ones for clarity.
inline Tensor squeeze_dim(const Tensor& self, int64_t dim) {
    return at::native::squeeze(self, dim);
}

inline std::vector<Tensor> unbind(const Tensor& self, int64_t dim = 0) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    int64_t n = self.size(dim);
    std::vector<Tensor> out;
    out.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        // select() returns a view; call contiguous() so caller can use raw ptr.
        out.push_back(at::native::select(self, dim, i).contiguous());
    }
    return out;
}

inline Tensor movedim(const Tensor& self, int64_t src, int64_t dst) {
    int64_t ndim = self.dim();
    if (src < 0) src += ndim;
    if (dst < 0) dst += ndim;
    PT_CHECK(src >= 0 && src < ndim && dst >= 0 && dst < ndim);
    if (src == dst) return self;
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != src) perm.push_back(i);
    perm.insert(perm.begin() + dst, src);
    return at::native::permute(self, perm);
}

inline Tensor permute_dims(const Tensor& self, c10::IntArrayRef dims) {
    return at::native::permute(self, dims);
}

inline Tensor view_as(const Tensor& self, const Tensor& other) {
    return at::native::view(self, other.sizes());
}

inline Tensor reshape_as(const Tensor& self, const Tensor& other) {
    return at::native::reshape(self, other.sizes());
}

inline Tensor expand_as(const Tensor& self, const Tensor& other) {
    return at::native::expand(self, other.sizes());
}

inline Tensor broadcast_to(const Tensor& self, c10::IntArrayRef sizes) {
    return at::native::expand(self, sizes);
}

// split with explicit sizes (torch.split_with_sizes)
inline std::vector<Tensor> split_sizes(const Tensor& self,
                                       c10::IntArrayRef sizes,
                                       int64_t dim = 0) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    int64_t start = 0;
    std::vector<Tensor> out;
    out.reserve(sizes.size());
    for (int64_t s : sizes) {
        PT_CHECK_MSG(s >= 0, "split_sizes: negative split size");
        out.push_back(at::native::narrow(self, dim, start, s).contiguous());
        start += s;
    }
    PT_CHECK_MSG(start == self.size(dim),
                 "split_sizes: sum of sizes != dim size");
    return out;
}

// ============================================================================
// Padding — constant / reflect / replicate, last-dim 1D only (most common).
// For multi-dim pad, specify pair-per-dim in (begin, end) order starting from
// the LAST dim, matching torch.nn.functional.pad semantics.
// ============================================================================

namespace detail_pad {
inline std::vector<int64_t> compute_output_shape(const Tensor& self,
                                                 c10::IntArrayRef pad) {
    PT_CHECK_MSG(pad.size() % 2 == 0, "pad: length must be even");
    int64_t npad = static_cast<int64_t>(pad.size() / 2);
    int64_t ndim = self.dim();
    PT_CHECK_MSG(npad <= ndim, "pad: too many padding pairs for tensor dim");
    std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
    for (int64_t i = 0; i < npad; ++i) {
        int64_t d = ndim - 1 - i;
        out_shape[d] += pad[2 * i] + pad[2 * i + 1];
    }
    return out_shape;
}
}  // namespace detail_pad

inline Tensor pad_constant(const Tensor& self,
                           c10::IntArrayRef pad,
                           Scalar value = Scalar(0)) {
    auto out_shape = detail_pad::compute_output_shape(self, pad);
    Tensor self_c = self.is_contiguous() ? self : self.contiguous();
    Tensor out = at::full(out_shape,
                          value,
                          TensorOptions().dtype(self.dtype()).device(self.device()));
    int64_t ndim = self.dim();
    int64_t npad = static_cast<int64_t>(pad.size() / 2);
    std::vector<int64_t> begin(ndim, 0);
    for (int64_t i = 0; i < npad; ++i) {
        int64_t d = ndim - 1 - i;
        begin[d] = pad[2 * i];
    }

    PT_DISPATCH_ALL_TYPES(self.dtype(), "pad_constant", [&] {
        const scalar_t* src = self_c.data_ptr<scalar_t>();
        scalar_t* dst = out.mutable_data_ptr<scalar_t>();
        int64_t n = self_c.numel();
        std::vector<int64_t> coord(ndim, 0);
        for (int64_t i = 0; i < n; ++i) {
            // linearize src coord -> dst offset with begin shift
            int64_t rem = i;
            int64_t dst_off = 0;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t c = rem % self_c.size(d);
                rem /= self_c.size(d);
                coord[d] = c;
                dst_off += (c + begin[d]) * out.stride(d);
            }
            dst[dst_off] = src[i];
        }
    });
    return out;
}

inline Tensor pad(const Tensor& self,
                  c10::IntArrayRef pad,
                  const std::string& mode = "constant",
                  Scalar value = Scalar(0)) {
    if (mode == "constant") return pad_constant(self, pad, value);
    // reflect / replicate: fall through to explicit helpers below.
    PT_CHECK_MSG(mode == "reflect" || mode == "replicate",
                 "pad: unknown mode (constant/reflect/replicate only)");

    auto out_shape = detail_pad::compute_output_shape(self, pad);
    Tensor self_c = self.is_contiguous() ? self : self.contiguous();
    Tensor out = empty(out_shape,
                       TensorOptions().dtype(self.dtype()).device(self.device()));
    int64_t ndim = self.dim();
    int64_t npad = static_cast<int64_t>(pad.size() / 2);
    std::vector<int64_t> begin(ndim, 0);
    for (int64_t i = 0; i < npad; ++i) {
        int64_t d = ndim - 1 - i;
        begin[d] = pad[2 * i];
    }

    PT_DISPATCH_ALL_TYPES(self.dtype(), "pad_reflect_replicate", [&] {
        const scalar_t* src = self_c.data_ptr<scalar_t>();
        scalar_t* dst = out.mutable_data_ptr<scalar_t>();
        int64_t on = out.numel();
        bool reflect = (mode == "reflect");
        for (int64_t i = 0; i < on; ++i) {
            int64_t rem = i;
            int64_t src_off = 0;
            bool valid = true;
            for (int64_t d = ndim - 1; d >= 0 && valid; --d) {
                int64_t oc = rem % out.size(d);
                rem /= out.size(d);
                int64_t sc = oc - begin[d];
                int64_t dim_sz = self_c.size(d);
                if (sc < 0) {
                    if (reflect) sc = -sc;
                    else sc = 0;  // replicate
                } else if (sc >= dim_sz) {
                    if (reflect) sc = 2 * (dim_sz - 1) - sc;
                    else sc = dim_sz - 1;
                }
                if (sc < 0 || sc >= dim_sz) { valid = false; break; }
                src_off += sc * self_c.stride(d);
            }
            dst[i] = valid ? src[src_off] : scalar_t{};
        }
    });
    return out;
}

inline Tensor pad_reflect(const Tensor& self, c10::IntArrayRef p) {
    return pad(self, p, "reflect");
}

inline Tensor pad_replicate(const Tensor& self, c10::IntArrayRef p) {
    return pad(self, p, "replicate");
}

// 1D sliding window. Returns tensor of shape (..., n_windows, window_size).
inline Tensor unfold_window(const Tensor& self,
                            int64_t dim,
                            int64_t size,
                            int64_t step) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(size > 0 && step > 0, "unfold_window: size/step must be > 0");
    int64_t src_len = self.size(dim);
    PT_CHECK_MSG(size <= src_len, "unfold_window: window larger than dim");

    int64_t n_windows = (src_len - size) / step + 1;
    std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
    out_shape[dim] = n_windows;
    out_shape.push_back(size);

    Tensor self_c = self.is_contiguous() ? self : self.contiguous();
    Tensor out = empty(out_shape,
                       TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "unfold_window", [&] {
        const scalar_t* src = self_c.data_ptr<scalar_t>();
        scalar_t* dst = out.mutable_data_ptr<scalar_t>();

        int64_t outer = 1;
        for (int64_t d = 0; d < dim; ++d) outer *= self_c.size(d);
        int64_t inner = 1;
        for (int64_t d = dim + 1; d < ndim; ++d) inner *= self_c.size(d);

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t w = 0; w < n_windows; ++w) {
                for (int64_t k = 0; k < size; ++k) {
                    int64_t src_pos = (o * src_len + w * step + k) * inner;
                    int64_t dst_pos = ((o * n_windows + w) * size + k) * inner;
                    std::memcpy(dst + dst_pos,
                                src + src_pos,
                                inner * sizeof(scalar_t));
                }
            }
        }
    });
    return out;
}

// Inverse of unfold_window along the last two axes. input: (..., n_windows, size)
inline Tensor fold_window(const Tensor& windows,
                          int64_t dim,
                          int64_t src_len,
                          int64_t step) {
    int64_t ndim = windows.dim();
    PT_CHECK_MSG(ndim >= 2, "fold_window: need >=2 dim tensor");
    if (dim < 0) dim += ndim - 1;  // output has 1 fewer dims
    PT_CHECK(dim >= 0 && dim < ndim - 1);

    int64_t n_windows = windows.size(ndim - 2);
    int64_t size      = windows.size(ndim - 1);

    std::vector<int64_t> out_shape(windows.sizes().begin(),
                                   windows.sizes().end() - 1);
    out_shape[dim] = src_len;

    Tensor out = zeros(out_shape,
                       TensorOptions().dtype(windows.dtype()).device(windows.device()));
    Tensor wc = windows.is_contiguous() ? windows : windows.contiguous();

    PT_DISPATCH_ALL_TYPES(windows.dtype(), "fold_window", [&] {
        const scalar_t* src = wc.data_ptr<scalar_t>();
        scalar_t* dst = out.mutable_data_ptr<scalar_t>();
        int64_t outer = 1;
        for (int64_t d = 0; d < dim; ++d) outer *= out.size(d);
        int64_t inner = 1;
        for (int64_t d = dim + 1; d < (int64_t)out_shape.size(); ++d) inner *= out.size(d);
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t w = 0; w < n_windows; ++w) {
                for (int64_t k = 0; k < size; ++k) {
                    int64_t dst_pos = (o * src_len + w * step + k) * inner;
                    int64_t src_pos = ((o * n_windows + w) * size + k) * inner;
                    if (w * step + k < src_len) {
                        for (int64_t j = 0; j < inner; ++j)
                            dst[dst_pos + j] += src[src_pos + j];
                    }
                }
            }
        }
    });
    return out;
}

// tile() — repeat along every axis
inline Tensor tile(const Tensor& self, c10::IntArrayRef reps) {
    // If fewer reps than ndim, prepend 1s; if more, unsqueeze.
    int64_t ndim = self.dim();
    int64_t nrep = static_cast<int64_t>(reps.size());
    Tensor t = self;
    while (t.dim() < nrep) t = at::native::unsqueeze(t, 0);
    std::vector<int64_t> rep_full;
    rep_full.reserve(t.dim());
    int64_t pad = t.dim() - nrep;
    for (int64_t i = 0; i < pad; ++i) rep_full.push_back(1);
    for (int64_t r : reps) rep_full.push_back(r);
    (void)ndim;
    return at::native::repeat(t, rep_full);
}

// ============================================================================
// Cat / stack helpers
// ============================================================================

inline Tensor stack_along_dim(const std::vector<Tensor>& ts, int64_t dim = 0) {
    return at::native::stack(ts, dim);
}

inline std::vector<Tensor> meshgrid_ij(const std::vector<Tensor>& ts) {
    return at::native::meshgrid(ts, "ij");
}

inline Tensor cartesian_prod(const std::vector<Tensor>& ts) {
    PT_CHECK_MSG(!ts.empty(), "cartesian_prod: empty list");
    // Each input must be 1D.
    int64_t total = 1;
    for (const auto& t : ts) {
        PT_CHECK_MSG(t.dim() == 1, "cartesian_prod: inputs must be 1D");
        total *= t.size(0);
    }
    int64_t n = static_cast<int64_t>(ts.size());
    Tensor out = empty({total, n},
                       TensorOptions().dtype(ts[0].dtype()).device(ts[0].device()));

    PT_DISPATCH_ALL_TYPES(ts[0].dtype(), "cartesian_prod", [&] {
        scalar_t* dst = out.mutable_data_ptr<scalar_t>();
        std::vector<const scalar_t*> ptrs(n);
        std::vector<int64_t> sizes(n);
        for (int64_t i = 0; i < n; ++i) {
            Tensor c = ts[i].contiguous();
            ptrs[i] = c.data_ptr<scalar_t>();
            sizes[i] = c.size(0);
        }
        for (int64_t row = 0; row < total; ++row) {
            int64_t rem = row;
            for (int64_t c = n - 1; c >= 0; --c) {
                int64_t idx = rem % sizes[c];
                rem /= sizes[c];
                dst[row * n + c] = ptrs[c][idx];
            }
        }
    });
    return out;
}

// ============================================================================
// Index family: *_dim aliases and new index_add / index_copy / masked_scatter
// ============================================================================

inline Tensor gather_dim(const Tensor& self, int64_t dim, const Tensor& index) {
    return at::native::gather(self, dim, index);
}

inline Tensor scatter_dim(const Tensor& self,
                          int64_t dim,
                          const Tensor& index,
                          const Tensor& src) {
    return at::native::scatter(self, dim, index, src);
}

inline Tensor scatter_add_dim(const Tensor& self,
                              int64_t dim,
                              const Tensor& index,
                              const Tensor& src) {
    Tensor out = self.clone();
    at::native::scatter_add_(out, dim, index, src);
    return out;
}

// out-of-place index_add: out = self; out[index along dim] += src rows
inline Tensor index_add(const Tensor& self,
                        int64_t dim,
                        const Tensor& index,
                        const Tensor& src) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
                 "index_add: index must be Long");
    PT_CHECK_MSG(index.dim() == 1, "index_add: index must be 1D");

    Tensor out = self.clone();
    Tensor src_c = src.is_contiguous() ? src : src.contiguous();
    const int64_t* idx = index.data_ptr<int64_t>();
    int64_t ni = index.size(0);
    PT_CHECK_MSG(src.size(dim) == ni,
                 "index_add: src.size(dim) must equal index length");

    PT_DISPATCH_ALL_TYPES(self.dtype(), "index_add", [&] {
        scalar_t* dst = out.mutable_data_ptr<scalar_t>();
        const scalar_t* sp = src_c.data_ptr<scalar_t>();
        // slab = product of dims except `dim`
        int64_t outer = 1;
        for (int64_t d = 0; d < dim; ++d) outer *= out.size(d);
        int64_t inner = 1;
        for (int64_t d = dim + 1; d < ndim; ++d) inner *= out.size(d);
        int64_t dim_sz = out.size(dim);
        for (int64_t i = 0; i < ni; ++i) {
            int64_t target = idx[i];
            PT_CHECK_MSG(target >= 0 && target < dim_sz,
                         "index_add: index out of range");
            for (int64_t o = 0; o < outer; ++o) {
                const scalar_t* src_row = sp + (o * ni + i) * inner;
                scalar_t* dst_row = dst + (o * dim_sz + target) * inner;
                for (int64_t j = 0; j < inner; ++j) dst_row[j] += src_row[j];
            }
        }
    });
    return out;
}

// index_copy: out = self; out[index along dim] = src rows
inline Tensor index_copy(const Tensor& self,
                         int64_t dim,
                         const Tensor& index,
                         const Tensor& src) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK_MSG(index.dtype() == c10::ScalarType::Long,
                 "index_copy: index must be Long");
    PT_CHECK_MSG(index.dim() == 1, "index_copy: index must be 1D");

    Tensor out = self.clone();
    Tensor src_c = src.is_contiguous() ? src : src.contiguous();
    const int64_t* idx = index.data_ptr<int64_t>();
    int64_t ni = index.size(0);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "index_copy", [&] {
        scalar_t* dst = out.mutable_data_ptr<scalar_t>();
        const scalar_t* sp = src_c.data_ptr<scalar_t>();
        int64_t outer = 1;
        for (int64_t d = 0; d < dim; ++d) outer *= out.size(d);
        int64_t inner = 1;
        for (int64_t d = dim + 1; d < ndim; ++d) inner *= out.size(d);
        int64_t dim_sz = out.size(dim);
        for (int64_t i = 0; i < ni; ++i) {
            int64_t target = idx[i];
            PT_CHECK_MSG(target >= 0 && target < dim_sz,
                         "index_copy: index out of range");
            for (int64_t o = 0; o < outer; ++o) {
                const scalar_t* src_row = sp + (o * ni + i) * inner;
                scalar_t* dst_row = dst + (o * dim_sz + target) * inner;
                std::memcpy(dst_row, src_row, inner * sizeof(scalar_t));
            }
        }
    });
    return out;
}

// masked_scatter: replace self's elements where mask=true with consecutive
// values from source (flat view). Returns a new tensor.
inline Tensor masked_scatter(const Tensor& self,
                             const Tensor& mask,
                             const Tensor& source) {
    PT_CHECK_MSG(mask.dtype() == c10::ScalarType::Bool,
                 "masked_scatter: mask must be Bool");
    PT_CHECK_MSG(mask.sizes() == self.sizes(),
                 "masked_scatter: mask must match self shape");
    Tensor out = self.clone();
    Tensor mask_c = mask.is_contiguous() ? mask : mask.contiguous();
    Tensor src_c = source.is_contiguous() ? source : source.contiguous();
    const bool* mp = mask_c.data_ptr<bool>();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "masked_scatter", [&] {
        scalar_t* dst = out.mutable_data_ptr<scalar_t>();
        const scalar_t* sp = src_c.data_ptr<scalar_t>();
        int64_t n = out.numel();
        int64_t src_pos = 0;
        int64_t src_n = src_c.numel();
        for (int64_t i = 0; i < n; ++i) {
            if (mp[i]) {
                PT_CHECK_MSG(src_pos < src_n,
                             "masked_scatter: source too short");
                dst[i] = sp[src_pos++];
            }
        }
    });
    return out;
}

// ============================================================================
// Reduce helpers — dim-variants and cum ops
// ============================================================================

inline Tensor argmax_dim(const Tensor& self, int64_t dim, bool keepdim = false) {
    return at::native::argmax(self, dim, keepdim);
}
inline Tensor argmin_dim(const Tensor& self, int64_t dim, bool keepdim = false) {
    return at::native::argmin(self, dim, keepdim);
}

inline std::tuple<Tensor, Tensor> topk_along_dim(const Tensor& self,
                                                 int64_t k,
                                                 int64_t dim = -1,
                                                 bool largest = true,
                                                 bool sorted = true) {
    return at::native::topk(self, k, dim, largest, sorted);
}

// Tensor-returning flavors of all()/any() (bool scalar -> {1} Tensor)
inline Tensor all_reduce(const Tensor& self) {
    Tensor out = empty({1},
                       TensorOptions().dtype(c10::ScalarType::Bool).device(self.device()));
    out.mutable_data_ptr<bool>()[0] = at::native::all(self);
    return out;
}

inline Tensor any_reduce(const Tensor& self) {
    Tensor out = empty({1},
                       TensorOptions().dtype(c10::ScalarType::Bool).device(self.device()));
    out.mutable_data_ptr<bool>()[0] = at::native::any(self);
    return out;
}

// ============================================================================
// Linear algebra helpers
// ============================================================================

inline Tensor diagonal(const Tensor& self, int64_t offset = 0) {
    // extract diagonal of 2D matrix (returns 1D tensor)
    PT_CHECK_MSG(self.dim() == 2, "diagonal: only 2D supported");
    return at::native::diag(self, offset);
}

inline Tensor outer_product(const Tensor& a, const Tensor& b) {
    return at::native::outer(a, b);
}

// Kronecker product, only 2D inputs (most common case)
inline Tensor kron(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.dim() == 2 && b.dim() == 2,
                 "kron: only 2D supported in OpsExpansion (expand later)");
    int64_t ar = a.size(0), ac = a.size(1);
    int64_t br = b.size(0), bc = b.size(1);
    Tensor out = empty({ar * br, ac * bc},
                       TensorOptions().dtype(a.dtype()).device(a.device()));
    Tensor ac_t = a.is_contiguous() ? a : a.contiguous();
    Tensor bc_t = b.is_contiguous() ? b : b.contiguous();

    PT_DISPATCH_FLOATING_TYPES(a.dtype(), "kron", [&] {
        const scalar_t* ap = ac_t.data_ptr<scalar_t>();
        const scalar_t* bp = bc_t.data_ptr<scalar_t>();
        scalar_t* op = out.mutable_data_ptr<scalar_t>();
        for (int64_t ai = 0; ai < ar; ++ai) {
            for (int64_t bi = 0; bi < br; ++bi) {
                for (int64_t aj = 0; aj < ac; ++aj) {
                    scalar_t av = ap[ai * ac + aj];
                    for (int64_t bj = 0; bj < bc; ++bj) {
                        int64_t out_r = ai * br + bi;
                        int64_t out_c = aj * bc + bj;
                        op[out_r * (ac * bc) + out_c] = av * bp[bi * bc + bj];
                    }
                }
            }
        }
    });
    return out;
}

// ============================================================================
// Logical ops
// ============================================================================

namespace detail_logical {

inline Tensor bool_binary(const Tensor& a, const Tensor& b,
                          const char* name,
                          bool (*fn)(bool, bool)) {
    PT_CHECK_MSG(a.sizes() == b.sizes(),
                 "logical op: shape mismatch (broadcasting via expand first)");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    Tensor out = empty(a.sizes(),
                       TensorOptions().dtype(c10::ScalarType::Bool).device(a.device()));
    bool* op = out.mutable_data_ptr<bool>();
    int64_t n = a.numel();

    auto to_bool = [](const Tensor& t, int64_t i) -> bool {
        // Any-nonzero interpretation, like torch.
        bool r = false;
        PT_DISPATCH_ALL_TYPES(t.dtype(), "to_bool", [&] {
            const scalar_t* d = t.data_ptr<scalar_t>();
            r = d[i] != static_cast<scalar_t>(0);
        });
        return r;
    };
    (void)name;

    for (int64_t i = 0; i < n; ++i) {
        op[i] = fn(to_bool(ac, i), to_bool(bc, i));
    }
    return out;
}

}  // namespace detail_logical

inline Tensor logical_and(const Tensor& a, const Tensor& b) {
    return detail_logical::bool_binary(a, b, "logical_and",
        [](bool x, bool y) { return x && y; });
}
inline Tensor logical_or(const Tensor& a, const Tensor& b) {
    return detail_logical::bool_binary(a, b, "logical_or",
        [](bool x, bool y) { return x || y; });
}
inline Tensor logical_xor(const Tensor& a, const Tensor& b) {
    return detail_logical::bool_binary(a, b, "logical_xor",
        [](bool x, bool y) { return x != y; });
}
inline Tensor logical_not(const Tensor& a) {
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor out = empty(a.sizes(),
                       TensorOptions().dtype(c10::ScalarType::Bool).device(a.device()));
    bool* op = out.mutable_data_ptr<bool>();
    int64_t n = a.numel();
    PT_DISPATCH_ALL_TYPES(a.dtype(), "logical_not", [&] {
        const scalar_t* d = ac.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) op[i] = !(d[i] != static_cast<scalar_t>(0));
    });
    return out;
}

// ============================================================================
// Float classification / closeness
// ============================================================================

inline Tensor isfinite(const Tensor& self) {
    Tensor sc = self.is_contiguous() ? self : self.contiguous();
    Tensor out = empty(self.sizes(),
                       TensorOptions().dtype(c10::ScalarType::Bool).device(self.device()));
    bool* op = out.mutable_data_ptr<bool>();
    int64_t n = self.numel();
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "isfinite", [&] {
        const scalar_t* d = sc.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) op[i] = std::isfinite(d[i]);
    });
    return out;
}

inline Tensor isinf(const Tensor& self) {
    Tensor sc = self.is_contiguous() ? self : self.contiguous();
    Tensor out = empty(self.sizes(),
                       TensorOptions().dtype(c10::ScalarType::Bool).device(self.device()));
    bool* op = out.mutable_data_ptr<bool>();
    int64_t n = self.numel();
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "isinf", [&] {
        const scalar_t* d = sc.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) op[i] = std::isinf(d[i]);
    });
    return out;
}

inline Tensor isnan(const Tensor& self) {
    Tensor sc = self.is_contiguous() ? self : self.contiguous();
    Tensor out = empty(self.sizes(),
                       TensorOptions().dtype(c10::ScalarType::Bool).device(self.device()));
    bool* op = out.mutable_data_ptr<bool>();
    int64_t n = self.numel();
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "isnan", [&] {
        const scalar_t* d = sc.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) op[i] = std::isnan(d[i]);
    });
    return out;
}

inline Tensor isclose(const Tensor& a, const Tensor& b,
                      double rtol = 1e-5,
                      double atol = 1e-8,
                      bool equal_nan = false) {
    PT_CHECK_MSG(a.sizes() == b.sizes(),
                 "isclose: shape mismatch (expand first)");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    Tensor out = empty(a.sizes(),
                       TensorOptions().dtype(c10::ScalarType::Bool).device(a.device()));
    bool* op = out.mutable_data_ptr<bool>();
    int64_t n = a.numel();
    PT_DISPATCH_FLOATING_TYPES(a.dtype(), "isclose", [&] {
        const scalar_t* ap = ac.data_ptr<scalar_t>();
        const scalar_t* bp = bc.data_ptr<scalar_t>();
        scalar_t rtol_s = static_cast<scalar_t>(rtol);
        scalar_t atol_s = static_cast<scalar_t>(atol);
        for (int64_t i = 0; i < n; ++i) {
            scalar_t x = ap[i], y = bp[i];
            if (std::isnan(x) && std::isnan(y)) { op[i] = equal_nan; continue; }
            scalar_t diff = std::abs(x - y);
            scalar_t tol = atol_s + rtol_s * std::abs(y);
            op[i] = diff <= tol;
        }
    });
    return out;
}

// ============================================================================
// Misc elementwise: lerp / hypot / atan2
// ============================================================================

inline Tensor lerp(const Tensor& start, const Tensor& end, Scalar weight) {
    PT_CHECK_MSG(start.sizes() == end.sizes(), "lerp: shape mismatch");
    Tensor sc = start.is_contiguous() ? start : start.contiguous();
    Tensor ec = end.is_contiguous() ? end : end.contiguous();
    Tensor out = empty(start.sizes(),
                       TensorOptions().dtype(start.dtype()).device(start.device()));
    PT_DISPATCH_FLOATING_TYPES(start.dtype(), "lerp", [&] {
        const scalar_t* sp = sc.data_ptr<scalar_t>();
        const scalar_t* ep = ec.data_ptr<scalar_t>();
        scalar_t* op = out.mutable_data_ptr<scalar_t>();
        scalar_t w = weight.to<scalar_t>();
        int64_t n = start.numel();
        for (int64_t i = 0; i < n; ++i) op[i] = sp[i] + w * (ep[i] - sp[i]);
    });
    return out;
}

inline Tensor hypot(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.sizes() == b.sizes(), "hypot: shape mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    Tensor out = empty(a.sizes(),
                       TensorOptions().dtype(a.dtype()).device(a.device()));
    PT_DISPATCH_FLOATING_TYPES(a.dtype(), "hypot", [&] {
        const scalar_t* ap = ac.data_ptr<scalar_t>();
        const scalar_t* bp = bc.data_ptr<scalar_t>();
        scalar_t* op = out.mutable_data_ptr<scalar_t>();
        int64_t n = a.numel();
        for (int64_t i = 0; i < n; ++i) op[i] = std::hypot(ap[i], bp[i]);
    });
    return out;
}

inline Tensor atan2(const Tensor& y, const Tensor& x) {
    PT_CHECK_MSG(y.sizes() == x.sizes(), "atan2: shape mismatch");
    Tensor yc = y.is_contiguous() ? y : y.contiguous();
    Tensor xc = x.is_contiguous() ? x : x.contiguous();
    Tensor out = empty(y.sizes(),
                       TensorOptions().dtype(y.dtype()).device(y.device()));
    PT_DISPATCH_FLOATING_TYPES(y.dtype(), "atan2", [&] {
        const scalar_t* yp = yc.data_ptr<scalar_t>();
        const scalar_t* xp = xc.data_ptr<scalar_t>();
        scalar_t* op = out.mutable_data_ptr<scalar_t>();
        int64_t n = y.numel();
        for (int64_t i = 0; i < n; ++i) op[i] = std::atan2(yp[i], xp[i]);
    });
    return out;
}

// ============================================================================
// one_hot — convert LongTensor [..., N] -> [..., N, num_classes] with 1 at idx.
// If num_classes == -1 we infer from max value in indices.
// ============================================================================

inline Tensor one_hot(const Tensor& indices, int64_t num_classes = -1) {
    PT_CHECK_MSG(indices.dtype() == c10::ScalarType::Long,
                 "one_hot: indices must be LongTensor");
    Tensor idx = indices.is_contiguous() ? indices : indices.contiguous();
    const int64_t* idx_data = idx.data_ptr<int64_t>();
    int64_t n = idx.numel();

    if (num_classes < 0) {
        int64_t m = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (idx_data[i] > m) m = idx_data[i];
        }
        num_classes = m + 1;
    }
    PT_CHECK_MSG(num_classes > 0, "one_hot: num_classes must be > 0");

    std::vector<int64_t> out_shape(idx.sizes().begin(), idx.sizes().end());
    out_shape.push_back(num_classes);

    Tensor out = zeros(out_shape,
                       TensorOptions().dtype(c10::ScalarType::Long).device(indices.device()));
    int64_t* op = out.mutable_data_ptr<int64_t>();

    for (int64_t i = 0; i < n; ++i) {
        int64_t c = idx_data[i];
        PT_CHECK_MSG(c >= 0 && c < num_classes,
                     "one_hot: index out of range [0, num_classes)");
        op[i * num_classes + c] = 1;
    }
    return out;
}

// ============================================================================
// allclose — single bool, reduction of isclose().
// ============================================================================

inline bool allclose(const Tensor& a, const Tensor& b,
                     double rtol = 1e-5,
                     double atol = 1e-8,
                     bool equal_nan = false) {
    PT_CHECK_MSG(a.sizes() == b.sizes(), "allclose: shape mismatch");
    Tensor mask = isclose(a, b, rtol, atol, equal_nan);
    const bool* mp = mask.data_ptr<bool>();
    int64_t n = mask.numel();
    for (int64_t i = 0; i < n; ++i) {
        if (!mp[i]) return false;
    }
    return true;
}

// ============================================================================
// equal — exact element-wise equality reduced to a single bool.
// Shape mismatch -> false (matches torch.equal).
// ============================================================================

inline bool equal(const Tensor& a, const Tensor& b) {
    if (a.sizes() != b.sizes()) return false;
    if (a.dtype() != b.dtype()) return false;
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    bool out = true;
    int64_t n = ac.numel();
    PT_DISPATCH_ALL_TYPES(ac.dtype(), "equal", [&] {
        const scalar_t* ap = ac.data_ptr<scalar_t>();
        const scalar_t* bp = bc.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n && out; ++i) {
            if (ap[i] != bp[i]) out = false;
        }
    });
    return out;
}

// ============================================================================
// floor_divide — element-wise a // b.
// For integer dtypes we follow Python floor-division semantics (round toward
// -inf, not toward zero), which differs from C's trunc-divide for negatives.
// For float dtypes we use std::floor on the true quotient.
// ============================================================================

inline Tensor floor_divide(const Tensor& a, const Tensor& b) {
    PT_CHECK_MSG(a.sizes() == b.sizes(),
                 "floor_divide: shape mismatch (broadcast first)");
    PT_CHECK_MSG(a.dtype() == b.dtype(),
                 "floor_divide: dtype mismatch");
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    Tensor out = empty(a.sizes(),
                       TensorOptions().dtype(a.dtype()).device(a.device()));
    int64_t n = ac.numel();
    c10::ScalarType dt = a.dtype();

    if (dt == c10::ScalarType::Float || dt == c10::ScalarType::Double) {
        PT_DISPATCH_FLOATING_TYPES(dt, "floor_divide_float", [&] {
            const scalar_t* ap = ac.data_ptr<scalar_t>();
            const scalar_t* bp = bc.data_ptr<scalar_t>();
            scalar_t* op = out.mutable_data_ptr<scalar_t>();
            for (int64_t i = 0; i < n; ++i) {
                op[i] = std::floor(ap[i] / bp[i]);
            }
        });
    } else {
        PT_DISPATCH_ALL_TYPES(dt, "floor_divide_int", [&] {
            const scalar_t* ap = ac.data_ptr<scalar_t>();
            const scalar_t* bp = bc.data_ptr<scalar_t>();
            scalar_t* op = out.mutable_data_ptr<scalar_t>();
            for (int64_t i = 0; i < n; ++i) {
                scalar_t x = ap[i], y = bp[i];
                PT_CHECK_MSG(y != 0, "floor_divide: integer divide by zero");
                // Python-style floor division: truncate then adjust if remainder
                // has opposite sign of divisor.
                scalar_t q = x / y;
                scalar_t r = x - q * y;
                if ((r != 0) && ((r < 0) != (y < 0))) q -= 1;
                op[i] = q;
            }
        });
    }
    return out;
}

}  // namespace native
}  // namespace at
