#pragma once

// ============================================================================
// Einstein Summation (einsum) — full implementation
// ----------------------------------------------------------------------------
// Supports:
//   - Arbitrary number of operands (greedy pairwise contraction).
//   - Ellipsis notation "..." for broadcast/batch dims.
//   - Implicit output (omit "->").
//   - Repeated labels within one operand (diagonal extraction).
//   - Labels that are summed out, carried forward (free), or batched.
// ============================================================================

// NOTE: this header depends on symbols (bmm, permute, mm) that come from
// LinearAlgebra.h and ShapeOps.h. To avoid a circular include with
// LinearAlgebra.h (which pulls in Einsum.h at its tail), we intentionally do
// NOT re-include LinearAlgebra.h here. Callers must always reach Einsum.h
// through LinearAlgebra.h (or through ATen.h, which includes it).
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/ReduceOps.h"
#include "aten/src/ATen/native/cpu/MathOps.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace at {
namespace native {
namespace einsum_full {

// We use `char` labels for user-provided subscripts (a-zA-Z) and an internal
// "extended" alphabet of wider integers to represent expanded ellipsis dims.
// Strategy: convert each operand's subscript string into a std::vector<int>
// where user labels map to their char value (>0) and ellipsis dims map to
// negative sentinels (-1, -2, ...) that are unique per logical ellipsis
// position (so each ellipsis axis becomes its own label, shared across the
// equation in left-to-right aligned broadcast order).
//
// This lets the rest of the algorithm treat ellipsis axes identically to
// regular labels (same permute / reshape / bmm machinery).

using Label = int;

struct Parsed {
    std::vector<std::vector<Label>> inputs;   // per-operand label lists
    std::vector<Label>              output;   // output label list
    std::unordered_map<Label, int64_t> sizes; // label -> dim size
};

// Parse "ab,bc->ac" style (with optional ellipsis). `shapes` is needed to
// resolve how many axes each `...` covers on each operand.
inline Parsed parse(const std::string& eq_in,
                    const std::vector<std::vector<int64_t>>& shapes) {
    // Strip whitespace.
    std::string eq;
    eq.reserve(eq_in.size());
    for (char c : eq_in) if (c != ' ' && c != '\t') eq.push_back(c);

    // Split lhs / rhs on "->".
    std::string lhs, rhs;
    bool has_arrow = false;
    auto arrow = eq.find("->");
    if (arrow != std::string::npos) {
        lhs = eq.substr(0, arrow);
        rhs = eq.substr(arrow + 2);
        has_arrow = true;
    } else {
        lhs = eq;
    }

    // Split lhs on "," into operand subscripts (raw strings, still contain "...").
    std::vector<std::string> raw_ops;
    {
        std::string cur;
        for (char c : lhs) {
            if (c == ',') { raw_ops.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        raw_ops.push_back(cur);
    }
    PT_CHECK_MSG(raw_ops.size() == shapes.size(),
        "einsum: number of subscripts (", raw_ops.size(),
        ") does not match number of operands (", shapes.size(), ")");

    // Determine ellipsis width: max axes covered by "..." across operands.
    int64_t ell_width = 0;
    std::vector<int> ell_count_per_op(raw_ops.size(), 0);
    std::vector<int> named_count_per_op(raw_ops.size(), 0);
    for (size_t i = 0; i < raw_ops.size(); ++i) {
        int named = 0;
        int dots = 0;
        for (size_t p = 0; p < raw_ops[i].size(); ) {
            if (p + 2 < raw_ops[i].size() && raw_ops[i].compare(p, 3, "...") == 0) {
                dots++;
                p += 3;
            } else {
                named++;
                p += 1;
            }
        }
        PT_CHECK_MSG(dots <= 1, "einsum: operand ", i, " has more than one ellipsis");
        ell_count_per_op[i] = dots;
        named_count_per_op[i] = named;
        int64_t rank = (int64_t)shapes[i].size();
        if (dots == 1) {
            int64_t cover = rank - named;
            PT_CHECK_MSG(cover >= 0,
                "einsum: operand ", i, " rank ", rank,
                " too small for subscript with ", named, " named axes");
            ell_width = std::max(ell_width, cover);
        } else {
            PT_CHECK_MSG((int64_t)named == rank,
                "einsum: operand ", i, " subscript has ", named,
                " labels but tensor has rank ", rank);
        }
    }

    // Assign sentinel labels -1, -2, ... for ellipsis axes (right-aligned),
    // i.e. last ellipsis axis = -1, second-to-last = -2, etc.
    // All operands share these sentinels for broadcast matching.
    auto ellipsis_label = [&](int64_t idx_from_right) -> Label {
        return -(Label)(idx_from_right + 1); // idx 0 -> -1, idx 1 -> -2, ...
    };

    Parsed out;
    out.inputs.resize(raw_ops.size());

    // Build per-operand label lists and register sizes.
    for (size_t i = 0; i < raw_ops.size(); ++i) {
        int64_t rank = (int64_t)shapes[i].size();
        int64_t cover = (ell_count_per_op[i] == 1)
                        ? (rank - named_count_per_op[i]) : 0;

        std::vector<Label>& labs = out.inputs[i];
        labs.reserve(rank);

        // Walk raw_ops[i] and expand ellipsis into `ell_width` sentinel labels.
        // Leading ellipsis axes align to the RIGHT side of ell_width window,
        // so if cover < ell_width, the operand broadcasts over the leftmost
        // (ell_width - cover) ellipsis positions. We skip those (they don't
        // exist in this operand).
        size_t p = 0;
        int axis = 0;
        while (p < raw_ops[i].size()) {
            if (p + 2 < raw_ops[i].size() && raw_ops[i].compare(p, 3, "...") == 0) {
                // Expand `cover` ellipsis axes using rightmost sentinels.
                for (int64_t k = 0; k < cover; ++k) {
                    int64_t idx_from_right = cover - 1 - k;
                    labs.push_back(ellipsis_label(idx_from_right));
                    axis++;
                }
                p += 3;
            } else {
                Label L = (Label)(unsigned char)raw_ops[i][p];
                labs.push_back(L);
                axis++;
                p += 1;
            }
        }
        PT_CHECK_MSG((int64_t)labs.size() == rank,
            "einsum: operand ", i, " label count != rank after expansion");

        // Register sizes, check consistency.
        for (int64_t d = 0; d < rank; ++d) {
            Label L = labs[d];
            int64_t sz = shapes[i][d];
            auto it = out.sizes.find(L);
            if (it == out.sizes.end()) {
                out.sizes[L] = sz;
            } else {
                // Broadcasting: allow size-1 to match any size (like PyTorch).
                if (it->second == 1)      it->second = sz;
                else if (sz == 1)         { /* ok */ }
                else PT_CHECK_MSG(it->second == sz,
                    "einsum: size mismatch for label (dim size ",
                    it->second, " vs ", sz, ")");
            }
        }
    }

    // Output labels.
    if (has_arrow) {
        // Parse rhs.
        int rhs_dots = 0, rhs_named = 0;
        for (size_t p = 0; p < rhs.size(); ) {
            if (p + 2 < rhs.size() && rhs.compare(p, 3, "...") == 0) {
                rhs_dots++;
                p += 3;
            } else {
                rhs_named++;
                p += 1;
            }
        }
        PT_CHECK_MSG(rhs_dots <= 1, "einsum: output has more than one ellipsis");

        size_t p = 0;
        while (p < rhs.size()) {
            if (p + 2 < rhs.size() && rhs.compare(p, 3, "...") == 0) {
                for (int64_t k = 0; k < ell_width; ++k) {
                    int64_t idx_from_right = ell_width - 1 - k;
                    out.output.push_back(ellipsis_label(idx_from_right));
                }
                p += 3;
            } else {
                out.output.push_back((Label)(unsigned char)rhs[p]);
                p += 1;
            }
        }
    } else {
        // Implicit output: ellipsis axes first (left-to-right), then named
        // labels appearing exactly once across all inputs, in alphabetic order.
        for (int64_t k = 0; k < ell_width; ++k) {
            out.output.push_back(ellipsis_label(ell_width - 1 - k));
        }
        std::map<Label, int> count; // sorted by label value -> alphabetic
        for (const auto& labs : out.inputs) {
            std::unordered_set<Label> seen_in_op;
            for (Label L : labs) {
                if (L < 0) continue; // ellipsis handled above
                if (seen_in_op.insert(L).second) count[L]++;
            }
        }
        for (auto& kv : count) {
            if (kv.second == 1) out.output.push_back(kv.first);
        }
    }

    return out;
}

// Given a tensor `t` with labels `labs` (may contain repeats -> diagonal),
// produce a tensor whose label list has no duplicates. Repeated labels are
// collapsed by taking the diagonal over those axes.
inline Tensor collapse_diagonals(Tensor t, std::vector<Label>& labs) {
    while (true) {
        // Find first duplicate.
        int dup_first = -1, dup_second = -1;
        for (size_t i = 0; i < labs.size() && dup_first < 0; ++i) {
            for (size_t j = i + 1; j < labs.size(); ++j) {
                if (labs[i] == labs[j]) { dup_first = (int)i; dup_second = (int)j; break; }
            }
        }
        if (dup_first < 0) break;

        // Take diagonal over (dup_first, dup_second).
        // Implementation: permute so these two dims come last, reshape to
        // (rest..., N, N), then gather diagonal.
        int64_t rank = (int64_t)labs.size();
        PT_CHECK_MSG(t.size(dup_first) == t.size(dup_second),
            "einsum: repeated label requires equal axis sizes");

        std::vector<int64_t> perm;
        std::vector<Label> new_labs;
        for (int64_t d = 0; d < rank; ++d) {
            if (d != dup_first && d != dup_second) {
                perm.push_back(d);
                new_labs.push_back(labs[d]);
            }
        }
        perm.push_back(dup_first);
        perm.push_back(dup_second);

        Tensor p = permute(t, perm).contiguous();
        // shape now: [rest..., N, N]
        int64_t N = p.size(rank - 1);
        int64_t rest = 1;
        for (int64_t d = 0; d < rank - 2; ++d) rest *= p.size(d);

        // Extract diagonal: result shape [rest..., N]
        // Simple loop (correctness over speed; rarely used path).
        std::vector<int64_t> out_shape;
        for (int64_t d = 0; d < rank - 2; ++d) out_shape.push_back(p.size(d));
        out_shape.push_back(N);
        Tensor diag_t = at::empty(out_shape, TensorOptions().dtype(p.dtype()).device(p.device()));

        PT_DISPATCH_FLOATING_TYPES(p.dtype(), "einsum_diag", [&] {
            const scalar_t* src = p.data_ptr<scalar_t>();
            scalar_t* dst = diag_t.mutable_data_ptr<scalar_t>();
            for (int64_t r = 0; r < rest; ++r) {
                for (int64_t i = 0; i < N; ++i) {
                    dst[r * N + i] = src[r * N * N + i * N + i];
                }
            }
        });

        new_labs.push_back(labs[dup_first]);
        t = diag_t;
        labs = new_labs;
    }
    return t;
}

// Sum out labels in `to_sum` from `t` whose labels are `labs`. Returns the
// reduced tensor and updates `labs` in-place.
inline Tensor sum_out_labels(Tensor t, std::vector<Label>& labs,
                             const std::unordered_set<Label>& to_sum) {
    // Collect dims to sum (from HIGHEST index to lowest so removal is stable).
    std::vector<int64_t> dims;
    for (int64_t d = (int64_t)labs.size() - 1; d >= 0; --d) {
        if (to_sum.count(labs[d])) dims.push_back(d);
    }
    if (dims.empty()) return t;

    Tensor cur = t;
    std::vector<Label> new_labs = labs;
    for (int64_t d : dims) {
        cur = at::native::sum(cur, d, /*keepdim=*/false);
        new_labs.erase(new_labs.begin() + d);
    }
    labs = new_labs;
    return cur;
}

// Broadcast `t` with labels `labs` to also include `missing` labels (size
// known via `sizes`), by inserting unsqueeze(1) then expand. The new labels
// are appended at the end.
inline Tensor broadcast_add_labels(Tensor t, std::vector<Label>& labs,
                                   const std::vector<Label>& missing,
                                   const std::unordered_map<Label, int64_t>& sizes) {
    for (Label L : missing) {
        // Unsqueeze at end.
        t = unsqueeze(t, (int64_t)labs.size());
        labs.push_back(L);
    }
    // Build target shape.
    std::vector<int64_t> target;
    for (Label L : labs) target.push_back(sizes.at(L));
    t = expand(t, target).contiguous();
    return t;
}

// Pair-contract two tensors with their label lists. Returns (result, result_labels).
// `output_labels_set` tells us which labels must survive (free), the rest of
// shared labels can be summed.
inline std::pair<Tensor, std::vector<Label>> pair_contract(
        Tensor A, std::vector<Label> lA,
        Tensor B, std::vector<Label> lB,
        const std::unordered_set<Label>& keep_labels,
        const std::unordered_map<Label, int64_t>& sizes) {

    // Classify labels.
    std::unordered_set<Label> setA(lA.begin(), lA.end());
    std::unordered_set<Label> setB(lB.begin(), lB.end());

    std::vector<Label> batch_labels;   // in both, in keep
    std::vector<Label> sum_labels;     // in both, not in keep
    std::vector<Label> freeA;          // only in A
    std::vector<Label> freeB;          // only in B

    for (Label L : lA) {
        if (setB.count(L)) {
            if (keep_labels.count(L)) {
                if (std::find(batch_labels.begin(), batch_labels.end(), L) == batch_labels.end())
                    batch_labels.push_back(L);
            } else {
                if (std::find(sum_labels.begin(), sum_labels.end(), L) == sum_labels.end())
                    sum_labels.push_back(L);
            }
        } else {
            freeA.push_back(L);
        }
    }
    for (Label L : lB) {
        if (!setA.count(L)) freeB.push_back(L);
    }

    // Permute A to [batch..., freeA..., sum...].
    auto make_perm = [](const std::vector<Label>& src,
                        const std::vector<std::vector<Label>*>& order) {
        std::vector<int64_t> perm;
        for (auto* grp : order) {
            for (Label L : *grp) {
                auto it = std::find(src.begin(), src.end(), L);
                if (it != src.end()) perm.push_back((int64_t)(it - src.begin()));
            }
        }
        return perm;
    };

    std::vector<Label> freeA_m = freeA, freeB_m = freeB;
    std::vector<Label> batch_m = batch_labels, sum_m = sum_labels;

    auto permA = make_perm(lA, {&batch_m, &freeA_m, &sum_m});
    auto permB = make_perm(lB, {&batch_m, &sum_m, &freeB_m});

    Tensor Ap = permute(A, permA).contiguous();
    Tensor Bp = permute(B, permB).contiguous();

    int64_t bnum = 1;
    for (Label L : batch_m) bnum *= sizes.at(L);
    int64_t fa = 1;
    for (Label L : freeA_m) fa *= sizes.at(L);
    int64_t fb = 1;
    for (Label L : freeB_m) fb *= sizes.at(L);
    int64_t sn = 1;
    for (Label L : sum_m)  sn *= sizes.at(L);

    // Broadcast: if any batch label actually had size 1 in one operand but
    // >1 in the other, permute+contiguous already placed it at the correct
    // position. We still need to expand along batch axis so shapes match
    // before reshape-to-3D.
    // Build expected batched shape.
    std::vector<int64_t> Ap_target, Bp_target;
    for (Label L : batch_m) { Ap_target.push_back(sizes.at(L)); Bp_target.push_back(sizes.at(L)); }
    for (Label L : freeA_m) Ap_target.push_back(sizes.at(L));
    for (Label L : sum_m)   Ap_target.push_back(sizes.at(L));
    for (Label L : sum_m)   Bp_target.push_back(sizes.at(L));
    for (Label L : freeB_m) Bp_target.push_back(sizes.at(L));
    if (Ap.sizes().vec() != Ap_target) Ap = expand(Ap, Ap_target).contiguous();
    if (Bp.sizes().vec() != Bp_target) Bp = expand(Bp, Bp_target).contiguous();

    Tensor A3 = Ap.reshape({bnum, fa, sn});
    Tensor B3 = Bp.reshape({bnum, sn, fb});

    Tensor C3 = bmm(A3, B3); // (bnum, fa, fb)

    // Reshape to [batch_dims..., freeA_dims..., freeB_dims...].
    std::vector<int64_t> out_shape;
    std::vector<Label> out_labels;
    for (Label L : batch_m) { out_shape.push_back(sizes.at(L)); out_labels.push_back(L); }
    for (Label L : freeA_m) { out_shape.push_back(sizes.at(L)); out_labels.push_back(L); }
    for (Label L : freeB_m) { out_shape.push_back(sizes.at(L)); out_labels.push_back(L); }
    if (out_shape.empty()) out_shape.push_back(1); // scalar-ish guard

    Tensor C = C3.reshape(out_shape);
    return {C, out_labels};
}

} // namespace einsum_full

// ============================================================================
// Public entry point.
// ============================================================================

inline Tensor einsum_impl(const std::string& equation,
                          const std::vector<Tensor>& operands) {
    using namespace einsum_full;
    PT_CHECK_MSG(!operands.empty(), "einsum: requires at least one operand");

    // Gather shapes.
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(operands.size());
    for (const auto& t : operands) shapes.push_back(t.sizes().vec());

    Parsed p = parse(equation, shapes);

    // Step 1: for each operand, collapse repeated labels via diagonal.
    std::vector<Tensor> tensors = operands;
    std::vector<std::vector<Label>> tlabs = p.inputs;
    for (size_t i = 0; i < tensors.size(); ++i) {
        tensors[i] = collapse_diagonals(tensors[i], tlabs[i]);
        // Sizes table may need refresh after diagonal collapse (unchanged sizes).
        for (size_t d = 0; d < tlabs[i].size(); ++d) {
            p.sizes[tlabs[i][d]] = tensors[i].size(d);
        }
    }

    // Step 2: for each operand, sum out labels that appear ONLY in that
    // operand and NOT in the output. This shrinks tensors cheaply before
    // combining.
    std::unordered_set<Label> output_set(p.output.begin(), p.output.end());
    // Count label occurrences across all operands.
    std::unordered_map<Label, int> label_op_count;
    for (const auto& labs : tlabs) {
        std::unordered_set<Label> seen;
        for (Label L : labs) if (seen.insert(L).second) label_op_count[L]++;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        std::unordered_set<Label> to_sum;
        for (Label L : tlabs[i]) {
            if (!output_set.count(L) && label_op_count[L] == 1)
                to_sum.insert(L);
        }
        if (!to_sum.empty()) {
            tensors[i] = sum_out_labels(tensors[i], tlabs[i], to_sum);
            // Rebuild op counts (these labels disappeared entirely).
            for (Label L : to_sum) label_op_count.erase(L);
        }
    }

    // Step 3: iteratively pair-contract. Greedy: pick the pair with the most
    // shared labels (not strictly optimal but fine for <= ~8 operands).
    while (tensors.size() > 1) {
        size_t best_i = 0, best_j = 1;
        int best_shared = -1;
        for (size_t i = 0; i < tensors.size(); ++i) {
            std::unordered_set<Label> si(tlabs[i].begin(), tlabs[i].end());
            for (size_t j = i + 1; j < tensors.size(); ++j) {
                int shared = 0;
                std::unordered_set<Label> seen;
                for (Label L : tlabs[j])
                    if (si.count(L) && seen.insert(L).second) shared++;
                if (shared > best_shared) {
                    best_shared = shared; best_i = i; best_j = j;
                }
            }
        }

        // For the contraction, "keep" labels are: output labels, plus any
        // label that still appears in OTHER remaining operands (not just the
        // pair we're contracting). Otherwise we'd wrongly sum them now.
        std::unordered_set<Label> keep = output_set;
        for (size_t k = 0; k < tensors.size(); ++k) {
            if (k == best_i || k == best_j) continue;
            for (Label L : tlabs[k]) keep.insert(L);
        }

        auto [C, clabs] = pair_contract(
            tensors[best_i], tlabs[best_i],
            tensors[best_j], tlabs[best_j],
            keep, p.sizes);

        // Remove the two, append result.
        // Remove j first (higher index) to keep i valid.
        tensors.erase(tensors.begin() + best_j);
        tlabs.erase(tlabs.begin() + best_j);
        tensors.erase(tensors.begin() + best_i);
        tlabs.erase(tlabs.begin() + best_i);
        tensors.push_back(C);
        tlabs.push_back(clabs);

        // Refresh op counts.
        label_op_count.clear();
        for (const auto& labs : tlabs) {
            std::unordered_set<Label> seen;
            for (Label L : labs) if (seen.insert(L).second) label_op_count[L]++;
        }
    }

    // Step 4: final tensor — sum out any remaining non-output labels, then
    // permute to output order (broadcasting missing output labels if any).
    Tensor R = tensors[0];
    std::vector<Label> rlabs = tlabs[0];

    std::unordered_set<Label> to_sum;
    for (Label L : rlabs) if (!output_set.count(L)) to_sum.insert(L);
    if (!to_sum.empty()) R = sum_out_labels(R, rlabs, to_sum);

    // Broadcast any output labels not yet present (e.g. "i,j->ij" done in a
    // way that left some free label orphan).
    std::vector<Label> missing;
    std::unordered_set<Label> rset(rlabs.begin(), rlabs.end());
    for (Label L : p.output) if (!rset.count(L)) missing.push_back(L);
    if (!missing.empty()) R = broadcast_add_labels(R, rlabs, missing, p.sizes);

    // Permute to output order.
    if (rlabs != p.output) {
        std::vector<int64_t> perm;
        perm.reserve(p.output.size());
        for (Label L : p.output) {
            auto it = std::find(rlabs.begin(), rlabs.end(), L);
            PT_CHECK_MSG(it != rlabs.end(), "einsum: internal label mismatch");
            perm.push_back((int64_t)(it - rlabs.begin()));
        }
        R = permute(R, perm).contiguous();
    }

    // Scalar output: if output label list is empty, ensure rank-0 tensor.
    if (p.output.empty() && R.dim() > 0 && R.numel() == 1) {
        R = R.reshape({});
    }

    return R;
}

} // namespace native

// Top-level at::einsum (matches PyTorch API).
inline Tensor einsum(const std::string& equation,
                     const std::vector<Tensor>& operands) {
    return at::native::einsum_impl(equation, operands);
}

} // namespace at
