// ============================================================================
// Phase 7 — speculative decode verify loop (MVP scaffold)
// ============================================================================
//
// MVP strategy: draft K-1 tokens via NgramDraft, run main model forward K
// times serially, accept greedy matches, rewind KV cache on first mismatch.
//
//   * K=1  → passthrough: no draft, no rewind, equivalent to plain decode.
//   * K=2+ → speculative: works correctly but gives NO SPEEDUP because each
//     forward pass reads the full 2.5 GB of weights. The speedup lives in
//     Phase 7.1 (batched inner kernels): one weight read + K compute passes.
//
// This file locks down the accept/reject protocol, KV rewind contract and
// NgramDraft bookkeeping so Phase 7.1 only has to swap the serial-forward
// loop for a batched one.
//
// Opt-in via env var PT_SPEC_K (1..6). Default 1 = classical decode.
// ============================================================================

#pragma once

#include "torch/io/speculative_draft.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace torch {
namespace io {

// Read PT_SPEC_K from env, clamp to [1, 6]. Cached per-process.
inline int spec_decode_k() {
    static const int k = [] {
        const char* e = std::getenv("PT_SPEC_K");
        if (!e || !e[0]) return 1;
        int v = std::atoi(e);
        if (v < 1) return 1;
        if (v > 6) return 6;
        return v;
    }();
    return k;
}

// Accumulators for reporting the actual acceptance rate at end of generate().
struct SpecStats {
    int64_t steps = 0;          // number of speculative verify attempts
    int64_t drafts_proposed = 0; // total draft tokens proposed (K-1 per step)
    int64_t drafts_accepted = 0; // total draft tokens accepted
    int64_t main_forwards = 0;   // total forward_decode_cpu invocations

    double acceptance_rate() const {
        if (drafts_proposed == 0) return 0.0;
        return double(drafts_accepted) / double(drafts_proposed);
    }
};

}  // namespace io
}  // namespace torch
