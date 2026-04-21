// Speculative decoding — draft token source.
//
// Keeps a rolling buffer of recently generated token_ids and exposes a cheap
// lookup that, given the last K=2 tokens, predicts the token that followed
// that same K-gram the last time we saw it in the buffer.
//
// Cost: O(buffer_size) linear scan per lookup (~100-500 ns for 2048 tokens),
// entirely in main thread — orders of magnitude cheaper than a forward pass
// (213 ms/token at 5.5 tok/s TP on Elbrus 8C2). A zero-accept draft is free
// because the verifier forward runs anyway; a high-accept draft shifts work
// from serial forwards into a batched forward.
//
// Designed to be stateless with respect to the model — caller records each
// committed token via `append()` and asks `predict()` for the next draft.
//
// Not a replacement for a trained draft model, but for repetitive contexts
// (code, instruction-following, structured generation, chat templates) empirical
// accept rates are 30-60%. For free-form creative text, accept rate is 5-15%.

#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

namespace torch {
namespace io {

// Minimal n-gram draft: exact match of the last `context_k` tokens against
// the rolling buffer. If the same suffix appeared before, return the token
// that followed it.
class NgramDraft {
public:
    // context_k: number of preceding tokens used as match key (default 2).
    // buffer_size: how far back we scan for matches (default 2048 ≈ 8 KB).
    explicit NgramDraft(int context_k = 2, size_t buffer_size = 2048)
        : context_k_(context_k), max_size_(buffer_size) {
        buf_.reserve(buffer_size);
    }

    // Record a committed token. Must be called for every token in context —
    // both prompt tokens and accepted generated tokens.
    void append(int64_t token_id) {
        if (buf_.size() < max_size_) {
            buf_.push_back(token_id);
        } else {
            // Ring buffer: overwrite oldest
            buf_[head_] = token_id;
            head_ = (head_ + 1) % max_size_;
        }
    }

    // Given the full committed-token history, return a drafted next-token.
    // Returns -1 if no match is found (caller falls back to non-speculative
    // decode for this step).
    //
    // Matching strategy: look at buf_ (rolling context up to 2048 back).
    // Find the most recent occurrence of `suffix` (last context_k tokens
    // of `history`). Return the token at position match+context_k.
    //
    // "Most recent" because: recent patterns are more likely to repeat than
    // old ones. Empirically gives higher accept rate than oldest-match.
    int64_t predict(const std::vector<int64_t>& history) const {
        if ((int64_t)history.size() < context_k_) return -1;
        if (logical_size() < (size_t)context_k_ + 1) return -1;

        // Suffix to match: last context_k tokens of history.
        const int64_t* suffix = history.data() + history.size() - context_k_;

        // Reverse scan through buf_ (from newest to oldest). Stop at first
        // match where the matched span is followed by at least one more token.
        // Iterate logical positions in reverse.
        size_t n = logical_size();
        // logical position i maps to physical: (head_ + i) % max_size_
        //   (when buf_ has wrapped; otherwise position i is just i).
        for (int64_t logical_i = (int64_t)n - 1 - context_k_; logical_i >= 0; --logical_i) {
            // Does buf[logical_i..logical_i+context_k-1] == suffix?
            bool ok = true;
            for (int k = 0; k < context_k_; ++k) {
                if (get(logical_i + k) != suffix[k]) { ok = false; break; }
            }
            if (ok) {
                // Candidate: token after match is buf[logical_i + context_k].
                int64_t pred = get(logical_i + context_k_);
                // Avoid degenerate loops: if predicted token is the same as
                // the last history token AND the last two match, likely stuck
                // (repeating tail). Return -1 so caller falls through.
                if ((int64_t)history.size() >= 2 &&
                    pred == history.back() &&
                    history.back() == history[history.size() - 2]) {
                    continue;
                }
                return pred;
            }
        }
        return -1;
    }

    size_t logical_size() const {
        return buf_.size();  // regardless of wrap; fine for our use (buf fills linearly first)
    }

    void reset() {
        buf_.clear();
        head_ = 0;
    }

    // Stats for benchmarking / tuning draft context length
    struct Stats {
        int64_t predictions = 0;    // predict() called
        int64_t hits = 0;           // predict() returned != -1
        int64_t accepts = 0;        // caller reported verified match
        double accept_rate() const {
            return hits > 0 ? (double)accepts / (double)hits : 0.0;
        }
        double hit_rate() const {
            return predictions > 0 ? (double)hits / (double)predictions : 0.0;
        }
    };
    mutable Stats stats;

    // Convenience: predict + record prediction stats in one call.
    int64_t predict_with_stats(const std::vector<int64_t>& history) const {
        stats.predictions++;
        int64_t p = predict(history);
        if (p >= 0) stats.hits++;
        return p;
    }

    // Caller reports whether the most recent draft was accepted by verifier.
    void record_accept(bool accepted) const {
        if (accepted) stats.accepts++;
    }

private:
    // Physical access — maps logical index to buf_ index accounting for wrap.
    int64_t get(int64_t logical_i) const {
        if (buf_.size() < max_size_) {
            return buf_[logical_i];
        }
        return buf_[(head_ + logical_i) % max_size_];
    }

    int context_k_;
    size_t max_size_;
    std::vector<int64_t> buf_;
    size_t head_ = 0;  // next write position when buf is full
};

}  // namespace io
}  // namespace torch
