// Correctness + perf microbench for torch::io::NgramDraft.

#include <cstdio>
#include <vector>
#include <cstdint>
#include <chrono>
#include <random>
#include <cassert>

#include "torch/io/speculative_draft.h"

int main() {
    using torch::io::NgramDraft;

    // Test 1 — empty history → no prediction.
    {
        NgramDraft d(2, 128);
        std::vector<int64_t> hist;
        assert(d.predict(hist) == -1);
    }

    // Test 2 — exact 2-gram match.
    {
        NgramDraft d(2, 128);
        for (int64_t t : {1, 2, 3, 4, 5, 1, 2}) d.append(t);
        // History ends with "1, 2" — seen before, followed by 3.
        std::vector<int64_t> hist = {1, 2, 3, 4, 5, 1, 2};
        int64_t p = d.predict(hist);
        std::printf("test2: pred=%ld (expected 3)\n", (long)p);
        assert(p == 3);
    }

    // Test 3 — prefers most recent match (newer prediction).
    {
        NgramDraft d(2, 128);
        for (int64_t t : {10, 20, 30, 10, 20, 40}) d.append(t);
        std::vector<int64_t> hist = {10, 20, 30, 10, 20, 40, 10, 20};
        // "10, 20" appears at position 0 (→30) and position 3 (→40).
        // Most recent match → predict 40.
        int64_t p = d.predict(hist);
        std::printf("test3: pred=%ld (expected 40, newest match)\n", (long)p);
        assert(p == 40);
    }

    // Test 4 — degenerate repeat guard.
    {
        NgramDraft d(2, 128);
        for (int64_t t : {5, 5, 5, 5}) d.append(t);
        std::vector<int64_t> hist = {5, 5, 5, 5};
        // last two are "5, 5", found at offset 0 (→5). But history ends in
        // "5, 5" where both are identical, so the degenerate-loop guard rejects.
        int64_t p = d.predict(hist);
        std::printf("test4: pred=%ld (expected -1, degenerate)\n", (long)p);
        assert(p == -1);
    }

    // Test 5 — no match.
    {
        NgramDraft d(2, 128);
        for (int64_t t : {100, 200, 300}) d.append(t);
        std::vector<int64_t> hist = {100, 200, 300, 999, 888};
        int64_t p = d.predict(hist);
        std::printf("test5: pred=%ld (expected -1, no match)\n", (long)p);
        assert(p == -1);
    }

    // Perf: 2048-token buffer, 1000 predictions. Time per predict.
    {
        NgramDraft d(2, 2048);
        std::mt19937 rng(42);
        std::vector<int64_t> hist;
        hist.reserve(2048);
        for (int i = 0; i < 2048; ++i) {
            int64_t t = rng() % 10000;
            d.append(t);
            hist.push_back(t);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        int64_t sum = 0;
        int iters = 10000;
        for (int i = 0; i < iters; ++i) {
            int64_t p = d.predict(hist);
            sum += (p >= 0 ? 1 : 0);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::printf("perf: %d predicts in %.0f us = %.2f us/predict, hits=%ld\n",
                    iters, us, us / iters, (long)sum);
    }

    std::printf("ALL TESTS PASSED\n");
    return 0;
}
