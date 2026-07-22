// ============================================================================
// diagnostics.h — Probes & diagnostics for PIR model debugging
// ============================================================================
// All tools for finding WHERE things break:
// 1. TensorStats — mean/std/min/max/nans/infs/zeros/saturated
// 2. ConsistencyDiagnostic — compare training_forward vs generate_forward layer-by-layer
// 3. GenerationProbe — per-token entropy, top-5, KL
// 4. ScanStateProbe — h[t] magnitudes inside parallel_scan
// 5. GateSaturationProbe — sigmoid(gate) histogram, scan_gate clamping rate
// ============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <string>
#include <vector>

namespace probes {

// ============================================================================
// 1. TensorStats — full health snapshot of a float buffer
// ============================================================================
struct TensorStats {
    double mean = 0;
    double std  = 0;
    float  min_v = 0;
    float  max_v = 0;
    int64_t n_nans = 0;
    int64_t n_infs = 0;
    int64_t n_zeros = 0;
    int64_t n_saturated_high = 0;  // |x| > 100
    int64_t n_total = 0;

    void compute(const float* x, int64_t n) {
        n_total = n;
        if (n == 0) return;
        double sum = 0, sumsq = 0;
        min_v = x[0]; max_v = x[0];
        for (int64_t i = 0; i < n; i++) {
            float v = x[i];
            if (std::isnan(v)) { n_nans++; continue; }
            if (std::isinf(v)) { n_infs++; continue; }
            if (v == 0.0f) n_zeros++;
            if (std::abs(v) > 100.0f) n_saturated_high++;
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            sum += v;
            sumsq += (double)v * v;
        }
        int64_t valid = n - n_nans - n_infs;
        if (valid > 0) {
            mean = sum / valid;
            std = std::sqrt(std::max(0.0, sumsq / valid - mean * mean));
        }
    }

    void print(const char* name) const {
        fprintf(stderr,
            "[%s] n=%lld mean=%+.4e std=%.4e min=%+.4e max=%+.4e "
            "nan=%lld inf=%lld zero=%lld sat>100=%lld\n",
            name, (long long)n_total, mean, std, min_v, max_v,
            (long long)n_nans, (long long)n_infs,
            (long long)n_zeros, (long long)n_saturated_high);
    }

    // Compact one-liner for layer-by-layer table
    void print_compact(const char* name) const {
        fprintf(stderr,
            "%-30s n=%9lld mean=%+.3e std=%.3e [%+.3e..%+.3e] nan=%lld inf=%lld\n",
            name, (long long)n_total, mean, std, min_v, max_v,
            (long long)n_nans, (long long)n_infs);
    }
};

// ============================================================================
// 2. Diff between two buffers — for ConsistencyDiagnostic
// ============================================================================
struct DiffStats {
    double max_abs_diff = 0;
    double max_rel_diff = 0;
    double mean_abs_diff = 0;
    int64_t n_diff = 0;        // count of elements where |a-b| > 1e-5
    int64_t n_total = 0;

    void compute(const float* a, const float* b, int64_t n) {
        n_total = n;
        double sum_abs = 0;
        for (int64_t i = 0; i < n; i++) {
            double d = std::abs((double)a[i] - (double)b[i]);
            sum_abs += d;
            if (d > 1e-5) n_diff++;
            if (d > max_abs_diff) max_abs_diff = d;
            double m = std::max(std::abs((double)a[i]), std::abs((double)b[i]));
            if (m > 1e-9) {
                double rd = d / m;
                if (rd > max_rel_diff) max_rel_diff = rd;
            }
        }
        mean_abs_diff = sum_abs / std::max((int64_t)1, n);
    }

    void print(const char* name) const {
        fprintf(stderr,
            "DIFF %-30s max_abs=%.3e max_rel=%.3e mean_abs=%.3e diverged=%lld/%lld (%.2f%%)\n",
            name, max_abs_diff, max_rel_diff, mean_abs_diff,
            (long long)n_diff, (long long)n_total,
            100.0 * n_diff / std::max((int64_t)1, n_total));
    }

    bool divergent() const { return max_abs_diff > 1e-3; }
};

// ============================================================================
// 3. GenerationProbe — analyze logit distribution per token
// ============================================================================
struct LogitProbe {
    float entropy = 0;
    float top1_prob = 0;
    int   top1_idx = 0;
    float top5_total = 0;
    int   top5_idx[5] = {0,0,0,0,0};
    float top5_prob[5] = {0,0,0,0,0};
    float kl_uniform = 0;  // KL(p || uniform). 0 = uniform (no info), high = peaked

    void compute(const float* logits, int V, float temperature = 1.0f) {
        // Softmax
        std::vector<float> p(V);
        float maxl = logits[0];
        for (int v = 1; v < V; v++) if (logits[v] > maxl) maxl = logits[v];
        float sum = 0;
        for (int v = 0; v < V; v++) {
            p[v] = std::exp((logits[v] - maxl) / temperature);
            sum += p[v];
        }
        for (int v = 0; v < V; v++) p[v] /= sum;

        // Entropy
        entropy = 0;
        for (int v = 0; v < V; v++) {
            if (p[v] > 1e-9) entropy -= p[v] * std::log(p[v]);
        }

        // KL from uniform: log(V) - H(p)
        kl_uniform = std::log((float)V) - entropy;

        // Top-5
        std::vector<int> idx(V);
        for (int v = 0; v < V; v++) idx[v] = v;
        std::partial_sort(idx.begin(), idx.begin() + std::min(5, V), idx.end(),
            [&](int a, int b) { return p[a] > p[b]; });

        top1_idx = idx[0];
        top1_prob = p[idx[0]];
        top5_total = 0;
        for (int i = 0; i < std::min(5, V); i++) {
            top5_idx[i] = idx[i];
            top5_prob[i] = p[idx[i]];
            top5_total += p[idx[i]];
        }
    }

    void print(int token_pos) const {
        fprintf(stderr, "  tok[%d] H=%.3f KL=%.3f top1=%d(p=%.3f) top5_sum=%.3f | ",
            token_pos, entropy, kl_uniform, top1_idx, top1_prob, top5_total);
        for (int i = 0; i < 5; i++) {
            char c = (top5_idx[i] >= 32 && top5_idx[i] < 127) ? (char)top5_idx[i] : '?';
            fprintf(stderr, "%d:%c(%.2f) ", top5_idx[i], c, top5_prob[i]);
        }
        fprintf(stderr, "\n");
    }
};

// ============================================================================
// 4. ScanStateProbe — magnitudes of h[t] inside parallel_scan
// ============================================================================
struct ScanStateProbe {
    std::vector<double> mag_per_t;  // ||h[t]||_2 / sqrt(D) at sampled t

    void sample_scan_state(const float* out_buf, int64_t T, int64_t D, int n_samples = 8) {
        mag_per_t.clear();
        for (int s = 0; s < n_samples; s++) {
            int64_t t = (T - 1) * s / std::max(1, n_samples - 1);
            const float* h = out_buf + t * D;
            double sum_sq = 0;
            for (int64_t d = 0; d < D; d++) sum_sq += (double)h[d] * h[d];
            mag_per_t.push_back(std::sqrt(sum_sq / D));
        }
    }

    void print(const char* name) const {
        fprintf(stderr, "[%s] |h[t]| at sampled t: ", name);
        for (double m : mag_per_t) fprintf(stderr, "%.3e ", m);
        fprintf(stderr, "\n");
    }
};

// ============================================================================
// 5. GateSaturationProbe — histogram of sigmoid output / scan gate clamping
// ============================================================================
struct GateSaturationProbe {
    int64_t bins[10] = {0,0,0,0,0,0,0,0,0,0};  // 10 bins [0..0.1, 0.1..0.2, ..., 0.9..1.0]
    int64_t n_clamped_low = 0;   // <= 0.5
    int64_t n_clamped_high = 0;  // >= 0.9999
    int64_t n_total = 0;

    void compute_sigmoid(const float* sig_out, int64_t n) {
        n_total = n;
        for (int64_t i = 0; i < n; i++) {
            float v = sig_out[i];
            if (v < 0) v = 0; if (v > 1) v = 1;
            int b = std::min(9, (int)(v * 10));
            bins[b]++;
        }
    }

    void compute_scan_gate(const float* gates, int64_t n) {
        n_total = n;
        for (int64_t i = 0; i < n; i++) {
            float v = gates[i];
            if (v <= 0.5001f) n_clamped_low++;
            if (v >= 0.9998f) n_clamped_high++;
            if (v < 0) v = 0; if (v > 1) v = 1;
            int b = std::min(9, (int)(v * 10));
            bins[b]++;
        }
    }

    void print(const char* name) const {
        fprintf(stderr, "[%s] hist[", name);
        for (int b = 0; b < 10; b++) {
            fprintf(stderr, "%lld%s",
                (long long)bins[b] * 100 / std::max((int64_t)1, n_total),
                b < 9 ? "," : "");
        }
        fprintf(stderr, "]%% clamp_lo=%lld clamp_hi=%lld n=%lld\n",
            (long long)n_clamped_low, (long long)n_clamped_high, (long long)n_total);
    }
};

// ============================================================================
// 6. Helpers for capturing intermediate activations
// ============================================================================
struct ActivationCapture {
    std::vector<float> data;
    std::string name;

    void capture(const std::string& n, const float* src, int64_t size) {
        name = n;
        data.assign(src, src + size);
    }
};

} // namespace probes
