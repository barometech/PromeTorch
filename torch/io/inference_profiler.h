#pragma once

// ============================================================================
// Inference Profiler for PromeTorch
// ============================================================================
// Lightweight CUDA event-based profiler for measuring per-operation GPU timing,
// VRAM usage timeline, and kernel launch counting.
//
// Usage:
//   profiler.enable();
//   profiler.begin("rms_norm"); ... profiler.end("rms_norm");
//   profiler.print_report(std::cout);

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#ifdef PT_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace torch {
namespace io {

class InferenceProfiler {
public:
    InferenceProfiler() = default;

    ~InferenceProfiler() {
#ifdef PT_USE_CUDA
        if (events_created_) {
            cudaEventDestroy(ev_start_);
            cudaEventDestroy(ev_stop_);
        }
#endif
    }

    void enable() {
        enabled_ = true;
#ifdef PT_USE_CUDA
        if (!events_created_) {
            cudaEventCreate(&ev_start_);
            cudaEventCreate(&ev_stop_);
            events_created_ = true;
        }
#endif
        wall_start_ = std::chrono::high_resolution_clock::now();
    }

    void disable() { enabled_ = false; }
    bool enabled() const { return enabled_; }

    void begin(const char* name) {
        if (!enabled_) return;
#ifdef PT_USE_CUDA
        cudaEventRecord(ev_start_, 0);
#endif
        current_name_ = name;
    }

    void end(const char* name) {
        if (!enabled_) return;
#ifdef PT_USE_CUDA
        cudaEventRecord(ev_stop_, 0);
        cudaEventSynchronize(ev_stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start_, ev_stop_);
        accumulated_ms_[name] += ms;
        accumulated_count_[name] += 1;
#endif
    }

    void sample_vram() {
        if (!enabled_) return;
#ifdef PT_USE_CUDA
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t used = total_mem - free_mem;

        auto now = std::chrono::high_resolution_clock::now();
        double t_ms = std::chrono::duration<double, std::milli>(now - wall_start_).count();
        vram_timeline_.push_back({t_ms, used, total_mem});
#endif
    }

    void count_tokens(int n) { total_tokens_ += n; }

    void reset() {
        accumulated_ms_.clear();
        accumulated_count_.clear();
        vram_timeline_.clear();
        total_tokens_ = 0;
        wall_start_ = std::chrono::high_resolution_clock::now();
    }

    void print_report(std::ostream& out) const {
        if (accumulated_ms_.empty()) {
            out << "[Profile] No data collected." << std::endl;
            return;
        }

        // Sort by total time descending
        struct Entry {
            std::string name;
            float total_ms;
            int count;
        };
        std::vector<Entry> entries;
        float grand_total_ms = 0.0f;

        for (auto& [name, ms] : accumulated_ms_) {
            int count = 0;
            auto it = accumulated_count_.find(name);
            if (it != accumulated_count_.end()) count = it->second;
            entries.push_back({name, ms, count});
            grand_total_ms += ms;
        }

        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.total_ms > b.total_ms; });

        out << "\n=== PromeTorch Inference Profile ===" << std::endl;
        if (total_tokens_ > 0) {
            out << "Tokens generated: " << total_tokens_ << std::endl;
            if (grand_total_ms > 0) {
                out << "GPU time per token: " << std::fixed << std::setprecision(2)
                    << (grand_total_ms / total_tokens_) << " ms" << std::endl;
            }
        }
        out << std::endl;

        // Header
        out << std::left << std::setw(28) << "Operation"
            << std::right << std::setw(10) << "Total ms"
            << std::setw(8) << "Calls"
            << std::setw(10) << "Avg us"
            << std::setw(8) << "%" << std::endl;
        out << std::string(64, '-') << std::endl;

        for (auto& e : entries) {
            float avg_us = (e.count > 0) ? (e.total_ms * 1000.0f / e.count) : 0.0f;
            float pct = (grand_total_ms > 0) ? (e.total_ms / grand_total_ms * 100.0f) : 0.0f;
            out << std::left << std::setw(28) << e.name
                << std::right << std::fixed << std::setprecision(2)
                << std::setw(10) << e.total_ms
                << std::setw(8) << e.count
                << std::setw(10) << avg_us
                << std::setw(7) << pct << "%" << std::endl;
        }

        out << std::string(64, '-') << std::endl;
        out << std::left << std::setw(28) << "TOTAL"
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(10) << grand_total_ms << std::endl;

        // Total kernel launches
        int total_launches = 0;
        for (auto& e : entries) total_launches += e.count;
        out << "Total kernel launches: " << total_launches << std::endl;

        if (total_tokens_ > 0) {
            out << "Launches per token: " << (total_launches / total_tokens_) << std::endl;
        }
        out << std::endl;
    }

    void print_vram_timeline(std::ostream& out) const {
        if (vram_timeline_.empty()) {
            out << "[Profile] No VRAM data." << std::endl;
            return;
        }

        out << "\n=== VRAM Timeline ===" << std::endl;
        out << std::left << std::setw(12) << "Time(ms)"
            << std::setw(12) << "Used(MB)"
            << std::setw(12) << "Total(MB)"
            << "Bar" << std::endl;
        out << std::string(60, '-') << std::endl;

        for (auto& s : vram_timeline_) {
            double used_mb = s.used_bytes / (1024.0 * 1024.0);
            double total_mb = s.total_bytes / (1024.0 * 1024.0);
            int bar_len = static_cast<int>(40.0 * s.used_bytes / s.total_bytes);
            if (bar_len < 0) bar_len = 0;
            if (bar_len > 40) bar_len = 40;

            out << std::fixed << std::setprecision(1)
                << std::left << std::setw(12) << s.time_ms
                << std::setw(12) << used_mb
                << std::setw(12) << total_mb
                << "[" << std::string(bar_len, '#')
                << std::string(40 - bar_len, '.') << "]" << std::endl;
        }
        out << std::endl;
    }

private:
    bool enabled_ = false;

#ifdef PT_USE_CUDA
    cudaEvent_t ev_start_{}, ev_stop_{};
    bool events_created_ = false;
#endif

    std::string current_name_;
    std::unordered_map<std::string, float> accumulated_ms_;
    std::unordered_map<std::string, int> accumulated_count_;

    struct VRAMSample {
        double time_ms;
        size_t used_bytes;
        size_t total_bytes;
    };
    std::vector<VRAMSample> vram_timeline_;

    int total_tokens_ = 0;
    std::chrono::high_resolution_clock::time_point wall_start_;
};

// Zero-overhead macros when profiler is disabled
#define PROF_BEGIN(profiler, name) do { if ((profiler).enabled()) (profiler).begin(name); } while(0)
#define PROF_END(profiler, name) do { if ((profiler).enabled()) (profiler).end(name); } while(0)
#define PROF_VRAM(profiler) do { if ((profiler).enabled()) (profiler).sample_vram(); } while(0)

} // namespace io
} // namespace torch
