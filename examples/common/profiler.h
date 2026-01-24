// ============================================================================
// ULTRA PROFILER - Detailed timing and memory tracking
// ============================================================================
#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

#ifdef PT_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace profiler {

// ============================================================================
// Timer
// ============================================================================

class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        return ms;
    }

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Scoped Timer with logging
// ============================================================================

class ScopedTimer {
public:
    ScopedTimer(const std::string& name, bool enabled = true)
        : name_(name), enabled_(enabled) {
        if (enabled_) {
            timer_.start();
            std::cout << "[TIMER] >>> " << name_ << " started" << std::endl;
        }
    }

    ~ScopedTimer() {
        if (enabled_) {
            double ms = timer_.stop();
            std::cout << "[TIMER] <<< " << name_ << " completed: "
                      << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
        }
    }

private:
    std::string name_;
    bool enabled_;
    Timer timer_;
};

// ============================================================================
// Memory Tracker
// ============================================================================

inline void log_memory(const std::string& label) {
#ifdef PT_USE_CUDA
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t used_bytes = total_bytes - free_bytes;

    std::cout << "[MEMORY] " << label
              << " | GPU Used: " << (used_bytes / 1024 / 1024) << " MB"
              << " | Free: " << (free_bytes / 1024 / 1024) << " MB"
              << " | Total: " << (total_bytes / 1024 / 1024) << " MB"
              << std::endl;
#else
    std::cout << "[MEMORY] " << label << " | CPU mode (no GPU tracking)" << std::endl;
#endif
}

// ============================================================================
// Ultra Logger
// ============================================================================

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }

    void trace(const std::string& msg) { log(LogLevel::TRACE, "TRACE", msg); }
    void debug(const std::string& msg) { log(LogLevel::DEBUG, "DEBUG", msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, "INFO", msg); }
    void warn(const std::string& msg) { log(LogLevel::WARN, "WARN", msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, "ERROR", msg); }

    // Tensor logging
    template<typename T>
    void log_tensor(const std::string& name, const T& tensor, LogLevel lvl = LogLevel::DEBUG) {
        if (lvl < level_) return;

        std::ostringstream oss;
        oss << "Tensor '" << name << "': shape=[";
        auto sizes = tensor.sizes();
        for (size_t i = 0; i < sizes.size(); ++i) {
            oss << sizes[i];
            if (i < sizes.size() - 1) oss << ", ";
        }
        oss << "], device=" << (tensor.is_cuda() ? "CUDA" : "CPU");
        oss << ", dtype=" << static_cast<int>(tensor.dtype());
        oss << ", numel=" << tensor.numel();

        log(lvl, "TENSOR", oss.str());
    }

    // Forward pass logging
    void log_forward(const std::string& module, const std::string& info = "") {
        if (LogLevel::TRACE < level_) return;
        std::ostringstream oss;
        oss << "Forward: " << module;
        if (!info.empty()) oss << " | " << info;
        log(LogLevel::TRACE, "FWD", oss.str());
    }

    // Backward pass logging
    void log_backward(const std::string& node, const std::string& info = "") {
        if (LogLevel::TRACE < level_) return;
        std::ostringstream oss;
        oss << "Backward: " << node;
        if (!info.empty()) oss << " | " << info;
        log(LogLevel::TRACE, "BWD", oss.str());
    }

    // Gradient logging
    void log_gradient(const std::string& param, float norm) {
        if (LogLevel::DEBUG < level_) return;
        std::ostringstream oss;
        oss << "Grad '" << param << "': norm=" << std::fixed << std::setprecision(6) << norm;
        log(LogLevel::DEBUG, "GRAD", oss.str());
    }

private:
    Logger() : level_(LogLevel::INFO) {}

    void log(LogLevel lvl, const std::string& tag, const std::string& msg) {
        if (lvl < level_) return;

        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        auto timer = std::chrono::system_clock::to_time_t(now);
        std::tm bt = *std::localtime(&timer);

        std::cout << std::put_time(&bt, "%H:%M:%S") << "."
                  << std::setfill('0') << std::setw(3) << ms.count()
                  << " [" << tag << "] " << msg << std::endl;
    }

    LogLevel level_;
};

// Convenience macros
#define LOG_TRACE(msg) profiler::Logger::instance().trace(msg)
#define LOG_DEBUG(msg) profiler::Logger::instance().debug(msg)
#define LOG_INFO(msg) profiler::Logger::instance().info(msg)
#define LOG_WARN(msg) profiler::Logger::instance().warn(msg)
#define LOG_ERROR(msg) profiler::Logger::instance().error(msg)

#define LOG_TENSOR(name, tensor) profiler::Logger::instance().log_tensor(name, tensor)
#define LOG_FORWARD(module) profiler::Logger::instance().log_forward(module)
#define LOG_BACKWARD(node) profiler::Logger::instance().log_backward(node)
#define LOG_GRAD(param, norm) profiler::Logger::instance().log_gradient(param, norm)

#define PROFILE_SCOPE(name) profiler::ScopedTimer _timer_##__LINE__(name)
#define LOG_MEMORY(label) profiler::log_memory(label)

// ============================================================================
// Statistics accumulator
// ============================================================================

class Stats {
public:
    void add(double value) {
        values_.push_back(value);
        sum_ += value;
        sum_sq_ += value * value;
        if (value < min_) min_ = value;
        if (value > max_) max_ = value;
    }

    size_t count() const { return values_.size(); }
    double sum() const { return sum_; }
    double mean() const { return values_.empty() ? 0 : sum_ / values_.size(); }
    double min() const { return min_; }
    double max() const { return max_; }
    double variance() const {
        if (values_.size() < 2) return 0;
        double m = mean();
        return (sum_sq_ / values_.size()) - m * m;
    }
    double std_dev() const { return std::sqrt(variance()); }

    void print(const std::string& name) const {
        std::cout << "[STATS] " << name
                  << " | count=" << count()
                  << " | mean=" << std::fixed << std::setprecision(3) << mean()
                  << " | std=" << std_dev()
                  << " | min=" << min()
                  << " | max=" << max()
                  << std::endl;
    }

    void reset() {
        values_.clear();
        sum_ = 0;
        sum_sq_ = 0;
        min_ = std::numeric_limits<double>::max();
        max_ = std::numeric_limits<double>::lowest();
    }

private:
    std::vector<double> values_;
    double sum_ = 0;
    double sum_sq_ = 0;
    double min_ = std::numeric_limits<double>::max();
    double max_ = std::numeric_limits<double>::lowest();
};

} // namespace profiler
