#pragma once

// ============================================================================
// GradMode - Global toggle for autograd recording
// ============================================================================
// When GradMode is disabled, no autograd graph is created.
// This is used for inference to prevent memory accumulation.
//
// IMPORTANT: get_enabled_() is defined in grad_mode.cpp, NOT here.
// This prevents the DLL singleton bug where each DLL gets its own
// thread_local copy (BUG-C9).

namespace torch {
namespace autograd {

class GradMode {
public:
    static bool is_enabled() {
        return get_enabled_();
    }

    static void set_enabled(bool enabled) {
        get_enabled_() = enabled;
    }

private:
    // Defined in grad_mode.cpp — single definition across all DLLs
    static bool& get_enabled_();
};

// RAII guard for disabling grad mode
class NoGradGuard {
public:
    NoGradGuard() : prev_enabled_(GradMode::is_enabled()) {
        GradMode::set_enabled(false);
    }

    ~NoGradGuard() {
        GradMode::set_enabled(prev_enabled_);
    }

    // Non-copyable
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_enabled_;
};

// RAII guard for enabling grad mode
class EnableGradGuard {
public:
    EnableGradGuard() : prev_enabled_(GradMode::is_enabled()) {
        GradMode::set_enabled(true);
    }

    ~EnableGradGuard() {
        GradMode::set_enabled(prev_enabled_);
    }

    // Non-copyable
    EnableGradGuard(const EnableGradGuard&) = delete;
    EnableGradGuard& operator=(const EnableGradGuard&) = delete;

private:
    bool prev_enabled_;
};

} // namespace autograd
} // namespace torch
