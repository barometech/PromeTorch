// ============================================================================
// GradMode singleton storage — MUST be in a .cpp file, NOT a header
// ============================================================================
// BUG-C9 FIX: When GradMode::get_enabled_() was defined in a header with
// static thread_local, each DLL (.pyd, .exe, .dll) got its own copy.
// Python's no_grad() would flip the Python DLL's copy, but the C++ autograd
// engine (in a different DLL or statically linked) would still see grad=true.
//
// Solution: Single .cpp file compiled into torch_autograd library.
// All users link against the same symbol.

#include "torch/csrc/autograd/grad_mode.h"

namespace torch {
namespace autograd {

bool& GradMode::get_enabled_() {
    static thread_local bool enabled = true;
    return enabled;
}

} // namespace autograd
} // namespace torch
