#pragma once
// ============================================================================
// torch::jit::codegen_cpp — TorchInductor-like C++ codegen for fused kernels.
//
// Flow:
//   1. A FUSED_EWISE OpRecord (chain of MicroOps) is translated into
//      element-wise C++ source code with a single OpenMP parallel for loop.
//   2. The source is written to a temp file and passed to a compiler
//      (PROMETORCH_CC env var, or platform default: gcc/l++/cl.exe).
//   3. The resulting .so/.dll is loaded (dlopen/LoadLibrary) and the
//      kernel symbol resolved (dlsym/GetProcAddress).
//   4. The returned function pointer has signature:
//        void(const float** ins, int ni, float* out, int64_t n)
//      where `ins[0]` is the base value and ins[1..ni-1] are rhs buffers in
//      order of appearance inside the fused chain.
//
// Cache:
//   * Compiled kernels are cached per (chain signature + compiler) in an
//     in-process map so the same chain shape recompiles at most once.
//   * Temp files live under ${PROMETORCH_CACHE_DIR} (default: system temp).
//
// Portability:
//   * POSIX: dlopen/dlsym, "gcc -O3 -ffast-math -fopenmp -fPIC -shared"
//   * Elbrus (LCC): "l++ -O3 -ffast-math -fopenmp -fPIC -shared"
//   * Windows (MSVC): cl.exe /O2 /LD /openmp /EHsc
// ============================================================================

#include "torch/jit/compile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
  #include <direct.h>
#else
  #include <dlfcn.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace torch {
namespace jit {

// ============================================================================
// Function pointer type for a compiled fused kernel.
// ============================================================================
using FusedKernelFn = void (*)(const float** inputs, int num_inputs,
                               float* output, int64_t n);

namespace codegen_detail {

// ----------------------------------------------------------------------------
// Environment helpers
// ----------------------------------------------------------------------------
inline std::string env_or(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : std::string(fallback);
}

inline std::string cache_dir() {
    const char* v = std::getenv("PROMETORCH_CACHE_DIR");
    if (v && *v) return std::string(v);
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMP");
    if (!tmp) tmp = "C:\\Temp";
    return std::string(tmp) + "\\prometorch_jit";
#else
    const char* tmp = std::getenv("TMPDIR");
    if (!tmp) tmp = "/tmp";
    return std::string(tmp) + "/prometorch_jit";
#endif
}

inline void ensure_dir(const std::string& d) {
#ifdef _WIN32
    _mkdir(d.c_str());
#else
    ::mkdir(d.c_str(), 0755);
#endif
}

// Cheap FNV-1a 64-bit hash — used to derive a kernel signature.
inline uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

// ----------------------------------------------------------------------------
// Signature of a chain = sequence of (op, has_rhs, scalar_bucket). We do NOT
// hash raw scalar values — different scalars at same positions produce same
// kernel shape but different captured constants. We emit scalars into the
// source directly (they become compile-time constants for fastmath), so the
// signature also encodes hashed scalars. This is still cheaper than shipping
// scalars as params at runtime.
// ----------------------------------------------------------------------------
inline std::string chain_signature(const std::vector<MicroOp>& chain) {
    std::ostringstream ss;
    ss << "v1|";
    for (const auto& m : chain) {
        ss << int(m.op) << ':';
        if (is_binary_tensor_ewise(m.op)) ss << 'T';
        else if (is_scalar_ewise(m.op))   ss << 'S' << m.scalar;
        else                              ss << 'U';
        ss << '|';
    }
    return ss.str();
}

// ----------------------------------------------------------------------------
// Emit the right-hand-side C++ expression for a MicroOp applied to `v`.
// For binary-tensor ops, `rhs_expr` is the C++ expression for the rhs value.
// ----------------------------------------------------------------------------
inline std::string emit_micro_expr(const MicroOp& m, const std::string& v,
                                   const std::string& rhs_expr) {
    auto scalar = [&](float s) {
        char buf[64];
        // use hex float literal-ish: use standard decimal with enough digits.
        std::snprintf(buf, sizeof(buf), "%.9gf", s);
        return std::string(buf);
    };
    switch (m.op) {
        case Op::ADD_T: return "(" + v + " + " + rhs_expr + ")";
        case Op::SUB_T: return "(" + v + " - " + rhs_expr + ")";
        case Op::MUL_T: return "(" + v + " * " + rhs_expr + ")";
        case Op::DIV_T: return "(" + v + " / " + rhs_expr + ")";
        case Op::ADD_S: return "(" + v + " + " + scalar(m.scalar) + ")";
        case Op::SUB_S: return "(" + v + " - " + scalar(m.scalar) + ")";
        case Op::MUL_S: return "(" + v + " * " + scalar(m.scalar) + ")";
        case Op::DIV_S: return "(" + v + " * (1.0f / " + scalar(m.scalar) + "))";
        case Op::RELU:    return "((" + v + ") > 0.0f ? (" + v + ") : 0.0f)";
        case Op::SIGMOID: return "(1.0f / (1.0f + std::exp(-(" + v + "))))";
        case Op::TANH:    return "std::tanh(" + v + ")";
        case Op::EXP:     return "std::exp(" + v + ")";
        case Op::LOG:     return "std::log(" + v + ")";
        default:          return v;  // shouldn't happen
    }
}

} // namespace codegen_detail

// ============================================================================
// Public API: emit C++ source for a fused element-wise chain.
//
// The generated function has signature:
//   extern "C" void <kernel_name>(const float** ins, int ni,
//                                 float* out, int64_t n);
//
// Layout: ins[0] is the base input; ins[1..K] are rhs buffers for each
// binary-tensor MicroOp (in chain order).
// ============================================================================
inline std::string emit_cpp_fused_kernel(const std::vector<MicroOp>& ops,
                                        const std::string& kernel_name) {
    std::ostringstream src;
    src << "// Auto-generated by torch::jit codegen_cpp\n"
        << "#include <cmath>\n"
        << "#include <cstdint>\n"
        << "\n"
        << "extern \"C\"\n"
#ifdef _WIN32
        << "__declspec(dllexport)\n"
#endif
        << "void " << kernel_name
        << "(const float** __restrict ins, int ni,\n"
        << "    float* __restrict out, int64_t n) {\n"
        << "    (void)ni;\n";

    // Bind rhs pointers for each binary-tensor micro-op.
    int rhs_idx = 1;
    for (size_t i = 0; i < ops.size(); ++i) {
        if (is_binary_tensor_ewise(ops[i].op)) {
            src << "    const float* __restrict r" << i
                << " = ins[" << rhs_idx << "];\n";
            ++rhs_idx;
        }
    }

    src << "    const float* __restrict x = ins[0];\n"
        << "#if defined(_OPENMP)\n"
        << "    #pragma omp parallel for schedule(static) if(n >= 4096)\n"
        << "#endif\n"
        << "    for (int64_t i = 0; i < n; ++i) {\n"
        << "        float v = x[i];\n";

    for (size_t i = 0; i < ops.size(); ++i) {
        const auto& m = ops[i];
        std::string rhs_expr;
        if (is_binary_tensor_ewise(m.op)) {
            rhs_expr = "r" + std::to_string(i) + "[i]";
        }
        src << "        v = "
            << codegen_detail::emit_micro_expr(m, "v", rhs_expr)
            << ";\n";
    }

    src << "        out[i] = v;\n"
        << "    }\n"
        << "}\n";
    return src.str();
}

namespace codegen_detail {

// ----------------------------------------------------------------------------
// Pick compiler command line based on env / platform.
// Returns: (command template, output extension). The template has two "{SRC}"
// and "{OUT}" placeholders replaced by the caller.
// ----------------------------------------------------------------------------
inline void pick_compiler(std::string& cmd_tmpl, std::string& out_ext) {
    const char* user_cc = std::getenv("PROMETORCH_CC");
    if (user_cc && *user_cc) {
        // Assume user provides a template with {SRC} and {OUT} placeholders.
        // If they don't, append a sane default.
        std::string s = user_cc;
        if (s.find("{SRC}") == std::string::npos) {
            s += " -O3 -ffast-math -fopenmp -fPIC -shared -o {OUT} {SRC}";
        }
        cmd_tmpl = s;
#ifdef _WIN32
        out_ext = ".dll";
#else
        out_ext = ".so";
#endif
        return;
    }

#ifdef _WIN32
    // MSVC cl.exe. /LD produces DLL.
    cmd_tmpl = "cl.exe /nologo /O2 /EHsc /LD /openmp /Fe:{OUT} {SRC}";
    out_ext  = ".dll";
#else
    #ifdef __e2k__
        // Elbrus LCC
        cmd_tmpl = "l++ -O3 -ffast-math -fopenmp -fPIC -shared -o {OUT} {SRC}";
    #else
        cmd_tmpl = "gcc -O3 -ffast-math -fopenmp -fPIC -shared -x c++ "
                   "-o {OUT} {SRC}";
    #endif
    out_ext = ".so";
#endif
}

inline std::string replace_all(std::string s, const std::string& from,
                               const std::string& to) {
    size_t p = 0;
    while ((p = s.find(from, p)) != std::string::npos) {
        s.replace(p, from.size(), to);
        p += to.size();
    }
    return s;
}

// ----------------------------------------------------------------------------
// Loaded-library handle + its resolved kernel pointer.
// ----------------------------------------------------------------------------
struct LoadedKernel {
#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
    FusedKernelFn fn = nullptr;
    std::string so_path;
};

// Global kernel cache keyed by (signature + compiler template).
inline std::unordered_map<std::string, LoadedKernel>& kernel_cache() {
    static std::unordered_map<std::string, LoadedKernel> m;
    return m;
}

inline std::mutex& kernel_cache_mutex() {
    static std::mutex m;
    return m;
}

} // namespace codegen_detail

// ============================================================================
// Public API: compile C++ source and return a function pointer to
// `kernel_name`, or nullptr on failure. Caches by signature.
// ============================================================================
inline FusedKernelFn compile_and_load(const std::string& cpp_src,
                                     const std::string& kernel_name) {
    using namespace codegen_detail;

    std::string cmd_tmpl, ext;
    pick_compiler(cmd_tmpl, ext);

    std::string cache_key = kernel_name + "|" + cmd_tmpl;
    {
        std::lock_guard<std::mutex> lk(kernel_cache_mutex());
        auto it = kernel_cache().find(cache_key);
        if (it != kernel_cache().end() && it->second.fn) return it->second.fn;
    }

    std::string dir = cache_dir();
    ensure_dir(dir);

    // Write source.
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    std::string src_path = dir + sep + kernel_name + ".cc";
    std::string out_path = dir + sep + kernel_name + ext;

    {
        std::ofstream f(src_path, std::ios::binary | std::ios::trunc);
        if (!f) return nullptr;
        f.write(cpp_src.data(), cpp_src.size());
        f.close();
    }

    // Build compile command.
    std::string cmd = replace_all(cmd_tmpl, "{SRC}", src_path);
    cmd = replace_all(cmd, "{OUT}", out_path);

    // Redirect diagnostics to a log file so we can inspect on failure.
    std::string log_path = out_path + ".log";
#ifdef _WIN32
    std::string full = cmd + " > \"" + log_path + "\" 2>&1";
#else
    std::string full = cmd + " > '" + log_path + "' 2>&1";
#endif
    int rc = std::system(full.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "[jit.codegen] compile failed (rc=%d): %s\n",
                     rc, full.c_str());
        std::fprintf(stderr, "[jit.codegen] see log: %s\n", log_path.c_str());
        return nullptr;
    }

    // Load the shared object.
    LoadedKernel lk;
    lk.so_path = out_path;

#ifdef _WIN32
    lk.handle = LoadLibraryA(out_path.c_str());
    if (!lk.handle) {
        std::fprintf(stderr, "[jit.codegen] LoadLibrary failed: %s\n",
                     out_path.c_str());
        return nullptr;
    }
    FARPROC p = GetProcAddress(lk.handle, kernel_name.c_str());
    lk.fn = reinterpret_cast<FusedKernelFn>(p);
#else
    lk.handle = ::dlopen(out_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lk.handle) {
        std::fprintf(stderr, "[jit.codegen] dlopen failed: %s\n", ::dlerror());
        return nullptr;
    }
    lk.fn = reinterpret_cast<FusedKernelFn>(
        ::dlsym(lk.handle, kernel_name.c_str()));
#endif
    if (!lk.fn) {
        std::fprintf(stderr, "[jit.codegen] symbol not found: %s\n",
                     kernel_name.c_str());
#ifdef _WIN32
        if (lk.handle) FreeLibrary(lk.handle);
#else
        if (lk.handle) ::dlclose(lk.handle);
#endif
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lk2(kernel_cache_mutex());
        kernel_cache()[cache_key] = lk;
    }
    return lk.fn;
}

// ============================================================================
// High-level helper: given a FUSED_EWISE chain, produce / retrieve its
// compiled kernel. Thread-safe via the kernel_cache mutex.
//
// Returns nullptr on compile failure (caller should fall back to interpreter).
// ============================================================================
inline FusedKernelFn codegen_fused(const std::vector<MicroOp>& chain) {
    using namespace codegen_detail;

    std::string sig = chain_signature(chain);
    uint64_t h = fnv1a(sig);
    char namebuf[64];
    std::snprintf(namebuf, sizeof(namebuf), "pt_fused_%016llx",
                  static_cast<unsigned long long>(h));
    std::string kernel_name = namebuf;

    // Fast path: already compiled?
    std::string cmd_tmpl, ext;
    pick_compiler(cmd_tmpl, ext);
    std::string cache_key = kernel_name + "|" + cmd_tmpl;
    {
        std::lock_guard<std::mutex> lk(kernel_cache_mutex());
        auto it = kernel_cache().find(cache_key);
        if (it != kernel_cache().end() && it->second.fn) return it->second.fn;
    }

    std::string src = emit_cpp_fused_kernel(chain, kernel_name);
    return compile_and_load(src, kernel_name);
}

// ============================================================================
// Convenience: run a chain via a compiled kernel. Builds the `ins` pointer
// array on the stack (small).
// ============================================================================
inline void run_fused_codegen(FusedKernelFn fn,
                              const float* base,
                              const std::vector<const float*>& rhs_ptrs,
                              float* out, int64_t n) {
    std::vector<const float*> ins;
    ins.reserve(rhs_ptrs.size() + 1);
    ins.push_back(base);
    for (auto* p : rhs_ptrs) ins.push_back(p);
    fn(ins.data(), static_cast<int>(ins.size()), out, n);
}

// ============================================================================
// Auto-registration: on first inclusion of this header in any TU, install
// codegen_fused() as the hook used by CompiledFn replay.
//
// The hook is a plain function pointer (not std::function) so installing it
// from multiple TUs is idempotent.
// ============================================================================
namespace codegen_detail {

inline CodegenFusedKernelFn codegen_hook_adapter(
        const std::vector<MicroOp>& chain) {
    return codegen_fused(chain);
}

struct CodegenHookRegistrar {
    CodegenHookRegistrar() {
        torch::jit::codegen_hook() = &codegen_hook_adapter;
    }
};

// inline static has exactly-one-initialization semantics across TUs (C++17).
inline const CodegenHookRegistrar registrar_instance{};

} // namespace codegen_detail

} // namespace jit
} // namespace torch
