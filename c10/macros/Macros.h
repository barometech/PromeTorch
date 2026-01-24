#pragma once

#include <string>
#include <sstream>
#include <stdexcept>
#include <cassert>

// ============================================================================
// PromeTorch - Core Macros
// ============================================================================

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define PT_PLATFORM_WINDOWS 1
#elif defined(__linux__)
    #define PT_PLATFORM_LINUX 1
#elif defined(__APPLE__)
    #define PT_PLATFORM_MACOS 1
#else
    #define PT_PLATFORM_UNKNOWN 1
#endif

// Compiler detection
#if defined(_MSC_VER)
    #define PT_COMPILER_MSVC 1
    #define PT_MSVC_VERSION _MSC_VER
#elif defined(__clang__)
    #define PT_COMPILER_CLANG 1
#elif defined(__GNUC__)
    #define PT_COMPILER_GCC 1
#else
    #define PT_COMPILER_UNKNOWN 1
#endif

// Export/Import macros for shared libraries
#if defined(PT_PLATFORM_WINDOWS)
    #if defined(PT_BUILD_SHARED_LIBS)
        #define PT_API __declspec(dllexport)
    #else
        #define PT_API __declspec(dllimport)
    #endif
#else
    #define PT_API __attribute__((visibility("default")))
#endif

// Force inline
#if defined(PT_COMPILER_MSVC)
    #define PT_FORCE_INLINE __forceinline
#else
    #define PT_FORCE_INLINE __attribute__((always_inline)) inline
#endif

// No inline
#if defined(PT_COMPILER_MSVC)
    #define PT_NOINLINE __declspec(noinline)
#else
    #define PT_NOINLINE __attribute__((noinline))
#endif

// Likely/Unlikely branch hints
#if defined(__GNUC__) || defined(__clang__)
    #define PT_LIKELY(x) __builtin_expect(!!(x), 1)
    #define PT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define PT_LIKELY(x) (x)
    #define PT_UNLIKELY(x) (x)
#endif

// Unused variable/parameter
#define PT_UNUSED(x) (void)(x)

// Alignment
#if defined(PT_COMPILER_MSVC)
    #define PT_ALIGN(n) __declspec(align(n))
#else
    #define PT_ALIGN(n) __attribute__((aligned(n)))
#endif

// Restrict pointer
#if defined(PT_COMPILER_MSVC)
    #define PT_RESTRICT __restrict
#else
    #define PT_RESTRICT __restrict__
#endif

// Deprecation warnings
#if defined(PT_COMPILER_MSVC)
    #define PT_DEPRECATED(msg) __declspec(deprecated(msg))
#else
    #define PT_DEPRECATED(msg) __attribute__((deprecated(msg)))
#endif

// Nodiscard attribute (C++17)
#define PT_NODISCARD [[nodiscard]]

// Assert macros
#include <cassert>
#define PT_ASSERT(cond) assert(cond)
#define PT_ASSERT_MSG(cond, msg) assert((cond) && (msg))

// Debug-only assert
#ifdef NDEBUG
    #define PT_DASSERT(cond) ((void)0)
#else
    #define PT_DASSERT(cond) assert(cond)
#endif

// Check macro - always active, throws on failure
#define PT_CHECK(cond) \
    do { \
        if (PT_UNLIKELY(!(cond))) { \
            throw std::runtime_error( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                ": Check failed: " #cond \
            ); \
        } \
    } while (0)

// Helper to concatenate messages
namespace c10 {
namespace detail {
inline std::string concat_msg() { return ""; }

template<typename T>
inline std::string concat_msg(const T& t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

template<typename T, typename... Args>
inline std::string concat_msg(const T& first, Args&&... rest) {
    std::ostringstream oss;
    oss << first;
    return oss.str() + concat_msg(std::forward<Args>(rest)...);
}
} // namespace detail
} // namespace c10

#define PT_CHECK_MSG(cond, ...) \
    do { \
        if (PT_UNLIKELY(!(cond))) { \
            throw std::runtime_error( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                ": " + ::c10::detail::concat_msg(__VA_ARGS__) \
            ); \
        } \
    } while (0)

// Stringify macro
#define PT_STRINGIFY(x) #x
#define PT_TOSTRING(x) PT_STRINGIFY(x)

// Concatenate macros
#define PT_CONCAT_IMPL(a, b) a##b
#define PT_CONCAT(a, b) PT_CONCAT_IMPL(a, b)

// Unique variable name generator
#define PT_UNIQUE_NAME(prefix) PT_CONCAT(prefix, __LINE__)

// Disable copy
#define PT_DISABLE_COPY(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete

// Disable move
#define PT_DISABLE_MOVE(ClassName) \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete

// Disable copy and move
#define PT_DISABLE_COPY_AND_MOVE(ClassName) \
    PT_DISABLE_COPY(ClassName); \
    PT_DISABLE_MOVE(ClassName)

// Default move
#define PT_DEFAULT_MOVE(ClassName) \
    ClassName(ClassName&&) = default; \
    ClassName& operator=(ClassName&&) = default

// CUDA support macros
#ifdef __CUDACC__
    #define PT_CUDA_ENABLED 1
    #define PT_HOST __host__
    #define PT_DEVICE __device__
    #define PT_HOST_DEVICE __host__ __device__
#else
    #define PT_CUDA_ENABLED 0
    #define PT_HOST
    #define PT_DEVICE
    #define PT_HOST_DEVICE
#endif

// Array size helper
template<typename T, size_t N>
constexpr size_t pt_array_size(const T (&)[N]) noexcept {
    return N;
}

#define PT_ARRAY_SIZE(arr) pt_array_size(arr)

// Version info
#define PT_VERSION_MAJOR 0
#define PT_VERSION_MINOR 1
#define PT_VERSION_PATCH 0
#define PT_VERSION_STRING "0.1.0"

// OpenMP support - make pragma conditional
#ifdef _OPENMP
    #include <omp.h>
    #define PT_OMP_ENABLED 1
#else
    #define PT_OMP_ENABLED 0
#endif
