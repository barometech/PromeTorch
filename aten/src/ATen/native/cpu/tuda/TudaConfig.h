#pragma once
// ============================================================================
// TudaConfig.h — TUDA: Compile-time CPU architecture detection & tuning
// ============================================================================
// TUDA (PromeTorch Unified Device Architecture) provides architecture-specific
// optimized kernels for Russian processors (Baikal, Elbrus) alongside x86 AVX2.
//
// Supported architectures:
//   AVX2     — Intel/AMD x86-64 with AVX2+FMA (default on x86)
//   NEON_A57 — ARM Cortex-A57 (Baikal-M, BE-M1000)
//   NEON_A75 — ARM Cortex-A75 (Baikal-S, BE-S1000)
//   E2K      — MCST Elbrus VLIW (8C, 8SV, 16C)
//   NMC4     — RC Module NeuroMatrix NMC4 (K1879VM8YA, NM Card Mini)
//   SCALAR   — Generic fallback for any platform
// ============================================================================

#include <cstdint>

namespace at {
namespace native {
namespace tuda {

// ============================================================================
// Architecture enum
// ============================================================================

enum class Arch {
    AVX2,
    NEON_A57,
    NEON_A75,
    E2K_V5,
    E2K_V6,
    NMC4,       // RC Module NeuroMatrix NMC4 (K1879VM8YA)
    SCALAR
};

// ============================================================================
// Compile-time architecture selection
// ============================================================================

#if defined(__AVX2__) && defined(__FMA__)
    constexpr Arch kArch = Arch::AVX2;
    #define TUDA_AVX2 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_FEATURE_DOTPROD)
        constexpr Arch kArch = Arch::NEON_A75;
        #define TUDA_NEON_A75 1
    #else
        constexpr Arch kArch = Arch::NEON_A57;
        #define TUDA_NEON_A57 1
    #endif
    #define TUDA_NEON 1
#elif defined(__e2k__) || defined(__elbrus__)
    #if defined(__iset__) && (__iset__ >= 6)
        constexpr Arch kArch = Arch::E2K_V6;
    #else
        constexpr Arch kArch = Arch::E2K_V5;
    #endif
    #define TUDA_E2K 1
#elif defined(__nmc__) || defined(__nmc4__) || defined(TUDA_FORCE_NMC4)
    constexpr Arch kArch = Arch::NMC4;
    #define TUDA_NMC4 1
#else
    constexpr Arch kArch = Arch::SCALAR;
    #define TUDA_SCALAR 1
#endif

// ============================================================================
// GEMM tuning constants per architecture
// ============================================================================

struct GemmTuning {
    int64_t MR, NR;
    int64_t MC, KC, NC;
    int64_t ALIGN;
};

// AVX2: 16 YMM regs → 6×2=12 accumulators + 2 B loads + 1 A broadcast + 1 spare
// MC*KC=72*256=18432 floats=72KB → fits L2
constexpr GemmTuning kAVX2 = {6, 16, 72, 256, 4096, 64};

// NEON A57 (Baikal-M): 32 V regs, 32KB L1d
// 4×2=8 accumulators + 2 B + 1 A = 11 regs
// MC*KC=48*128=6144 floats=24KB → fits L1d
constexpr GemmTuning kNEON_A57 = {4, 8, 48, 128, 2048, 16};

// NEON A75 (Baikal-S): 32 V regs, 64KB L1d, wider pipeline
// 8×3=24 accumulators + 3 B + 1 A = 28 regs
// MC*KC=64*256=16384 floats=64KB → fits L1d
constexpr GemmTuning kNEON_A75 = {8, 12, 64, 256, 2048, 16};

// E2K Elbrus: 256 regs, 4 FMA units, 64KB L1d
// 4×4=16 scalar FMA per K-step → fills 4 FMA units in 4 cycles
constexpr GemmTuning kE2K = {4, 4, 64, 256, 2048, 16};

// NMC4 NeuroMatrix: 4 FPU cores, 512KB NMMB local memory
// On-card GEMM via nmpp (nmppmMul_mm_32f), TUDA tiles for host-side fallback
// MR=4, NR=4: 16 scalar FMA accumulators → maps to 4 FPU cores
constexpr GemmTuning kNMC4 = {4, 4, 64, 128, 2048, 16};

// Scalar fallback
constexpr GemmTuning kScalar = {4, 4, 64, 64, 512, 8};

// Select tuning for current arch
constexpr GemmTuning kTuning =
#if defined(TUDA_AVX2)
    kAVX2;
#elif defined(TUDA_NEON_A75)
    kNEON_A75;
#elif defined(TUDA_NEON)
    kNEON_A57;
#elif defined(TUDA_E2K)
    kE2K;
#elif defined(TUDA_NMC4)
    kNMC4;
#else
    kScalar;
#endif

// Vector width in floats
constexpr int64_t kVecWidth =
#if defined(TUDA_AVX2)
    8;
#elif defined(TUDA_NEON)
    4;
#elif defined(TUDA_E2K)
    4;
#elif defined(TUDA_NMC4)
    4;  // 4 FPU cores process in parallel
#else
    1;
#endif

// Architecture name string
constexpr const char* kArchName =
#if defined(TUDA_AVX2)
    "AVX2";
#elif defined(TUDA_NEON_A75)
    "NEON-A75 (Baikal-S)";
#elif defined(TUDA_NEON)
    "NEON-A57 (Baikal-M)";
#elif defined(TUDA_E2K)
    "E2K (Elbrus)";
#elif defined(TUDA_NMC4)
    "NMC4 (NeuroMatrix)";
#else
    "Scalar";
#endif

} // namespace tuda
} // namespace native
} // namespace at
