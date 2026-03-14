# ============================================================================
# Toolchain: Astra Linux SE (x86_64, Debian-based)
# ============================================================================
# Usage: cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/x86_64-astra.cmake
# Target: Astra Linux Special Edition 1.7+ (ГОСТ Р 56939, ФСТЭК certified)
# Based on: Debian 10 (Buster) / Debian 11 (Bullseye)
# Package manager: apt
# ============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Standard GCC (Astra ships GCC 8.3+ / 10+)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# x86_64 with AVX2+FMA (Intel/AMD server CPUs)
set(CMAKE_C_FLAGS_INIT "-march=x86-64-v3 -mavx2 -mfma")
set(CMAKE_CXX_FLAGS_INIT "-march=x86-64-v3 -mavx2 -mfma")

# PromeTorch options
set(PT_USE_AVX ON CACHE BOOL "AVX on x86_64" FORCE)
set(PT_USE_AVX2 ON CACHE BOOL "AVX2 on x86_64" FORCE)
set(PT_USE_TUDA ON CACHE BOOL "TUDA AVX2 dispatch" FORCE)
