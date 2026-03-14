# ============================================================================
# Toolchain: ALT Linux (x86_64, apt-rpm based)
# ============================================================================
# Usage: cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/x86_64-alt.cmake
# Target: ALT Linux SP 10+ (ФСТЭК certified)
# Based on: ALT Sisyphus (independent distro, apt-rpm)
# Package manager: apt-get (RPM backend)
# ============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Standard GCC (ALT ships GCC 10+)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# x86_64 with AVX2+FMA
set(CMAKE_C_FLAGS_INIT "-march=x86-64-v3 -mavx2 -mfma")
set(CMAKE_CXX_FLAGS_INIT "-march=x86-64-v3 -mavx2 -mfma")

# PromeTorch options
set(PT_USE_AVX ON CACHE BOOL "AVX on x86_64" FORCE)
set(PT_USE_AVX2 ON CACHE BOOL "AVX2 on x86_64" FORCE)
set(PT_USE_TUDA ON CACHE BOOL "TUDA AVX2 dispatch" FORCE)
