# ============================================================================
# Toolchain: Baikal-M (ARM Cortex-A57, NEON)
# ============================================================================
# Usage: cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-baikal-m.cmake
# Target: Baikal-M BE-M1000 SoC (8x Cortex-A57, Mali-T628 GPU)
# OS: Astra Linux SE 1.7+, ALT Linux
# ============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler (install: apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Cortex-A57 tuning (no dotprod, no fp16 — ARMv8.0-A)
set(CMAKE_C_FLAGS_INIT "-march=armv8-a+simd -mtune=cortex-a57")
set(CMAKE_CXX_FLAGS_INIT "-march=armv8-a+simd -mtune=cortex-a57")

# Sysroot (optional, set if cross-compiling from x86)
# set(CMAKE_SYSROOT /opt/baikal-m-sysroot)

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Disable x86-only features
set(PT_USE_AVX OFF CACHE BOOL "No AVX on ARM" FORCE)
set(PT_USE_AVX2 OFF CACHE BOOL "No AVX2 on ARM" FORCE)
set(PT_USE_CUDA OFF CACHE BOOL "No CUDA on Baikal-M" FORCE)
set(PT_USE_TUDA ON CACHE BOOL "TUDA NEON dispatch" FORCE)
