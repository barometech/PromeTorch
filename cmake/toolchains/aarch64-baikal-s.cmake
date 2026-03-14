# ============================================================================
# Toolchain: Baikal-S (ARM Cortex-A75, NEON + dotprod)
# ============================================================================
# Usage: cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-baikal-s.cmake
# Target: Baikal-S BE-S1000 SoC (48x Cortex-A75, Mali-G57 GPU)
# OS: Astra Linux SE 1.8+, ALT Linux
# ============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Cortex-A75 tuning (ARMv8.2-A with fp16 + dotprod)
set(CMAKE_C_FLAGS_INIT "-march=armv8.2-a+fp16+dotprod -mtune=cortex-a75")
set(CMAKE_CXX_FLAGS_INIT "-march=armv8.2-a+fp16+dotprod -mtune=cortex-a75")

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Disable x86-only features
set(PT_USE_AVX OFF CACHE BOOL "No AVX on ARM" FORCE)
set(PT_USE_AVX2 OFF CACHE BOOL "No AVX2 on ARM" FORCE)
set(PT_USE_CUDA OFF CACHE BOOL "No CUDA on Baikal-S" FORCE)
set(PT_USE_TUDA ON CACHE BOOL "TUDA NEON dispatch" FORCE)
