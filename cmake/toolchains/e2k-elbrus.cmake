# ============================================================================
# Toolchain: Elbrus E2K (MCST Elbrus 8C/16C/8SV, LCC compiler)
# ============================================================================
# Usage: cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/e2k-elbrus.cmake
# Target: Elbrus 8C (8 cores, 250 MHz FMA), Elbrus 16C, Elbrus 8SV
# OS: Elbrus OS (PDK), Astra Linux SE for Elbrus
# Compiler: LCC (MCST C/C++ compiler) — GCC-compatible frontend
# ============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR e2k)

# LCC compiler (default on Elbrus OS)
set(CMAKE_C_COMPILER lcc)
set(CMAKE_CXX_COMPILER lcc++)

# Elbrus 8C tuning — LCC auto-vectorizes with software pipelining
# Use -march=elbrus-8c for 8C, -march=elbrus-v6 for 8SV/16C
set(CMAKE_C_FLAGS_INIT "-march=elbrus-8c -O3")
set(CMAKE_CXX_FLAGS_INIT "-march=elbrus-8c -O3")

# LCC supports most GCC flags
# -fwhole — LCC whole-program optimization (optional)
# -qp — enable software pipelining (default at -O3)

# Disable x86/ARM features
set(PT_USE_AVX OFF CACHE BOOL "No AVX on E2K" FORCE)
set(PT_USE_AVX2 OFF CACHE BOOL "No AVX2 on E2K" FORCE)
set(PT_USE_CUDA OFF CACHE BOOL "No CUDA on E2K" FORCE)
set(PT_USE_TUDA ON CACHE BOOL "TUDA E2K dispatch" FORCE)
