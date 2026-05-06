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

# Compiler auto-detect — script (build-elbrus.sh) обычно прокидывает
# CMAKE_C_COMPILER / CMAKE_CXX_COMPILER через env. Если не передано,
# пробуем варианты по порядку:
#   - lcc / lcc++ (Альт Линукс под Эльбрус, pkg lcc-c++)
#   - lcc / l++   (MCST PDK, /opt/mcst/bin/l++)
#   - gcc-elbrus / g++-elbrus
#   - gcc / g++ (если e2k-aware)
if(NOT CMAKE_C_COMPILER)
    find_program(_LCC_BIN lcc)
    if(_LCC_BIN)
        set(CMAKE_C_COMPILER ${_LCC_BIN})
    else()
        find_program(_GCC_BIN gcc-elbrus gcc)
        set(CMAKE_C_COMPILER ${_GCC_BIN})
    endif()
endif()
if(NOT CMAKE_CXX_COMPILER)
    find_program(_LCCXX_BIN lcc++)
    if(NOT _LCCXX_BIN)
        find_program(_LCCXX_BIN l++)
    endif()
    if(_LCCXX_BIN)
        set(CMAKE_CXX_COMPILER ${_LCCXX_BIN})
    else()
        find_program(_GXX_BIN g++-elbrus g++)
        set(CMAKE_CXX_COMPILER ${_GXX_BIN})
    endif()
endif()

# Elbrus tuning — универсальный target по ISA-версии:
#   -march=elbrus-v4 — все 8C/8C2/8СВ (covers everything since 2019)
#   -march=elbrus-v5 — 8СВ
#   -march=elbrus-v6 — 16С
# Старые конкретные имена (-march=elbrus-8c) новые версии LCC и gcc-elbrus
# не понимают: используй -march=elbrus-v4 как safe baseline.
set(CMAKE_C_FLAGS_INIT "-march=elbrus-v4 -O3")
set(CMAKE_CXX_FLAGS_INIT "-march=elbrus-v4 -O3")

# LCC supports most GCC flags
# -fwhole — LCC whole-program optimization (optional)
# -qp — enable software pipelining (default at -O3)

# Disable x86/ARM features
set(PT_USE_AVX OFF CACHE BOOL "No AVX on E2K" FORCE)
set(PT_USE_AVX2 OFF CACHE BOOL "No AVX2 on E2K" FORCE)
set(PT_USE_CUDA OFF CACHE BOOL "No CUDA on E2K" FORCE)
set(PT_USE_TUDA ON CACHE BOOL "TUDA E2K dispatch" FORCE)
