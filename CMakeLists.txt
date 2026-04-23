cmake_minimum_required(VERSION 3.18)

project(PromeTorch
    VERSION 0.1.0
    DESCRIPTION "Deep Learning Framework from Scratch"
    LANGUAGES CXX
)

# ============================================================================
# Build Options
# ============================================================================

option(PT_BUILD_SHARED_LIBS "Build shared libraries" ON)
option(PT_BUILD_TESTS "Build tests" ON)
option(PT_USE_CUDA "Enable CUDA support" OFF)
option(PT_USE_ROCM "Enable AMD ROCm/HIP backend (hipify CUDA sources)" OFF)
option(PT_USE_OPENMP "Enable OpenMP for CPU parallelization" OFF)

# CUDA and ROCm share CMake's CUDA language slot but target different
# toolchains (nvcc vs hipcc) — they cannot coexist in one build.
if(PT_USE_CUDA AND PT_USE_ROCM)
    message(FATAL_ERROR
        "PT_USE_CUDA and PT_USE_ROCM are mutually exclusive. Pick one GPU "
        "backend. ROCm builds run scripts/hipify.sh over the CUDA tree and "
        "compile the translated sources with hipcc.")
endif()
option(PT_USE_AVX "Enable AVX instructions" ON)
option(PT_USE_AVX2 "Enable AVX2 instructions" ON)
option(PT_USE_NMCARD "Enable NM Card Mini support (emulator)" OFF)
option(PT_USE_NMQUAD "Enable NM QUAD support (4x NM6408 hardware)" OFF)
option(PT_USE_TUDA "Enable TUDA CPU arch dispatch (AVX2/NEON/E2K/Scalar)" ON)
option(PT_USE_LINQ "Enable LinQ H1M accelerator (emulator)" OFF)

# Optional test suites — disabled by default because they are partial /
# unverified deliverables from the 2026-04-19 agent-burst session. Enable
# manually after integration review.
option(PT_BUILD_GENERATED_TESTS "Build auto-generated op test suite (partial)" OFF)
option(PT_BUILD_DEEPSPEED_TESTS "Build DeepSpeed-style self-tests (partial)" OFF)
option(PT_USE_MPS "Enable Apple Metal Performance Shaders backend (Apple only)" OFF)

# ============================================================================
# C++ Standard
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Compiler Flags
# ============================================================================

if(MSVC)
    # MSVC compiler flags - ONLY for C/CXX, not CUDA
    # Use generator expression to avoid passing these to nvcc
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>           # Warning level 4
        $<$<COMPILE_LANGUAGE:CXX>:/WX->          # Warnings not as errors
        $<$<COMPILE_LANGUAGE:CXX>:/MP>           # Multi-processor compilation
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->  # Strict conformance
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>  # Enable __cplusplus macro
    )

    if(PT_USE_AVX2)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    elseif(PT_USE_AVX)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
    endif()

    # Define for DLL export
    if(PT_BUILD_SHARED_LIBS)
        add_compile_definitions(PT_BUILD_SHARED_LIBS)
    endif()

else()
    # GCC/Clang compiler flags - ONLY for C/CXX
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-function>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-function>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-error>
        $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
    )

    if(PT_USE_AVX2)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
    elseif(PT_USE_AVX)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx>)
    endif()

    # Enable fast math (optional, can cause precision issues)
    # add_compile_options(-ffast-math)
endif()

# ============================================================================
# TUDA — CPU ISA auto-detection for cross-compilation
# ============================================================================

if(PT_USE_TUDA)
    add_compile_definitions(PT_USE_TUDA)

    # Auto-detect target architecture for cross-compile
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        message(STATUS "TUDA: ARM64 detected — NEON path will be used")
        # Check for dotprod (Baikal-S / Cortex-A75+)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=armv8.2-a+dotprod" HAS_DOTPROD)
        if(HAS_DOTPROD)
            message(STATUS "TUDA: ARM dotprod available — NEON_A75 micro-kernel")
        else()
            message(STATUS "TUDA: ARM without dotprod — NEON_A57 micro-kernel")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
        message(STATUS "TUDA: Elbrus E2K detected — E2K path will be used")
        add_compile_definitions(TUDA_E2K)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
        message(STATUS "TUDA: x86_64 detected — AVX2 path will be used")
    else()
        message(STATUS "TUDA: Unknown arch '${CMAKE_SYSTEM_PROCESSOR}' — Scalar fallback")
    endif()
endif()

# ============================================================================
# Elbrus E2K — LCC-specific optimization flags
# ============================================================================

if(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
    message(STATUS "Elbrus E2K: enabling -O3, -ffast-math, LTO, EML BLAS")

    # Aggressive optimization: LCC VLIW scheduling benefits hugely from -O3
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

    # LCC-specific performance flags (from docs/elbrus/LCC_OPTIMIZATION.md +
    # habr smartengines / mcst official optimization guide). These unlock:
    #   * APB  — hardware array prefetch buffer, critical for memory-bound
    #             workloads. Needs both `-fprefetch` AND int64_t loop
    #             counters AND #pragma loop count (see cpu_quant_gemv.h).
    #   * software pipelining with wider window (default 300 ops too small
    #             for our 256-element Q4_K super-block bodies).
    #   * cache-blocked pipelined loops (-fcache-opt).
    #   * mtune=elbrus-8c2 picks the E8C2-specific instruction scheduler
    #             weights instead of the generic e2k default.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprefetch -fcache-opt")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=elbrus-8c2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frestrict-params")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fswp-maxopers=800")

    # Link-Time Optimization: lets LCC optimize across TU boundaries
    # Disabled by default — LCC ar wrapper may not be found
    # Enable manually: -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
    # set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

    # EML SIGILL ROOT CAUSE (2026-04-01): cblas_sgemm from ANY pthread = SIGILL on E2K.
    # Even single-threaded, even 64x64 matrices. Only works from main thread.
    # Fix: removed all pthread/std::thread NUMA wrappers. EML handles NUMA via OMP internally.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")
    add_compile_definitions(TUDA_E2K PT_USE_EML_BLAS)

    # NUMA support for multi-chip Elbrus (4-chip E8C2: 1840 GFLOPS with NUMA-aware GEMM)
    # libnuma enables per-node memory allocation and thread pinning.
    # Without NUMA: 4-chip E8C2 gets only 330 GFLOPS (worse than 1-chip 463 GFLOPS!)
    # With NUMA: 4× node-local parallel = 1840 GFLOPS (linear scaling)
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        message(STATUS "  NUMA: ${NUMA_LIB} — NUMA-aware tiled GEMM enabled")
        add_compile_definitions(PT_USE_NUMA)
    else()
        message(STATUS "  NUMA: not found — single-node EML BLAS only")
    endif()
endif()

# ============================================================================
# OpenMP
# ============================================================================

# Auto-enable OpenMP on Elbrus (32 cores, critical for performance)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
    if(NOT PT_USE_OPENMP)
        message(STATUS "Elbrus detected -- auto-enabling OpenMP for 32-core parallelism")
        set(PT_USE_OPENMP ON)
    endif()
endif()

if(PT_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        # On Elbrus/LCC, OpenMP may not be detected by find_package but -fopenmp works
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
            message(STATUS "OpenMP: Using -fopenmp directly for Elbrus LCC")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
        else()
            message(WARNING "OpenMP not found, disabling")
            set(PT_USE_OPENMP OFF)
        endif()
    endif()
endif()

# ============================================================================
# CUDA
# ============================================================================

if(PT_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA architectures (>= 75 to avoid deprecated warnings)
    # 75 = Turing (RTX 20xx), 80 = Ampere (A100), 86 = Ampere (RTX 30xx), 89 = Ada (RTX 40xx)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

    # CUDA compiler flags — aggressive optimization for inference
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -O3,-v")

    # NOTE: --default-stream per-thread tested but was 30% slower (35 vs 50 tok/s).
    # Keeping explicit blocking stream + CUDA Graph capture instead.

    # Host compiler flags for CUDA (pass MSVC flags through -Xcompiler)
    if(MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W3,/WX-")
    endif()

    # Add CUDA include directories
    include_directories(${CUDAToolkit_INCLUDE_DIRS})

    # ======================================================================
    # cuDNN (optional but recommended for high-performance convolutions)
    # ======================================================================
    option(PT_USE_CUDNN "Enable cuDNN support" ON)

    if(PT_USE_CUDNN)
        # Add cmake module path for FindcuDNN
        list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

        find_package(cuDNN)

        if(cuDNN_FOUND)
            message(STATUS "cuDNN found: ${cuDNN_VERSION}")
            add_compile_definitions(PT_USE_CUDNN)
            include_directories(${cuDNN_INCLUDE_DIRS})
        else()
            message(WARNING "cuDNN not found, disabling cuDNN support")
            set(PT_USE_CUDNN OFF)
        endif()
    endif()
endif()

# ============================================================================
# ROCm / HIP (AMD GPUs)
# ============================================================================
# Strategy: one CUDA source tree. At configure time we invoke
# scripts/hipify.sh to translate aten/src/ATen/cuda/*.{cu,h} into
# ${CMAKE_BINARY_DIR}/hipified/... and compile those with hipcc. The
# checked-in HIPCompat.h catches tokens hipify misses.
#
# Users pass -DCMAKE_CXX_COMPILER=hipcc on the cmake line (see docs/ROCM.md).
# ============================================================================

if(PT_USE_ROCM)
    message(STATUS "ROCm/HIP backend: ENABLED")

    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH $ENV{ROCM_PATH})
        else()
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()
    message(STATUS "  ROCM_PATH: ${ROCM_PATH}")

    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake")

    # HIP as a first-class CMake language (requires CMake 3.21+). hipcc
    # drives compilation; *.cu files are dispatched through it.
    enable_language(HIP)

    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES
            gfx906   # MI50 / Radeon VII
            gfx908   # MI100
            gfx90a   # MI210 / MI250
            gfx1030  # RX 6800/6900 (RDNA2)
            gfx1100  # RX 7900 XTX (RDNA3)
            gfx1101) # RX 7800/7700 (RDNA3)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    message(STATUS "  HIP archs: ${CMAKE_HIP_ARCHITECTURES}")

    find_package(hip REQUIRED)
    find_package(rocblas REQUIRED)
    find_package(miopen QUIET)  # cuDNN analogue — optional
    if(miopen_FOUND)
        message(STATUS "  MIOpen: FOUND")
    else()
        message(STATUS "  MIOpen: not found (convolution path disabled)")
    endif()

    # Translate CUDA sources at configure time.
    set(PT_HIP_SRC_DIR "${CMAKE_SOURCE_DIR}/aten/src/ATen/cuda")
    set(PT_HIP_OUT_DIR "${CMAKE_BINARY_DIR}/hipified/aten/src/ATen/cuda")

    message(STATUS "  Running hipify: ${PT_HIP_SRC_DIR} -> ${PT_HIP_OUT_DIR}")
    execute_process(
        COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/hipify.sh"
                     "${PT_HIP_SRC_DIR}" "${PT_HIP_OUT_DIR}"
        RESULT_VARIABLE _hipify_rc
        OUTPUT_VARIABLE _hipify_out
        ERROR_VARIABLE  _hipify_err)
    if(NOT _hipify_rc EQUAL 0)
        message(FATAL_ERROR "hipify.sh failed (rc=${_hipify_rc}):\n"
                            "stdout: ${_hipify_out}\n"
                            "stderr: ${_hipify_err}")
    endif()
    message(STATUS "  hipify: OK")

    include_directories("${CMAKE_BINARY_DIR}/hipified" "${ROCM_PATH}/include")
    add_compile_definitions(PT_USE_ROCM __HIP_PLATFORM_AMD__)
endif()

# ============================================================================
# Include Directories
# ============================================================================

include_directories(${CMAKE_SOURCE_DIR})

# ============================================================================
# c10 Library (Core)
# ============================================================================

set(C10_SOURCES
    c10/core/Allocator.cpp
    c10/core/Device.cpp
    c10/core/TensorImpl.cpp
    c10/util/Exception.cpp
)

# Check if source files exist, if not create dummy ones
foreach(src ${C10_SOURCES})
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${src})
        # Get directory
        get_filename_component(src_dir ${src} DIRECTORY)
        file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/${src_dir})

        # Create stub file
        file(WRITE ${CMAKE_SOURCE_DIR}/${src}
            "// Auto-generated stub\n#include \"${src_dir}/../${src_dir}.h\"\n"
        )
    endif()
endforeach()

set(C10_HEADERS
    c10/macros/Macros.h
    c10/core/ScalarType.h
    c10/core/Device.h
    c10/core/Allocator.h
    c10/core/Storage.h
    c10/core/TensorImpl.h
    c10/util/Exception.h
)

if(PT_BUILD_SHARED_LIBS)
    add_library(c10 SHARED ${C10_SOURCES})
else()
    add_library(c10 STATIC ${C10_SOURCES})
endif()

target_include_directories(c10 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(PT_USE_OPENMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(c10 PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# ============================================================================
# ATen Library (Tensor Operations)
# ============================================================================

set(ATEN_HEADERS
    aten/src/ATen/ATen.h
    aten/src/ATen/core/Tensor.h
    aten/src/ATen/core/TensorFactory.h
    aten/src/ATen/native/cpu/MathOps.h
    aten/src/ATen/native/cpu/ReduceOps.h
    aten/src/ATen/native/cpu/LinearAlgebra.h
    aten/src/ATen/native/cpu/Einsum.h
    aten/src/ATen/native/cpu/ShapeOps.h
    aten/src/ATen/native/cpu/IndexOps.h
)

# CUDA sources for ATen
if(PT_USE_CUDA)
    set(ATEN_CUDA_SOURCES
        aten/src/ATen/cuda/CUDAKernels.cu
        aten/src/ATen/cuda/FP16Kernels.cu
        aten/src/ATen/cuda/CUDAReduce.cu
        aten/src/ATen/cuda/CUDABlas.cu
        aten/src/ATen/cuda/CUDAConv.cu
        aten/src/ATen/cuda/CUDAInference.cu
        aten/src/ATen/cuda/FlashDecoding.cu
        aten/src/ATen/cuda/CUDAQuantGemv.cu
        aten/src/ATen/cuda/FlashAttention.cu
        # CRITICAL: CUDAAllocator singleton must be in a .cpp file (not header)
        # to avoid DLL boundary issues on Windows. Each DLL must use the SAME
        # allocator instance, otherwise heap corruption occurs.
        c10/cuda/CUDAAllocator.cpp
    )

    set(ATEN_CUDA_HEADERS
        aten/src/ATen/cuda/CUDAOps.h
    )

    # Build CUDA library as SHARED to properly export CUDACachingAllocator singleton
    # CRITICAL: Must be SHARED (DLL) to ensure single allocator instance across all code!
    if(PT_BUILD_SHARED_LIBS)
        add_library(aten_cuda SHARED ${ATEN_CUDA_SOURCES})
        # Define ATEN_CUDA_EXPORTS so ATEN_CUDA_API exports symbols (dllexport)
        # When other targets include CUDAOps.h, ATEN_CUDA_EXPORTS is NOT defined,
        # so ATEN_CUDA_API becomes dllimport - correct behavior!
        target_compile_definitions(aten_cuda PRIVATE ATEN_CUDA_EXPORTS PT_BUILD_SHARED_LIBS)
    else()
        add_library(aten_cuda STATIC ${ATEN_CUDA_SOURCES})
    endif()

    target_include_directories(aten_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(aten_cuda PUBLIC
        c10
        CUDA::cudart
        CUDA::cublas
    )

    # Link cuDNN if available
    if(PT_USE_CUDNN AND cuDNN_FOUND)
        target_link_libraries(aten_cuda PUBLIC ${cuDNN_LIBRARIES})
        target_compile_definitions(aten_cuda PUBLIC PT_USE_CUDNN)
    endif()

    set_target_properties(aten_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    # nvcc does not reliably forward __declspec(dllexport) to host-side
    # functions defined in .cu files, so ATEN_CUDA_API on the launch_* defs
    # does not end up as /EXPORT in the resulting object and they are missing
    # from the final DLL. Fix: inject an explicit module-definition file that
    # names the host launchers we need cross-DLL. This is narrower (and
    # safer) than WINDOWS_EXPORT_ALL_SYMBOLS, which also re-exports every RTTI
    # / std:: template symbol and produces ODR conflicts with c10.dll.
    if(MSVC AND PT_USE_CUDA AND PT_BUILD_SHARED_LIBS)
        target_sources(aten_cuda PRIVATE
            ${CMAKE_SOURCE_DIR}/aten/src/ATen/cuda/aten_cuda_exports.def)
    endif()
endif()

# ROCm/HIP path: build aten_cuda from the hipified sources under
# ${CMAKE_BINARY_DIR}/hipified/. The filename stays "aten_cuda" so the rest
# of the build graph (INTERFACE links, Python bindings) is untouched.
if(PT_USE_ROCM)
    set(_HIPIFIED_DIR "${CMAKE_BINARY_DIR}/hipified/aten/src/ATen/cuda")
    set(ATEN_HIP_SOURCES
        ${_HIPIFIED_DIR}/CUDAKernels.cu
        ${_HIPIFIED_DIR}/CUDAReduce.cu
        ${_HIPIFIED_DIR}/CUDABlas.cu
        ${_HIPIFIED_DIR}/CUDAConv.cu
        ${_HIPIFIED_DIR}/CUDAInference.cu
        ${_HIPIFIED_DIR}/FlashDecoding.cu
        ${_HIPIFIED_DIR}/CUDAQuantGemv.cu
        ${_HIPIFIED_DIR}/FlashAttention.cu
        # Allocator stays in the original tree — it's pure host C++ and the
        # DLL-singleton constraint is identical for ROCm on Linux.
        c10/cuda/CUDAAllocator.cpp
    )
    # Only pull in FP16Kernels.cu if the CUDA tree shipped it (some branches do).
    if(EXISTS "${_HIPIFIED_DIR}/FP16Kernels.cu")
        list(APPEND ATEN_HIP_SOURCES ${_HIPIFIED_DIR}/FP16Kernels.cu)
    endif()

    # Mark .cu files as HIP language so CMake dispatches hipcc instead of nvcc.
    foreach(_src ${ATEN_HIP_SOURCES})
        if(_src MATCHES "\\.cu$")
            set_source_files_properties(${_src} PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()

    if(PT_BUILD_SHARED_LIBS)
        add_library(aten_cuda SHARED ${ATEN_HIP_SOURCES})
        target_compile_definitions(aten_cuda PRIVATE ATEN_CUDA_EXPORTS PT_BUILD_SHARED_LIBS)
    else()
        add_library(aten_cuda STATIC ${ATEN_HIP_SOURCES})
    endif()

    target_include_directories(aten_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/hipified>
        $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(aten_cuda PUBLIC c10 hip::host roc::rocblas)
    if(miopen_FOUND)
        target_link_libraries(aten_cuda PUBLIC MIOpen)
        target_compile_definitions(aten_cuda PUBLIC PT_USE_MIOPEN)
    endif()

    target_compile_definitions(aten_cuda PUBLIC PT_USE_ROCM __HIP_PLATFORM_AMD__)

    set_target_properties(aten_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ATen CPU: compiled hot-path inner loops for LTO optimization on Elbrus.
# Previously INTERFACE (header-only), now STATIC with hot_loops.cpp.
# This lets LCC schedule VLIW bundles across mm/add/relu/reduce inner loops.
set(ATEN_CPU_SOURCES
    aten/src/ATen/native/cpu/hot_loops.cpp
)

add_library(aten_cpu STATIC ${ATEN_CPU_SOURCES})

target_include_directories(aten_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(aten_cpu PUBLIC c10)

# On Elbrus: link EML BLAS for optimized SGEMM
# STRATEGY (2026-04-06): libeml_mt gives 245 GFLOPS/node vs 50 ST.
#   libeml_mt from pthreads = SIGILL. From main thread (numactl) = OK.
#   So: link libeml_mt, disable NUMA pthread pool at runtime (PT_NO_NUMA_POOL=1),
#   run 4 processes with numactl --cpunodebind=N --membind=N.
# FALLBACK: PT_NO_NUMA_POOL unset → uses libeml ST pthread pool (180 GFLOPS).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
    target_compile_options(aten_cpu PRIVATE -O3 -ffast-math -D_GNU_SOURCE)
    target_link_libraries(aten_cpu PUBLIC eml_mt pthread)
    # NUMA: link libnuma for NUMA-aware tiled GEMM on multi-chip systems
    if(NUMA_LIB)
        target_link_libraries(aten_cpu PUBLIC ${NUMA_LIB})
    endif()
endif()

# On x86: try to find system BLAS (MKL, OpenBLAS) for optimized GEMM
# MKL gives ~2-3x speedup over our AVX2 GEMM on Intel CPUs (cache tuning)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "System BLAS found: ${BLAS_LIBRARIES}")
        target_link_libraries(aten_cpu PUBLIC ${BLAS_LIBRARIES})
        target_compile_definitions(aten_cpu PUBLIC PT_USE_SYSTEM_BLAS)
        # Try to find cblas.h header
        find_path(CBLAS_INCLUDE_DIR cblas.h
            HINTS
                ENV MKLROOT
                ENV OpenBLAS_HOME
                ENV BLAS_HOME
            PATH_SUFFIXES include include/mkl include/openblas
        )
        if(CBLAS_INCLUDE_DIR)
            message(STATUS "  cblas.h found at: ${CBLAS_INCLUDE_DIR}")
            target_include_directories(aten_cpu PUBLIC ${CBLAS_INCLUDE_DIR})
        endif()
    else()
        message(STATUS "System BLAS not found — using built-in AVX2 GEMM")
    endif()
endif()

if(PT_USE_OPENMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aten_cpu PUBLIC OpenMP::OpenMP_CXX)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
        # LCC: -fopenmp already set globally via CMAKE_CXX_FLAGS
        target_compile_options(aten_cpu PRIVATE -fopenmp)
    endif()
endif()

set_target_properties(aten_cpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ATen umbrella library: headers + compiled CPU ops + optional backends
add_library(aten INTERFACE)

target_include_directories(aten INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(aten INTERFACE c10 aten_cpu)

if(PT_USE_CUDA)
    target_link_libraries(aten INTERFACE aten_cuda)
    target_compile_definitions(aten INTERFACE PT_USE_CUDA)
endif()

if(PT_USE_ROCM)
    # aten_cuda is built from hipified sources; code that guards on
    # PT_USE_CUDA sees it too so callers don't have to ifdef twice.
    target_link_libraries(aten INTERFACE aten_cuda)
    target_compile_definitions(aten INTERFACE PT_USE_CUDA PT_USE_ROCM)
endif()

# ============================================================================
# NM Card Mini Backend (Emulator + Hardware)
# ============================================================================

if(PT_USE_NMCARD)
    set(NMCARD_SOURCES
        c10/nmcard/NMCardAllocator.cpp
        aten/src/ATen/nmcard/NMCardEmulator.cpp
        aten/src/ATen/nmcard/NMCardHardware.cpp
    )

    if(PT_BUILD_SHARED_LIBS)
        add_library(aten_nmcard SHARED ${NMCARD_SOURCES})
        target_compile_definitions(aten_nmcard PRIVATE ATEN_NMCARD_EXPORTS PT_BUILD_SHARED_LIBS)
    else()
        add_library(aten_nmcard STATIC ${NMCARD_SOURCES})
    endif()

    target_include_directories(aten_nmcard PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    target_link_libraries(aten_nmcard PUBLIC c10)

    set_target_properties(aten_nmcard PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    target_link_libraries(aten INTERFACE aten_nmcard)
    target_compile_definitions(aten INTERFACE PT_USE_NMCARD)
endif()

if(PT_USE_NMQUAD)
    # NM QUAD backend — links against libnm_quad_load.so (real hardware)
    target_compile_definitions(aten_cpu PUBLIC PT_USE_NMQUAD)
    target_compile_definitions(c10 PUBLIC PT_USE_NMQUAD)
    target_link_libraries(aten_cpu PUBLIC nm_quad_load)
endif()

# ============================================================================
# LinQ H1M Backend (Emulator)
# ============================================================================

if(PT_USE_LINQ)
    set(LINQ_SOURCES
        c10/linq/LinQAllocator.cpp
    )

    if(PT_BUILD_SHARED_LIBS)
        add_library(aten_linq SHARED ${LINQ_SOURCES})
        target_compile_definitions(aten_linq PRIVATE BUILDING_ATEN_LINQ PT_BUILD_SHARED_LIBS)
    else()
        add_library(aten_linq STATIC ${LINQ_SOURCES})
        target_compile_definitions(aten_linq PRIVATE BUILDING_ATEN_LINQ)
    endif()

    target_include_directories(aten_linq PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    target_link_libraries(aten_linq PUBLIC c10)

    set_target_properties(aten_linq PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    target_link_libraries(aten INTERFACE aten_linq)
    target_compile_definitions(aten INTERFACE PT_USE_LINQ)
endif()

# ============================================================================
# MPS Backend (Apple Metal Performance Shaders — Apple platforms only)
# ============================================================================
# Enable via -DPT_USE_MPS=ON on macOS. CMake links Metal, MPS and MPSGraph
# frameworks. If anyone toggles this on a non-Apple host, we fall back to
# the error-only stubs (the .mm files compile the `#else` branch), so it
# won't break the build but won't actually do anything either.

if(PT_USE_MPS)
    if(NOT APPLE)
        message(WARNING "PT_USE_MPS=ON but CMAKE_HOST_SYSTEM is not Apple. "
                        "MPS stubs will throw at runtime. Set OFF on Elbrus/Linux/Windows.")
    endif()

    # Objective-C++ sources. Safe to list on non-Apple hosts: CMake marks
    # them as OBJCXX, and the #if-guards inside make them no-ops.
    set(MPS_SOURCES
        c10/mps/MPSAllocator.mm
        aten/src/ATen/mps/MPSDevice.mm
        aten/src/ATen/mps/MPSKernels.mm
    )

    if(APPLE)
        enable_language(OBJCXX)
        set_source_files_properties(${MPS_SOURCES} PROPERTIES
            COMPILE_FLAGS "-fobjc-arc")
    endif()

    if(PT_BUILD_SHARED_LIBS)
        add_library(aten_mps SHARED ${MPS_SOURCES})
        target_compile_definitions(aten_mps PRIVATE ATEN_MPS_EXPORTS PT_BUILD_SHARED_LIBS)
    else()
        add_library(aten_mps STATIC ${MPS_SOURCES})
    endif()

    target_include_directories(aten_mps PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    target_compile_definitions(aten_mps PUBLIC PT_USE_MPS)
    target_link_libraries(aten_mps PUBLIC c10)

    if(APPLE)
        # Link against the actual Metal frameworks only when we're on Apple.
        find_library(METAL_FRAMEWORK   Metal   REQUIRED)
        find_library(MPS_FRAMEWORK     MetalPerformanceShaders REQUIRED)
        find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        target_link_libraries(aten_mps PUBLIC
            ${METAL_FRAMEWORK}
            ${MPS_FRAMEWORK}
            ${MPSGRAPH_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
        )
    endif()

    set_target_properties(aten_mps PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    target_link_libraries(aten INTERFACE aten_mps)
    target_compile_definitions(aten INTERFACE PT_USE_MPS)
endif()

if(PT_USE_OPENMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aten INTERFACE OpenMP::OpenMP_CXX)
    endif()
    # On Elbrus without find_package: -fopenmp is already in CMAKE_CXX_FLAGS globally
endif()

# ============================================================================
# Torch Autograd Library
# ============================================================================

set(AUTOGRAD_HEADERS
    torch/csrc/autograd/autograd.h
    torch/csrc/autograd/edge.h
    torch/csrc/autograd/node.h
    torch/csrc/autograd/autograd_meta.h
    torch/csrc/autograd/engine.h
    torch/csrc/autograd/functions/MathBackward.h
    torch/csrc/autograd/functions/ReduceBackward.h
    torch/csrc/autograd/functions/LinearAlgebraBackward.h
    torch/csrc/autograd/functions/ShapeBackward.h
)

# Autograd library (grad_mode.cpp provides DLL-safe singleton for GradMode)
# Also hosts torch::distributed (TCP-socket DDP) — header-only on Windows,
# real POSIX implementation on Elbrus/Linux.
add_library(torch_autograd STATIC
    torch/csrc/autograd/grad_mode.cpp
    torch/distributed/ddp.cpp
)

target_include_directories(torch_autograd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(torch_autograd PUBLIC aten)

# DDP needs sockets. On Windows we link Ws2_32 via #pragma comment; on POSIX
# nothing extra is required (libc provides socket(), close()).
if(NOT WIN32)
    target_link_libraries(torch_autograd PUBLIC pthread)
endif()

# ============================================================================
# Torch NN Library (Neural Network Modules)
# ============================================================================

set(NN_HEADERS
    torch/nn/parameter.h
    torch/nn/module.h
    torch/nn/init.h
    torch/nn/nn.h
    torch/nn/functional.h
    torch/nn/modules/container.h
    torch/nn/modules/linear.h
    torch/nn/modules/activation.h
    torch/nn/modules/conv.h
    torch/nn/modules/pooling.h
    torch/nn/modules/normalization.h
    torch/nn/modules/dropout.h
    torch/nn/modules/sparse.h
    torch/nn/modules/loss.h
    torch/nn/parallel/tensor_parallel.h
    torch/nn/parallel/pipeline.h
)

# NN is header-only for now, create interface library
add_library(torch_nn INTERFACE)

target_include_directories(torch_nn INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(torch_nn INTERFACE torch_autograd)

# ============================================================================
# Torch Optim Library (Optimizers)
# ============================================================================

set(OPTIM_HEADERS
    torch/optim/optimizer.h
    torch/optim/sgd.h
    torch/optim/adam.h
    torch/optim/adamw.h
    torch/optim/rmsprop.h
    torch/optim/lr_scheduler.h
    torch/optim/optim.h
    # New optimizers added 2026-04-19
    torch/optim/lion.h
    torch/optim/sophia.h
    torch/optim/lamb.h
    torch/optim/adafactor.h
    torch/optim/nadam.h
    torch/optim/radam.h
    torch/optim/adagrad.h
    torch/optim/adamax.h
    torch/optim/asgd.h
    torch/optim/lbfgs.h
)

# Optim is header-only for now, create interface library
add_library(torch_optim INTERFACE)

target_include_directories(torch_optim INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(torch_optim INTERFACE torch_nn)

# ============================================================================
# Torch Data Library (DataLoader)
# ============================================================================

set(DATA_HEADERS
    torch/data/dataset.h
    torch/data/sampler.h
    torch/data/dataloader.h
    torch/data/data.h
    torch/data/distributed_sampler.h
)

# Data is header-only for now, create interface library
add_library(torch_data INTERFACE)

target_include_directories(torch_data INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(torch_data INTERFACE torch_optim)

# ============================================================================
# Torch Extras — header-only modules added 2026-04-19.
# ONNX, JIT, Mobile, MLIR, Vision, Trainer, NN-parallel, Quantization, Distributed,
# Forward-AD, vmap. Each is an INTERFACE library so IDEs see the headers and
# install targets export them.
# ============================================================================

set(TORCH_EXTRAS_HEADERS
    # ONNX export
    torch/onnx/export.h
    # JIT compile prototype
    torch/jit/compile.h
    torch/jit/codegen_cpp.h
    # Mobile serialization (ExecuTorch-like)
    torch/mobile/executor.h
    # MLIR text export
    torch/mlir/export.h
    # Vision (torchvision-compatible)
    torch/vision/vision.h
    torch/vision/transforms.h
    torch/vision/models.h
    # Text (torchtext-compatible)
    torch/text/text.h
    torch/text/vocab.h
    torch/text/tokenizers.h
    # Audio (torchaudio-compatible)
    torch/audio/audio.h
    torch/audio/transforms.h
    torch/audio/functional.h
    # Lightning-style trainer
    torch/trainer/trainer.h
    # Quantization (INT8 QAT)
    torch/quantization/qat.h
    torch/quantization/observer.h
    torch/quantization/quantization.h
    torch/quantization/quantize.h
    torch/quantization/quant4.h
    # Distributed (TCP DDP, FSDP, cluster launcher, DeepSpeed-style)
    torch/distributed/ddp.h
    torch/distributed/fsdp.h
    torch/distributed/launcher.h
    torch/distributed/deepspeed.h
    torch/distributed/deepspeed_offload.h
    torch/distributed/pipeline_schedule.h
    torch/distributed/zero3.h
    # Forward-mode AD, vmap
    torch/autograd/forward_ad.h
    torch/autograd/vmap.h
    # LLM serving engine
    torch/serve/llm.h
)

add_library(torch_extras INTERFACE)

target_include_directories(torch_extras INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(torch_extras INTERFACE torch_data)

# ============================================================================
# Tests
# ============================================================================

if(PT_BUILD_TESTS)
    enable_testing()

    # Download and configure Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    # c10 Test executable
    add_executable(c10_tests
        test/cpp/test_scalar_type.cpp
        test/cpp/test_device.cpp
        test/cpp/test_allocator.cpp
        test/cpp/test_storage.cpp
        test/cpp/test_tensor_impl.cpp
    )

    target_link_libraries(c10_tests
        PRIVATE
        c10
        GTest::gtest
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(c10_tests)

    # ATen Test executable
    add_executable(aten_tests
        test/cpp/test_tensor.cpp
    )

    target_link_libraries(aten_tests
        PRIVATE
        aten
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(aten_tests)

    # Autograd Test executable
    add_executable(autograd_tests
        test/cpp/test_autograd.cpp
    )

    target_link_libraries(autograd_tests
        PRIVATE
        torch_autograd
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(autograd_tests)

    # NN Module Test executable (standalone, not using GTest)
    add_executable(nn_tests
        test/cpp/test_nn.cpp
    )

    target_link_libraries(nn_tests
        PRIVATE
        torch_nn
    )

    if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(nn_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME nn_tests COMMAND nn_tests)

    # Optimizer Test executable
    add_executable(optim_tests
        test/cpp/test_optim.cpp
    )

    target_link_libraries(optim_tests
        PRIVATE
        torch_optim
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(optim_tests)

    # Comprehensive Optimizer Tests (all 9 optimizers)
    add_executable(optimizer_tests
        test/cpp/test_optimizers.cpp
    )

    target_link_libraries(optimizer_tests
        PRIVATE
        torch_optim
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(optimizer_tests)

    # Data Loading Test executable
    add_executable(data_tests
        test/cpp/test_data.cpp
    )

    target_link_libraries(data_tests
        PRIVATE
        torch_data
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(data_tests)

    # Comprehensive All-Ops Test
    add_executable(all_ops_tests
        test/cpp/test_all_ops.cpp
    )

    target_link_libraries(all_ops_tests
        PRIVATE
        aten
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(all_ops_tests)

    # Auto-generated comprehensive op test suite (parameterized across
    # ops x shapes x layouts). Targets 5000+ gtest cases.
    # Gated behind PT_BUILD_GENERATED_TESTS (default OFF) — currently has linker
    # issues with GradMode singleton. Enable after linker fix is in.
    if(PT_BUILD_GENERATED_TESTS)
        add_executable(test_ops_generated
            test/cpp/test_ops_generated.cpp
            test/cpp/ops_spec.cpp
        )

        target_link_libraries(test_ops_generated
            PRIVATE
            aten
            torch_autograd
            GTest::gtest
            GTest::gtest_main
        )

        add_test(NAME test_ops_generated COMMAND test_ops_generated)
    endif()

    # Comprehensive Autograd Test
    add_executable(autograd_full_tests
        test/cpp/test_autograd_full.cpp
    )

    target_link_libraries(autograd_full_tests
        PRIVATE
        torch_autograd
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(autograd_full_tests)

    # NN Modules Test
    add_executable(nn_modules_tests
        test/cpp/test_nn_modules.cpp
    )

    target_link_libraries(nn_modules_tests
        PRIVATE
        torch_nn
        GTest::gtest
        GTest::gtest_main
    )

    if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(nn_modules_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()

    gtest_discover_tests(nn_modules_tests)

    # Attention / SDPA Test (forward + backward, bool/causal/additive masks, dropout)
    add_executable(attention_tests
        test/cpp/test_attention.cpp
    )
    target_link_libraries(attention_tests
        PRIVATE
        torch_autograd
        torch_nn
        GTest::gtest
        GTest::gtest_main
    )
    if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(attention_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    gtest_discover_tests(attention_tests)

    # NN Functional Test
    add_executable(nn_functional_tests
        test/cpp/test_nn_functional_full.cpp
    )

    target_link_libraries(nn_functional_tests
        PRIVATE
        torch_nn
        GTest::gtest
        GTest::gtest_main
    )

    if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(nn_functional_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()

    gtest_discover_tests(nn_functional_tests)

    # Edge Cases Test
    add_executable(edge_case_tests
        test/cpp/test_edge_cases.cpp
    )

    target_link_libraries(edge_case_tests
        PRIVATE
        aten
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(edge_case_tests)

    # DDP (TCP-socket distributed) self-test — fork-based, POSIX only.
    add_executable(ddp_tests
        test/cpp/test_ddp.cpp
    )
    target_link_libraries(ddp_tests PRIVATE torch_autograd)
    if(NOT WIN32)
        add_test(NAME ddp_tests COMMAND ddp_tests)
    endif()

    # FSDP / ZeRO self-test — fork-based, POSIX only.
    add_executable(fsdp_tests
        test/cpp/test_fsdp.cpp
    )
    target_link_libraries(fsdp_tests PRIVATE torch_autograd)
    if(NOT WIN32)
        add_test(NAME fsdp_tests COMMAND fsdp_tests)
    endif()

    # Tensor-parallel GGUF inference self-test — fork-based, POSIX only.
    # Loads a GGUF model (path passed as CLI arg or GGUF_DDP_MODEL env var),
    # runs single-process reference decode + world_size-process TP decode,
    # verifies logits cos-sim >= 0.999 and argmax match.
    add_executable(gguf_ddp_tests
        test/cpp/test_gguf_ddp.cpp
    )
    target_include_directories(gguf_ddp_tests PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(gguf_ddp_tests PRIVATE c10 aten torch_autograd)
    if(MSVC)
        target_link_options(gguf_ddp_tests PRIVATE /STACK:67108864)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gguf_ddp_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    # Not registered via add_test — requires a GGUF model path supplied at run-time.

    # DeepSpeed-style wrappers (CPU offload, 1F1B pipeline, ZeRO-3).
    # Gated behind PT_BUILD_DEEPSPEED_TESTS (default OFF) — partial from killed
    # agent burst, not fully verified. Enable after integration review.
    if(PT_BUILD_DEEPSPEED_TESTS)
        add_executable(deepspeed_tests
            test/cpp/test_deepspeed.cpp
        )
        target_link_libraries(deepspeed_tests PRIVATE torch_autograd)
        if(NOT WIN32)
            add_test(NAME deepspeed_tests COMMAND deepspeed_tests)
        endif()
    endif()

    # NM Card Mini Emulator Test (only when NMCard is enabled)
    if(PT_USE_NMCARD)
        add_executable(nmcard_tests
            test/cpp/test_nmcard.cpp
        )
        target_link_libraries(nmcard_tests
            PRIVATE
            aten
        )
        add_test(NAME nmcard_tests COMMAND nmcard_tests)
    endif()

    # LinQ H1M Emulator Test (only when LinQ is enabled)
    if(PT_USE_LINQ)
        add_executable(linq_tests
            test/cpp/test_linq.cpp
        )
        target_link_libraries(linq_tests
            PRIVATE
            aten
        )
        add_test(NAME linq_tests COMMAND linq_tests)
    endif()

    # TUDA CPU Architecture Test
    if(PT_USE_TUDA)
        add_executable(tuda_tests
            test/cpp/test_tuda.cpp
        )
        target_link_libraries(tuda_tests
            PRIVATE
            aten
        )
        add_test(NAME tuda_tests COMMAND tuda_tests)
    endif()
endif()

# ============================================================================
# Python Bindings (pybind11)
# ============================================================================

option(PT_BUILD_PYTHON "Build Python bindings" OFF)

if(PT_BUILD_PYTHON)
    # Find Python
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    # Add pybind11
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    # Python extension module
    pybind11_add_module(_C MODULE
        python/csrc/init.cpp
        python/csrc/tensor_bindings.cpp
        python/csrc/autograd_bindings.cpp
        python/csrc/nn_bindings.cpp
        python/csrc/optim_bindings.cpp
        python/csrc/bindings_new.cpp
    )

    target_include_directories(_C PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${Python3_INCLUDE_DIRS}
    )

    target_link_libraries(_C PRIVATE
        c10
        torch_data
    )

    # Enable detailed pybind11 error messages
    target_compile_definitions(_C PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

    if(PT_USE_CUDA)
        target_link_libraries(_C PRIVATE aten_cuda)
        target_compile_definitions(_C PRIVATE PT_USE_CUDA)
    endif()

    if(PT_USE_ROCM)
        target_link_libraries(_C PRIVATE aten_cuda)
        target_compile_definitions(_C PRIVATE PT_USE_CUDA PT_USE_ROCM)
    endif()

    # Set output directory for Python package
    set_target_properties(_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/promethorch
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/promethorch
    )

    message(STATUS "  Python: ${Python3_VERSION}")
    message(STATUS "  pybind11: Enabled")
endif()

# ============================================================================
# Examples
# ============================================================================

# All example subdirs are optional — guarded by `if(EXISTS ...)` so a
# partial clone / missing sub-CMakeLists.txt doesn't break the whole build.
foreach(_ex_dir mnist vae mobile jit mlir rnn transformer vit cifar pir gguf gan shakespeare)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/${_ex_dir}/CMakeLists.txt")
        add_subdirectory(examples/${_ex_dir})
    endif()
endforeach()
if(PT_USE_NMCARD)
    add_subdirectory(examples/nmcard)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/promeserve/CMakeLists.txt")
    add_subdirectory(promeserve)
endif()

# ============================================================================
# Installation
# ============================================================================

set(INSTALL_TARGETS c10 aten_cpu aten torch_autograd torch_nn torch_optim torch_data torch_extras)
if(PT_USE_CUDA OR PT_USE_ROCM)
    list(APPEND INSTALL_TARGETS aten_cuda)
endif()
if(PT_USE_NMCARD)
    list(APPEND INSTALL_TARGETS aten_nmcard)
endif()
if(PT_USE_LINQ)
    list(APPEND INSTALL_TARGETS aten_linq)
endif()
if(PT_USE_MPS)
    list(APPEND INSTALL_TARGETS aten_mps)
endif()

install(TARGETS ${INSTALL_TARGETS}
    EXPORT PromeTorchTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY c10/
    DESTINATION include/c10
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY aten/
    DESTINATION include/aten
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY torch/
    DESTINATION include/torch
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT PromeTorchTargets
    FILE PromeTorchTargets.cmake
    NAMESPACE PromeTorch::
    DESTINATION lib/cmake/PromeTorch
)

# ============================================================================
# Package Configuration
# ============================================================================

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/PromeTorchConfig.cmake.in
    ${CMAKE_BINARY_DIR}/PromeTorchConfig.cmake
    INSTALL_DESTINATION lib/cmake/PromeTorch
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/PromeTorchConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_BINARY_DIR}/PromeTorchConfig.cmake
    ${CMAKE_BINARY_DIR}/PromeTorchConfigVersion.cmake
    DESTINATION lib/cmake/PromeTorch
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "PromeTorch Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared Libraries: ${PT_BUILD_SHARED_LIBS}")
message(STATUS "  Build Tests: ${PT_BUILD_TESTS}")
message(STATUS "  Python Bindings: ${PT_BUILD_PYTHON}")
message(STATUS "  OpenMP: ${PT_USE_OPENMP}")
message(STATUS "  CUDA: ${PT_USE_CUDA}")
if(PT_USE_CUDA AND PT_USE_CUDNN)
    message(STATUS "  cuDNN: ${cuDNN_VERSION}")
else()
    message(STATUS "  cuDNN: OFF")
endif()
message(STATUS "  ROCm/HIP: ${PT_USE_ROCM}")
if(PT_USE_ROCM)
    message(STATUS "  HIP archs: ${CMAKE_HIP_ARCHITECTURES}")
    message(STATUS "  ROCM_PATH: ${ROCM_PATH}")
endif()
message(STATUS "  NMCard Mini: ${PT_USE_NMCARD}")
message(STATUS "  LinQ H1M: ${PT_USE_LINQ}")
message(STATUS "  MPS (Apple Metal): ${PT_USE_MPS}")
message(STATUS "  TUDA CPU Dispatch: ${PT_USE_TUDA}")
message(STATUS "  AVX: ${PT_USE_AVX}")
message(STATUS "  AVX2: ${PT_USE_AVX2}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
    message(STATUS "  Elbrus LTO: ON")
    message(STATUS "  EML BLAS: ON")
    if(NUMA_LIB)
        message(STATUS "  NUMA-aware GEMM: ON (${NUMA_LIB})")
    else()
        message(STATUS "  NUMA-aware GEMM: OFF (libnuma not found)")
    endif()
endif()
message(STATUS "")
