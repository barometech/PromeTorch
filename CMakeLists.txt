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
option(PT_USE_OPENMP "Enable OpenMP for CPU parallelization" OFF)
option(PT_USE_AVX "Enable AVX instructions" ON)
option(PT_USE_AVX2 "Enable AVX2 instructions" ON)
option(PT_USE_NMCARD "Enable NM Card Mini support (emulator)" OFF)
option(PT_USE_NMQUAD "Enable NM QUAD support (4x NM6408 hardware)" OFF)
option(PT_USE_TUDA "Enable TUDA CPU arch dispatch (AVX2/NEON/E2K/Scalar)" ON)
option(PT_USE_LINQ "Enable LinQ H1M accelerator (emulator)" OFF)

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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
    add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM)

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
    aten/src/ATen/native/cpu/ShapeOps.h
    aten/src/ATen/native/cpu/IndexOps.h
)

# CUDA sources for ATen
if(PT_USE_CUDA)
    set(ATEN_CUDA_SOURCES
        aten/src/ATen/cuda/CUDAKernels.cu
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
add_library(torch_autograd STATIC
    torch/csrc/autograd/grad_mode.cpp
)

target_include_directories(torch_autograd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(torch_autograd PUBLIC aten)

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
    torch/optim/rmsprop.h
    torch/optim/lr_scheduler.h
    torch/optim/optim.h
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
)

# Data is header-only for now, create interface library
add_library(torch_data INTERFACE)

target_include_directories(torch_data INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(torch_data INTERFACE torch_optim)

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

add_subdirectory(examples/mnist)
add_subdirectory(examples/rnn)
add_subdirectory(examples/transformer)
add_subdirectory(examples/vit)
add_subdirectory(examples/pir)
add_subdirectory(examples/gguf)
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

set(INSTALL_TARGETS c10 aten_cpu aten torch_autograd torch_nn torch_optim torch_data)
if(PT_USE_CUDA)
    list(APPEND INSTALL_TARGETS aten_cuda)
endif()
if(PT_USE_NMCARD)
    list(APPEND INSTALL_TARGETS aten_nmcard)
endif()
if(PT_USE_LINQ)
    list(APPEND INSTALL_TARGETS aten_linq)
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
message(STATUS "  NMCard Mini: ${PT_USE_NMCARD}")
message(STATUS "  LinQ H1M: ${PT_USE_LINQ}")
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
