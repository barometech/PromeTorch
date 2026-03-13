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
# OpenMP
# ============================================================================

if(PT_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found, disabling")
        set(PT_USE_OPENMP OFF)
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

    # CUDA compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

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

if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(c10 PUBLIC OpenMP::OpenMP_CXX)
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

# ATen is header-only for now, create interface library
add_library(aten INTERFACE)

target_include_directories(aten INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(aten INTERFACE c10)

if(PT_USE_CUDA)
    target_link_libraries(aten INTERFACE aten_cuda)
    target_compile_definitions(aten INTERFACE PT_USE_CUDA)
endif()

if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(aten INTERFACE OpenMP::OpenMP_CXX)
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

# Autograd is header-only for now, create interface library
add_library(torch_autograd INTERFACE)

target_include_directories(torch_autograd INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(torch_autograd INTERFACE aten)

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

    # PIR 270M Test executable
    add_executable(pir_tests
        test/cpp/test_pir.cpp
    )

    target_link_libraries(pir_tests
        PRIVATE
        torch_data
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(pir_tests)
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

add_subdirectory(examples/pir)
add_subdirectory(examples/mnist)
add_subdirectory(examples/rnn)
add_subdirectory(examples/transformer)
add_subdirectory(examples/vit)
add_subdirectory(examples/gguf)

# ============================================================================
# Installation
# ============================================================================

set(INSTALL_TARGETS c10 aten torch_autograd torch_nn torch_optim torch_data)
if(PT_USE_CUDA)
    list(APPEND INSTALL_TARGETS aten_cuda)
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
message(STATUS "  AVX: ${PT_USE_AVX}")
message(STATUS "  AVX2: ${PT_USE_AVX2}")
message(STATUS "")
