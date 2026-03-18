# PIR 270M Training Example

add_executable(train_pir train_pir.cpp)

target_link_libraries(train_pir
    PRIVATE
    torch_data
)

# Increase stack size to 10MB for deep autograd graphs
if(MSVC)
    target_link_options(train_pir PRIVATE /STACK:10485760)
endif()

if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(train_pir PRIVATE OpenMP::OpenMP_CXX)
endif()

# Simple MLP Training Example (proof of autograd working)
add_executable(train_mlp train_mlp.cpp)

# MLP Char Model for GPU utilization comparison
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_mlp_char.cpp")
add_executable(train_mlp_char train_mlp_char.cpp)

target_link_libraries(train_mlp_char
    PRIVATE
    torch_data
)

if(MSVC)
    target_link_options(train_mlp_char PRIVATE /STACK:10485760)
endif()

if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(train_mlp_char PRIVATE OpenMP::OpenMP_CXX)
endif()
endif() # EXISTS train_mlp_char.cpp

target_link_libraries(train_mlp
    PRIVATE
    torch_data
)

if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(train_mlp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Memory Leak Test - Minimal MLP for debugging GPU memory leaks
add_executable(test_mem_leak ${CMAKE_SOURCE_DIR}/examples/test_mem_leak.cpp)

target_link_libraries(test_mem_leak
    PRIVATE
    torch_data
)

if(MSVC)
    target_link_options(test_mem_leak PRIVATE /STACK:10485760)
endif()

if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_mem_leak PRIVATE OpenMP::OpenMP_CXX)
endif()

# CUDA Isolation Test - for debugging CUDA crash
if(PT_USE_CUDA)
    add_executable(cuda_isolation_test ${CMAKE_SOURCE_DIR}/examples/cuda_isolation_test.cpp)

    target_link_libraries(cuda_isolation_test
        PRIVATE
        c10
        aten_cuda
    )

    # CRITICAL: Add PT_USE_CUDA definition
    target_compile_definitions(cuda_isolation_test PRIVATE PT_USE_CUDA)

    if(MSVC)
        target_link_options(cuda_isolation_test PRIVATE /STACK:10485760)
    endif()
endif()
