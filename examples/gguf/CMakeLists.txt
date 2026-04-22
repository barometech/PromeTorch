# GGUF Inference Test

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_gguf_inference.cpp")
    add_executable(test_gguf_inference test_gguf_inference.cpp)
    target_include_directories(test_gguf_inference PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(test_gguf_inference PRIVATE c10 aten torch_autograd)
    if(PT_USE_CUDA)
        target_link_libraries(test_gguf_inference PRIVATE aten_cuda)
        target_compile_definitions(test_gguf_inference PRIVATE PT_USE_CUDA)
    endif()
    if(MSVC)
        target_link_options(test_gguf_inference PRIVATE /STACK:67108864)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_gguf_inference PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Phase 7.1 — batched Q4_K GEMV benchmark
if(EXISTS "${PROJECT_SOURCE_DIR}/examples/benchmarks/q4k_batched_vs_serial.cpp")
    add_executable(q4k_batched_vs_serial
        ${PROJECT_SOURCE_DIR}/examples/benchmarks/q4k_batched_vs_serial.cpp)
    target_include_directories(q4k_batched_vs_serial PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(q4k_batched_vs_serial PRIVATE c10 aten torch_autograd)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(q4k_batched_vs_serial PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
