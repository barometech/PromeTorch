# PromeTorch Benchmark Suite
add_executable(bench_promethorch bench_promethorch.cpp)
target_include_directories(bench_promethorch PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/aten/src
)
target_link_libraries(bench_promethorch PRIVATE
    c10 aten torch_autograd torch_nn torch_optim torch_data
)
if(PT_USE_CUDA)
    target_link_libraries(bench_promethorch PRIVATE aten_cuda)
    target_compile_definitions(bench_promethorch PRIVATE PT_USE_CUDA)
endif()
if(MSVC)
    target_link_options(bench_promethorch PRIVATE /STACK:10485760)
endif()

# Focused optimization benchmark
add_executable(bench_optimized bench_optimized.cpp)
target_include_directories(bench_optimized PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/aten/src
)
target_link_libraries(bench_optimized PRIVATE
    c10 aten torch_autograd torch_nn torch_optim torch_data
)
if(PT_USE_CUDA)
    target_link_libraries(bench_optimized PRIVATE aten_cuda)
    target_compile_definitions(bench_optimized PRIVATE PT_USE_CUDA)
endif()
if(MSVC)
    target_link_options(bench_optimized PRIVATE /STACK:10485760)
endif()

# === Round 4 — ThreadPool overhead microbench (Step 1 gate) ===
add_executable(bench_threadpool_overhead
    ${PROJECT_SOURCE_DIR}/examples/benchmarks/threadpool_overhead_bench.cpp)
target_include_directories(bench_threadpool_overhead PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/aten/src
)
target_link_libraries(bench_threadpool_overhead PRIVATE c10)
if(NOT MSVC)
    target_link_libraries(bench_threadpool_overhead PRIVATE pthread)
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        target_link_libraries(bench_threadpool_overhead PRIVATE ${NUMA_LIB})
    endif()
endif()
