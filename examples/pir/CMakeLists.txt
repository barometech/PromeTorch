# PIR Training Examples

# PIR 250M Elbrus Training
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_pir_elbrus.cpp")
    add_executable(train_pir_elbrus train_pir_elbrus.cpp)
    target_include_directories(train_pir_elbrus PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(train_pir_elbrus PRIVATE c10 aten torch_autograd torch_nn torch_optim torch_data)
    if(PT_USE_CUDA)
        target_link_libraries(train_pir_elbrus PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(train_pir_elbrus PRIVATE PT_USE_CUDA)
    endif()
    if(MSVC)
        target_link_options(train_pir_elbrus PRIVATE /STACK:10485760)
    endif()
    # OpenMP for NUMA-aware parallelism on Elbrus
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(train_pir_elbrus PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# MLP Char Model (existing)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_mlp_char.cpp")
    add_executable(train_mlp_char train_mlp_char.cpp)
    target_include_directories(train_mlp_char PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(train_mlp_char PRIVATE c10 aten torch_autograd torch_nn torch_optim torch_data)
    if(PT_USE_CUDA)
        target_link_libraries(train_mlp_char PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(train_mlp_char PRIVATE PT_USE_CUDA)
    endif()
    if(MSVC)
        target_link_options(train_mlp_char PRIVATE /STACK:10485760)
    endif()
endif()
