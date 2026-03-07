# MNIST Training Examples

# MLP Example
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_mnist_mlp.cpp")
    add_executable(train_mnist_mlp train_mnist_mlp.cpp)
    target_include_directories(train_mnist_mlp PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(train_mnist_mlp PRIVATE c10 aten torch_autograd torch_nn torch_optim torch_data)
    if(PT_USE_CUDA)
        target_link_libraries(train_mnist_mlp PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(train_mnist_mlp PRIVATE PT_USE_CUDA)
    endif()
    if(MSVC)
        target_link_options(train_mnist_mlp PRIVATE /STACK:10485760)
    endif()
endif()

# CNN Example
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_mnist_cnn.cpp")
    add_executable(train_mnist_cnn train_mnist_cnn.cpp)
    target_include_directories(train_mnist_cnn PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(train_mnist_cnn PRIVATE c10 aten torch_autograd torch_nn torch_optim torch_data)
    if(PT_USE_CUDA)
        target_link_libraries(train_mnist_cnn PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(train_mnist_cnn PRIVATE PT_USE_CUDA)
    endif()
    if(MSVC)
        target_link_options(train_mnist_cnn PRIVATE /STACK:10485760)
    endif()
endif()

# Legacy train_mnist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_mnist.cpp")
    add_executable(train_mnist train_mnist.cpp)
    target_include_directories(train_mnist PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(train_mnist PRIVATE c10 aten torch_autograd torch_nn torch_optim torch_data)
    if(PT_USE_CUDA)
        target_link_libraries(train_mnist PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(train_mnist PRIVATE PT_USE_CUDA)
    endif()
    if(MSVC)
        target_link_options(train_mnist PRIVATE /STACK:10485760)
    endif()
endif()

# 10 Models Training
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_10_models.cpp")
    add_executable(train_10_models train_10_models.cpp)
    target_include_directories(train_10_models PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(train_10_models PRIVATE c10 aten torch_autograd torch_nn torch_optim torch_data)
    if(PT_USE_CUDA)
        target_link_libraries(train_10_models PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(train_10_models PRIVATE PT_USE_CUDA)
    endif()
    if(MSVC)
        target_link_options(train_10_models PRIVATE /STACK:10485760)
    endif()
endif()

# CUDA Debug Test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_debug.cpp")
    add_executable(test_cuda_debug test_cuda_debug.cpp)
    target_include_directories(test_cuda_debug PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(test_cuda_debug PRIVATE c10 aten torch_autograd torch_nn torch_optim torch_data)
    if(PT_USE_CUDA)
        target_link_libraries(test_cuda_debug PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(test_cuda_debug PRIVATE PT_USE_CUDA)
    endif()
    if(MSVC)
        target_link_options(test_cuda_debug PRIVATE /STACK:10485760)
    endif()
endif()
