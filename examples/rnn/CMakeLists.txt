# RNN Examples

# Simple RNN example
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_rnn.cpp")
    add_executable(train_rnn train_rnn.cpp)
    target_include_directories(train_rnn PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(train_rnn PRIVATE c10 aten torch_autograd torch_nn torch_optim)
    if(PT_USE_CUDA)
        target_link_libraries(train_rnn PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(train_rnn PRIVATE PT_USE_CUDA)
    endif()
endif()

# Full RNN (all variants) example
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/train_rnn_full.cpp")
    add_executable(train_rnn_full train_rnn_full.cpp)
    target_include_directories(train_rnn_full PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
        ${PROJECT_SOURCE_DIR}/examples
    )
    target_link_libraries(train_rnn_full PRIVATE c10 aten torch_autograd torch_nn torch_optim)
    if(PT_USE_CUDA)
        target_link_libraries(train_rnn_full PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(train_rnn_full PRIVATE PT_USE_CUDA)
    endif()
endif()
