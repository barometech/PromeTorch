# Transformer Examples

add_executable(train_transformer train_transformer.cpp)
target_include_directories(train_transformer PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/aten/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(train_transformer PRIVATE c10 aten torch_autograd torch_nn torch_optim)
if(PT_USE_CUDA)
    target_link_libraries(train_transformer PRIVATE aten_cuda ${CUDA_LIBRARIES})
    target_compile_definitions(train_transformer PRIVATE PT_USE_CUDA)
endif()

# TEST_PLAN §5.9 standalone verifier: TransformerEncoderLayer CUDA forward
add_executable(test_transformer_cuda_forward ${PROJECT_SOURCE_DIR}/tests/test_transformer_cuda_forward.cpp)
target_include_directories(test_transformer_cuda_forward PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/aten/src
)
target_link_libraries(test_transformer_cuda_forward PRIVATE c10 aten torch_autograd torch_nn torch_optim)
if(PT_USE_CUDA)
    target_link_libraries(test_transformer_cuda_forward PRIVATE aten_cuda ${CUDA_LIBRARIES})
    target_compile_definitions(test_transformer_cuda_forward PRIVATE PT_USE_CUDA)
endif()
