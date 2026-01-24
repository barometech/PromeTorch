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
