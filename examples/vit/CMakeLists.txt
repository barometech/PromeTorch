# Vision Transformer (ViT) Examples

add_executable(train_vit train_vit.cpp)
target_include_directories(train_vit PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/aten/src
    ${PROJECT_SOURCE_DIR}/examples
)
target_link_libraries(train_vit PRIVATE c10 aten torch_autograd torch_nn torch_optim)
if(PT_USE_CUDA)
    target_link_libraries(train_vit PRIVATE aten_cuda)
    target_compile_definitions(train_vit PRIVATE PT_USE_CUDA)
endif()
