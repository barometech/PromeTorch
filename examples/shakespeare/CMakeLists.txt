# CMakeLists.txt for Shakespeare example

add_executable(shakespeare_train
    train.cpp
)

target_link_libraries(shakespeare_train
    PRIVATE
    torch_data
)

target_include_directories(shakespeare_train
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples/shakespeare
)

if(PT_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(shakespeare_train PRIVATE OpenMP::OpenMP_CXX)
endif()
