# PromeServe — HTTP inference server

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/promeserve.cpp")
    add_executable(promeserve promeserve.cpp)
    target_include_directories(promeserve PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/aten/src
    )
    target_link_libraries(promeserve PRIVATE c10 aten)

    # Windows sockets
    if(WIN32)
        target_link_libraries(promeserve PRIVATE ws2_32)
    endif()

    # CUDA support
    if(PT_USE_CUDA)
        target_link_libraries(promeserve PRIVATE aten_cuda ${CUDA_LIBRARIES})
        target_compile_definitions(promeserve PRIVATE PT_USE_CUDA)
    endif()

    # Large stack for deep model inference
    if(MSVC)
        target_link_options(promeserve PRIVATE /STACK:67108864)
    endif()

    # OpenMP if available
    if(OpenMP_CXX_FOUND)
        target_link_libraries(promeserve PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
