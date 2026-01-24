# FindcuDNN.cmake
# ================
# Find NVIDIA cuDNN library
#
# This module finds the cuDNN library and headers.
#
# Result Variables:
#   cuDNN_FOUND        - True if cuDNN was found
#   cuDNN_INCLUDE_DIRS - cuDNN include directories
#   cuDNN_LIBRARIES    - cuDNN libraries to link
#   cuDNN_VERSION      - cuDNN version string
#   cuDNN_VERSION_MAJOR - Major version
#   cuDNN_VERSION_MINOR - Minor version
#   cuDNN_VERSION_PATCH - Patch version
#
# Hints:
#   CUDNN_ROOT         - Root directory to search
#   CUDA_TOOLKIT_ROOT_DIR - CUDA toolkit directory (cuDNN often installed with CUDA)

include(FindPackageHandleStandardArgs)

# Search paths
set(_cuDNN_SEARCH_PATHS
    ${CUDNN_ROOT}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDAToolkit_ROOT}
    $ENV{CUDNN_ROOT}
    $ENV{CUDA_PATH}
    /usr/local/cuda
    /usr/local
    /opt/cuda
)

# For Anaconda on Windows
if(WIN32)
    list(APPEND _cuDNN_SEARCH_PATHS
        $ENV{CONDA_PREFIX}
        $ENV{CONDA_PREFIX}/Library
        "C:/ProgramData/anaconda3/Library"
        "C:/Users/$ENV{USERNAME}/anaconda3/Library"
    )
endif()

# Find header
find_path(cuDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS ${_cuDNN_SEARCH_PATHS}
    PATH_SUFFIXES include include/cudnn
    DOC "cuDNN include directory"
)

# Find library
if(WIN32)
    set(_cuDNN_LIB_NAMES cudnn cudnn64_8 cudnn64_7 cudnn64)
else()
    set(_cuDNN_LIB_NAMES cudnn)
endif()

find_library(cuDNN_LIBRARY
    NAMES ${_cuDNN_LIB_NAMES}
    HINTS ${_cuDNN_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu
    DOC "cuDNN library"
)

# Get version from header
if(cuDNN_INCLUDE_DIR)
    # Try cudnn_version.h first (cuDNN 8+)
    if(EXISTS "${cuDNN_INCLUDE_DIR}/cudnn_version.h")
        file(READ "${cuDNN_INCLUDE_DIR}/cudnn_version.h" _cuDNN_VERSION_HEADER)
    elseif(EXISTS "${cuDNN_INCLUDE_DIR}/cudnn.h")
        file(READ "${cuDNN_INCLUDE_DIR}/cudnn.h" _cuDNN_VERSION_HEADER)
    endif()

    if(_cuDNN_VERSION_HEADER)
        string(REGEX MATCH "#define CUDNN_MAJOR ([0-9]+)" _ "${_cuDNN_VERSION_HEADER}")
        set(cuDNN_VERSION_MAJOR ${CMAKE_MATCH_1})

        string(REGEX MATCH "#define CUDNN_MINOR ([0-9]+)" _ "${_cuDNN_VERSION_HEADER}")
        set(cuDNN_VERSION_MINOR ${CMAKE_MATCH_1})

        string(REGEX MATCH "#define CUDNN_PATCHLEVEL ([0-9]+)" _ "${_cuDNN_VERSION_HEADER}")
        set(cuDNN_VERSION_PATCH ${CMAKE_MATCH_1})

        if(cuDNN_VERSION_MAJOR AND cuDNN_VERSION_MINOR AND cuDNN_VERSION_PATCH)
            set(cuDNN_VERSION "${cuDNN_VERSION_MAJOR}.${cuDNN_VERSION_MINOR}.${cuDNN_VERSION_PATCH}")
        endif()
    endif()
endif()

# Standard args handling
find_package_handle_standard_args(cuDNN
    REQUIRED_VARS cuDNN_LIBRARY cuDNN_INCLUDE_DIR
    VERSION_VAR cuDNN_VERSION
)

# Set output variables
if(cuDNN_FOUND)
    set(cuDNN_INCLUDE_DIRS ${cuDNN_INCLUDE_DIR})
    set(cuDNN_LIBRARIES ${cuDNN_LIBRARY})

    # Create imported target
    if(NOT TARGET cuDNN::cuDNN)
        add_library(cuDNN::cuDNN UNKNOWN IMPORTED)
        set_target_properties(cuDNN::cuDNN PROPERTIES
            IMPORTED_LOCATION "${cuDNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${cuDNN_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(cuDNN_INCLUDE_DIR cuDNN_LIBRARY)
