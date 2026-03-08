@echo off

rem === Setup MSVC Environment ===
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

rem === Explicitly add Windows SDK include/lib paths ===
set "WINSDK=C:\Program Files (x86)\Windows Kits\10"
set "WINSDK_VER=10.0.19041.0"
set "MSVC_VER=14.29.30133"
set "MSVC_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\%MSVC_VER%"

set "INCLUDE=%MSVC_ROOT%\include;%WINSDK%\Include\%WINSDK_VER%\ucrt;%WINSDK%\Include\%WINSDK_VER%\shared;%WINSDK%\Include\%WINSDK_VER%\um;%WINSDK%\Include\%WINSDK_VER%\winrt;%INCLUDE%"
set "LIB=%MSVC_ROOT%\lib\x64;%WINSDK%\Lib\%WINSDK_VER%\ucrt\x64;%WINSDK%\Lib\%WINSDK_VER%\um\x64;%LIB%"
set "PATH=%MSVC_ROOT%\bin\HostX64\x64;%WINSDK%\bin\%WINSDK_VER%\x64;%PATH%"

set "CUDA_PATH=C:/ProgramData/anaconda3/Library"
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

set "BUILD_DIR=C:\Users\paper\Desktop\promethorch\build_gguf_cuda"
set "SRC_DIR=C:\Users\paper\Desktop\promethorch"
set "LOG=%SRC_DIR%\build_gguf_cuda_log.txt"

rem === Create build dir if needed ===
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

rem === Run cmake if no Makefile ===
if not exist "Makefile" (
    echo [CMAKE] Configuring with CUDA... >> "%LOG%" 2>&1
    "%CMAKE%" "%SRC_DIR%" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=ON -DPT_USE_CUDNN=OFF -DPT_BUILD_TESTS=OFF "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_PATH%" >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo CMAKE_FAILED >> "%LOG%"
        exit /b 1
    )
)

rem === Build target ===
echo [BUILD] Building test_gguf_inference with CUDA... > "%LOG%" 2>&1
nmake test_gguf_inference >> "%LOG%" 2>&1

echo EXIT_CODE=%ERRORLEVEL% >> "%LOG%"
if exist examples\gguf\test_gguf_inference.exe (
    echo BUILD_SUCCESS >> "%LOG%"
) else (
    echo BUILD_FAILED >> "%LOG%"
)
