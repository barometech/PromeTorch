@echo off
REM =============================================================================
REM PromeTorch CUDA build reference script (Windows, MSVC 2019, NVIDIA CUDA 12.4)
REM =============================================================================
REM Known working on Windows 10 + MSVC 2019 BuildTools + NVIDIA CUDA Toolkit 12.4
REM (standalone install, NOT the anaconda-packaged CUDA which is missing
REM  `nv/target` header required by cuda_fp16.h).
REM
REM Usage: call this from any Windows command line (cmd.exe), NOT from bash.
REM   scripts\build-cuda-windows.bat [target]
REM
REM Examples:
REM   scripts\build-cuda-windows.bat                    REM all
REM   scripts\build-cuda-windows.bat train_10_models
REM   scripts\build-cuda-windows.bat test_gguf_inference
REM   scripts\build-cuda-windows.bat promeserve
REM
REM Output goes to build_cuda124/. Rerun freely — incremental rebuilds.
REM =============================================================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set "WindowsSdkDir=C:\Program Files (x86)\Windows Kits\10"
set "WindowsSDKVersion=10.0.19041.0"
set INCLUDE=%INCLUDE%;%WindowsSdkDir%\Include\%WindowsSDKVersion%\ucrt;%WindowsSdkDir%\Include\%WindowsSDKVersion%\shared;%WindowsSdkDir%\Include\%WindowsSDKVersion%\um
set LIB=%LIB%;%WindowsSdkDir%\Lib\%WindowsSDKVersion%\ucrt\x64;%WindowsSdkDir%\Lib\%WindowsSDKVersion%\um\x64

REM NVIDIA CUDA Toolkit 12.4 (adjust path if you have a different version)
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_PATH_FWD=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
set "PATH=%CUDA_PATH%\bin;%WindowsSdkDir%\bin\%WindowsSDKVersion%\x64;%PATH%"
set "INCLUDE=%CUDA_PATH%\include;%INCLUDE%"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"

REM cuDNN from anaconda (9.x). cuDNN RNN legacy API is guarded for cuDNN 9
REM (see aten/src/ATen/cudnn/CuDNNRNN.h). cuDNN 8 works too.
set "PATH=%PATH%;C:\ProgramData\anaconda3\Library\bin"
set "INCLUDE=%INCLUDE%;C:\ProgramData\anaconda3\Library\include"
set "LIB=%LIB%;C:\ProgramData\anaconda3\Library\lib"

if not exist "build_cuda124" mkdir "build_cuda124"
cd build_cuda124

echo === CMake configure (CUDA 12.4) ===
if not exist "CMakeCache.txt" (
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" .. ^
        -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
        -DPT_USE_CUDA=ON -DPT_USE_CUDNN=OFF -DPT_BUILD_TESTS=OFF ^
        "-DCMAKE_CUDA_COMPILER=%CUDA_PATH_FWD%/bin/nvcc.exe" ^
        "-DCUDAToolkit_ROOT=%CUDA_PATH_FWD%" 2>&1
    if errorlevel 1 (
        echo CMake configure failed.
        cd ..
        exit /b 1
    )
)

echo.
echo === Building target: %1 (empty = all) ===
echo Start: %date% %time%
nmake %* 2>&1
set RC=%errorlevel%
echo End: %date% %time%
echo Build result: %RC%

cd ..
exit /b %RC%
