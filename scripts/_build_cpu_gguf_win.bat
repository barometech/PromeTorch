@echo off
REM Build PromeTorch CPU-only on Windows for GGUF inference
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
cd /d C:\Users\paper\Desktop\promethorch
if not exist build_cpu_gguf mkdir build_cpu_gguf
cd build_cpu_gguf
"%CMAKE%" .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=OFF -DPT_USE_CUDNN=OFF -DPT_BUILD_TESTS=OFF -DPT_BUILD_PYTHON=OFF
echo === CMAKE EXIT: %ERRORLEVEL% ===
nmake test_gguf_inference
echo === BUILD EXIT: %ERRORLEVEL% ===
