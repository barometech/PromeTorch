@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

cd /d C:\Users\paper\Desktop\promethorch\build_final3

echo === Re-running CMake to pick up new gguf target ===
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=OFF -DPT_BUILD_TESTS=OFF

echo === Building test_gguf_inference ===
nmake test_gguf_inference

echo === Done ===
if exist examples\gguf\test_gguf_inference.exe (
    echo SUCCESS: examples\gguf\test_gguf_inference.exe built
) else (
    echo FAILED: test_gguf_inference.exe not found
)
