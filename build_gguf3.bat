@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\Users\paper\Desktop\promethorch\build_final3
nmake test_gguf_inference > C:\Users\paper\Desktop\promethorch\build_gguf_log.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> C:\Users\paper\Desktop\promethorch\build_gguf_log.txt
if exist examples\gguf\test_gguf_inference.exe (
    echo BUILD_SUCCESS >> C:\Users\paper\Desktop\promethorch\build_gguf_log.txt
) else (
    echo BUILD_FAILED >> C:\Users\paper\Desktop\promethorch\build_gguf_log.txt
)
