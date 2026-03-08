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

cd /d C:\Users\paper\Desktop\promethorch\build_final3
nmake test_gguf_inference > C:\Users\paper\Desktop\promethorch\build_gguf_log.txt 2>&1

echo EXIT_CODE=%ERRORLEVEL% >> C:\Users\paper\Desktop\promethorch\build_gguf_log.txt
if exist examples\gguf\test_gguf_inference.exe (
    echo BUILD_SUCCESS >> C:\Users\paper\Desktop\promethorch\build_gguf_log.txt
) else (
    echo BUILD_FAILED >> C:\Users\paper\Desktop\promethorch\build_gguf_log.txt
)
