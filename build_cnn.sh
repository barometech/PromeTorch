#!/bin/bash
MSVC_VER=14.29.30133
MSVC_BASE="C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/$MSVC_VER"
MSVC_BIN="$MSVC_BASE/bin/Hostx64/x64"
WINSDK_BASE="C:/Program Files (x86)/Windows Kits/10"
WINSDK_VER=10.0.19041.0
WINSDK_BIN="$WINSDK_BASE/bin/$WINSDK_VER/x64"

export INCLUDE="$MSVC_BASE/include;$WINSDK_BASE/Include/$WINSDK_VER/ucrt;$WINSDK_BASE/Include/$WINSDK_VER/shared;$WINSDK_BASE/Include/$WINSDK_VER/um;$WINSDK_BASE/Include/$WINSDK_VER/winrt"
export LIB="$MSVC_BASE/lib/x64;$WINSDK_BASE/Lib/$WINSDK_VER/ucrt/x64;$WINSDK_BASE/Lib/$WINSDK_VER/um/x64"
export PATH="$MSVC_BIN:$WINSDK_BIN:$PATH"

CMAKE_EXE="C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"

cd C:/Users/paper/Desktop/promethorch/build_final3

echo "=== Reconfiguring CMake ==="
"$CMAKE_EXE" .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=OFF -DPT_BUILD_TESTS=OFF 2>&1

echo "=== Building train_mnist_cnn_autograd ==="
nmake.exe train_mnist_cnn_autograd 2>&1

echo "=== Done ==="
