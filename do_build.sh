#!/bin/bash
MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64"
WINSDK_BIN="/c/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64"

export INCLUDE="C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/include;C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/ucrt;C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/shared;C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/um;C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/winrt"
export LIB="C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/lib/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/ucrt/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/um/x64"
export PATH="$MSVC_BIN:$WINSDK_BIN:/c/ProgramData/anaconda3:/c/ProgramData/anaconda3/Library/bin:$PATH"

cd "$(dirname "$0")/build_pybind"
rm -f CMakeFiles/_C.dir/python/csrc/*.obj 2>/dev/null
nmake.exe _C 2>&1
