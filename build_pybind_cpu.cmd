@echo off
echo BUILD_START > C:\Users\USER\Desktop\promethorch\build_pybind_log.txt
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >> C:\Users\USER\Desktop\promethorch\build_pybind_log.txt 2>&1
set PATH=C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Library\bin;%PATH%
cd /d C:\Users\USER\Desktop\promethorch\build_pybind
del /q CMakeFiles\_C.dir\python\csrc\*.obj >> C:\Users\USER\Desktop\promethorch\build_pybind_log.txt 2>&1
nmake _C >> C:\Users\USER\Desktop\promethorch\build_pybind_log.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> C:\Users\USER\Desktop\promethorch\build_pybind_log.txt
