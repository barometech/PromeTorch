@echo off
set NMC_GCC_TOOLPATH=C:\Program Files\Module\NMC-SDK
set IDE=%NMC_GCC_TOOLPATH%\nmc4-ide
set CC1PLUS=%IDE%\libexec\gcc\nmc\4.8.3\cc1plus.exe
set AS_GNU=%IDE%\bin\nmc-as-gnu.exe
set LD=%IDE%\bin\nmc-ld.exe
set LIBGCC_FLOAT=%IDE%\lib\gcc\nmc\4.8.3\nmc4-float\libgcc.a

echo Building dispatcher_suda.abs (SUDA generated)...
"%CC1PLUS%" -quiet dispatcher_suda.cpp -mnmc4 -mgas -O2 -o dispatcher_suda_gas.s
"%AS_GNU%" -o dispatcher_suda_gas.o dispatcher_suda_gas.s
"%LD%" -o dispatcher_suda.abs nmc_startup.o dispatcher_suda_gas.o MullMatrix_f.o TmpBuffers64G.o "%LIBGCC_FLOAT%" -T nm6408brd.lds -e start
if exist dispatcher_suda.abs (
    echo [OK] dispatcher_suda.abs
    del dispatcher_suda_gas.s dispatcher_suda_gas.o 2>nul
) else (
    echo [FAIL] Build failed
)
