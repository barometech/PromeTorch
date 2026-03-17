@echo off
set NMC_GCC_TOOLPATH=C:\Program Files\Module\NMC-SDK
set IDE=%NMC_GCC_TOOLPATH%\nmc4-ide
set CC1PLUS=%IDE%\libexec\gcc\nmc\4.8.3\cc1plus.exe
set AS_GNU=%IDE%\bin\nmc-as-gnu.exe
set LD=%IDE%\bin\nmc-ld.exe

call "%NMC_GCC_TOOLPATH%\nmc4cmd.bat" "%~dp0"

echo =============================================
echo Building with GAS format for NMC4
echo Using custom startup (nmc_startup.s)
echo =============================================
echo.

:: First assemble the startup code
echo Building startup code...
"%AS_GNU%" -o nmc_startup.o nmc_startup.s 2>nul
if not exist nmc_startup.o (
    echo FATAL: Cannot assemble nmc_startup.s
    exit /b 1
)
echo   [OK] nmc_startup.o
echo.

set SUCCESS=0
set FAILED=0

goto :main

:compile_nmpp
set NAME=%~1
set NM_CARD_INC=C:\Program Files\Module\NM_Card\libload\include
set NM_CARD_LIB=C:\Program Files\Module\NM_Card\libload\lib
echo Building %NAME%.abs (with nmpp vectorized BLAS)...
del %NAME%_gas.s %NAME%_gas.o %NAME%.abs 2>nul

"%CC1PLUS%" -quiet -I"%NM_CARD_INC%" %NAME%.cpp -mnmc4 -mgas -O2 -o %NAME%_gas.s 2>nul
if not exist %NAME%_gas.s (
    echo   [FAIL] cc1plus
    set /a FAILED+=1
    goto :eof
)

"%AS_GNU%" -o %NAME%_gas.o %NAME%_gas.s 2>nul
if not exist %NAME%_gas.o (
    echo   [FAIL] nmc-as-gnu
    set /a FAILED+=1
    goto :eof
)

"%LD%" -o %NAME%.abs nmc_startup.o %NAME%_gas.o -L"%NM_CARD_LIB%" -lnmpps-nmc4 -T nm6408brd.lds -e start 2>nul
if exist %NAME%.abs (
    echo   [OK] VECTORIZED
    set /a SUCCESS+=1
    del %NAME%_gas.s %NAME%_gas.o 2>nul
) else (
    echo   [FAIL] linker (nmpp)
    set /a FAILED+=1
)
goto :eof

:compile_mc
set NAME=%~1
set NM_CARD_INC=C:\Program Files\Module\NM_Card\libload\include
set NM_CARD_LIB=C:\Program Files\Module\NM_Card\libload\lib
echo Building %NAME%.abs (multi-core, with nm6408load_nmc)...
del %NAME%_gas.s %NAME%_gas.o %NAME%.abs 2>nul

"%CC1PLUS%" -quiet -I"%NM_CARD_INC%" %NAME%.cpp -mnmc4 -mgas -O2 -o %NAME%_gas.s 2>nul
if not exist %NAME%_gas.s (
    echo   [FAIL] cc1plus
    set /a FAILED+=1
    goto :eof
)

"%AS_GNU%" -o %NAME%_gas.o %NAME%_gas.s 2>nul
if not exist %NAME%_gas.o (
    echo   [FAIL] nmc-as-gnu
    set /a FAILED+=1
    goto :eof
)

"%LD%" -o %NAME%.abs nmc_startup.o %NAME%_gas.o -L"%NM_CARD_LIB%" -lnm6408load_nmc -T nm6408brd.lds -e start 2>nul
if exist %NAME%.abs (
    echo   [OK]
    set /a SUCCESS+=1
    del %NAME%_gas.s %NAME%_gas.o 2>nul
) else (
    echo   [FAIL] linker
    set /a FAILED+=1
)
goto :eof

:compile
set NAME=%~1
echo Building %NAME%.abs...
del %NAME%_gas.s %NAME%_gas.o %NAME%.abs 2>nul

:: Step 1: C++ to GAS assembly with -mgas -mnmc4
"%CC1PLUS%" -quiet %NAME%.cpp -mnmc4 -mgas -O2 -o %NAME%_gas.s 2>nul
if not exist %NAME%_gas.s (
    echo   [FAIL] cc1plus
    set /a FAILED+=1
    goto :eof
)

:: Step 2: GNU Assembler
"%AS_GNU%" -o %NAME%_gas.o %NAME%_gas.s 2>nul
if not exist %NAME%_gas.o (
    echo   [FAIL] nmc-as-gnu
    set /a FAILED+=1
    goto :eof
)

:: Step 3: Link with custom startup (no problematic library!)
"%LD%" -o %NAME%.abs nmc_startup.o %NAME%_gas.o -T nm6408brd.lds -e start 2>nul
if exist %NAME%.abs (
    echo   [OK]
    set /a SUCCESS+=1
    del %NAME%_gas.s %NAME%_gas.o 2>nul
) else (
    echo   [FAIL] linker
    set /a FAILED+=1
)
goto :eof

:main
:: Main kernels (forward)
call :compile matmul_custom
call :compile matmul_quant
call :compile rmsnorm
call :compile silu
call :compile softmax
call :compile rope
call :compile attention
call :compile elementwise
call :compile layernorm
call :compile dispatcher

:: Backward kernels (for training)
call :compile matmul_backward
call :compile silu_backward
call :compile gelu_backward
call :compile softmax_backward
call :compile rmsnorm_backward
call :compile rope_backward
call :compile attention_backward

:: Optimizers
call :compile sgd_update
call :compile adam_update

:: Loss functions
call :compile cross_entropy

:: Multi-core dispatcher (needs nm6408load_nmc library)
call :compile_mc dispatcher_mc

:: VECTORIZED float dispatchers (link with nmpp BLAS for vector pipeline)
call :compile_nmpp dispatcher_float_vec
call :compile_nmpp dispatcher_nmpp

:: Test files
call :compile simple_test
call :compile echo_test
call :compile add_test
call :compile mul_test
call :compile mymul_test
call :compile matmul_int
call :compile matmul

echo.
echo =============================================
echo Results: %SUCCESS% success, %FAILED% failed
echo =============================================
echo.
dir /b *.abs 2>nul
