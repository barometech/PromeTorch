# PromeTorch Build Guide

## ВАЖНО: Читай это перед любой сборкой!

---

## КРИТИЧНО: Проблема со сборкой через bash/cygwin

На Windows есть проблема: при запуске через bash/cygwin окружение Visual Studio не передаётся правильно, и rc.exe (Resource Compiler) не находится.

### РЕШЕНИЕ: Запускать сборку из НАСТОЯЩЕГО Developer Command Prompt

1. Открыть **Start Menu**
2. Найти **"Developer Command Prompt for VS 2019"** или **"x64 Native Tools Command Prompt for VS 2019"**
3. В этом терминале выполнить команды сборки

### Команды для копирования в Developer Command Prompt:

```cmd
cd C:\Users\paper\Desktop\promethorch

:: Добавить CUDA
set PATH=%PATH%;C:\ProgramData\anaconda3\Library\bin
set CUDA_PATH=C:\ProgramData\anaconda3\Library

:: Чистая сборка
if exist build_cuda rmdir /s /q build_cuda
mkdir build_cuda
cd build_cuda

:: CMake (использовать VS CMake!)
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=ON -DPT_USE_CUDNN=ON -DPT_BUILD_TESTS=OFF -DPT_BUILD_PYTHON=OFF -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_PATH%"

:: Build
nmake c10
nmake aten_cuda
nmake train_pir
nmake train_mnist_mlp
```

---

## ОКРУЖЕНИЕ

- **OS:** Windows 10/11
- **Компилятор:** Visual Studio 2019 Build Tools
- **CUDA:** 12.9 (Anaconda: `C:\ProgramData\anaconda3\Library`)
- **cuDNN:** 9.14 (Anaconda)
- **CMake:** 3.20 (VS2019 встроенный!)

### КРИТИЧНО: Какой CMake использовать

**НЕ ИСПОЛЬЗУЙ** CMake из Anaconda (`C:\ProgramData\anaconda3\Lib\site-packages\cmake`) - он вызывает проблемы с Windows SDK!

**ИСПОЛЬЗУЙ** CMake из Visual Studio:
```
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
```

---

## CUDA СБОРКА (РАБОЧАЯ КОНФИГУРАЦИЯ)

### Шаг 1: Создать batch файл `build_cuda.bat`

```batch
@echo off
:: Настройка VS окружения
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Используем VS CMake!
set CMAKE_EXE="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

:: Добавляем CUDA в PATH
set PATH=C:\ProgramData\anaconda3\Library\bin;C:\ProgramData\anaconda3\Scripts;%PATH%
set CUDA_PATH=C:\ProgramData\anaconda3\Library

cd /d C:\Users\paper\Desktop\promethorch

:: Создаём/очищаем build директорию
if exist build_cuda rmdir /s /q build_cuda
mkdir build_cuda
cd build_cuda

:: CMake configure
%CMAKE_EXE% .. -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPT_USE_CUDA=ON ^
    -DPT_USE_CUDNN=ON ^
    -DPT_BUILD_TESTS=OFF ^
    -DPT_BUILD_PYTHON=OFF ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" ^
    -DCUDAToolkit_ROOT="%CUDA_PATH%"

:: Build
nmake c10
nmake aten_cuda
nmake train_mnist_mlp

echo Build complete!
```

### Шаг 2: Запустить из CMD (не из bash!)

Открыть **cmd.exe** и выполнить:
```cmd
cd C:\Users\paper\Desktop\promethorch
build_cuda.bat
```

### Шаг 3: Проверить результат

```
build_cuda/
├── c10.dll              # Core library
├── aten_cuda.lib        # CUDA operations
└── examples/mnist/
    └── train_mnist_mlp.exe
```

---

## ЗАПУСК ТРЕНИРОВКИ НА GPU

### Batch файл `run_mnist_cuda.bat`

```batch
@echo off
set PATH=C:\Users\paper\Desktop\promethorch\build_cuda;C:\ProgramData\anaconda3\Library\bin;%PATH%
cd /d C:\Users\paper\Desktop\promethorch\build_cuda\examples\mnist
train_mnist_mlp.exe --data C:\Users\paper\Desktop\promethorch\data\mnist --device cuda --epochs 5 --batch_size 64
```

---

## ИЗВЕСТНЫЕ ПРОБЛЕМЫ

### 1. CMake: "rc.exe not found" или "corecrt.h not found"

**Причина:** Используется CMake из Anaconda который не видит Windows SDK.

**Решение:** Использовать CMake из Visual Studio (см. выше).

### 2. Exit code -1073740791 (heap corruption)

**Причина:** DLL singleton проблема с CUDACachingAllocator.

**Решение:** Убедиться что:
1. `c10/cuda/CUDAAllocator.cpp` содержит singleton implementation
2. `aten_cuda` собирается как SHARED (не STATIC)
3. ВСЕ файлы пересобраны после исправления

### 3. nvcc: "A single input file is required"

**Причина:** MSVC флаги попадают в nvcc.

**Решение:** В CMakeLists.txt использовать generator expressions:
```cmake
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
```

---

## СУЩЕСТВУЮЩИЕ СБОРКИ

| Директория | Дата | Статус |
|------------|------|--------|
| build_cudnn | 22-23 янв | c10.dll новый, aten_cuda старый |
| build_cuda_examples | 23-24 янв | c10.dll новый, MLP старый |
| build_cpu | - | CPU only, работает |

**Рекомендация:** Делать чистую сборку `build_cuda` по гайду выше.

---

## ДАННЫЕ MNIST

Путь: `C:\Users\paper\Desktop\promethorch\data\mnist\`

Файлы:
- train-images-idx3-ubyte (60000 images)
- train-labels-idx1-ubyte
- t10k-images-idx3-ubyte (10000 images)
- t10k-labels-idx1-ubyte

---

## БЫСТРЫЙ СТАРТ

```cmd
:: 1. Открыть cmd.exe (НЕ bash!)
:: 2. Запустить сборку
cd C:\Users\paper\Desktop\promethorch
build_cuda.bat

:: 3. Запустить тренировку
run_mnist_cuda.bat
```
