# Сборка на Windows

PromeTorch — cross-platform проект. Primary dev/test path — Linux.
Windows тоже поддерживается, но требует работы из Developer Command
Prompt (или явного `vcvarsall.bat`).

## Требования

- MSVC 2019 Build Tools или Visual Studio 2019+
- CMake 3.15+ (рекомендуется бандл из Visual Studio — НЕ anaconda)
- Для CUDA: NVIDIA CUDA Toolkit **12.4+** (standalone installer)
  - Anaconda-packaged CUDA 12.9 **не подходит** — в нём отсутствует
    `nv/target` header, требуемый `cuda_fp16.h`.

## CPU build

```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

cd /d C:\path\to\promethorch
mkdir build && cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
nmake
```

**Важно:** из Git Bash / PowerShell / WSL сборка не пройдёт — `rc.exe`
(Windows Resource Compiler) не находится без `vcvarsall.bat`.

## CUDA build

Используйте готовый reference script `scripts/build-cuda-windows.bat`:

```batch
scripts\build-cuda-windows.bat train_10_models
scripts\build-cuda-windows.bat test_gguf_inference
scripts\build-cuda-windows.bat promeserve
```

Скрипт:
- подключает `vcvarsall.bat x64`
- выставляет Windows SDK пути
- указывает `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` (standalone)
- passes quoted CUDA paths с forward slashes в CMake (escape-safe)
- генерирует в `build_cuda124/`

Incremental rebuild — просто перезапустите скрипт с нужным target.

## Частые проблемы

**`cuda_fp16.h(4492): fatal error C1083: nv/target: No such file or directory`**

→ Вы используете anaconda CUDA. Установите standalone NVIDIA CUDA
Toolkit 12.4+ с сайта NVIDIA и выставите `CUDA_PATH` на его путь.
Скрипт `build-cuda-windows.bat` это делает автоматически.

**`rc.exe: command not found`**

→ Сборка запущена не из Developer Command Prompt. Либо откройте
его через меню Пуск, либо сделайте `call vcvarsall.bat x64` в начале
вашей batch-сессии.

**cuDNN RNN compile errors**

→ cuDNN 9 убрал legacy RNN API (`cudnnSetRNNDescriptor_v6`). В коде
добавлен guard `CUDNN_VERSION < 9000` — на cuDNN 9 RNN путь
автоматически no-op'ится, LSTM/GRU работают через pure-C++ forward.
Если нужна cuDNN-accelerated RNN, используйте cuDNN 8.

## Проверка

```batch
build\examples\mnist\train_mnist_mlp.exe --device cpu --data data\mnist --epochs 1
```

Ожидаемо: MNIST MLP ~92-97 % test accuracy (в зависимости от arch).

Для CUDA:
```batch
build_cuda124\examples\mnist\train_10_models.exe --device cuda --data data\mnist
```
