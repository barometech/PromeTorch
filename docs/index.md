# PromeTorch

**Deep learning framework, написанный с нуля на C++17/CUDA**, с нативной
поддержкой российских процессоров (Эльбрус E2K, Baikal aarch64) и
ускорителей (NM Card / NM Quad DSP), а также x86/AVX2 и NVIDIA CUDA.

> Числа проекта (backward functions, CUDA kernels, optimizers, tests,
> строки кода) — в [STATS.md](https://github.com/barometech/PromeTorch/blob/main/STATS.md),
> генерируется из исходников (`scripts/gen_stats.py`), single source of truth.

## Быстрый старт

=== "pip (CPU)"

    ```bash
    pip install prometorch
    python -c "import prometorch as pt; print(pt.tensor([1.0,2.0,3.0]))"
    ```

=== "Из исходников (presets)"

    ```bash
    cmake --preset linux-x86_64
    cmake --build --preset linux-x86_64
    ```

=== "Эльбрус (E2K)"

    ```bash
    ./scripts/build-elbrus.sh        # auto-detect ISA v3/v4/v5/v6 + BLAS
    ./scripts/run_tp_elbrus.sh "prompt"
    ```

Полное руководство по сборке — [BUILD.md](BUILD.md).

## Ключевое

- **Эльбрус LLM inference** — qwen3-4B TP-4 на 8C2 = 10.9 tok/s; 10 GGUF
  моделей работают. Multi-arch: v3/v4/v5/v6 (auto-detect ISA + BLAS
  EML→OpenBLAS→TUDA). См. [PERFORMANCE_BY_ISA.md](elbrus_isa/PERFORMANCE_BY_ISA.md).
- **pre-SINQ GGUF** — drop-in поддержка Huawei Sinkhorn-квантизации
  (Apache-2.0), на 8C2 = 11.5 tok/s. См. [SINQ_INTEGRATION.md](SINQ_INTEGRATION.md).
- **PromeServe** — Ollama-совместимый LLM сервер с tool-calling + MCP.
  См. [MCP_INTEGRATION.md](MCP_INTEGRATION.md).
- **Полный PyTorch-подобный API** — Tensor, autograd, nn.Module, 16
  optimizers, CPU SIMD (AVX2/NEON/E2K) + CUDA.

## Поддерживаемые платформы

| Платформа | Сборка | BLAS |
|-----------|--------|------|
| x86_64 Linux/Win/macOS | `cmake --preset` / pip | AVX2 / system BLAS |
| Эльбрус E2K v3–v6 | `build-elbrus.sh` | EML / OpenBLAS / TUDA |
| Baikal-M/S (aarch64) | `cmake --preset baikal-*` | NEON |
| ALT / Astra / RED OS | `cmake --preset {alt,astra,redos}-x86_64` | AVX2 |
| NVIDIA CUDA | `cmake --preset cuda` | cuBLAS / cuDNN |
| NM Card / NM Quad (DSP) | отдельный SDK | — |

## Лицензия

PromeTorch License — модифицированная BSD 3-Clause + два условия (атрибуция в
коммерческих продуктах и запрет перепродажи самого фреймворка). См.
[LICENSE](https://github.com/barometech/PromeTorch/blob/main/LICENSE)
и [NOTICE](https://github.com/barometech/PromeTorch/blob/main/NOTICE)
(сторонние: llama.cpp/ggml MIT, SINQ Apache-2.0, PyTorch API-вдохновение).
Торговая марка — [TRADEMARKS.md](https://github.com/barometech/PromeTorch/blob/main/TRADEMARKS.md).
