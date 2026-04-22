# Third-Party Notices

PromeTorch is an independent C++17/CUDA implementation, but it consumes and
produces files that follow the on-disk formats and packing conventions of
several third-party projects. This document records those projects, their
licenses, and the scope of use within this codebase.

No third-party source code is vendored in this repository — only format
constants, bit layouts and a handful of well-documented helper conventions are
reproduced so that PromeTorch can read / write binary-compatible weights.

---

## 1. GGML / llama.cpp (MIT License)

- **Upstream:** https://github.com/ggerganov/ggml, https://github.com/ggerganov/llama.cpp
- **License:** MIT (Copyright (c) 2023-2024 Georgi Gerganov)
- **Scope of use in PromeTorch:**
  - GGUF container format (magic bytes, metadata KV layout, tensor descriptor
    table, 32-byte alignment rule).
  - Quantization super-block layouts (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`,
    including `qs`/`qh`/`scales`/`mins` / `d`, `dmin` packing).
  - `GGML_FP16 <-> FP32` conversion convention.
- **Files touching these conventions:**
  - `torch/io/gguf_dequant.h` — type enums & CPU dequantizers
  - `torch/io/gguf_loader.h` — GGUF file parser / mmap backend
  - `torch/io/cpu_quant_gemv.h` — AVX2 fused dequant+GEMV
  - `aten/src/ATen/cuda/CUDAQuantGemv.cu` — CUDA fused dequant+GEMV
- **What is *not* taken from llama.cpp:** `ggml-cuda.cu`, `ggml-quants.c`,
  any source files or code; all kernels here are independent implementations.

```
MIT License

Copyright (c) 2023-2024 Georgi Gerganov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. PyTorch (BSD-3-Clause)

- **Upstream:** https://github.com/pytorch/pytorch
- **License:** BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
- **Scope of use in PromeTorch:**
  - Public API shape and naming (`torch::Tensor`, `torch::nn::Linear`,
    `torch::optim::Adam`, `torch.save` / `torch.load` container tags) so that
    user code and `.pt` checkpoints can be moved across with minimal changes.
  - No PyTorch source code is vendored. Implementations are independent.
- `.pt` / pickle I/O conventions reproduced in `torch/serialization.h` follow
  the publicly documented container layout.

---

## 3. Ollama REST API (MIT License)

- **Upstream:** https://github.com/ollama/ollama
- **License:** MIT (Copyright (c) Ollama)
- **Scope of use in PromeTorch:**
  - HTTP endpoints (`/api/tags`, `/api/generate`, `/api/chat`, `/api/pull`,
    `/api/show`, `/api/ps`) and NDJSON streaming body shape in `promeserve/`
    so that existing Ollama-compatible clients can talk to PromeServe.
  - No Ollama source code is vendored.

---

## 4. GGUF Test Weights (Qwen / Gemma / DeepSeek / etc.)

Model weights (`qwen3-4b-Q4_K_M.gguf` etc.) that appear in benchmark scripts
are **not** redistributed in this repository. They must be downloaded from
the respective model authors' channels (Hugging Face / Ollama registry) and
are governed by their own licenses (Qwen: Tongyi Qianwen, Gemma: Gemma TOS,
DeepSeek: DeepSeek License, etc.). See the models' own license files.

---

## Reporting

If you believe content in this repository reproduces a third-party project in
a way that is not disclosed here, please open an issue at
https://github.com/barometech/PromeTorch/issues so we can add the notice.
