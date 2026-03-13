# PromeTorch

A complete deep learning framework built from scratch in C++17/CUDA. Implements the PyTorch API with custom tensor operations, automatic differentiation, neural network modules, and GPU acceleration.

**~45,000 lines of C++/CUDA** | **110+ tensor operations** | **90+ autograd backward functions** | **57+ NN modules**

## Features

### Core Tensor Library
- N-dimensional tensors with broadcasting, strides, views, and memory formats
- 110+ operations: math, reductions, linear algebra (LU, QR, SVD, Cholesky), shape manipulation, advanced indexing
- Factory functions: zeros, ones, rand, randn, arange, linspace, eye, multinomial
- Type system: Float, Double, Half, Int, Long, Bool, BFloat16
- Channels-last memory format (NHWC)

### Automatic Differentiation
- Reverse-mode autograd with dynamic computational graph
- 90+ backward functions covering all differentiable operations
- Custom autograd functions (CRTP pattern)
- Gradient checkpointing for memory-efficient training
- Hook system (forward pre-hooks, forward hooks, backward hooks)

### Neural Network Modules (57+)
- **Layers**: Linear, Bilinear, LazyLinear, Conv1d/2d/3d, ConvTranspose2d
- **Activations**: ReLU, GELU, SiLU, Mish, Sigmoid, Tanh, ELU, SELU, Softmax, and 10+ more
- **Normalization**: BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, RMSNorm
- **Pooling**: MaxPool, AvgPool, AdaptiveAvgPool, GlobalAvgPool
- **Recurrent**: RNN, LSTM, GRU (multi-layer, bidirectional support)
- **Transformer**: MultiheadAttention, TransformerEncoder/Decoder, PositionalEncoding
- **Loss**: CrossEntropy, MSE, L1, BCE, NLL, KLDiv, CTC, Focal, Dice, and 10+ more
- **Containers**: Sequential, ModuleList, ModuleDict
- **Embedding**: Embedding, EmbeddingBag

### Optimizers & Schedulers
- **Optimizers**: SGD (with momentum), Adam, AdamW, RMSprop
- **LR Schedulers**: StepLR, CosineAnnealing, OneCycleLR, WarmupLR, and 9 more

### CUDA Backend
- Custom CUDA kernels: GEMM, GEMV, reductions, element-wise ops, softmax
- cuDNN integration: convolution, pooling, batch normalization, activations
- FlashAttention (O(N) memory, causal masking)
- Mixed precision training (AMP): GradScaler + Autocast
- Quantized GEMV kernels (Q4_K, Q5_K, Q6_K) with warp-cooperative coalesced access

### GGUF Inference Engine
- Load and run Ollama/llama.cpp models directly (GGUF format)
- Supported architectures: Qwen3, Gemma3, DeepSeek-R1, Llama
- Chat template support with special token handling
- GPU inference: ~55 tok/s on A100 (qwen3:4b)
- Quantization formats: Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32

### Additional Features
- Serialization (save/load tensors and state dicts)
- Python bindings via pybind11
- FFT operations (fft, ifft, rfft, fft2)
- Einsum with optimized paths

## Building

### Prerequisites
- C++17 compiler (MSVC 2019+ or GCC 9+)
- CMake 3.18+
- (Optional) CUDA Toolkit 12.x for GPU support
- (Optional) cuDNN 9.x for accelerated convolutions

### CPU Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CUDA Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=ON -DPT_USE_CUDNN=ON
make -j$(nproc)
```

### Windows (MSVC)
```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
mkdir build && cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=ON
nmake
```

## Examples

### MNIST Training
```cpp
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"

// Define model
Sequential model;
model.add(std::make_shared<Linear>(784, 256));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(256, 10));

// Train
auto optimizer = torch::optim::Adam(model.parameters(), /*lr=*/0.001);
for (auto& [data, target] : dataloader) {
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    torch::autograd::backward(loss);
    optimizer.step();
}
```

### GGUF Inference
```cpp
#include "torch/io/gguf_model.h"

GGUFModel model;
model.load_from_ollama("qwen3:4b", /*use_cuda=*/true);
std::string response = model.chat("What is 2+2?", /*max_tokens=*/64);
```

## Benchmarks

| Task | PromeTorch | PyTorch |
|------|-----------|---------|
| MNIST MLP (accuracy) | 97.65% | 97.8% |
| LSTM Shakespeare | 98.44% | ~98% |
| GRU Classification | 95.3% | ~95% |
| GGUF qwen3:4b (A100) | 55 tok/s | N/A |
| GGUF deepseek-r1:8b (A100) | 30 tok/s | N/A |

## Architecture

```
c10/                    Core: Allocator, Device, Storage, TensorImpl, ScalarType
aten/src/ATen/
  core/                 Tensor, TensorFactory, TensorOptions
  native/cpu/           MathOps, ReduceOps, LinearAlgebra, ShapeOps, IndexOps, FFTOps
  cuda/                 CUDAKernels, CUDABlas, CUDAReduce, FlashAttention, CUDAQuantGemv
  cudnn/                CuDNN wrappers (Conv, Pool, BatchNorm, Activation)
torch/
  csrc/autograd/        Engine, Node, Edge, 90+ backward functions
  nn/modules/           57+ NN module implementations
  optim/                Optimizers and LR schedulers
  io/                   GGUF model loading and inference
  amp/                  Mixed precision (GradScaler, Autocast)
  data/                 Dataset, DataLoader, Samplers
test/cpp/               Google Test suite (16 test files, 300+ tests)
examples/               MNIST, PIR, RNN, Transformer, ViT, GGUF inference
python/                 pybind11 bindings
```

## Testing

```bash
cd build
cmake .. -DPT_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

## Future Work

- Distributed training (DDP, NCCL)
- JIT compilation / TorchScript
- ONNX export
- Sparse tensor support
- Complex number dtype

## License

MIT License. See [LICENSE](LICENSE) for details.
