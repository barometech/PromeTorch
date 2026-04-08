# Changelog

## [0.1.0] - 2026-04-08

### Core
- c10: Allocator (arena + caching), Device, Storage, TensorImpl, ScalarType
- ATen: 149 tensor operations (math, reduce, linear algebra, shape, index)
- Autograd: Engine, 107 backward functions, custom autograd functions
- TUDA: CPU architecture dispatch (AVX2, E2K, NEON, Scalar)

### NN Modules
- 50+ modules: Linear, Conv1d/2d/3d, BatchNorm, LayerNorm, RMSNorm, GroupNorm
- Activations: ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, 15+ more
- Transformer: Encoder, Decoder, MultiheadAttention, PositionalEncoding
- PIR Architecture: Parallel Infinite Retention (O(T) memory, no attention)
- RNN/LSTM/GRU with multi-layer support
- 12 loss functions

### Training
- Optimizers: SGD, Adam, AdamW, RMSprop + fused multi-parameter versions
- 9 LR Schedulers
- Mixed Precision (GradScaler, Autocast)
- Gradient Checkpointing
- Data loading (Dataset, DataLoader, Sampler)

### Backends
- CPU: AVX2 SIMD, cache-tiled GEMM, OpenMP
- CUDA: Custom kernels, cuDNN, FlashAttention, Quantized inference (Q4-Q8)
- Elbrus E2K: EML BLAS, NUMA-aware 4-chip training, VLIW fused ops
- NMCard Mini: Q16.16 fixed-point emulator + hardware driver

### Infrastructure
- Python bindings (pybind11)
- Serialization (PTOR format)
- PromeServe: Ollama-compatible LLM inference server
- Docker support (Ubuntu, Astra, ALT, Elbrus)
- 18 C++ test suites

### Performance
- 4-NUMA data-parallel training: 936 tok/s on Elbrus 32-core
- CPU SIMD: 46x→1.75x vs PyTorch (weighted benchmark)
- CUDA quantized inference: 49.9 tok/s (qwen3:4b)
