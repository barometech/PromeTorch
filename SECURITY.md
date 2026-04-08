# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please **DO NOT** create public GitHub Issues for security vulnerabilities.

Instead, report security issues to: **security@prometorch.ru**

We will respond within 48 hours and provide a fix timeline.

## Known Security Considerations

- **Buffer overflow protection**: Allocator has alignment overflow checks (Allocator.h)
- **Integer overflow protection**: numel overflow checks in TensorImpl.h
- **Type safety**: copy_() validates dtype match
- **CUDA**: No `cudaFree` on shutdown (prevents double-free, matches PyTorch pattern)
