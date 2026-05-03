## Summary

<!-- 1-3 bullets describing what this PR changes and why. -->

## Test plan

<!-- How did you verify this change? Local builds, benchmarks, regression checks. -->

- [ ] Built locally (specify backend: CPU/CUDA/Эльбрус/NM Card)
- [ ] Existing tests pass (`build_*/test_*`)
- [ ] No regression on the 11.4 tok/s qwen3-4B Q4_K_M TP-4 baseline (Эльбрус)
- [ ] No regression on MNIST/LSTM/GRU end-to-end accuracy (CPU CLI)

## Risk / scope

<!--
Touches inference hot path? Quantization layout? GGUF loader? CMake?
Any breaking changes to existing public APIs?
-->

## Linked issue / task

<!-- e.g. #59, JOURNAL.md anchor, INFRASTRUCTURE_AUDIT.md bug ID. -->
