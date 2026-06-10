# dev_archive — архив dev-артефактов

Сюда переносятся файлы, которые исторически жили в корне репозитория, но
не являются частью сборки/публичного API. Хранятся для истории, не
подключены к CMake.

## scratch_tests/

8 одноразовых отладочных тестов из корня (orphan, 0 CMake refs, audit
2026-06-02 `_dead_code`): `test_adam_minimal`, `test_backward_debug`,
`test_batch_by_batch`, `test_cuda_init`, `test_fp16_kernels`,
`test_gemm_native`, `test_gradient_direction`, `test_lstm_grads`.
Использовались при отладке autograd/CUDA в ранних фазах. Канонические
тесты — в `test/cpp/` и `tests/`.
