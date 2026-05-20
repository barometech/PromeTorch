# Аудит ДВА — stub / throw / no-op функции с claimed-DONE статусом

Дата: 2026-05-20
Скоуп: `c10/`, `aten/`, `torch/`, `python/csrc/`, `promeserve/`
Метод: grep на anti-patterns (`PT_ERROR("not implemented")`, `throw std::runtime_error("not implemented")`, `return Tensor()` сразу после declaration, `// stub`, `// TODO`, `not yet implemented`).

Severity:
- **critical** — публичный API + branch reachable + tensors-возвращающий путь, который заявлен в README/CLAUDE.md как DONE
- **major** — публичный API падает на типичном для документации use-case
- **minor** — internal-only / acceptable design / документировано как not-yet

## Таблица находок

| # | Функция | Файл:строка | Тип stub | Claimed где | Severity | Рекомендуемое действие |
|---|---------|-------------|----------|-------------|----------|------------------------|
| 1 | `at::scaled_dot_product_attention` CPU fallback | `aten/src/ATen/cuda/FlashAttention.cu:584` | `PT_ERROR("Standard attention fallback not implemented")` — публичный API через `torch::nn::F::scaled_dot_product_attention` | README.md L1005 *"CUDA: ... FlashDecoding ... FlashAttention forward+backward временно отключён"*; README L1112 *"FlashAttention — forward + backward с online softmax"*; CLAUDE.md фаза 14 = **BROKEN** | **critical** | Реализовать CPU reference: explicit softmax(QK^T / sqrt(d))V (это уже описано в README L1545 как ожидаемое поведение CPU-fallback'а, но код просто кидает) |
| 2 | `cudnn::convolution_dispatch` CPU branch | `aten/src/ATen/cudnn/CuDNN.h:125` | `PT_ERROR("CPU convolution fallback not implemented in cuDNN dispatch")` | README.md L661 *"cuDNN wiring: ... CPU fallback"*; CLAUDE.md фаза 12 = BROKEN, no callsites | **critical** | Заменить PT_ERROR на вызов `at::native::cpu::conv2d_forward` (есть в native/cpu) — это и есть документированный CPU fallback |
| 3 | `cudnn::max_pool2d_dispatch` CPU branch | `aten/src/ATen/cudnn/CuDNN.h:138` | `PT_ERROR("CPU max_pool2d fallback not implemented")` | README L661, CLAUDE.md фаза 12 | **critical** | Заменить на `at::native::cpu::max_pool2d_forward` |
| 4 | `cudnn::avg_pool2d_dispatch` CPU branch | `aten/src/ATen/cudnn/CuDNN.h:154` | `PT_ERROR("CPU avg_pool2d fallback not implemented")` | README L661 | **critical** | Заменить на native CPU avg_pool2d |
| 5 | `cudnn::batch_norm_training_dispatch` CPU branch | `aten/src/ATen/cudnn/CuDNN.h:172` | `PT_ERROR("CPU batch_norm_training fallback not implemented")` | README L661 cuDNN wiring claim | **critical** | Заменить на `at::native::cpu::batch_norm2d_forward_training` |
| 6 | `cudnn::batch_norm_inference_dispatch` CPU branch | `aten/src/ATen/cudnn/CuDNN.h:189` | `PT_ERROR("CPU batch_norm_inference fallback not implemented")` | README L661 | **critical** | Native CPU batch_norm inference |
| 7 | `cudnn::relu_dispatch` CPU branch | `aten/src/ATen/cudnn/CuDNN.h:199` | `PT_ERROR("CPU relu fallback not implemented")` | README L661 | **major** | relu CPU есть везде — просто `at::relu(input)` |
| 8 | `cudnn::softmax_dispatch` CPU branch | `aten/src/ATen/cudnn/CuDNN.h:207` | `PT_ERROR("CPU softmax fallback not implemented")` | README L661 | **major** | `at::softmax(input, dim)` |
| 9 | `CTCLoss::forward(input)` single-arg | `torch/nn/modules/loss.h:1711-1713` | `throw std::runtime_error("CTCLoss requires log_probs, targets, input_lengths, and target_lengths")` | README.md L980 *"CTCLoss"* как стандартная loss; CLAUDE.md фаза 4 = "DONE (Conv3d — stub, CTCLoss — throw)"; README L617 *"CTCLoss полный Graves DP (ранее throw)"* — multi-arg реализован, но single-arg override базового `nn::Module::forward(input)` всё ещё throws | **major** | Однопараметрный `forward(input)` должен throw с явным сообщением *"use forward(log_probs, targets, input_lengths, target_lengths)"* — текущий текст ОК, но это значит CTCLoss НЕ usable как obj в `nn::Sequential` или generic trainer'е. Документировать в README что CTCLoss требует 4-arg API |
| 10 | `flash_attention_backward` ограничение `head_dim != {64,128}` | `aten/src/ATen/cuda/FlashAttention.cu:555` | `PT_ERROR("FlashAttention backward: head_dim must be 64 or 128, got ", head_dim)` | README L1005/L1112 — claim "FlashAttention forward+backward". На реальных моделях head_dim бывает 80/96/192/256 (GigaChat3 = 192) | **major** | Добавить generic backward (без template tile) либо документировать ограничение явно в API docstring |
| 11 | `matmul_multi_chip` копирует только 1-й чип | `aten/src/ATen/nmquad/NMQuadOps.h:181, 204-213` | Два TODO: `B_slice = B.contiguous()  // TODO: proper slicing` (берёт **весь** B вместо колонок `[col_start:col_end]`) и `// Copy to output (column-major concatenation) // For now just copy first chip result` | CLAUDE.md фаза 16 "NM Quad Backend = PARTIAL (100× vs own scalar; max 16 cores stable)"; MEMORY.md заявляет "16 cores (4 chips × 4) BIT-EXACT measured 24× speedup на M=2048" → существующие данные о bit-exact не могли проходить через эту функцию | **critical** | Либо реализовать column-slice + concat, либо удалить функцию и оставить только working 1-chip path. Сейчас функция silently возвращает мусор (M×N тензор где только первые `chip_cols[0]` колонок верные) |
| 12 | `gguf_model.h` lm_head FP16 tied-weights | `torch/io/gguf_model.h:1349-1354` | `// TODO: FP32→FP16 conversion kernel for cuBLAS lm_head` + `cudaFree(lm_head_fp16_); lm_head_fp16_ = nullptr;` — allocate then free | CLAUDE.md фаза 18 *"GGUF inference DONE (Q4_K/Q6_K; 49.9 tok/s на A100)"*; README L1005 *"Quantized inference (Q4_K_M, Q5_K, Q6_K, Q8_1)"*. Tied-weights модели (gemma3:27b и др.) попадут в эту ветку | **major** | Написать `__global__ void fp32_to_fp16_kernel` (тривиально, 5 строк). Сейчас silently fallback на quant path — функционально ОК но performance regression необъявлен |
| 13 | `Q3_K dequant` неполный unpack | `torch/io/gguf_dequant.h:514` | `// 2-bit base from qs (Q3_K stub — full unpack TBD)` + код использует `qs[byte_idx % 64]` (mod), что неправильно для полного блока QK_K=256 | README L1005 указывает Q4_K_M/Q5_K/Q6_K/Q8_1 — Q3_K **не** в списке, но `dequantize()` dispatcher (line 530) принимает GGML_TYPE_Q3_K. CLAUDE.md фаза 18 DONE — если кто-то загрузит Q3_K модель, тихо получит мусор | **major** | Либо реализовать Q3_K корректно (Graves описание есть в llama.cpp), либо `PT_CHECK_MSG(false, "Q3_K not supported")` в dispatcher вместо тихого мусора |
| 14 | `pt8` loader без metadata KV section | `torch/io/gguf_model.h:2188-2191` | `throw std::runtime_error("[load_pt8] PT8 metadata KV section not yet present in Agent B's writer")` | Формат `.pt8` упоминается в io/PT8_FORMAT.md как production format — load throws если нет sibling GGUF | **minor** | Документировано в комментарии. Нужно убрать упоминания PT8 как production-ready пока не дописан writer |
| 15 | `PT8_Q8_SOA4` K-slice TP | `torch/io/gguf_model.h:5302-5306` | `std::cerr << ... "PT8_Q8_SOA4 K-slice not yet implemented — falling back to replicated full-weight path"` + `return false` | MEMORY.md *"PT_Q8_SOA=1 ОБЯЗАТЕЛЬНА для TP-4 qwen3:4b"* — TP-4 заявлен как 11.4 tok/s baseline. Текущий код silently falls back на replicated full-weight (KV-кэш memory wasted на каждом ranke) | **major** | Реализовать K-slice для Q8_SOA4 (Round 4 work) либо warn пользователя что TP экономит compute но не память |
| 16 | `transformers` safetensors loader пропускает `pytorch_model.bin` | `torch/serve/llm.h:598-600` | `// Last resort: pytorch_model.bin — requires pickle parser (not implemented here)` + `return false` | README.md описывает llm serve как ready — но при HuggingFace модели без safetensors, load_weights silently returns false → tokenizer работает, веса нули → garbage output | **major** | Либо использовать external pickle parser (cnpy-style), либо явно throw "pytorch_model.bin not supported, convert to safetensors first" вместо silent false |
| 17 | `promeserve` `/api/embeddings` endpoint | `promeserve/api_handlers.h:556-561` | `resp.status = 501; resp.set_json("{\"error\":\"embeddings not implemented\"}")` | promeserve позиционируется как Ollama-совместимый сервер. Ollama API `/api/embeddings` — standard endpoint | **minor** | 501 — корректно, но удалить из README/docs если есть упоминания "Ollama-compatible API" |
| 18 | `nn::Module::forward()` базовые overrides | `torch/nn/module.h:69, 74, 84` | Три `throw std::runtime_error("forward(...) not implemented for " + name_)` для разных arity (single, two-arg, vector) | По дизайну виртуальные методы базового класса. **Acceptable** — но: если пользователь делает `MyModule.forward(other_tensor)` без override, текст ошибки общий и не указывает stack | **minor** | OK by design |
| 19 | `CumsumBackward` без `flip()` | `torch/csrc/autograd/functions/ReduceBackward.h:630` | `// Implement directly without flip (flip not yet available)` — реализовано через ручной reverse loop | Autograd объявлен DONE (CLAUDE.md фаза 3 "110 backward functions"). Реализация работает, но: `at::flip()` объявлен в IndexOps но не используется — либо `flip` есть и комментарий устарел, либо `flip` нет и нужно добавить | **minor** | Проверить наличие `at::flip` — если есть, переписать через `flip(cumsum(flip(g)))` (canonical PyTorch form) |
| 20 | `mps_ops::unary_dispatch` generic path | `aten/src/ATen/mps/MPSDispatch.h:174` | `PT_CHECK_MSG(false, "mps_ops::unary_dispatch: generic path not implemented")` | README L40 *"compile-verified only (MPS/ROCm — нет Mac/AMD GPU)"*; CLAUDE.md не претендует на MPS DONE. **Acceptable** | **minor** | Документировано как compile-only — OK |
| 21 | `mps_ops::binary_dispatch` generic path | `aten/src/ATen/mps/MPSDispatch.h:189` | `PT_CHECK_MSG(false, "mps_ops::binary_dispatch: call add/mul directly")` | См. #20 | **minor** | OK |
| 22 | `pipeline_schedule.h::run()` non-last stage возвращает undefined tensor | `torch/distributed/pipeline_schedule.h:187` | `return at::Tensor()` для всех ranks кроме последнего | По дизайну (последний stage возвращает выход) — но caller-facing API может на этом упасть если не знает | **minor** | Документировано в комментарии — OK |
| 23 | `mlir/export.h` large tensor placeholder | `torch/mlir/export.h:77-83` | Для tensor.numel > 4096 emits `arith.constant dense<0.0>` placeholder | README не упоминает MLIR export как DONE; это experimental | **minor** | Документировано — OK |
| 24 | `Q8_SOA4 NUMA K-slice` warn-and-fall-back | `torch/io/gguf_model.h:5309-5313` | unknown qtype branch: `std::cerr << ... "unsupported qtype=... (will use replicated fallback)"` + `return false` | См. #15 — TP-4 wiring | **major** | Тот же fix как #15 |
| 25 | `dispatcher_nmquad.cpp` recompute Q/K/V в backward | `aten/src/ATen/nmquad/nmc_programs/dispatcher_nmquad.cpp:689` | `// Recompute them (or cache in forward — TODO for speed)` — пересчитывает QKV в backward вместо cache | MEMORY.md "nanoGPT TinyStories на NM Quad — РАБОЧИЙ пайплайн 16 cores" — пайплайн работает, но 2× slowdown на backward | **minor** | Кэшировать Q/K/V в forward — perf оптимизация, корректность не страдает |
| 26 | `gguf_model.h:980 invalidate_graph()` | `torch/io/gguf_model.h:980` | `void invalidate_graph() {}` — пустое тело для non-CUDA build | По дизайну (no CUDA graphs to invalidate) | **minor** | OK |

## Сводка по severity

| Severity | Count | Notes |
|----------|-------|-------|
| critical | 7 | FlashAttention CPU fallback, 6 cuDNN CPU fallbacks, NMQuad multi-chip silent garbage |
| major | 8 | CTCLoss API, Flash backward head_dim, GGUF tied lm_head, Q3_K silent garbage, PT8 K-slice, safetensors-only loader, Q8_SOA4 K-slice |
| minor | 11 | by-design empties, documented compile-only, perf TODOs |

## Корреляция с CLAUDE.md "DONE" claims

| CLAUDE.md фаза | Реальность | Несоответствие |
|---|---|---|
| 4 NN Modules = DONE (Conv3d — stub, CTCLoss — throw) | Conv3d **починен** 2026-04-18 (см. conv.h:672 коммент); CTCLoss multi-arg работает, single-arg throws by design | Обновить CLAUDE.md: "Conv3d — DONE (im2col); CTCLoss — multi-arg DONE, single-arg N/A" |
| 12 cuDNN Integration = BROKEN | Подтверждено: 7 PT_ERROR stubs во всех dispatch функциях для CPU branch. cuDNN ветка работает, fallback мёртв | OK match |
| 14 FlashAttention = BROKEN (6 багов) | `dim3(64,64)` fix НЕ в коде (CU:310 показывает `dim3 block(BKV, BQ)` где BKV/BQ задаются template params 32/16/32, 1024 макс) — это исправлено. Но `scaled_dot_product_attention` CPU branch всё ещё throws | Обновить: forward+backward работают для head_dim={64,128} only; CPU fallback throws |
| 18 GGUF inference = DONE | Q3_K stub в dequant + tied-weights FP16 stub + PT8_Q8_SOA4 K-slice missing | Уточнить: "GGUF Q4_K_M/Q5_K/Q6_K/Q8_0 DONE; Q3_K incomplete; tied lm_head fallback to quant" |

## Файлы где stubs прячутся в production-paths (highest priority fixes)

1. `aten/src/ATen/cudnn/CuDNN.h` — 7× PT_ERROR (если PT_USE_CUDNN=ON но input на CPU)
2. `aten/src/ATen/cuda/FlashAttention.cu:584` — public API throws на CPU
3. `aten/src/ATen/nmquad/NMQuadOps.h:152-217` — `matmul_multi_chip` silent garbage
4. `torch/io/gguf_dequant.h:505-523` — Q3_K silent garbage
5. `torch/serve/llm.h:598-600` — silent false для pytorch_model.bin
