#pragma once

#include "torch/io/gguf_loader.h"
#include "torch/io/gguf_dequant.h"
#include "torch/io/cpu_quant_gemv.h"
#include "torch/io/speculative_decode.h"
#include "torch/io/sliding_window_attn.h"
#include "torch/io/sparse_gemv.h"
#include "torch/io/ollama.h"
#include "torch/io/tokenizer.h"
#include "torch/io/inference_profiler.h"
#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

namespace torch {
namespace io {

using at::Tensor;

// ============================================================================
// Transformer Configuration (parsed from GGUF metadata)
// ============================================================================

struct TransformerConfig {
    std::string architecture;
    std::string model_name;

    int64_t vocab_size = 0;
    int64_t hidden_size = 0;       // embedding_length
    int64_t num_layers = 0;        // block_count
    int64_t num_heads = 0;         // attention.head_count
    int64_t num_kv_heads = 0;      // attention.head_count_kv (for GQA)
    int64_t intermediate_size = 0; // feed_forward_length
    int64_t head_dim = 0;          // from attention.key_length or hidden/heads
    int64_t context_length = 0;

    float rope_freq_base = 10000.0f;
    float rms_norm_eps = 1e-6f;

    // Architecture-specific
    bool tie_word_embeddings = false;
    bool scale_embeddings = false;     // Gemma: multiply embeddings by sqrt(hidden)
    bool gemma_norm_add_one = false;   // Gemma: RMSNorm weight += 1
    bool has_qk_norm = false;          // Gemma3: per-head Q/K normalization
    bool has_post_norm = false;        // Gemma3: post-attention + post-FFN norms

    void parse(const gguf::GGUFReader& reader) {
        architecture = reader.architecture();
        model_name = reader.get_string("general.name", architecture);

        hidden_size = reader.get_arch_int("embedding_length", 0);
        num_layers = reader.get_arch_int("block_count", 0);
        num_heads = reader.get_arch_int("attention.head_count", 0);
        num_kv_heads = reader.get_arch_int("attention.head_count_kv", num_heads);
        intermediate_size = reader.get_arch_int("feed_forward_length", 0);
        context_length = reader.get_arch_int("context_length", 8192);

        rope_freq_base = reader.get_arch_float("rope.freq_base", 10000.0f);
        rms_norm_eps = reader.get_arch_float("attention.layer_norm_rms_epsilon", 1e-6f);

        // Head dim: prefer explicit key_length, fallback to hidden/heads
        int64_t key_length = reader.get_arch_int("attention.key_length", 0);
        head_dim = (key_length > 0) ? key_length : ((num_heads > 0) ? hidden_size / num_heads : 0);

        // Gemma-specific
        if (architecture == "gemma" || architecture == "gemma2" || architecture == "gemma3") {
            scale_embeddings = true;
            tie_word_embeddings = true;
            // Note: GGUF converter already bakes in the +1 (layernorm1p)
            // so we do NOT add 1 again during inference
            gemma_norm_add_one = false;
        }
        if (architecture == "gemma3") {
            has_qk_norm = true;
            has_post_norm = true;
        }
        // Qwen3 also has QK-norm
        if (architecture == "qwen3") {
            has_qk_norm = true;
        }
    }

    void print() const {
        std::cout << "\n=== Model Config ===" << std::endl;
        std::cout << "  Architecture: " << architecture << std::endl;
        std::cout << "  Name: " << model_name << std::endl;
        std::cout << "  Hidden size: " << hidden_size << std::endl;
        std::cout << "  Layers: " << num_layers << std::endl;
        std::cout << "  Heads: " << num_heads << " (KV: " << num_kv_heads << ")" << std::endl;
        std::cout << "  Head dim: " << head_dim << std::endl;
        std::cout << "  Q dim: " << (num_heads * head_dim) << ", KV dim: " << (num_kv_heads * head_dim) << std::endl;
        std::cout << "  FFN size: " << intermediate_size << std::endl;
        std::cout << "  Vocab size: " << vocab_size << std::endl;
        std::cout << "  Context length: " << context_length << std::endl;
        std::cout << "  RoPE base: " << rope_freq_base << std::endl;
        std::cout << "  RMS norm eps: " << rms_norm_eps << std::endl;
        std::cout << "  Gemma features: scale_emb=" << scale_embeddings
                  << " norm+1=" << gemma_norm_add_one
                  << " qk_norm=" << has_qk_norm
                  << " post_norm=" << has_post_norm << std::endl;
    }
};

// ============================================================================
// Transformer Layer Weights
// ============================================================================

// ============================================================================
// Quantized Weight — raw quant blocks stored on GPU/CPU
// Supports Q4_K (144B/256vals) and Q6_K (210B/256vals)
// ============================================================================

struct QuantizedWeight {
    void* gpu_data = nullptr;     // raw quant blocks on GPU
    void* cpu_data = nullptr;     // raw quant blocks on CPU (heap or mmap'd)
    void* fp16_data = nullptr;    // dequantized FP16 weights on GPU [N, K]
    int64_t rows = 0;             // N (out_features) — original [N, K] layout
    int64_t cols = 0;             // K (in_features)
    int64_t row_stride_bytes = 0; // bytes per row
    int64_t total_bytes = 0;
    uint32_t quant_type = 0;      // gguf::GGML_TYPE_Q4_K or Q6_K
    bool valid = false;

    bool is_q4k() const { return quant_type == 12; }  // GGML_TYPE_Q4_K
    bool is_q6k() const { return quant_type == 14; }  // GGML_TYPE_Q6_K
    bool is_q5k() const { return quant_type == 13; }  // GGML_TYPE_Q5_K
    bool is_q8_0() const { return quant_type == 8; }  // GGML_TYPE_Q8_0

    bool mmap_owned = false;  // true if cpu_data points into mmap region (don't free!)

    void free_gpu() {
        gpu_data = nullptr;
        valid = false;
    }

    void free_cpu() {
        if (cpu_data && !mmap_owned) {
            std::free(cpu_data);
        }
        cpu_data = nullptr;
    }

#ifdef PT_USE_CUDA
    void dequant_to_fp16() {
        if (!gpu_data || !valid || !is_q4k()) return;
        int64_t fp16_bytes = rows * cols * 2;  // sizeof(half) = 2
        cudaMalloc(&fp16_data, fp16_bytes);
        at::cuda::launch_dequant_q4k_to_fp16(
            gpu_data, fp16_data,
            static_cast<int>(cols), static_cast<int>(rows),
            row_stride_bytes, nullptr);
        cudaDeviceSynchronize();
    }
#endif
};

struct TransformerLayer {
    // Attention
    Tensor attn_norm;    // [hidden]
    Tensor attn_q;       // [num_heads * head_dim, hidden]
    Tensor attn_k;       // [num_kv_heads * head_dim, hidden]
    Tensor attn_v;       // [num_kv_heads * head_dim, hidden]
    Tensor attn_output;  // [hidden, num_heads * head_dim]

    // QK-norm (Gemma3)
    Tensor attn_q_norm;  // [head_dim] per-head Q normalization
    Tensor attn_k_norm;  // [head_dim] per-head K normalization

    // Post-norms (Gemma3)
    Tensor post_attention_norm;  // [hidden]
    Tensor post_ffw_norm;        // [hidden]

    // FFN (SwiGLU)
    Tensor ffn_norm;     // [hidden]
    Tensor ffn_gate;     // [intermediate, hidden]
    Tensor ffn_up;       // [intermediate, hidden]
    Tensor ffn_down;     // [hidden, intermediate]

    // Optional biases (Qwen has some)
    Tensor attn_q_bias;
    Tensor attn_k_bias;
    Tensor attn_v_bias;

    // Quantized weights (GPU, Q4_K_M) — used for decode GEMV
    QuantizedWeight q_attn_q, q_attn_k, q_attn_v, q_attn_output;
    QuantizedWeight q_ffn_gate, q_ffn_up, q_ffn_down;
};

// ============================================================================
// KV Cache for inference — Pre-allocated for zero-allocation decode
// ============================================================================

struct KVCache {
    std::vector<Tensor> key_cache;   // per layer: [max_seq, kv_dim] pre-allocated
    std::vector<Tensor> value_cache; // per layer: [max_seq, kv_dim] pre-allocated

    // FP16 KV cache (halves attention memory bandwidth)
    std::vector<void*> key_cache_fp16;   // per layer: [max_seq, kv_dim] FP16
    std::vector<void*> value_cache_fp16; // per layer: [max_seq, kv_dim] FP16
    bool use_fp16_kv = false;

    int64_t seq_len = 0;
    int64_t max_seq = 0;
    int64_t kv_dim_ = 0;
    bool allocated = false;

    void reset() {
        seq_len = 0;
        // Don't deallocate — reuse buffers
    }

    void allocate(int64_t num_layers, int64_t max_seq_len, int64_t kv_dim, bool use_cuda) {
        max_seq = max_seq_len;
        kv_dim_ = kv_dim;
        key_cache.resize(num_layers);
        value_cache.resize(num_layers);
        for (int64_t i = 0; i < num_layers; ++i) {
#ifdef PT_USE_CUDA
            if (use_cuda) {
                key_cache[i] = at::empty_cuda({max_seq, kv_dim});
                value_cache[i] = at::empty_cuda({max_seq, kv_dim});
            } else
#endif
            {
                key_cache[i] = at::empty({max_seq, kv_dim});
                value_cache[i] = at::empty({max_seq, kv_dim});
            }
        }

        // Allocate FP16 KV cache for CUDA decode (halves attention bandwidth)
#ifdef PT_USE_CUDA
        if (use_cuda) {
            use_fp16_kv = true;
            key_cache_fp16.resize(num_layers, nullptr);
            value_cache_fp16.resize(num_layers, nullptr);
            for (int64_t i = 0; i < num_layers; ++i) {
                cudaMalloc(&key_cache_fp16[i], max_seq * kv_dim * sizeof(uint16_t));
                cudaMalloc(&value_cache_fp16[i], max_seq * kv_dim * sizeof(uint16_t));
            }
            size_t fp16_bytes = num_layers * 2 * max_seq * kv_dim * sizeof(uint16_t);
            std::cout << "[KVCache] FP16 KV cache: " << (fp16_bytes / (1024*1024))
                      << " MB (" << num_layers << " layers)" << std::endl;
        }
#endif

        allocated = true;
        seq_len = 0;
    }

    // Append new K/V rows at current seq_len offset (no reallocation!)
    void append(int64_t layer_idx, const Tensor& new_k, const Tensor& new_v, bool use_cuda) {
        int64_t num_new = new_k.size(0);
        int64_t kv_dim = new_k.size(1);
        (void)kv_dim;
#ifdef PT_USE_CUDA
        if (use_cuda) {
            at::cuda::launch_kv_cache_write(
                new_k.data_ptr<float>(), key_cache[layer_idx].mutable_data_ptr<float>(),
                num_new, new_k.size(1), seq_len, nullptr);
            at::cuda::launch_kv_cache_write(
                new_v.data_ptr<float>(), value_cache[layer_idx].mutable_data_ptr<float>(),
                num_new, new_v.size(1), seq_len, nullptr);
            // Also write to FP16 cache if enabled
            if (use_fp16_kv) {
                at::cuda::launch_fp16_kv_cache_write(
                    new_k.data_ptr<float>(), key_cache_fp16[layer_idx],
                    num_new, new_k.size(1), seq_len, nullptr);
                at::cuda::launch_fp16_kv_cache_write(
                    new_v.data_ptr<float>(), value_cache_fp16[layer_idx],
                    num_new, new_v.size(1), seq_len, nullptr);
            }
            return;
        }
#endif
        // CPU fallback: memcpy
        int64_t cols = new_k.size(1);
        std::memcpy(key_cache[layer_idx].mutable_data_ptr<float>() + seq_len * cols,
                     new_k.data_ptr<float>(), num_new * cols * sizeof(float));
        std::memcpy(value_cache[layer_idx].mutable_data_ptr<float>() + seq_len * cols,
                     new_v.data_ptr<float>(), num_new * cols * sizeof(float));
    }

    // Get view of K cache [0..seq_len+new_rows-1, kv_dim] — just adjust size interpretation
    // Since we pre-allocated, we return the full buffer and pass total_seq to kernels
};

// ============================================================================
// Scratch Pool — Pre-allocated GPU buffers for zero-allocation decode
// ============================================================================

struct InferenceScratchPool {
    Tensor buf_x[2];       // [1, H] — double-buffered hidden state
    Tensor buf_normed;     // [1, H]
    Tensor buf_q;          // [1, q_dim]
    Tensor buf_k;          // [1, kv_dim]
    Tensor buf_v;          // [1, kv_dim]
    Tensor buf_attn;       // [1, q_dim] — attention output
    Tensor buf_attn_proj;  // [1, H]
    Tensor buf_h;          // [1, H] — residual intermediate
    Tensor buf_gate;       // [1, intermediate]
    Tensor buf_up;         // [1, intermediate]
    Tensor buf_silu;       // [1, intermediate]
    Tensor buf_down;       // [1, H]
    Tensor buf_logits;     // [1, vocab_size]
    void* q8_buf = nullptr;     // Q8_1 quantized x for dp4a GEMV (legacy)
    void* x_fp16_buf = nullptr; // FP16 scratch for cuBLAS GEMV input
    void* y_fp16_buf = nullptr; // FP16 scratch for cuBLAS GEMV output

    // Flash-decode scratch buffers
    float* fd_partial_O = nullptr;    // [max_splits * n_heads * head_dim]
    float* fd_partial_lse = nullptr;  // [max_splits * n_heads]
    float* fd_partial_max = nullptr;  // [max_splits * n_heads]
    int fd_max_splits = 0;

    bool allocated = false;

    void allocate(const TransformerConfig& config) {
#ifdef PT_USE_CUDA
        int64_t H = config.hidden_size;
        int64_t q_dim = config.num_heads * config.head_dim;
        int64_t kv_dim = config.num_kv_heads * config.head_dim;
        int64_t inter = config.intermediate_size;
        int64_t V = config.vocab_size;

        buf_x[0] = at::empty_cuda({1, H});
        buf_x[1] = at::empty_cuda({1, H});
        buf_normed = at::empty_cuda({1, H});
        buf_q = at::empty_cuda({1, q_dim});
        buf_k = at::empty_cuda({1, kv_dim});
        buf_v = at::empty_cuda({1, kv_dim});
        buf_attn = at::empty_cuda({1, q_dim});
        buf_attn_proj = at::empty_cuda({1, H});
        buf_h = at::empty_cuda({1, H});
        buf_gate = at::empty_cuda({1, inter});
        buf_up = at::empty_cuda({1, inter});
        buf_silu = at::empty_cuda({1, inter});
        buf_down = at::empty_cuda({1, H});
        buf_logits = at::empty_cuda({1, V});

        // FP16 scratch buffers for cuBLAS HGEMV
        int64_t max_K = H > inter ? H : inter;
        if (q_dim > max_K) max_K = q_dim;
        int64_t max_N = V > inter ? V : inter;
        if (q_dim > max_N) max_N = q_dim;
        cudaMalloc(&x_fp16_buf, max_K * 2);  // sizeof(half) = 2
        cudaMalloc(&y_fp16_buf, max_N * 2);

        // Flash-decode scratch buffers
        // Max context = max_seq_len, splits = ceil(max_seq / 256)
        int64_t max_seq = config.context_length > 0 ? config.context_length : 8192;
        fd_max_splits = at::cuda::flash_decode_num_splits(static_cast<int>(max_seq));
        int64_t n_heads = config.num_heads;
        int64_t hdim = config.head_dim;
        cudaMalloc(&fd_partial_O, fd_max_splits * n_heads * hdim * sizeof(float));
        cudaMalloc(&fd_partial_lse, fd_max_splits * n_heads * sizeof(float));
        cudaMalloc(&fd_partial_max, fd_max_splits * n_heads * sizeof(float));

        allocated = true;

        size_t total_bytes = (6*H + 2*q_dim + 2*kv_dim + 3*inter + V) * sizeof(float);
        size_t fd_bytes = fd_max_splits * n_heads * (hdim + 2) * sizeof(float);
        std::cout << "[Scratch] Allocated decode buffers: "
                  << (total_bytes / 1024) << " KB"
                  << " + flash-decode: " << (fd_bytes / 1024) << " KB"
                  << " (" << fd_max_splits << " splits)" << std::endl;
#endif
    }
};

// ============================================================================
// CPU Scratch Pool — Pre-allocated raw float buffers for zero-allocation
// CPU decode. Unlike GPU scratch (Tensor-based), these are raw pointers
// to avoid Tensor allocation overhead (~1us per at::empty call x 14 calls
// x 28 layers = 392us per token wasted on allocations alone).
// ============================================================================

struct CPUScratchPool {
    float* x_buf[2] = {nullptr, nullptr};  // double-buffered [H]
    float* q_buf = nullptr;       // [q_dim]
    float* k_buf = nullptr;       // [kv_dim]
    float* v_buf = nullptr;       // [kv_dim]
    float* attn_buf = nullptr;    // [q_dim]
    float* h_buf = nullptr;       // [H] residual intermediate
    float* gate_buf = nullptr;    // [intermediate]
    float* up_buf = nullptr;      // [intermediate]
    float* down_buf = nullptr;    // [H]
    float* logits_buf = nullptr;  // [vocab_size]
    float* scores_buf = nullptr;  // [max_seq] for attention scores

    bool allocated = false;

    void allocate(const TransformerConfig& config, int64_t max_seq) {
        int64_t H = config.hidden_size;
        int64_t q_dim = config.num_heads * config.head_dim;
        int64_t kv_dim = config.num_kv_heads * config.head_dim;
        int64_t inter = config.intermediate_size;
        int64_t V = config.vocab_size;

        // Aligned allocation for AVX2 (32-byte aligned)
        auto alloc = [](int64_t n) -> float* {
            // _aligned_malloc on Windows, aligned_alloc on Linux
#ifdef _WIN32
            return static_cast<float*>(_aligned_malloc(n * sizeof(float), 32));
#else
            void* ptr = nullptr;
            posix_memalign(&ptr, 32, n * sizeof(float));
            return static_cast<float*>(ptr);
#endif
        };

        x_buf[0] = alloc(H);
        x_buf[1] = alloc(H);
        q_buf = alloc(q_dim);
        k_buf = alloc(kv_dim);
        v_buf = alloc(kv_dim);
        attn_buf = alloc(q_dim);
        h_buf = alloc(H);
        gate_buf = alloc(inter);
        up_buf = alloc(inter);
        down_buf = alloc(H);
        logits_buf = alloc(V);
        scores_buf = alloc(max_seq);

        allocated = true;

        size_t total = (3*H + 2*q_dim + 2*kv_dim + 2*inter + V + max_seq) * sizeof(float);
        std::cout << "[CPUScratch] Allocated " << (total / 1024) << " KB decode buffers" << std::endl;
    }

    ~CPUScratchPool() {
        auto dealloc = [](float*& p) {
            if (p) {
#ifdef _WIN32
                _aligned_free(p);
#else
                free(p);
#endif
                p = nullptr;
            }
        };
        dealloc(x_buf[0]); dealloc(x_buf[1]);
        dealloc(q_buf); dealloc(k_buf); dealloc(v_buf);
        dealloc(attn_buf); dealloc(h_buf);
        dealloc(gate_buf); dealloc(up_buf); dealloc(down_buf);
        dealloc(logits_buf); dealloc(scores_buf);
    }
};

// ============================================================================
// GGUFModel — Load and run inference with GGUF transformer models
// ============================================================================

class GGUFModel {
public:
    TransformerConfig config;
    GGUFTokenizer tokenizer;

    // Weights
    Tensor token_embedding;  // [vocab_size, hidden] (CPU)
    Tensor emb_gpu_;         // [vocab_size, hidden] (GPU copy for CUDA Graph)
    Tensor output_norm;      // [hidden]
    Tensor output_weight;    // [vocab_size, hidden]
    std::vector<TransformerLayer> layers;

    // KV cache
    KVCache kv_cache;

    // Device
    bool use_cuda_ = false;
    bool use_quant_gemv_ = false;  // Q4_K_M decode acceleration
    bool output_weight_needs_float32_ = false;  // true if output.weight has no Q4_K_M

    // GGUF file path (for reloading raw quant data)
    std::string gguf_file_path_;
    QuantizedWeight q_output_weight;  // quantized output projection

    // Memory-mapped GGUF file (kept alive for zero-copy weight access)
    // When active, QuantizedWeight::cpu_data points directly into mmap'd region
    // No malloc/memcpy needed — OS pages in data on demand
    gguf::MmapHandle mmap_handle_;
    bool use_mmap_ = false;  // true if weights are mmap'd (don't free cpu_data!)

    // CUDA Graph for decode acceleration (eliminates kernel launch overhead)
#ifdef PT_USE_CUDA
    cudaGraph_t decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    bool graph_captured_ = false;
    int graph_token_id_ = 0;      // Updated before each graph launch
    int* d_token_id_ = nullptr;   // Device memory for token_id (graph-updateable)
    int64_t* d_past_len_ = nullptr;  // Device memory for past_len (graph-compatible RoPE/KV)
    cudaStream_t decode_stream_ = nullptr;
#endif

    // Profiler (enabled with --profile flag)
    InferenceProfiler profiler;

    // Scratch pool for zero-allocation decode
    InferenceScratchPool scratch_;

    // CPU scratch pool for zero-allocation CPU decode
    CPUScratchPool cpu_scratch_;

    // === Algorithmic acceleration ===
    LowRankOutputProj low_rank_output_;      // Speculative decode via low-rank output proj
    SlidingWindowAttention sliding_window_;   // Sliding window attention
    SparseQ4KWeight sparse_output_;           // Sparse GEMV for output projection
    std::vector<SparseQ4KWeight> sparse_ffn_; // Sparse GEMV for FFN layers
    bool use_speculative_output_ = false;     // Enable speculative output decode
    bool use_sparse_gemv_ = false;            // Enable sparse GEMV
    int64_t sliding_window_size_ = 0;         // 0 = disabled

    // Helper: move tensor to CPU
    Tensor to_cpu_tensor(const Tensor& t) const {
#ifdef PT_USE_CUDA
        return at::to_cpu(t);
#else
        return t;
#endif
    }

    // Helper: move tensor to CUDA
    Tensor to_cuda_tensor(const Tensor& t) const {
#ifdef PT_USE_CUDA
        return at::to_cuda(t);
#else
        return t;
#endif
    }

    // ========================================================================
    // Move weights to CUDA (quant-only mode)
    // Only moves small tensors (norms, biases). Large weight matrices stay
    // as Q4_K_M on GPU — no float32 duplication. Saves ~14 GB VRAM.
    // ========================================================================

    void to_cuda() {
#ifdef PT_USE_CUDA
        auto t_start = std::chrono::high_resolution_clock::now();
        std::cout << "[Model] Moving weights to CUDA (quant-only mode)..." << std::endl;

        auto move = [](Tensor& t) {
            if (t.defined() && t.is_cpu()) {
                t = at::to_cuda(t);
            }
        };

        // Keep token_embedding on CPU — embedding lookup is CPU-based

        // Output weight: only move if no Q4_K_M version available (e.g. tied embeddings)
        // Will be loaded as Q4_K_M later if possible; otherwise needs float32
        // Note: we defer this — load_quantized_to_cuda() handles Q4_K_M.
        // If output.weight has Q4_K_M, we skip float32. Otherwise, move it.
        output_weight_needs_float32_ = true;  // assume yes, corrected after quant load

        // Output norm — small 1D tensor, always move
        move(output_norm);

        for (auto& layer : layers) {
            // 1D norms — always move (tiny: hidden_size floats each)
            move(layer.attn_norm);
            move(layer.ffn_norm);
            move(layer.attn_q_norm);
            move(layer.attn_k_norm);
            move(layer.post_attention_norm);
            move(layer.post_ffw_norm);

            // 1D biases — always move
            move(layer.attn_q_bias);
            move(layer.attn_k_bias);
            move(layer.attn_v_bias);

            // SKIP large 2D weight matrices — they stay on CPU for now
            // load_quantized_to_cuda() will either:
            //   - load Q4_K_M and free float32 (saves 14 GB VRAM)
            //   - or move float32 to GPU as fallback (non-Q4_K_M weights)
        }

        use_cuda_ = true;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[Model] Moved to CUDA in " << (ms / 1000.0) << " seconds" << std::endl;

        // Report VRAM usage
        size_t vram_free = 0, vram_total = 0;
        cudaMemGetInfo(&vram_free, &vram_total);
        double used_gb = (vram_total - vram_free) / (1024.0 * 1024.0 * 1024.0);
        double total_gb = vram_total / (1024.0 * 1024.0 * 1024.0);
        std::cout << "[Model] VRAM: " << std::fixed << std::setprecision(1)
                  << used_gb << " / " << total_gb << " GB" << std::endl;
#else
        std::cerr << "[Model] CUDA not available (compiled without PT_USE_CUDA)" << std::endl;
#endif
    }

    // ========================================================================
    // Load raw Q4_K_M weights to GPU for quantized GEMV inference
    // Call AFTER to_cuda(). Re-reads GGUF file to get raw quant blocks.
    // ========================================================================

    void load_quantized_to_cuda() {
#ifdef PT_USE_CUDA
        if (gguf_file_path_.empty()) {
            std::cerr << "[Quant] No GGUF file path stored — skipping quantized loading" << std::endl;
            return;
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        std::cout << "[Quant] Loading quantized weights to GPU..." << std::endl;

        gguf::GGUFReader reader;
        reader.open(gguf_file_path_);

        auto upload_quant = [&](const std::string& name, QuantizedWeight& qw) {
            if (!reader.has_tensor(name)) return;
            const auto& info = reader.get_tensor_info(name);
            // Accept Q4_K, Q5_K, Q6_K
            uint32_t type = info.type;
            int64_t block_bytes = 0;
            if (type == gguf::GGML_TYPE_Q4_K) block_bytes = 144;
            else if (type == gguf::GGML_TYPE_Q6_K) block_bytes = 210;
            else if (type == gguf::GGML_TYPE_Q5_K) block_bytes = 176;
            else return;  // F16/F32/other → handled via float32 fallback

            auto raw = reader.load_raw_tensor(name);
            auto shape = raw.shape;
            qw.rows = shape[0];
            qw.cols = shape[1];
            qw.row_stride_bytes = (qw.cols / 256) * block_bytes;
            qw.total_bytes = static_cast<int64_t>(raw.data.size());
            qw.quant_type = type;

            cudaError_t err = cudaMalloc(&qw.gpu_data, qw.total_bytes);
            if (err != cudaSuccess) {
                qw.gpu_data = nullptr;
                qw.valid = false;
                return;  // fallback to float32
            }
            cudaMemcpy(qw.gpu_data, raw.data.data(), qw.total_bytes, cudaMemcpyHostToDevice);
            qw.valid = true;
        };

        // Helper: move float32 to GPU with pre-transpose (for weights without Q4_K_M)
        auto move_t_gpu = [](Tensor& t) {
            if (t.defined() && t.dim() == 2 && t.is_cpu()) {
                Tensor tr = t.t().contiguous();
                t = at::to_cuda(tr);
            } else if (t.defined() && t.is_cpu()) {
                t = at::to_cuda(t);
            }
        };

        // Helper: if Q4_K_M loaded → free float32; else → move float32 to GPU
        auto finalize_weight = [&](QuantizedWeight& qw, Tensor& float_w) {
            if (qw.valid) {
                float_w = Tensor();  // free float32 — Q4_K_M is used
            } else if (float_w.defined()) {
                move_t_gpu(float_w);  // fallback: move float32 to GPU
            }
        };

        // Load per-layer weights + finalize
        int64_t freed_count = 0, gpu_count = 0;
        for (int64_t i = 0; i < config.num_layers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";
            auto& layer = layers[i];

            upload_quant(prefix + "attn_q.weight", layer.q_attn_q);
            upload_quant(prefix + "attn_k.weight", layer.q_attn_k);
            upload_quant(prefix + "attn_v.weight", layer.q_attn_v);
            upload_quant(prefix + "attn_output.weight", layer.q_attn_output);
            upload_quant(prefix + "ffn_gate.weight", layer.q_ffn_gate);
            upload_quant(prefix + "ffn_up.weight", layer.q_ffn_up);
            upload_quant(prefix + "ffn_down.weight", layer.q_ffn_down);

            // For each weight: Q4_K_M → free float32, else → move float32 to GPU
            finalize_weight(layer.q_attn_q, layer.attn_q);
            finalize_weight(layer.q_attn_k, layer.attn_k);
            finalize_weight(layer.q_attn_v, layer.attn_v);
            finalize_weight(layer.q_attn_output, layer.attn_output);
            finalize_weight(layer.q_ffn_gate, layer.ffn_gate);
            finalize_weight(layer.q_ffn_up, layer.ffn_up);
            finalize_weight(layer.q_ffn_down, layer.ffn_down);

            // Count for logging
            auto count = [&](const QuantizedWeight& qw) {
                if (qw.valid) freed_count++; else gpu_count++;
            };
            count(layer.q_attn_q); count(layer.q_attn_k); count(layer.q_attn_v);
            count(layer.q_attn_output);
            count(layer.q_ffn_gate); count(layer.q_ffn_up); count(layer.q_ffn_down);
        }

        // Output projection — try output.weight first, then tied token_embd.weight
        if (reader.has_tensor("output.weight")) {
            upload_quant("output.weight", q_output_weight);
        } else if (config.tie_word_embeddings && reader.has_tensor("token_embd.weight")) {
            upload_quant("token_embd.weight", q_output_weight);
        }

        use_quant_gemv_ = true;

        // Handle output_weight
        if (q_output_weight.valid) {
            output_weight_needs_float32_ = false;
            output_weight = Tensor();
            freed_count++;
        } else if (output_weight_needs_float32_ && output_weight.defined()) {
            Tensor out_cpu = output_weight.is_cpu() ? output_weight : to_cpu_tensor(output_weight);
            Tensor out_tr = out_cpu.t().contiguous();
            output_weight = at::to_cuda(out_tr);
            gpu_count++;
        }

        std::cout << "[Quant] " << freed_count << " weights → quantized (float32 freed), "
                  << gpu_count << " weights → float32 on GPU (fallback)" << std::endl;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[Quant] Loaded in " << (ms / 1000.0) << " seconds" << std::endl;

        // Report VRAM usage
        size_t vram_free = 0, vram_total = 0;
        cudaMemGetInfo(&vram_free, &vram_total);
        double used_gb = (vram_total - vram_free) / (1024.0 * 1024.0 * 1024.0);
        double total_gb = vram_total / (1024.0 * 1024.0 * 1024.0);
        std::cout << "[Quant] VRAM: " << std::fixed << std::setprecision(1)
                  << used_gb << " / " << total_gb << " GB" << std::endl;

        // Note: cuBLAS FP16 GEMV is available but slower than custom Q4_K GEMV
        // for M=1 (reads 3.5x more data). Keeping infrastructure for batch GEMM.
#else
        std::cerr << "[Quant] CUDA not available" << std::endl;
#endif
    }

    // ========================================================================
    // Load raw Q4_K_M weights to CPU for fused dequant-GEMV
    // Avoids reading 14 GB of float32 — reads 2.6 GB of Q4_K_M instead (7x less bandwidth)
    // ========================================================================

    void load_quantized_to_cpu() {
        if (gguf_file_path_.empty()) {
            std::cerr << "[Quant] No GGUF file path stored — skipping" << std::endl;
            return;
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        std::cout << "[Quant] Loading quantized weights to CPU..." << std::endl;

        gguf::GGUFReader reader;
        reader.open(gguf_file_path_);

        auto upload_quant_cpu = [&](const std::string& name, QuantizedWeight& qw) {
            if (!reader.has_tensor(name)) return;
            const auto& info = reader.get_tensor_info(name);
            uint32_t type = info.type;
            int64_t block_bytes = 0;
            int64_t group_size = 256;  // QK_K for K-quants
            if (type == gguf::GGML_TYPE_Q4_K) block_bytes = 144;
            else if (type == gguf::GGML_TYPE_Q6_K) block_bytes = 210;
            else if (type == gguf::GGML_TYPE_Q5_K) block_bytes = 176;
            else if (type == gguf::GGML_TYPE_Q8_0) { block_bytes = 34; group_size = 32; }
            else return;

            auto raw = reader.load_raw_tensor(name);
            auto shape = raw.shape;
            qw.rows = shape[0];
            qw.cols = shape[1];
            qw.row_stride_bytes = (qw.cols / group_size) * block_bytes;
            qw.total_bytes = static_cast<int64_t>(raw.data.size());
            qw.quant_type = type;

            qw.cpu_data = std::malloc(qw.total_bytes);
            std::memcpy(qw.cpu_data, raw.data.data(), qw.total_bytes);
            qw.valid = true;
        };

        for (int64_t i = 0; i < config.num_layers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";
            auto& layer = layers[i];
            upload_quant_cpu(prefix + "attn_q.weight", layer.q_attn_q);
            upload_quant_cpu(prefix + "attn_k.weight", layer.q_attn_k);
            upload_quant_cpu(prefix + "attn_v.weight", layer.q_attn_v);
            upload_quant_cpu(prefix + "attn_output.weight", layer.q_attn_output);
            upload_quant_cpu(prefix + "ffn_gate.weight", layer.q_ffn_gate);
            upload_quant_cpu(prefix + "ffn_up.weight", layer.q_ffn_up);
            upload_quant_cpu(prefix + "ffn_down.weight", layer.q_ffn_down);
        }

        if (reader.has_tensor("output.weight")) {
            upload_quant_cpu("output.weight", q_output_weight);
        }

        use_quant_gemv_ = true;

        // === Initialize algorithmic accelerations ===
        init_cpu_accelerations();

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[Quant] CPU Q4_K_M loaded in " << (ms / 1000.0) << " seconds" << std::endl;
    }

    // ========================================================================
    // Initialize CPU algorithmic accelerations
    // Called after quantized weights are loaded to CPU
    // ========================================================================
    void init_cpu_accelerations() {
        // 1. Low-rank output projection (speculative decode)
        if (q_output_weight.valid && q_output_weight.cpu_data &&
            config.vocab_size > 10000) {
            int64_t rank = 256;
            if (config.hidden_size < 512) rank = 64;
            else if (config.hidden_size < 2048) rank = 128;

            low_rank_output_.candidate_k = 64;
            low_rank_output_.init_from_quantized(
                q_output_weight.cpu_data, q_output_weight.quant_type,
                config.vocab_size, q_output_weight.cols,
                q_output_weight.row_stride_bytes, rank);

            if (low_rank_output_.valid) {
                use_speculative_output_ = true;
                std::cout << "[Accel] Speculative output decode enabled"
                          << " (rank=" << rank << ", top-k=" << low_rank_output_.candidate_k << ")"
                          << std::endl;
            }
        }

        // 2. Sparse GEMV for output projection
        if (q_output_weight.valid && q_output_weight.cpu_data && q_output_weight.is_q4k()) {
            sparse_output_.analyze(q_output_weight.cpu_data,
                                    q_output_weight.rows, q_output_weight.cols,
                                    q_output_weight.row_stride_bytes, 0.01f);
            if (sparse_output_.valid) {
                use_sparse_gemv_ = true;
            }
        }

        // 3. Sparse GEMV for FFN layers
        sparse_ffn_.resize(config.num_layers * 3);
        int64_t total_sparse_blocks = 0, total_blocks = 0;
        for (int64_t i = 0; i < config.num_layers; ++i) {
            auto& layer = layers[i];
            auto try_sparse = [&](const QuantizedWeight& qw, SparseQ4KWeight& sw) {
                if (qw.valid && qw.cpu_data && qw.is_q4k()) {
                    sw.analyze(qw.cpu_data, qw.rows, qw.cols, qw.row_stride_bytes, 0.005f);
                }
            };
            try_sparse(layer.q_ffn_gate, sparse_ffn_[i * 3 + 0]);
            try_sparse(layer.q_ffn_up,   sparse_ffn_[i * 3 + 1]);
            try_sparse(layer.q_ffn_down,  sparse_ffn_[i * 3 + 2]);
        }
    }

    // ========================================================================
    // Enable/disable sliding window attention
    // ========================================================================
    void enable_sliding_window(int64_t window_size) {
        sliding_window_size_ = window_size;
        if (window_size > 0) {
            sliding_window_.init(config.num_layers, config.num_kv_heads,
                                  config.head_dim, window_size);
            std::cout << "[Accel] Sliding window attention enabled (window="
                      << window_size << ")" << std::endl;
        }
    }

    // ========================================================================
    // Load quantized weights via memory-mapping (ZERO-COPY)
    // Instead of malloc + memcpy, points cpu_data directly into mmap'd file.
    // Load time: ~0s (OS pages in on demand). RAM: only accessed pages.
    // ========================================================================

    void load_quantized_mmap(gguf::GGUFReader& reader) {
        if (gguf_file_path_.empty()) {
            std::cerr << "[mmap] No GGUF file path stored — falling back to read" << std::endl;
            load_quantized_to_cpu();
            return;
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        std::cout << "[mmap] Memory-mapping quantized weights (zero-copy)..." << std::endl;

        // Open mmap handle (kept alive in GGUFModel for lifetime of weights)
        if (!mmap_handle_.open(gguf_file_path_)) {
            std::cerr << "[mmap] Failed to mmap file — falling back to read-based loading" << std::endl;
            load_quantized_to_cpu();
            return;
        }

        std::cout << "[mmap] Mapped " << (mmap_handle_.size() / (1024 * 1024)) << " MB" << std::endl;

        // Setup mmap on the reader too (for get_tensor_data_ptr)
        reader.mmap_file();

        auto map_quant_mmap = [&](const std::string& name, QuantizedWeight& qw) {
            if (!reader.has_tensor(name)) return;
            const auto& info = reader.get_tensor_info(name);
            uint32_t type = info.type;
            int64_t block_bytes = 0;
            int64_t group_size = 256;  // QK_K for K-quants
            if (type == gguf::GGML_TYPE_Q4_K) block_bytes = 144;
            else if (type == gguf::GGML_TYPE_Q6_K) block_bytes = 210;
            else if (type == gguf::GGML_TYPE_Q5_K) block_bytes = 176;
            else if (type == gguf::GGML_TYPE_Q8_0) { block_bytes = 34; group_size = 32; }
            else return;

            auto shape = info.shape();
            qw.rows = shape[0];
            qw.cols = shape[1];
            qw.row_stride_bytes = (qw.cols / group_size) * block_bytes;
            qw.total_bytes = info.data_bytes();
            qw.quant_type = type;

            // ZERO-COPY: point directly into mmap'd region
            uint64_t abs_offset = reader.data_offset + info.offset;
            qw.cpu_data = const_cast<void*>(mmap_handle_.at_offset(abs_offset));
            qw.mmap_owned = true;  // Don't free — it's part of mmap
            qw.valid = (qw.cpu_data != nullptr);
        };

        int64_t total_bytes_mapped = 0;
        for (int64_t i = 0; i < config.num_layers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";
            auto& layer = layers[i];
            map_quant_mmap(prefix + "attn_q.weight", layer.q_attn_q);
            map_quant_mmap(prefix + "attn_k.weight", layer.q_attn_k);
            map_quant_mmap(prefix + "attn_v.weight", layer.q_attn_v);
            map_quant_mmap(prefix + "attn_output.weight", layer.q_attn_output);
            map_quant_mmap(prefix + "ffn_gate.weight", layer.q_ffn_gate);
            map_quant_mmap(prefix + "ffn_up.weight", layer.q_ffn_up);
            map_quant_mmap(prefix + "ffn_down.weight", layer.q_ffn_down);

            // Sum up mapped bytes for reporting
            if (layer.q_attn_q.valid) total_bytes_mapped += layer.q_attn_q.total_bytes;
            if (layer.q_attn_k.valid) total_bytes_mapped += layer.q_attn_k.total_bytes;
            if (layer.q_attn_v.valid) total_bytes_mapped += layer.q_attn_v.total_bytes;
            if (layer.q_attn_output.valid) total_bytes_mapped += layer.q_attn_output.total_bytes;
            if (layer.q_ffn_gate.valid) total_bytes_mapped += layer.q_ffn_gate.total_bytes;
            if (layer.q_ffn_up.valid) total_bytes_mapped += layer.q_ffn_up.total_bytes;
            if (layer.q_ffn_down.valid) total_bytes_mapped += layer.q_ffn_down.total_bytes;
        }

        if (reader.has_tensor("output.weight")) {
            map_quant_mmap("output.weight", q_output_weight);
            if (q_output_weight.valid) total_bytes_mapped += q_output_weight.total_bytes;
        }

        use_quant_gemv_ = true;
        use_mmap_ = true;

        // Lock critical weights in RAM: output projection + first/last layer norms
        // These are always accessed and should never be paged out
        lock_critical_weights();

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[mmap] " << (total_bytes_mapped / (1024 * 1024)) << " MB mapped in "
                  << std::fixed << std::setprecision(1) << ms << " ms (zero-copy)" << std::endl;
    }

    // ========================================================================
    // Lock critical weights in physical RAM (mlock/VirtualLock)
    // Prevents OS from paging out always-needed weights:
    //   - Output projection (used every token)
    //   - First layer norms (always first to be accessed)
    //   - Last layer norms (always last before output)
    // ========================================================================

    void lock_critical_weights() {
        if (!use_mmap_ || !mmap_handle_.is_open()) return;

        size_t locked_bytes = 0;

        // Output projection — used for every single token
        if (q_output_weight.valid && q_output_weight.cpu_data && q_output_weight.mmap_owned) {
#ifdef _WIN32
            if (VirtualLock(const_cast<void*>(q_output_weight.cpu_data),
                           static_cast<SIZE_T>(q_output_weight.total_bytes))) {
                locked_bytes += q_output_weight.total_bytes;
            }
#else
            if (mlock(q_output_weight.cpu_data, q_output_weight.total_bytes) == 0) {
                locked_bytes += q_output_weight.total_bytes;
            }
#endif
        }

        // First layer attention weights (always first to be touched)
        if (config.num_layers > 0) {
            auto& first = layers[0];
            QuantizedWeight* first_weights[] = {
                &first.q_attn_q, &first.q_attn_k, &first.q_attn_v
            };
            for (auto* qw : first_weights) {
                if (qw->valid && qw->cpu_data && qw->mmap_owned) {
#ifdef _WIN32
                    if (VirtualLock(const_cast<void*>(qw->cpu_data),
                                   static_cast<SIZE_T>(qw->total_bytes))) {
                        locked_bytes += qw->total_bytes;
                    }
#else
                    if (mlock(qw->cpu_data, qw->total_bytes) == 0) {
                        locked_bytes += qw->total_bytes;
                    }
#endif
                }
            }
        }

        if (locked_bytes > 0) {
            std::cout << "[mmap] Locked " << (locked_bytes / (1024 * 1024))
                      << " MB of critical weights in RAM" << std::endl;
        }
    }

    // ========================================================================
    // Load from GGUF file path
    // ========================================================================

    void load(const std::string& gguf_path) {
        gguf_file_path_ = gguf_path;  // Save for later quantized loading
        auto t_start = std::chrono::high_resolution_clock::now();

        gguf::GGUFReader reader;
        reader.open(gguf_path);

        // Parse config from metadata
        config.parse(reader);

        // Load tokenizer
        tokenizer.load(reader);

        // Determine vocab size from embedding tensor shape
        if (reader.has_tensor("token_embd.weight")) {
            auto& info = reader.get_tensor_info("token_embd.weight");
            auto shape = info.shape();
            config.vocab_size = shape[0];
        }

        config.print();

        // Load FP32 weights (norms, embeddings)
        load_weights(reader);

        // Quantized weight loading: prefer mmap (zero-copy) over malloc+memcpy
        // mmap: ~0ms load time, lazy paging, shared across processes
        // fallback: malloc + memcpy (~9s for 2.5GB)
        load_quantized_mmap(reader);

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "\n[Model] Loaded in " << (ms / 1000.0) << " seconds" << std::endl;
    }

    // ========================================================================
    // Load from Ollama model name
    // ========================================================================

    void load_ollama(const std::string& model_name) {
        std::string path = ollama::resolve_model(model_name);
        load(path);
    }

    // ========================================================================
    // Forward pass: tokens → logits
    // Input: token IDs [seq_len]
    // Output: logits [seq_len, vocab_size] (or [1, vocab_size] with KV cache)
    // ========================================================================

    Tensor forward(const std::vector<int64_t>& tokens, bool use_cache = false) {
        int64_t seq_len = static_cast<int64_t>(tokens.size());
        int64_t H = config.hidden_size;

        // 1. Token embedding lookup
        PROF_BEGIN(profiler, "embedding");
        Tensor x = embedding_lookup(tokens);  // [seq_len, hidden]
        PROF_END(profiler, "embedding");

        // Scale embeddings (Gemma)
        if (config.scale_embeddings) {
            float scale = std::sqrt(static_cast<float>(H));
#ifdef PT_USE_CUDA
            if (use_cuda_) {
                x = at::cuda_ops::mul_scalar(x, scale);
            } else
#endif
            {
                Tensor scaled = at::empty(x.sizes().vec());
                const float* src = x.data_ptr<float>();
                float* dst = scaled.mutable_data_ptr<float>();
                for (int64_t i = 0; i < x.numel(); ++i) {
                    dst[i] = src[i] * scale;
                }
                x = scaled;
            }
        }

        // 2. Transformer layers
        int64_t past_len = use_cache ? kv_cache.seq_len : 0;

        for (int64_t i = 0; i < config.num_layers; ++i) {
            x = transformer_layer(x, i, past_len, use_cache);
        }

        // 3. Final RMS norm
        PROF_BEGIN(profiler, "final_norm");
        x = rms_norm(x, output_norm, config.rms_norm_eps);
        PROF_END(profiler, "final_norm");

        // 4. Output projection → logits
        // For prefill: only project LAST position (saves seq_len-1 × vocab matmuls)
        Tensor x_last;
        if (seq_len > 1) {
            x_last = x.select(0, seq_len - 1).unsqueeze(0);  // [1, hidden]
        } else {
            x_last = x;
        }
        PROF_BEGIN(profiler, "output_proj");
        Tensor logits = matmul_q(x_last, output_weight, q_output_weight, true);
        PROF_END(profiler, "output_proj");

        if (use_cache) {
            kv_cache.seq_len += seq_len;
        }

        return logits;
    }

    // ========================================================================
    // Zero-allocation decode forward (single token, CUDA only)
    // Uses pre-allocated scratch buffers — no cudaMalloc during decode.
    // ========================================================================

#ifdef PT_USE_CUDA
    Tensor forward_decode(int64_t token_id) {
        if (!scratch_.allocated) {
            scratch_.allocate(config);
        }

        auto& sp = scratch_;
        int64_t H = config.hidden_size;
        int64_t q_dim = config.num_heads * config.head_dim;
        int64_t kv_dim = config.num_kv_heads * config.head_dim;
        int64_t n_heads = config.num_heads;
        int64_t n_kv_heads = config.num_kv_heads;
        int64_t head_dim = config.head_dim;
        int64_t inter = config.intermediate_size;
        int64_t past_len = kv_cache.seq_len;
        float eps = config.rms_norm_eps;
        bool add_one = config.gemma_norm_add_one;

        // 0. Initialize device past_len pointer (once) + update each token
        if (!d_past_len_) {
            cudaMalloc(&d_past_len_, sizeof(int64_t));
        }
        cudaMemcpyAsync(d_past_len_, &past_len, sizeof(int64_t), cudaMemcpyHostToDevice, nullptr);

        // 1. Embedding: GPU lookup (zero H2D copy for CUDA Graph compatibility)
        PROF_BEGIN(profiler, "embedding");
        if (!emb_gpu_.defined() && token_embedding.defined()) {
            // First call: copy embedding table to GPU (once)
            emb_gpu_ = at::empty_cuda({config.vocab_size, H});
            cudaMemcpy(emb_gpu_.mutable_data_ptr<float>(), token_embedding.data_ptr<float>(),
                       config.vocab_size * H * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (emb_gpu_.defined()) {
            // GPU→GPU copy (stays on device, CUDA Graph friendly)
            cudaMemcpyAsync(sp.buf_x[0].mutable_data_ptr<float>(),
                       emb_gpu_.data_ptr<float>() + token_id * H,
                       H * sizeof(float), cudaMemcpyDeviceToDevice, nullptr);
        } else {
            // Fallback: CPU embedding
            const float* emb_table = token_embedding.data_ptr<float>();
            cudaMemcpy(sp.buf_x[0].mutable_data_ptr<float>(),
                       emb_table + token_id * H, H * sizeof(float), cudaMemcpyHostToDevice);
        }
        PROF_END(profiler, "embedding");

        // Scale embeddings (Gemma)
        if (config.scale_embeddings) {
            float scale = std::sqrt(static_cast<float>(H));
            at::cuda::launch_mul_scalar(sp.buf_x[0].data_ptr<float>(), scale,
                                         sp.buf_x[0].mutable_data_ptr<float>(), H, nullptr);
        }

        int cur = 0; // which buf_x holds current hidden state

        // 2. Transformer layers

        for (int64_t i = 0; i < config.num_layers; ++i) {
            auto& layer = layers[i];
            float* x_ptr = sp.buf_x[cur].mutable_data_ptr<float>();

            // Check if all QKV weights are Q4_K with same K and stride (for fused path)
            bool can_fuse_qkv = use_quant_gemv_ &&
                layer.q_attn_q.valid && layer.q_attn_k.valid && layer.q_attn_v.valid &&
                layer.q_attn_q.is_q4k() && layer.q_attn_k.is_q4k() && layer.q_attn_v.is_q4k() &&
                layer.q_attn_q.gpu_data && layer.q_attn_k.gpu_data && layer.q_attn_v.gpu_data &&
                layer.q_attn_q.cols == layer.q_attn_k.cols &&
                layer.q_attn_q.cols == layer.q_attn_v.cols &&
                layer.q_attn_q.row_stride_bytes == layer.q_attn_k.row_stride_bytes &&
                layer.q_attn_q.row_stride_bytes == layer.q_attn_v.row_stride_bytes;

            if (can_fuse_qkv) {
                // -- FUSED: attn_norm + Q/K/V projections (1 kernel instead of 4) --
                PROF_BEGIN(profiler, "fused_norm_qkv");
                at::cuda::launch_q4km_fused_rmsnorm_qkv_gemv(
                    x_ptr, layer.attn_norm.data_ptr<float>(),
                    layer.q_attn_q.gpu_data, layer.q_attn_k.gpu_data, layer.q_attn_v.gpu_data,
                    sp.buf_q.mutable_data_ptr<float>(),
                    sp.buf_k.mutable_data_ptr<float>(),
                    sp.buf_v.mutable_data_ptr<float>(),
                    static_cast<int>(H),
                    static_cast<int>(layer.q_attn_q.rows),
                    static_cast<int>(layer.q_attn_k.rows),
                    static_cast<int>(layer.q_attn_v.rows),
                    layer.q_attn_q.row_stride_bytes,
                    eps, add_one, nullptr);
                PROF_END(profiler, "fused_norm_qkv");
            } else {
                // Fallback: separate attn_norm + 3 GEMVs
                PROF_BEGIN(profiler, "attn_norm");
                at::cuda::launch_rms_norm(
                    x_ptr, layer.attn_norm.data_ptr<float>(),
                    sp.buf_normed.mutable_data_ptr<float>(),
                    1, static_cast<int>(H), eps, add_one, nullptr);
                PROF_END(profiler, "attn_norm");

                const float* normed_ptr = sp.buf_normed.data_ptr<float>();

                PROF_BEGIN(profiler, "qkv_proj");
                gemv_scratch(layer.q_attn_q, layer.attn_q, normed_ptr,
                            sp.buf_q.mutable_data_ptr<float>(), q_dim);
                gemv_scratch(layer.q_attn_k, layer.attn_k, normed_ptr,
                            sp.buf_k.mutable_data_ptr<float>(), kv_dim);
                gemv_scratch(layer.q_attn_v, layer.attn_v, normed_ptr,
                            sp.buf_v.mutable_data_ptr<float>(), kv_dim);
                PROF_END(profiler, "qkv_proj");
            }

            // -- Biases (Qwen3 has Q/K/V biases) --
            if (layer.attn_q_bias.defined()) {
                at::cuda::launch_add(
                    sp.buf_q.data_ptr<float>(), layer.attn_q_bias.data_ptr<float>(),
                    sp.buf_q.mutable_data_ptr<float>(), q_dim, nullptr);
                at::cuda::launch_add(
                    sp.buf_k.data_ptr<float>(), layer.attn_k_bias.data_ptr<float>(),
                    sp.buf_k.mutable_data_ptr<float>(), kv_dim, nullptr);
                at::cuda::launch_add(
                    sp.buf_v.data_ptr<float>(), layer.attn_v_bias.data_ptr<float>(),
                    sp.buf_v.mutable_data_ptr<float>(), kv_dim, nullptr);
            }

            // -- Fused QK-norm + RoPE + KV cache write --
            // Use graph-compatible version with device pointer past_len
            PROF_BEGIN(profiler, "fused_qknorm_rope_kv");
            if (d_past_len_) {
                at::cuda::launch_fused_qknorm_rope_kvwrite_graph(
                    sp.buf_q.mutable_data_ptr<float>(),
                    sp.buf_k.mutable_data_ptr<float>(),
                    sp.buf_v.data_ptr<float>(),
                    layer.attn_q_norm.defined() ? layer.attn_q_norm.data_ptr<float>() : nullptr,
                    layer.attn_q_norm.defined() ? layer.attn_k_norm.data_ptr<float>() : nullptr,
                    kv_cache.key_cache[i].mutable_data_ptr<float>(),
                    kv_cache.value_cache[i].mutable_data_ptr<float>(),
                    static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                    static_cast<int>(head_dim),
                    d_past_len_, config.rope_freq_base,
                    eps, add_one, nullptr);
            } else {
                at::cuda::launch_fused_qknorm_rope_kvwrite(
                    sp.buf_q.mutable_data_ptr<float>(),
                    sp.buf_k.mutable_data_ptr<float>(),
                    sp.buf_v.data_ptr<float>(),
                    layer.attn_q_norm.defined() ? layer.attn_q_norm.data_ptr<float>() : nullptr,
                    layer.attn_q_norm.defined() ? layer.attn_k_norm.data_ptr<float>() : nullptr,
                    kv_cache.key_cache[i].mutable_data_ptr<float>(),
                    kv_cache.value_cache[i].mutable_data_ptr<float>(),
                    static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                    static_cast<int>(head_dim),
                    static_cast<int>(past_len), config.rope_freq_base,
                    eps, add_one, past_len, nullptr);
            }
            PROF_END(profiler, "fused_qknorm_rope_kv");

            // Also write K/V to FP16 cache for attention bandwidth reduction
            if (kv_cache.use_fp16_kv) {
                at::cuda::launch_fp16_kv_cache_write(
                    kv_cache.key_cache[i].data_ptr<float>() + past_len * kv_dim,
                    kv_cache.key_cache_fp16[i],
                    1, kv_dim, past_len, nullptr);
                at::cuda::launch_fp16_kv_cache_write(
                    kv_cache.value_cache[i].data_ptr<float>() + past_len * kv_dim,
                    kv_cache.value_cache_fp16[i],
                    1, kv_dim, past_len, nullptr);
            }

            int64_t total_seq = past_len + 1;
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // -- Flash-Decode attention: FP16 KV cache if available --
            PROF_BEGIN(profiler, "flash_decode");
            if (kv_cache.use_fp16_kv) {
                at::cuda::launch_flash_decode_fp16(
                    sp.buf_q.data_ptr<float>(),
                    kv_cache.key_cache_fp16[i],
                    kv_cache.value_cache_fp16[i],
                    sp.buf_attn.mutable_data_ptr<float>(),
                    sp.fd_partial_O, sp.fd_partial_lse, sp.fd_partial_max,
                    static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                    static_cast<int>(head_dim),
                    static_cast<int>(total_seq), scale, nullptr);
            } else {
                at::cuda::launch_flash_decode(
                    sp.buf_q.data_ptr<float>(),
                    kv_cache.key_cache[i].data_ptr<float>(),
                    kv_cache.value_cache[i].data_ptr<float>(),
                    sp.buf_attn.mutable_data_ptr<float>(),
                    sp.fd_partial_O, sp.fd_partial_lse, sp.fd_partial_max,
                    static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                    static_cast<int>(head_dim),
                    static_cast<int>(total_seq), scale, nullptr);
            }
            PROF_END(profiler, "flash_decode");

            // -- Output projection with fused residual add --
            // Instead of: output_proj → buf_attn_proj, then add(x, buf_attn_proj) → buf_h
            // We do: copy x → buf_h, then accumulate: buf_h += W @ attn_out
            // This saves 1 kernel launch (residual_add) per layer.
            bool can_fuse_output_residual = use_quant_gemv_ &&
                layer.q_attn_output.valid && layer.q_attn_output.is_q4k() &&
                layer.q_attn_output.gpu_data &&
                !layer.post_attention_norm.defined();  // Can't fuse if post-norm needed

            if (can_fuse_output_residual) {
                PROF_BEGIN(profiler, "fused_output_residual");
                // Copy x → buf_h (residual base)
                at::cuda::launch_copy(x_ptr, sp.buf_h.mutable_data_ptr<float>(), H, nullptr);
                // buf_h += W_output @ attn_out (accumulate)
                at::cuda::launch_q4km_persistent_gemv_accumulate(
                    layer.q_attn_output.gpu_data,
                    sp.buf_attn.data_ptr<float>(),
                    sp.buf_h.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_attn_output.cols),
                    static_cast<int>(layer.q_attn_output.rows),
                    layer.q_attn_output.row_stride_bytes, nullptr);
                PROF_END(profiler, "fused_output_residual");
            } else {
                // Fallback: separate output_proj + residual_add
                PROF_BEGIN(profiler, "attn_output_proj");
                gemv_scratch(layer.q_attn_output, layer.attn_output,
                            sp.buf_attn.data_ptr<float>(),
                            sp.buf_attn_proj.mutable_data_ptr<float>(), H);
                PROF_END(profiler, "attn_output_proj");

                if (layer.post_attention_norm.defined()) {
                    at::cuda::launch_rms_norm(
                        sp.buf_attn_proj.data_ptr<float>(),
                        layer.post_attention_norm.data_ptr<float>(),
                        sp.buf_attn_proj.mutable_data_ptr<float>(),
                        1, static_cast<int>(H), eps, add_one, nullptr);
                }

                PROF_BEGIN(profiler, "residual_add");
                at::cuda::launch_add(x_ptr, sp.buf_attn_proj.data_ptr<float>(),
                                      sp.buf_h.mutable_data_ptr<float>(), H, nullptr);
                PROF_END(profiler, "residual_add");
            }

            // -- FFN: Check if we can fuse norm + gate+up --
            bool can_fuse_ffn = use_quant_gemv_ &&
                layer.q_ffn_gate.valid && layer.q_ffn_up.valid &&
                layer.q_ffn_gate.is_q4k() && layer.q_ffn_up.is_q4k() &&
                layer.q_ffn_gate.gpu_data && layer.q_ffn_up.gpu_data &&
                layer.q_ffn_gate.cols == layer.q_ffn_up.cols &&
                layer.q_ffn_gate.row_stride_bytes == layer.q_ffn_up.row_stride_bytes;

            if (can_fuse_ffn) {
                // -- FUSED: ffn_norm + gate+up projections (1 kernel instead of 3) --
                PROF_BEGIN(profiler, "fused_norm_gate_up");
                at::cuda::launch_q4km_fused_rmsnorm_gate_up_gemv(
                    sp.buf_h.data_ptr<float>(),
                    layer.ffn_norm.data_ptr<float>(),
                    layer.q_ffn_gate.gpu_data, layer.q_ffn_up.gpu_data,
                    sp.buf_gate.mutable_data_ptr<float>(),
                    sp.buf_up.mutable_data_ptr<float>(),
                    static_cast<int>(H),
                    static_cast<int>(layer.q_ffn_gate.rows),
                    static_cast<int>(layer.q_ffn_up.rows),
                    layer.q_ffn_gate.row_stride_bytes,
                    eps, add_one, nullptr);
                PROF_END(profiler, "fused_norm_gate_up");
            } else {
                // Fallback: separate ffn_norm + gate/up GEMVs
                PROF_BEGIN(profiler, "ffn_norm");
                at::cuda::launch_rms_norm(
                    sp.buf_h.data_ptr<float>(), layer.ffn_norm.data_ptr<float>(),
                    sp.buf_normed.mutable_data_ptr<float>(),
                    1, static_cast<int>(H), eps, add_one, nullptr);
                PROF_END(profiler, "ffn_norm");

                const float* normed_ptr = sp.buf_normed.data_ptr<float>();

                PROF_BEGIN(profiler, "ffn_gate_up");
                fused_gate_up_gemv(layer, normed_ptr,
                                  sp.buf_gate.mutable_data_ptr<float>(),
                                  sp.buf_up.mutable_data_ptr<float>(), inter);
                PROF_END(profiler, "ffn_gate_up");
            }

            // -- SiLU-Mul: silu(gate) * up → buf_silu --
            PROF_BEGIN(profiler, "silu_mul");
            at::cuda::launch_silu_mul(
                sp.buf_gate.data_ptr<float>(), sp.buf_up.data_ptr<float>(),
                sp.buf_silu.mutable_data_ptr<float>(), inter, nullptr);
            PROF_END(profiler, "silu_mul");

            // -- Down projection with fused residual add --
            bool can_fuse_down_residual = use_quant_gemv_ &&
                layer.q_ffn_down.valid && layer.q_ffn_down.is_q4k() &&
                layer.q_ffn_down.gpu_data &&
                !layer.post_ffw_norm.defined();  // Can't fuse if post-norm needed

            int next = 1 - cur;
            if (can_fuse_down_residual) {
                PROF_BEGIN(profiler, "fused_down_residual");
                // Copy buf_h → buf_x[next] (residual base)
                at::cuda::launch_copy(sp.buf_h.data_ptr<float>(),
                                       sp.buf_x[next].mutable_data_ptr<float>(), H, nullptr);
                // buf_x[next] += W_down @ silu_out (accumulate)
                at::cuda::launch_q4km_persistent_gemv_accumulate(
                    layer.q_ffn_down.gpu_data,
                    sp.buf_silu.data_ptr<float>(),
                    sp.buf_x[next].mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_ffn_down.cols),
                    static_cast<int>(layer.q_ffn_down.rows),
                    layer.q_ffn_down.row_stride_bytes, nullptr);
                PROF_END(profiler, "fused_down_residual");
            } else {
                // Fallback: separate down_proj + post-norm + residual
                PROF_BEGIN(profiler, "ffn_down");
                gemv_scratch(layer.q_ffn_down, layer.ffn_down,
                            sp.buf_silu.data_ptr<float>(),
                            sp.buf_down.mutable_data_ptr<float>(), H);
                PROF_END(profiler, "ffn_down");

                if (layer.post_ffw_norm.defined()) {
                    at::cuda::launch_rms_norm(
                        sp.buf_down.data_ptr<float>(),
                        layer.post_ffw_norm.data_ptr<float>(),
                        sp.buf_down.mutable_data_ptr<float>(),
                        1, static_cast<int>(H), eps, add_one, nullptr);
                }

                PROF_BEGIN(profiler, "residual_add");
                at::cuda::launch_add(sp.buf_h.data_ptr<float>(), sp.buf_down.data_ptr<float>(),
                                      sp.buf_x[next].mutable_data_ptr<float>(), H, nullptr);
                PROF_END(profiler, "residual_add");
            }
            cur = next;
        }

        // End transformer layers

        // 3. Final RMS norm: buf_x[cur] → buf_normed
        PROF_BEGIN(profiler, "final_norm");
        at::cuda::launch_rms_norm(
            sp.buf_x[cur].data_ptr<float>(), output_norm.data_ptr<float>(),
            sp.buf_normed.mutable_data_ptr<float>(),
            1, static_cast<int>(H), eps, add_one, nullptr);
        PROF_END(profiler, "final_norm");

        // 4. Output projection: buf_normed → buf_logits
        PROF_BEGIN(profiler, "output_proj");
        gemv_scratch(q_output_weight, output_weight,
                    sp.buf_normed.data_ptr<float>(),
                    sp.buf_logits.mutable_data_ptr<float>(), config.vocab_size);
        PROF_END(profiler, "output_proj");

        kv_cache.seq_len += 1;
        return sp.buf_logits;
    }

    // GEMV helper for scratch pool: dispatch persistent > quant > float32
    // Uses persistent kernel for Q4_K to reduce launch overhead (grid-stride over rows).
    void gemv_scratch(const QuantizedWeight& qw, const Tensor& float_w,
                      const float* x, float* y, int64_t N) {
        if (use_quant_gemv_ && qw.valid) {
            int K = static_cast<int>(qw.cols);
            int Nr = static_cast<int>(qw.rows);
            if (qw.is_q4k() && qw.gpu_data) {
                at::cuda::launch_q4km_persistent_gemv(qw.gpu_data, x, y, K, Nr, qw.row_stride_bytes, nullptr);
            } else if (qw.is_q6k() && qw.gpu_data) {
                at::cuda::launch_q6k_gemv(qw.gpu_data, x, y, K, Nr, qw.row_stride_bytes, nullptr);
            } else if (qw.is_q5k() && qw.gpu_data) {
                at::cuda::launch_q5k_gemv(qw.gpu_data, x, y, K, Nr, qw.row_stride_bytes, nullptr);
            }
        } else if (float_w.defined()) {
            int K = static_cast<int>(float_w.size(0));
            int Ni = static_cast<int>(float_w.size(1));
            at::cuda::launch_inference_gemv(x, float_w.data_ptr<float>(), y, K, Ni, nullptr);
        }
    }

    // Fused gate+up GEMV: single kernel launch for both gate and up projections.
    // Saves one kernel launch per layer (~28 launches saved for 28-layer model).
    void fused_gate_up_gemv(const TransformerLayer& layer,
                            const float* x, float* y_gate, float* y_up,
                            int64_t inter) {
        const auto& qg = layer.q_ffn_gate;
        const auto& qu = layer.q_ffn_up;
        // Both must be Q4_K with same K and row_stride for fused kernel
        if (use_quant_gemv_ && qg.valid && qu.valid &&
            qg.is_q4k() && qu.is_q4k() && qg.gpu_data && qu.gpu_data &&
            qg.cols == qu.cols && qg.row_stride_bytes == qu.row_stride_bytes) {
            int K = static_cast<int>(qg.cols);
            int N_gate = static_cast<int>(qg.rows);
            int N_up = static_cast<int>(qu.rows);
            at::cuda::launch_q4km_fused_gate_up_gemv(
                qg.gpu_data, qu.gpu_data, x, y_gate, y_up,
                K, N_gate, N_up, qg.row_stride_bytes, nullptr);
        } else {
            // Fallback: two separate GEMVs
            gemv_scratch(qg, layer.ffn_gate, x, y_gate, inter);
            gemv_scratch(qu, layer.ffn_up, x, y_up, inter);
        }
    }
#endif

    // ========================================================================
    // Zero-allocation CPU decode forward (single token)
    //
    // Optimizations vs generic forward():
    //   1. Zero allocations: all buffers pre-allocated in CPUScratchPool
    //   2. Fused RMSNorm + QKV GEMV: RMSNorm output stays in L1, shared x
    //   3. Fused RMSNorm + gate+up GEMV: same fusion for FFN
    //   4. Prefetch next layer weights while current layer computes
    //   5. Optimized single-token attention: no causal mask needed
    //   6. In-place residual adds: no temp tensors
    //   7. In-place SiLU-mul: gate_buf modified in place
    //
    // Expected speedup: ~1.5-2x over generic forward() on CPU
    // ========================================================================

    Tensor forward_decode_cpu(int64_t token_id) {
        int64_t H = config.hidden_size;
        int64_t q_dim = config.num_heads * config.head_dim;
        int64_t kv_dim = config.num_kv_heads * config.head_dim;
        int64_t n_heads = config.num_heads;
        int64_t n_kv_heads = config.num_kv_heads;
        int64_t head_dim = config.head_dim;
        int64_t inter = config.intermediate_size;
        int64_t past_len = kv_cache.seq_len;
        float eps = config.rms_norm_eps;
        bool add_one = config.gemma_norm_add_one;
        int64_t heads_per_group = n_heads / n_kv_heads;

        // Allocate scratch if needed
        if (!cpu_scratch_.allocated) {
            int64_t max_seq = kv_cache.max_seq > 0 ? kv_cache.max_seq : 4096;
            cpu_scratch_.allocate(config, max_seq);
        }

        auto& sp = cpu_scratch_;
        int cur = 0;  // which x_buf holds current hidden state

        // 1. Embedding lookup directly into scratch buffer
        const float* emb_table = token_embedding.data_ptr<float>();
        std::memcpy(sp.x_buf[cur], emb_table + token_id * H, H * sizeof(float));

        // Scale embeddings (Gemma)
        if (config.scale_embeddings) {
            float scale = std::sqrt(static_cast<float>(H));
#ifdef __AVX2__
            __m256 vscale = _mm256_set1_ps(scale);
            int64_t j = 0;
            for (; j + 7 < H; j += 8) {
                __m256 vx = _mm256_loadu_ps(sp.x_buf[cur] + j);
                _mm256_storeu_ps(sp.x_buf[cur] + j, _mm256_mul_ps(vx, vscale));
            }
            for (; j < H; ++j) sp.x_buf[cur][j] *= scale;
#else
            for (int64_t j = 0; j < H; ++j) sp.x_buf[cur][j] *= scale;
#endif
        }

        // 2. Transformer layers
        for (int64_t i = 0; i < config.num_layers; ++i) {
            auto& layer = layers[i];
            float* x_ptr = sp.x_buf[cur];

            // -- Async prefetch: next layer's weights into L2 cache + TLB --
            // With mmap, this is critical: triggers page faults BEFORE we need the data.
            // Prefetches first 64KB of each weight matrix (fills ~16 TLB entries per weight).
            // Without this, mmap'd first access to each layer takes ~0.5ms in page faults.
#if defined(__AVX2__) || defined(_MSC_VER)
            if (i + 1 < config.num_layers) {
                auto& next = layers[i + 1];
                // All 7 weight matrices in next layer
                const void* next_ptrs[] = {
                    next.q_attn_q.cpu_data,
                    next.q_attn_k.cpu_data,
                    next.q_attn_v.cpu_data,
                    next.q_attn_output.cpu_data,
                    next.q_ffn_gate.cpu_data,
                    next.q_ffn_up.cpu_data,
                    next.q_ffn_down.cpu_data,
                };
                for (const void* p : next_ptrs) {
                    if (p) {
                        const char* cp = static_cast<const char*>(p);
                        // Prefetch first 64KB — fills TLB entries, hides page fault latency
                        // _MM_HINT_T1 = L2 cache (won't pollute L1 with data not yet needed)
                        for (int off = 0; off < 65536; off += 4096) {
                            _mm_prefetch(cp + off, _MM_HINT_T1);
                        }
                    }
                }
                // Prefetch norm weights into L1 (small: ~14KB, will be used very soon)
                if (next.attn_norm.defined()) {
                    const char* np = reinterpret_cast<const char*>(next.attn_norm.data_ptr<float>());
                    int64_t norm_bytes = next.attn_norm.numel() * sizeof(float);
                    for (int64_t off = 0; off < norm_bytes; off += 64) {
                        _mm_prefetch(np + off, _MM_HINT_T0);
                    }
                }
                if (next.ffn_norm.defined()) {
                    const char* np = reinterpret_cast<const char*>(next.ffn_norm.data_ptr<float>());
                    int64_t norm_bytes = next.ffn_norm.numel() * sizeof(float);
                    for (int64_t off = 0; off < norm_bytes; off += 64) {
                        _mm_prefetch(np + off, _MM_HINT_T0);
                    }
                }
            }
#endif

            // -- Check if we can use fused RMSNorm + batched QKV GEMV --
            bool can_fuse = use_quant_gemv_ &&
                layer.q_attn_q.valid && layer.q_attn_k.valid && layer.q_attn_v.valid &&
                layer.q_attn_q.cpu_data && layer.q_attn_k.cpu_data && layer.q_attn_v.cpu_data &&
                cpu_quant::cpu_quant_gemv_supported(layer.q_attn_q.quant_type) &&
                layer.q_attn_q.quant_type == layer.q_attn_k.quant_type &&
                layer.q_attn_q.quant_type == layer.q_attn_v.quant_type &&
                layer.q_attn_q.cols == layer.q_attn_k.cols &&
                layer.q_attn_q.cols == layer.q_attn_v.cols &&
                layer.q_attn_q.row_stride_bytes == layer.q_attn_k.row_stride_bytes;

            if (can_fuse) {
                // FUSED: RMSNorm + QKV projection (1 RMSNorm, shared x across 3 GEMVs)
                cpu_quant::cpu_fused_rmsnorm_qkv_gemv(
                    x_ptr, layer.attn_norm.data_ptr<float>(), eps, add_one,
                    layer.q_attn_q.quant_type,
                    layer.q_attn_q.cpu_data, layer.q_attn_k.cpu_data, layer.q_attn_v.cpu_data,
                    sp.q_buf, sp.k_buf, sp.v_buf,
                    H, layer.q_attn_q.rows, layer.q_attn_k.rows, layer.q_attn_v.rows,
                    layer.q_attn_q.row_stride_bytes);
            } else {
                // Fallback: separate RMSNorm + 3 GEMVs
                // RMSNorm into q_buf as temp (reuse buffer)
                float norm_buf[8192];  // stack buffer for normalized x
                float* x_normed = (H <= 8192) ? norm_buf
                    : static_cast<float*>(std::malloc(H * sizeof(float)));

                // Compute RMSNorm
                float sum_sq = 0.0f;
                for (int64_t j = 0; j < H; ++j) sum_sq += x_ptr[j] * x_ptr[j];
                float rms = 1.0f / std::sqrt(sum_sq / H + eps);
                const float* gamma = layer.attn_norm.data_ptr<float>();
                for (int64_t j = 0; j < H; ++j) {
                    float w = add_one ? (1.0f + gamma[j]) : gamma[j];
                    x_normed[j] = x_ptr[j] * rms * w;
                }

                // Q, K, V projections
                if (layer.q_attn_q.valid && layer.q_attn_q.cpu_data) {
                    cpu_quant::cpu_quant_gemv(layer.q_attn_q.quant_type, layer.q_attn_q.cpu_data,
                        x_normed, sp.q_buf, H, layer.q_attn_q.rows, layer.q_attn_q.row_stride_bytes);
                }
                if (layer.q_attn_k.valid && layer.q_attn_k.cpu_data) {
                    cpu_quant::cpu_quant_gemv(layer.q_attn_k.quant_type, layer.q_attn_k.cpu_data,
                        x_normed, sp.k_buf, H, layer.q_attn_k.rows, layer.q_attn_k.row_stride_bytes);
                }
                if (layer.q_attn_v.valid && layer.q_attn_v.cpu_data) {
                    cpu_quant::cpu_quant_gemv(layer.q_attn_v.quant_type, layer.q_attn_v.cpu_data,
                        x_normed, sp.v_buf, H, layer.q_attn_v.rows, layer.q_attn_v.row_stride_bytes);
                }

                if (H > 8192) std::free(x_normed);
            }

            // -- Add biases if present (Qwen3) --
            if (layer.attn_q_bias.defined()) {
                const float* bq = layer.attn_q_bias.data_ptr<float>();
                const float* bk = layer.attn_k_bias.data_ptr<float>();
                const float* bv = layer.attn_v_bias.data_ptr<float>();
#ifdef __AVX2__
                for (int64_t j = 0; j + 7 < q_dim; j += 8) {
                    _mm256_storeu_ps(sp.q_buf + j,
                        _mm256_add_ps(_mm256_loadu_ps(sp.q_buf + j), _mm256_loadu_ps(bq + j)));
                }
                for (int64_t j = (q_dim / 8) * 8; j < q_dim; ++j) sp.q_buf[j] += bq[j];
                for (int64_t j = 0; j + 7 < kv_dim; j += 8) {
                    _mm256_storeu_ps(sp.k_buf + j,
                        _mm256_add_ps(_mm256_loadu_ps(sp.k_buf + j), _mm256_loadu_ps(bk + j)));
                    _mm256_storeu_ps(sp.v_buf + j,
                        _mm256_add_ps(_mm256_loadu_ps(sp.v_buf + j), _mm256_loadu_ps(bv + j)));
                }
                for (int64_t j = (kv_dim / 8) * 8; j < kv_dim; ++j) {
                    sp.k_buf[j] += bk[j];
                    sp.v_buf[j] += bv[j];
                }
#else
                for (int64_t j = 0; j < q_dim; ++j) sp.q_buf[j] += bq[j];
                for (int64_t j = 0; j < kv_dim; ++j) { sp.k_buf[j] += bk[j]; sp.v_buf[j] += bv[j]; }
#endif
            }

            // -- QK-norm (Qwen3, Gemma3): per-head RMSNorm --
            if (layer.attn_q_norm.defined()) {
                const float* qn_w = layer.attn_q_norm.data_ptr<float>();
                const float* kn_w = layer.attn_k_norm.data_ptr<float>();
                for (int64_t h = 0; h < n_heads; ++h) {
                    cpu_quant::cpu_rmsnorm_inplace(sp.q_buf + h * head_dim, qn_w, eps, add_one, head_dim);
                }
                for (int64_t h = 0; h < n_kv_heads; ++h) {
                    cpu_quant::cpu_rmsnorm_inplace(sp.k_buf + h * head_dim, kn_w, eps, add_one, head_dim);
                }
            }

            // -- RoPE in-place on Q, K (FUSED: precompute cos/sin ONCE, reuse for all heads) --
            // Before: (n_heads + n_kv_heads) * head_dim/2 calls to pow() + cos() + sin()
            // After:  head_dim/2 calls to pow() + cos() + sin(), then simple FMA multiply
            // Speedup on trig: ~36x for qwen3:4b (32 Q heads + 4 KV heads share same table)
            {
                float rope_cos[256], rope_sin[256];  // head_dim/2 <= 256 for all models
                at::native::hot::rope_precompute(rope_cos, rope_sin,
                    past_len, head_dim, config.rope_freq_base);
                at::native::hot::rope_apply_fused(sp.q_buf, sp.k_buf,
                    rope_cos, rope_sin, n_heads, n_kv_heads, head_dim);
            }

            // -- KV cache append (zero-copy: write directly into cache) --
            {
                float* k_cache = kv_cache.key_cache[i].mutable_data_ptr<float>();
                float* v_cache = kv_cache.value_cache[i].mutable_data_ptr<float>();
                std::memcpy(k_cache + past_len * kv_dim, sp.k_buf, kv_dim * sizeof(float));
                std::memcpy(v_cache + past_len * kv_dim, sp.v_buf, kv_dim * sizeof(float));
            }

            // -- Single-token attention (optimized: no causal mask needed) --
            // For seq_len=1: Q is [1, q_dim], K is [total_seq, kv_dim], V is [total_seq, kv_dim]
            // For each head: score = Q_h . K_h[t] for all t, then softmax, then weighted V sum
            // With sliding window: only attend to last window_size positions + summary
            {
                int64_t total_seq = past_len + 1;
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                const float* k_cache = kv_cache.key_cache[i].data_ptr<float>();
                const float* v_cache = kv_cache.value_cache[i].data_ptr<float>();

                // Update sliding window state (summarize evicted positions)
                if (sliding_window_.enabled) {
                    sliding_window_.update_window(i, total_seq,
                        k_cache, v_cache, n_kv_heads, head_dim);
                }

                // Parallel over heads
                c10::get_thread_pool().parallel_for(0, n_heads, [&](int64_t h_start, int64_t h_end) {
                for (int64_t h = h_start; h < h_end; ++h) {
                    int64_t kv_h = h / heads_per_group;
                    const float* q_head = sp.q_buf + h * head_dim;
                    float* out_head = sp.attn_buf + h * head_dim;

                    // Thread-local scores buffer
                    float local_scores[4096];
                    float* scores = (total_seq <= 4096) ? local_scores : sp.scores_buf;

                    // Use sliding window attention if enabled and context exceeds window
                    if (sliding_window_.enabled && total_seq > sliding_window_.window_size) {
                        sliding_window_.compute_attention(
                            q_head, k_cache, v_cache, out_head,
                            total_seq, kv_h, head_dim, kv_dim,
                            i, scale, scores);
                    } else {
                    // Standard full attention
#ifdef __AVX2__
                    for (int64_t t = 0; t < total_seq; ++t) {
                        const float* k_head = k_cache + t * kv_dim + kv_h * head_dim;
                        __m256 dot_acc = _mm256_setzero_ps();
                        int64_t d = 0;
                        for (; d + 7 < head_dim; d += 8) {
                            dot_acc = _mm256_fmadd_ps(
                                _mm256_loadu_ps(q_head + d),
                                _mm256_loadu_ps(k_head + d), dot_acc);
                        }
                        float dot = cpu_quant::hsum_avx(dot_acc);
                        for (; d < head_dim; ++d) dot += q_head[d] * k_head[d];
                        scores[t] = dot * scale;
                    }
#else
                    for (int64_t t = 0; t < total_seq; ++t) {
                        const float* k_head = k_cache + t * kv_dim + kv_h * head_dim;
                        float dot = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * k_head[d];
                        scores[t] = dot * scale;
                    }
#endif

                    // Softmax (online: max + exp + normalize)
                    float max_score = scores[0];
                    for (int64_t t = 1; t < total_seq; ++t) {
                        if (scores[t] > max_score) max_score = scores[t];
                    }
                    float sum_exp = 0.0f;
                    for (int64_t t = 0; t < total_seq; ++t) {
                        scores[t] = std::exp(scores[t] - max_score);
                        sum_exp += scores[t];
                    }
                    float inv_sum = 1.0f / (sum_exp + 1e-10f);
                    for (int64_t t = 0; t < total_seq; ++t) scores[t] *= inv_sum;

                    // Weighted sum of V
                    std::fill(out_head, out_head + head_dim, 0.0f);
#ifdef __AVX2__
                    for (int64_t t = 0; t < total_seq; ++t) {
                        const float* v_head = v_cache + t * kv_dim + kv_h * head_dim;
                        __m256 vw = _mm256_set1_ps(scores[t]);
                        int64_t d = 0;
                        for (; d + 7 < head_dim; d += 8) {
                            _mm256_storeu_ps(out_head + d,
                                _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_head + d),
                                    _mm256_loadu_ps(out_head + d)));
                        }
                        for (; d < head_dim; ++d) out_head[d] += scores[t] * v_head[d];
                    }
#else
                    for (int64_t t = 0; t < total_seq; ++t) {
                        const float* v_head = v_cache + t * kv_dim + kv_h * head_dim;
                        float w = scores[t];
                        for (int64_t d = 0; d < head_dim; ++d) out_head[d] += w * v_head[d];
                    }
#endif
                    }  // end standard full attention
                }
                }, 1);  // min_grain=1: parallelize across heads
            }

            // -- Output projection: attn_buf @ W_o -> h_buf --
            // Post-attention norm (Gemma3): in-place on attn_buf
            if (layer.post_attention_norm.defined()) {
                cpu_quant::cpu_rmsnorm_inplace(sp.attn_buf, layer.post_attention_norm.data_ptr<float>(),
                    eps, add_one, q_dim);
            }

            if (use_quant_gemv_ && layer.q_attn_output.valid && layer.q_attn_output.cpu_data) {
                cpu_quant::cpu_quant_gemv(layer.q_attn_output.quant_type, layer.q_attn_output.cpu_data,
                    sp.attn_buf, sp.h_buf, q_dim, layer.q_attn_output.rows, layer.q_attn_output.row_stride_bytes);
            } else if (layer.attn_output.defined()) {
                // Float32 fallback
                const float* w = layer.attn_output.data_ptr<float>();
                int64_t N_out = layer.attn_output.size(0);
                for (int64_t n = 0; n < N_out; ++n) {
                    float dot = 0.0f;
                    for (int64_t k = 0; k < q_dim; ++k) dot += sp.attn_buf[k] * w[n * q_dim + k];
                    sp.h_buf[n] = dot;
                }
            }

            // -- Residual add: x = x + attn_output (in-place into other x_buf) --
            int next = 1 - cur;
#ifdef __AVX2__
            {
                int64_t j = 0;
                for (; j + 7 < H; j += 8) {
                    _mm256_storeu_ps(sp.x_buf[next] + j,
                        _mm256_add_ps(_mm256_loadu_ps(sp.x_buf[cur] + j),
                                      _mm256_loadu_ps(sp.h_buf + j)));
                }
                for (; j < H; ++j) sp.x_buf[next][j] = sp.x_buf[cur][j] + sp.h_buf[j];
            }
#else
            for (int64_t j = 0; j < H; ++j) sp.x_buf[next][j] = sp.x_buf[cur][j] + sp.h_buf[j];
#endif
            cur = next;

            // -- FFN: fused RMSNorm + gate+up GEMV --
            bool can_fuse_ffn = use_quant_gemv_ &&
                layer.q_ffn_gate.valid && layer.q_ffn_up.valid &&
                layer.q_ffn_gate.cpu_data && layer.q_ffn_up.cpu_data &&
                cpu_quant::cpu_quant_gemv_supported(layer.q_ffn_gate.quant_type) &&
                cpu_quant::cpu_quant_gemv_supported(layer.q_ffn_up.quant_type);

            if (can_fuse_ffn) {
                cpu_quant::cpu_fused_rmsnorm_gate_up_gemv(
                    sp.x_buf[cur], layer.ffn_norm.data_ptr<float>(), eps, add_one,
                    layer.q_ffn_gate.quant_type, layer.q_ffn_gate.cpu_data,
                    layer.q_ffn_up.quant_type, layer.q_ffn_up.cpu_data,
                    sp.gate_buf, sp.up_buf,
                    H, layer.q_ffn_gate.rows, layer.q_ffn_up.rows,
                    layer.q_ffn_gate.row_stride_bytes, layer.q_ffn_up.row_stride_bytes);
            } else {
                // Fallback: separate RMSNorm + GEMVs
                float norm2_buf[8192];
                float* x_normed2 = (H <= 8192) ? norm2_buf
                    : static_cast<float*>(std::malloc(H * sizeof(float)));
                float sum_sq = 0.0f;
                for (int64_t j = 0; j < H; ++j) sum_sq += sp.x_buf[cur][j] * sp.x_buf[cur][j];
                float rms = 1.0f / std::sqrt(sum_sq / H + eps);
                const float* gamma = layer.ffn_norm.data_ptr<float>();
                for (int64_t j = 0; j < H; ++j) {
                    float w = add_one ? (1.0f + gamma[j]) : gamma[j];
                    x_normed2[j] = sp.x_buf[cur][j] * rms * w;
                }
                if (layer.q_ffn_gate.valid && layer.q_ffn_gate.cpu_data)
                    cpu_quant::cpu_quant_gemv(layer.q_ffn_gate.quant_type, layer.q_ffn_gate.cpu_data,
                        x_normed2, sp.gate_buf, H, layer.q_ffn_gate.rows, layer.q_ffn_gate.row_stride_bytes);
                if (layer.q_ffn_up.valid && layer.q_ffn_up.cpu_data)
                    cpu_quant::cpu_quant_gemv(layer.q_ffn_up.quant_type, layer.q_ffn_up.cpu_data,
                        x_normed2, sp.up_buf, H, layer.q_ffn_up.rows, layer.q_ffn_up.row_stride_bytes);
                if (H > 8192) std::free(x_normed2);
            }

            // -- SiLU(gate) * up: in-place into gate_buf --
#ifdef __AVX2__
            {
                int64_t j = 0;
                __m256 one = _mm256_set1_ps(1.0f);
                __m256 neg_one = _mm256_set1_ps(-1.0f);
                for (; j + 7 < inter; j += 8) {
                    __m256 g = _mm256_loadu_ps(sp.gate_buf + j);
                    __m256 u = _mm256_loadu_ps(sp.up_buf + j);
                    __m256 neg_g = _mm256_mul_ps(g, neg_one);
                    neg_g = _mm256_max_ps(neg_g, _mm256_set1_ps(-88.0f));
                    neg_g = _mm256_min_ps(neg_g, _mm256_set1_ps(88.0f));
                    // Scalar exp fallback (same as existing code)
                    float tmp[8];
                    _mm256_storeu_ps(tmp, neg_g);
                    __m256 exp_neg_g = _mm256_set_ps(
                        std::exp(tmp[7]), std::exp(tmp[6]), std::exp(tmp[5]), std::exp(tmp[4]),
                        std::exp(tmp[3]), std::exp(tmp[2]), std::exp(tmp[1]), std::exp(tmp[0]));
                    __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_g));
                    __m256 silu = _mm256_mul_ps(g, sigmoid);
                    _mm256_storeu_ps(sp.gate_buf + j, _mm256_mul_ps(silu, u));
                }
                for (; j < inter; ++j) {
                    float g = sp.gate_buf[j];
                    sp.gate_buf[j] = (g / (1.0f + std::exp(-g))) * sp.up_buf[j];
                }
            }
#else
            for (int64_t j = 0; j < inter; ++j) {
                float g = sp.gate_buf[j];
                sp.gate_buf[j] = (g / (1.0f + std::exp(-g))) * sp.up_buf[j];
            }
#endif

            // -- Post-FFN norm (Gemma3) --
            // Applied to gate_buf before down projection
            // Note: for Gemma3, post_ffw_norm is applied AFTER SiLU*up but BEFORE down proj
            // Actually, Gemma3 applies post_ffw_norm after the entire FFN output
            // So we apply it after down_proj below

            // -- Down projection: gate_buf @ W_down -> h_buf --
            // Use sparse GEMV if available for this layer
            if (use_quant_gemv_ && layer.q_ffn_down.valid && layer.q_ffn_down.cpu_data) {
                if (use_sparse_gemv_ && i * 3 + 2 < (int64_t)sparse_ffn_.size() &&
                    sparse_ffn_[i * 3 + 2].valid && layer.q_ffn_down.is_q4k()) {
                    sparse_q4k_gemv(layer.q_ffn_down.cpu_data, sp.gate_buf, sp.h_buf,
                                    inter, layer.q_ffn_down.rows, layer.q_ffn_down.row_stride_bytes,
                                    sparse_ffn_[i * 3 + 2]);
                } else {
                    cpu_quant::cpu_quant_gemv(layer.q_ffn_down.quant_type, layer.q_ffn_down.cpu_data,
                        sp.gate_buf, sp.h_buf, inter, layer.q_ffn_down.rows, layer.q_ffn_down.row_stride_bytes);
                }
            }

            // Post-FFN norm (Gemma3)
            if (layer.post_ffw_norm.defined()) {
                cpu_quant::cpu_rmsnorm_inplace(sp.h_buf, layer.post_ffw_norm.data_ptr<float>(),
                    eps, add_one, H);
            }

            // -- Residual add: x = x + ffn_output (in-place into other x_buf) --
            next = 1 - cur;
#ifdef __AVX2__
            {
                int64_t j = 0;
                for (; j + 7 < H; j += 8) {
                    _mm256_storeu_ps(sp.x_buf[next] + j,
                        _mm256_add_ps(_mm256_loadu_ps(sp.x_buf[cur] + j),
                                      _mm256_loadu_ps(sp.h_buf + j)));
                }
                for (; j < H; ++j) sp.x_buf[next][j] = sp.x_buf[cur][j] + sp.h_buf[j];
            }
#else
            for (int64_t j = 0; j < H; ++j) sp.x_buf[next][j] = sp.x_buf[cur][j] + sp.h_buf[j];
#endif
            cur = next;
        }  // end layer loop

        // 3. Final RMS norm (in-place)
        cpu_quant::cpu_rmsnorm_inplace(sp.x_buf[cur], output_norm.data_ptr<float>(), eps, add_one, H);

        // 4. Output projection -> logits (into scratch logits_buf)
        //    Use sparse GEMV for output projection if available
        if (use_quant_gemv_ && q_output_weight.valid && q_output_weight.cpu_data) {
            if (use_sparse_gemv_ && sparse_output_.valid && q_output_weight.is_q4k()) {
                sparse_q4k_gemv(q_output_weight.cpu_data, sp.x_buf[cur], sp.logits_buf,
                                H, q_output_weight.rows, q_output_weight.row_stride_bytes,
                                sparse_output_);
            } else {
                cpu_quant::cpu_quant_gemv(q_output_weight.quant_type, q_output_weight.cpu_data,
                    sp.x_buf[cur], sp.logits_buf, H, q_output_weight.rows, q_output_weight.row_stride_bytes);
            }
        } else if (output_weight.defined()) {
            // Float32 fallback — AVX2 + threaded GEMV
            const float* w = output_weight.data_ptr<float>();
            const float* x_ptr_out = sp.x_buf[cur];
            int64_t V = config.vocab_size;
            c10::get_thread_pool().parallel_for(0, V, [&](int64_t start, int64_t end) {
                for (int64_t n = start; n < end; ++n) {
                    const float* w_row = w + n * H;
#ifdef __AVX2__
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    int64_t k = 0;
                    for (; k + 15 < H; k += 16) {
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr_out + k),
                                               _mm256_loadu_ps(w_row + k), acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr_out + k + 8),
                                               _mm256_loadu_ps(w_row + k + 8), acc1);
                    }
                    acc0 = _mm256_add_ps(acc0, acc1);
                    for (; k + 7 < H; k += 8) {
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr_out + k),
                                               _mm256_loadu_ps(w_row + k), acc0);
                    }
                    float dot = cpu_quant::hsum_avx(acc0);
                    for (; k < H; ++k) dot += x_ptr_out[k] * w_row[k];
                    sp.logits_buf[n] = dot;
#else
                    float dot = 0.0f;
                    for (int64_t k = 0; k < H; ++k) dot += x_ptr_out[k] * w_row[k];
                    sp.logits_buf[n] = dot;
#endif
                }
            }, 64);  // min_grain=64 to avoid excessive thread overhead
        }

        // Wrap logits in a Tensor (zero-copy view of scratch buffer)
        // We must copy because Tensor might outlive this decode call
        Tensor logits = at::empty({1, config.vocab_size});
        std::memcpy(logits.mutable_data_ptr<float>(), sp.logits_buf,
                     config.vocab_size * sizeof(float));

        kv_cache.seq_len += 1;
        return logits;
    }

    // ========================================================================
    // Speculative CPU decode: skip full output GEMV, use low-rank + exact top-k
    // Returns token ID directly (greedy). Much faster for large vocab models.
    // Only call when use_speculative_output_ is true and greedy decode.
    // ========================================================================
    int32_t forward_decode_cpu_speculative(int64_t token_id) {
        int64_t H = config.hidden_size;
        int64_t q_dim = config.num_heads * config.head_dim;
        int64_t kv_dim = config.num_kv_heads * config.head_dim;
        int64_t n_heads = config.num_heads;
        int64_t n_kv_heads = config.num_kv_heads;
        int64_t head_dim = config.head_dim;
        int64_t inter = config.intermediate_size;
        int64_t past_len = kv_cache.seq_len;
        float eps = config.rms_norm_eps;
        bool add_one = config.gemma_norm_add_one;
        int64_t heads_per_group = n_heads / n_kv_heads;

        if (!cpu_scratch_.allocated) {
            int64_t max_seq = kv_cache.max_seq > 0 ? kv_cache.max_seq : 4096;
            cpu_scratch_.allocate(config, max_seq);
        }

        auto& sp = cpu_scratch_;
        int cur = 0;

        // 1. Embedding lookup
        const float* emb_table = token_embedding.data_ptr<float>();
        std::memcpy(sp.x_buf[cur], emb_table + token_id * H, H * sizeof(float));

        if (config.scale_embeddings) {
            float scale = std::sqrt(static_cast<float>(H));
            for (int64_t j = 0; j < H; ++j) sp.x_buf[cur][j] *= scale;
        }

        // 2. Run all transformer layers (same logic as forward_decode_cpu)
        for (int64_t i = 0; i < config.num_layers; ++i) {
            auto& layer = layers[i];
            float* x_ptr = sp.x_buf[cur];

            // Fused RMSNorm + QKV
            bool can_fuse = use_quant_gemv_ &&
                layer.q_attn_q.valid && layer.q_attn_k.valid && layer.q_attn_v.valid &&
                layer.q_attn_q.cpu_data && layer.q_attn_k.cpu_data && layer.q_attn_v.cpu_data &&
                cpu_quant::cpu_quant_gemv_supported(layer.q_attn_q.quant_type) &&
                layer.q_attn_q.quant_type == layer.q_attn_k.quant_type &&
                layer.q_attn_q.quant_type == layer.q_attn_v.quant_type &&
                layer.q_attn_q.cols == layer.q_attn_k.cols &&
                layer.q_attn_q.cols == layer.q_attn_v.cols &&
                layer.q_attn_q.row_stride_bytes == layer.q_attn_k.row_stride_bytes;

            if (can_fuse) {
                cpu_quant::cpu_fused_rmsnorm_qkv_gemv(
                    x_ptr, layer.attn_norm.data_ptr<float>(), eps, add_one,
                    layer.q_attn_q.quant_type,
                    layer.q_attn_q.cpu_data, layer.q_attn_k.cpu_data, layer.q_attn_v.cpu_data,
                    sp.q_buf, sp.k_buf, sp.v_buf,
                    H, layer.q_attn_q.rows, layer.q_attn_k.rows, layer.q_attn_v.rows,
                    layer.q_attn_q.row_stride_bytes);
            } else {
                float norm_buf[8192];
                float* x_normed = (H <= 8192) ? norm_buf
                    : static_cast<float*>(std::malloc(H * sizeof(float)));
                float sum_sq = 0.0f;
                for (int64_t j = 0; j < H; ++j) sum_sq += x_ptr[j] * x_ptr[j];
                float rms = 1.0f / std::sqrt(sum_sq / H + eps);
                const float* gamma = layer.attn_norm.data_ptr<float>();
                for (int64_t j = 0; j < H; ++j) {
                    float w = add_one ? (1.0f + gamma[j]) : gamma[j];
                    x_normed[j] = x_ptr[j] * rms * w;
                }
                if (layer.q_attn_q.valid && layer.q_attn_q.cpu_data)
                    cpu_quant::cpu_quant_gemv(layer.q_attn_q.quant_type, layer.q_attn_q.cpu_data,
                        x_normed, sp.q_buf, H, layer.q_attn_q.rows, layer.q_attn_q.row_stride_bytes);
                if (layer.q_attn_k.valid && layer.q_attn_k.cpu_data)
                    cpu_quant::cpu_quant_gemv(layer.q_attn_k.quant_type, layer.q_attn_k.cpu_data,
                        x_normed, sp.k_buf, H, layer.q_attn_k.rows, layer.q_attn_k.row_stride_bytes);
                if (layer.q_attn_v.valid && layer.q_attn_v.cpu_data)
                    cpu_quant::cpu_quant_gemv(layer.q_attn_v.quant_type, layer.q_attn_v.cpu_data,
                        x_normed, sp.v_buf, H, layer.q_attn_v.rows, layer.q_attn_v.row_stride_bytes);
                if (H > 8192) std::free(x_normed);
            }

            // Biases (Qwen3)
            if (layer.attn_q_bias.defined()) {
                const float* bq = layer.attn_q_bias.data_ptr<float>();
                const float* bk = layer.attn_k_bias.data_ptr<float>();
                const float* bv = layer.attn_v_bias.data_ptr<float>();
                for (int64_t j = 0; j < q_dim; ++j) sp.q_buf[j] += bq[j];
                for (int64_t j = 0; j < kv_dim; ++j) { sp.k_buf[j] += bk[j]; sp.v_buf[j] += bv[j]; }
            }

            // QK-norm
            if (layer.attn_q_norm.defined()) {
                const float* qn_w = layer.attn_q_norm.data_ptr<float>();
                const float* kn_w = layer.attn_k_norm.data_ptr<float>();
                for (int64_t h = 0; h < n_heads; ++h)
                    cpu_quant::cpu_rmsnorm_inplace(sp.q_buf + h * head_dim, qn_w, eps, add_one, head_dim);
                for (int64_t h = 0; h < n_kv_heads; ++h)
                    cpu_quant::cpu_rmsnorm_inplace(sp.k_buf + h * head_dim, kn_w, eps, add_one, head_dim);
            }

            // RoPE
            {
                int64_t pos = past_len;
                float freq_base = config.rope_freq_base;
                for (int64_t h = 0; h < n_heads; ++h) {
                    float* hd = sp.q_buf + h * head_dim;
                    for (int64_t d = 0; d < head_dim / 2; ++d) {
                        float freq = 1.0f / std::pow(freq_base, 2.0f * d / head_dim);
                        float theta = pos * freq;
                        float ct = std::cos(theta), st = std::sin(theta);
                        float x0 = hd[2*d], x1 = hd[2*d+1];
                        hd[2*d] = x0*ct - x1*st; hd[2*d+1] = x0*st + x1*ct;
                    }
                }
                for (int64_t h = 0; h < n_kv_heads; ++h) {
                    float* hd = sp.k_buf + h * head_dim;
                    for (int64_t d = 0; d < head_dim / 2; ++d) {
                        float freq = 1.0f / std::pow(freq_base, 2.0f * d / head_dim);
                        float theta = pos * freq;
                        float ct = std::cos(theta), st = std::sin(theta);
                        float x0 = hd[2*d], x1 = hd[2*d+1];
                        hd[2*d] = x0*ct - x1*st; hd[2*d+1] = x0*st + x1*ct;
                    }
                }
            }

            // KV cache append
            {
                float* k_cache_w = kv_cache.key_cache[i].mutable_data_ptr<float>();
                float* v_cache_w = kv_cache.value_cache[i].mutable_data_ptr<float>();
                std::memcpy(k_cache_w + past_len * kv_dim, sp.k_buf, kv_dim * sizeof(float));
                std::memcpy(v_cache_w + past_len * kv_dim, sp.v_buf, kv_dim * sizeof(float));
            }

            // Attention (with sliding window)
            {
                int64_t total_seq = past_len + 1;
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                const float* k_cache_r = kv_cache.key_cache[i].data_ptr<float>();
                const float* v_cache_r = kv_cache.value_cache[i].data_ptr<float>();

                if (sliding_window_.enabled)
                    sliding_window_.update_window(i, total_seq, k_cache_r, v_cache_r, n_kv_heads, head_dim);

                for (int64_t h = 0; h < n_heads; ++h) {
                    int64_t kv_h = h / heads_per_group;
                    const float* q_head = sp.q_buf + h * head_dim;
                    float* out_head = sp.attn_buf + h * head_dim;
                    float local_scores[4096];
                    float* scores = (total_seq <= 4096) ? local_scores : sp.scores_buf;

                    if (sliding_window_.enabled && total_seq > sliding_window_.window_size) {
                        sliding_window_.compute_attention(q_head, k_cache_r, v_cache_r, out_head,
                            total_seq, kv_h, head_dim, kv_dim, i, scale, scores);
                    } else {
                        for (int64_t t = 0; t < total_seq; ++t) {
                            const float* kh = k_cache_r + t * kv_dim + kv_h * head_dim;
                            float dot = 0.0f;
                            for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * kh[d];
                            scores[t] = dot * scale;
                        }
                        float mx = scores[0];
                        for (int64_t t = 1; t < total_seq; ++t) if (scores[t] > mx) mx = scores[t];
                        float se = 0.0f;
                        for (int64_t t = 0; t < total_seq; ++t) { scores[t] = std::exp(scores[t] - mx); se += scores[t]; }
                        float inv_s = 1.0f / (se + 1e-10f);
                        for (int64_t t = 0; t < total_seq; ++t) scores[t] *= inv_s;
                        std::fill(out_head, out_head + head_dim, 0.0f);
                        for (int64_t t = 0; t < total_seq; ++t) {
                            const float* vh = v_cache_r + t * kv_dim + kv_h * head_dim;
                            for (int64_t d = 0; d < head_dim; ++d) out_head[d] += scores[t] * vh[d];
                        }
                    }
                }
            }

            // Output projection
            if (layer.post_attention_norm.defined())
                cpu_quant::cpu_rmsnorm_inplace(sp.attn_buf, layer.post_attention_norm.data_ptr<float>(), eps, add_one, q_dim);
            if (use_quant_gemv_ && layer.q_attn_output.valid && layer.q_attn_output.cpu_data)
                cpu_quant::cpu_quant_gemv(layer.q_attn_output.quant_type, layer.q_attn_output.cpu_data,
                    sp.attn_buf, sp.h_buf, q_dim, layer.q_attn_output.rows, layer.q_attn_output.row_stride_bytes);

            // Residual
            int next = 1 - cur;
            for (int64_t j = 0; j < H; ++j) sp.x_buf[next][j] = sp.x_buf[cur][j] + sp.h_buf[j];
            cur = next;

            // FFN
            bool can_fuse_ffn = use_quant_gemv_ &&
                layer.q_ffn_gate.valid && layer.q_ffn_up.valid &&
                layer.q_ffn_gate.cpu_data && layer.q_ffn_up.cpu_data &&
                cpu_quant::cpu_quant_gemv_supported(layer.q_ffn_gate.quant_type) &&
                cpu_quant::cpu_quant_gemv_supported(layer.q_ffn_up.quant_type);

            if (can_fuse_ffn) {
                cpu_quant::cpu_fused_rmsnorm_gate_up_gemv(
                    sp.x_buf[cur], layer.ffn_norm.data_ptr<float>(), eps, add_one,
                    layer.q_ffn_gate.quant_type, layer.q_ffn_gate.cpu_data,
                    layer.q_ffn_up.quant_type, layer.q_ffn_up.cpu_data,
                    sp.gate_buf, sp.up_buf,
                    H, layer.q_ffn_gate.rows, layer.q_ffn_up.rows,
                    layer.q_ffn_gate.row_stride_bytes, layer.q_ffn_up.row_stride_bytes);
            } else {
                float nb[8192];
                float* xn = (H <= 8192) ? nb : static_cast<float*>(std::malloc(H * sizeof(float)));
                float ssq = 0.0f;
                for (int64_t j = 0; j < H; ++j) ssq += sp.x_buf[cur][j] * sp.x_buf[cur][j];
                float rms_v = 1.0f / std::sqrt(ssq / H + eps);
                const float* gamma = layer.ffn_norm.data_ptr<float>();
                for (int64_t j = 0; j < H; ++j) {
                    float w = add_one ? (1.0f + gamma[j]) : gamma[j];
                    xn[j] = sp.x_buf[cur][j] * rms_v * w;
                }
                if (layer.q_ffn_gate.valid && layer.q_ffn_gate.cpu_data)
                    cpu_quant::cpu_quant_gemv(layer.q_ffn_gate.quant_type, layer.q_ffn_gate.cpu_data,
                        xn, sp.gate_buf, H, layer.q_ffn_gate.rows, layer.q_ffn_gate.row_stride_bytes);
                if (layer.q_ffn_up.valid && layer.q_ffn_up.cpu_data)
                    cpu_quant::cpu_quant_gemv(layer.q_ffn_up.quant_type, layer.q_ffn_up.cpu_data,
                        xn, sp.up_buf, H, layer.q_ffn_up.rows, layer.q_ffn_up.row_stride_bytes);
                if (H > 8192) std::free(xn);
            }

            // SiLU(gate) * up
            for (int64_t j = 0; j < inter; ++j) {
                float g = sp.gate_buf[j];
                sp.gate_buf[j] = (g / (1.0f + std::exp(-g))) * sp.up_buf[j];
            }

            // Down projection (with sparse GEMV)
            if (use_quant_gemv_ && layer.q_ffn_down.valid && layer.q_ffn_down.cpu_data) {
                if (use_sparse_gemv_ && i * 3 + 2 < (int64_t)sparse_ffn_.size() &&
                    sparse_ffn_[i * 3 + 2].valid && layer.q_ffn_down.is_q4k()) {
                    sparse_q4k_gemv(layer.q_ffn_down.cpu_data, sp.gate_buf, sp.h_buf,
                                    inter, layer.q_ffn_down.rows, layer.q_ffn_down.row_stride_bytes,
                                    sparse_ffn_[i * 3 + 2]);
                } else {
                    cpu_quant::cpu_quant_gemv(layer.q_ffn_down.quant_type, layer.q_ffn_down.cpu_data,
                        sp.gate_buf, sp.h_buf, inter, layer.q_ffn_down.rows, layer.q_ffn_down.row_stride_bytes);
                }
            }

            if (layer.post_ffw_norm.defined())
                cpu_quant::cpu_rmsnorm_inplace(sp.h_buf, layer.post_ffw_norm.data_ptr<float>(), eps, add_one, H);

            // Residual
            next = 1 - cur;
            for (int64_t j = 0; j < H; ++j) sp.x_buf[next][j] = sp.x_buf[cur][j] + sp.h_buf[j];
            cur = next;
        }

        // 3. Final RMS norm
        cpu_quant::cpu_rmsnorm_inplace(sp.x_buf[cur], output_norm.data_ptr<float>(), eps, add_one, H);

        // 4. SPECULATIVE output: low-rank approx -> top-k -> exact -> argmax
        //    Skips computing 151936 dot products, does only ~40M FMAs + 64 exact dots
        int32_t result = low_rank_output_.decode_greedy(
            sp.x_buf[cur],
            q_output_weight.cpu_data,
            q_output_weight.quant_type,
            q_output_weight.row_stride_bytes);

        kv_cache.seq_len += 1;
        return result;
    }


    // ========================================================================
    // Chat template formatting
    // ========================================================================

    std::string apply_chat_template(const std::string& prompt) const {
        if (config.architecture == "qwen3") {
            // Qwen3 chat format with /no_think system message to disable thinking
            return "<|im_start|>system\n/no_think<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        } else if (config.architecture == "gemma3" || config.architecture == "gemma2") {
            // Gemma chat format
            return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
        } else if (config.architecture == "llama") {
            // Llama 3 chat format
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                   + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        }
        // Default: raw prompt
        return prompt;
    }

    // Generate with chat template applied
    std::string chat(const std::string& prompt, int max_tokens = 128,
                     float temperature = 0.7f, int top_k = 40, float top_p = 0.9f,
                     float repetition_penalty = 1.05f) {
        return generate(apply_chat_template(prompt), max_tokens, temperature, top_k, top_p, repetition_penalty);
    }

    // ========================================================================
    // Generate text (raw prompt, no template)
    // ========================================================================

    std::string generate(const std::string& prompt, int max_tokens = 128,
                         float temperature = 0.7f, int top_k = 40, float top_p = 0.9f,
                         float repetition_penalty = 1.05f) {
        // Reset KV cache — allocate pre-sized buffers if not yet done
        kv_cache.reset();
        int64_t kv_dim = config.num_kv_heads * config.head_dim;
        int64_t max_total_seq = static_cast<int64_t>(max_tokens) + 2048; // prompt + generation
        if (max_total_seq > config.context_length) max_total_seq = config.context_length;
        if (!kv_cache.allocated || kv_cache.max_seq < max_total_seq) {
            kv_cache.allocate(config.num_layers, max_total_seq, kv_dim, use_cuda_);
        }

        // Encode prompt
        auto input_tokens = tokenizer.encode(prompt, true);
        std::cout << "[Generate] Prompt tokens: " << input_tokens.size() << std::endl;

        // Process prompt (prefill)
        std::vector<int64_t> tokens_i64(input_tokens.begin(), input_tokens.end());
        Tensor logits = forward(tokens_i64, true);

        // Get last token logits
        std::vector<int32_t> generated;
        auto t_start = std::chrono::high_resolution_clock::now();

        // Pre-check: can we use GPU-only greedy path? (no D2H transfer)
        bool gpu_greedy = use_cuda_ && (temperature < 1e-6f) && (repetition_penalty <= 1.0f);

        // Pre-allocate GPU argmax buffer (reused every token, 8 bytes)
#ifdef PT_USE_CUDA
        int64_t* d_argmax_idx = nullptr;
        if (gpu_greedy) {
            cudaMalloc(&d_argmax_idx, sizeof(int64_t));
        }
#endif

        // Speculative decode state: when speculative decode is used, we get the
        // next token directly from forward_decode_cpu_speculative (which runs
        // the full transformer + low-rank output + exact top-k). We store it
        // here so the next iteration can skip the sampling step.
        int32_t speculative_next_token = -1;
        bool speculative_token_ready = false;

        // Can we use speculative decode at all?
        bool can_speculative = use_speculative_output_ && !use_cuda_ &&
            use_quant_gemv_ && temperature < 1e-6f && repetition_penalty <= 1.0f;

        for (int step = 0; step < max_tokens; ++step) {
            int32_t next_token;

            // If speculative decode already computed the token, use it directly
            if (speculative_token_ready) {
                next_token = speculative_next_token;
                speculative_token_ready = false;
            } else {
            // Standard sampling from logits
#ifdef PT_USE_CUDA
            if (gpu_greedy && logits.is_cuda()) {
                PROF_BEGIN(profiler, "argmax");
                int64_t last_pos = logits.size(0) - 1;
                int64_t V = logits.size(1);
                const float* logit_row = logits.data_ptr<float>() + last_pos * V;
                at::cuda::launch_argmax(logit_row, d_argmax_idx, V, nullptr);
                int64_t h_idx = 0;
                cudaMemcpy(&h_idx, d_argmax_idx, sizeof(int64_t), cudaMemcpyDeviceToHost);
                next_token = static_cast<int32_t>(h_idx);
                PROF_END(profiler, "argmax");
            } else
#endif
            {
                // CPU greedy fast path: argmax directly on scratch buffer (no tensor copy)
                if (use_quant_gemv_ && !use_cuda_ && cpu_scratch_.allocated &&
                    temperature < 1e-6f && (repetition_penalty <= 1.0f || generated.empty())) {
                    const float* lbuf = cpu_scratch_.logits_buf;
                    int64_t V = config.vocab_size;
                    int32_t best = 0;
                    float best_val = lbuf[0];
#ifdef __AVX2__
                    // AVX2 argmax
                    __m256 vmax = _mm256_set1_ps(lbuf[0]);
                    __m256i vidx = _mm256_setzero_si256();
                    __m256i vstep = _mm256_set1_epi32(8);
                    __m256i vcur = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
                    int64_t j = 0;
                    for (; j + 7 < V; j += 8) {
                        __m256 vv = _mm256_loadu_ps(lbuf + j);
                        __m256 cmp = _mm256_cmp_ps(vv, vmax, _CMP_GT_OS);
                        vmax = _mm256_blendv_ps(vmax, vv, cmp);
                        vidx = _mm256_castps_si256(_mm256_blendv_ps(
                            _mm256_castsi256_ps(vidx), _mm256_castsi256_ps(vcur), cmp));
                        vcur = _mm256_add_epi32(vcur, vstep);
                    }
                    // Reduce 8 lanes
                    alignas(32) float vals[8];
                    alignas(32) int32_t idxs[8];
                    _mm256_store_ps(vals, vmax);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(idxs), vidx);
                    for (int k = 0; k < 8; ++k) {
                        if (vals[k] > best_val) { best_val = vals[k]; best = idxs[k]; }
                    }
                    for (; j < V; ++j) {
                        if (lbuf[j] > best_val) { best_val = lbuf[j]; best = static_cast<int32_t>(j); }
                    }
#else
                    for (int64_t j = 1; j < V; ++j) {
                        if (lbuf[j] > best_val) { best_val = lbuf[j]; best = static_cast<int32_t>(j); }
                    }
#endif
                    next_token = best;
                } else {
                    // General path: extract row, apply penalties, sample
                    int64_t last_pos = logits.size(0) - 1;
                    Tensor last_logits = get_row(logits, last_pos);

                    if (repetition_penalty > 1.0f && !generated.empty()) {
                        float* logit_data = last_logits.mutable_data_ptr<float>();
                        for (int32_t prev_token : generated) {
                            if (prev_token >= 0 && prev_token < static_cast<int32_t>(tokenizer.vocab.size())) {
                                if (logit_data[prev_token] > 0) {
                                    logit_data[prev_token] /= repetition_penalty;
                                } else {
                                    logit_data[prev_token] *= repetition_penalty;
                                }
                            }
                        }
                    }

                    next_token = sample_token(last_logits, temperature, top_k, top_p);
                }
            }
            }  // end else (not speculative_token_ready)

            if (next_token == tokenizer.eos_id) {
                break;
            }

            // Check for model-specific stop tokens
            if (is_stop_token(next_token)) {
                break;
            }

            generated.push_back(next_token);

            // Print token as it's generated
            std::string token_str = tokenizer.decode_token(next_token);
            std::cout << token_str << std::flush;

            // Forward with single token (using KV cache)
#ifdef PT_USE_CUDA
            if (use_cuda_) {
                logits = forward_decode(static_cast<int64_t>(next_token));
            } else
#endif
            if (can_speculative) {
                // SPECULATIVE DECODE: forward + low-rank output → token directly
                // Skips 151936-dim GEMV, does rank-256 approx + 64 exact dots instead
                speculative_next_token = forward_decode_cpu_speculative(
                    static_cast<int64_t>(next_token));
                speculative_token_ready = true;
            } else {
                // Standard CPU decode path
                if (use_quant_gemv_) {
                    logits = forward_decode_cpu(static_cast<int64_t>(next_token));
                } else {
                    std::vector<int64_t> next_input = {static_cast<int64_t>(next_token)};
                    logits = forward(next_input, true);
                }
            }

            // Profiler: count token and sample VRAM periodically
            if (profiler.enabled()) {
                profiler.count_tokens(1);
                if (step % 16 == 0) profiler.sample_vram();
            }
        }

        // Free GPU argmax buffer
#ifdef PT_USE_CUDA
        if (d_argmax_idx) {
            cudaFree(d_argmax_idx);
            d_argmax_idx = nullptr;
        }
#endif

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tokens_per_sec = generated.size() / (ms / 1000.0);

        std::cout << "\n\n[Generate] " << generated.size() << " tokens in "
                  << (ms / 1000.0) << "s (" << tokens_per_sec << " tok/s)" << std::endl;

        // Print profiler report if enabled
        if (profiler.enabled()) {
            profiler.print_report(std::cout);
            profiler.print_vram_timeline(std::cout);
        }

        std::string result = tokenizer.decode(generated, true);

        // Strip thinking blocks: <think>...</think>
        for (;;) {
            size_t s = result.find("<think>");
            if (s == std::string::npos) break;
            size_t e = result.find("</think>", s);
            if (e != std::string::npos) {
                result.erase(s, e + 8 - s);
            } else {
                // No closing tag — remove everything from <think> to end
                result.erase(s);
                break;
            }
        }
        // Also remove standalone </think>
        for (;;) {
            size_t s = result.find("</think>");
            if (s == std::string::npos) break;
            result.erase(s, 8);
        }

        // Trim leading whitespace
        size_t first_non_ws = result.find_first_not_of(" \n\r\t");
        if (first_non_ws != std::string::npos && first_non_ws > 0) {
            result = result.substr(first_non_ws);
        } else if (first_non_ws == std::string::npos) {
            result.clear();
        }

        return result;
    }

    // Public access for PromeServe streaming API
    bool is_stop_token(int32_t token_id) const {
        if (token_id < 0 || token_id >= static_cast<int32_t>(tokenizer.vocab.size())) return false;
        const auto& tok = tokenizer.vocab[token_id];
        // Common stop tokens across architectures
        if (tok == "<|im_end|>" || tok == "<|endoftext|>" ||
            tok == "<end_of_turn>" || tok == "<|eot_id|>" ||
            tok == "</s>" || tok == "<|end|>") {
            return true;
        }
        return false;
    }

    // ========================================================================
    // CPU-lite loader: only small tensors (norms, embeddings, biases)
    // Large weight matrices (attn_q/k/v/output, ffn_gate/up/down) are
    // skipped — they will be loaded as raw quant blocks by load_quantized_to_cpu()
    // ========================================================================

    void load_weights_cpu_lite(const gguf::GGUFReader& reader) {
        std::cout << "\n[Model] Loading weights (CPU-lite: norms+embeddings only)..." << std::endl;

        // Token embeddings — always needed for embedding lookup
        token_embedding = reader.load_tensor("token_embd.weight");
        std::cout << "  token_embd: [" << token_embedding.size(0) << ", "
                  << token_embedding.size(1) << "]" << std::endl;

        // Output norm — always needed (small: [hidden])
        output_norm = reader.load_tensor("output_norm.weight");

        // Output weight: needed for tied embeddings, otherwise skip (use quant)
        if (!reader.has_tensor("output.weight")) {
            output_weight = token_embedding;
            config.tie_word_embeddings = true;
            std::cout << "  output: tied to token_embd" << std::endl;
        } else {
            // Check if quant type is supported — if not, load as FP32 fallback
            const auto& info = reader.get_tensor_info("output.weight");
            uint32_t type = info.type;
            if (!cpu_quant::cpu_quant_gemv_supported(type)) {
                output_weight = reader.load_tensor("output.weight");
                std::cout << "  output: FP32 fallback (quant type " << type << ")" << std::endl;
            } else {
                std::cout << "  output: quant GEMV (skipping FP32 dequant)" << std::endl;
            }
        }

        // Layers: only load small tensors (norms, biases, qk_norm)
        layers.resize(config.num_layers);
        for (int64_t i = 0; i < config.num_layers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";
            auto& layer = layers[i];

            // Norms — small: [hidden] each
            layer.attn_norm = reader.load_tensor(prefix + "attn_norm.weight");
            layer.ffn_norm = reader.load_tensor(prefix + "ffn_norm.weight");

            // QK-norm (Gemma3, Qwen3) — small: [head_dim]
            if (reader.has_tensor(prefix + "attn_q_norm.weight")) {
                layer.attn_q_norm = reader.load_tensor(prefix + "attn_q_norm.weight");
                layer.attn_k_norm = reader.load_tensor(prefix + "attn_k_norm.weight");
            }

            // Post-norms (Gemma3) — small: [hidden]
            if (reader.has_tensor(prefix + "post_attention_norm.weight")) {
                layer.post_attention_norm = reader.load_tensor(prefix + "post_attention_norm.weight");
            }
            if (reader.has_tensor(prefix + "post_ffw_norm.weight")) {
                layer.post_ffw_norm = reader.load_tensor(prefix + "post_ffw_norm.weight");
            }

            // Biases — small
            if (reader.has_tensor(prefix + "attn_q.bias")) {
                layer.attn_q_bias = reader.load_tensor(prefix + "attn_q.bias");
                layer.attn_k_bias = reader.load_tensor(prefix + "attn_k.bias");
                layer.attn_v_bias = reader.load_tensor(prefix + "attn_v.bias");
            }

            // SKIP: attn_q/k/v/output, ffn_gate/up/down weights
            // These are loaded as raw quant blocks by load_quantized_to_cpu()

            if ((i + 1) % 5 == 0 || i == config.num_layers - 1) {
                std::cout << "  Layer " << (i + 1) << "/" << config.num_layers
                          << " loaded (lite)" << std::endl;
            }
        }
    }

    // ========================================================================
    // Load weights from GGUF reader
    // ========================================================================

    void load_weights(const gguf::GGUFReader& reader) {
        std::cout << "\n[Model] Loading weights..." << std::endl;

        // Token embeddings
        token_embedding = reader.load_tensor("token_embd.weight");
        std::cout << "  token_embd: [" << token_embedding.size(0) << ", "
                  << token_embedding.size(1) << "]" << std::endl;

        // Output norm
        output_norm = reader.load_tensor("output_norm.weight");

        // Output weight (may be tied to embeddings)
        if (reader.has_tensor("output.weight")) {
            output_weight = reader.load_tensor("output.weight");
        } else {
            // Tie output to embeddings (Gemma, Qwen3, etc.)
            output_weight = token_embedding;
            config.tie_word_embeddings = true;
            std::cout << "  output: tied to token_embd" << std::endl;
        }

        // Layers
        layers.resize(config.num_layers);
        for (int64_t i = 0; i < config.num_layers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";

            auto& layer = layers[i];

            // Attention norm
            layer.attn_norm = reader.load_tensor(prefix + "attn_norm.weight");

            // Attention weights
            layer.attn_q = reader.load_tensor(prefix + "attn_q.weight");
            layer.attn_k = reader.load_tensor(prefix + "attn_k.weight");
            layer.attn_v = reader.load_tensor(prefix + "attn_v.weight");
            layer.attn_output = reader.load_tensor(prefix + "attn_output.weight");

            // QK-norm (Gemma3)
            if (reader.has_tensor(prefix + "attn_q_norm.weight")) {
                layer.attn_q_norm = reader.load_tensor(prefix + "attn_q_norm.weight");
                layer.attn_k_norm = reader.load_tensor(prefix + "attn_k_norm.weight");
            }

            // Post-norms (Gemma3)
            if (reader.has_tensor(prefix + "post_attention_norm.weight")) {
                layer.post_attention_norm = reader.load_tensor(prefix + "post_attention_norm.weight");
            }
            if (reader.has_tensor(prefix + "post_ffw_norm.weight")) {
                layer.post_ffw_norm = reader.load_tensor(prefix + "post_ffw_norm.weight");
            }

            // FFN norm
            layer.ffn_norm = reader.load_tensor(prefix + "ffn_norm.weight");

            // FFN weights
            layer.ffn_gate = reader.load_tensor(prefix + "ffn_gate.weight");
            layer.ffn_up = reader.load_tensor(prefix + "ffn_up.weight");
            layer.ffn_down = reader.load_tensor(prefix + "ffn_down.weight");

            // Optional biases
            if (reader.has_tensor(prefix + "attn_q.bias")) {
                layer.attn_q_bias = reader.load_tensor(prefix + "attn_q.bias");
                layer.attn_k_bias = reader.load_tensor(prefix + "attn_k.bias");
                layer.attn_v_bias = reader.load_tensor(prefix + "attn_v.bias");
            }

            if ((i + 1) % 5 == 0 || i == config.num_layers - 1) {
                std::cout << "  Layer " << (i + 1) << "/" << config.num_layers
                          << " loaded" << std::endl;
            }
        }
    }

    // ========================================================================
    // Embedding lookup: token_ids → [seq_len, hidden]
    // ========================================================================

    Tensor embedding_lookup(const std::vector<int64_t>& tokens) {
        int64_t seq_len = static_cast<int64_t>(tokens.size());
        int64_t H = config.hidden_size;

        // token_embedding is always on CPU (kept there for fast lookup)
        Tensor emb_cpu = token_embedding;

        Tensor result = at::empty({seq_len, H});
        float* dst = result.mutable_data_ptr<float>();
        const float* emb = emb_cpu.data_ptr<float>();

        for (int64_t i = 0; i < seq_len; ++i) {
            int64_t token = tokens[i];
            if (token < 0 || token >= config.vocab_size) {
                throw std::runtime_error("Token ID out of range: " + std::to_string(token));
            }
            std::memcpy(dst + i * H, emb + token * H, H * sizeof(float));
        }

        // Move to GPU if using CUDA
        if (use_cuda_) return to_cuda_tensor(result);
        return result;
    }

    // ========================================================================
    // RMS Normalization
    // ========================================================================

    Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
#ifdef PT_USE_CUDA
        if (use_cuda_ && x.is_cuda()) {
            int64_t rows = x.size(0);
            int64_t hidden = x.size(-1);
            auto output = at::empty_cuda(x.sizes().vec(), x.dtype(), x.device().index());
            at::cuda::launch_rms_norm(
                x.data_ptr<float>(), weight.data_ptr<float>(),
                output.mutable_data_ptr<float>(),
                static_cast<int>(rows), static_cast<int>(hidden),
                eps, config.gemma_norm_add_one, nullptr);
            return output;
        }
#endif
        // CPU fallback
        int64_t outer = x.size(0);
        int64_t hidden = x.size(-1);
        Tensor output = at::empty(x.sizes().vec());
        const float* x_data = x.data_ptr<float>();
        const float* w_data = weight.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();
        bool add_one = config.gemma_norm_add_one;

        for (int64_t s = 0; s < outer; ++s) {
            const float* row = x_data + s * hidden;
            float* out_row = out_data + s * hidden;
            float sum_sq = 0.0f;
            for (int64_t j = 0; j < hidden; ++j) sum_sq += row[j] * row[j];
            float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
            for (int64_t j = 0; j < hidden; ++j) {
                float w = add_one ? (1.0f + w_data[j]) : w_data[j];
                out_row[j] = row[j] * rms * w;
            }
        }
        return output;
    }

    // Per-head RMS norm for QK-norm (in-place on current device)
    void apply_qk_norm_inplace(Tensor& x, const Tensor& weight,
                                int64_t n_heads, int64_t head_dim) {
#ifdef PT_USE_CUDA
        if (use_cuda_ && x.is_cuda()) {
            int64_t seq_len = x.size(0);
            at::cuda::launch_per_head_rms_norm(
                x.mutable_data_ptr<float>(), weight.data_ptr<float>(),
                static_cast<int>(seq_len), static_cast<int>(n_heads),
                static_cast<int>(head_dim),
                config.rms_norm_eps, config.gemma_norm_add_one, nullptr);
            return;
        }
#endif
        // CPU fallback
        int64_t seq_len = x.size(0);
        float* data = x.mutable_data_ptr<float>();
        const float* w_data = weight.data_ptr<float>();
        float eps = config.rms_norm_eps;
        bool add_one = config.gemma_norm_add_one;
        for (int64_t s = 0; s < seq_len; ++s) {
            for (int64_t h = 0; h < n_heads; ++h) {
                float* head = data + s * (n_heads * head_dim) + h * head_dim;
                float sum_sq = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) sum_sq += head[d] * head[d];
                float rms = 1.0f / std::sqrt(sum_sq / head_dim + eps);
                for (int64_t d = 0; d < head_dim; ++d) {
                    float w = add_one ? (1.0f + w_data[d]) : w_data[d];
                    head[d] = head[d] * rms * w;
                }
            }
        }
    }

    // ========================================================================
    // Matrix multiplication: A @ B or A @ B^T
    // ========================================================================

    // Quantized matmul: uses Q4_K/Q6_K/Q5_K GEMV kernels
    // qw: quantized weight in original [N, K] layout
    // Falls back to float32 matmul if quant not available
    Tensor matmul_q(const Tensor& a, const Tensor& b,
                    const QuantizedWeight& qw, bool transpose_b = false) {
#ifdef PT_USE_CUDA
        if (use_cuda_ && use_quant_gemv_ && qw.valid && qw.gpu_data) {
            int64_t M = a.size(0);
            int K = static_cast<int>(qw.cols);
            int N = static_cast<int>(qw.rows);

            // Select launch function based on quant type
            // Use persistent kernel for Q4_K single-token decode
            auto launch_gemv = [&](const float* x_ptr, float* y_ptr) {
                if (qw.is_q4k()) {
                    at::cuda::launch_q4km_persistent_gemv(qw.gpu_data, x_ptr, y_ptr,
                        K, N, qw.row_stride_bytes, nullptr);
                } else if (qw.is_q6k()) {
                    at::cuda::launch_q6k_gemv(qw.gpu_data, x_ptr, y_ptr,
                        K, N, qw.row_stride_bytes, nullptr);
                } else if (qw.is_q5k()) {
                    at::cuda::launch_q5k_gemv(qw.gpu_data, x_ptr, y_ptr,
                        K, N, qw.row_stride_bytes, nullptr);
                }
            };

            if (M == 1) {
                auto output = at::empty_cuda({1, N}, a.dtype(), a.device().index());
                launch_gemv(a.data_ptr<float>(), output.mutable_data_ptr<float>());
                return output;
            } else {
                // Prefill (M>1): batch GEMV — one per input row
                auto output = at::empty_cuda({M, N}, a.dtype(), a.device().index());
                const float* a_ptr = a.data_ptr<float>();
                float* out_ptr = output.mutable_data_ptr<float>();
                for (int64_t m = 0; m < M; ++m) {
                    launch_gemv(a_ptr + m * K, out_ptr + m * N);
                }
                return output;
            }
        }
#endif
        // CPU fused dequant-GEMV for all supported quant types (Q4_K, Q6_K, Q5_K, Q8_0)
        if (!use_cuda_ && use_quant_gemv_ && qw.valid && qw.cpu_data &&
            cpu_quant::cpu_quant_gemv_supported(qw.quant_type)) {
            int64_t M = a.size(0);
            int64_t K_dim = static_cast<int64_t>(qw.cols);
            int64_t N_dim = static_cast<int64_t>(qw.rows);

            Tensor output = at::zeros({M, N_dim});
            const float* x_data = a.data_ptr<float>();
            float* y_data = output.mutable_data_ptr<float>();

            if (M == 1) {
                // Single-token decode: direct GEMV
                cpu_quant::cpu_quant_gemv(qw.quant_type, qw.cpu_data,
                    x_data, y_data, K_dim, N_dim, qw.row_stride_bytes);
            } else {
                // Prefill (M>1): batch GEMV — one per input row
                for (int64_t m = 0; m < M; ++m) {
                    cpu_quant::cpu_quant_gemv(qw.quant_type, qw.cpu_data,
                        x_data + m * K_dim, y_data + m * N_dim,
                        K_dim, N_dim, qw.row_stride_bytes);
                }
            }
            return output;
        }
        return matmul(a, b, transpose_b);
    }

    // ========================================================================
    // CPU Q4_K_M fused dequant-GEMV
    // For each output row: read Q4_K_M blocks, dequant in-register, dot with input
    // Q4_K_M block = 144 bytes = 256 values: d(fp16) + dmin(fp16) + scales[12] + qs[128]
    // ========================================================================

    Tensor cpu_q4km_gemv(const Tensor& x, const QuantizedWeight& qw) {
        int64_t K = qw.cols;
        int64_t N = qw.rows;
        int64_t blocks_per_row = K / 256;  // QK_K = 256

        Tensor output = at::zeros({1, N});
        const float* x_data = x.data_ptr<float>();
        float* y_data = output.mutable_data_ptr<float>();
        const uint8_t* raw = static_cast<const uint8_t*>(qw.cpu_data);

        for (int64_t n = 0; n < N; ++n) {
            const uint8_t* row_data = raw + n * qw.row_stride_bytes;

#ifdef __AVX2__
            // Accumulate entire row in vector registers — single horizontal sum at end
            __m256 acc = _mm256_setzero_ps();
            __m256i mask_lo = _mm256_set1_epi32(0xF);

            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* block = row_data + bi * 144;
                int64_t base_k = bi * 256;

                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits, block, 2);
                std::memcpy(&dmin_bits, block + 2, 2);
                const float d = gguf::fp16_to_fp32(d_bits);
                const float dmin = gguf::fp16_to_fp32(dmin_bits);
                const uint8_t* scales = block + 4;
                const uint8_t* qs = block + 16;

                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m_val;
                    gguf::get_scale_min_k4(is, scales, &sc, &m_val);
                    float d1 = d * sc;
                    float m1 = dmin * m_val;
                    gguf::get_scale_min_k4(is + 1, scales, &sc, &m_val);
                    float d2 = d * sc;
                    float m2 = dmin * m_val;

                    // Compute: sum_x += (d1 * q_lo - m1) * x  and  (d2 * q_hi - m2) * x
                    // Rewrite: d1 * sum(q_lo * x) - m1 * sum(x)  + d2 * sum(q_hi * x) - m2 * sum(x_hi)
                    // This avoids per-element dequant: just accumulate q*x and x separately

                    __m256 sum_qx_lo = _mm256_setzero_ps();
                    __m256 sum_x_lo = _mm256_setzero_ps();
                    __m256 sum_qx_hi = _mm256_setzero_ps();
                    __m256 sum_x_hi = _mm256_setzero_ps();

                    for (int l = 0; l < 32; l += 8) {
                        // Load 8 quant bytes, extract low/high nibbles as int32
                        // Use scalar extract since we need nibble separation
                        __m256i qi = _mm256_set_epi32(
                            qs[l+7], qs[l+6], qs[l+5], qs[l+4],
                            qs[l+3], qs[l+2], qs[l+1], qs[l+0]);
                        __m256i q_lo_i = _mm256_and_si256(qi, mask_lo);
                        __m256i q_hi_i = _mm256_srli_epi32(qi, 4);

                        __m256 q_lo_f = _mm256_cvtepi32_ps(q_lo_i);
                        __m256 q_hi_f = _mm256_cvtepi32_ps(q_hi_i);

                        __m256 vx_lo = _mm256_loadu_ps(x_data + base_k + j + l);
                        __m256 vx_hi = _mm256_loadu_ps(x_data + base_k + j + 32 + l);

                        sum_qx_lo = _mm256_fmadd_ps(q_lo_f, vx_lo, sum_qx_lo);
                        sum_x_lo = _mm256_add_ps(sum_x_lo, vx_lo);
                        sum_qx_hi = _mm256_fmadd_ps(q_hi_f, vx_hi, sum_qx_hi);
                        sum_x_hi = _mm256_add_ps(sum_x_hi, vx_hi);
                    }
                    // acc += d1 * sum(q_lo * x) - m1 * sum(x_lo) + d2 * sum(q_hi * x) - m2 * sum(x_hi)
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d1), sum_qx_lo, acc);
                    acc = _mm256_fnmadd_ps(_mm256_set1_ps(m1), sum_x_lo, acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d2), sum_qx_hi, acc);
                    acc = _mm256_fnmadd_ps(_mm256_set1_ps(m2), sum_x_hi, acc);

                    qs += 32;
                    is += 2;
                }
            }
            // Single horizontal sum at the end
            __m128 hi128 = _mm256_extractf128_ps(acc, 1);
            __m128 lo128 = _mm256_castps256_ps128(acc);
            __m128 sum4 = _mm_add_ps(lo128, hi128);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            y_data[n] = _mm_cvtss_f32(sum4);
#else
            float dot = 0.0f;
            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* block = row_data + bi * 144;
                int64_t base_k = bi * 256;
                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits, block, 2);
                std::memcpy(&dmin_bits, block + 2, 2);
                const float d = gguf::fp16_to_fp32(d_bits);
                const float dmin = gguf::fp16_to_fp32(dmin_bits);
                const uint8_t* scales = block + 4;
                const uint8_t* qs = block + 16;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m_val;
                    gguf::get_scale_min_k4(is, scales, &sc, &m_val);
                    float d1 = d * sc;
                    float m1 = dmin * m_val;
                    gguf::get_scale_min_k4(is + 1, scales, &sc, &m_val);
                    float d2 = d * sc;
                    float m2 = dmin * m_val;
                    for (int l = 0; l < 32; ++l) {
                        dot += (d1 * (qs[l] & 0xF) - m1) * x_data[base_k + j + l];
                        dot += (d2 * (qs[l] >> 4) - m2) * x_data[base_k + j + 32 + l];
                    }
                    qs += 32;
                    is += 2;
                }
            }
            y_data[n] = dot;
#endif
        }
        return output;
    }

    Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_b = false) {
#ifdef PT_USE_CUDA
        if (use_cuda_) {
            // On GPU, weights are pre-transposed during to_cuda().
            // So when transpose_b=true, B is already [K, N] (not [N, K]).
            // For decode (M=1): use dedicated GEMV kernel (much faster than GEMM)
            if (a.size(0) == 1) {
                int K = a.size(1);
                int N = b.size(1);
                auto output = at::empty_cuda({1, N}, a.dtype(), a.device().index());
                at::cuda::launch_inference_gemv(
                    a.data_ptr<float>(), b.data_ptr<float>(),
                    output.mutable_data_ptr<float>(), K, N, nullptr);
                return output;
            }
            // Prefill: use tiled GEMM
            return at::cuda_ops::mm(a, b);
        }
#endif
        int64_t M = a.size(0);
        int64_t K = a.size(1);
        int64_t N = transpose_b ? b.size(0) : b.size(1);

        if (transpose_b) {
            if (b.size(1) != K) {
                throw std::runtime_error("matmul: dimension mismatch (transpose_b)");
            }
        } else {
            if (b.size(0) != K) {
                throw std::runtime_error("matmul: dimension mismatch");
            }
        }

        Tensor result = at::zeros({M, N});
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* c_data = result.mutable_data_ptr<float>();

        if (transpose_b) {
            // C[m,n] = sum_k A[m,k] * B[n,k]  (B is [N, K])
#ifdef __AVX2__
            if (M == 1) {
                // AVX2 GEMV: one dot product per output element
                const float* x = a_data;
                for (int64_t n = 0; n < N; ++n) {
                    const float* w = b_data + n * K;
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    int64_t k = 0;
                    for (; k + 15 < K; k += 16) {
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k),     _mm256_loadu_ps(w + k),     acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), _mm256_loadu_ps(w + k + 8), acc1);
                    }
                    acc0 = _mm256_add_ps(acc0, acc1);
                    for (; k + 7 < K; k += 8) {
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), _mm256_loadu_ps(w + k), acc0);
                    }
                    // Horizontal sum
                    __m128 hi = _mm256_extractf128_ps(acc0, 1);
                    __m128 lo = _mm256_castps256_ps128(acc0);
                    __m128 sum4 = _mm_add_ps(lo, hi);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    float sum = _mm_cvtss_f32(sum4);
                    for (; k < K; ++k) sum += x[k] * w[k];
                    c_data[n] = sum;
                }
                return result;
            }
#endif
            #pragma omp parallel for schedule(dynamic) if(M > 4)
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    const float* a_row = a_data + m * K;
                    const float* b_row = b_data + n * K;
                    int64_t k = 0;
                    for (; k + 3 < K; k += 4) {
                        sum += a_row[k] * b_row[k] + a_row[k+1] * b_row[k+1]
                             + a_row[k+2] * b_row[k+2] + a_row[k+3] * b_row[k+3];
                    }
                    for (; k < K; ++k) sum += a_row[k] * b_row[k];
                    c_data[m * N + n] = sum;
                }
            }
        } else {
            // C[m,n] = sum_k A[m,k] * B[k,n]
            #pragma omp parallel for schedule(dynamic) if(M > 4)
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t k = 0; k < K; ++k) {
                    float a_val = a_data[m * K + k];
                    for (int64_t n = 0; n < N; ++n) {
                        c_data[m * N + n] += a_val * b_data[k * N + n];
                    }
                }
            }
        }

        return result;
    }

    // ========================================================================
    // Transformer layer forward
    // ========================================================================

    Tensor transformer_layer(const Tensor& x, int64_t layer_idx,
                              int64_t past_len, bool use_cache) {
        auto& layer = layers[layer_idx];

        // 1. Attention pre-norm
        PROF_BEGIN(profiler, "attn_norm");
        Tensor normed = rms_norm(x, layer.attn_norm, config.rms_norm_eps);
        PROF_END(profiler, "attn_norm");

        // Self-attention
        Tensor attn_out = self_attention(normed, layer, layer_idx, past_len, use_cache);

        // 2. Post-attention norm (Gemma3)
        if (layer.post_attention_norm.defined()) {
            attn_out = rms_norm(attn_out, layer.post_attention_norm, config.rms_norm_eps);
        }

        // 3. Residual
        PROF_BEGIN(profiler, "residual_add");
        Tensor h = add_tensors(x, attn_out);
        PROF_END(profiler, "residual_add");

        // 4. FFN pre-norm
        PROF_BEGIN(profiler, "ffn_norm");
        Tensor normed2 = rms_norm(h, layer.ffn_norm, config.rms_norm_eps);
        PROF_END(profiler, "ffn_norm");

        // SwiGLU FFN
        Tensor ffn_out = swiglu_ffn(normed2, layer);

        // 5. Post-FFN norm (Gemma3)
        if (layer.post_ffw_norm.defined()) {
            ffn_out = rms_norm(ffn_out, layer.post_ffw_norm, config.rms_norm_eps);
        }

        // 6. Residual
        PROF_BEGIN(profiler, "residual_add");
        Tensor result = add_tensors(h, ffn_out);
        PROF_END(profiler, "residual_add");
        return result;
    }

    // ========================================================================
    // Self-attention with RoPE and GQA
    // ========================================================================

    Tensor self_attention(const Tensor& x, const TransformerLayer& layer,
                           int64_t layer_idx, int64_t past_len, bool use_cache) {
        int64_t seq_len = x.size(0);
        int64_t n_heads = config.num_heads;
        int64_t n_kv_heads = config.num_kv_heads;
        int64_t head_dim = config.head_dim;
        int64_t q_dim = n_heads * head_dim;
        int64_t kv_dim = n_kv_heads * head_dim;

        // Q, K, V projections
        PROF_BEGIN(profiler, "qkv_proj");
        Tensor q = matmul_q(x, layer.attn_q, layer.q_attn_q, true);   // [seq, q_dim]
        Tensor k = matmul_q(x, layer.attn_k, layer.q_attn_k, true);   // [seq, kv_dim]
        Tensor v = matmul_q(x, layer.attn_v, layer.q_attn_v, true);   // [seq, kv_dim]
        PROF_END(profiler, "qkv_proj");

        // Add biases if present
        if (layer.attn_q_bias.defined()) {
            q = add_tensors(q, layer.attn_q_bias);
            k = add_tensors(k, layer.attn_k_bias);
            v = add_tensors(v, layer.attn_v_bias);
        }

        // QK-norm
        if (layer.attn_q_norm.defined()) {
            PROF_BEGIN(profiler, "qk_norm");
            apply_qk_norm_inplace(q, layer.attn_q_norm, n_heads, head_dim);
            apply_qk_norm_inplace(k, layer.attn_k_norm, n_kv_heads, head_dim);
            PROF_END(profiler, "qk_norm");
        }

        // RoPE
        PROF_BEGIN(profiler, "rope");
        apply_rope_inplace(q, n_heads, head_dim, past_len);
        apply_rope_inplace(k, n_kv_heads, head_dim, past_len);
        PROF_END(profiler, "rope");

        // KV cache
        if (use_cache) {
            PROF_BEGIN(profiler, "kv_cache");
            kv_cache.append(layer_idx, k, v, use_cuda_);
            k = kv_cache.key_cache[layer_idx];
            v = kv_cache.value_cache[layer_idx];
            PROF_END(profiler, "kv_cache");
        }

        int64_t total_seq = use_cache ? (kv_cache.seq_len + seq_len) : k.size(0);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#ifdef PT_USE_CUDA
        if (use_cuda_ && q.is_cuda()) {
            PROF_BEGIN(profiler, "attention");
            auto output = at::empty_cuda({seq_len, q_dim}, q.dtype(), q.device().index());
            at::cuda::launch_causal_attention(
                q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                output.mutable_data_ptr<float>(),
                static_cast<int>(seq_len), static_cast<int>(total_seq),
                static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                static_cast<int>(head_dim),
                static_cast<int>(past_len), scale, nullptr);
            PROF_END(profiler, "attention");
            // Output projection
            PROF_BEGIN(profiler, "attn_output_proj");
            Tensor attn_result = matmul_q(output, layer.attn_output, layer.q_attn_output, true);
            PROF_END(profiler, "attn_output_proj");
            return attn_result;
        }
#endif
        // CPU fallback
        int64_t heads_per_group = n_heads / n_kv_heads;
        Tensor output = at::empty({seq_len, q_dim});
        float* out_data = output.mutable_data_ptr<float>();
        const float* q_data = q.data_ptr<float>();
        const float* k_data = k.data_ptr<float>();
        const float* v_data = v.data_ptr<float>();

        #pragma omp parallel for if(n_heads > 2)
        for (int64_t h = 0; h < n_heads; ++h) {
            int64_t kv_h = h / heads_per_group;
            std::vector<float> scores(total_seq);

            for (int64_t s = 0; s < seq_len; ++s) {
                const float* q_head = q_data + s * q_dim + h * head_dim;
                for (int64_t t = 0; t < total_seq; ++t) {
                    const float* k_head = k_data + t * kv_dim + kv_h * head_dim;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * k_head[d];
                    scores[t] = dot * scale;
                }
                int64_t max_pos = past_len + s;
                for (int64_t t = max_pos + 1; t < total_seq; ++t) scores[t] = -1e9f;

                float max_score = *std::max_element(scores.begin(), scores.begin() + total_seq);
                float sum_exp = 0.0f;
                for (int64_t t = 0; t < total_seq; ++t) {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                float inv_sum = 1.0f / (sum_exp + 1e-10f);
                for (int64_t t = 0; t < total_seq; ++t) scores[t] *= inv_sum;

                float* out_head = out_data + s * q_dim + h * head_dim;
                std::fill(out_head, out_head + head_dim, 0.0f);
                for (int64_t t = 0; t < total_seq; ++t) {
                    const float* v_head = v_data + t * kv_dim + kv_h * head_dim;
                    float w = scores[t];
                    for (int64_t d = 0; d < head_dim; ++d) out_head[d] += w * v_head[d];
                }
            }
        }
        return matmul(output, layer.attn_output, true);
    }

    // ========================================================================
    // RoPE (Rotary Position Embeddings) — applied in-place
    // ========================================================================

    void apply_rope_inplace(Tensor& x, int64_t n_heads, int64_t head_dim,
                            int64_t position_offset) {
#ifdef PT_USE_CUDA
        if (use_cuda_ && x.is_cuda()) {
            int64_t seq_len = x.size(0);
            at::cuda::launch_rope(
                x.mutable_data_ptr<float>(),
                static_cast<int>(seq_len), static_cast<int>(n_heads),
                static_cast<int>(head_dim),
                static_cast<int>(position_offset), config.rope_freq_base, nullptr);
            return;
        }
#endif
        int64_t seq_len = x.size(0);
        float* data = x.mutable_data_ptr<float>();
        float freq_base = config.rope_freq_base;
        for (int64_t s = 0; s < seq_len; ++s) {
            int64_t pos = position_offset + s;
            for (int64_t h = 0; h < n_heads; ++h) {
                float* head_data = data + s * (n_heads * head_dim) + h * head_dim;
                for (int64_t d = 0; d < head_dim / 2; ++d) {
                    float freq = 1.0f / std::pow(freq_base, 2.0f * d / head_dim);
                    float theta = pos * freq;
                    float cos_theta = std::cos(theta);
                    float sin_theta = std::sin(theta);
                    float x0 = head_data[2 * d];
                    float x1 = head_data[2 * d + 1];
                    head_data[2 * d]     = x0 * cos_theta - x1 * sin_theta;
                    head_data[2 * d + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }

    // ========================================================================
    // SwiGLU Feed-Forward Network
    // gate_proj, up_proj, down_proj
    // output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    // ========================================================================

    Tensor swiglu_ffn(const Tensor& x, const TransformerLayer& layer) {
        // gate + up projections
        PROF_BEGIN(profiler, "ffn_gate_up");
        Tensor gate = matmul_q(x, layer.ffn_gate, layer.q_ffn_gate, true);
        Tensor up = matmul_q(x, layer.ffn_up, layer.q_ffn_up, true);
        PROF_END(profiler, "ffn_gate_up");

        // Fused SiLU(gate) * up
#ifdef PT_USE_CUDA
        if (use_cuda_ && gate.is_cuda()) {
            PROF_BEGIN(profiler, "silu_mul");
            auto hidden = at::empty_cuda(gate.sizes().vec(), gate.dtype(), gate.device().index());
            at::cuda::launch_silu_mul(gate.data_ptr<float>(), up.data_ptr<float>(),
                                       hidden.mutable_data_ptr<float>(), gate.numel(), nullptr);
            PROF_END(profiler, "silu_mul");
            PROF_BEGIN(profiler, "ffn_down");
            Tensor result = matmul_q(hidden, layer.ffn_down, layer.q_ffn_down, true);
            PROF_END(profiler, "ffn_down");
            return result;
        }
#endif
        {
            int64_t n = gate.numel();
            float* gate_data = gate.mutable_data_ptr<float>();
            const float* up_data = up.data_ptr<float>();
#ifdef __AVX2__
            // AVX2 fused SiLU-Mul: silu(g) * u = g * sigmoid(g) * u
            // Fast exp approximation via polynomial (good enough for inference)
            int64_t i = 0;
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 neg_one = _mm256_set1_ps(-1.0f);
            for (; i + 7 < n; i += 8) {
                __m256 g = _mm256_loadu_ps(gate_data + i);
                __m256 u = _mm256_loadu_ps(up_data + i);
                // sigmoid(g) = 1/(1+exp(-g))
                // Use fast approximation: exp(-x) ≈ via Schraudolph/clamp
                __m256 neg_g = _mm256_mul_ps(g, neg_one);
                // Clamp to [-88, 88] to avoid overflow
                neg_g = _mm256_max_ps(neg_g, _mm256_set1_ps(-88.0f));
                neg_g = _mm256_min_ps(neg_g, _mm256_set1_ps(88.0f));
                // exp via Cephes-style polynomial (4th order, ~1e-4 accuracy)
                // exp(x) ≈ (1 + x/256)^256 ≈ use integer trick
                // Simpler: use scalar fallback for exp
                float tmp[8];
                _mm256_storeu_ps(tmp, neg_g);
                __m256 exp_neg_g = _mm256_set_ps(
                    std::exp(tmp[7]), std::exp(tmp[6]), std::exp(tmp[5]), std::exp(tmp[4]),
                    std::exp(tmp[3]), std::exp(tmp[2]), std::exp(tmp[1]), std::exp(tmp[0]));
                __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_g));
                __m256 silu = _mm256_mul_ps(g, sigmoid);
                __m256 result = _mm256_mul_ps(silu, u);
                _mm256_storeu_ps(gate_data + i, result);
            }
            for (; i < n; ++i) {
                float g = gate_data[i];
                gate_data[i] = (g / (1.0f + std::exp(-g))) * up_data[i];
            }
#else
            for (int64_t i = 0; i < n; ++i) {
                float g = gate_data[i];
                float silu = g / (1.0f + std::exp(-g));
                gate_data[i] = silu * up_data[i];
            }
#endif
            return matmul_q(gate, layer.ffn_down, layer.q_ffn_down, true);
        }
    }

    // ========================================================================
    // Tensor utilities (no autograd needed for inference)
    // ========================================================================

    Tensor add_tensors(const Tensor& a, const Tensor& b) {
#ifdef PT_USE_CUDA
        if (use_cuda_ && a.is_cuda()) {
            if (a.dim() == 2 && b.dim() == 1) {
                return at::cuda_ops::add_broadcast(a, b);
            }
            return at::cuda_ops::add(a, b);
        }
#endif
        if (a.dim() == 2 && b.dim() == 1) {
            // Broadcasting: [M, N] + [N]
            int64_t M = a.size(0), N = a.size(1);
            Tensor result = at::empty({M, N});
            const float* a_data = a.data_ptr<float>();
            const float* b_data = b.data_ptr<float>();
            float* out = result.mutable_data_ptr<float>();
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    out[i * N + j] = a_data[i * N + j] + b_data[j];
                }
            }
            return result;
        }

        // Element-wise add (same shape)
        int64_t n = a.numel();
        Tensor result = at::empty(a.sizes().vec());
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* out = result.mutable_data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) {
            out[i] = a_data[i] + b_data[i];
        }
        return result;
    }

    Tensor concat_tensors(const Tensor& a, const Tensor& b) {
#ifdef PT_USE_CUDA
        if (use_cuda_ && a.is_cuda()) {
            int64_t M1 = a.size(0), M2 = b.size(0), K = a.size(1);
            auto result = at::empty_cuda({M1 + M2, K}, a.dtype(), a.device().index());
            at::cuda::launch_concat(
                a.data_ptr<float>(), b.data_ptr<float>(),
                result.mutable_data_ptr<float>(), M1, M2, K, nullptr);
            return result;
        }
#endif
        int64_t M1 = a.size(0), M2 = b.size(0), K = a.size(1);
        Tensor result = at::empty({M1 + M2, K});
        float* out = result.mutable_data_ptr<float>();
        std::memcpy(out, a.data_ptr<float>(), M1 * K * sizeof(float));
        std::memcpy(out + M1 * K, b.data_ptr<float>(), M2 * K * sizeof(float));
        return result;
    }

    Tensor get_row(const Tensor& x, int64_t row) {
        // Extract single row from [M, N] → [N] (always returns CPU for sampling)
        Tensor x_cpu = x.is_cpu() ? x : to_cpu_tensor(x);
        int64_t N = x_cpu.size(1);
        Tensor result = at::empty({N});
        const float* src = x_cpu.data_ptr<float>() + row * N;
        std::memcpy(result.mutable_data_ptr<float>(), src, N * sizeof(float));
        return result;
    }

    // ========================================================================
    // Token sampling
    // ========================================================================

    int32_t sample_token(const Tensor& logits, float temperature, int top_k, float top_p) {
        int64_t vocab = logits.numel();
        const float* data = logits.data_ptr<float>();

        // Greedy (temperature = 0)
        if (temperature < 1e-6f) {
            int32_t best = 0;
            float best_val = data[0];
            for (int64_t i = 1; i < vocab; ++i) {
                if (data[i] > best_val) {
                    best_val = data[i];
                    best = static_cast<int32_t>(i);
                }
            }
            return best;
        }

        // Apply temperature
        std::vector<float> probs(vocab);
        for (int64_t i = 0; i < vocab; ++i) {
            probs[i] = data[i] / temperature;
        }

        // Top-k filtering
        if (top_k > 0 && top_k < vocab) {
            // Find the k-th largest value
            std::vector<float> sorted_probs(probs);
            std::partial_sort(sorted_probs.begin(), sorted_probs.begin() + top_k,
                            sorted_probs.end(), std::greater<float>());
            float threshold = sorted_probs[top_k - 1];
            for (int64_t i = 0; i < vocab; ++i) {
                if (probs[i] < threshold) probs[i] = -1e9f;
            }
        }

        // Softmax
        float max_val = *std::max_element(probs.begin(), probs.end());
        float sum_exp = 0.0f;
        for (int64_t i = 0; i < vocab; ++i) {
            probs[i] = std::exp(probs[i] - max_val);
            sum_exp += probs[i];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int64_t i = 0; i < vocab; ++i) {
            probs[i] *= inv_sum;
        }

        // Top-p (nucleus) filtering
        if (top_p < 1.0f) {
            std::vector<std::pair<float, int32_t>> sorted;
            sorted.reserve(vocab);
            for (int64_t i = 0; i < vocab; ++i) {
                if (probs[i] > 1e-10f) {
                    sorted.push_back({probs[i], static_cast<int32_t>(i)});
                }
            }
            std::sort(sorted.begin(), sorted.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

            float cumsum = 0.0f;
            std::vector<bool> keep(vocab, false);
            for (const auto& [p, idx] : sorted) {
                keep[idx] = true;
                cumsum += p;
                if (cumsum >= top_p) break;
            }
            for (int64_t i = 0; i < vocab; ++i) {
                if (!keep[i]) probs[i] = 0.0f;
            }

            // Renormalize
            sum_exp = 0.0f;
            for (int64_t i = 0; i < vocab; ++i) sum_exp += probs[i];
            if (sum_exp > 0) {
                inv_sum = 1.0f / sum_exp;
                for (int64_t i = 0; i < vocab; ++i) probs[i] *= inv_sum;
            }
        }

        // Sample from distribution
        static std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cumsum = 0.0f;
        for (int64_t i = 0; i < vocab; ++i) {
            cumsum += probs[i];
            if (cumsum >= r) return static_cast<int32_t>(i);
        }

        return static_cast<int32_t>(vocab - 1);
    }
};

// ============================================================================
// Convenience functions
// ============================================================================

inline GGUFModel load_gguf_model(const std::string& gguf_path) {
    GGUFModel model;
    model.load(gguf_path);
    return model;
}

inline GGUFModel load_ollama_model(const std::string& model_name) {
    GGUFModel model;
    model.load_ollama(model_name);
    return model;
}

} // namespace io
} // namespace torch
