#pragma once

#include "torch/io/gguf_loader.h"
#include "torch/io/gguf_dequant.h"
#include "torch/io/cpu_quant_gemv.h"
#include "torch/io/q8_soa_repack.h"
#include "torch/io/pt8_reader.h"
#include "torch/io/numa_weight_replica.h"
#include "torch/io/speculative_decode.h"
#include "torch/io/speculative_verify.h"
#include "torch/io/sliding_window_attn.h"
#include "torch/io/sparse_gemv.h"
#include "torch/io/ollama.h"
#include "torch/io/tokenizer.h"
#include "torch/io/inference_profiler.h"
#include "torch/distributed/ddp.h"
#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "aten/src/ATen/native/cpu/VectorizedOps.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#include "aten/src/ATen/cuda/CuBLASHandle.h"
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
    bool is_f16() const { return quant_type == 1; }   // GGML_TYPE_F16
    bool is_q8_0() const { return quant_type == 8; }  // GGML_TYPE_Q8_0

    bool mmap_owned = false;  // true if cpu_data points into mmap region (don't free!)

    // Optional: per-NUMA-node local copies of cpu_data, built lazily when
    // PT_NUMA_REPLICATE=1. When populated, GEMV can pick `replica.get(node)`
    // instead of `cpu_data` to read weights from thread's local DDR.
    torch::io::ReplicatedWeight numa_replica;

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
            // FP16 KV cache allocation (attention memory bandwidth reduction).
            // Decode path uses graph-compatible FP32 flash_decode (reads d_past_len from device) —
            // FP16 flag gates the fp16-reading kernel which is currently used for prefill only.
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
        // Bounds check: prevent writing past allocated cache
        if (seq_len + num_new > max_seq) {
            std::cerr << "[KVCache] ERROR: seq_len(" << seq_len << ") + num_new(" << num_new
                      << ") > max_seq(" << max_seq << "). Truncating." << std::endl;
            num_new = max_seq - seq_len;
            if (num_new <= 0) return;
        }
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

        // Cacheline-aligned allocation (64-byte). AVX2 only requires 32 B but
        // cachelines are 64 B on E8C2 and x86 — aligning scratch buffers to
        // cacheline boundary eliminates false sharing when adjacent thread
        // chunks write to y[i..i+chunk] regions.
        // (agent_4_threadpool_audit.md Q2 / P2)
        auto alloc = [](int64_t n) -> float* {
            // _aligned_malloc on Windows, posix_memalign on Linux
#ifdef _WIN32
            return static_cast<float*>(_aligned_malloc(n * sizeof(float), 64));
#else
            void* ptr = nullptr;
            posix_memalign(&ptr, 64, n * sizeof(float));
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
// Tensor-Parallel (DDP) configuration for multi-process CPU inference
// ============================================================================
// Splits transformer compute across N processes for throughput on ONE request:
//   - attn_q / attn_k / attn_v / ffn_gate / ffn_up: ROW-SPLIT (output dim)
//       rank r owns rows [r*rows/N, (r+1)*rows/N) — contiguous byte-slice of Q_K
//   - attn_output / ffn_down / output_weight: REPLICATED (full weight on every rank)
//       rank r fills its local column-slice of the input buffer, zeros elsewhere,
//       runs the full GEMV locally, then AllReduce-SUM across ranks.
//   - norms, biases, token_embedding: REPLICATED.
//
// Collective points (per layer):
//   1. After attention output proj: AllReduce(h_buf, SUM) across ranks.
//   2. After FFN down proj:         AllReduce(h_buf, SUM) across ranks.
// Logits are fully-replicated since final hidden state is fully-replicated.
//
// Constraints: num_heads % nprocs == 0 AND num_kv_heads % nprocs == 0.
// ============================================================================

struct TPSlicedWeight {
    // Row-sliced copy of a QuantizedWeight. Points into an owned malloc'd buffer
    // (never mmap). rows = local_rows, cols = full cols, stride = full row stride.
    void* cpu_data = nullptr;
    int64_t rows = 0;            // local rows (N / nprocs)
    int64_t cols = 0;            // same as full cols (K)
    int64_t row_stride_bytes = 0;
    int64_t total_bytes = 0;
    uint32_t quant_type = 0;
    bool valid = false;

    // Round 3 Agent 5: Q8 SoA4 4-row interleaved weight, populated only when
    // PT_Q8_SOA=1 at TP setup. When valid, forward path uses q8_soa4_gemv
    // instead of cpu_quant_gemv. Falls back to Q4_K kernel otherwise.
    cpu_quant::Q8SoA4 q8_soa;

    ~TPSlicedWeight() {
        if (cpu_data) std::free(cpu_data);
    }
    TPSlicedWeight() = default;
    TPSlicedWeight(const TPSlicedWeight&) = delete;
    TPSlicedWeight& operator=(const TPSlicedWeight&) = delete;
    TPSlicedWeight(TPSlicedWeight&& o) noexcept { swap_from(o); }
    TPSlicedWeight& operator=(TPSlicedWeight&& o) noexcept {
        if (this != &o) {
            if (cpu_data) std::free(cpu_data);
            cpu_data = nullptr;
            swap_from(o);
        }
        return *this;
    }
private:
    void swap_from(TPSlicedWeight& o) {
        cpu_data = o.cpu_data; o.cpu_data = nullptr;
        rows = o.rows; cols = o.cols;
        row_stride_bytes = o.row_stride_bytes;
        total_bytes = o.total_bytes;
        quant_type = o.quant_type;
        valid = o.valid;
        q8_soa = std::move(o.q8_soa);
    }
};

struct TPLayer {
    // N-dim row-sliced Q/K/V/gate/up (ColumnParallel-style: each rank gets its
    // output-dim slice; input is replicated; output is its own slice).
    TPSlicedWeight q_attn_q;   // rows = n_heads_local * head_dim
    TPSlicedWeight q_attn_k;   // rows = n_kv_heads_local * head_dim
    TPSlicedWeight q_attn_v;   // rows = n_kv_heads_local * head_dim
    TPSlicedWeight q_ffn_gate; // rows = inter_local
    TPSlicedWeight q_ffn_up;   // rows = inter_local

    // K-dim sliced attn_output and ffn_down (RowParallel-style: each rank gets
    // its input-dim K-slice of a full-N weight; input is its own slice; output
    // is a partial N-vector that must be AllReduce-sum'd across ranks).
    // Reuses TPSlicedWeight: cpu_data = malloc'd buffer for N×(local_blocks*144)
    // row_stride_bytes = local_blocks * 144 (local, not global)
    // rows = full N (4096 for attn_output; H=2560 for ffn_down)
    // cols = local K = local_blocks * 256
    TPSlicedWeight q_attn_output;  // K-slice of attn_output (K=q_dim=4096)
    TPSlicedWeight q_ffn_down;     // K-slice of ffn_down (K=inter=9728)

    // K-slice metadata (in super-block units = 256 elements each)
    int64_t attn_output_k_start = 0;  // in super-blocks
    int64_t attn_output_k_end = 0;
    int64_t attn_output_k_local = 0;  // = (k_end - k_start) * 256
    int64_t ffn_down_k_start = 0;
    int64_t ffn_down_k_end = 0;
    int64_t ffn_down_k_local = 0;
};

struct GGUFTPConfig {
    bool enabled = false;
    int rank = 0;
    int nprocs = 1;

    // Local head partitioning
    int64_t n_heads_local = 0;     // num_heads / nprocs
    int64_t n_kv_heads_local = 0;  // num_kv_heads / nprocs
    int64_t head_start = 0;        // rank * n_heads_local
    int64_t head_end = 0;          // head_start + n_heads_local
    int64_t kv_head_start = 0;
    int64_t kv_head_end = 0;

    // Local dim slices
    int64_t q_dim_local = 0;       // n_heads_local * head_dim
    int64_t kv_dim_local = 0;      // n_kv_heads_local * head_dim
    int64_t inter_local = 0;       // this rank's slice of intermediate (= local_blocks * 256,
                                   // may differ across ranks if inter/256 not divisible by nprocs)
    int64_t inter_offset = 0;      // this rank's start offset into full inter (= k_start * 256)

    // Per-rank KV cache (local KV dim)
    std::vector<at::Tensor> k_cache_local;  // per layer: [max_seq, kv_dim_local]
    std::vector<at::Tensor> v_cache_local;  // per layer: [max_seq, kv_dim_local]
    int64_t kv_seq_len = 0;
    int64_t kv_max_seq = 0;

    // Per-layer row-sliced quantized weights
    std::vector<TPLayer> layers;

    // Round 3 Option D: K-slice the output (lm_head) projection so each rank
    // reads only 1/N of its bytes. Currently 175 MB/token of output_weight
    // gets read replicated; K-slicing → 44 MB/rank → saves ~131 MB/token
    // aggregate bandwidth.
    TPSlicedWeight q_output_weight_k_slice;
    int64_t output_weight_k_start  = 0;
    int64_t output_weight_k_end    = 0;
    int64_t output_weight_k_local  = 0;

    // Scratch buffers sized to local dims
    std::vector<float> x_buf[2];       // [H] (hidden state, replicated)
    std::vector<float> q_local_buf;    // [q_dim_local]
    std::vector<float> k_local_buf;    // [kv_dim_local]
    std::vector<float> v_local_buf;    // [kv_dim_local]
    std::vector<float> attn_full_buf;  // [q_dim]  (zero-padded; local heads filled)
    std::vector<float> h_buf;          // [H]
    std::vector<float> gate_local_buf; // [inter_local]
    std::vector<float> up_local_buf;   // [inter_local]
    std::vector<float> silu_full_buf;  // [inter] (zero-padded; local slice filled)
    std::vector<float> logits_buf;     // [vocab]
    bool scratch_ready = false;

    // Round 3 Agent 5: Q8 SoA4 path scratch buffers. Sized at TP setup to
    // max K seen across all GEMVs (= hidden_size). Activation broadcast uses
    // 4 bytes per input element; sum_a is 4 bytes per 32-element block.
    std::vector<uint8_t> soa_act_b16;   // up to K_max*4 bytes
    std::vector<int32_t> soa_sum_a;     // up to K_max/32 ints
    float                soa_scale_a = 1.0f;

    // Round 4 Step 9: per-layer scratch (избегает 36 vec alloc/dealloc per token)
    std::vector<float> x_normed_buf;    // [H] для RMSNorm output в attention
    std::vector<float> silu_scratch_buf; // [inter_local] для SiLU(gate) * up результата
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
    void* lm_head_fp16_ = nullptr;  // FP16 dequantized lm_head for cuBLAS [vocab×hidden]
    std::vector<TransformerLayer> layers;

    // KV cache
    KVCache kv_cache;

    // Device
    bool use_cuda_ = false;
    bool use_quant_gemv_ = false;  // Q4_K_M decode acceleration
    bool use_fp16_weights_ = false;  // Dequant-at-load FP16 weights + cuBLAS HGEMV decode path
    bool use_llama_gemv_ = false;  // Route Q4_K GEMV through launch_q4km_persistent_gemv_v2 (llama.cpp-style)
    size_t fp16_weights_bytes_ = 0;  // Total FP16 weight VRAM (for reporting)
    bool output_weight_needs_float32_ = false;  // true if output.weight has no Q4_K_M

    // GGUF file path (for reloading raw quant data)
    std::string gguf_file_path_;
    QuantizedWeight q_output_weight;  // quantized output projection

    // Memory-mapped GGUF file (kept alive for zero-copy weight access)
    // When active, QuantizedWeight::cpu_data points directly into mmap'd region
    // No malloc/memcpy needed — OS pages in data on demand
    gguf::MmapHandle mmap_handle_;
    bool use_mmap_ = false;  // true if weights are mmap'd (don't free cpu_data!)

    // ========================================================================
    // PT8 native loader (Round 4, Agent C). When the input file is .pt8
    // (magic 'PT8\0'), load() routes through load_pt8() which mmaps the
    // file via this reader and points QuantizedWeight::cpu_data into the
    // mmap'd region. The reader is kept alive for the lifetime of the
    // model so that all zero-copy pointers stay valid.
    //
    // When use_pt8_ is true, init_tensor_parallel() detects the
    // PT8_TYPE_Q8_0_SOA4 layout and skips the on-load Q4_K → Q8 SoA4 repack
    // step (the file is already in the Q8 SoA4 byte layout).
    // ========================================================================
    PT8Reader pt8_reader_;
    bool use_pt8_ = false;

    // CUDA Graph for decode acceleration (eliminates kernel launch overhead)
#ifdef PT_USE_CUDA
    cudaGraph_t decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    // Pinned host memory for async H2D copies (Gemini 3.1 Pro: without pinned, cudaMemcpyAsync syncs)
    int64_t* h_past_len_pinned_ = nullptr;
    int* h_token_id_pinned_ = nullptr;
    bool graph_captured_ = false;
    int graph_token_id_ = 0;      // Updated before each graph launch
    int* d_token_id_ = nullptr;   // Device memory for token_id (graph-updateable)
    int64_t* d_past_len_ = nullptr;  // Device memory for past_len (graph-compatible RoPE/KV)
    cudaStream_t decode_stream_ = nullptr;

    void invalidate_graph() {
        if (decode_graph_exec_) {
            cudaGraphExecDestroy(decode_graph_exec_);
            decode_graph_exec_ = nullptr;
        }
        if (decode_graph_) {
            cudaGraphDestroy(decode_graph_);
            decode_graph_ = nullptr;
        }
        graph_captured_ = false;
        graph_token_id_ = 0;
        std::cerr << "[PromeGraph] Graph invalidated (KV cache reallocated)" << std::endl;
    }
#else
    void invalidate_graph() {}  // no-op for non-CUDA
#endif

    // Profiler (enabled with --profile flag)
    InferenceProfiler profiler;

    // Scratch pool for zero-allocation decode
    InferenceScratchPool scratch_;

    // CPU scratch pool for zero-allocation CPU decode
    CPUScratchPool cpu_scratch_;

    // Tensor-parallel (multi-process DDP) state for CPU inference.
    // When enabled, forward_decode_cpu_tp() is used instead of forward_decode_cpu().
    GGUFTPConfig tp_;
    bool tp_enabled() const { return tp_.enabled; }

    // === Algorithmic acceleration ===
    LowRankOutputProj low_rank_output_;      // Speculative decode via low-rank output proj
    SlidingWindowAttention sliding_window_;   // Sliding window attention
    SparseQ4KWeight sparse_output_;           // Sparse GEMV for output projection
    std::vector<SparseQ4KWeight> sparse_ffn_; // Sparse GEMV for FFN layers
    bool use_speculative_output_ = false;     // Enable speculative output decode
    bool use_sparse_gemv_ = false;            // Enable sparse GEMV
    int64_t sliding_window_size_ = 0;         // 0 = disabled

    // Phase 7.5 — draft model for multi-token speculative verify. When
    // PT_SPEC_DRAFT_PATH env points at a smaller same-family GGUF, we load
    // it here and feed its argmax predictions into spec_decode_step_cpu
    // instead of NgramDraft. Same tokenizer is REQUIRED — we compare raw
    // token ids, no decode/encode in the hot path.
    std::unique_ptr<GGUFModel> draft_model_;

    // Ensure draft is loaded and its KV cache is allocated for max_seq.
    // Called lazily from generate() when PT_SPEC_K>1 and draft path is set.
    void ensure_draft_model(int64_t max_seq) {
        if (draft_model_) {
            // Already loaded — only re-allocate KV if the new max_seq is larger.
            if (draft_model_->kv_cache.max_seq < max_seq) {
                int64_t kv_dim = draft_model_->config.num_kv_heads
                               * draft_model_->config.head_dim;
                draft_model_->kv_cache.allocate(
                    draft_model_->config.num_layers, max_seq, kv_dim, false);
            }
            return;
        }
        const char* p = std::getenv("PT_SPEC_DRAFT_PATH");
        if (!p || !p[0]) return;
        std::fprintf(stderr, "[spec-draft] loading draft model: %s\n", p);
        draft_model_.reset(new GGUFModel());
        draft_model_->load(p);
        int64_t kv_dim = draft_model_->config.num_kv_heads
                       * draft_model_->config.head_dim;
        draft_model_->kv_cache.allocate(
            draft_model_->config.num_layers, max_seq, kv_dim, false);
        std::fprintf(stderr, "[spec-draft] loaded. vocab=%ld hidden=%ld layers=%ld\n",
            (long)draft_model_->config.vocab_size,
            (long)draft_model_->config.hidden_size,
            (long)draft_model_->config.num_layers);
    }

    // Predict K-1 drafts via draft_model. Pre-condition: draft_model's KV
    // is synced to main's KV length BEFORE current_token. Uses a temporary
    // token that we forward on draft to produce draft_1, then draft_1 →
    // draft_2, etc. Returns drafts in order. Stops early if draft's vocab
    // is smaller than main's (shouldn't happen in same-family pairs).
    std::vector<int64_t> draft_predict_model(int64_t current_token, int K_minus_1) {
        std::vector<int64_t> out;
        if (!draft_model_ || K_minus_1 <= 0) return out;
        out.reserve(K_minus_1);
        const int64_t V_draft = draft_model_->config.vocab_size;
        const int64_t V_limit = std::min(V_draft, config.vocab_size);
        int64_t tok = current_token;
        for (int i = 0; i < K_minus_1; ++i) {
            Tensor lg = draft_model_->forward_decode_cpu(tok);
            const float* lr = lg.data_ptr<float>();
            int64_t best = 0; float best_v = lr[0];
            for (int64_t v = 1; v < V_limit; ++v) {
                if (lr[v] > best_v) { best_v = lr[v]; best = v; }
            }
            out.push_back(best);
            tok = best;
        }
        return out;
    }

    // After a spec step, main committed j+1 tokens (accepted drafts + best_j
    // OR K accepted + best_{K-1}). Draft's KV has been advanced by K-1
    // during draft_predict_model (for all the proposed drafts, only the
    // first j of which turned out to be correct). Sync: rewind draft to
    // new_main_seq_len - 1 (dropping K-1 - j slots), then forward the last
    // committed token so draft's KV matches main's exactly.
    void draft_sync_after_step(int64_t main_seq_len_after,
                                int64_t last_committed_token,
                                int drafts_proposed) {
        if (!draft_model_) return;
        const int64_t expected_after = main_seq_len_after;
        const int64_t draft_seq      = draft_model_->kv_cache.seq_len;
        if (draft_seq < expected_after) return;   // shouldn't happen
        // draft is AT expected_after if all drafts matched + we want
        // best_{K-1}'s KV appended. Or DRAFT IS PAST expected_after if a
        // mismatch rejected drafts: drop the excess first, then forward
        // last_committed_token to add its KV.
        draft_model_->kv_cache.seq_len = expected_after - 1;
        (void)draft_model_->forward_decode_cpu(last_committed_token);
        (void)drafts_proposed;
    }

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
            else if (type == gguf::GGML_TYPE_F16) {
                // FP16 weight — upload raw bytes to GPU for FP16 GEMV
                auto raw = reader.load_raw_tensor(name);
                auto shape = raw.shape;
                qw.rows = shape[0];
                qw.cols = shape[1];
                qw.row_stride_bytes = qw.cols * 2;  // 2 bytes per FP16
                qw.total_bytes = static_cast<int64_t>(raw.data.size());
                qw.quant_type = type;

                cudaError_t err = cudaMalloc(&qw.gpu_data, qw.total_bytes);
                if (err == cudaSuccess) {
                    cudaMemcpy(qw.gpu_data, raw.data.data(), qw.total_bytes, cudaMemcpyHostToDevice);
                    qw.valid = true;
                }
                return;
            }
            else return;  // F32/other → handled via float32 fallback

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

        // Prepare lm_head FP16 for cuBLAS HGEMV (Tensor Cores)
        // Handles both: quantized Q4_K output weights AND tied embeddings (FP32)
        if (q_output_weight.valid && q_output_weight.is_q4k() && q_output_weight.gpu_data) {
            int K_lm = static_cast<int>(q_output_weight.cols);
            int N_lm = static_cast<int>(q_output_weight.rows);
            size_t fp16_bytes = static_cast<size_t>(N_lm) * K_lm * sizeof(uint16_t);
            cudaError_t err = cudaMalloc(&lm_head_fp16_, fp16_bytes);
            if (err == cudaSuccess) {
                at::cuda::launch_dequant_q4k_to_fp16(
                    q_output_weight.gpu_data, lm_head_fp16_,
                    K_lm, N_lm, q_output_weight.row_stride_bytes, nullptr);
                cudaDeviceSynchronize();
                std::cout << "[Quant] lm_head dequantized to FP16: "
                          << (fp16_bytes / (1024*1024)) << " MB" << std::endl;
            } else {
                std::cerr << "[Quant] lm_head FP16 alloc failed (need "
                          << (fp16_bytes / (1024*1024)) << " MB)" << std::endl;
                lm_head_fp16_ = nullptr;
            }
        } else if (output_weight.defined() && output_weight.is_cuda()) {
            // Tied weights: output = embedding (FP32 on GPU). Convert to FP16 for cuBLAS.
            int64_t V = config.vocab_size;
            int64_t H_dim = config.hidden_size;
            size_t fp16_bytes = V * H_dim * sizeof(uint16_t);
            cudaError_t err = cudaMalloc(&lm_head_fp16_, fp16_bytes);
            if (err == cudaSuccess) {
                // TODO: FP32→FP16 conversion kernel for cuBLAS lm_head
                // For now, free — need a proper conversion kernel
                std::cout << "[Quant] lm_head FP16 alloc OK (" << (fp16_bytes/(1024*1024))
                          << " MB) — conversion not yet implemented" << std::endl;
                cudaFree(lm_head_fp16_);
                lm_head_fp16_ = nullptr;
            }
        }
#else
        std::cerr << "[Quant] CUDA not available" << std::endl;
#endif
    }

    // ========================================================================
    // Dequantize ALL layer Q4_K weights to FP16 on GPU (one-shot at load).
    // Enables cuBLAS HGEMV decode path (Tensor Cores) — expected 2-3x speedup
    // on A100 vs custom Q4_K fused kernels. Call AFTER load_quantized_to_cuda().
    //
    // VRAM cost per weight: rows * cols * 2 bytes (FP16). Total for qwen3:4b:
    // 7 weights * 28 layers ~ 196 weights, ~5-6 GB extra FP16 buffers.
    //
    // On OOM: frees any fp16_data already allocated, resets use_fp16_weights_
    // to false, and returns false. Caller should fall back to the quant path.
    //
    // Returns true on success, false on fallback-to-quant.
    // ========================================================================
    bool dequant_all_to_fp16(double max_vram_fraction = 0.85) {
#ifdef PT_USE_CUDA
        if (!use_quant_gemv_) {
            std::cerr << "[FP16] Quantized weights not loaded — call load_quantized_to_cuda() first" << std::endl;
            return false;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // Estimate total FP16 bytes required before touching any malloc.
        size_t need_bytes = 0;
        auto add_w = [&](const QuantizedWeight& qw) {
            if (qw.valid && qw.is_q4k() && qw.gpu_data && !qw.fp16_data) {
                need_bytes += static_cast<size_t>(qw.rows) * qw.cols * sizeof(uint16_t);
            }
        };
        for (const auto& layer : layers) {
            add_w(layer.q_attn_q); add_w(layer.q_attn_k); add_w(layer.q_attn_v);
            add_w(layer.q_attn_output);
            add_w(layer.q_ffn_gate); add_w(layer.q_ffn_up); add_w(layer.q_ffn_down);
        }

        size_t vram_free = 0, vram_total = 0;
        cudaMemGetInfo(&vram_free, &vram_total);
        double need_mb = need_bytes / (1024.0 * 1024.0);
        double free_mb = vram_free / (1024.0 * 1024.0);
        double total_mb = vram_total / (1024.0 * 1024.0);
        std::cout << "[FP16] Dequant plan: need " << std::fixed << std::setprecision(1)
                  << need_mb << " MB, free " << free_mb << " / " << total_mb << " MB" << std::endl;

        // Headroom check: leave room for KV cache + activations + cuBLAS workspace.
        size_t budget = static_cast<size_t>(vram_total * max_vram_fraction);
        size_t used = vram_total - vram_free;
        if (used + need_bytes > budget) {
            std::cerr << "[FP16] Would exceed " << (max_vram_fraction*100.0) << "% VRAM budget ("
                      << ((used + need_bytes) / (1024.0*1024.0)) << " MB > "
                      << (budget / (1024.0*1024.0)) << " MB). Falling back to quant path." << std::endl;
            use_fp16_weights_ = false;
            return false;
        }

        std::cout << "[FP16] Dequantizing Q4_K -> FP16 on GPU..." << std::endl;
        size_t allocated_bytes = 0;
        bool ok = true;

        auto dequant_one = [&](QuantizedWeight& qw) -> bool {
            if (!ok) return false;
            if (!qw.valid || !qw.is_q4k() || !qw.gpu_data) return true;  // skip non-Q4K
            if (qw.fp16_data) return true;  // already done
            size_t bytes = static_cast<size_t>(qw.rows) * qw.cols * sizeof(uint16_t);
            cudaError_t err = cudaMalloc(&qw.fp16_data, bytes);
            if (err != cudaSuccess) {
                std::cerr << "[FP16] cudaMalloc failed (" << (bytes/(1024*1024)) << " MB): "
                          << cudaGetErrorString(err) << std::endl;
                qw.fp16_data = nullptr;
                ok = false;
                return false;
            }
            at::cuda::launch_dequant_q4k_to_fp16(
                qw.gpu_data, qw.fp16_data,
                static_cast<int>(qw.cols), static_cast<int>(qw.rows),
                qw.row_stride_bytes, nullptr);
            allocated_bytes += bytes;
            return true;
        };

        for (auto& layer : layers) {
            if (!dequant_one(layer.q_attn_q)) break;
            if (!dequant_one(layer.q_attn_k)) break;
            if (!dequant_one(layer.q_attn_v)) break;
            if (!dequant_one(layer.q_attn_output)) break;
            if (!dequant_one(layer.q_ffn_gate)) break;
            if (!dequant_one(layer.q_ffn_up)) break;
            if (!dequant_one(layer.q_ffn_down)) break;
        }
        cudaDeviceSynchronize();

        if (!ok) {
            std::cerr << "[FP16] OOM or dequant failure — releasing " << (allocated_bytes/(1024*1024))
                      << " MB and falling back to quant path." << std::endl;
            for (auto& layer : layers) {
                auto free_one = [](QuantizedWeight& qw) {
                    if (qw.fp16_data) { cudaFree(qw.fp16_data); qw.fp16_data = nullptr; }
                };
                free_one(layer.q_attn_q); free_one(layer.q_attn_k); free_one(layer.q_attn_v);
                free_one(layer.q_attn_output);
                free_one(layer.q_ffn_gate); free_one(layer.q_ffn_up); free_one(layer.q_ffn_down);
            }
            use_fp16_weights_ = false;
            fp16_weights_bytes_ = 0;
            return false;
        }

        use_fp16_weights_ = true;
        fp16_weights_bytes_ = allocated_bytes;
        // Force graph re-capture since hot path dispatches differ now.
        invalidate_graph();

        auto t_end = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t_end - t_start).count();
        cudaMemGetInfo(&vram_free, &vram_total);
        double used_gb = (vram_total - vram_free) / (1024.0 * 1024.0 * 1024.0);
        double total_gb = vram_total / (1024.0 * 1024.0 * 1024.0);
        std::cout << "[FP16] Dequantized in " << s << "s — added "
                  << (allocated_bytes / (1024.0*1024.0)) << " MB, VRAM "
                  << std::fixed << std::setprecision(1) << used_gb << " / " << total_gb << " GB" << std::endl;
        return true;
#else
        (void)max_vram_fraction;
        std::cerr << "[FP16] CUDA not available" << std::endl;
        return false;
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
        } else if (config.tie_word_embeddings && reader.has_tensor("token_embd.weight")) {
            upload_quant_cpu("token_embd.weight", q_output_weight);
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
            else if (type == gguf::GGML_TYPE_F16) {
                // FP16 weight — upload raw bytes to GPU for cuBLAS HGEMV
                auto shape = info.shape();
                qw.rows = shape[0];
                qw.cols = shape[1];
                qw.row_stride_bytes = qw.cols * 2;  // FP16: 2 bytes per element
                qw.total_bytes = info.data_bytes();
                qw.quant_type = type;
                uint64_t abs_offset = reader.data_offset + info.offset;
                qw.cpu_data = const_cast<void*>(mmap_handle_.at_offset(abs_offset));
                qw.mmap_owned = true;
                qw.valid = (qw.cpu_data != nullptr);
                return;
            }
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
        } else if (config.tie_word_embeddings && reader.has_tensor("token_embd.weight")) {
            // Tied output weights: use quantized token embedding as output projection.
            // Without this, the TP forward falls back to a scalar main-thread FP32
            // GEMV for vocab=152k — 500+ ms/token on Elbrus, the single biggest
            // cost of TP decode. Fixing the tie saves 50%+ of tok time.
            map_quant_mmap("token_embd.weight", q_output_weight);
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
        // Round 4 (Agent C): auto-detect PT8 vs GGUF by magic bytes. Set
        // PT_FORMAT_AUTO=0 to force the GGUF path even when the file's
        // first 8 bytes look like a .pt8 (debug only — do not use in prod).
        const char* auto_env = std::getenv("PT_FORMAT_AUTO");
        const bool  auto_detect = !auto_env || auto_env[0] != '0';
        if (auto_detect && PT8Reader::is_pt8_file(gguf_path)) {
            std::cout << "[load] PT8 magic detected — routing through "
                         "load_pt8() (zero-repack path)" << std::endl;
            load_pt8(gguf_path);
            return;
        }

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

        // Optional: replicate hot weights across NUMA nodes (PT_NUMA_REPLICATE=1).
        // Each worker thread then reads from its local DDR controller at full
        // per-channel bandwidth instead of contending through a single node's
        // DDR via the inter-chip interconnect.
        replicate_weights_for_numa();

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "\n[Model] Loaded in " << (ms / 1000.0) << " seconds" << std::endl;
    }

    // ========================================================================
    // Load from a PromeTorch .pt8 file (Round 4 Agent C, 2026-04-30).
    //
    // The .pt8 layout (header + tensor data + tail tensor table) is parsed
    // by torch::io::PT8Reader, which mmaps the file zero-copy. For every
    // expected weight name we point QuantizedWeight::cpu_data /
    // mmap-friendly Tensor::from_blob into the mapped region.
    //
    // Agent A's primary contract: when a 2D weight is stored as
    // PT8_TYPE_Q8_0_SOA4, the bytes are byte-identical to the in-memory
    // cpu_quant::Q8SoA4 layout — init_tensor_parallel() will detect
    // the synthetic quant_type and skip the repack step.
    //
    // What works in this v1 (passthrough variants):
    //   - F32 norms / biases / embeddings — direct from_blob view
    //   - F16 token_embd (when emitted as PT8_TYPE_F16) — small dequant
    //   - Q4_K passthrough (when the encoder is registered, otherwise
    //     this branch is never hit and we fall back gracefully)
    //
    // What's reserved for Agent A's encoder register pass:
    //   - PT8_TYPE_Q4K_SOA4 (the 0.6875 B/param hot dtype, spec §10)
    //   - PT8_TYPE_Q8_0_SOA4 (1.75 B/param, ready microbench-validated path)
    //
    // Either way, the loader builds a complete, decode-ready model — Q8 SoA4
    // weights live directly inside the mmap, no per-rank malloc duplication.
    // ========================================================================
    void load_pt8(const std::string& pt8_path) {
        gguf_file_path_ = pt8_path;  // Stored for diagnostic/log purposes only
        auto t_start = std::chrono::high_resolution_clock::now();

        if (!pt8_reader_.open(pt8_path)) {
            throw std::runtime_error("[load_pt8] failed to open " + pt8_path);
        }
        use_pt8_ = true;

        const auto& hdr = pt8_reader_.header();
        std::cout << "[pt8] opened " << pt8_path << " (" << hdr.tensor_count
                  << " tensors, " << (pt8_reader_.mmap_size() / (1024 * 1024))
                  << " MB mmap)" << std::endl;

        // ====================================================================
        // 1. Parse model config from PT8 metadata blob, with GGUF fallback.
        //    Agent A's spec §5 calls for a metadata KV section; the current
        //    Agent B writer doesn't yet emit it. Until it does, we recover
        //    the config from a sibling .gguf or by inspecting tensor shapes.
        //
        //    Path 1 (preferred): companion .gguf at same basename. This is
        //    by far the simplest interim contract — the converter already
        //    consumes the .gguf, so the user has it.
        //
        //    Path 2 (fallback): infer from tensor shapes alone. Used when no
        //    .gguf is alongside (acceptance §10: "prometorch run model.pt8"
        //    standalone). Agent A's spec promises a metadata KV; once Agent B
        //    writes it, we'll prefer that over both paths above.
        // ====================================================================
        std::string gguf_companion;
        {
            std::string p = pt8_path;
            size_t dot = p.find_last_of('.');
            if (dot != std::string::npos) p.erase(dot);
            gguf_companion = p + ".gguf";
        }

        bool used_gguf_companion = false;
        {
            FILE* f = std::fopen(gguf_companion.c_str(), "rb");
            if (f) {
                std::fclose(f);
                std::cout << "[pt8] reading config from companion GGUF: "
                          << gguf_companion << std::endl;
                gguf::GGUFReader cfg_reader;
                cfg_reader.open(gguf_companion);
                config.parse(cfg_reader);
                tokenizer.load(cfg_reader);
                if (cfg_reader.has_tensor("token_embd.weight")) {
                    auto& info = cfg_reader.get_tensor_info("token_embd.weight");
                    auto shape = info.shape();
                    config.vocab_size = shape[0];
                }
                used_gguf_companion = true;
            }
        }
        if (!used_gguf_companion) {
            // Minimal shape-inference fallback. Until Agent B emits metadata
            // KV (spec §5), we infer dimensions from the token_embd tensor.
            const auto* embd = pt8_reader_.find("token_embd.weight");
            if (!embd || embd->dims.size() != 2) {
                throw std::runtime_error(
                    "[load_pt8] no companion .gguf and no token_embd.weight in "
                    "PT8 file — cannot infer config. Place the source GGUF "
                    "alongside the .pt8 (same basename) for now.");
            }
            config.vocab_size  = embd->dims[0];
            config.hidden_size = embd->dims[1];
            // Rest of config (num_heads/layers/...) requires the metadata KV
            // section. Without it, the loader cannot construct an inference-
            // ready model. Throw with a clear message.
            throw std::runtime_error(
                "[load_pt8] PT8 metadata KV section not yet present in "
                "Agent B's writer. Add a sibling " + gguf_companion +
                " for now.");
        }

        config.print();

        // ====================================================================
        // 2. FP weights (norms, embeddings, biases) — zero-copy from_blob.
        //    Falls back to companion-GGUF dequant if a tensor is missing
        //    from the .pt8 (Agent B may not yet emit every tensor).
        // ====================================================================
        load_pt8_fp_weights_(used_gguf_companion ? gguf_companion : "");

        // ====================================================================
        // 3. Quantized weight 'mmap' — point QuantizedWeight::cpu_data into
        //    the .pt8 mmap region. quant_type encodes the PT8 tag:
        //      - passthrough Q4_K (raw GGML bytes) → quant_type = 12
        //      - PT8_TYPE_Q8_0_SOA4 (already-repacked) → quant_type = 0xPT8 + tag
        //    init_tensor_parallel() learns to dispatch on either.
        // ====================================================================
        load_pt8_quant_weights_();

        use_quant_gemv_ = true;
        use_mmap_ = true;

        // NUMA replication mirrors the GGUF path. For PT8_TYPE_Q8_0_SOA4 the
        // bytes are already in the runtime layout, so a memcpy gives us
        // node-local Q8 SoA4 weights with no per-token cross-chip traffic.
        replicate_weights_for_numa();

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "\n[Model] PT8 loaded in " << (ms / 1000.0) << " seconds"
                  << " (zero repack, mmap_size="
                  << (pt8_reader_.mmap_size() / (1024 * 1024)) << " MB)"
                  << std::endl;
    }

private:
    // ------------------------------------------------------------------
    // load_pt8_fp_weights_ / load_pt8_quant_weights_ — internal helpers
    // ------------------------------------------------------------------

    // Build an owned Tensor by copying the mmap'd float bytes. Norms /
    // embeddings / biases are tiny (< 100 MB total) so the copy cost is
    // negligible vs the convenience of independent lifetime from the mmap.
    Tensor pt8_fp32_view_(const PT8TensorRecord& r) const {
        const float* p = reinterpret_cast<const float*>(
            pt8_reader_.tensor_data(r.name));
        std::vector<int64_t> sizes(r.dims.begin(), r.dims.end());
        Tensor out = at::empty(sizes);  // default kFloat
        size_t n = 1;
        for (auto d : sizes) n *= static_cast<size_t>(d);
        std::memcpy(out.mutable_data_ptr<float>(), p, n * sizeof(float));
        return out;
    }

    // FP16 → FP32 dequant view. Used for token_embd if writer emits it
    // as PT8_TYPE_F16 (Agent A spec §6).
    Tensor pt8_fp16_to_fp32_(const PT8TensorRecord& r) const {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(
            pt8_reader_.tensor_data(r.name));
        size_t n_elems = 1;
        for (auto d : r.dims) n_elems *= static_cast<size_t>(d);
        std::vector<int64_t> sizes(r.dims.begin(), r.dims.end());
        Tensor out = at::empty(sizes);
        float* dst = out.mutable_data_ptr<float>();
        for (size_t i = 0; i < n_elems; ++i) {
            dst[i] = gguf::fp16_to_fp32(src[i]);
        }
        return out;
    }

    void load_pt8_fp_weights_(const std::string& gguf_companion_for_fallback) {
        std::cout << "\n[pt8] Loading FP weights..." << std::endl;

        // We need a GGUFReader fallback for tensors the .pt8 doesn't yet
        // carry (e.g. tokenizer-only artefacts). Open lazily.
        std::unique_ptr<gguf::GGUFReader> fallback_reader;
        auto get_fallback = [&]() -> gguf::GGUFReader* {
            if (!fallback_reader && !gguf_companion_for_fallback.empty()) {
                fallback_reader.reset(new gguf::GGUFReader());
                fallback_reader->open(gguf_companion_for_fallback);
            }
            return fallback_reader.get();
        };

        auto load_fp_named = [&](const std::string& name, Tensor& dst) {
            const auto* r = pt8_reader_.find(name);
            if (r) {
                if (r->pt8_type == PT8_TYPE_F32) {
                    dst = pt8_fp32_view_(*r);
                    return true;
                }
                if (r->pt8_type == PT8_TYPE_F16) {
                    dst = pt8_fp16_to_fp32_(*r);
                    return true;
                }
                // BF16 / Quant fall through to fallback.
            }
            if (auto* fb = get_fallback()) {
                if (fb->has_tensor(name)) {
                    dst = fb->load_tensor(name);
                    return true;
                }
            }
            return false;
        };

        load_fp_named("token_embd.weight", token_embedding);
        if (token_embedding.defined()) {
            std::cout << "  token_embd: [" << token_embedding.size(0) << ", "
                      << token_embedding.size(1) << "]" << std::endl;
        }

        load_fp_named("output_norm.weight", output_norm);

        // Tied output: when output.weight isn't its own tensor, alias to
        // token_embd. Matches the GGUF path (Qwen3, Gemma, etc.).
        if (!load_fp_named("output.weight", output_weight)) {
            output_weight = token_embedding;
            config.tie_word_embeddings = true;
            std::cout << "  output: tied to token_embd" << std::endl;
        }

        layers.resize(config.num_layers);
        for (int64_t i = 0; i < config.num_layers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";
            auto& layer = layers[i];

            load_fp_named(prefix + "attn_norm.weight", layer.attn_norm);
            load_fp_named(prefix + "ffn_norm.weight",  layer.ffn_norm);
            // Optional norms / biases — silent no-op if absent.
            load_fp_named(prefix + "attn_q_norm.weight", layer.attn_q_norm);
            load_fp_named(prefix + "attn_k_norm.weight", layer.attn_k_norm);
            load_fp_named(prefix + "post_attention_norm.weight",
                          layer.post_attention_norm);
            load_fp_named(prefix + "post_ffw_norm.weight",
                          layer.post_ffw_norm);
            load_fp_named(prefix + "attn_q.bias", layer.attn_q_bias);
            load_fp_named(prefix + "attn_k.bias", layer.attn_k_bias);
            load_fp_named(prefix + "attn_v.bias", layer.attn_v_bias);

            if ((i + 1) % 5 == 0 || i == config.num_layers - 1) {
                std::cout << "  Layer " << (i + 1) << "/" << config.num_layers
                          << " FP weights loaded" << std::endl;
            }
        }
    }

    // Map one quant tensor from the PT8 mmap into a QuantizedWeight without
    // copying. Sets quant_type to either:
    //   - the original GGML type id (12 = Q4_K, 14 = Q6_K, 8 = Q8_0) for
    //     passthrough variants — existing dispatch in init_tensor_parallel
    //     and forward_decode_cpu_tp Just Works
    //   - 0x100 + Pt8Type for SoA4-native layouts. init_tensor_parallel
    //     branches on this to skip repack.
    bool map_pt8_quant_(const std::string& name, QuantizedWeight& qw) {
        const auto* r = pt8_reader_.find(name);
        if (!r) return false;
        const void* data = pt8_reader_.tensor_data(name);
        if (!data) return false;

        if (r->dims.size() != 2) return false;
        qw.rows         = r->dims[0];
        qw.cols         = r->dims[1];
        qw.total_bytes  = static_cast<int64_t>(r->data_size);
        qw.cpu_data     = const_cast<void*>(data);
        qw.mmap_owned   = true;
        qw.row_stride_bytes = static_cast<int64_t>(r->row_stride);

        // Map PT8 dtype tag to a quant_type the rest of the pipeline groks.
        // Passthrough types preserve their original GGML id so the GGUF
        // dispatch path is untouched.
        switch (r->pt8_type) {
            case PT8_TYPE_F32:
                qw.quant_type = 0;   // GGML_TYPE_F32 — caller should treat as raw FP32
                break;
            case PT8_TYPE_F16:
                qw.quant_type = 1;   // GGML_TYPE_F16
                if (qw.row_stride_bytes == 0)
                    qw.row_stride_bytes = qw.cols * 2;
                break;
            case PT8_TYPE_Q8_0_SOA4:
                // Synthetic ID — init_tensor_parallel sees this and skips
                // the on-the-fly Q4_K → Q8 SoA4 repack (already done at
                // conversion time). Forward path then reads the bytes
                // directly into a Q8SoA4 view of the mmap.
                qw.quant_type = 0x100u | PT8_TYPE_Q8_0_SOA4;
                break;
            case PT8_TYPE_Q4K_SOA4:
                qw.quant_type = 0x100u | PT8_TYPE_Q4K_SOA4;
                break;
            default:
                std::cerr << "[pt8] unknown pt8_type " << r->pt8_type
                          << " for " << name << " — skipping" << std::endl;
                return false;
        }
        qw.valid = true;
        return true;
    }

    void load_pt8_quant_weights_() {
        std::cout << "[pt8] Mapping quantized weights (zero-copy)..." << std::endl;
        int64_t total_mapped = 0;
        int64_t soa4_count   = 0;

        auto try_map = [&](const std::string& name, QuantizedWeight& qw) {
            if (map_pt8_quant_(name, qw)) {
                total_mapped += qw.total_bytes;
                if ((qw.quant_type & 0xF00) == 0x100) ++soa4_count;
            }
        };

        for (int64_t i = 0; i < config.num_layers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";
            auto& layer = layers[i];
            try_map(prefix + "attn_q.weight",      layer.q_attn_q);
            try_map(prefix + "attn_k.weight",      layer.q_attn_k);
            try_map(prefix + "attn_v.weight",      layer.q_attn_v);
            try_map(prefix + "attn_output.weight", layer.q_attn_output);
            try_map(prefix + "ffn_gate.weight",    layer.q_ffn_gate);
            try_map(prefix + "ffn_up.weight",      layer.q_ffn_up);
            try_map(prefix + "ffn_down.weight",    layer.q_ffn_down);
        }

        if (pt8_reader_.has("output.weight")) {
            try_map("output.weight", q_output_weight);
        } else if (config.tie_word_embeddings &&
                   pt8_reader_.has("token_embd.weight")) {
            try_map("token_embd.weight", q_output_weight);
        }

        std::cout << "[pt8] Mapped " << (total_mapped / (1024 * 1024))
                  << " MB of quantized weights"
                  << " (" << soa4_count << " native SoA4 tensors — repack skipped)"
                  << std::endl;
    }

public:

    // Replicate the hot CPU weights (ffn_gate/up/down, attn_output) across NUMA
    // nodes. Safe no-op when PT_NUMA_REPLICATE is not set or libnuma missing.
    // Only replicates weights whose cpu_data is non-null (CPU-loaded models).
    void replicate_weights_for_numa() {
        int n = torch::io::numa_replicate_count();
        if (n <= 1) return;

        auto rep = [&](QuantizedWeight& qw) {
            if (!qw.valid || !qw.cpu_data || qw.total_bytes == 0) return;
            qw.numa_replica.replicate(qw.cpu_data, qw.total_bytes);
        };

        size_t total_bytes = 0;
        for (auto& layer : layers) {
            rep(layer.q_ffn_gate);
            rep(layer.q_ffn_up);
            rep(layer.q_ffn_down);
            rep(layer.q_attn_output);
            // Q/K/V: agent_3_numa_audit.md flagged ~264 MB/token of
            // attn_q/k/v reads as 30% cross-chip when replication is on for
            // the other weights. Replicate these too now that infra supports it.
            rep(layer.q_attn_q);
            rep(layer.q_attn_k);
            rep(layer.q_attn_v);
            total_bytes += layer.q_ffn_gate.total_bytes +
                           layer.q_ffn_up.total_bytes +
                           layer.q_ffn_down.total_bytes +
                           layer.q_attn_output.total_bytes +
                           layer.q_attn_q.total_bytes +
                           layer.q_attn_k.total_bytes +
                           layer.q_attn_v.total_bytes;
        }
        // LM head: 208 MB of Q4_K read once per decoded token from whatever
        // NUMA node got the first-touch on load. Ranks 1-3 then pay cross-
        // chip latency for every token — agent_2 measured that at ~12% of
        // serial budget.
        rep(q_output_weight);
        total_bytes += q_output_weight.total_bytes;

        std::cout << "[NUMA] Replicated hot weights across " << n << " nodes ("
                  << (total_bytes * n / (1024.0 * 1024.0)) << " MB total allocated)"
                  << std::endl;
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

        // 0. Initialize device + pinned host pointers (once)
        if (!d_past_len_) {
            cudaMalloc(&d_past_len_, sizeof(int64_t));
            cudaHostAlloc(&h_past_len_pinned_, sizeof(int64_t), cudaHostAllocDefault);
        }
        if (!d_token_id_) {
            cudaMalloc(&d_token_id_, sizeof(int));
            cudaHostAlloc(&h_token_id_pinned_, sizeof(int), cudaHostAllocDefault);
        }

        // Ensure embedding table is on GPU (one-time)
        if (!emb_gpu_.defined() && token_embedding.defined()) {
            emb_gpu_ = at::empty_cuda({config.vocab_size, H});
            cudaMemcpy(emb_gpu_.mutable_data_ptr<float>(), token_embedding.data_ptr<float>(),
                       config.vocab_size * H * sizeof(float), cudaMemcpyHostToDevice);
        }

        int token_id_int = static_cast<int>(token_id);
        int cur = 0;
        bool capturing = false;

        // Use blocking stream for correct numerics + capture-friendly behavior.
        // Per-thread default stream (nullptr) was 30% slower in testing.
        if (!decode_stream_) {
            cudaStreamCreate(&decode_stream_);
            static bool smem_inited = false;
            if (!smem_inited) {
                // MSVC + windows.h max macro collision: use extra parens to bypass macro expansion.
                int64_t m1 = (H > q_dim) ? (int64_t)H : (int64_t)q_dim;
                int64_t m2 = (kv_dim > inter) ? (int64_t)kv_dim : (int64_t)inter;
                int max_K = static_cast<int>(m1 > m2 ? m1 : m2);
                at::cuda::init_cuda_kernel_smem_attributes(max_K, static_cast<int>(inter));
                smem_inited = true;
            }
        }
        cudaStream_t s = decode_stream_;

        // CUDA Graph: replay or capture
        if (graph_captured_ && decode_graph_exec_) {
            // FAST PATH: update device ptrs + replay
            *h_past_len_pinned_ = past_len;
            *h_token_id_pinned_ = token_id_int;
            cudaMemcpyAsync(d_past_len_, h_past_len_pinned_, sizeof(int64_t), cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(d_token_id_, h_token_id_pinned_, sizeof(int), cudaMemcpyHostToDevice, s);
            cudaGraphLaunch(decode_graph_exec_, s);
            cudaStreamSynchronize(s);  // CRITICAL: wait before CPU reads logits
            kv_cache.seq_len += 1;
            return sp.buf_logits;
        }

        // Update device pointers
        *h_past_len_pinned_ = past_len;
        *h_token_id_pinned_ = token_id_int;
        cudaMemcpyAsync(d_past_len_, h_past_len_pinned_, sizeof(int64_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_token_id_, h_token_id_pinned_, sizeof(int), cudaMemcpyHostToDevice, s);

        // Graph capture: reads device pointers (d_past_len_, d_token_id_) — cuBLAS
        // HGEMV path sets its own stream per call (see launch_cublas_hgemv), which is
        // capture-friendly on CUDA 11.1+. Disable for first token (warmup) so scratch
        // buffers are stable; capture on 2nd token, then replay for the rest.
        capturing = (d_past_len_ && !graph_captured_ && graph_token_id_ > 0);
        if (capturing) {
            // Ensure cuBLAS handle is bound to our capture stream before any cuBLAS call
            // inside the captured region (redundant with per-call SetStream, but safe).
#ifdef PT_USE_CUDA
            cublasHandle_t cublas_h = at::cuda::CuBLASHandle::get();
            cublasSetStream(cublas_h, s);
#endif
            cudaError_t cap_err = cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
            if (cap_err != cudaSuccess) {
                // Silent fall back per user spec
                cudaGetLastError();
                capturing = false;
            }
        }

        // 1. Embedding lookup (captured inside graph when capturing)
        PROF_BEGIN(profiler, "embedding");
        if (emb_gpu_.defined()) {
            at::cuda::launch_embedding_lookup(
                emb_gpu_.data_ptr<float>(),
                sp.buf_x[0].mutable_data_ptr<float>(),
                d_token_id_, static_cast<int>(H), s);
        }
        PROF_END(profiler, "embedding");


        // Scale embeddings (Gemma)
        if (config.scale_embeddings) {
            float scale = std::sqrt(static_cast<float>(H));
            at::cuda::launch_mul_scalar(sp.buf_x[0].data_ptr<float>(), scale,
                                         sp.buf_x[0].mutable_data_ptr<float>(), H, s);
        }

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

            // FP16 path: have fp16 for Q/K/V and user requested FP16
            bool use_fp16_qkv = use_fp16_weights_ &&
                layer.q_attn_q.fp16_data && layer.q_attn_k.fp16_data && layer.q_attn_v.fp16_data;

            if (use_fp16_qkv) {
                // -- FP16: rmsnorm then 3x cuBLAS HGEMV on Tensor Cores --
                PROF_BEGIN(profiler, "fp16_norm_qkv");
                at::cuda::launch_rms_norm(
                    x_ptr, layer.attn_norm.data_ptr<float>(),
                    sp.buf_normed.mutable_data_ptr<float>(),
                    1, static_cast<int>(H), eps, add_one, s);
                const float* normed_ptr = sp.buf_normed.data_ptr<float>();
                at::cuda::launch_cublas_hgemv(
                    layer.q_attn_q.fp16_data, normed_ptr,
                    sp.buf_q.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_attn_q.cols),
                    static_cast<int>(layer.q_attn_q.rows),
                    sp.x_fp16_buf, sp.y_fp16_buf, s);
                at::cuda::launch_cublas_hgemv(
                    layer.q_attn_k.fp16_data, normed_ptr,
                    sp.buf_k.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_attn_k.cols),
                    static_cast<int>(layer.q_attn_k.rows),
                    sp.x_fp16_buf, sp.y_fp16_buf, s);
                at::cuda::launch_cublas_hgemv(
                    layer.q_attn_v.fp16_data, normed_ptr,
                    sp.buf_v.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_attn_v.cols),
                    static_cast<int>(layer.q_attn_v.rows),
                    sp.x_fp16_buf, sp.y_fp16_buf, s);
                PROF_END(profiler, "fp16_norm_qkv");
            } else if (can_fuse_qkv) {
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
                    eps, add_one, s);
                PROF_END(profiler, "fused_norm_qkv");
            } else {
                // Fallback: separate attn_norm + 3 GEMVs
                PROF_BEGIN(profiler, "attn_norm");
                at::cuda::launch_rms_norm(
                    x_ptr, layer.attn_norm.data_ptr<float>(),
                    sp.buf_normed.mutable_data_ptr<float>(),
                    1, static_cast<int>(H), eps, add_one, s);
                PROF_END(profiler, "attn_norm");

                const float* normed_ptr = sp.buf_normed.data_ptr<float>();

                PROF_BEGIN(profiler, "qkv_proj");
                gemv_scratch(layer.q_attn_q, layer.attn_q, normed_ptr,
                            sp.buf_q.mutable_data_ptr<float>(), q_dim, s);
                gemv_scratch(layer.q_attn_k, layer.attn_k, normed_ptr,
                            sp.buf_k.mutable_data_ptr<float>(), kv_dim, s);
                gemv_scratch(layer.q_attn_v, layer.attn_v, normed_ptr,
                            sp.buf_v.mutable_data_ptr<float>(), kv_dim, s);
                PROF_END(profiler, "qkv_proj");
            }

            // -- Biases (Qwen3 has Q/K/V biases) --
            if (layer.attn_q_bias.defined()) {
                at::cuda::launch_add(
                    sp.buf_q.data_ptr<float>(), layer.attn_q_bias.data_ptr<float>(),
                    sp.buf_q.mutable_data_ptr<float>(), q_dim, s);
                at::cuda::launch_add(
                    sp.buf_k.data_ptr<float>(), layer.attn_k_bias.data_ptr<float>(),
                    sp.buf_k.mutable_data_ptr<float>(), kv_dim, s);
                at::cuda::launch_add(
                    sp.buf_v.data_ptr<float>(), layer.attn_v_bias.data_ptr<float>(),
                    sp.buf_v.mutable_data_ptr<float>(), kv_dim, s);
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
                    eps, add_one, s);
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
                    eps, add_one, past_len, s);
            }
            PROF_END(profiler, "fused_qknorm_rope_kv");

            // Also write K/V to FP16 cache for attention bandwidth reduction
            // NOTE: FP16 KV cache writes disabled in decode — baked offsets break CUDA Graph.
            // FP32 flash_decode is used instead (graph-compatible, reads d_past_len from GPU).
            // The FP16 KV cache is still used for PREFILL (via kv_cache.append()).

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // -- Flash-Decode attention (graph-compatible FP32: reads d_past_len from GPU) --
            PROF_BEGIN(profiler, "flash_decode");
            if (d_past_len_) {
                // Graph-compatible FP32 fallback
                at::cuda::launch_flash_decode_graph(
                    sp.buf_q.data_ptr<float>(),
                    kv_cache.key_cache[i].data_ptr<float>(),
                    kv_cache.value_cache[i].data_ptr<float>(),
                    sp.buf_attn.mutable_data_ptr<float>(),
                    sp.fd_partial_O, sp.fd_partial_lse, sp.fd_partial_max,
                    static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                    static_cast<int>(head_dim),
                    d_past_len_, static_cast<int>(kv_cache.max_seq), scale, s);
            } else if (kv_cache.use_fp16_kv) {
                int64_t total_seq = past_len + 1;
                at::cuda::launch_flash_decode_fp16(
                    sp.buf_q.data_ptr<float>(),
                    kv_cache.key_cache_fp16[i],
                    kv_cache.value_cache_fp16[i],
                    sp.buf_attn.mutable_data_ptr<float>(),
                    sp.fd_partial_O, sp.fd_partial_lse, sp.fd_partial_max,
                    static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                    static_cast<int>(head_dim),
                    static_cast<int>(total_seq), scale, s);
            } else {
                int64_t total_seq = past_len + 1;
                at::cuda::launch_flash_decode(
                    sp.buf_q.data_ptr<float>(),
                    kv_cache.key_cache[i].data_ptr<float>(),
                    kv_cache.value_cache[i].data_ptr<float>(),
                    sp.buf_attn.mutable_data_ptr<float>(),
                    sp.fd_partial_O, sp.fd_partial_lse, sp.fd_partial_max,
                    static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                    static_cast<int>(head_dim),
                    static_cast<int>(total_seq), scale, s);
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

            // FP16 path: cublas HGEMV + add (2 kernels instead of 1 fused, but Tensor Cores win)
            bool use_fp16_output = use_fp16_weights_ &&
                layer.q_attn_output.fp16_data &&
                !layer.post_attention_norm.defined();

            if (use_fp16_output) {
                PROF_BEGIN(profiler, "fp16_output_residual");
                at::cuda::launch_cublas_hgemv(
                    layer.q_attn_output.fp16_data,
                    sp.buf_attn.data_ptr<float>(),
                    sp.buf_attn_proj.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_attn_output.cols),
                    static_cast<int>(layer.q_attn_output.rows),
                    sp.x_fp16_buf, sp.y_fp16_buf, s);
                at::cuda::launch_add(x_ptr, sp.buf_attn_proj.data_ptr<float>(),
                                      sp.buf_h.mutable_data_ptr<float>(), H, s);
                PROF_END(profiler, "fp16_output_residual");
            } else if (can_fuse_output_residual) {
                PROF_BEGIN(profiler, "fused_output_residual");
                // Copy x → buf_h (residual base)
                at::cuda::launch_copy(x_ptr, sp.buf_h.mutable_data_ptr<float>(), H, s);
                // buf_h += W_output @ attn_out (accumulate)
                at::cuda::launch_q4km_persistent_gemv_accumulate(
                    layer.q_attn_output.gpu_data,
                    sp.buf_attn.data_ptr<float>(),
                    sp.buf_h.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_attn_output.cols),
                    static_cast<int>(layer.q_attn_output.rows),
                    layer.q_attn_output.row_stride_bytes, s);
                PROF_END(profiler, "fused_output_residual");
            } else {
                // Fallback: separate output_proj + residual_add
                PROF_BEGIN(profiler, "attn_output_proj");
                gemv_scratch(layer.q_attn_output, layer.attn_output,
                            sp.buf_attn.data_ptr<float>(),
                            sp.buf_attn_proj.mutable_data_ptr<float>(), H, s);
                PROF_END(profiler, "attn_output_proj");

                if (layer.post_attention_norm.defined()) {
                    at::cuda::launch_rms_norm(
                        sp.buf_attn_proj.data_ptr<float>(),
                        layer.post_attention_norm.data_ptr<float>(),
                        sp.buf_attn_proj.mutable_data_ptr<float>(),
                        1, static_cast<int>(H), eps, add_one, s);
                }

                PROF_BEGIN(profiler, "residual_add");
                at::cuda::launch_add(x_ptr, sp.buf_attn_proj.data_ptr<float>(),
                                      sp.buf_h.mutable_data_ptr<float>(), H, s);
                PROF_END(profiler, "residual_add");
            }

            // -- FFN: Check if we can fuse norm + gate+up --
            bool can_fuse_ffn = use_quant_gemv_ &&
                layer.q_ffn_gate.valid && layer.q_ffn_up.valid &&
                layer.q_ffn_gate.is_q4k() && layer.q_ffn_up.is_q4k() &&
                layer.q_ffn_gate.gpu_data && layer.q_ffn_up.gpu_data &&
                layer.q_ffn_gate.cols == layer.q_ffn_up.cols &&
                layer.q_ffn_gate.row_stride_bytes == layer.q_ffn_up.row_stride_bytes;

            bool use_fp16_ffn_gu = use_fp16_weights_ &&
                layer.q_ffn_gate.fp16_data && layer.q_ffn_up.fp16_data;

            if (use_fp16_ffn_gu) {
                // -- FP16: ffn_norm + 2x cuBLAS HGEMV (Tensor Cores) --
                PROF_BEGIN(profiler, "fp16_norm_gate_up");
                at::cuda::launch_rms_norm(
                    sp.buf_h.data_ptr<float>(), layer.ffn_norm.data_ptr<float>(),
                    sp.buf_normed.mutable_data_ptr<float>(),
                    1, static_cast<int>(H), eps, add_one, s);
                const float* normed_ptr = sp.buf_normed.data_ptr<float>();
                at::cuda::launch_cublas_hgemv(
                    layer.q_ffn_gate.fp16_data, normed_ptr,
                    sp.buf_gate.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_ffn_gate.cols),
                    static_cast<int>(layer.q_ffn_gate.rows),
                    sp.x_fp16_buf, sp.y_fp16_buf, s);
                at::cuda::launch_cublas_hgemv(
                    layer.q_ffn_up.fp16_data, normed_ptr,
                    sp.buf_up.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_ffn_up.cols),
                    static_cast<int>(layer.q_ffn_up.rows),
                    sp.x_fp16_buf, sp.y_fp16_buf, s);
                PROF_END(profiler, "fp16_norm_gate_up");
            } else if (can_fuse_ffn) {
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
                    eps, add_one, s);
                PROF_END(profiler, "fused_norm_gate_up");
            } else {
                // Fallback: separate ffn_norm + gate/up GEMVs
                PROF_BEGIN(profiler, "ffn_norm");
                at::cuda::launch_rms_norm(
                    sp.buf_h.data_ptr<float>(), layer.ffn_norm.data_ptr<float>(),
                    sp.buf_normed.mutable_data_ptr<float>(),
                    1, static_cast<int>(H), eps, add_one, s);
                PROF_END(profiler, "ffn_norm");

                const float* normed_ptr = sp.buf_normed.data_ptr<float>();

                PROF_BEGIN(profiler, "ffn_gate_up");
                fused_gate_up_gemv(layer, normed_ptr,
                                  sp.buf_gate.mutable_data_ptr<float>(),
                                  sp.buf_up.mutable_data_ptr<float>(), inter, s);
                PROF_END(profiler, "ffn_gate_up");
            }

            // -- SiLU-Mul: silu(gate) * up → buf_silu --
            PROF_BEGIN(profiler, "silu_mul");
            at::cuda::launch_silu_mul(
                sp.buf_gate.data_ptr<float>(), sp.buf_up.data_ptr<float>(),
                sp.buf_silu.mutable_data_ptr<float>(), inter, s);
            PROF_END(profiler, "silu_mul");

            // -- Down projection with fused residual add --
            bool can_fuse_down_residual = use_quant_gemv_ &&
                layer.q_ffn_down.valid && layer.q_ffn_down.is_q4k() &&
                layer.q_ffn_down.gpu_data &&
                !layer.post_ffw_norm.defined();  // Can't fuse if post-norm needed

            bool use_fp16_down = use_fp16_weights_ &&
                layer.q_ffn_down.fp16_data &&
                !layer.post_ffw_norm.defined();

            int next = 1 - cur;
            if (use_fp16_down) {
                PROF_BEGIN(profiler, "fp16_down_residual");
                at::cuda::launch_cublas_hgemv(
                    layer.q_ffn_down.fp16_data,
                    sp.buf_silu.data_ptr<float>(),
                    sp.buf_down.mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_ffn_down.cols),
                    static_cast<int>(layer.q_ffn_down.rows),
                    sp.x_fp16_buf, sp.y_fp16_buf, s);
                at::cuda::launch_add(sp.buf_h.data_ptr<float>(), sp.buf_down.data_ptr<float>(),
                                      sp.buf_x[next].mutable_data_ptr<float>(), H, s);
                PROF_END(profiler, "fp16_down_residual");
            } else if (can_fuse_down_residual) {
                PROF_BEGIN(profiler, "fused_down_residual");
                // Copy buf_h → buf_x[next] (residual base)
                at::cuda::launch_copy(sp.buf_h.data_ptr<float>(),
                                       sp.buf_x[next].mutable_data_ptr<float>(), H, s);
                // buf_x[next] += W_down @ silu_out (accumulate)
                at::cuda::launch_q4km_persistent_gemv_accumulate(
                    layer.q_ffn_down.gpu_data,
                    sp.buf_silu.data_ptr<float>(),
                    sp.buf_x[next].mutable_data_ptr<float>(),
                    static_cast<int>(layer.q_ffn_down.cols),
                    static_cast<int>(layer.q_ffn_down.rows),
                    layer.q_ffn_down.row_stride_bytes, s);
                PROF_END(profiler, "fused_down_residual");
            } else {
                // Fallback: separate down_proj + post-norm + residual
                PROF_BEGIN(profiler, "ffn_down");
                gemv_scratch(layer.q_ffn_down, layer.ffn_down,
                            sp.buf_silu.data_ptr<float>(),
                            sp.buf_down.mutable_data_ptr<float>(), H, s);
                PROF_END(profiler, "ffn_down");

                if (layer.post_ffw_norm.defined()) {
                    at::cuda::launch_rms_norm(
                        sp.buf_down.data_ptr<float>(),
                        layer.post_ffw_norm.data_ptr<float>(),
                        sp.buf_down.mutable_data_ptr<float>(),
                        1, static_cast<int>(H), eps, add_one, s);
                }

                PROF_BEGIN(profiler, "residual_add");
                at::cuda::launch_add(sp.buf_h.data_ptr<float>(), sp.buf_down.data_ptr<float>(),
                                      sp.buf_x[next].mutable_data_ptr<float>(), H, s);
                PROF_END(profiler, "residual_add");
            }
            cur = next;
        }

        // 3. Final RMS norm (INSIDE graph capture — uses s=nullptr)
        PROF_BEGIN(profiler, "final_norm");
        at::cuda::launch_rms_norm(
            sp.buf_x[cur].data_ptr<float>(), output_norm.data_ptr<float>(),
            sp.buf_normed.mutable_data_ptr<float>(),
            1, static_cast<int>(H), eps, add_one, s);
        PROF_END(profiler, "final_norm");

        // 4. Output projection — cuBLAS FP16 for lm_head (Tensor Cores!)
        PROF_BEGIN(profiler, "output_proj");
        if (lm_head_fp16_ && sp.x_fp16_buf && sp.y_fp16_buf) {
            // cuBLAS HGEMV: FP16 weights × FP32 input → FP32 output (via Tensor Cores)
            at::cuda::launch_cublas_hgemv(
                lm_head_fp16_,
                sp.buf_normed.data_ptr<float>(),
                sp.buf_logits.mutable_data_ptr<float>(),
                static_cast<int>(q_output_weight.cols),
                static_cast<int>(q_output_weight.rows),
                sp.x_fp16_buf, sp.y_fp16_buf, s);
        } else if (use_quant_gemv_ && q_output_weight.valid && q_output_weight.gpu_data) {
            // Fallback: custom Q4_K GEMV
            int K_out = static_cast<int>(q_output_weight.cols);
            int N_out = static_cast<int>(q_output_weight.rows);
            if (q_output_weight.is_q4k()) {
                at::cuda::launch_q4km_persistent_gemv(q_output_weight.gpu_data,
                    sp.buf_normed.data_ptr<float>(), sp.buf_logits.mutable_data_ptr<float>(),
                    K_out, N_out, q_output_weight.row_stride_bytes, s);
            } else {
                gemv_scratch(q_output_weight, output_weight, sp.buf_normed.data_ptr<float>(),
                    sp.buf_logits.mutable_data_ptr<float>(), config.vocab_size, s);
            }
        } else if (output_weight.defined()) {
            at::cuda::launch_gemv(output_weight.data_ptr<float>(),
                sp.buf_normed.data_ptr<float>(), sp.buf_logits.mutable_data_ptr<float>(),
                config.vocab_size, static_cast<int>(H), s);
        }
        PROF_END(profiler, "output_proj");

        // End CUDA Graph capture (on decode_stream_, NOT nullptr!)
        if (capturing) {
            cudaStreamEndCapture(s, &decode_graph_);
            cudaError_t err = cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, NULL, NULL, 0);
            if (err == cudaSuccess) {
                graph_captured_ = true;
                std::cout << "[PromeGraph] Captured full decode graph!" << std::endl;
                // RE-EXECUTE: capture didn't run kernels
                cudaGraphLaunch(decode_graph_exec_, s);
                cudaStreamSynchronize(s);
            } else {
                std::cerr << "[PromeGraph] Capture failed: " << cudaGetErrorString(err) << std::endl;
                decode_graph_ = nullptr;
                decode_graph_exec_ = nullptr;
            }
        } else {
            // Normal path: sync decode_stream before CPU reads logits
            cudaStreamSynchronize(s);
        }

        kv_cache.seq_len += 1;
        graph_token_id_++;
        return sp.buf_logits;
    }

    // GEMV helper for scratch pool: dispatch persistent > quant > float32
    // Uses persistent kernel for Q4_K to reduce launch overhead (grid-stride over rows).
    void gemv_scratch(const QuantizedWeight& qw, const Tensor& float_w,
                      const float* x, float* y, int64_t N,
                      cudaStream_t stream = nullptr) {
        if (use_quant_gemv_ && qw.valid) {
            int K = static_cast<int>(qw.cols);
            int Nr = static_cast<int>(qw.rows);
            if (use_llama_gemv_ && qw.is_q4k() && qw.gpu_data) {
                at::cuda::launch_q4km_persistent_gemv_v2(qw.gpu_data, x, y, K, Nr, qw.row_stride_bytes, stream);
            } else if (qw.is_q4k() && qw.gpu_data) {
                at::cuda::launch_q4km_persistent_gemv(qw.gpu_data, x, y, K, Nr, qw.row_stride_bytes, stream);
            } else if (qw.is_q6k() && qw.gpu_data) {
                at::cuda::launch_q6k_gemv(qw.gpu_data, x, y, K, Nr, qw.row_stride_bytes, stream);
            } else if (qw.is_q5k() && qw.gpu_data) {
                at::cuda::launch_q5k_gemv(qw.gpu_data, x, y, K, Nr, qw.row_stride_bytes, stream);
            } else if (qw.is_f16() && qw.gpu_data) {
                // FP16 weight: use simple FP16→FP32 dequant GEMV
                at::cuda::launch_fp16_gemv(qw.gpu_data, x, y, K, Nr, stream);
            }
        } else if (float_w.defined()) {
            int K = static_cast<int>(float_w.size(0));
            int Ni = static_cast<int>(float_w.size(1));
            at::cuda::launch_inference_gemv(x, float_w.data_ptr<float>(), y, K, Ni, stream);
        }
    }

    // Fused gate+up GEMV: single kernel launch for both gate and up projections.
    // Saves one kernel launch per layer (~28 launches saved for 28-layer model).
    void fused_gate_up_gemv(const TransformerLayer& layer,
                            const float* x, float* y_gate, float* y_up,
                            int64_t inter, cudaStream_t stream = nullptr) {
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
                K, N_gate, N_up, qg.row_stride_bytes, stream);
        } else {
            // Fallback: two separate GEMVs
            gemv_scratch(qg, layer.ffn_gate, x, y_gate, inter, stream);
            gemv_scratch(qu, layer.ffn_up, x, y_up, inter, stream);
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

    // Per-section cumulative timers, summed across layers+tokens, dumped on
    // reset. Toggle via env PT_PROFILE_LAYER=1. Near-zero overhead when off.
    mutable struct SectionTimers {
        bool on = false;
        bool inited = false;
        double qkv_ms = 0, attn_ms = 0, ao_ms = 0, gateup_ms = 0, silu_ms = 0, fdown_ms = 0;
        int64_t tokens = 0;
        void init() {
            if (inited) return;
            const char* env = std::getenv("PT_PROFILE_LAYER");
            on = (env && env[0] == '1');
            inited = true;
        }
        void dump() const {
            if (!on || tokens == 0) return;
            double total = qkv_ms + attn_ms + ao_ms + gateup_ms + silu_ms + fdown_ms;
            std::fprintf(stderr,
                "[prof] %ld tokens, avg ms/token:\n"
                "  qkv_fused:     %6.2f\n"
                "  attention:     %6.2f\n"
                "  attn_output:   %6.2f\n"
                "  gate_up_fused: %6.2f\n"
                "  silu:          %6.2f\n"
                "  ffn_down:      %6.2f\n"
                "  sum:           %6.2f\n",
                (long)tokens,
                qkv_ms/tokens, attn_ms/tokens, ao_ms/tokens,
                gateup_ms/tokens, silu_ms/tokens, fdown_ms/tokens,
                total/tokens);
        }
    } sec_timers_;

    Tensor forward_decode_cpu(int64_t token_id) {
        sec_timers_.init();
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

            auto _t_section = std::chrono::high_resolution_clock::now();
            auto _elapsed_ms = [&_t_section]() {
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - _t_section).count();
                _t_section = t1;
                return ms;
            };

            if (can_fuse) {
                // FUSED: RMSNorm + QKV projection (1 RMSNorm, shared x across 3 GEMVs)
                // Pick per-NUMA replica for each weight. get() falls back to
                // the original cpu_data when replication is off, so this is a
                // no-op unless PT_NUMA_REPLICATE=1 was set.
                int _node = c10::current_numa_node();
                const void* w_q = layer.q_attn_q.numa_replica.get(_node);
                const void* w_k = layer.q_attn_k.numa_replica.get(_node);
                const void* w_v = layer.q_attn_v.numa_replica.get(_node);
                if (!w_q) w_q = layer.q_attn_q.cpu_data;
                if (!w_k) w_k = layer.q_attn_k.cpu_data;
                if (!w_v) w_v = layer.q_attn_v.cpu_data;
                cpu_quant::cpu_fused_rmsnorm_qkv_gemv(
                    x_ptr, layer.attn_norm.data_ptr<float>(), eps, add_one,
                    layer.q_attn_q.quant_type,
                    w_q, w_k, w_v,
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

                if (sec_timers_.on) sec_timers_.qkv_ms += _elapsed_ms();
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

                    // Softmax (AVX2 vectorized: max + exp + normalize).
                    // Round2 agent_5 P2: scalar std::exp over past_len=1024
                    // tokens × 32 heads × 36 layers ≈ 1.2M exp/token wasted
                    // on libm scalar. VectorizedOps.h provides exp256_ps.
#ifdef __AVX2__
                    float max_score = scores[0];
                    {
                        int64_t t = 1;
                        __m256 vmax = _mm256_set1_ps(max_score);
                        for (; t + 7 < total_seq; t += 8) {
                            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(scores + t));
                        }
                        // horizontal max of vmax
                        float lanes[8]; _mm256_storeu_ps(lanes, vmax);
                        for (int k = 0; k < 8; ++k) if (lanes[k] > max_score) max_score = lanes[k];
                        for (; t < total_seq; ++t) if (scores[t] > max_score) max_score = scores[t];
                    }
                    float sum_exp = 0.0f;
                    {
                        __m256 vmaxs = _mm256_set1_ps(max_score);
                        __m256 vsum = _mm256_setzero_ps();
                        int64_t t = 0;
                        for (; t + 7 < total_seq; t += 8) {
                            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(scores + t), vmaxs);
                            v = at::native::vec::exp256_ps(v);
                            _mm256_storeu_ps(scores + t, v);
                            vsum = _mm256_add_ps(vsum, v);
                        }
                        sum_exp = at::native::vec::hsum_avx2(vsum);
                        for (; t < total_seq; ++t) {
                            scores[t] = std::exp(scores[t] - max_score);
                            sum_exp += scores[t];
                        }
                    }
                    float inv_sum = 1.0f / (sum_exp + 1e-10f);
                    {
                        __m256 vinv = _mm256_set1_ps(inv_sum);
                        int64_t t = 0;
                        for (; t + 7 < total_seq; t += 8) {
                            _mm256_storeu_ps(scores + t,
                                _mm256_mul_ps(_mm256_loadu_ps(scores + t), vinv));
                        }
                        for (; t < total_seq; ++t) scores[t] *= inv_sum;
                    }
#else
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
#endif

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
            if (sec_timers_.on) sec_timers_.attn_ms += _elapsed_ms();

            // -- Output projection: attn_buf @ W_o -> h_buf --
            // Post-attention norm (Gemma3): in-place on attn_buf
            if (layer.post_attention_norm.defined()) {
                cpu_quant::cpu_rmsnorm_inplace(sp.attn_buf, layer.post_attention_norm.data_ptr<float>(),
                    eps, add_one, q_dim);
            }

            if (use_quant_gemv_ && layer.q_attn_output.valid && layer.q_attn_output.cpu_data) {
                cpu_quant::cpu_quant_gemv(layer.q_attn_output.quant_type, layer.q_attn_output.cpu_data,
                    sp.attn_buf, sp.h_buf, q_dim, layer.q_attn_output.rows, layer.q_attn_output.row_stride_bytes,
                    &layer.q_attn_output.numa_replica);
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
            if (sec_timers_.on) sec_timers_.ao_ms += _elapsed_ms();

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
                    layer.q_ffn_gate.row_stride_bytes, layer.q_ffn_up.row_stride_bytes,
                    &layer.q_ffn_gate.numa_replica, &layer.q_ffn_up.numa_replica);
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

            if (sec_timers_.on) sec_timers_.gateup_ms += _elapsed_ms();

            // -- SiLU(gate) * up: in-place into gate_buf --
            // Parallelized: std::exp on E2K libm is scalar software, ~200ns/call.
            // Per layer inter=9728 × 8 exps per AVX2 iter (prev scalar fallback)
            // = ~10K exps/layer × 36 = 350K/token = ~70 ms serial. Fan-out across
            // 24 threads cuts this to single-digit ms.
            c10::get_thread_pool().parallel_for(0, inter, [&](int64_t s, int64_t e) {
                int64_t j = s;
#ifdef __AVX2__
                __m256 one = _mm256_set1_ps(1.0f);
                for (; j + 7 < e; j += 8) {
                    __m256 g = _mm256_loadu_ps(sp.gate_buf + j);
                    __m256 u = _mm256_loadu_ps(sp.up_buf + j);
                    __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
                    neg_g = _mm256_max_ps(neg_g, _mm256_set1_ps(-88.0f));
                    neg_g = _mm256_min_ps(neg_g, _mm256_set1_ps(88.0f));
                    float tmp[8];
                    _mm256_storeu_ps(tmp, neg_g);
                    __m256 exp_neg_g = _mm256_set_ps(
                        std::exp(tmp[7]), std::exp(tmp[6]), std::exp(tmp[5]), std::exp(tmp[4]),
                        std::exp(tmp[3]), std::exp(tmp[2]), std::exp(tmp[1]), std::exp(tmp[0]));
                    __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_g));
                    __m256 silu = _mm256_mul_ps(g, sigmoid);
                    _mm256_storeu_ps(sp.gate_buf + j, _mm256_mul_ps(silu, u));
                }
#endif
                for (; j < e; ++j) {
                    float g = sp.gate_buf[j];
                    sp.gate_buf[j] = (g / (1.0f + std::exp(-g))) * sp.up_buf[j];
                }
            }, /*min_grain=*/256);
            if (sec_timers_.on) sec_timers_.silu_ms += _elapsed_ms();

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
                        sp.gate_buf, sp.h_buf, inter, layer.q_ffn_down.rows, layer.q_ffn_down.row_stride_bytes,
                        &layer.q_ffn_down.numa_replica);
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
            if (sec_timers_.on) sec_timers_.fdown_ms += _elapsed_ms();
        }  // end layer loop
        if (sec_timers_.on) sec_timers_.tokens++;

        // 3. Final RMS norm (in-place)
        cpu_quant::cpu_rmsnorm_inplace(sp.x_buf[cur], output_norm.data_ptr<float>(), eps, add_one, H);

        // 4. Output projection -> logits (into scratch logits_buf)
        //    Use sparse GEMV for output projection if available
        if (use_quant_gemv_ && q_output_weight.valid && q_output_weight.cpu_data) {
            // LM head is 208 MB/token Q4_K read per decoded step. Agent 2
            // measured ~12% of serial budget spent on cross-chip fetch when
            // it lived on one first-touch NUMA node. Pick local replica;
            // falls back to original pointer when replication disabled.
            int _node = c10::current_numa_node();
            const void* w_out = q_output_weight.numa_replica.get(_node);
            if (!w_out) w_out = q_output_weight.cpu_data;
            if (use_sparse_gemv_ && sparse_output_.valid && q_output_weight.is_q4k()) {
                sparse_q4k_gemv(w_out, sp.x_buf[cur], sp.logits_buf,
                                H, q_output_weight.rows, q_output_weight.row_stride_bytes,
                                sparse_output_);
            } else {
                // Round2 agent_5 item 3b: pass &numa_replica so each worker
                // picks its LOCAL copy at chunk-start. Previously we passed
                // only the pre-resolved master-thread pointer, causing
                // workers on other NUMA nodes to cross-chip-fetch the
                // 608 KB vocab stride.
                cpu_quant::cpu_quant_gemv(q_output_weight.quant_type,
                    q_output_weight.cpu_data,
                    sp.x_buf[cur], sp.logits_buf,
                    H, q_output_weight.rows, q_output_weight.row_stride_bytes,
                    &q_output_weight.numa_replica);
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
                int _node = c10::current_numa_node();
                const void* w_q = layer.q_attn_q.numa_replica.get(_node);
                const void* w_k = layer.q_attn_k.numa_replica.get(_node);
                const void* w_v = layer.q_attn_v.numa_replica.get(_node);
                if (!w_q) w_q = layer.q_attn_q.cpu_data;
                if (!w_k) w_k = layer.q_attn_k.cpu_data;
                if (!w_v) w_v = layer.q_attn_v.cpu_data;
                cpu_quant::cpu_fused_rmsnorm_qkv_gemv(
                    x_ptr, layer.attn_norm.data_ptr<float>(), eps, add_one,
                    layer.q_attn_q.quant_type,
                    w_q, w_k, w_v,
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
                    layer.q_ffn_gate.row_stride_bytes, layer.q_ffn_up.row_stride_bytes,
                    &layer.q_ffn_gate.numa_replica, &layer.q_ffn_up.numa_replica);
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
                        sp.gate_buf, sp.h_buf, inter, layer.q_ffn_down.rows, layer.q_ffn_down.row_stride_bytes,
                        &layer.q_ffn_down.numa_replica);
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
    // Phase 7 — speculative decode step (MVP: serial verify, no speedup yet).
    //
    // Returns a vector of 1..K committed tokens. Caller passes:
    //   - `current_token`: the last-committed token (input to the first verify)
    //   - `draft`: the running NgramDraft, already populated with all committed
    //     tokens up to but not including `current_token`.
    //   - `history`: vector of all committed tokens, INCLUDING current_token
    //     (i.e. history.back() == current_token). Random-access mirror of
    //     draft's internal ring buffer, used for suffix lookup.
    //   - `K_max`: desired verify width (1..6).
    //   - `stats` (optional): increments counters for reporting.
    //
    // Semantics:
    //   1. Use NgramDraft to predict up to K_max-1 draft tokens starting from
    //      `history` + [current_token]. Stop drafting on the first -1.
    //   2. Run forward_decode_cpu serially on `current_token` then each draft
    //      token in order. After each forward, compute argmax.
    //   3. If argmax at position j == draft[j], accept the draft token, feed
    //      it to history + NgramDraft, continue.
    //   4. On first mismatch (or end of drafts), commit argmax as the main
    //      model's answer for position j, REWIND kv_cache.seq_len back to
    //      `past_len + j + 1` (so KV no longer contains the now-rejected
    //      draft tokens' K/V entries), and return.
    //
    // Invariants after return:
    //   - `kv_cache.seq_len` reflects exactly the number of committed tokens.
    //   - Every returned token has been argmax'd by the main model (correct).
    //   - `draft` has been updated for every returned token.
    //   - `history` is UNCHANGED by this function; caller updates it.
    //
    // For K_max=1 this degenerates to exactly one forward + one argmax and
    // is bit-for-bit identical to the existing decode path.
    //
    // No speedup yet — serial forwards mean N tokens still cost N×forward.
    // Phase 7.1 will replace the serial loop with a batched forward pass
    // that reads each weight matrix once and computes K parallel outputs.
    // ========================================================================
    // ========================================================================
    // Phase 7.2 — batched forward pass scaffold.
    //
    // Entry point that runs the main model on K consecutive tokens and
    // returns K logit rows. Today (scaffold) this is literally K serial
    // calls to forward_decode_cpu — same cost, same output, no speedup.
    //
    // Next iterations will replace specific inner calls with batched Q4_K
    // kernels (cpu_quant_gemv_batched) reading each weight matrix ONCE for
    // all K queries. The top-level signature will not change so callers
    // (spec_decode_step_cpu) don't need revisiting.
    //
    // Call sites to convert in the per-layer loop (cumulative 180 ms / 192 ms
    // decode budget, from prof-TP profile 2026-04-22):
    //   1. cpu_fused_rmsnorm_qkv_gemv       (34.5 ms)   → batched QKV
    //   2. cpu_quant_gemv(attn_output)      (20.4 ms)   → batched
    //   3. cpu_fused_rmsnorm_gate_up_gemv   (74.4 ms)   → batched gate+up
    //   4. cpu_quant_gemv(ffn_down)         (48.5 ms)   → batched
    //   5. cpu_quant_gemv(output_proj)      (20.0 ms)   → batched
    //
    // Not batched (small, not BW-bound):
    //   * RMSNorm preambles, SiLU, residual adds, embedding lookup
    //   * Attention softmax + Q·K·V math (needs per-query causal mask)
    //
    // KV cache contract: on return, kv_cache.seq_len == past_len + K.
    // Caller (spec_decode_step_cpu) may rewind if draft tokens were rejected.
    //
    // logits_out layout: row-major [K × vocab_size]. Row k is the logits
    // distribution for predicting the (past_len+k+1)-th token, conditioned
    // on tokens[0..k] appended to the KV cache before it.
    // ========================================================================
    void forward_decode_cpu_batched(const int64_t* tokens, int K,
                                    float* logits_out) {
        const int64_t H          = config.hidden_size;
        const int64_t q_dim      = config.num_heads * config.head_dim;
        const int64_t kv_dim     = config.num_kv_heads * config.head_dim;
        const int64_t head_dim   = config.head_dim;
        const int64_t n_heads    = config.num_heads;
        const int64_t n_kv_heads = config.num_kv_heads;
        const int64_t inter      = config.intermediate_size;
        const int64_t heads_per_group = n_heads / n_kv_heads;
        const int64_t V          = config.vocab_size;
        const float   eps        = config.rms_norm_eps;
        const bool    add_one    = config.gemma_norm_add_one;
        const int64_t past_len_0 = kv_cache.seq_len;
        const int     _node      = c10::current_numa_node();

        // Per-call batched scratch. Per-token forward uses CPUScratchPool
        // (size K=1 buffers). Here we need K copies of every intermediate.
        // For qwen3:4b with K=5 the total heap is ≈ 5 × (2*H + 2*q_dim +
        // 2*kv_dim + q_dim + H + H + 2*inter + inter + H + V) × 4 B ≈
        // 5 × (5 120 + 8 192 + 2 048 + 4 096 + 2 560 + 2 560 + 19 456 +
        // 9 728 + 2 560 + 151 936) × 4 B ≈ 4.1 MB. Allocated on heap once
        // per verify step; amortised over 36 layers.
        std::vector<float> x_a  (K * H),          x_b  (K * H);
        std::vector<float> x_na (K * H),          x_nf (K * H);
        std::vector<float> q_b  (K * q_dim),      k_b  (K * kv_dim), v_b(K * kv_dim);
        std::vector<float> attn (K * q_dim),      hbuf (K * H);
        std::vector<float> gate (K * inter),      up_b (K * inter),  siluup(K * inter);
        std::vector<float> down (K * H),          x_fin(K * H);
        float* cur  = x_a.data();
        float* next = x_b.data();

        // 1. Embedding: K lookups.
        {
            const float* emb = token_embedding.data_ptr<float>();
            for (int k = 0; k < K; ++k) {
                std::memcpy(cur + k * H, emb + tokens[k] * H, H * sizeof(float));
            }
            if (config.scale_embeddings) {
                float s = std::sqrt((float)H);
                for (int64_t j = 0; j < (int64_t)K * H; ++j) cur[j] *= s;
            }
        }

        // 2. Transformer layers.
        for (int64_t i = 0; i < config.num_layers; ++i) {
            auto& layer = layers[i];

            // 2a. K × RMSNorm (attn preamble).
            const float* an_w = layer.attn_norm.data_ptr<float>();
            for (int k = 0; k < K; ++k) {
                cpu_quant::cpu_rmsnorm_out(cur + k * H, x_na.data() + k * H, an_w, eps, add_one, H);
            }

            // 2b. Batched QKV (weights read once per matrix, K outputs each).
            {
                const void* w_q = layer.q_attn_q.numa_replica.get(_node);
                const void* w_k = layer.q_attn_k.numa_replica.get(_node);
                const void* w_v = layer.q_attn_v.numa_replica.get(_node);
                if (!w_q) w_q = layer.q_attn_q.cpu_data;
                if (!w_k) w_k = layer.q_attn_k.cpu_data;
                if (!w_v) w_v = layer.q_attn_v.cpu_data;
                cpu_quant::cpu_quant_gemv_batched(
                    layer.q_attn_q.quant_type, w_q, x_na.data(), q_b.data(),
                    K, H, q_dim, layer.q_attn_q.row_stride_bytes);
                cpu_quant::cpu_quant_gemv_batched(
                    layer.q_attn_k.quant_type, w_k, x_na.data(), k_b.data(),
                    K, H, kv_dim, layer.q_attn_k.row_stride_bytes);
                cpu_quant::cpu_quant_gemv_batched(
                    layer.q_attn_v.quant_type, w_v, x_na.data(), v_b.data(),
                    K, H, kv_dim, layer.q_attn_v.row_stride_bytes);
            }

            // 2c. Per-token: bias + QK-norm + RoPE + KV cache write.
            float* k_cache = kv_cache.key_cache[i].mutable_data_ptr<float>();
            float* v_cache = kv_cache.value_cache[i].mutable_data_ptr<float>();
            for (int k = 0; k < K; ++k) {
                float* q  = q_b.data() + k * q_dim;
                float* kk = k_b.data() + k * kv_dim;
                float* vv = v_b.data() + k * kv_dim;
                if (layer.attn_q_bias.defined()) {
                    const float* bq = layer.attn_q_bias.data_ptr<float>();
                    const float* bk = layer.attn_k_bias.data_ptr<float>();
                    const float* bv = layer.attn_v_bias.data_ptr<float>();
                    for (int64_t j = 0; j < q_dim;  ++j) q[j]  += bq[j];
                    for (int64_t j = 0; j < kv_dim; ++j) { kk[j] += bk[j]; vv[j] += bv[j]; }
                }
                if (layer.attn_q_norm.defined()) {
                    const float* qnw = layer.attn_q_norm.data_ptr<float>();
                    const float* knw = layer.attn_k_norm.data_ptr<float>();
                    for (int64_t h = 0; h < n_heads;    ++h)
                        cpu_quant::cpu_rmsnorm_inplace(q  + h * head_dim, qnw, eps, add_one, head_dim);
                    for (int64_t h = 0; h < n_kv_heads; ++h)
                        cpu_quant::cpu_rmsnorm_inplace(kk + h * head_dim, knw, eps, add_one, head_dim);
                }
                float rope_cos[256], rope_sin[256];
                at::native::hot::rope_precompute(rope_cos, rope_sin,
                    past_len_0 + k, head_dim, config.rope_freq_base);
                at::native::hot::rope_apply_fused(q, kk, rope_cos, rope_sin,
                    n_heads, n_kv_heads, head_dim);
                std::memcpy(k_cache + (past_len_0 + k) * kv_dim, kk, kv_dim * sizeof(float));
                std::memcpy(v_cache + (past_len_0 + k) * kv_dim, vv, kv_dim * sizeof(float));
            }

            // 2d. Batched attention — K queries × n_heads, each attends to
            //     its causal prefix of length past_len_0 + k + 1.
            const float scale = 1.0f / std::sqrt((float)head_dim);
            const float* k_cache_ro = kv_cache.key_cache[i].data_ptr<float>();
            const float* v_cache_ro = kv_cache.value_cache[i].data_ptr<float>();
            c10::get_thread_pool().parallel_for(0, (int64_t)K * n_heads,
                [&](int64_t a_start, int64_t a_end) {
                for (int64_t idx = a_start; idx < a_end; ++idx) {
                    int64_t k   = idx / n_heads;
                    int64_t h   = idx % n_heads;
                    int64_t kv_h = h / heads_per_group;
                    int64_t total_seq = past_len_0 + k + 1;
                    const float* q_head = q_b.data() + k * q_dim + h * head_dim;
                    float* out_head = attn.data() + k * q_dim + h * head_dim;
                    float local_scores[8192];
                    float* scores = (total_seq <= 8192) ? local_scores
                                                         : new float[total_seq];
                    for (int64_t t = 0; t < total_seq; ++t) {
                        const float* kh = k_cache_ro + t * kv_dim + kv_h * head_dim;
                        float dot = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * kh[d];
                        scores[t] = dot * scale;
                    }
                    float mx = scores[0];
                    for (int64_t t = 1; t < total_seq; ++t) if (scores[t] > mx) mx = scores[t];
                    float sm = 0.0f;
                    for (int64_t t = 0; t < total_seq; ++t) {
                        scores[t] = std::exp(scores[t] - mx);
                        sm += scores[t];
                    }
                    float inv = 1.0f / (sm + 1e-10f);
                    for (int64_t t = 0; t < total_seq; ++t) scores[t] *= inv;
                    std::fill(out_head, out_head + head_dim, 0.0f);
                    for (int64_t t = 0; t < total_seq; ++t) {
                        const float* vh = v_cache_ro + t * kv_dim + kv_h * head_dim;
                        float w = scores[t];
                        for (int64_t d = 0; d < head_dim; ++d) out_head[d] += w * vh[d];
                    }
                    if (total_seq > 8192) delete[] scores;
                }
            }, 1);

            // 2e. Batched attn_output.
            {
                const void* w_o = layer.q_attn_output.numa_replica.get(_node);
                if (!w_o) w_o = layer.q_attn_output.cpu_data;
                cpu_quant::cpu_quant_gemv_batched(
                    layer.q_attn_output.quant_type, w_o, attn.data(), hbuf.data(),
                    K, q_dim, H, layer.q_attn_output.row_stride_bytes);
            }

            // 2f. Residual: next = cur + hbuf — parallel AVX2.
            {
                const int64_t total = (int64_t)K * H;
                c10::get_thread_pool().parallel_for(0, total,
                    [&](int64_t start, int64_t end) {
                    int64_t j = start;
#ifdef __AVX2__
                    for (; j + 7 < end; j += 8) {
                        _mm256_storeu_ps(next + j,
                            _mm256_add_ps(_mm256_loadu_ps(cur + j),
                                          _mm256_loadu_ps(hbuf.data() + j)));
                    }
#endif
                    for (; j < end; ++j) next[j] = cur[j] + hbuf[j];
                }, 1024);
            }
            std::swap(cur, next);

            // 2g. K × RMSNorm (FFN preamble).
            const float* fn_w = layer.ffn_norm.data_ptr<float>();
            for (int k = 0; k < K; ++k) {
                cpu_quant::cpu_rmsnorm_out(cur + k * H, x_nf.data() + k * H, fn_w, eps, add_one, H);
            }

            // 2h. Batched gate + up (two separate weight reads, K outputs each).
            {
                const void* w_g = layer.q_ffn_gate.numa_replica.get(_node);
                const void* w_u = layer.q_ffn_up.numa_replica.get(_node);
                if (!w_g) w_g = layer.q_ffn_gate.cpu_data;
                if (!w_u) w_u = layer.q_ffn_up.cpu_data;
                cpu_quant::cpu_quant_gemv_batched(
                    layer.q_ffn_gate.quant_type, w_g, x_nf.data(), gate.data(),
                    K, H, inter, layer.q_ffn_gate.row_stride_bytes);
                cpu_quant::cpu_quant_gemv_batched(
                    layer.q_ffn_up.quant_type, w_u, x_nf.data(), up_b.data(),
                    K, H, inter, layer.q_ffn_up.row_stride_bytes);
            }

            // 2i. SiLU(gate) * up — parallel + AVX2.
            // Round2 agent_5 P1: was scalar + serial, 48k exp/token at K=5
            // wasted. Main decode has the AVX2+parallel pattern already;
            // just wasn't ported to the batched path.
            {
                const int64_t total = (int64_t)K * inter;
                c10::get_thread_pool().parallel_for(0, total,
                    [&](int64_t start, int64_t end) {
                    int64_t j = start;
#ifdef __AVX2__
                    for (; j + 7 < end; j += 8) {
                        __m256 g = _mm256_loadu_ps(gate.data() + j);
                        __m256 negg = _mm256_sub_ps(_mm256_setzero_ps(), g);
                        __m256 ex = at::native::vec::exp256_ps(negg);
                        __m256 den = _mm256_add_ps(_mm256_set1_ps(1.0f), ex);
                        __m256 s = _mm256_div_ps(g, den);
                        __m256 u = _mm256_loadu_ps(up_b.data() + j);
                        _mm256_storeu_ps(siluup.data() + j, _mm256_mul_ps(s, u));
                    }
#endif
                    for (; j < end; ++j) {
                        float g = gate[j];
                        float s = g / (1.0f + std::exp(-g));
                        siluup[j] = s * up_b[j];
                    }
                }, 1024);
            }

            // 2j. Batched ffn_down.
            {
                const void* w_d = layer.q_ffn_down.numa_replica.get(_node);
                if (!w_d) w_d = layer.q_ffn_down.cpu_data;
                cpu_quant::cpu_quant_gemv_batched(
                    layer.q_ffn_down.quant_type, w_d, siluup.data(), down.data(),
                    K, inter, H, layer.q_ffn_down.row_stride_bytes);
            }

            // 2k. Residual: next = cur + down — parallel AVX2.
            {
                const int64_t total = (int64_t)K * H;
                c10::get_thread_pool().parallel_for(0, total,
                    [&](int64_t start, int64_t end) {
                    int64_t j = start;
#ifdef __AVX2__
                    for (; j + 7 < end; j += 8) {
                        _mm256_storeu_ps(next + j,
                            _mm256_add_ps(_mm256_loadu_ps(cur + j),
                                          _mm256_loadu_ps(down.data() + j)));
                    }
#endif
                    for (; j < end; ++j) next[j] = cur[j] + down[j];
                }, 1024);
            }
            std::swap(cur, next);
        }

        // 3. Final RMSNorm (K parallel).
        const float* on_w = output_norm.data_ptr<float>();
        for (int k = 0; k < K; ++k) {
            cpu_quant::cpu_rmsnorm_out(cur + k * H, x_fin.data() + k * H, on_w, eps, add_one, H);
        }

        // 4. Batched output projection (biggest single GEMV — N = vocab_size).
        if (q_output_weight.valid && q_output_weight.cpu_data) {
            const void* w_o = q_output_weight.numa_replica.get(_node);
            if (!w_o) w_o = q_output_weight.cpu_data;
            cpu_quant::cpu_quant_gemv_batched(
                q_output_weight.quant_type, w_o, x_fin.data(), logits_out,
                K, H, V, q_output_weight.row_stride_bytes);
        } else {
            // FP32 fallback: scalar K × V × H (rare, never hot path on qwen3).
            const float* W = output_weight.data_ptr<float>();
            for (int k = 0; k < K; ++k) {
                for (int64_t n = 0; n < V; ++n) {
                    float dot = 0.0f;
                    const float* xp = x_fin.data() + k * H;
                    for (int64_t j = 0; j < H; ++j) dot += xp[j] * W[n * H + j];
                    logits_out[k * V + n] = dot;
                }
            }
        }

        // 5. Advance seq_len by K — every draft position is now in the KV
        // cache. Caller (spec_decode_step_cpu) rewinds if drafts rejected.
        kv_cache.seq_len = past_len_0 + K;
    }

    std::vector<int64_t> spec_decode_step_cpu(
            int64_t current_token,
            NgramDraft& draft,
            const std::vector<int64_t>& history,
            int K_max,
            SpecStats* stats = nullptr,
            float repetition_penalty = 1.0f) {

        std::vector<int64_t> out;
        out.reserve(K_max);

        // Draft up to K_max-1 additional tokens after current_token.
        // Contract: history already ends with current_token.
        std::vector<int64_t> rolling = history;

        std::vector<int64_t> drafted;
        drafted.reserve(K_max - 1);
        // Phase 7.5 — if a draft GGUF model is loaded, generate K-1 drafts
        // by running draft.forward_decode_cpu serially on current + drafts.
        // This typically achieves 0.5-0.85 acceptance on same-family pairs
        // vs ~0.1 for n-gram — the whole point of spec decode.
        if (draft_model_) {
            drafted = draft_predict_model(current_token, K_max - 1);
            if (stats) stats->drafts_proposed += static_cast<int64_t>(drafted.size());
        } else {
            // PT_PLD=1 enables Round 3 PLD draft (Saxena 2023): variable-n
            // match (3→2→1) over full history, returning up to K-1 tokens at
            // once. ON ELBRUS THIS REGRESSES (4.3 → 2.7 tok/s) because
            // forward_decode_cpu_batched is currently K-serial internally,
            // not truly batched — needs Phase 7.3 finishing work to deliver
            // the win. Default = old single-token NgramDraft.predict() loop.
            static const bool use_pld = [] {
                const char* e = std::getenv("PT_PLD");
                return e && e[0] == '1';
            }();
            if (use_pld) {
                drafted = draft.predict_pld(rolling, K_max - 1);
                if (stats) stats->drafts_proposed += static_cast<int64_t>(drafted.size());
            } else {
                for (int j = 0; j + 1 < K_max; ++j) {
                    int64_t d = draft.predict(rolling);
                    if (d < 0) break;
                    drafted.push_back(d);
                    rolling.push_back(d);
                    if (stats) ++stats->drafts_proposed;
                }
            }
        }

        // The sequence of tokens we'll verify, in order:
        //   inputs[0] = current_token
        //   inputs[1..D] = drafted
        // We run main forward on inputs[i] and check if argmax(logits_i)
        // matches inputs[i+1] (the expected next token).
        std::vector<int64_t> inputs;
        inputs.reserve(1 + drafted.size());
        inputs.push_back(current_token);
        for (auto t : drafted) inputs.push_back(t);

        // One batched forward pass over all K inputs at once. For now the
        // internal implementation is K serial forwards (same cost as the old
        // loop), but Phase 7.2 will swap inner GEMVs for batched variants —
        // no API changes needed here.
        const int64_t V = config.vocab_size;
        std::vector<float> logits_buf(inputs.size() * V);
        forward_decode_cpu_batched(inputs.data(),
                                   static_cast<int>(inputs.size()),
                                   logits_buf.data());
        if (stats) stats->main_forwards += static_cast<int64_t>(inputs.size());

        // Accept/reject loop against the batched logits.
        // Build a set of tokens for rep-pen (greedy path applies the penalty
        // by dividing positive logits / multiplying negative ones for tokens
        // that have appeared in history — matches the main decode's sampler).
        const bool apply_rep = repetition_penalty > 1.0001f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            float* lr = logits_buf.data() + i * V;
            if (apply_rep) {
                const float inv_rp = 1.0f / repetition_penalty;
                for (int64_t t : history) {
                    if (t >= 0 && t < V) {
                        lr[t] = (lr[t] > 0) ? (lr[t] * inv_rp)
                                            : (lr[t] * repetition_penalty);
                    }
                }
                // Also penalize any tokens committed earlier in THIS step.
                for (size_t j = 0; j < i; ++j) {
                    int64_t t = (j == 0) ? current_token : inputs[j];
                    if (t >= 0 && t < V) {
                        lr[t] = (lr[t] > 0) ? (lr[t] * inv_rp)
                                            : (lr[t] * repetition_penalty);
                    }
                }
            }
            int64_t best = 0;
            float best_v = lr[0];
            for (int64_t v = 1; v < V; ++v) {
                if (lr[v] > best_v) { best_v = lr[v]; best = v; }
            }

            if (i + 1 < inputs.size()) {
                if (best == inputs[i + 1]) {
                    out.push_back(inputs[i + 1]);
                    if (stats) ++stats->drafts_accepted;
                } else {
                    // REJECT: commit main's argmax. KV cache has entries for
                    // positions past_len_start .. past_len_start + (i-1) from
                    // this verify pass. Entry at past_len_start+i was created
                    // by the forward on inputs[i] itself. The "wrong" draft
                    // inputs[i+1] was NEVER forwarded, so its K/V slot was
                    // never written. But forward_decode_cpu_batched advanced
                    // seq_len by inputs.size(); we need to rewind by
                    // (inputs.size() - i - 1) slots.
                    const int64_t rewind = static_cast<int64_t>(inputs.size() - i - 1);
                    if (rewind > 0) kv_cache.seq_len -= rewind;
                    out.push_back(best);
                    break;
                }
            } else {
                out.push_back(best);
            }
        }

        if (stats) ++stats->steps;
        return out;
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

        // Phase 7 — multi-token speculative decode with batched verify. Opt
        // in via PT_SPEC_K=N (1..6). Only runs on the CPU greedy quant path.
        // NgramDraft is populated with prompt tokens so early decode steps
        // can already propose drafts based on prompt patterns.
        const int spec_k = io::spec_decode_k();
        // Allow rep-pen with spec: we apply it on the K batched logit rows
        // before argmax (see spec_decode_step_cpu). Greedy-only for now —
        // true probabilistic accept/reject needs temperature > 0.
        const bool can_spec_verify = (spec_k > 1) && !use_cuda_ &&
            use_quant_gemv_ && temperature < 1e-6f;

        // Low-rank output speculative (rank-256 GEMV + top-k exact). MUTEX
        // with multi-token speculative verify: the latter needs full-vocab
        // logits from forward_decode_cpu_batched, not a single top-k answer.
        bool can_speculative = !can_spec_verify &&
            use_speculative_output_ && !use_cuda_ &&
            use_quant_gemv_ && temperature < 1e-6f && repetition_penalty <= 1.0f;
        NgramDraft ngram_draft(2, 2048);
        io::SpecStats spec_stats;
        if (can_spec_verify) {
            for (auto t : tokens_i64) ngram_draft.append(t);
            // Phase 7.5 — load draft model if PT_SPEC_DRAFT_PATH is set.
            // Warm its KV cache by feeding the prompt tokens so it's aligned
            // with main at entry of the first decode step.
            ensure_draft_model(max_total_seq);
            if (draft_model_) {
                draft_model_->kv_cache.seq_len = 0;
                // Prefill draft with prompt. forward_decode_cpu appends
                // one token's KV per call.
                for (auto t : tokens_i64) {
                    (void)draft_model_->forward_decode_cpu(t);
                }
                std::fprintf(stderr,
                    "[spec-draft] prefilled KV to seq_len=%ld\n",
                    (long)draft_model_->kv_cache.seq_len);
            }
        }
        std::fprintf(stderr,
            "[spec-init] spec_k=%d can_spec_verify=%d use_cuda=%d "
            "use_quant_gemv=%d temperature=%.6f rep_pen=%.6f use_spec_out=%d\n",
            spec_k, (int)can_spec_verify, (int)use_cuda_,
            (int)use_quant_gemv_, temperature, repetition_penalty,
            (int)use_speculative_output_);

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

            // Print token as it's generated (no per-token flush on hot path — flush below at end)
            std::string token_str = tokenizer.decode_token(next_token);
#ifdef PT_DEBUG_DECODE
            std::cout << token_str << std::flush;
#else
            std::cout << token_str;
#endif

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
            } else if (can_spec_verify) {
                // PHASE 7 — multi-token speculative verify with batched forward.
                // spec_decode_step_cpu runs forward_decode_cpu_batched on
                // [next_token, draft_1, ..., draft_{K-1}], greedy-argmaxes
                // each logit row, accepts drafts that match and rewinds KV
                // on first mismatch.
                ngram_draft.append(static_cast<int64_t>(next_token));
                std::vector<int64_t> history;
                history.reserve(tokens_i64.size() + generated.size() + 1);
                for (auto t : tokens_i64) history.push_back(t);
                for (auto t : generated)  history.push_back((int64_t)t);
                // generated was just appended with next_token above, so it's
                // already in history. Do NOT push next_token again.
                auto out = spec_decode_step_cpu(
                    static_cast<int64_t>(next_token),
                    ngram_draft, history, spec_k, &spec_stats,
                    repetition_penalty);

                // Commit every accepted draft plus the main's answer at the
                // end. generate()'s main loop increments `step` by 1 per
                // iteration, so we print the extra tokens here without
                // advancing the loop's step counter — decode will naturally
                // exit when max_tokens is reached via generated.size().
                // out[0] is ALWAYS either an accepted draft or main's first
                // argmax, both of which are the "second" committed token
                // for this verify step (next_token itself is the first and
                // was already printed + generated.push_back'd above).
                int32_t last_committed = next_token;
                for (size_t i = 0; i < out.size(); ++i) {
                    int32_t tok = (int32_t)out[i];
                    generated.push_back(tok);
                    ngram_draft.append(out[i]);
                    std::string ts = tokenizer.decode_token(tok);
#ifdef PT_DEBUG_DECODE
                    std::cout << ts << std::flush;
#else
                    std::cout << ts;
#endif
                    last_committed = tok;
                    if (tok == tokenizer.eos_id || is_stop_token(tok)) break;
                    if ((int)generated.size() >= max_tokens) break;
                }
                // After spec_decode_step, the LAST committed token's KV is
                // NOT yet in the cache (by contract). Forward it now to
                // advance KV and get logits for the next iteration.
                if (!out.empty() &&
                    last_committed != tokenizer.eos_id &&
                    !is_stop_token(last_committed) &&
                    (int)generated.size() < max_tokens) {
                    logits = forward_decode_cpu(static_cast<int64_t>(last_committed));
                    // Phase 7.5 — sync draft KV to main's new seq_len AND
                    // ensure draft has last_committed's KV too (so next step
                    // can propose drafts from the correct context).
                    if (draft_model_) {
                        // Drop draft's excess KV (it had forwarded K-1
                        // drafts; main committed fewer on mismatch, or
                        // committed best_{K-1} which draft never saw).
                        int64_t target = kv_cache.seq_len - 1;  // seq BEFORE last_committed
                        if (draft_model_->kv_cache.seq_len > target) {
                            draft_model_->kv_cache.seq_len = target;
                        }
                        (void)draft_model_->forward_decode_cpu(
                            static_cast<int64_t>(last_committed));
                    }
                }
                // Honour the user-facing max_tokens budget: spec commits up
                // to K tokens per iteration, which would otherwise overshoot.
                if ((int)generated.size() >= max_tokens) break;
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

        // Phase 7 — speculative decode stats dump (only if activated).
        if (can_spec_verify && spec_stats.steps > 0) {
            std::fprintf(stderr,
                "[spec] K=%d  steps=%ld  main_forwards=%ld  drafts_proposed=%ld  "
                "drafts_accepted=%ld  acceptance=%.3f  tokens_per_step=%.2f\n",
                spec_k,
                (long)spec_stats.steps,
                (long)spec_stats.main_forwards,
                (long)spec_stats.drafts_proposed,
                (long)spec_stats.drafts_accepted,
                spec_stats.acceptance_rate(),
                (double)generated.size() / (double)spec_stats.steps);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tokens_per_sec = generated.size() / (ms / 1000.0);

        std::cout << "\n\n[Generate] " << generated.size() << " tokens in "
                  << (ms / 1000.0) << "s (" << tokens_per_sec << " tok/s)" << std::endl;

        sec_timers_.dump();

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
    // Tensor-Parallel initialization (multi-process DDP CPU inference).
    //
    // Must be called AFTER load() (full weights present as QuantizedWeight on
    // each rank) AND AFTER torch::distributed::init_process_group(cfg).
    //
    // Effect: row-slices the Q/K/V and FFN gate/up quantized weights by output
    // rows so rank r only keeps its local slice. attn_output / ffn_down /
    // output_weight stay replicated — each rank keeps the full weight pointer.
    //
    // Returns true on success, false if constraints are violated (in which
    // case tp_ stays disabled and the caller should fall back to single
    // process decode).
    // ========================================================================
    bool init_tensor_parallel(int rank, int nprocs) {
        if (nprocs <= 1) {
            tp_.enabled = false;
            return false;
        }
        if (!use_quant_gemv_) {
            std::cerr << "[TP] init_tensor_parallel requires quantized weights "
                         "loaded (call load() first). Skipping." << std::endl;
            return false;
        }
        int64_t n_heads    = config.num_heads;
        int64_t n_kv_heads = config.num_kv_heads;
        int64_t head_dim   = config.head_dim;
        int64_t inter      = config.intermediate_size;

        if (n_heads % nprocs != 0) {
            std::cerr << "[TP] num_heads (" << n_heads << ") must be divisible "
                         "by nprocs (" << nprocs << ")." << std::endl;
            return false;
        }
        if (n_kv_heads % nprocs != 0) {
            std::cerr << "[TP] num_kv_heads (" << n_kv_heads << ") must be "
                         "divisible by nprocs (" << nprocs << ")." << std::endl;
            return false;
        }
        // inter_local is non-uniform: rank gets super-block count =
        //   per_rank_blocks + (rank < rem ? 1 : 0), each 256 elements.
        // This must match the K-slicing of ffn_down for correctness.
        if (inter % 256 != 0) {
            std::cerr << "[TP] intermediate_size (" << inter << ") must be "
                         "multiple of 256 (Q4_K super-block granularity)." << std::endl;
            return false;
        }

        tp_.enabled          = true;
        tp_.rank             = rank;
        tp_.nprocs           = nprocs;
        tp_.n_heads_local    = n_heads / nprocs;
        tp_.n_kv_heads_local = n_kv_heads / nprocs;
        tp_.head_start       = rank * tp_.n_heads_local;
        tp_.head_end         = tp_.head_start + tp_.n_heads_local;
        tp_.kv_head_start    = rank * tp_.n_kv_heads_local;
        tp_.kv_head_end      = tp_.kv_head_start + tp_.n_kv_heads_local;
        tp_.q_dim_local      = tp_.n_heads_local * head_dim;
        tp_.kv_dim_local     = tp_.n_kv_heads_local * head_dim;

        // Inter partition: uniform (inter/nprocs per rank) when PT_TP_GATHER=1
        // because all_gather_inplace requires equal per_rank_count across ranks.
        // Legacy path keeps the non-uniform super-block partition so that the
        // K-slice of ffn_down respects Q4_K super-block (256-element) alignment.
        const char* gather_env_setup = std::getenv("PT_TP_GATHER");
        const bool gather_mode_setup = gather_env_setup && gather_env_setup[0] == '1';
        if (gather_mode_setup) {
            if (inter % nprocs != 0) {
                std::cerr << "[TP] PT_TP_GATHER requires intermediate_size ("
                          << inter << ") divisible by nprocs (" << nprocs << ")."
                          << std::endl;
                return false;
            }
            tp_.inter_local  = inter / nprocs;
            tp_.inter_offset = rank * tp_.inter_local;
        } else {
            // Non-uniform super-block partition of inter dim. Granularity must
            // match the K-slice block size used for ffn_down (and the row-slice
            // alignment for ffn_gate/ffn_up which feed silu_local). Default 256
            // for Q4_K/Q6_K; Q8_0 has 32-elem blocks so partition must align to 32.
            // Mismatch here was the root cause of Q8_0 TP-4 producing NaN
            // ("!!!!!!" decode): rank's silu region (256-aligned) didn't agree
            // with the K-slice partition (32-aligned), causing OOB reads in
            // q8_0_gemv_k_slice → uninitialized memory → NaN cascade.
            int64_t inter_qbe = 256;
            if (config.num_layers > 0 && layers[0].q_ffn_down.valid) {
                uint32_t qt = layers[0].q_ffn_down.quant_type;
                if (qt == 8)                     inter_qbe = 32;   // Q8_0
                else if (qt == 12 || qt == 14)   inter_qbe = 256;  // Q4_K / Q6_K
            }
            if (inter % inter_qbe != 0) {
                std::cerr << "[TP] inter (" << inter << ") not divisible by "
                          << inter_qbe << " for ffn_down quant_type" << std::endl;
                return false;
            }
            int64_t inter_total_blocks = inter / inter_qbe;
            int64_t per_rank_blocks = inter_total_blocks / nprocs;
            int64_t rem_blocks = inter_total_blocks % nprocs;
            int64_t my_blocks = per_rank_blocks + (rank < rem_blocks ? 1 : 0);
            int64_t my_block_start = rank * per_rank_blocks + std::min<int64_t>(rank, rem_blocks);
            tp_.inter_local  = my_blocks * inter_qbe;
            tp_.inter_offset = my_block_start * inter_qbe;
        }

        // Row-slice Q/K/V/gate/up from the already-loaded full quantized weights
        tp_.layers.resize(config.num_layers);
        auto slice_rows = [&](const QuantizedWeight& full, TPSlicedWeight& out,
                              int64_t row_start_elems, int64_t row_count_elems) -> bool {
            if (!full.valid || !full.cpu_data) return false;
            if (row_start_elems + row_count_elems > full.rows) return false;
            // Each output "row" in quantized form is full.row_stride_bytes bytes.
            int64_t stride = full.row_stride_bytes;
            int64_t local_bytes = row_count_elems * stride;
            out.cpu_data = std::malloc(local_bytes);
            if (!out.cpu_data) return false;
            std::memcpy(out.cpu_data,
                        static_cast<const char*>(full.cpu_data) + row_start_elems * stride,
                        local_bytes);
            out.rows = row_count_elems;
            out.cols = full.cols;
            out.row_stride_bytes = stride;
            out.total_bytes = local_bytes;
            out.quant_type = full.quant_type;
            out.valid = true;
            return true;
        };

        int64_t q_row_start    = rank * tp_.q_dim_local;
        int64_t kv_row_start   = rank * tp_.kv_dim_local;
        int64_t inter_row_start = tp_.inter_offset;  // non-uniform offset

        // K-DIM SLICING lambda — pulled out of the layer loop so it can be
        // reused for output_weight K-slice (Round 3 Option D) after the loop.
        auto slice_k_blocks = [&](const QuantizedWeight& full, TPSlicedWeight& out,
                                   int64_t& k_start_out, int64_t& k_end_out,
                                   int64_t& k_local_out, const char* dbg_name) -> bool {
            if (!full.valid || !full.cpu_data) {
                std::cerr << "[TP slice_k] " << dbg_name
                          << " fail: valid=" << full.valid
                          << " cpu=" << (full.cpu_data != nullptr) << std::endl;
                return false;
            }
            int64_t bytes_per_block, elems_per_block;
            constexpr uint32_t PT8_NATIVE_Q8_SOA4_K = 0x100u | PT8_TYPE_Q8_0_SOA4;
            if (full.quant_type == 12)      { bytes_per_block = 144; elems_per_block = 256; }
            else if (full.quant_type == 14) { bytes_per_block = 210; elems_per_block = 256; }
            else if (full.quant_type ==  8) { bytes_per_block =  34; elems_per_block =  32; }
            else if (full.quant_type == PT8_NATIVE_Q8_SOA4_K) {
                // PT8 Q8_SoA4 K-slice has non-row-major super-block headers
                // (4× fp32 d_w / dmin_m / i32 sum_q live at the head of each
                // 176-byte super-row block, shared across the 4 rows). The
                // generic per-row memcpy below assumes row-major bytes —
                // applying it here would scramble the headers. Until a
                // dedicated K-slice path lands (Agent D, alongside the q8/q4
                // _soa4_gemv K-slice variant), fall back to the replicated
                // full-weight path. For TP-4 on qwen3:4b that means each rank
                // reads the full attn_output / ffn_down once per token —
                // identical to Round 3's behaviour for Q5_K/Q6_K weights.
                std::cerr << "[TP slice_k] " << dbg_name
                          << " PT8_Q8_SOA4 K-slice not yet implemented "
                          << "— falling back to replicated full-weight path."
                          << std::endl;
                return false;
            }
            else {
                std::cerr << "[TP slice_k] " << dbg_name
                          << " unsupported qtype=" << full.quant_type
                          << " (will use replicated fallback)" << std::endl;
                return false;
            }
            int64_t K_full = full.cols;
            if (K_full % elems_per_block != 0) {
                std::cerr << "[TP slice_k] " << dbg_name << " K_full=" << K_full
                          << " not multiple of " << elems_per_block << std::endl;
                return false;
            }
            int64_t total_blocks = K_full / elems_per_block;
            int64_t per_rank = total_blocks / nprocs;
            int64_t rem = total_blocks % nprocs;
            int64_t k_start = rank * per_rank + std::min((int64_t)rank, rem);
            int64_t local_blocks = per_rank + (rank < rem ? 1 : 0);
            int64_t k_end = k_start + local_blocks;

            int64_t local_row_stride = local_blocks * bytes_per_block;
            int64_t full_row_stride = full.row_stride_bytes;
            int64_t total_local_bytes = full.rows * local_row_stride;

            out.cpu_data = std::malloc(total_local_bytes);
            if (!out.cpu_data) return false;

            char* dst = static_cast<char*>(out.cpu_data);
            const char* src = static_cast<const char*>(full.cpu_data);
            int64_t offset_bytes = k_start * bytes_per_block;
            for (int64_t n = 0; n < full.rows; ++n) {
                std::memcpy(dst + n * local_row_stride,
                            src + n * full_row_stride + offset_bytes,
                            local_row_stride);
            }
            out.rows = full.rows;
            out.cols = local_blocks * elems_per_block;
            out.row_stride_bytes = local_row_stride;
            out.total_bytes = total_local_bytes;
            out.quant_type = full.quant_type;
            out.valid = true;
            k_start_out = k_start;
            k_end_out = k_end;
            k_local_out = local_blocks * elems_per_block;
            return true;
        };

        for (int64_t i = 0; i < config.num_layers; ++i) {
            const auto& layer = layers[i];
            auto& tl = tp_.layers[i];
            // Attention projections: output rows correspond to head lanes.
            // attn_q has rows = n_heads * head_dim, so slicing rows
            // [rank * q_dim_local, (rank+1) * q_dim_local) picks rank's heads.
            if (!slice_rows(layer.q_attn_q, tl.q_attn_q, q_row_start, tp_.q_dim_local)) {
                std::cerr << "[TP] failed to row-slice layer " << i << " attn_q" << std::endl;
                tp_.enabled = false;
                return false;
            }
            if (!slice_rows(layer.q_attn_k, tl.q_attn_k, kv_row_start, tp_.kv_dim_local)) {
                std::cerr << "[TP] failed to row-slice layer " << i << " attn_k" << std::endl;
                tp_.enabled = false;
                return false;
            }
            if (!slice_rows(layer.q_attn_v, tl.q_attn_v, kv_row_start, tp_.kv_dim_local)) {
                std::cerr << "[TP] failed to row-slice layer " << i << " attn_v" << std::endl;
                tp_.enabled = false;
                return false;
            }
            if (!slice_rows(layer.q_ffn_gate, tl.q_ffn_gate, inter_row_start, tp_.inter_local)) {
                std::cerr << "[TP] failed to row-slice layer " << i << " ffn_gate" << std::endl;
                tp_.enabled = false;
                return false;
            }
            if (!slice_rows(layer.q_ffn_up, tl.q_ffn_up, inter_row_start, tp_.inter_local)) {
                std::cerr << "[TP] failed to row-slice layer " << i << " ffn_up" << std::endl;
                tp_.enabled = false;
                return false;
            }

            // ============================================================
            // K-DIM SLICING for attn_output and ffn_down (RowParallel).
            // Lambda hoisted out to function scope above to be reused for
            // output_weight after the layer loop (Round 3 Option D).
            // ============================================================

            // K-slicing for Q4_K only. Q5_K/Q6_K fall back to replicated path
            // (marked via tl.q_attn_output.valid=false; forward will see valid=0
            // and use the full-weight zero-padded path).
            //
            // Skip K-slice allocation under PT_TP_GATHER — gather path uses
            // replicated weights + pointer-offset N-slicing, never touches
            // tl.q_attn_output / tl.q_ffn_down. Saves ~(attn_output + ffn_down)
            // / nprocs bytes per layer (~115 MB for qwen3:4b Q4_K_M on N=4).
            if (!gather_mode_setup) {
                slice_k_blocks(layer.q_attn_output, tl.q_attn_output,
                               tl.attn_output_k_start, tl.attn_output_k_end,
                               tl.attn_output_k_local, "attn_output");
                slice_k_blocks(layer.q_ffn_down, tl.q_ffn_down,
                               tl.ffn_down_k_start, tl.ffn_down_k_end,
                               tl.ffn_down_k_local, "ffn_down");
            }
        }

        // Round 3 Option D: K-slice the output (lm_head) projection ONCE
        // (not per-layer). Currently each rank reads full output_weight
        // ~175 MB/token; K-sliced → 44 MB/rank/token = -131 MB aggregate
        // per-token bandwidth. Followed by AllReduce-sum on logits buffer.
        // Per Round 2 agent_9 §4.2: +14% (about +0.7 tok/s) alone.
        if (!gather_mode_setup && q_output_weight.valid && q_output_weight.cpu_data) {
            slice_k_blocks(q_output_weight, tp_.q_output_weight_k_slice,
                           tp_.output_weight_k_start, tp_.output_weight_k_end,
                           tp_.output_weight_k_local, "output_weight");
        }

        // Allocate scratch buffers (local dims for Q/K/V/gate/up,
        // full dims for attn (zero-padded) and silu (zero-padded)).
        int64_t H     = config.hidden_size;
        int64_t q_dim = n_heads * head_dim;
        tp_.x_buf[0].assign(H, 0.0f);
        tp_.x_buf[1].assign(H, 0.0f);
        tp_.q_local_buf.assign(tp_.q_dim_local, 0.0f);
        tp_.k_local_buf.assign(tp_.kv_dim_local, 0.0f);
        tp_.v_local_buf.assign(tp_.kv_dim_local, 0.0f);
        tp_.attn_full_buf.assign(q_dim, 0.0f);
        tp_.h_buf.assign(H, 0.0f);
        tp_.gate_local_buf.assign(tp_.inter_local, 0.0f);
        tp_.up_local_buf.assign(tp_.inter_local, 0.0f);
        tp_.silu_full_buf.assign(inter, 0.0f);
        tp_.logits_buf.assign(config.vocab_size, 0.0f);
        // Q8 SoA4 activation broadcast scratch sized to max K seen at runtime.
        // K_max = max(hidden, q_dim_local, inter, vocab_K_slice) — the largest
        // is `inter` (9728 for qwen3:4b). a_b16 = K*4 bytes; sum_a = K/32 ints.
        int64_t K_act_max = std::max<int64_t>({(int64_t)H, (int64_t)inter,
                                                (int64_t)tp_.q_dim_local,
                                                (int64_t)config.vocab_size});
        tp_.soa_act_b16.assign(K_act_max * 4, 0);
        tp_.soa_sum_a.assign(K_act_max / 32 + 1, 0);
        // Round 4 Step 9: persistent per-layer scratch
        tp_.x_normed_buf.assign(H, 0.0f);
        tp_.silu_scratch_buf.assign(tp_.inter_local, 0.0f);
        tp_.scratch_ready = true;

        // Round 3 Agent 5: Q8 SoA4 weight repack — gated by PT_Q8_SOA=1.
        // For each TP-sliced Q4_K weight, build the 4-row interleaved Q8
        // layout used by q8_soa4_gemv. ~1.32× memory inflation per weight,
        // but enables 4-row parallel qpmaddubsh in inner loop. Microbench
        // shows 0.85× EML (1.21 ms vs 1.03 ms for K=2560 N=2432 single-core).
        //
        // Round 4 Agent C: when input was a PT8 file with PT8_TYPE_Q8_0_SOA4
        // tensors, the sliced bytes are already in the runtime Q8SoA4
        // layout — `try_repack` becomes a zero-cost view binding rather than
        // a Q4_K → SoA4 repack. PT_Q8_SOA=1 is implied for PT8 SoA4 files.
        const char* soa_env = std::getenv("PT_Q8_SOA");
        bool use_q8_soa = (soa_env && soa_env[0] == '1') || use_pt8_;
        if (use_q8_soa) {
            int64_t soa_count = 0;
            int64_t soa_bytes = 0;
            int64_t pt8_native_count = 0;
            constexpr uint32_t PT8_NATIVE_Q8_SOA4 = 0x100u | PT8_TYPE_Q8_0_SOA4;

            auto try_repack = [&](TPSlicedWeight& tl, const char* name) {
                if (!tl.valid || !tl.cpu_data) return;
                if (tl.rows % 4 != 0) return;

                // -- PT8 NATIVE FAST PATH ---------------------------------
                // The sliced bytes ARE the Q8SoA4 storage. Bind tl.q8_soa
                // to point at them — no allocation, no math, no repack.
                // Lifetime: tl owns cpu_data via malloc (slice_rows malloc'd
                // it from the mmap), so we hand off ownership by moving the
                // pointer into q8_soa.mem. We then null cpu_data so the
                // TPSlicedWeight destructor doesn't double-free.
                if (tl.quant_type == PT8_NATIVE_Q8_SOA4) {
                    int64_t bpr = tl.cols / 32;
                    if (tl.cols % 32 != 0) {
                        std::cerr << "[pt8-soa4] " << name
                                  << " cols=" << tl.cols
                                  << " not multiple of 32 — skipping" << std::endl;
                        return;
                    }
                    tl.q8_soa.mem          = static_cast<uint8_t*>(tl.cpu_data);
                    tl.q8_soa.N            = tl.rows;
                    tl.q8_soa.K            = tl.cols;
                    tl.q8_soa.group_stride = bpr * SOA4_GROUP_BYTES;
                    tl.q8_soa.valid        = true;
                    // Hand off ownership: cpu_data was malloc'd by slice_rows
                    // — destructor is in cpu_quant::Q8SoA4::~Q8SoA4 (frees
                    // via std::free). Null cpu_data on the slice so its
                    // destructor doesn't double-free.
                    tl.cpu_data = nullptr;
                    soa_count++;
                    pt8_native_count++;
                    soa_bytes += (tl.rows / 4) * tl.q8_soa.group_stride;
                    return;
                }

                // -- PT8 Q4_SOA4 (Agent D kernel pending) -----------------
                // Format spec §10's hot dtype. The byte layout is known
                // (88 B / 4-row × 32-K block), but `q4_soa4_gemv` is
                // Agent D's deliverable. Until that lands, we surface a
                // clear diagnostic instead of silently mismatching.
                constexpr uint32_t PT8_NATIVE_Q4_SOA4 = 0x100u | PT8_TYPE_Q4K_SOA4;
                if (tl.quant_type == PT8_NATIVE_Q4_SOA4) {
                    static bool warned = false;
                    if (!warned) {
                        std::cerr << "[pt8-q4soa4] PT8_TYPE_Q4K_SOA4 detected for "
                                  << name << " but q4_soa4_gemv kernel is not "
                                     "yet implemented (Agent D dependency). "
                                     "Forward path will fall back to whatever "
                                     "non-SoA dispatcher matches quant_type."
                                  << std::endl;
                        warned = true;
                    }
                    return;
                }

                // -- Q4_K REPACK PATH (legacy) ----------------------------
                if (tl.quant_type != 12) return;  // Q4_K only for now
                if (tl.cols % 256 != 0) return;
                if (cpu_quant::repack_q4k_to_q8soa4(
                        tl.cpu_data, tl.rows, tl.cols,
                        tl.row_stride_bytes, &tl.q8_soa)) {
                    soa_count++;
                    soa_bytes += (tl.rows / 4) * tl.q8_soa.group_stride;
                } else {
                    std::cerr << "[Q8_SOA] failed to repack " << name
                              << " rows=" << tl.rows << " cols=" << tl.cols << std::endl;
                }
            };
            for (int64_t i = 0; i < config.num_layers; ++i) {
                auto& tl = tp_.layers[i];
                try_repack(tl.q_attn_q,    "attn_q");
                try_repack(tl.q_attn_k,    "attn_k");
                try_repack(tl.q_attn_v,    "attn_v");
                try_repack(tl.q_ffn_gate,  "ffn_gate");
                try_repack(tl.q_ffn_up,    "ffn_up");
                try_repack(tl.q_attn_output,"attn_output");
                try_repack(tl.q_ffn_down,  "ffn_down");
            }
            try_repack(tp_.q_output_weight_k_slice, "output_weight");
            std::cout << "[Q8_SOA] " << soa_count
                      << " TP-sliced weights ready, " << (soa_bytes / (1024*1024))
                      << " MB SoA4 storage";
            if (pt8_native_count > 0) {
                std::cout << " (" << pt8_native_count
                          << " native PT8 — repack skipped)";
            }
            std::cout << std::endl;
        }

        std::cout << "[TP] Tensor-parallel enabled: rank " << rank << "/" << nprocs
                  << " heads[" << tp_.head_start << ".." << tp_.head_end << ") "
                  << "kv_heads[" << tp_.kv_head_start << ".." << tp_.kv_head_end << ") "
                  << "inter[" << inter_row_start << ".." << (inter_row_start + tp_.inter_local) << ")"
                  << std::endl;

        // Free the full-size row-split weights (q_attn_q/k/v, ffn_gate/up) to save RAM.
        // attn_output, ffn_down, output_weight stay as full QuantizedWeight per rank.
        // Do this only if the full weights are NOT mmap'd (we cannot free mmap pointers).
        for (int64_t i = 0; i < config.num_layers; ++i) {
            auto& full = layers[i];
            auto free_if_owned = [](QuantizedWeight& qw) {
                if (qw.valid && qw.cpu_data && !qw.mmap_owned) {
                    std::free(qw.cpu_data);
                    qw.cpu_data = nullptr;
                    qw.valid = false;
                }
            };
            free_if_owned(full.q_attn_q);
            free_if_owned(full.q_attn_k);
            free_if_owned(full.q_attn_v);
            free_if_owned(full.q_ffn_gate);
            free_if_owned(full.q_ffn_up);
        }

        return true;
    }

    // Allocate local KV cache (per-rank, kv_dim_local columns)
    void tp_allocate_kv_cache(int64_t max_seq_len) {
        tp_.k_cache_local.resize(config.num_layers);
        tp_.v_cache_local.resize(config.num_layers);
        for (int64_t i = 0; i < config.num_layers; ++i) {
            tp_.k_cache_local[i] = at::empty({max_seq_len, tp_.kv_dim_local});
            tp_.v_cache_local[i] = at::empty({max_seq_len, tp_.kv_dim_local});
        }
        tp_.kv_max_seq = max_seq_len;
        tp_.kv_seq_len = 0;
    }

    // ========================================================================
    // Tensor-parallel CPU decode (1 token).
    // Mirrors forward_decode_cpu() but:
    //   - Q/K/V use row-sliced local weights (local_q_dim, local_kv_dim).
    //   - Attention runs on local heads only (uses local KV cache).
    //   - Output proj runs on FULL replicated weight with zero-padded input
    //     (only rank's heads filled in attn_full_buf) → AllReduce SUM.
    //   - FFN gate/up use row-sliced local weights.
    //   - FFN down runs on FULL replicated weight with zero-padded silu_full_buf
    //     → AllReduce SUM.
    //   - Final RMSNorm + output projection: replicated (no AllReduce).
    // Returns logits [1, vocab_size] (identical on all ranks).
    // ========================================================================
    // Per-section TP timers. Same shape as sec_timers_ but with an extra
    // `allreduce_ms` bucket — this is the TP-unique cost we want to isolate.
    mutable struct TPSectionTimers {
        bool on = false;
        bool inited = false;
        double qkv_ms = 0, attn_ms = 0, ao_ms = 0, allreduce_ao_ms = 0;
        double gateup_ms = 0, silu_ms = 0, fdown_ms = 0, allreduce_fdown_ms = 0;
        double output_proj_ms = 0, tail_ms = 0;
        int64_t tokens = 0;
        void init() {
            if (inited) return;
            const char* env = std::getenv("PT_PROFILE_LAYER");
            on = (env && env[0] == '1');
            inited = true;
        }
        void dump(int rank) const {
            if (!on || tokens == 0 || rank != 0) return;
            double total = qkv_ms+attn_ms+ao_ms+allreduce_ao_ms+gateup_ms+silu_ms+fdown_ms
                          +allreduce_fdown_ms+output_proj_ms+tail_ms;
            std::fprintf(stderr,
                "[prof-TP rank0] %ld tokens, avg ms/token:\n"
                "  attn_phase:        %6.2f  (RMSNorm+QKV+bias+QKnorm+RoPE+KVcache+attn math)\n"
                "  attn_output:       %6.2f\n"
                "  allreduce(ao):     %6.2f\n"
                "  gate_up:           %6.2f  (RMSNorm+gate+up GEMVs)\n"
                "  ffn_down:          %6.2f  (incl. SiLU*up)\n"
                "  allreduce(fdown):  %6.2f\n"
                "  output_proj:       %6.2f  (final RMSNorm + full vocab GEMV)\n"
                "  tail:              %6.2f  (embedding lookup + tensor wrap)\n"
                "  sum:               %6.2f\n",
                (long)tokens, attn_ms/tokens, ao_ms/tokens,
                allreduce_ao_ms/tokens, gateup_ms/tokens,
                fdown_ms/tokens, allreduce_fdown_ms/tokens,
                output_proj_ms/tokens, tail_ms/tokens, total/tokens);
        }
    } tp_sec_timers_;

    Tensor forward_decode_cpu_tp(int64_t token_id) {
        if (!tp_.enabled) {
            throw std::runtime_error("forward_decode_cpu_tp: tp_ not initialized");
        }
        tp_sec_timers_.init();
        const int64_t H          = config.hidden_size;
        const int64_t head_dim   = config.head_dim;
        const int64_t inter      = config.intermediate_size;
        const int64_t q_dim      = config.num_heads * config.head_dim;
        const int64_t kv_dim     = config.num_kv_heads * config.head_dim;
        const int64_t past_len   = tp_.kv_seq_len;
        const float eps          = config.rms_norm_eps;
        const bool add_one       = config.gemma_norm_add_one;
        const int64_t n_heads_l  = tp_.n_heads_local;
        const int64_t n_kv_l     = tp_.n_kv_heads_local;
        const int64_t heads_per_group = config.num_heads / config.num_kv_heads;

        // Option F gather path: replaces the 2 per-layer AllReduce-SUM calls
        // with 4 AllGather-CONCAT calls. Each GEMV becomes N-slice (rank r
        // computes 1/N of output rows from the FULL replicated weight), and
        // adjacent ops are joined by all_gather_inplace. Net wins come from
        // (1) no serial reducer bottleneck in all_gather, (2) smaller per-call
        // payload (H/N vs H), (3) more opportunities for futex-accelerated
        // barriers. Requires H, q_dim, inter divisible by nprocs.
        static const bool use_gather = [] {
            const char* e = std::getenv("PT_TP_GATHER");
            return e && e[0] == '1';
        }();
        if (use_gather) {
            if (H % tp_.nprocs != 0 || q_dim % tp_.nprocs != 0 || inter % tp_.nprocs != 0) {
                throw std::runtime_error(
                    "PT_TP_GATHER: H/q_dim/inter must be divisible by nprocs");
            }
        }
        const int64_t H_local = H / tp_.nprocs;
        const int64_t h_row_start = tp_.rank * H_local;
        // Within-rank group offset mapping (local Q head -> local KV head):
        // Global Q head h → global KV head h/heads_per_group. Since Q and KV
        // heads are sliced in aligned chunks (rank gets contiguous head range
        // of both), local_kv_h = local_q_h / heads_per_group.

        // Ensure KV cache is allocated
        if (tp_.kv_max_seq == 0) {
            tp_allocate_kv_cache(config.context_length > 0 ? config.context_length : 4096);
        }

        int cur = 0;
        // 1. Embedding lookup (replicated on every rank)
        const float* emb_table = token_embedding.data_ptr<float>();
        std::memcpy(tp_.x_buf[cur].data(), emb_table + token_id * H, H * sizeof(float));
        if (config.scale_embeddings) {
            float scale = std::sqrt(static_cast<float>(H));
            for (int64_t j = 0; j < H; ++j) tp_.x_buf[cur][j] *= scale;
        }

        // 2. Transformer layers
        auto _tp_t = std::chrono::high_resolution_clock::now();
        auto _tp_elapsed = [&_tp_t]() {
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - _tp_t).count();
            _tp_t = t1;
            return ms;
        };

        for (int64_t i = 0; i < config.num_layers; ++i) {
            const auto& layer   = layers[i];
            const auto& tl      = tp_.layers[i];
            float* x_ptr = tp_.x_buf[cur].data();

            if (tp_sec_timers_.on) _tp_elapsed();  // reset baseline for this layer

            // --- RMSNorm(x) + fused Q/K/V GEMV (row-sliced weights) ---
            // Profiler shows attn_phase = 34 ms/token; most of it is RMSNorm-
            // scan of x + 3 separate GEMV passes. Fusing RMSNorm into the
            // first GEMV pass (and batching Q/K/V through a single input
            // stream) eliminates 3× reads of x_normed and 2× redundant
            // RMSNorm scans. See cpu_fused_rmsnorm_qkv_gemv for details.
            // x_normed reuses persistent tp_.x_normed_buf (Step 9: avoids
            // 36 layers × 1 vector alloc/dealloc per token).
            std::vector<float>& x_normed = tp_.x_normed_buf;
            float* q_l = tp_.q_local_buf.data();
            float* k_l = tp_.k_local_buf.data();
            float* v_l = tp_.v_local_buf.data();
            // Q8_SoA4 path (PT_Q8_SOA=1): if SoA4 builds exist for Q/K/V,
            // use the 4-row interleaved kernel. Otherwise fall back to
            // existing fused/scalar Q4_K kernel.
            bool use_soa_qkv = tl.q_attn_q.q8_soa.valid &&
                               tl.q_attn_k.q8_soa.valid &&
                               tl.q_attn_v.q8_soa.valid;
            bool can_fuse_qkv = tl.q_attn_q.valid && tl.q_attn_k.valid && tl.q_attn_v.valid &&
                tl.q_attn_q.quant_type == tl.q_attn_k.quant_type &&
                tl.q_attn_q.quant_type == tl.q_attn_v.quant_type &&
                tl.q_attn_q.row_stride_bytes == tl.q_attn_k.row_stride_bytes &&
                tl.q_attn_q.row_stride_bytes == tl.q_attn_v.row_stride_bytes &&
                cpu_quant::cpu_quant_gemv_supported(tl.q_attn_q.quant_type);
            if (use_soa_qkv) {
                // RMSNorm into x_normed, then int8-quant + broadcast once for QKV.
                std::memcpy(x_normed.data(), x_ptr, H * sizeof(float));
                cpu_quant::cpu_rmsnorm_inplace(x_normed.data(),
                    layer.attn_norm.data_ptr<float>(), eps, add_one, H);
                cpu_quant::q8_soa4_quant_activation(x_normed.data(), H,
                    tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                    &tp_.soa_scale_a);
                // Triple-fused: 1 parallel_for dispatch вместо 3, shared
                // activation reads, shared pool wakeup. Save ~3×200μs ×
                // 36 layers ≈ 20 ms/token overhead.
                cpu_quant::q8_soa4_gemv_triple(
                    &tl.q_attn_q.q8_soa, &tl.q_attn_k.q8_soa, &tl.q_attn_v.q8_soa,
                    tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                    tp_.soa_scale_a, q_l, k_l, v_l);
            } else if (can_fuse_qkv) {
                cpu_quant::cpu_fused_rmsnorm_qkv_gemv(
                    x_ptr, layer.attn_norm.data_ptr<float>(), eps, add_one,
                    tl.q_attn_q.quant_type,
                    tl.q_attn_q.cpu_data, tl.q_attn_k.cpu_data, tl.q_attn_v.cpu_data,
                    q_l, k_l, v_l,
                    H, tl.q_attn_q.rows, tl.q_attn_k.rows, tl.q_attn_v.rows,
                    tl.q_attn_q.row_stride_bytes);
            } else {
                std::memcpy(x_normed.data(), x_ptr, H * sizeof(float));
                cpu_quant::cpu_rmsnorm_inplace(x_normed.data(),
                    layer.attn_norm.data_ptr<float>(), eps, add_one, H);
                if (tl.q_attn_q.valid)
                    cpu_quant::cpu_quant_gemv(tl.q_attn_q.quant_type, tl.q_attn_q.cpu_data,
                        x_normed.data(), q_l, H, tl.q_attn_q.rows, tl.q_attn_q.row_stride_bytes);
                if (tl.q_attn_k.valid)
                    cpu_quant::cpu_quant_gemv(tl.q_attn_k.quant_type, tl.q_attn_k.cpu_data,
                        x_normed.data(), k_l, H, tl.q_attn_k.rows, tl.q_attn_k.row_stride_bytes);
                if (tl.q_attn_v.valid)
                    cpu_quant::cpu_quant_gemv(tl.q_attn_v.quant_type, tl.q_attn_v.cpu_data,
                        x_normed.data(), v_l, H, tl.q_attn_v.rows, tl.q_attn_v.row_stride_bytes);
            }

            // --- Biases (Qwen3) on rank-local slice ---
            if (layer.attn_q_bias.defined()) {
                const float* bq = layer.attn_q_bias.data_ptr<float>();
                const float* bk = layer.attn_k_bias.data_ptr<float>();
                const float* bv = layer.attn_v_bias.data_ptr<float>();
                // bq is sized [q_dim]; rank gets slice [q_dim_local_start, ...)
                int64_t q_off = tp_.rank * tp_.q_dim_local;
                int64_t kv_off = tp_.rank * tp_.kv_dim_local;
                for (int64_t j = 0; j < tp_.q_dim_local; ++j)  q_l[j] += bq[q_off + j];
                for (int64_t j = 0; j < tp_.kv_dim_local; ++j) k_l[j] += bk[kv_off + j];
                for (int64_t j = 0; j < tp_.kv_dim_local; ++j) v_l[j] += bv[kv_off + j];
            }

            // --- QK-norm (per-head) ---
            if (layer.attn_q_norm.defined()) {
                const float* qn_w = layer.attn_q_norm.data_ptr<float>();
                const float* kn_w = layer.attn_k_norm.data_ptr<float>();
                for (int64_t h = 0; h < n_heads_l; ++h)
                    cpu_quant::cpu_rmsnorm_inplace(q_l + h * head_dim, qn_w, eps, add_one, head_dim);
                for (int64_t h = 0; h < n_kv_l; ++h)
                    cpu_quant::cpu_rmsnorm_inplace(k_l + h * head_dim, kn_w, eps, add_one, head_dim);
            }

            // --- RoPE (uses current past_len position) ---
            {
                float rope_cos[256], rope_sin[256];
                at::native::hot::rope_precompute(rope_cos, rope_sin,
                    past_len, head_dim, config.rope_freq_base);
                at::native::hot::rope_apply_fused(q_l, k_l, rope_cos, rope_sin,
                    n_heads_l, n_kv_l, head_dim);
            }

            // --- KV cache append (local) ---
            float* k_cache = tp_.k_cache_local[i].mutable_data_ptr<float>();
            float* v_cache = tp_.v_cache_local[i].mutable_data_ptr<float>();
            std::memcpy(k_cache + past_len * tp_.kv_dim_local, k_l, tp_.kv_dim_local * sizeof(float));
            std::memcpy(v_cache + past_len * tp_.kv_dim_local, v_l, tp_.kv_dim_local * sizeof(float));

            // --- Attention over local heads (fill rank's slice of attn_full_buf, zero rest) ---
            // AllReduce path: zero full buffer first so other ranks' slices stay
            // 0 (for the replicated output-proj GEMV + AllReduce-sum trick).
            // AllGather path: no zero-pad — other slices are overwritten by
            // all_gather_inplace after attention.
            if (!use_gather) {
                std::fill(tp_.attn_full_buf.begin(), tp_.attn_full_buf.end(), 0.0f);
            }
            {
                int64_t total_seq = past_len + 1;
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                int64_t q_off = tp_.rank * tp_.q_dim_local;
                for (int64_t hl = 0; hl < n_heads_l; ++hl) {
                    int64_t global_h = tp_.head_start + hl;
                    int64_t kv_hl    = hl / heads_per_group;
                    (void)global_h;
                    const float* q_head = q_l + hl * head_dim;
                    float* out_head = tp_.attn_full_buf.data() + q_off + hl * head_dim;
                    std::vector<float> scores(total_seq);
                    // Round 4: AVX2 Q@K (head_dim=128 → 16 iters of 8-fp32)
                    for (int64_t t = 0; t < total_seq; ++t) {
                        const float* k_head = k_cache + t * tp_.kv_dim_local + kv_hl * head_dim;
#if defined(__AVX2__)
                        __m256 acc = _mm256_setzero_ps();
                        int64_t d = 0;
                        for (; d + 8 <= head_dim; d += 8) {
                            __m256 q = _mm256_loadu_ps(q_head + d);
                            __m256 k = _mm256_loadu_ps(k_head + d);
                            acc = _mm256_fmadd_ps(q, k, acc);
                        }
                        // horizontal sum 8 floats
                        __m128 lo = _mm256_castps256_ps128(acc);
                        __m128 hi = _mm256_extractf128_ps(acc, 1);
                        __m128 s = _mm_add_ps(lo, hi);
                        s = _mm_hadd_ps(s, s);
                        s = _mm_hadd_ps(s, s);
                        float dot = _mm_cvtss_f32(s);
                        for (; d < head_dim; ++d) dot += q_head[d] * k_head[d];
#else
                        float dot = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * k_head[d];
#endif
                        scores[t] = dot * scale;
                    }
                    float mx = scores[0];
                    for (int64_t t = 1; t < total_seq; ++t) if (scores[t] > mx) mx = scores[t];
                    float se = 0.0f;
                    for (int64_t t = 0; t < total_seq; ++t) {
                        scores[t] = std::exp(scores[t] - mx);
                        se += scores[t];
                    }
                    float inv = 1.0f / (se + 1e-10f);
                    for (int64_t t = 0; t < total_seq; ++t) scores[t] *= inv;
                    std::fill(out_head, out_head + head_dim, 0.0f);
                    // Round 4: AVX2 weighted-sum @V
                    for (int64_t t = 0; t < total_seq; ++t) {
                        const float* v_head = v_cache + t * tp_.kv_dim_local + kv_hl * head_dim;
                        float w = scores[t];
#if defined(__AVX2__)
                        __m256 wv = _mm256_set1_ps(w);
                        int64_t d = 0;
                        for (; d + 8 <= head_dim; d += 8) {
                            __m256 v = _mm256_loadu_ps(v_head + d);
                            __m256 o = _mm256_loadu_ps(out_head + d);
                            o = _mm256_fmadd_ps(wv, v, o);
                            _mm256_storeu_ps(out_head + d, o);
                        }
                        for (; d < head_dim; ++d) out_head[d] += w * v_head[d];
#else
                        for (int64_t d = 0; d < head_dim; ++d) out_head[d] += w * v_head[d];
#endif
                    }
                }
            }

            // --- Post-attention norm (Gemma3): in-place on full buf (norm is elementwise / scale) ---
            // Note: RMSNorm of zero-padded vector ≠ per-slice RMSNorm. Since this
            // path is Gemma3-specific and we target qwen3, we don't support
            // post_attention_norm in TP mode; guard and throw for clarity.
            if (layer.post_attention_norm.defined()) {
                throw std::runtime_error("TP: post_attention_norm unsupported (Gemma3 not yet wired)");
            }

            if (tp_sec_timers_.on) tp_sec_timers_.attn_ms += _tp_elapsed();
            // qkv+attention combined into attn_ms for now; we split by also
            // tracking QKV upstream — done below after the simple integration
            // proves the data. (qkv_ms initially 0, attn_ms holds combined time.)

            // --- Output projection ---
            float* h_buf = tp_.h_buf.data();
            if (use_gather) {
                // Option F: first gather local attention slices into full
                // attn_full_buf (all ranks get full q_dim vector), then every
                // rank computes its N-row slice of h from the full replicated
                // W_o, finally gather h slices.
                //
                // Split-collective pattern (Step 4): call `post` to deposit+
                // signal, issue weight prefetch loop over the upcoming GEMV's
                // cachelines while barriers resolve, then `wait` to consume.
                // Overlap is best-effort — if LCC ignores __builtin_prefetch
                // this is a no-op and correctness is unchanged.
                const int _node = c10::current_numa_node();
                const void* w_ao_full = nullptr;
                const uint8_t* w_ao_slice = nullptr;
                int64_t w_ao_stride = 0;
                if (use_quant_gemv_ && layer.q_attn_output.valid) {
                    w_ao_full = layer.q_attn_output.numa_replica.get(_node);
                    if (!w_ao_full) w_ao_full = layer.q_attn_output.cpu_data;
                    w_ao_stride = layer.q_attn_output.row_stride_bytes;
                    w_ao_slice = static_cast<const uint8_t*>(w_ao_full)
                               + h_row_start * w_ao_stride;
                }

                torch::distributed::all_gather_post(
                    tp_.attn_full_buf.data(), tp_.q_dim_local);
                // Overlap: warm the first ~2 KB of each of the first 4 rows
                // of the attn_output weight slice. Total: ~8 KB of prefetch
                // hints issued during the arrival barrier.
                if (w_ao_slice) {
                    int64_t pf_rows = std::min<int64_t>(H_local, 4);
                    int64_t pf_bytes = std::min<int64_t>(w_ao_stride, 2048);
                    for (int64_t r = 0; r < pf_rows; ++r) {
                        const uint8_t* row_ptr = w_ao_slice + r * w_ao_stride;
                        for (int64_t off = 0; off < pf_bytes; off += 64) {
                            __builtin_prefetch(row_ptr + off, 0, 2);
                        }
                    }
                }
                torch::distributed::all_gather_wait();
                if (tp_sec_timers_.on) tp_sec_timers_.allreduce_ao_ms += _tp_elapsed();

                // N-slice GEMV on full replicated attn_output weight.
                if (w_ao_slice) {
                    cpu_quant::cpu_quant_gemv(
                        layer.q_attn_output.quant_type, w_ao_slice,
                        tp_.attn_full_buf.data(),
                        h_buf + h_row_start,
                        q_dim, H_local, w_ao_stride,
                        /*numa=*/nullptr);
                } else if (layer.attn_output.defined()) {
                    const float* w = layer.attn_output.data_ptr<float>();
                    for (int64_t n = 0; n < H_local; ++n) {
                        int64_t global_n = h_row_start + n;
                        float dot = 0.0f;
                        for (int64_t k = 0; k < q_dim; ++k)
                            dot += tp_.attn_full_buf[k] * w[global_n * q_dim + k];
                        h_buf[h_row_start + n] = dot;
                    }
                }
                if (tp_sec_timers_.on) tp_sec_timers_.ao_ms += _tp_elapsed();

                // Gather N-slices into full h_buf. Overlap: local residual
                // + partial sum_sq on own slice — both operands already known
                // before gather completes. Writes to x_next at own offset so
                // we can skip redoing this part after wait().
                torch::distributed::all_gather_post(h_buf, H_local);
                {
                    // Touch the first few rows of gate/up weights for the
                    // FFN block to warm L2 while the arrival barrier resolves.
                    const void* w_g = tl.q_ffn_gate.valid ? tl.q_ffn_gate.cpu_data : nullptr;
                    const void* w_u = tl.q_ffn_up.valid   ? tl.q_ffn_up.cpu_data   : nullptr;
                    int64_t pf_stride = tl.q_ffn_gate.valid ? tl.q_ffn_gate.row_stride_bytes : 0;
                    if (w_g && w_u && pf_stride > 0) {
                        int64_t pf_rows = 4;
                        int64_t pf_bytes = std::min<int64_t>(pf_stride, 2048);
                        for (int64_t r = 0; r < pf_rows; ++r) {
                            const uint8_t* gp = static_cast<const uint8_t*>(w_g) + r * pf_stride;
                            const uint8_t* up_ = static_cast<const uint8_t*>(w_u) + r * pf_stride;
                            for (int64_t off = 0; off < pf_bytes; off += 64) {
                                __builtin_prefetch(gp + off, 0, 2);
                                __builtin_prefetch(up_ + off, 0, 2);
                            }
                        }
                    }
                }
                torch::distributed::all_gather_wait();
                if (tp_sec_timers_.on) tp_sec_timers_.allreduce_ao_ms += _tp_elapsed();
            } else {
                // Legacy path: K-slice GEMV + AllReduce-sum.
                if (tl.q_attn_output.q8_soa.valid) {
                    // Round 3 SoA path: input is rank's q_dim_local slice of attn_full_buf.
                    const float* input_slice = tp_.attn_full_buf.data() + (tp_.rank * tp_.q_dim_local);
                    cpu_quant::q8_soa4_quant_activation(input_slice,
                        tp_.q_dim_local,
                        tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                        &tp_.soa_scale_a);
                    cpu_quant::q8_soa4_gemv(&tl.q_attn_output.q8_soa,
                        tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                        tp_.soa_scale_a, h_buf);
                } else if (tl.q_attn_output.valid && tl.q_attn_output.cpu_data) {
                    const float* input_slice = tp_.attn_full_buf.data() + (tp_.rank * tp_.q_dim_local);
                    int64_t local_blocks = tl.attn_output_k_end - tl.attn_output_k_start;
                    cpu_quant::cpu_quant_gemv_k_slice(
                        tl.q_attn_output.quant_type,
                        tl.q_attn_output.cpu_data,
                        input_slice,
                        h_buf,
                        local_blocks,
                        tl.q_attn_output.rows,             // full H
                        tl.q_attn_output.row_stride_bytes);
                } else if (use_quant_gemv_ && layer.q_attn_output.valid && layer.q_attn_output.cpu_data) {
                    cpu_quant::cpu_quant_gemv(
                        layer.q_attn_output.quant_type, layer.q_attn_output.cpu_data,
                        tp_.attn_full_buf.data(), h_buf, q_dim,
                        layer.q_attn_output.rows, layer.q_attn_output.row_stride_bytes);
                } else if (layer.attn_output.defined()) {
                    const float* w = layer.attn_output.data_ptr<float>();
                    int64_t N_out = layer.attn_output.size(0);
                    for (int64_t n = 0; n < N_out; ++n) {
                        float dot = 0.0f;
                        for (int64_t k = 0; k < q_dim; ++k) dot += tp_.attn_full_buf[k] * w[n * q_dim + k];
                        h_buf[n] = dot;
                    }
                }

                if (tp_sec_timers_.on) tp_sec_timers_.ao_ms += _tp_elapsed();

                torch::distributed::all_reduce_inplace(h_buf, H);
                if (tp_sec_timers_.on) tp_sec_timers_.allreduce_ao_ms += _tp_elapsed();
            }

            // --- Residual add: x_next = x + h ---
            int next = 1 - cur;
            for (int64_t j = 0; j < H; ++j) tp_.x_buf[next][j] = tp_.x_buf[cur][j] + h_buf[j];
            cur = next;

            // --- FFN: fused RMSNorm(x) + gate + up GEMV ---
            // Profiler: gate_up = 73 ms/token (34% of total). Fused kernel
            // eliminates 1 full RMSNorm scan + 1 x_normed memcpy + streams
            // gate/up GEMV through single input tile — cuts ~20-30% on this
            // hottest section.
            float* x_cur = tp_.x_buf[cur].data();
            float* gate_l = tp_.gate_local_buf.data();
            float* up_l   = tp_.up_local_buf.data();
            bool use_soa_ffn = tl.q_ffn_gate.q8_soa.valid && tl.q_ffn_up.q8_soa.valid;
            bool can_fuse_ffn = tl.q_ffn_gate.valid && tl.q_ffn_up.valid &&
                cpu_quant::cpu_quant_gemv_supported(tl.q_ffn_gate.quant_type) &&
                cpu_quant::cpu_quant_gemv_supported(tl.q_ffn_up.quant_type);
            if (use_soa_ffn) {
                std::memcpy(x_normed.data(), x_cur, H * sizeof(float));
                cpu_quant::cpu_rmsnorm_inplace(x_normed.data(),
                    layer.ffn_norm.data_ptr<float>(), eps, add_one, H);
                cpu_quant::q8_soa4_quant_activation(x_normed.data(), H,
                    tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                    &tp_.soa_scale_a);
                // Round 4: fused gate+up — single parallel_for dispatch,
                // shared activation reads per N-row group.
                cpu_quant::q8_soa4_gemv_dual(
                    &tl.q_ffn_gate.q8_soa, &tl.q_ffn_up.q8_soa,
                    tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                    tp_.soa_scale_a, gate_l, up_l);
            } else if (can_fuse_ffn) {
                cpu_quant::cpu_fused_rmsnorm_gate_up_gemv(
                    x_cur, layer.ffn_norm.data_ptr<float>(), eps, add_one,
                    tl.q_ffn_gate.quant_type, tl.q_ffn_gate.cpu_data,
                    tl.q_ffn_up.quant_type,   tl.q_ffn_up.cpu_data,
                    gate_l, up_l,
                    H, tl.q_ffn_gate.rows, tl.q_ffn_up.rows,
                    tl.q_ffn_gate.row_stride_bytes, tl.q_ffn_up.row_stride_bytes,
                    /*numa_gate=*/nullptr, /*numa_up=*/nullptr);
            } else {
                std::memcpy(x_normed.data(), x_cur, H * sizeof(float));
                cpu_quant::cpu_rmsnorm_inplace(x_normed.data(),
                    layer.ffn_norm.data_ptr<float>(), eps, add_one, H);
                if (tl.q_ffn_gate.valid)
                    cpu_quant::cpu_quant_gemv(tl.q_ffn_gate.quant_type, tl.q_ffn_gate.cpu_data,
                        x_normed.data(), gate_l, H, tl.q_ffn_gate.rows, tl.q_ffn_gate.row_stride_bytes);
                if (tl.q_ffn_up.valid)
                    cpu_quant::cpu_quant_gemv(tl.q_ffn_up.quant_type, tl.q_ffn_up.cpu_data,
                        x_normed.data(), up_l, H, tl.q_ffn_up.rows, tl.q_ffn_up.row_stride_bytes);
            }
            if (tp_sec_timers_.on) tp_sec_timers_.gateup_ms += _tp_elapsed();

            // --- SiLU(gate) * up + ffn_down ---
            if (use_gather) {
                // Option F: write local silu into silu_full_buf at rank's
                // offset, all-gather to get full inter-vector on every rank,
                // then compute N-slice of ffn_down.
                for (int64_t j = 0; j < tp_.inter_local; ++j) {
                    float g = gate_l[j];
                    tp_.silu_full_buf[tp_.inter_offset + j] = (g / (1.0f + std::exp(-g))) * up_l[j];
                }
                if (tp_sec_timers_.on) tp_sec_timers_.gateup_ms += _tp_elapsed();

                // Pre-compute ffn_down N-slice pointer so we can prefetch
                // during the arrival barrier.
                const int _node_fd = c10::current_numa_node();
                const void* w_fd_full = nullptr;
                const uint8_t* w_fd_slice = nullptr;
                int64_t w_fd_stride = 0;
                if (use_quant_gemv_ && layer.q_ffn_down.valid) {
                    w_fd_full = layer.q_ffn_down.numa_replica.get(_node_fd);
                    if (!w_fd_full) w_fd_full = layer.q_ffn_down.cpu_data;
                    w_fd_stride = layer.q_ffn_down.row_stride_bytes;
                    w_fd_slice = static_cast<const uint8_t*>(w_fd_full)
                               + h_row_start * w_fd_stride;
                }

                torch::distributed::all_gather_post(
                    tp_.silu_full_buf.data(), tp_.inter_local);
                if (w_fd_slice) {
                    int64_t pf_rows = std::min<int64_t>(H_local, 4);
                    int64_t pf_bytes = std::min<int64_t>(w_fd_stride, 2048);
                    for (int64_t r = 0; r < pf_rows; ++r) {
                        const uint8_t* row_ptr = w_fd_slice + r * w_fd_stride;
                        for (int64_t off = 0; off < pf_bytes; off += 64) {
                            __builtin_prefetch(row_ptr + off, 0, 2);
                        }
                    }
                }
                torch::distributed::all_gather_wait();
                if (tp_sec_timers_.on) tp_sec_timers_.allreduce_fdown_ms += _tp_elapsed();

                if (w_fd_slice) {
                    cpu_quant::cpu_quant_gemv(
                        layer.q_ffn_down.quant_type, w_fd_slice,
                        tp_.silu_full_buf.data(),
                        h_buf + h_row_start,
                        inter, H_local, w_fd_stride,
                        /*numa=*/nullptr);
                } else if (layer.ffn_down.defined()) {
                    const float* w = layer.ffn_down.data_ptr<float>();
                    for (int64_t n = 0; n < H_local; ++n) {
                        int64_t global_n = h_row_start + n;
                        float dot = 0.0f;
                        for (int64_t k = 0; k < inter; ++k)
                            dot += tp_.silu_full_buf[k] * w[global_n * inter + k];
                        h_buf[h_row_start + n] = dot;
                    }
                }

                if (layer.post_ffw_norm.defined()) {
                    throw std::runtime_error("TP: post_ffw_norm unsupported (Gemma3 not yet wired)");
                }
                if (tp_sec_timers_.on) tp_sec_timers_.fdown_ms += _tp_elapsed();

                // Final gather of layer's output h. Overlap: prefetch the
                // NEXT layer's attn_q/k/v row-sliced weights (tl entries are
                // local slices, no NUMA resolution needed).
                torch::distributed::all_gather_post(h_buf, H_local);
                {
                    int64_t next_i = i + 1;
                    if (next_i < config.num_layers) {
                        const auto& ntl = tp_.layers[next_i];
                        const void* w_nq = ntl.q_attn_q.valid ? ntl.q_attn_q.cpu_data : nullptr;
                        int64_t nq_stride = ntl.q_attn_q.valid ? ntl.q_attn_q.row_stride_bytes : 0;
                        if (w_nq && nq_stride > 0) {
                            int64_t pf_rows = 4;
                            int64_t pf_bytes = std::min<int64_t>(nq_stride, 2048);
                            for (int64_t r = 0; r < pf_rows; ++r) {
                                const uint8_t* row_ptr = static_cast<const uint8_t*>(w_nq)
                                                       + r * nq_stride;
                                for (int64_t off = 0; off < pf_bytes; off += 64) {
                                    __builtin_prefetch(row_ptr + off, 0, 2);
                                }
                            }
                        }
                    }
                }
                torch::distributed::all_gather_wait();
                if (tp_sec_timers_.on) tp_sec_timers_.allreduce_fdown_ms += _tp_elapsed();
            } else {
                // Legacy path: K-slice ffn_down + AllReduce-sum.
                if (tl.q_ffn_down.q8_soa.valid) {
                    // Round 4 Item 2: fused SiLU + Q8 SoA4 quant in
                    // q8_soa_repack.h (out-of-line so LCC can optimize it).
                    // 2 passes vs old 5 + per-call vector<uint8_t>(K) alloc.
                    // Lossless — identical FP/INT arithmetic.
                    std::vector<float>& silu_local = tp_.silu_scratch_buf;
                    cpu_quant::q8_soa4_silu_quant_activation_fused(
                        gate_l, up_l,
                        tp_.inter_local, silu_local.data(),
                        tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                        &tp_.soa_scale_a);
                    cpu_quant::q8_soa4_gemv(&tl.q_ffn_down.q8_soa,
                        tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                        tp_.soa_scale_a, h_buf);
                } else if (tl.q_ffn_down.valid && tl.q_ffn_down.cpu_data) {
                    std::vector<float>& silu_local = tp_.silu_scratch_buf;
                    for (int64_t j = 0; j < tp_.inter_local; ++j) {
                        float g = gate_l[j];
                        silu_local[j] = (g / (1.0f + std::exp(-g))) * up_l[j];
                    }
                    int64_t local_blocks = tl.ffn_down_k_end - tl.ffn_down_k_start;
                    cpu_quant::cpu_quant_gemv_k_slice(
                        tl.q_ffn_down.quant_type,
                        tl.q_ffn_down.cpu_data,
                        silu_local.data(),
                        h_buf,
                        local_blocks,
                        tl.q_ffn_down.rows,                   // full H
                        tl.q_ffn_down.row_stride_bytes);
                } else if (use_quant_gemv_ && layer.q_ffn_down.valid && layer.q_ffn_down.cpu_data) {
                    std::fill(tp_.silu_full_buf.begin(), tp_.silu_full_buf.end(), 0.0f);
                    for (int64_t j = 0; j < tp_.inter_local; ++j) {
                        float g = gate_l[j];
                        tp_.silu_full_buf[tp_.inter_offset + j] = (g / (1.0f + std::exp(-g))) * up_l[j];
                    }
                    cpu_quant::cpu_quant_gemv(
                        layer.q_ffn_down.quant_type, layer.q_ffn_down.cpu_data,
                        tp_.silu_full_buf.data(), h_buf, inter,
                        layer.q_ffn_down.rows, layer.q_ffn_down.row_stride_bytes);
                }

                if (layer.post_ffw_norm.defined()) {
                    throw std::runtime_error("TP: post_ffw_norm unsupported (Gemma3 not yet wired)");
                }
                if (tp_sec_timers_.on) tp_sec_timers_.fdown_ms += _tp_elapsed();

                torch::distributed::all_reduce_inplace(h_buf, H);
                if (tp_sec_timers_.on) tp_sec_timers_.allreduce_fdown_ms += _tp_elapsed();
            }

            // --- Residual add ---
            next = 1 - cur;
            for (int64_t j = 0; j < H; ++j) tp_.x_buf[next][j] = tp_.x_buf[cur][j] + h_buf[j];
            cur = next;
        }  // end layer loop
        if (tp_sec_timers_.on) _tp_elapsed();  // reset — tail measured below

        // 3. Final RMSNorm (in-place, replicated)
        cpu_quant::cpu_rmsnorm_inplace(tp_.x_buf[cur].data(),
            output_norm.data_ptr<float>(), eps, add_one, H);

        // 4. Output projection — split across ranks and AllReduce.
        // Each rank computes a disjoint row-range of the vocab; zero-pads the
        // rest; AllReduce-SUM concatenates into full logits_buf on all ranks.
        // Per-rank work: 1/nprocs of vocab rows → ~4× speedup on output_proj.
        // Needs kShmSlotSize >= V*4 bytes — raised to 1 MB to fit vocab=152k.
        int64_t V = config.vocab_size;
        // Round 3 Option D: prefer K-slice path. Each rank reads only 1/N
        // of the output_weight bytes (~44 MB vs 175 MB of N-row split),
        // computes partial logits over FULL vocab, AllReduce-sum to combine.
        // Eliminates the replicated-K-read that was costing ~131 MB/token
        // aggregate bandwidth. Per Round 2 agent_9 §4.2: +14% alone (~+0.7 tok/s).
        if (tp_.q_output_weight_k_slice.q8_soa.valid) {
            // Round 3 SoA path: each rank's K-slice of output_weight on SoA4.
            int64_t elems_per_block = (tp_.q_output_weight_k_slice.quant_type == 8) ? 32 : 256;
            const float* x_local =
                tp_.x_buf[cur].data() + tp_.output_weight_k_start * elems_per_block;
            int64_t local_K = tp_.q_output_weight_k_slice.q8_soa.K;
            cpu_quant::q8_soa4_quant_activation(x_local,
                local_K,
                tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                &tp_.soa_scale_a);
            cpu_quant::q8_soa4_gemv(&tp_.q_output_weight_k_slice.q8_soa,
                tp_.soa_act_b16.data(), tp_.soa_sum_a.data(),
                tp_.soa_scale_a, tp_.logits_buf.data());
            torch::distributed::all_reduce_inplace(tp_.logits_buf.data(), V);
        } else if (use_quant_gemv_ && tp_.q_output_weight_k_slice.valid &&
            tp_.q_output_weight_k_slice.cpu_data) {
            // K-slice path: rank reads its 1/N of input dim K (=H)
            int64_t local_blocks = tp_.output_weight_k_end - tp_.output_weight_k_start;
            // Q4_K is 256 elements/block; output_weight uses Q4_K typically.
            int64_t elems_per_block = (tp_.q_output_weight_k_slice.quant_type == 8) ? 32 : 256;
            const float* x_local =
                tp_.x_buf[cur].data() + tp_.output_weight_k_start * elems_per_block;
            cpu_quant::cpu_quant_gemv_k_slice(
                tp_.q_output_weight_k_slice.quant_type,
                tp_.q_output_weight_k_slice.cpu_data,
                x_local,
                tp_.logits_buf.data(),
                local_blocks,
                tp_.q_output_weight_k_slice.rows,    // = V (full vocab)
                tp_.q_output_weight_k_slice.row_stride_bytes);
            torch::distributed::all_reduce_inplace(tp_.logits_buf.data(), V);
        } else if (use_quant_gemv_ && q_output_weight.valid && q_output_weight.cpu_data) {
            // Legacy fallback: N-row split (each rank reads full K of its 1/N rows).
            // This path runs when K-slice setup failed (e.g. PT_TP_GATHER=1 mode
            // skipped slicing) or for quant types that don't support k-slice.
            int64_t rank_rows = V / tp_.nprocs;
            int64_t row_start = tp_.rank * rank_rows;
            int64_t row_end   = (tp_.rank == tp_.nprocs - 1) ? V : row_start + rank_rows;
            int64_t local_rows = row_end - row_start;

            std::memset(tp_.logits_buf.data(), 0, V * sizeof(float));

            const uint8_t* w_slice = static_cast<const uint8_t*>(q_output_weight.cpu_data)
                                   + row_start * q_output_weight.row_stride_bytes;
            cpu_quant::cpu_quant_gemv(
                q_output_weight.quant_type, w_slice,
                tp_.x_buf[cur].data(), tp_.logits_buf.data() + row_start,
                H, local_rows, q_output_weight.row_stride_bytes);

            torch::distributed::all_reduce_inplace(tp_.logits_buf.data(), V);
        } else if (output_weight.defined()) {
            // FP32 fallback — no split, scalar main-thread path (slow but rare
            // since tied q_output_weight is the common case for qwen3:4b etc.).
            const float* w = output_weight.data_ptr<float>();
            for (int64_t n = 0; n < V; ++n) {
                float dot = 0.0f;
                for (int64_t k = 0; k < H; ++k) dot += tp_.x_buf[cur][k] * w[n * H + k];
                tp_.logits_buf[n] = dot;
            }
        }
        if (tp_sec_timers_.on) tp_sec_timers_.output_proj_ms += _tp_elapsed();

        // Wrap into tensor (copy out)
        Tensor logits = at::empty({1, config.vocab_size});
        std::memcpy(logits.mutable_data_ptr<float>(), tp_.logits_buf.data(),
                    config.vocab_size * sizeof(float));

        tp_.kv_seq_len += 1;
        if (tp_sec_timers_.on) {
            tp_sec_timers_.tail_ms += _tp_elapsed();
            tp_sec_timers_.tokens++;
        }
        return logits;
    }

    // ========================================================================
    // Tensor-parallel generate: drives prefill + decode via forward_decode_cpu_tp.
    // Only rank 0 prints tokens. Returns the decoded string on rank 0; other
    // ranks return "" after finishing the same number of forward passes
    // (to keep collectives in lockstep).
    // ========================================================================
    std::string generate_tp(const std::string& prompt, int max_tokens = 128,
                            float temperature = 0.0f) {
        if (!tp_.enabled) {
            throw std::runtime_error("generate_tp: call init_tensor_parallel() first");
        }
        // Reset KV cache
        int64_t max_total_seq = static_cast<int64_t>(max_tokens) + 2048;
        if (max_total_seq > config.context_length) max_total_seq = config.context_length;
        if (tp_.kv_max_seq < max_total_seq) {
            tp_allocate_kv_cache(max_total_seq);
        } else {
            tp_.kv_seq_len = 0;
        }

        // Tokenize prompt
        auto input_tokens = tokenizer.encode(prompt, true);
        if (tp_.rank == 0) {
            std::cout << "[Generate-TP] Prompt tokens: " << input_tokens.size() << std::endl;
        }

        // Prefill: run every prompt token through forward_decode_cpu_tp one at a time
        Tensor logits;
        for (int64_t t : input_tokens) {
            logits = forward_decode_cpu_tp(t);
        }

        // Generation loop
        std::vector<int32_t> generated;
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < max_tokens; ++step) {
            // Greedy argmax on identical replicated logits (all ranks agree)
            int32_t best = 0;
            const float* lbuf = logits.data_ptr<float>();
            float best_val = lbuf[0];
            for (int64_t j = 1; j < config.vocab_size; ++j) {
                if (lbuf[j] > best_val) { best_val = lbuf[j]; best = static_cast<int32_t>(j); }
            }
            (void)temperature;  // only greedy supported in TP mode

            if (best == tokenizer.eos_id) break;
            if (is_stop_token(best)) break;

            generated.push_back(best);
            if (tp_.rank == 0) {
                std::cout << tokenizer.decode_token(best);
            }
            logits = forward_decode_cpu_tp(static_cast<int64_t>(best));
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tok_per_s = generated.size() / (ms / 1000.0);

        if (tp_.rank == 0) {
            std::cout << "\n\n[Generate-TP] " << generated.size() << " tokens in "
                      << (ms / 1000.0) << "s (" << tok_per_s << " tok/s)" << std::endl;
            tp_sec_timers_.dump(tp_.rank);
            return tokenizer.decode(generated, true);
        }
        return "";
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
                } else if (qw.is_f16()) {
                    at::cuda::launch_fp16_gemv(qw.gpu_data, x_ptr, y_ptr, K, N, nullptr);
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
