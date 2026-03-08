#pragma once

#include "torch/io/gguf_loader.h"
#include "torch/io/gguf_dequant.h"
#include "torch/io/ollama.h"
#include "torch/io/tokenizer.h"
#include "aten/src/ATen/ATen.h"
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
};

// ============================================================================
// KV Cache for inference
// ============================================================================

struct KVCache {
    std::vector<Tensor> key_cache;   // per layer: [seq_so_far, num_kv_heads, head_dim]
    std::vector<Tensor> value_cache; // per layer: [seq_so_far, num_kv_heads, head_dim]
    int64_t seq_len = 0;

    void reset() {
        key_cache.clear();
        value_cache.clear();
        seq_len = 0;
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
    Tensor token_embedding;  // [vocab_size, hidden]
    Tensor output_norm;      // [hidden]
    Tensor output_weight;    // [vocab_size, hidden]
    std::vector<TransformerLayer> layers;

    // KV cache
    KVCache kv_cache;

    // Device
    bool use_cuda_ = false;

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
    // Move all weights to CUDA
    // Pre-transposes 2D weights for faster mm (no runtime transpose needed)
    // ========================================================================

    void to_cuda() {
#ifdef PT_USE_CUDA
        auto t_start = std::chrono::high_resolution_clock::now();
        std::cout << "[Model] Moving weights to CUDA..." << std::endl;

        auto move = [](Tensor& t) {
            if (t.defined() && t.is_cpu()) {
                t = at::to_cuda(t);
            }
        };
        // Pre-transpose and move 2D weights (for mm without runtime transpose)
        auto move_t = [](Tensor& t) {
            if (t.defined() && t.dim() == 2 && t.is_cpu()) {
                // Transpose: [out, in] → [in, out] and make contiguous
                Tensor tr = t.t().contiguous();
                t = at::to_cuda(tr);
            } else if (t.defined() && t.is_cpu()) {
                t = at::to_cuda(t);
            }
        };

        // Keep token_embedding on CPU — embedding lookup is CPU-based
        // (copying vocab*hidden from GPU→CPU every token would be worse)

        // Output weight: create transposed copy on GPU
        // Must handle tied embeddings (output_weight == token_embedding)
        {
            Tensor out_cpu = output_weight.is_cpu() ? output_weight : to_cpu_tensor(output_weight);
            Tensor out_tr = out_cpu.t().contiguous();
            output_weight = at::to_cuda(out_tr);
        }
        move(output_norm);

        for (auto& layer : layers) {
            move(layer.attn_norm);
            move_t(layer.attn_q);
            move_t(layer.attn_k);
            move_t(layer.attn_v);
            move_t(layer.attn_output);
            move(layer.ffn_norm);
            move_t(layer.ffn_gate);
            move_t(layer.ffn_up);
            move_t(layer.ffn_down);

            // 1D weights/biases
            move(layer.attn_q_norm);
            move(layer.attn_k_norm);
            move(layer.post_attention_norm);
            move(layer.post_ffw_norm);
            move(layer.attn_q_bias);
            move(layer.attn_k_bias);
            move(layer.attn_v_bias);
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
    // Load from GGUF file path
    // ========================================================================

    void load(const std::string& gguf_path) {
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

        // Load all weights
        load_weights(reader);

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
        Tensor x = embedding_lookup(tokens);  // [seq_len, hidden]



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
        x = rms_norm(x, output_norm, config.rms_norm_eps);

        // 4. Output projection → logits
        // For single-token generation, only compute logits for last position
        Tensor x_last = x;
        if (use_cache && seq_len > 1) {
            // During prefill, we only need last position logits for generation
            // But we compute all for correctness (KV cache needs all positions processed)
        }
        Tensor logits = matmul(x_last, output_weight, true);

        if (use_cache) {
            kv_cache.seq_len += seq_len;
        }

        return logits;
    }



    // ========================================================================
    // Chat template formatting
    // ========================================================================

    std::string apply_chat_template(const std::string& prompt) const {
        if (config.architecture == "qwen3") {
            // Qwen3 chat format
            return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
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
        // Reset KV cache
        kv_cache.reset();
        kv_cache.key_cache.resize(config.num_layers);
        kv_cache.value_cache.resize(config.num_layers);

        // Encode prompt
        auto input_tokens = tokenizer.encode(prompt, true);
        std::cout << "[Generate] Prompt tokens: " << input_tokens.size() << std::endl;

        // Process prompt (prefill)
        std::vector<int64_t> tokens_i64(input_tokens.begin(), input_tokens.end());
        Tensor logits = forward(tokens_i64, true);

        // Get last token logits
        std::vector<int32_t> generated;
        auto t_start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < max_tokens; ++step) {
            // Sample next token from last position logits
            int64_t last_pos = logits.size(0) - 1;
            Tensor last_logits = get_row(logits, last_pos);  // [vocab_size]

            // Apply repetition penalty
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

            int32_t next_token = sample_token(last_logits, temperature, top_k, top_p);

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
            std::vector<int64_t> next_input = {static_cast<int64_t>(next_token)};
            logits = forward(next_input, true);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tokens_per_sec = generated.size() / (ms / 1000.0);

        std::cout << "\n\n[Generate] " << generated.size() << " tokens in "
                  << (ms / 1000.0) << "s (" << tokens_per_sec << " tok/s)" << std::endl;

        std::string result = tokenizer.decode(generated, true);

        // Strip thinking blocks: <think>...</think>
        for (;;) {
            size_t s = result.find("<think>");
            if (s == std::string::npos) break;
            size_t e = result.find("</think>", s);
            if (e != std::string::npos) {
                result.erase(s, e + 8 - s);
            } else {
                // No closing tag — just remove the <think> tag itself
                result.erase(s, 7);
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

private:
    // Check if token is a stop/EOS token for the model
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

    Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_b = false) {
#ifdef PT_USE_CUDA
        if (use_cuda_) {
            // On GPU, weights are pre-transposed during to_cuda().
            // So when transpose_b=true, B is already [K, N] (not [N, K]).
            // Just do A @ B directly.
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

        // Basic GEMM with tiling for cache efficiency
        constexpr int64_t TILE = 32;

        if (transpose_b) {
            // C[m,n] = sum_k A[m,k] * B[n,k]
            #pragma omp parallel for schedule(dynamic) if(M > 4)
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n_tile = 0; n_tile < N; n_tile += TILE) {
                    int64_t n_end = (std::min)(n_tile + TILE, N);
                    for (int64_t n = n_tile; n < n_end; ++n) {
                        float sum = 0.0f;
                        const float* a_row = a_data + m * K;
                        const float* b_row = b_data + n * K;
                        int64_t k = 0;
                        // Unrolled accumulation
                        for (; k + 3 < K; k += 4) {
                            sum += a_row[k] * b_row[k] + a_row[k+1] * b_row[k+1]
                                 + a_row[k+2] * b_row[k+2] + a_row[k+3] * b_row[k+3];
                        }
                        for (; k < K; ++k) {
                            sum += a_row[k] * b_row[k];
                        }
                        c_data[m * N + n] = sum;
                    }
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

        // 1. Attention pre-norm → Self-attention
        Tensor normed = rms_norm(x, layer.attn_norm, config.rms_norm_eps);
        Tensor attn_out = self_attention(normed, layer, layer_idx, past_len, use_cache);

        // 2. Post-attention norm (Gemma3)
        if (layer.post_attention_norm.defined()) {
            attn_out = rms_norm(attn_out, layer.post_attention_norm, config.rms_norm_eps);
        }

        // 3. Residual
        Tensor h = add_tensors(x, attn_out);

        // 4. FFN pre-norm → SwiGLU FFN
        Tensor normed2 = rms_norm(h, layer.ffn_norm, config.rms_norm_eps);
        Tensor ffn_out = swiglu_ffn(normed2, layer);

        // 5. Post-FFN norm (Gemma3)
        if (layer.post_ffw_norm.defined()) {
            ffn_out = rms_norm(ffn_out, layer.post_ffw_norm, config.rms_norm_eps);
        }

        // 6. Residual
        return add_tensors(h, ffn_out);
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

        // Q, K, V projections: x @ W^T (GPU GEMM when CUDA)
        Tensor q = matmul(x, layer.attn_q, true);   // [seq, q_dim]
        Tensor k = matmul(x, layer.attn_k, true);   // [seq, kv_dim]
        Tensor v = matmul(x, layer.attn_v, true);   // [seq, kv_dim]

        // Add biases if present
        if (layer.attn_q_bias.defined()) {
            q = add_tensors(q, layer.attn_q_bias);
            k = add_tensors(k, layer.attn_k_bias);
            v = add_tensors(v, layer.attn_v_bias);
        }

        // QK-norm (operates on current device — GPU or CPU)
        if (layer.attn_q_norm.defined()) {
            apply_qk_norm_inplace(q, layer.attn_q_norm, n_heads, head_dim);
            apply_qk_norm_inplace(k, layer.attn_k_norm, n_kv_heads, head_dim);
        }

        // RoPE (operates on current device — GPU or CPU)
        apply_rope_inplace(q, n_heads, head_dim, past_len);
        apply_rope_inplace(k, n_kv_heads, head_dim, past_len);

        // KV cache (on same device as K,V)
        if (use_cache) {
            if (kv_cache.key_cache[layer_idx].defined()) {
                k = concat_tensors(kv_cache.key_cache[layer_idx], k);
                v = concat_tensors(kv_cache.value_cache[layer_idx], v);
            }
            kv_cache.key_cache[layer_idx] = k;
            kv_cache.value_cache[layer_idx] = v;
        }

        int64_t total_seq = k.size(0);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#ifdef PT_USE_CUDA
        if (use_cuda_ && q.is_cuda()) {
            // Full GPU attention using CUDA kernel
            auto output = at::empty_cuda({seq_len, q_dim}, q.dtype(), q.device().index());
            at::cuda::launch_causal_attention(
                q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                output.mutable_data_ptr<float>(),
                static_cast<int>(seq_len), static_cast<int>(total_seq),
                static_cast<int>(n_heads), static_cast<int>(n_kv_heads),
                static_cast<int>(head_dim),
                static_cast<int>(past_len), scale, nullptr);
            // Output projection
            return matmul(output, layer.attn_output, true);
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
        // gate = x @ gate_weight^T  [seq, intermediate]
        Tensor gate = matmul(x, layer.ffn_gate, true);
        // up = x @ up_weight^T  [seq, intermediate]
        Tensor up = matmul(x, layer.ffn_up, true);

        // SiLU(gate) * up
#ifdef PT_USE_CUDA
        if (use_cuda_ && gate.is_cuda()) {
            Tensor activated = at::cuda_ops::silu(gate);
            Tensor hidden = at::cuda_ops::mul(activated, up);
            return matmul(hidden, layer.ffn_down, true);
        }
#endif
        {
            int64_t n = gate.numel();
            float* gate_data = gate.mutable_data_ptr<float>();
            const float* up_data = up.data_ptr<float>();
            for (int64_t i = 0; i < n; ++i) {
                float g = gate_data[i];
                float silu = g / (1.0f + std::exp(-g));
                gate_data[i] = silu * up_data[i];
            }
            return matmul(gate, layer.ffn_down, true);
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
