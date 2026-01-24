#pragma once

// ============================================================================
// PIR 270M Model
// ============================================================================
// ~270M parameters with O(T) linear complexity
// Architecture:
// - 22 transformer blocks
// - 768 hidden dimension
// - 3 PIR layers per block
// - SwiGLU FFN
// - RMSNorm + RoPE

#include "torch/nn/module.h"
#include "torch/nn/modules/pir.h"
#include "torch/nn/modules/sparse.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include <random>
#include <set>

namespace torch {
namespace nn {

// ============================================================================
// PIR270MConfig - Configuration
// ============================================================================

struct PIR270MConfig {
    int64_t vocab_size = 50257;       // GPT-2 vocabulary
    int64_t n_embd = 768;             // Hidden dimension
    int64_t n_layers = 22;            // Transformer blocks
    int64_t n_pir_layers = 3;         // PIR layers per block
    int64_t block_size = 2048;        // Context length
    double ffn_mult = 4.0;            // FFN expansion
    double dropout = 0.0;             // Dropout rate
    bool tie_weights = true;          // Tie embedding weights
    double rope_base = 10000.0;       // RoPE base frequency

    // Computed FFN hidden size (SwiGLU uses 2/3)
    int64_t ffn_hidden() const {
        int64_t h = static_cast<int64_t>(n_embd * ffn_mult * 2 / 3);
        return (h + 63) / 64 * 64;  // Round to 64
    }
};

// ============================================================================
// PIR270M Model
// ============================================================================

class PIR270M : public Module {
public:
    explicit PIR270M(const PIR270MConfig& config = PIR270MConfig())
        : Module("PIR270M")
        , config_(config)
    {
        // Token embedding
        tok_emb_ = std::make_shared<Embedding>(config.vocab_size, config.n_embd);
        register_module("tok_emb", tok_emb_);

        // RoPE
        rope_ = std::make_shared<RotaryEmbedding>(
            config.n_embd / 2,
            config.block_size,
            config.rope_base
        );
        register_module("rope", rope_);

        // Transformer blocks
        for (int64_t i = 0; i < config.n_layers; ++i) {
            auto block = std::make_shared<PIRTransformerBlock>(
                config.n_embd,
                config.ffn_hidden(),
                config.n_pir_layers,
                config.dropout
            );
            blocks_.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }

        // Output normalization
        norm_out_ = std::make_shared<RMSNorm>(config.n_embd);
        register_module("norm_out", norm_out_);

        // Language model head
        lm_head_ = std::make_shared<Linear>(config.n_embd, config.vocab_size, false);
        register_module("lm_head", lm_head_);

        // Weight tying
        if (config.tie_weights) {
            // Share weights between embedding and lm_head
            auto* emb_weight = tok_emb_->get_parameter("weight");
            auto* lm_weight = lm_head_->get_parameter("weight");
            if (emb_weight && lm_weight) {
                // Copy embedding weights to lm_head
                Tensor emb_w = emb_weight->data();
                lm_weight->set_data(emb_w);
            }
        }

        // Initialize weights
        init_weights();

        // Count parameters
        int64_t n_params = count_params();
        std::cout << "PIR 270M initialized: " << n_params / 1e6 << "M parameters" << std::endl;
    }

    void init_weights() {
        // Initialize token embeddings
        auto* emb_weight = tok_emb_->get_parameter("weight");
        if (emb_weight) {
            Tensor w = emb_weight->data();
            float* data = w.mutable_data_ptr<float>();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (int64_t i = 0; i < w.numel(); ++i) {
                data[i] = dist(gen);
            }
        }
    }

    int64_t count_params() const {
        int64_t count = 0;
        for (auto* param : const_cast<PIR270M*>(this)->parameters()) {
            count += param->data().numel();
        }
        return count;
    }

    // Forward pass
    // idx: [batch, seq_len] token indices (as float)
    // targets: [batch, seq_len] target indices for loss (optional)
    // Returns: logits [batch, seq_len, vocab_size], loss (if targets provided)
    std::pair<Tensor, Tensor> forward_with_loss(
        const Tensor& idx,
        const Tensor& targets = Tensor()
    ) {
        auto sizes = idx.sizes().vec();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        if (T > config_.block_size) {
            throw std::runtime_error(
                "Sequence length " + std::to_string(T) +
                " exceeds block_size " + std::to_string(config_.block_size)
            );
        }

        // Token embeddings: [B, T] -> [B, T, n_embd]
        Tensor x = embed_tokens(idx);

        // Apply RoPE
        x = rope_->apply(x, T, true);

        // Transformer blocks
        for (size_t i = 0; i < blocks_.size(); ++i) {
            x = blocks_[i]->forward(x);
        }

        // Output normalization
        x = norm_out_->forward(x);

        // Language model head: [B, T, n_embd] -> [B, T, vocab_size]
        Tensor logits = project_to_vocab(x);

        // Compute loss if targets provided
        Tensor loss;
        if (targets.defined()) {
            loss = cross_entropy_loss(logits, targets);
        }

        return {logits, loss};
    }

    // Single tensor forward (for Module interface)
    Tensor forward(const Tensor& idx) override {
        return forward_with_loss(idx).first;
    }

    // Generate tokens autoregressively
    std::vector<int64_t> generate(
        const std::vector<int64_t>& prompt,
        int64_t max_new_tokens,
        double temperature = 0.85,
        int64_t top_k = 40,
        double top_p = 0.92,
        int64_t eos_token_id = 50256,
        double repetition_penalty = 1.15
    ) {
        eval();  // Set to eval mode

        // CRITICAL: Disable autograd for inference to prevent memory leak!
        torch::autograd::NoGradGuard no_grad;

        std::vector<int64_t> tokens = prompt;
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int64_t i = 0; i < max_new_tokens; ++i) {
            // Crop to block size
            int64_t context_len = std::min(
                static_cast<int64_t>(tokens.size()),
                config_.block_size
            );
            int64_t start_idx = tokens.size() > static_cast<size_t>(config_.block_size) ?
                tokens.size() - config_.block_size : 0;

            // Prepare input: [1, context_len]
            Tensor input = at::empty({1, context_len});
            float* input_data = input.mutable_data_ptr<float>();
            for (int64_t j = 0; j < context_len; ++j) {
                input_data[j] = static_cast<float>(tokens[start_idx + j]);
            }

            // Move input to same device as model
#ifdef PT_USE_CUDA
            if (tok_emb_->get_parameter("weight")->data().is_cuda()) {
                input = at::to_cuda(input);
            }
#endif

            // Forward pass (no gradient needed for inference)
            input.set_requires_grad(false);
            auto [logits, _] = forward_with_loss(input);

            // Copy logits to CPU for reading
            Tensor logits_cpu = logits;
#ifdef PT_USE_CUDA
            if (logits.is_cuda()) {
                logits_cpu = at::to_cpu(logits);
            }
#endif

            // Get last position logits: [vocab_size]
            std::vector<float> last_logits(config_.vocab_size);
            const float* logits_data = logits_cpu.data_ptr<float>();
            int64_t last_pos = context_len - 1;
            for (int64_t v = 0; v < config_.vocab_size; ++v) {
                last_logits[v] = logits_data[(last_pos) * config_.vocab_size + v];
            }

            // Apply repetition penalty
            if (repetition_penalty != 1.0) {
                std::set<int64_t> seen(tokens.begin(), tokens.end());
                for (int64_t token_id : seen) {
                    if (token_id < config_.vocab_size) {
                        if (last_logits[token_id] > 0) {
                            last_logits[token_id] /= static_cast<float>(repetition_penalty);
                        } else {
                            last_logits[token_id] *= static_cast<float>(repetition_penalty);
                        }
                    }
                }
            }

            // Apply temperature
            for (auto& logit : last_logits) {
                logit /= static_cast<float>(temperature);
            }

            // Top-k filtering
            if (top_k > 0 && top_k < config_.vocab_size) {
                std::vector<float> sorted_logits = last_logits;
                std::partial_sort(
                    sorted_logits.begin(),
                    sorted_logits.begin() + top_k,
                    sorted_logits.end(),
                    std::greater<float>()
                );
                float threshold = sorted_logits[top_k - 1];
                for (auto& logit : last_logits) {
                    if (logit < threshold) {
                        logit = -std::numeric_limits<float>::infinity();
                    }
                }
            }

            // Convert to probabilities (softmax)
            float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
            std::vector<float> probs(config_.vocab_size);
            float sum = 0.0f;
            for (int64_t v = 0; v < config_.vocab_size; ++v) {
                probs[v] = std::exp(last_logits[v] - max_logit);
                sum += probs[v];
            }
            for (auto& p : probs) {
                p /= sum;
            }

            // Top-p (nucleus) filtering
            if (top_p < 1.0) {
                // Create sorted indices
                std::vector<std::pair<float, int64_t>> sorted_probs;
                for (int64_t v = 0; v < config_.vocab_size; ++v) {
                    sorted_probs.emplace_back(probs[v], v);
                }
                std::sort(sorted_probs.begin(), sorted_probs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

                // Compute cumulative probability
                float cumsum = 0.0f;
                for (auto& [prob, idx] : sorted_probs) {
                    cumsum += prob;
                    if (cumsum > top_p) {
                        probs[idx] = 0.0f;
                    }
                }

                // Renormalize
                sum = 0.0f;
                for (auto& p : probs) sum += p;
                for (auto& p : probs) p /= sum;
            }

            // Sample
            std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
            int64_t next_token = dist(gen);

            // Stop on EOS
            if (next_token == eos_token_id) {
                break;
            }

            tokens.push_back(next_token);
        }

        return tokens;
    }

    const PIR270MConfig& config() const { return config_; }

private:
    // Embed tokens: [B, T] -> [B, T, n_embd]
    Tensor embed_tokens(const Tensor& idx) {
        auto sizes = idx.sizes().vec();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        // Flatten for embedding lookup (idx doesn't require grad)
        Tensor flat = idx.reshape({-1});
        Tensor embedded = tok_emb_->forward(flat);

        // Reshape back - use autograd reshape to preserve gradient flow
        return torch::autograd::reshape_autograd(embedded, {B, T, config_.n_embd});
    }

    // Project to vocab: [B, T, n_embd] -> [B, T, vocab_size]
    Tensor project_to_vocab(const Tensor& hidden) {
        auto sizes = hidden.sizes().vec();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        // Flatten for linear (use autograd reshape to preserve gradient flow)
        Tensor flat = torch::autograd::reshape_autograd(hidden, {B * T, config_.n_embd});
        Tensor logits = lm_head_->forward(flat);

        // Reshape back (use autograd reshape)
        return torch::autograd::reshape_autograd(logits, {B, T, config_.vocab_size});
    }

    // Cross-entropy loss with autograd tracking
    Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
        // logits: [B, T, vocab_size]
        // targets: [B, T]
        auto sizes = logits.sizes().vec();
        int64_t B = sizes[0];
        int64_t T = sizes[1];
        int64_t V = sizes[2];
        int64_t total = B * T;

        bool is_cuda = logits.is_cuda();

        // Flatten logits to [B*T, vocab_size] - use autograd reshape
        Tensor flat_logits = torch::autograd::reshape_autograd(logits, {total, V});
        Tensor flat_targets = targets.reshape({total});  // Targets don't need grad

        // Move to CPU for computation if on CUDA (temporary until CUDA cross_entropy kernel)
        Tensor logits_cpu = flat_logits;
        Tensor targets_cpu = flat_targets;
#ifdef PT_USE_CUDA
        if (is_cuda) {
            logits_cpu = at::to_cpu(flat_logits);
            targets_cpu = at::to_cpu(flat_targets);
        }
#endif

        const float* logits_data = logits_cpu.data_ptr<float>();
        const float* targets_data = targets_cpu.data_ptr<float>();

        // Compute softmax and loss
        Tensor softmax = at::empty({total, V});
        float* softmax_data = softmax.mutable_data_ptr<float>();

        float total_loss = 0.0f;
        int64_t count = 0;

        for (int64_t i = 0; i < total; ++i) {
            int64_t offset = i * V;
            int64_t target_idx = static_cast<int64_t>(targets_data[i]);

            // Compute softmax for numerical stability
            float max_logit = logits_data[offset];
            for (int64_t v = 1; v < V; ++v) {
                max_logit = std::max(max_logit, logits_data[offset + v]);
            }

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                float exp_val = std::exp(logits_data[offset + v] - max_logit);
                softmax_data[offset + v] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize softmax
            for (int64_t v = 0; v < V; ++v) {
                softmax_data[offset + v] /= sum_exp;
            }

            // Compute loss for valid targets
            if (target_idx >= 0 && target_idx < V) {
                float log_prob = std::log(softmax_data[offset + target_idx] + 1e-10f);
                total_loss -= log_prob;
                count++;
            }
        }

        // Create loss tensor
        Tensor loss = at::empty({});
        if (count > 0) {
            loss.mutable_data_ptr<float>()[0] = total_loss / static_cast<float>(count);
        } else {
            loss.mutable_data_ptr<float>()[0] = 0.0f;
        }

        // Move loss back to GPU if needed
#ifdef PT_USE_CUDA
        if (is_cuda) {
            loss = at::to_cuda(loss);
        }
#endif

        // Set up autograd - only if logits requires grad
        if (logits.requires_grad()) {
            auto backward_fn = std::make_shared<torch::autograd::CrossEntropyBackward>(
                softmax, targets_cpu, -100, V, count, is_cuda  // Use CPU targets, track CUDA flag!
            );
            backward_fn->add_input_metadata(flat_logits);

            // Directly set autograd metadata
            auto* meta = torch::autograd::ensure_autograd_meta_impl(loss);
            meta->grad_fn = backward_fn;
            meta->output_nr_ = 0;
            meta->is_leaf_ = false;
            meta->requires_grad_ = true;
        }

        return loss;
    }

    PIR270MConfig config_;
    std::shared_ptr<Embedding> tok_emb_;
    std::shared_ptr<RotaryEmbedding> rope_;
    std::vector<std::shared_ptr<PIRTransformerBlock>> blocks_;
    std::shared_ptr<RMSNorm> norm_out_;
    std::shared_ptr<Linear> lm_head_;
};

// ============================================================================
// Factory function
// ============================================================================

inline std::shared_ptr<PIR270M> create_pir_270m(const PIR270MConfig& config = PIR270MConfig()) {
    return std::make_shared<PIR270M>(config);
}

} // namespace nn
} // namespace torch
