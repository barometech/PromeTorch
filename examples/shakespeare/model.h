#pragma once

// ============================================================================
// TransformerLM - Character-level Language Model
// ============================================================================
// A decoder-only Transformer model for character-level text generation.
// Similar to GPT architecture but smaller for proof of concept.
//
// Architecture:
//   - Token Embedding (vocab_size -> d_model)
//   - Positional Encoding (sinusoidal)
//   - N x TransformerEncoderLayer (with causal masking)
//   - Output Linear (d_model -> vocab_size)

#include "torch/nn/nn.h"
#include <memory>
#include <set>
#include <fstream>
#include <string>
#include <unordered_map>

namespace shakespeare {

using namespace torch;
using namespace torch::nn;

class TransformerLM : public Module {
public:
    TransformerLM(
        int64_t vocab_size,
        int64_t d_model = 256,
        int64_t nhead = 4,
        int64_t num_layers = 4,
        int64_t dim_feedforward = 512,
        double dropout = 0.1,
        int64_t max_seq_len = 256
    )
        : Module("TransformerLM")
        , vocab_size_(vocab_size)
        , d_model_(d_model)
        , max_seq_len_(max_seq_len)
    {
        // Token embedding
        embedding_ = std::make_shared<Embedding>(vocab_size, d_model);
        register_module("embedding", embedding_);

        // Positional encoding
        pos_encoder_ = std::make_shared<PositionalEncoding>(d_model, dropout, max_seq_len);
        register_module("pos_encoder", pos_encoder_);

        // Transformer encoder (used as decoder with causal mask)
        transformer_ = std::make_shared<TransformerEncoder>(
            d_model, nhead, num_layers, dim_feedforward,
            dropout, "gelu", 1e-5, false, true  // Pre-norm for better training
        );
        register_module("transformer", transformer_);

        // Output projection
        output_ = std::make_shared<Linear>(d_model, vocab_size, false);  // No bias
        register_module("output", output_);

        // Initialize weights
        init_weights();
    }

    void init_weights() {
        // Weight tying: share embedding and output weights
        // This is a common technique that improves performance and reduces parameters
        auto* emb_weight = embedding_->get_parameter("weight");
        auto* out_weight = output_->get_parameter("weight");
        if (emb_weight && out_weight) {
            // Copy embedding weights to output (they share the same dimension)
            Tensor emb_w = emb_weight->data();  // [vocab_size, d_model]
            Tensor out_w = out_weight->data();  // [vocab_size, d_model]

            // Initialize embedding with small values
            float init_range = 0.1f;
            float* emb_data = emb_w.mutable_data_ptr<float>();
            for (int64_t i = 0; i < emb_w.numel(); ++i) {
                emb_data[i] = (2.0f * static_cast<float>(::rand()) / RAND_MAX - 1.0f) * init_range;
            }

            // Copy to output
            float* out_data = out_w.mutable_data_ptr<float>();
            for (int64_t i = 0; i < out_w.numel(); ++i) {
                out_data[i] = emb_data[i];
            }
        }
    }

    // Forward pass
    // Input: [seq_len, batch] tensor of token indices (stored as float)
    // Output: [seq_len, batch, vocab_size] logits
    Tensor forward(const Tensor& input) override {
        int64_t seq_len = input.size(0);
        int64_t batch_size = input.size(1);

        // Create causal mask
        Tensor mask = generate_square_subsequent_mask(seq_len);

        // Get embeddings: [seq_len, batch] -> [seq_len, batch, d_model]
        Tensor embedded = embed_tokens(input);

        // Add positional encoding
        Tensor encoded = pos_encoder_->forward(embedded);

        // Scale embeddings (as in original Transformer paper)
        float scale = std::sqrt(static_cast<float>(d_model_));
        encoded = encoded.mul(at::Scalar(scale));

        // Transformer forward with causal mask
        Tensor transformer_out = transformer_->forward_with_mask(encoded, mask, Tensor());

        // Output projection: [seq_len, batch, d_model] -> [seq_len, batch, vocab_size]
        Tensor logits = output_projection(transformer_out);

        return logits;
    }

    // Generate text autoregressively
    std::vector<int64_t> generate(
        const std::vector<int64_t>& prompt,
        int64_t max_new_tokens,
        double temperature = 1.0,
        bool do_sample = true
    ) {
        eval();  // Set to evaluation mode

        std::vector<int64_t> tokens = prompt;
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int64_t i = 0; i < max_new_tokens; ++i) {
            // Prepare input: [seq_len, 1]
            int64_t context_len = std::min(static_cast<int64_t>(tokens.size()), max_seq_len_);
            int64_t start_idx = tokens.size() > static_cast<size_t>(max_seq_len_) ?
                                tokens.size() - max_seq_len_ : 0;

            Tensor input = at::empty({context_len, 1});
            float* input_data = input.mutable_data_ptr<float>();
            for (int64_t j = 0; j < context_len; ++j) {
                input_data[j] = static_cast<float>(tokens[start_idx + j]);
            }

            // Forward pass
            Tensor logits = forward(input);  // [seq_len, 1, vocab_size]

            // Get logits for last position
            // logits[-1, 0, :] -> [vocab_size]
            int64_t last_pos = context_len - 1;
            std::vector<float> last_logits(vocab_size_);
            const float* logits_data = logits.data_ptr<float>();
            for (int64_t v = 0; v < vocab_size_; ++v) {
                last_logits[v] = logits_data[(last_pos * 1 + 0) * vocab_size_ + v];
            }

            // Apply temperature
            if (temperature != 1.0) {
                for (auto& logit : last_logits) {
                    logit /= static_cast<float>(temperature);
                }
            }

            int64_t next_token;
            if (do_sample) {
                // Sample from softmax distribution
                // First compute softmax
                float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
                std::vector<float> probs(vocab_size_);
                float sum = 0.0f;
                for (int64_t v = 0; v < vocab_size_; ++v) {
                    probs[v] = std::exp(last_logits[v] - max_logit);
                    sum += probs[v];
                }
                for (auto& p : probs) {
                    p /= sum;
                }

                // Sample
                std::discrete_distribution<> dist(probs.begin(), probs.end());
                next_token = dist(gen);
            } else {
                // Greedy: take argmax
                next_token = std::distance(
                    last_logits.begin(),
                    std::max_element(last_logits.begin(), last_logits.end())
                );
            }

            tokens.push_back(next_token);
        }

        return tokens;
    }

    int64_t vocab_size() const { return vocab_size_; }
    int64_t d_model() const { return d_model_; }
    int64_t max_seq_len() const { return max_seq_len_; }

private:
    // Embed tokens: [seq_len, batch] -> [seq_len, batch, d_model]
    Tensor embed_tokens(const Tensor& input) {
        int64_t seq_len = input.size(0);
        int64_t batch_size = input.size(1);

        // Flatten input for embedding lookup
        Tensor flat_input = input.reshape({-1});  // [seq_len * batch]
        Tensor embedded = embedding_->forward(flat_input);  // [seq_len * batch, d_model]

        // Reshape back to [seq_len, batch, d_model]
        return embedded.reshape({seq_len, batch_size, d_model_});
    }

    // Output projection: [seq_len, batch, d_model] -> [seq_len, batch, vocab_size]
    Tensor output_projection(const Tensor& hidden) {
        int64_t seq_len = hidden.size(0);
        int64_t batch_size = hidden.size(1);

        // Flatten for linear layer
        Tensor flat_hidden = hidden.reshape({seq_len * batch_size, d_model_});
        Tensor logits = output_->forward(flat_hidden);  // [seq_len * batch, vocab_size]

        // Reshape back
        return logits.reshape({seq_len, batch_size, vocab_size_});
    }

    int64_t vocab_size_;
    int64_t d_model_;
    int64_t max_seq_len_;

    std::shared_ptr<Embedding> embedding_;
    std::shared_ptr<PositionalEncoding> pos_encoder_;
    std::shared_ptr<TransformerEncoder> transformer_;
    std::shared_ptr<Linear> output_;
};

// ============================================================================
// Character-level tokenizer
// ============================================================================

class CharTokenizer {
public:
    CharTokenizer() = default;

    // Build vocabulary from text
    void build_vocab(const std::string& text) {
        std::set<char> chars(text.begin(), text.end());

        int64_t idx = 0;
        for (char c : chars) {
            if (char_to_idx_.find(c) == char_to_idx_.end()) {
                char_to_idx_[c] = idx;
                idx_to_char_[idx] = c;
                idx++;
            }
        }
        vocab_size_ = idx;
    }

    // Encode string to token indices
    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> tokens;
        tokens.reserve(text.size());
        for (char c : text) {
            auto it = char_to_idx_.find(c);
            if (it != char_to_idx_.end()) {
                tokens.push_back(it->second);
            }
        }
        return tokens;
    }

    // Decode token indices to string
    std::string decode(const std::vector<int64_t>& tokens) const {
        std::string text;
        text.reserve(tokens.size());
        for (int64_t idx : tokens) {
            auto it = idx_to_char_.find(idx);
            if (it != idx_to_char_.end()) {
                text += it->second;
            }
        }
        return text;
    }

    int64_t vocab_size() const { return vocab_size_; }

    // Save/Load vocabulary
    void save(const std::string& path) const {
        std::ofstream file(path);
        for (const auto& [c, idx] : char_to_idx_) {
            file << static_cast<int>(c) << " " << idx << "\n";
        }
    }

    void load(const std::string& path) {
        std::ifstream file(path);
        char_to_idx_.clear();
        idx_to_char_.clear();

        int char_code;
        int64_t idx;
        while (file >> char_code >> idx) {
            char c = static_cast<char>(char_code);
            char_to_idx_[c] = idx;
            idx_to_char_[idx] = c;
        }
        vocab_size_ = idx_to_char_.size();
    }

private:
    std::map<char, int64_t> char_to_idx_;
    std::map<int64_t, char> idx_to_char_;
    int64_t vocab_size_ = 0;
};

// ============================================================================
// Text Dataset for character-level modeling
// ============================================================================

class TextDataset {
public:
    TextDataset(
        const std::vector<int64_t>& tokens,
        int64_t block_size
    )
        : tokens_(tokens)
        , block_size_(block_size)
    {
        if (tokens_.size() < static_cast<size_t>(block_size_ + 1)) {
            throw std::runtime_error("Text too short for given block_size");
        }
    }

    size_t size() const {
        return tokens_.size() - block_size_;
    }

    // Get a training example (input, target)
    // Input: tokens[i:i+block_size]
    // Target: tokens[i+1:i+block_size+1]
    std::pair<Tensor, Tensor> get(size_t index) const {
        Tensor input = at::empty({block_size_});
        Tensor target = at::empty({block_size_});

        float* input_data = input.mutable_data_ptr<float>();
        float* target_data = target.mutable_data_ptr<float>();

        for (int64_t i = 0; i < block_size_; ++i) {
            input_data[i] = static_cast<float>(tokens_[index + i]);
            target_data[i] = static_cast<float>(tokens_[index + i + 1]);
        }

        return {input, target};
    }

    // Get a random batch
    std::pair<Tensor, Tensor> get_batch(int64_t batch_size, std::mt19937& gen) const {
        std::uniform_int_distribution<size_t> dist(0, size() - 1);

        // [block_size, batch_size]
        Tensor input = at::empty({block_size_, batch_size});
        Tensor target = at::empty({block_size_, batch_size});

        float* input_data = input.mutable_data_ptr<float>();
        float* target_data = target.mutable_data_ptr<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            size_t idx = dist(gen);
            for (int64_t i = 0; i < block_size_; ++i) {
                input_data[i * batch_size + b] = static_cast<float>(tokens_[idx + i]);
                target_data[i * batch_size + b] = static_cast<float>(tokens_[idx + i + 1]);
            }
        }

        return {input, target};
    }

private:
    std::vector<int64_t> tokens_;
    int64_t block_size_;
};

} // namespace shakespeare
