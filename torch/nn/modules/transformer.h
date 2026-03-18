#pragma once

#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/dropout.h"
#include "torch/nn/modules/normalization.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/modules/attention.h"
#include <cmath>

namespace torch {
namespace nn {

// ============================================================================
// PositionalEncoding - Sinusoidal positional embeddings
// ============================================================================
// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

class PositionalEncoding : public Module {
public:
    PositionalEncoding(
        int64_t d_model,
        double dropout = 0.1,
        int64_t max_len = 5000
    )
        : Module("PositionalEncoding")
        , d_model_(d_model)
        , dropout_prob_(dropout)
        , max_len_(max_len)
    {
        // Precompute positional encodings
        Tensor pe = at::zeros({max_len, d_model});
        float* pe_data = pe.mutable_data_ptr<float>();

        for (int64_t pos = 0; pos < max_len; ++pos) {
            for (int64_t i = 0; i < d_model; i += 2) {
                double div_term = std::exp(-static_cast<double>(i) * std::log(10000.0) / d_model);
                pe_data[pos * d_model + i] = static_cast<float>(std::sin(pos * div_term));
                if (i + 1 < d_model) {
                    pe_data[pos * d_model + i + 1] = static_cast<float>(std::cos(pos * div_term));
                }
            }
        }

        // Register as buffer (not trained)
        register_buffer("pe", Buffer(pe, false));  // non-persistent

        // Dropout
        dropout_ = std::make_shared<Dropout>(dropout);
        register_module("dropout", dropout_);
    }

    // Input: [seq_len, batch, d_model] or [batch, seq_len, d_model] with batch_first
    Tensor forward(const Tensor& input) override {
        int64_t seq_len = input.size(0);  // Assuming not batch_first

        Buffer* pe_buf = get_buffer("pe");
        Tensor pe = pe_buf->data();

        // Get positional encoding for this sequence length
        // pe: [max_len, d_model] -> [seq_len, d_model]
        Tensor pos_enc = pe.slice(0, 0, seq_len);

        // Add to input (broadcasting across batch dimension)
        Tensor result = input.clone();
        float* out_data = result.mutable_data_ptr<float>();
        const float* pe_data = pos_enc.data_ptr<float>();

        int64_t batch_size = input.size(1);

        for (int64_t s = 0; s < seq_len; ++s) {
            for (int64_t b = 0; b < batch_size; ++b) {
                for (int64_t d = 0; d < d_model_; ++d) {
                    int64_t idx = (s * batch_size + b) * d_model_ + d;
                    out_data[idx] += pe_data[s * d_model_ + d];
                }
            }
        }

        return dropout_->forward(result);
    }

    // Batch-first version
    Tensor forward_batch_first(const Tensor& input) {
        // Input: [batch, seq_len, d_model]
        int64_t batch_size = input.size(0);
        int64_t seq_len = input.size(1);

        Buffer* pe_buf = get_buffer("pe");
        Tensor pe = pe_buf->data();
        Tensor pos_enc = pe.slice(0, 0, seq_len);

        Tensor result = input.clone();
        float* out_data = result.mutable_data_ptr<float>();
        const float* pe_data = pos_enc.data_ptr<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                for (int64_t d = 0; d < d_model_; ++d) {
                    int64_t idx = (b * seq_len + s) * d_model_ + d;
                    out_data[idx] += pe_data[s * d_model_ + d];
                }
            }
        }

        return dropout_->forward(result);
    }

private:
    int64_t d_model_;
    double dropout_prob_;
    int64_t max_len_;
    std::shared_ptr<Dropout> dropout_;
};

// ============================================================================
// TransformerEncoderLayer - Single transformer encoder layer
// ============================================================================
// Consists of self-attention + feedforward network with residual connections
// and layer normalization.

class TransformerEncoderLayer : public Module {
public:
    TransformerEncoderLayer(
        int64_t d_model,
        int64_t nhead,
        int64_t dim_feedforward = 2048,
        double dropout = 0.1,
        const std::string& activation = "relu",
        double layer_norm_eps = 1e-5,
        bool batch_first = false,
        bool norm_first = false
    )
        : Module("TransformerEncoderLayer")
        , d_model_(d_model)
        , nhead_(nhead)
        , dim_feedforward_(dim_feedforward)
        , dropout_prob_(dropout)
        , batch_first_(batch_first)
        , norm_first_(norm_first)
    {
        // Self-attention
        self_attn_ = std::make_shared<MultiheadAttention>(
            d_model, nhead, dropout, true, false, false, 0, 0, batch_first
        );
        register_module("self_attn", self_attn_);

        // Feedforward network
        linear1_ = std::make_shared<Linear>(d_model, dim_feedforward);
        linear2_ = std::make_shared<Linear>(dim_feedforward, d_model);
        register_module("linear1", linear1_);
        register_module("linear2", linear2_);

        // Layer normalization
        norm1_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model}, layer_norm_eps);
        norm2_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model}, layer_norm_eps);
        register_module("norm1", norm1_);
        register_module("norm2", norm2_);

        // Dropout
        dropout1_ = std::make_shared<Dropout>(dropout);
        dropout2_ = std::make_shared<Dropout>(dropout);
        register_module("dropout1", dropout1_);
        register_module("dropout2", dropout2_);

        // Activation
        if (activation == "gelu") {
            activation_ = std::make_shared<GELU>();
        } else {
            activation_ = std::make_shared<ReLU>();
        }
        register_module("activation", activation_);
    }

    // Forward pass
    // Input shape: [seq_len, batch, d_model] or [batch, seq_len, d_model] if batch_first
    Tensor forward(const Tensor& src) override {
        return forward_with_mask(src, Tensor(), Tensor());
    }

    Tensor forward_with_mask(
        const Tensor& src,
        const Tensor& src_mask,
        const Tensor& src_key_padding_mask
    ) {
        Tensor x = src;

        if (norm_first_) {
            // Pre-norm: Norm -> Attention -> Add -> Norm -> FFN -> Add
            Tensor normed = norm1_->forward(x);
            auto [attn_out, _] = self_attn_->forward_attention(
                normed, normed, normed, src_mask, src_key_padding_mask, false
            );
            x = x.add(dropout1_->forward(attn_out));

            normed = norm2_->forward(x);
            Tensor ff_out = feedforward(normed);
            x = x.add(dropout2_->forward(ff_out));
        } else {
            // Post-norm: Attention -> Add -> Norm -> FFN -> Add -> Norm
            auto [attn_out, _] = self_attn_->forward_attention(
                x, x, x, src_mask, src_key_padding_mask, false
            );
            x = norm1_->forward(x.add(dropout1_->forward(attn_out)));

            Tensor ff_out = feedforward(x);
            x = norm2_->forward(x.add(dropout2_->forward(ff_out)));
        }

        return x;
    }

    friend class TransformerEncoder;

private:
    Tensor feedforward(const Tensor& x) {
        Tensor out = linear1_->forward(x);
        out = activation_->forward(out);
        out = linear2_->forward(out);
        return out;
    }

    int64_t d_model_;
    int64_t nhead_;
    int64_t dim_feedforward_;
    double dropout_prob_;
    bool batch_first_;
    bool norm_first_;

    std::shared_ptr<MultiheadAttention> self_attn_;
    std::shared_ptr<Linear> linear1_;
    std::shared_ptr<Linear> linear2_;
    std::shared_ptr<LayerNorm> norm1_;
    std::shared_ptr<LayerNorm> norm2_;
    std::shared_ptr<Dropout> dropout1_;
    std::shared_ptr<Dropout> dropout2_;
    std::shared_ptr<Module> activation_;
};

// ============================================================================
// TransformerEncoder - Stack of transformer encoder layers
// ============================================================================

class TransformerEncoder : public Module {
public:
    TransformerEncoder(
        std::shared_ptr<TransformerEncoderLayer> encoder_layer,
        int64_t num_layers,
        bool enable_nested_tensor = false
    )
        : Module("TransformerEncoder")
        , num_layers_(num_layers)
    {
        // Create separate layers with the same config as encoder_layer
        // (pushing the same shared_ptr would share weights across all layers!)
        for (int64_t i = 0; i < num_layers; ++i) {
            auto layer = std::make_shared<TransformerEncoderLayer>(
                encoder_layer->d_model_,
                encoder_layer->nhead_,
                encoder_layer->dim_feedforward_,
                encoder_layer->dropout_prob_,
                "relu",  // default activation
                1e-5,    // default layer_norm_eps
                encoder_layer->batch_first_,
                encoder_layer->norm_first_
            );
            layers_.push_back(layer);
            register_module("layers." + std::to_string(i), layer);
        }
    }

    // Alternative constructor: create layers internally
    TransformerEncoder(
        int64_t d_model,
        int64_t nhead,
        int64_t num_layers,
        int64_t dim_feedforward = 2048,
        double dropout = 0.1,
        const std::string& activation = "relu",
        double layer_norm_eps = 1e-5,
        bool batch_first = false,
        bool norm_first = false
    )
        : Module("TransformerEncoder")
        , num_layers_(num_layers)
    {
        for (int64_t i = 0; i < num_layers; ++i) {
            auto layer = std::make_shared<TransformerEncoderLayer>(
                d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first
            );
            layers_.push_back(layer);
            register_module("layers." + std::to_string(i), layer);
        }

        // Optional final layer norm
        norm_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model}, layer_norm_eps);
        register_module("norm", norm_);
    }

    Tensor forward(const Tensor& src) override {
        return forward_with_mask(src, Tensor(), Tensor());
    }

    Tensor forward_with_mask(
        const Tensor& src,
        const Tensor& mask,
        const Tensor& src_key_padding_mask
    ) {
        Tensor output = src;

        for (auto& layer : layers_) {
            output = layer->forward_with_mask(output, mask, src_key_padding_mask);
        }

        if (norm_) {
            output = norm_->forward(output);
        }

        return output;
    }

private:
    int64_t num_layers_;
    std::vector<std::shared_ptr<TransformerEncoderLayer>> layers_;
    std::shared_ptr<LayerNorm> norm_;
};

// ============================================================================
// TransformerDecoderLayer - Single transformer decoder layer
// ============================================================================
// Consists of self-attention + cross-attention + FFN

class TransformerDecoderLayer : public Module {
public:
    TransformerDecoderLayer(
        int64_t d_model,
        int64_t nhead,
        int64_t dim_feedforward = 2048,
        double dropout = 0.1,
        const std::string& activation = "relu",
        double layer_norm_eps = 1e-5,
        bool batch_first = false,
        bool norm_first = false
    )
        : Module("TransformerDecoderLayer")
        , d_model_(d_model)
        , batch_first_(batch_first)
        , norm_first_(norm_first)
    {
        // Self-attention
        self_attn_ = std::make_shared<MultiheadAttention>(
            d_model, nhead, dropout, true, false, false, 0, 0, batch_first
        );
        register_module("self_attn", self_attn_);

        // Cross-attention (encoder-decoder attention)
        multihead_attn_ = std::make_shared<MultiheadAttention>(
            d_model, nhead, dropout, true, false, false, 0, 0, batch_first
        );
        register_module("multihead_attn", multihead_attn_);

        // Feedforward
        linear1_ = std::make_shared<Linear>(d_model, dim_feedforward);
        linear2_ = std::make_shared<Linear>(dim_feedforward, d_model);
        register_module("linear1", linear1_);
        register_module("linear2", linear2_);

        // Layer norms
        norm1_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model}, layer_norm_eps);
        norm2_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model}, layer_norm_eps);
        norm3_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model}, layer_norm_eps);
        register_module("norm1", norm1_);
        register_module("norm2", norm2_);
        register_module("norm3", norm3_);

        // Dropout
        dropout1_ = std::make_shared<Dropout>(dropout);
        dropout2_ = std::make_shared<Dropout>(dropout);
        dropout3_ = std::make_shared<Dropout>(dropout);
        register_module("dropout1", dropout1_);
        register_module("dropout2", dropout2_);
        register_module("dropout3", dropout3_);

        // Activation
        if (activation == "gelu") {
            activation_ = std::make_shared<GELU>();
        } else {
            activation_ = std::make_shared<ReLU>();
        }
        register_module("activation", activation_);
    }

    Tensor forward(const Tensor& tgt) override {
        throw std::runtime_error("TransformerDecoderLayer requires both tgt and memory");
    }

    Tensor forward_with_memory(
        const Tensor& tgt,
        const Tensor& memory,
        const Tensor& tgt_mask = Tensor(),
        const Tensor& memory_mask = Tensor(),
        const Tensor& tgt_key_padding_mask = Tensor(),
        const Tensor& memory_key_padding_mask = Tensor()
    ) {
        Tensor x = tgt;

        if (norm_first_) {
            // Pre-norm
            Tensor normed = norm1_->forward(x);
            auto [sa_out, _] = self_attn_->forward_attention(
                normed, normed, normed, tgt_mask, tgt_key_padding_mask, false
            );
            x = x.add(dropout1_->forward(sa_out));

            normed = norm2_->forward(x);
            auto [ca_out, __] = multihead_attn_->forward_attention(
                normed, memory, memory, memory_mask, memory_key_padding_mask, false
            );
            x = x.add(dropout2_->forward(ca_out));

            normed = norm3_->forward(x);
            Tensor ff_out = feedforward(normed);
            x = x.add(dropout3_->forward(ff_out));
        } else {
            // Post-norm
            auto [sa_out, _] = self_attn_->forward_attention(
                x, x, x, tgt_mask, tgt_key_padding_mask, false
            );
            x = norm1_->forward(x.add(dropout1_->forward(sa_out)));

            auto [ca_out, __] = multihead_attn_->forward_attention(
                x, memory, memory, memory_mask, memory_key_padding_mask, false
            );
            x = norm2_->forward(x.add(dropout2_->forward(ca_out)));

            Tensor ff_out = feedforward(x);
            x = norm3_->forward(x.add(dropout3_->forward(ff_out)));
        }

        return x;
    }

private:
    Tensor feedforward(const Tensor& x) {
        Tensor out = linear1_->forward(x);
        out = activation_->forward(out);
        out = linear2_->forward(out);
        return out;
    }

    int64_t d_model_;
    bool batch_first_;
    bool norm_first_;

    std::shared_ptr<MultiheadAttention> self_attn_;
    std::shared_ptr<MultiheadAttention> multihead_attn_;
    std::shared_ptr<Linear> linear1_;
    std::shared_ptr<Linear> linear2_;
    std::shared_ptr<LayerNorm> norm1_;
    std::shared_ptr<LayerNorm> norm2_;
    std::shared_ptr<LayerNorm> norm3_;
    std::shared_ptr<Dropout> dropout1_;
    std::shared_ptr<Dropout> dropout2_;
    std::shared_ptr<Dropout> dropout3_;
    std::shared_ptr<Module> activation_;
};

// ============================================================================
// TransformerDecoder - Stack of decoder layers
// ============================================================================

class TransformerDecoder : public Module {
public:
    TransformerDecoder(
        int64_t d_model,
        int64_t nhead,
        int64_t num_layers,
        int64_t dim_feedforward = 2048,
        double dropout = 0.1,
        const std::string& activation = "relu",
        double layer_norm_eps = 1e-5,
        bool batch_first = false,
        bool norm_first = false
    )
        : Module("TransformerDecoder")
        , num_layers_(num_layers)
    {
        for (int64_t i = 0; i < num_layers; ++i) {
            auto layer = std::make_shared<TransformerDecoderLayer>(
                d_model, nhead, dim_feedforward, dropout,
                activation, layer_norm_eps, batch_first, norm_first
            );
            layers_.push_back(layer);
            register_module("layers." + std::to_string(i), layer);
        }

        norm_ = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model}, layer_norm_eps);
        register_module("norm", norm_);
    }

    Tensor forward(const Tensor& tgt) override {
        throw std::runtime_error("TransformerDecoder requires both tgt and memory");
    }

    Tensor forward_with_memory(
        const Tensor& tgt,
        const Tensor& memory,
        const Tensor& tgt_mask = Tensor(),
        const Tensor& memory_mask = Tensor(),
        const Tensor& tgt_key_padding_mask = Tensor(),
        const Tensor& memory_key_padding_mask = Tensor()
    ) {
        Tensor output = tgt;

        for (auto& layer : layers_) {
            output = layer->forward_with_memory(
                output, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask
            );
        }

        if (norm_) {
            output = norm_->forward(output);
        }

        return output;
    }

private:
    int64_t num_layers_;
    std::vector<std::shared_ptr<TransformerDecoderLayer>> layers_;
    std::shared_ptr<LayerNorm> norm_;
};

// ============================================================================
// Transformer - Full encoder-decoder transformer
// ============================================================================

class Transformer : public Module {
public:
    Transformer(
        int64_t d_model = 512,
        int64_t nhead = 8,
        int64_t num_encoder_layers = 6,
        int64_t num_decoder_layers = 6,
        int64_t dim_feedforward = 2048,
        double dropout = 0.1,
        const std::string& activation = "relu",
        double layer_norm_eps = 1e-5,
        bool batch_first = false,
        bool norm_first = false
    )
        : Module("Transformer")
        , d_model_(d_model)
        , batch_first_(batch_first)
    {
        encoder_ = std::make_shared<TransformerEncoder>(
            d_model, nhead, num_encoder_layers, dim_feedforward,
            dropout, activation, layer_norm_eps, batch_first, norm_first
        );
        register_module("encoder", encoder_);

        decoder_ = std::make_shared<TransformerDecoder>(
            d_model, nhead, num_decoder_layers, dim_feedforward,
            dropout, activation, layer_norm_eps, batch_first, norm_first
        );
        register_module("decoder", decoder_);
    }

    Tensor forward(const Tensor& src) override {
        throw std::runtime_error("Transformer requires both src and tgt");
    }

    Tensor forward(
        const Tensor& src,
        const Tensor& tgt,
        const Tensor& src_mask = Tensor(),
        const Tensor& tgt_mask = Tensor(),
        const Tensor& memory_mask = Tensor(),
        const Tensor& src_key_padding_mask = Tensor(),
        const Tensor& tgt_key_padding_mask = Tensor(),
        const Tensor& memory_key_padding_mask = Tensor()
    ) {
        // Encode
        Tensor memory = encoder_->forward_with_mask(src, src_mask, src_key_padding_mask);

        // Decode
        Tensor output = decoder_->forward_with_memory(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask
        );

        return output;
    }

    // Encoder-only forward
    Tensor encode(
        const Tensor& src,
        const Tensor& src_mask = Tensor(),
        const Tensor& src_key_padding_mask = Tensor()
    ) {
        return encoder_->forward_with_mask(src, src_mask, src_key_padding_mask);
    }

    // Decoder-only forward (given memory)
    Tensor decode(
        const Tensor& tgt,
        const Tensor& memory,
        const Tensor& tgt_mask = Tensor(),
        const Tensor& memory_mask = Tensor(),
        const Tensor& tgt_key_padding_mask = Tensor(),
        const Tensor& memory_key_padding_mask = Tensor()
    ) {
        return decoder_->forward_with_memory(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask
        );
    }

private:
    int64_t d_model_;
    bool batch_first_;
    std::shared_ptr<TransformerEncoder> encoder_;
    std::shared_ptr<TransformerDecoder> decoder_;
};

} // namespace nn
} // namespace torch
