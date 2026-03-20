#pragma once

#include "../module.h"
#include "../init.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/functions/MathBackward.h"
#include <cmath>
#include <algorithm>

namespace torch {
namespace nn {

// ============================================================================
// Embedding - A simple lookup table that stores embeddings of a fixed dictionary
// ============================================================================
// This module is often used to store word embeddings and retrieve them using indices.
// The input to the module is a list of indices, and the output is the corresponding
// word embeddings.
//
// Shape:
//   Input: (*) where * means any number of dimensions
//   Output: (*, H) where H = embedding_dim
//
// Attributes:
//   weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)

class Embedding : public Module {
private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
    bool has_padding_idx_;
    double max_norm_;
    bool has_max_norm_;
    double norm_type_;
    bool scale_grad_by_freq_;
    bool sparse_;

public:
    Embedding(
        int64_t num_embeddings,
        int64_t embedding_dim,
        int64_t padding_idx = -1,          // -1 means no padding
        double max_norm = -1.0,             // -1 means no max_norm
        double norm_type = 2.0,
        bool scale_grad_by_freq = false,
        bool sparse = false
    ) : num_embeddings_(num_embeddings),
        embedding_dim_(embedding_dim),
        padding_idx_(padding_idx),
        has_padding_idx_(padding_idx >= 0),
        max_norm_(max_norm),
        has_max_norm_(max_norm > 0),
        norm_type_(norm_type),
        scale_grad_by_freq_(scale_grad_by_freq),
        sparse_(sparse) {

        if (num_embeddings <= 0) {
            throw std::invalid_argument("num_embeddings must be positive");
        }
        if (embedding_dim <= 0) {
            throw std::invalid_argument("embedding_dim must be positive");
        }
        if (has_padding_idx_ && (padding_idx_ < 0 || padding_idx_ >= num_embeddings_)) {
            throw std::invalid_argument(
                "padding_idx must be within num_embeddings, got " + std::to_string(padding_idx_)
            );
        }

        // Initialize weight
        Tensor weight = at::empty({num_embeddings, embedding_dim});
        init::normal_(weight, 0.0, 1.0);

        // Zero out padding index if specified
        if (has_padding_idx_) {
            float* data = weight.mutable_data_ptr<float>();
            for (int64_t i = 0; i < embedding_dim_; ++i) {
                data[padding_idx_ * embedding_dim_ + i] = 0.0f;
            }
        }

        register_parameter("weight", Parameter(weight));
    }

    // Create from pretrained embeddings
    static std::shared_ptr<Embedding> from_pretrained(
        const Tensor& embeddings,
        bool freeze = true,
        int64_t padding_idx = -1,
        double max_norm = -1.0,
        double norm_type = 2.0,
        bool scale_grad_by_freq = false,
        bool sparse = false
    ) {
        if (embeddings.dim() != 2) {
            throw std::invalid_argument("Embeddings must be 2D");
        }

        int64_t num_embeddings = embeddings.size(0);
        int64_t embedding_dim = embeddings.size(1);

        auto embedding = std::make_shared<Embedding>(
            num_embeddings, embedding_dim, padding_idx,
            max_norm, norm_type, scale_grad_by_freq, sparse
        );

        // Copy pretrained weights
        embedding->get_parameter("weight")->data().copy_(embeddings);

        // Optionally freeze
        if (freeze) {
            embedding->get_parameter("weight")->set_requires_grad(false);
        }

        return embedding;
    }

    Tensor forward(const Tensor& input) override {
        // Input contains indices, output is embeddings
        // Input shape: (*)
        // Output shape: (*, embedding_dim)

        Parameter* weight_param = get_parameter("weight");
        const Tensor& weight = weight_param->data();

        // Get input dimensions
        std::vector<int64_t> input_sizes = input.sizes().vec();
        int64_t num_indices = input.numel();

        // Create output shape: input_shape + [embedding_dim]
        std::vector<int64_t> output_sizes = input_sizes;
        output_sizes.push_back(embedding_dim_);

        // Check if we need to handle CUDA tensors
        bool is_cuda = weight.is_cuda() || input.is_cuda();

        // Move to CPU for lookup if needed (until CUDA embedding kernel)
        Tensor input_cpu = input;
        Tensor weight_cpu = weight;
#ifdef PT_USE_CUDA
        if (input.is_cuda()) {
            input_cpu = at::to_cpu(input);
        }
        if (weight.is_cuda()) {
            weight_cpu = at::to_cpu(weight);
        }
#endif

        Tensor output = at::empty(output_sizes);

        // Get data pointers
        // Note: We assume input contains integer indices stored as float
        // In production, you'd want a separate integer tensor type
        const float* indices_data = input_cpu.data_ptr<float>();
        const float* weight_data = weight_cpu.data_ptr<float>();
        float* output_data = output.mutable_data_ptr<float>();

        // Look up embeddings (no OpenMP — throw inside parallel is illegal on LCC)
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = static_cast<int64_t>(indices_data[i]);

            if (idx < 0 || idx >= num_embeddings_) {
                throw std::out_of_range(
                    "index out of range: got " + std::to_string(idx) +
                    " but num_embeddings is " + std::to_string(num_embeddings_)
                );
            }

            // Handle max_norm if specified
            if (has_max_norm_) {
                // Compute norm of the embedding
                double norm = 0.0;
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    double val = weight_data[idx * embedding_dim_ + j];
                    if (norm_type_ == 2.0) {
                        norm += val * val;
                    } else {
                        norm += std::pow(std::abs(val), norm_type_);
                    }
                }
                norm = std::pow(norm, 1.0 / norm_type_);

                // Scale if exceeds max_norm
                double scale = (norm > max_norm_) ? (max_norm_ / norm) : 1.0;

                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_data[i * embedding_dim_ + j] =
                        static_cast<float>(weight_data[idx * embedding_dim_ + j] * scale);
                }
            } else {
                // Simple copy
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_data[i * embedding_dim_ + j] = weight_data[idx * embedding_dim_ + j];
                }
            }
        }

        // Move output to GPU if weight is on GPU
#ifdef PT_USE_CUDA
        if (is_cuda) {
            output = at::to_cuda(output);
        }
#endif

        // Set up autograd if weight requires gradients
        if (weight_param->requires_grad()) {
            auto backward_fn = std::make_shared<torch::autograd::EmbeddingBackward>(
                input.clone(),  // Store indices for backward
                weight_param->data(),  // Reference to weight for gradient accumulation
                num_embeddings_,
                embedding_dim_,
                padding_idx_,
                has_padding_idx_
            );
            auto* meta = torch::autograd::ensure_autograd_meta_impl(output);
            meta->grad_fn = backward_fn;
            meta->output_nr_ = 0;
            meta->is_leaf_ = false;
            meta->requires_grad_ = true;
        }

        return output;
    }

    std::string name() const override { return "Embedding"; }

    int64_t num_embeddings() const { return num_embeddings_; }
    int64_t embedding_dim() const { return embedding_dim_; }
    int64_t padding_idx() const { return padding_idx_; }
    bool has_padding_idx() const { return has_padding_idx_; }
};

// ============================================================================
// EmbeddingBag Mode
// ============================================================================

enum class EmbeddingBagMode {
    Sum,
    Mean,
    Max
};

// ============================================================================
// EmbeddingBag - Computes sums/means of bags of embeddings
// ============================================================================
// Computes sums or means of "bags" of embeddings, without instantiating
// the intermediate embeddings. For bags of constant length and no per_sample_weights,
// this is more efficient than creating an embedding and then reducing.
//
// Shape:
//   Input: (B, sequence_length) or 1D tensor of indices with offsets
//   Output: (B, embedding_dim)

class EmbeddingBag : public Module {
private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    double max_norm_;
    bool has_max_norm_;
    double norm_type_;
    bool scale_grad_by_freq_;
    EmbeddingBagMode mode_;
    bool sparse_;
    bool include_last_offset_;
    int64_t padding_idx_;
    bool has_padding_idx_;

public:
    EmbeddingBag(
        int64_t num_embeddings,
        int64_t embedding_dim,
        double max_norm = -1.0,
        double norm_type = 2.0,
        bool scale_grad_by_freq = false,
        EmbeddingBagMode mode = EmbeddingBagMode::Mean,
        bool sparse = false,
        bool include_last_offset = false,
        int64_t padding_idx = -1
    ) : num_embeddings_(num_embeddings),
        embedding_dim_(embedding_dim),
        max_norm_(max_norm),
        has_max_norm_(max_norm > 0),
        norm_type_(norm_type),
        scale_grad_by_freq_(scale_grad_by_freq),
        mode_(mode),
        sparse_(sparse),
        include_last_offset_(include_last_offset),
        padding_idx_(padding_idx),
        has_padding_idx_(padding_idx >= 0) {

        if (num_embeddings <= 0) {
            throw std::invalid_argument("num_embeddings must be positive");
        }
        if (embedding_dim <= 0) {
            throw std::invalid_argument("embedding_dim must be positive");
        }

        // Initialize weight
        Tensor weight = at::empty({num_embeddings, embedding_dim});
        init::normal_(weight, 0.0, 1.0);

        // Zero out padding index if specified
        if (has_padding_idx_) {
            float* data = weight.mutable_data_ptr<float>();
            for (int64_t i = 0; i < embedding_dim_; ++i) {
                data[padding_idx_ * embedding_dim_ + i] = 0.0f;
            }
        }

        register_parameter("weight", Parameter(weight));
    }

    // Forward with 2D input (batch of sequences with same length)
    Tensor forward(const Tensor& input) override {
        if (input.dim() == 1) {
            throw std::runtime_error(
                "EmbeddingBag with 1D input requires offsets tensor. "
                "Use forward_with_offsets() instead."
            );
        }

        if (input.dim() != 2) {
            throw std::runtime_error(
                "EmbeddingBag: Expected 1D or 2D input, got " +
                std::to_string(input.dim()) + "D"
            );
        }

        int64_t batch_size = input.size(0);
        int64_t seq_length = input.size(1);

        return forward_impl(input, batch_size, seq_length, nullptr);
    }

    // Forward with 1D input and offsets
    Tensor forward_with_offsets(
        const Tensor& input,
        const Tensor& offsets,
        const Tensor* per_sample_weights = nullptr
    ) {
        if (input.dim() != 1) {
            throw std::runtime_error("With offsets, input must be 1D");
        }
        if (offsets.dim() != 1) {
            throw std::runtime_error("offsets must be 1D");
        }

        const Tensor& weight = get_parameter("weight")->data();
        const float* indices_data = input.data_ptr<float>();
        const float* offsets_data = offsets.data_ptr<float>();
        const float* weight_data = weight.data_ptr<float>();
        const float* sample_weights = per_sample_weights ?
            per_sample_weights->data_ptr<float>() : nullptr;

        int64_t num_bags = offsets.numel();
        if (include_last_offset_) {
            num_bags -= 1;
        }

        Tensor output = at::zeros({num_bags, embedding_dim_});
        float* output_data = output.mutable_data_ptr<float>();

        int64_t num_indices = input.numel();

        for (int64_t bag = 0; bag < num_bags; ++bag) {
            int64_t start = static_cast<int64_t>(offsets_data[bag]);
            int64_t end;
            if (include_last_offset_) {
                end = static_cast<int64_t>(offsets_data[bag + 1]);
            } else {
                end = (bag + 1 < offsets.numel()) ?
                    static_cast<int64_t>(offsets_data[bag + 1]) : num_indices;
            }

            int64_t bag_size = end - start;
            if (bag_size == 0) continue;

            // Initialize for max mode
            if (mode_ == EmbeddingBagMode::Max) {
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_data[bag * embedding_dim_ + j] = -std::numeric_limits<float>::infinity();
                }
            }

            // Aggregate embeddings
            for (int64_t i = start; i < end; ++i) {
                int64_t idx = static_cast<int64_t>(indices_data[i]);

                if (has_padding_idx_ && idx == padding_idx_) {
                    continue;  // Skip padding index
                }

                if (idx < 0 || idx >= num_embeddings_) {
                    throw std::out_of_range("index out of range");
                }

                float sample_weight = sample_weights ? sample_weights[i] : 1.0f;

                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    float val = weight_data[idx * embedding_dim_ + j] * sample_weight;

                    switch (mode_) {
                        case EmbeddingBagMode::Sum:
                        case EmbeddingBagMode::Mean:
                            output_data[bag * embedding_dim_ + j] += val;
                            break;
                        case EmbeddingBagMode::Max:
                            output_data[bag * embedding_dim_ + j] =
                                std::max(output_data[bag * embedding_dim_ + j], val);
                            break;
                    }
                }
            }

            // Apply mean
            if (mode_ == EmbeddingBagMode::Mean && bag_size > 0) {
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_data[bag * embedding_dim_ + j] /= static_cast<float>(bag_size);
                }
            }
        }

        return output;
    }

    std::string name() const override { return "EmbeddingBag"; }

    int64_t num_embeddings() const { return num_embeddings_; }
    int64_t embedding_dim() const { return embedding_dim_; }
    EmbeddingBagMode mode() const { return mode_; }

private:
    Tensor forward_impl(
        const Tensor& input,
        int64_t batch_size,
        int64_t seq_length,
        const float* per_sample_weights
    ) {
        const Tensor& weight = get_parameter("weight")->data();
        const float* indices_data = input.data_ptr<float>();
        const float* weight_data = weight.data_ptr<float>();

        Tensor output = at::zeros({batch_size, embedding_dim_});
        float* output_data = output.mutable_data_ptr<float>();

        // omp removed for LCC
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t valid_count = 0;

            // Initialize for max mode
            if (mode_ == EmbeddingBagMode::Max) {
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_data[b * embedding_dim_ + j] = -std::numeric_limits<float>::infinity();
                }
            }

            for (int64_t s = 0; s < seq_length; ++s) {
                int64_t idx = static_cast<int64_t>(indices_data[b * seq_length + s]);

                if (has_padding_idx_ && idx == padding_idx_) {
                    continue;
                }

                if (idx < 0 || idx >= num_embeddings_) {
                    throw std::out_of_range("index out of range");
                }

                valid_count++;
                float sample_weight = per_sample_weights ?
                    per_sample_weights[b * seq_length + s] : 1.0f;

                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    float val = weight_data[idx * embedding_dim_ + j] * sample_weight;

                    switch (mode_) {
                        case EmbeddingBagMode::Sum:
                        case EmbeddingBagMode::Mean:
                            output_data[b * embedding_dim_ + j] += val;
                            break;
                        case EmbeddingBagMode::Max:
                            output_data[b * embedding_dim_ + j] =
                                std::max(output_data[b * embedding_dim_ + j], val);
                            break;
                    }
                }
            }

            // Apply mean
            if (mode_ == EmbeddingBagMode::Mean && valid_count > 0) {
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_data[b * embedding_dim_ + j] /= static_cast<float>(valid_count);
                }
            }
        }

        return output;
    }
};

// ============================================================================
// One-Hot Encoding (utility function, not a module)
// ============================================================================

inline Tensor one_hot(const Tensor& indices, int64_t num_classes = -1) {
    // Find max index if num_classes not specified
    if (num_classes < 0) {
        const float* data = indices.data_ptr<float>();
        int64_t max_idx = 0;
        for (int64_t i = 0; i < indices.numel(); ++i) {
            max_idx = std::max(max_idx, static_cast<int64_t>(data[i]));
        }
        num_classes = max_idx + 1;
    }

    // Create output shape: indices_shape + [num_classes]
    std::vector<int64_t> output_sizes = indices.sizes().vec();
    output_sizes.push_back(num_classes);

    Tensor output = at::zeros(output_sizes);
    const float* indices_data = indices.data_ptr<float>();
    float* output_data = output.mutable_data_ptr<float>();

    int64_t num_indices = indices.numel();

    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = static_cast<int64_t>(indices_data[i]);
        if (idx >= 0 && idx < num_classes) {
            output_data[i * num_classes + idx] = 1.0f;
        }
    }

    return output;
}

} // namespace nn
} // namespace torch
