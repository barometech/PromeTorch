#pragma once

// ============================================================================
// PIR (Parallel Information Routing) Module with Autograd
// ============================================================================
// Implementation of PIR architecture with O(T) linear complexity.
// All operations are autograd-tracked for gradient computation.

#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include <cmath>
#include <algorithm>

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDAOps.h"
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif

namespace torch {
namespace nn {

// ============================================================================
// Autograd-enabled helper functions
// ============================================================================

// Sigmoid with autograd
inline Tensor sigmoid_autograd(const Tensor& input) {
    // Use tensor sigmoid which has CUDA dispatch
    Tensor output = input.sigmoid();

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<torch::autograd::SigmoidBackward>(output);
        backward_fn->add_input_metadata(input);
        // ensure_autograd_meta_impl creates AutogradMetaImpl and sets requires_grad
        auto* meta = torch::autograd::ensure_autograd_meta_impl(output);
        meta->grad_fn = backward_fn;
        meta->output_nr_ = 0;
        meta->is_leaf_ = false;
        meta->requires_grad_ = true;  // Set via meta, not set_requires_grad()
    }

    return output;
}

// SiLU with autograd: silu(x) = x * sigmoid(x)
inline Tensor silu_autograd(const Tensor& input) {
    // Use tensor operations with CUDA dispatch
    Tensor sigmoid = input.sigmoid();
    Tensor output = torch::mul(input, sigmoid);

    if (input.requires_grad()) {
        auto backward_fn = std::make_shared<torch::autograd::SiLUBackward>(input, sigmoid);
        backward_fn->add_input_metadata(input);
        auto* meta = torch::autograd::ensure_autograd_meta_impl(output);
        meta->grad_fn = backward_fn;
        meta->output_nr_ = 0;
        meta->is_leaf_ = false;
        meta->requires_grad_ = true;
    }

    return output;
}

// Element-wise multiply with autograd
inline Tensor mul_autograd(const Tensor& self, const Tensor& other) {
    Tensor result = self.mul(other);

    if (self.requires_grad() || other.requires_grad()) {
        auto backward_fn = std::make_shared<torch::autograd::MulTensorBackward>(
            self, other, self.requires_grad(), other.requires_grad()
        );
        if (self.requires_grad()) backward_fn->add_input_metadata(self);
        if (other.requires_grad()) backward_fn->add_input_metadata(other);
        auto* meta = torch::autograd::ensure_autograd_meta_impl(result);
        meta->grad_fn = backward_fn;
        meta->output_nr_ = 0;
        meta->is_leaf_ = false;
        meta->requires_grad_ = true;
    }

    return result;
}

// ============================================================================
// RMSNorm with Autograd
// ============================================================================

class RMSNorm : public Module {
public:
    explicit RMSNorm(int64_t dim, double eps = 1e-6)
        : Module("RMSNorm")
        , dim_(dim)
        , eps_(eps)
    {
        register_parameter("weight", Parameter(at::ones({dim})));
    }

    Tensor forward(const Tensor& input) override {
        auto input_sizes = input.sizes().vec();
        int64_t last_dim = input_sizes.back();
        int64_t outer_size = input.numel() / last_dim;

        // 1. Square input (same shape, element-wise)
        Tensor input_sq = input.mul(input);

        // 2. Sum over last dim -> [outer_size]
        Tensor input_sq_2d = input_sq.view({outer_size, last_dim});
        Tensor sum_sq = input_sq_2d.sum(1, false);

        // 3. Mean and rsqrt
        Tensor mean_sq = sum_sq.mul(at::Scalar(1.0f / static_cast<float>(last_dim)));
        Tensor mean_sq_eps = mean_sq.add(at::Scalar(static_cast<float>(eps_)));
        Tensor inv_rms = mean_sq_eps.rsqrt();  // [outer_size]

        // 4. Normalize: need broadcasting [outer, dim] * [outer, 1]
        Tensor input_2d = input.view({outer_size, last_dim});
        Tensor inv_rms_2d = inv_rms.view({outer_size, 1});
        Tensor normalized = input_2d.mul_broadcast(inv_rms_2d);  // Broadcasting mul

        // 5. Scale by weight: [outer, dim] * [dim]
        Tensor weight = get_parameter("weight")->data();
        Tensor output_2d = normalized.mul_broadcast(weight);
        Tensor output = output_2d.view(input_sizes);

        // Set up autograd
        if (input.requires_grad()) {
            auto backward_fn = std::make_shared<torch::autograd::RMSNormBackward>(
                input, weight, inv_rms, last_dim, eps_
            );
            backward_fn->add_input_metadata(input);
            auto* meta = torch::autograd::ensure_autograd_meta_impl(output);
            meta->grad_fn = backward_fn;
            meta->output_nr_ = 0;
            meta->is_leaf_ = false;
            meta->requires_grad_ = true;
        }

        return output;
    }

    std::string extra_repr() const override {
        return std::to_string(dim_) + ", eps=" + std::to_string(eps_);
    }

private:
    int64_t dim_;
    double eps_;
};

// ============================================================================
// RotaryEmbedding with Autograd
// ============================================================================

class RotaryEmbedding : public Module {
public:
    RotaryEmbedding(int64_t dim, int64_t max_seq_len = 2048, double base = 10000.0)
        : Module("RotaryEmbedding")
        , dim_(dim)
        , max_seq_len_(max_seq_len)
        , base_(base)
    {
        std::vector<float> inv_freq(dim / 2);
        for (int64_t i = 0; i < dim / 2; ++i) {
            inv_freq[i] = 1.0f / std::pow(static_cast<float>(base), 2.0f * i / dim);
        }

        Tensor cos_cache = at::empty({max_seq_len, dim});
        Tensor sin_cache = at::empty({max_seq_len, dim});
        float* cos_data = cos_cache.mutable_data_ptr<float>();
        float* sin_data = sin_cache.mutable_data_ptr<float>();

        for (int64_t pos = 0; pos < max_seq_len; ++pos) {
            for (int64_t i = 0; i < dim / 2; ++i) {
                float freq = pos * inv_freq[i];
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                cos_data[pos * dim + i] = cos_val;
                cos_data[pos * dim + dim / 2 + i] = cos_val;
                sin_data[pos * dim + i] = sin_val;
                sin_data[pos * dim + dim / 2 + i] = sin_val;
            }
        }

        register_buffer("cos_cached", Buffer(cos_cache, false));
        register_buffer("sin_cached", Buffer(sin_cache, false));
    }

    Tensor apply(const Tensor& x, int64_t seq_len, bool batch_first = true) {
        Tensor cos_cache = get_buffer("cos_cached")->data();
        Tensor sin_cache = get_buffer("sin_cached")->data();

        auto sizes = x.sizes().vec();
        int64_t batch_size = batch_first ? sizes[0] : sizes[1];
        int64_t dim = sizes.back();

        Tensor output;

#ifdef PT_USE_CUDA
        if (x.is_cuda()) {
            // Use CUDA kernel - no GPU<->CPU transfer!
            output = at::empty_cuda(sizes);

            // Ensure caches are on GPU
            Tensor cos_cuda = cos_cache.is_cuda() ? cos_cache : at::to_cuda(cos_cache);
            Tensor sin_cuda = sin_cache.is_cuda() ? sin_cache : at::to_cuda(sin_cache);

            at::cuda::launch_rotary_embedding(
                x.data_ptr<float>(),
                cos_cuda.data_ptr<float>(),
                sin_cuda.data_ptr<float>(),
                output.mutable_data_ptr<float>(),
                batch_size, seq_len, dim, dim_,
                nullptr  // default stream
            );
            cudaDeviceSynchronize();

        } else
#endif
        {
            // CPU implementation
            output = at::empty(sizes);
            float* out_data = output.mutable_data_ptr<float>();
            const float* in_data = x.data_ptr<float>();
            const float* cos_data = cos_cache.data_ptr<float>();
            const float* sin_data = sin_cache.data_ptr<float>();

            int64_t half_dim = dim / 2;

            for (int64_t b = 0; b < batch_size; ++b) {
                for (int64_t s = 0; s < seq_len; ++s) {
                    int64_t offset = batch_first ?
                        (b * seq_len + s) * dim :
                        (s * batch_size + b) * dim;

                    for (int64_t i = 0; i < half_dim; ++i) {
                        float x1 = in_data[offset + i];
                        float x2 = in_data[offset + half_dim + i];
                        float cos_val = cos_data[s * dim_ + i];
                        float sin_val = sin_data[s * dim_ + i];

                        out_data[offset + i] = x1 * cos_val - x2 * sin_val;
                        out_data[offset + half_dim + i] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }

        // Set up autograd
        if (x.requires_grad()) {
            auto backward_fn = std::make_shared<torch::autograd::RotaryEmbeddingBackward>(
                cos_cache, sin_cache, seq_len, dim_, batch_first
            );
            backward_fn->add_input_metadata(x);
            auto* meta = torch::autograd::ensure_autograd_meta_impl(output);
            meta->grad_fn = backward_fn;
            meta->output_nr_ = 0;
            meta->is_leaf_ = false;
            meta->requires_grad_ = true;
        }

        return output;
    }

    Tensor forward(const Tensor& input) override {
        int64_t seq_len = input.size(1);
        return apply(input, seq_len, true);
    }

private:
    int64_t dim_;
    int64_t max_seq_len_;
    double base_;
};

// ============================================================================
// Dynamic Parallel Scan with Autograd
// ============================================================================

inline Tensor dynamic_parallel_scan(
    const Tensor& x,
    const Tensor& gate_logits,
    const Tensor& base_decay,
    int64_t segment_size = 64
) {
    auto sizes = x.sizes().vec();
    int64_t B = sizes[0];
    int64_t T = sizes[1];
    int64_t D = sizes[2];

    Tensor output;
    Tensor gates;

#ifdef PT_USE_CUDA
    if (x.is_cuda()) {
        // Use CUDA kernel - no GPU<->CPU transfer!
        output = at::empty_cuda(sizes);
        gates = at::empty_cuda(sizes);

        // Ensure base_decay is on GPU
        Tensor decay_cuda = base_decay.is_cuda() ? base_decay : at::to_cuda(base_decay);

        at::cuda::launch_parallel_scan(
            x.data_ptr<float>(),
            gate_logits.data_ptr<float>(),
            decay_cuda.data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            gates.mutable_data_ptr<float>(),
            B, T, D,
            nullptr  // default stream
        );
        cudaDeviceSynchronize();  // Ensure kernel completes

    } else
#endif
    {
        // CPU implementation
        const float* x_data = x.data_ptr<float>();
        const float* gate_data = gate_logits.data_ptr<float>();
        const float* decay_data = base_decay.data_ptr<float>();

        output = at::zeros(sizes);
        gates = at::empty(sizes);
        float* out_data = output.mutable_data_ptr<float>();
        float* gates_data = gates.mutable_data_ptr<float>();

        #pragma omp parallel for if(B > 4)
        for (int64_t b = 0; b < B; ++b) {
            std::vector<float> h(D, 0.0f);

            for (int64_t t = 0; t < T; ++t) {
                int64_t offset = (b * T + t) * D;

                for (int64_t d = 0; d < D; ++d) {
                    float gate_logit = gate_data[offset + d];
                    float modulation = std::tanh(gate_logit) * 0.1f;
                    float gate = decay_data[d] * (1.0f + modulation);
                    gate = std::max(0.5f, std::min(0.999f, gate));
                    gates_data[offset + d] = gate;

                    h[d] = gate * h[d] + x_data[offset + d];
                    out_data[offset + d] = h[d];
                }
            }
        }
    }

    // Set up autograd
    if (x.requires_grad() || gate_logits.requires_grad()) {
        auto backward_fn = std::make_shared<torch::autograd::ParallelScanBackward>(
            x, gates, gate_logits, base_decay, output,
            x.requires_grad(), gate_logits.requires_grad()
        );
        if (x.requires_grad()) backward_fn->add_input_metadata(x);
        if (gate_logits.requires_grad()) backward_fn->add_input_metadata(gate_logits);
        auto* meta = torch::autograd::ensure_autograd_meta_impl(output);
        meta->grad_fn = backward_fn;
        meta->output_nr_ = 0;
        meta->is_leaf_ = false;
        meta->requires_grad_ = true;
    }

    return output;
}

// ============================================================================
// PIRLayer with Autograd
// ============================================================================

class PIRLayer : public Module {
public:
    PIRLayer(int64_t n_embd, double decay_min, double decay_max)
        : Module("PIRLayer")
        , n_embd_(n_embd)
    {
        Tensor decay = at::empty({n_embd});
        float* decay_data = decay.mutable_data_ptr<float>();
        for (int64_t i = 0; i < n_embd; ++i) {
            decay_data[i] = static_cast<float>(
                decay_min + (decay_max - decay_min) * i / (n_embd - 1)
            );
        }
        register_buffer("base_decay", Buffer(decay, false));

        gate_proj_ = std::make_shared<Linear>(n_embd, n_embd, false);
        value_proj_ = std::make_shared<Linear>(n_embd, n_embd, false);
        out_proj_ = std::make_shared<Linear>(n_embd, n_embd, false);
        norm_ = std::make_shared<RMSNorm>(n_embd);

        register_module("gate_proj", gate_proj_);
        register_module("value_proj", value_proj_);
        register_module("out_proj", out_proj_);
        register_module("norm", norm_);

        init_weights();
    }

    void init_weights() {
        init_orthogonal(gate_proj_->get_parameter("weight")->data(), 0.1f);
        init_orthogonal(value_proj_->get_parameter("weight")->data(), 1.0f);
        init_orthogonal(out_proj_->get_parameter("weight")->data(), 0.5f);
    }

    void init_orthogonal(Tensor& weight, float gain) {
        float* data = weight.mutable_data_ptr<float>();
        int64_t rows = weight.size(0);
        int64_t cols = weight.size(1);
        float scale = gain * std::sqrt(2.0f / (rows + cols));

        for (int64_t i = 0; i < weight.numel(); ++i) {
            data[i] = (2.0f * static_cast<float>(::rand()) / RAND_MAX - 1.0f) * scale;
        }
    }

    Tensor forward(const Tensor& x) override {
        Tensor gate_logits = gate_proj_->forward(x);
        Tensor values = value_proj_->forward(x);

        // Content-adaptive gating with autograd
        Tensor value_gate = sigmoid_autograd(gate_logits);
        Tensor gated_values = mul_autograd(values, value_gate);

        // Parallel scan with autograd
        Tensor base_decay = get_buffer("base_decay")->data();
        Tensor scanned = dynamic_parallel_scan(gated_values, gate_logits, base_decay);

        // Output
        Tensor out = out_proj_->forward(scanned);
        return norm_->forward(out);
    }

private:
    int64_t n_embd_;
    std::shared_ptr<Linear> gate_proj_;
    std::shared_ptr<Linear> value_proj_;
    std::shared_ptr<Linear> out_proj_;
    std::shared_ptr<RMSNorm> norm_;
};

// ============================================================================
// PIRBlock with Autograd
// ============================================================================

class PIRBlock : public Module {
public:
    PIRBlock(int64_t n_embd, int64_t n_pir_layers = 3)
        : Module("PIRBlock")
        , n_embd_(n_embd)
    {
        std::vector<std::pair<double, double>> decay_ranges = {
            {0.80, 0.92},
            {0.88, 0.96},
            {0.94, 0.995}
        };

        for (int64_t i = 0; i < n_pir_layers; ++i) {
            auto& range = decay_ranges[i % decay_ranges.size()];
            auto layer = std::make_shared<PIRLayer>(n_embd, range.first, range.second);
            layers_.push_back(layer);
            register_module("layer_" + std::to_string(i), layer);
        }

        mix_proj_ = std::make_shared<Linear>(n_embd, n_embd, false);
        norm_ = std::make_shared<RMSNorm>(n_embd);
        register_module("mix_proj", mix_proj_);
        register_module("norm", norm_);

        init_orthogonal(mix_proj_->get_parameter("weight")->data(), 0.5f);
    }

    void init_orthogonal(Tensor& weight, float gain) {
        float* data = weight.mutable_data_ptr<float>();
        int64_t rows = weight.size(0);
        int64_t cols = weight.size(1);
        float scale = gain * std::sqrt(2.0f / (rows + cols));

        for (int64_t i = 0; i < weight.numel(); ++i) {
            data[i] = (2.0f * static_cast<float>(::rand()) / RAND_MAX - 1.0f) * scale;
        }
    }

    Tensor forward(const Tensor& x) override {
        Tensor h = x;
        for (auto& layer : layers_) {
            // Use autograd add
            h = torch::autograd::add_autograd(h, layer->forward(h));
        }

        Tensor out = mix_proj_->forward(h);
        return norm_->forward(out);
    }

private:
    int64_t n_embd_;
    std::vector<std::shared_ptr<PIRLayer>> layers_;
    std::shared_ptr<Linear> mix_proj_;
    std::shared_ptr<RMSNorm> norm_;
};

// ============================================================================
// SwiGLU FeedForward with Autograd
// ============================================================================

class SwiGLUFeedForward : public Module {
public:
    SwiGLUFeedForward(int64_t n_embd, int64_t hidden)
        : Module("SwiGLUFeedForward")
        , n_embd_(n_embd)
        , hidden_(hidden)
    {
        w1_ = std::make_shared<Linear>(n_embd, hidden, false);
        w2_ = std::make_shared<Linear>(hidden, n_embd, false);
        w3_ = std::make_shared<Linear>(n_embd, hidden, false);

        register_module("w1", w1_);
        register_module("w2", w2_);
        register_module("w3", w3_);

        init_weights();
    }

    void init_weights() {
        init_orthogonal(w1_->get_parameter("weight")->data(), 1.0f);
        init_orthogonal(w2_->get_parameter("weight")->data(), 0.5f);
        init_orthogonal(w3_->get_parameter("weight")->data(), 1.0f);
    }

    void init_orthogonal(Tensor& weight, float gain) {
        float* data = weight.mutable_data_ptr<float>();
        int64_t rows = weight.size(0);
        int64_t cols = weight.size(1);
        float scale = gain * std::sqrt(2.0f / (rows + cols));

        for (int64_t i = 0; i < weight.numel(); ++i) {
            data[i] = (2.0f * static_cast<float>(::rand()) / RAND_MAX - 1.0f) * scale;
        }
    }

    Tensor forward(const Tensor& x) override {
        Tensor gate = w1_->forward(x);
        Tensor up = w3_->forward(x);

        // SiLU with autograd
        gate = silu_autograd(gate);

        // Element-wise multiply with autograd
        Tensor hidden = mul_autograd(gate, up);

        return w2_->forward(hidden);
    }

private:
    int64_t n_embd_;
    int64_t hidden_;
    std::shared_ptr<Linear> w1_;
    std::shared_ptr<Linear> w2_;
    std::shared_ptr<Linear> w3_;
};

// ============================================================================
// PIR Transformer Block with Autograd
// ============================================================================

class PIRTransformerBlock : public Module {
public:
    PIRTransformerBlock(int64_t n_embd, int64_t ffn_hidden, int64_t n_pir_layers = 3, double dropout = 0.0)
        : Module("PIRTransformerBlock")
    {
        pir_ = std::make_shared<PIRBlock>(n_embd, n_pir_layers);
        ffn_ = std::make_shared<SwiGLUFeedForward>(n_embd, ffn_hidden);
        norm1_ = std::make_shared<RMSNorm>(n_embd);
        norm2_ = std::make_shared<RMSNorm>(n_embd);

        register_module("pir", pir_);
        register_module("ffn", ffn_);
        register_module("norm1", norm1_);
        register_module("norm2", norm2_);
    }

    Tensor forward(const Tensor& x) override {
        std::cout << "BLK: norm1" << std::endl; std::cout.flush();
        Tensor x_norm = norm1_->forward(x);
        std::cout << "BLK: pir" << std::endl; std::cout.flush();
        Tensor pir_out = pir_->forward(x_norm);
        std::cout << "BLK: add1" << std::endl; std::cout.flush();
        Tensor h = torch::autograd::add_autograd(x, pir_out);
        std::cout << "BLK: norm2" << std::endl; std::cout.flush();
        Tensor h_norm = norm2_->forward(h);
        std::cout << "BLK: ffn" << std::endl; std::cout.flush();
        Tensor ffn_out = ffn_->forward(h_norm);
        std::cout << "BLK: add2" << std::endl; std::cout.flush();
        h = torch::autograd::add_autograd(h, ffn_out);
        std::cout << "BLK: done" << std::endl; std::cout.flush();
        return h;
    }

private:
    std::shared_ptr<PIRBlock> pir_;
    std::shared_ptr<SwiGLUFeedForward> ffn_;
    std::shared_ptr<RMSNorm> norm1_;
    std::shared_ptr<RMSNorm> norm2_;
};

} // namespace nn
} // namespace torch
