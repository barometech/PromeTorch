// ============================================================================
// PIR 270M Tests
// ============================================================================

#include "torch/nn/nn.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace torch;
using namespace torch::nn;

// ============================================================================
// RMSNorm Tests
// ============================================================================

TEST(RMSNormTest, BasicForward) {
    RMSNorm norm(64);

    Tensor input = at::randn({2, 10, 64});
    Tensor output = norm.forward(input);

    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 10);
    EXPECT_EQ(output.size(2), 64);
}

TEST(RMSNormTest, OutputMagnitude) {
    RMSNorm norm(32);

    // Create input with known properties
    Tensor input = at::ones({1, 1, 32});
    Tensor output = norm.forward(input);

    // RMS of all ones is 1, so output should be close to weight (all ones)
    const float* data = output.data_ptr<float>();
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(data[i], 1.0f, 0.1f);
    }
}

// ============================================================================
// RotaryEmbedding Tests
// ============================================================================

TEST(RotaryEmbeddingTest, Construction) {
    RotaryEmbedding rope(64, 512, 10000.0);

    // Check that buffers are created
    auto* cos_buf = rope.get_buffer("cos_cached");
    auto* sin_buf = rope.get_buffer("sin_cached");

    EXPECT_TRUE(cos_buf != nullptr);
    EXPECT_TRUE(sin_buf != nullptr);
}

TEST(RotaryEmbeddingTest, Apply) {
    RotaryEmbedding rope(64, 128, 10000.0);

    Tensor input = at::randn({2, 16, 64});  // [batch, seq, dim]
    Tensor output = rope.apply(input, 16, true);

    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 16);
    EXPECT_EQ(output.size(2), 64);
}

// ============================================================================
// PIRLayer Tests
// ============================================================================

TEST(PIRLayerTest, Forward) {
    PIRLayer layer(128, 0.8, 0.95);

    Tensor input = at::randn({2, 32, 128});  // [batch, seq, dim]
    Tensor output = layer.forward(input);

    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 32);
    EXPECT_EQ(output.size(2), 128);
}

// ============================================================================
// PIRBlock Tests
// ============================================================================

TEST(PIRBlockTest, Forward) {
    PIRBlock block(128, 3);

    Tensor input = at::randn({2, 32, 128});
    Tensor output = block.forward(input);

    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 32);
    EXPECT_EQ(output.size(2), 128);
}

// ============================================================================
// SwiGLU Tests
// ============================================================================

TEST(SwiGLUTest, Forward) {
    SwiGLUFeedForward ffn(128, 256);

    Tensor input = at::randn({2, 16, 128});
    Tensor output = ffn.forward(input);

    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 16);
    EXPECT_EQ(output.size(2), 128);
}

// ============================================================================
// PIRTransformerBlock Tests
// ============================================================================

TEST(PIRTransformerBlockTest, Forward) {
    PIRTransformerBlock block(128, 256, 3, 0.0);

    Tensor input = at::randn({2, 32, 128});
    Tensor output = block.forward(input);

    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 32);
    EXPECT_EQ(output.size(2), 128);
}

// ============================================================================
// PIR270M Config Tests
// ============================================================================

TEST(PIR270MConfigTest, DefaultValues) {
    PIR270MConfig config;

    EXPECT_EQ(config.vocab_size, 50257);
    EXPECT_EQ(config.n_embd, 768);
    EXPECT_EQ(config.n_layers, 22);
    EXPECT_EQ(config.n_pir_layers, 3);
    EXPECT_EQ(config.block_size, 2048);
}

TEST(PIR270MConfigTest, FFNHidden) {
    PIR270MConfig config;
    config.n_embd = 768;
    config.ffn_mult = 4.0;

    int64_t ffn_hidden = config.ffn_hidden();

    // Should be ~2048 (768 * 4 * 2/3), rounded to 64
    EXPECT_GT(ffn_hidden, 1500);
    EXPECT_LT(ffn_hidden, 2500);
    EXPECT_EQ(ffn_hidden % 64, 0);  // Should be multiple of 64
}

// ============================================================================
// Small PIR Model Test (reduced size for testing)
// ============================================================================

TEST(PIR270MTest, SmallModelForward) {
    // Create a small model for testing
    PIR270MConfig config;
    config.vocab_size = 1000;
    config.n_embd = 64;
    config.n_layers = 2;
    config.n_pir_layers = 2;
    config.block_size = 64;

    PIR270M model(config);

    // Create input: [batch=2, seq=16] of token indices
    Tensor input = at::empty({2, 16});
    float* data = input.mutable_data_ptr<float>();
    for (int i = 0; i < 32; ++i) {
        data[i] = static_cast<float>(rand() % 1000);
    }

    // Forward pass
    auto [logits, loss] = model.forward_with_loss(input);

    EXPECT_EQ(logits.size(0), 2);
    EXPECT_EQ(logits.size(1), 16);
    EXPECT_EQ(logits.size(2), 1000);
}

TEST(PIR270MTest, SmallModelWithTargets) {
    PIR270MConfig config;
    config.vocab_size = 1000;
    config.n_embd = 64;
    config.n_layers = 2;
    config.n_pir_layers = 2;
    config.block_size = 64;

    PIR270M model(config);

    // Create input and targets
    Tensor input = at::empty({2, 16});
    Tensor targets = at::empty({2, 16});
    float* in_data = input.mutable_data_ptr<float>();
    float* tgt_data = targets.mutable_data_ptr<float>();

    for (int i = 0; i < 32; ++i) {
        in_data[i] = static_cast<float>(rand() % 1000);
        tgt_data[i] = static_cast<float>(rand() % 1000);
    }

    // Forward with loss
    auto [logits, loss] = model.forward_with_loss(input, targets);

    EXPECT_TRUE(loss.defined());
    float loss_value = loss.data_ptr<float>()[0];
    EXPECT_GT(loss_value, 0.0f);  // Loss should be positive
    EXPECT_LT(loss_value, 20.0f); // Loss shouldn't be too high
}

TEST(PIR270MTest, Generate) {
    PIR270MConfig config;
    config.vocab_size = 100;
    config.n_embd = 32;
    config.n_layers = 1;
    config.n_pir_layers = 1;
    config.block_size = 32;

    PIR270M model(config);

    std::vector<int64_t> prompt = {1, 2, 3};
    std::vector<int64_t> generated = model.generate(
        prompt,
        5,      // Generate 5 tokens
        1.0,    // Temperature
        10,     // Top-k
        0.9,    // Top-p
        99,     // EOS token (high so it won't trigger)
        1.0     // No repetition penalty
    );

    EXPECT_GE(generated.size(), 3);  // At least the prompt
    EXPECT_LE(generated.size(), 8);  // Prompt + max 5 new tokens
}

// ============================================================================
// Parameter Count Test
// ============================================================================

TEST(PIR270MTest, ParameterCount) {
    PIR270MConfig config;
    config.vocab_size = 50257;
    config.n_embd = 768;
    config.n_layers = 22;
    config.n_pir_layers = 3;

    PIR270M model(config);

    int64_t n_params = model.count_params();

    // Should be around 270M parameters
    EXPECT_GT(n_params, 200e6);
    EXPECT_LT(n_params, 350e6);

    std::cout << "PIR 270M parameter count: " << n_params / 1e6 << "M" << std::endl;
}
