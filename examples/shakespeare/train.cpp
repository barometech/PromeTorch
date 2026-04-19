// ============================================================================
// Shakespeare Character-Level Language Model Training
// ============================================================================
// This example trains a small Transformer model on Shakespeare text for
// character-level text generation.
//
// Usage:
//   ./shakespeare_train [path_to_shakespeare.txt]
//
// If no path is provided, uses a small built-in sample.

#include "model.h"
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

using namespace shakespeare;
using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

// Default Shakespeare sample (from Hamlet)
const char* DEFAULT_TEXT = R"(
HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take Arms against a Sea of troubles,
And by opposing end them: to die, to sleep;
No more; and by a sleep, to say we end
The heart-ache, and the thousand natural shocks
That Flesh is heir to? 'Tis a consummation
Devoutly to be wished. To die, to sleep,
To sleep, perchance to Dream; aye, there's the rub,
For in that sleep of death, what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes Calamity of so long life:
For who would bear the Whips and Scorns of time,
The Oppressor's wrong, the proud man's Contumely,
The pangs of despised Love, the Law's delay,
The insolence of Office, and the Spurns
That patient merit of the unworthy takes,
When he himself might his Quietus make
With a bare Bodkin? Who would Fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered Country, from whose bourn
No Traveller returns, Puzzles the will,
And makes us rather bear those ills we have,
Than fly to others that we know not of.
Thus Conscience does make Cowards of us all,
And thus the Native hue of Resolution
Is sicklied o'er, with the pale cast of Thought,
And enterprises of great pitch and moment,
With this regard their Currents turn awry,
And lose the name of Action. Soft you now,
The fair Ophelia? Nymph, in thy Orisons
Be all my sins remembered.

KING CLAUDIUS: How fares our cousin Hamlet?

HAMLET: Excellent, i' faith; of the chameleon's dish: I eat
the air, promise-crammed: you cannot feed capons so.

KING CLAUDIUS: I have nothing with this answer, Hamlet; these words
are not mine.

HAMLET: No, nor mine now.
)";

// Cross-entropy loss for language modeling
float cross_entropy_loss(const Tensor& logits, const Tensor& targets, int64_t vocab_size) {
    // logits: [seq_len, batch, vocab_size]
    // targets: [seq_len, batch]
    int64_t seq_len = logits.size(0);
    int64_t batch_size = logits.size(1);
    int64_t total_tokens = seq_len * batch_size;

    const float* logits_data = logits.data_ptr<float>();
    const float* targets_data = targets.data_ptr<float>();

    float loss = 0.0f;

    for (int64_t s = 0; s < seq_len; ++s) {
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t target_idx = static_cast<int64_t>(targets_data[s * batch_size + b]);
            int64_t offset = (s * batch_size + b) * vocab_size;

            // Log-softmax for numerical stability
            float max_logit = logits_data[offset];
            for (int64_t v = 1; v < vocab_size; ++v) {
                max_logit = std::max(max_logit, logits_data[offset + v]);
            }

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < vocab_size; ++v) {
                sum_exp += std::exp(logits_data[offset + v] - max_logit);
            }

            float log_softmax = logits_data[offset + target_idx] - max_logit - std::log(sum_exp);
            loss -= log_softmax;
        }
    }

    return loss / static_cast<float>(total_tokens);
}

// Compute cross-entropy loss with gradient
Tensor cross_entropy_loss_with_grad(const Tensor& logits, const Tensor& targets, int64_t vocab_size) {
    // logits: [seq_len, batch, vocab_size]
    // targets: [seq_len, batch]
    int64_t seq_len = logits.size(0);
    int64_t batch_size = logits.size(1);
    int64_t total_tokens = seq_len * batch_size;

    // For now, we compute the loss manually and store gradient
    Tensor loss_tensor = at::zeros({});
    float* loss_data = loss_tensor.mutable_data_ptr<float>();

    const float* logits_data = logits.data_ptr<float>();
    const float* targets_data = targets.data_ptr<float>();

    // Compute softmax and loss
    Tensor grad = at::zeros(logits.sizes());
    float* grad_data = grad.mutable_data_ptr<float>();

    float total_loss = 0.0f;

    for (int64_t s = 0; s < seq_len; ++s) {
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t target_idx = static_cast<int64_t>(targets_data[s * batch_size + b]);
            int64_t offset = (s * batch_size + b) * vocab_size;

            // Compute softmax
            float max_logit = logits_data[offset];
            for (int64_t v = 1; v < vocab_size; ++v) {
                max_logit = std::max(max_logit, logits_data[offset + v]);
            }

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < vocab_size; ++v) {
                float exp_val = std::exp(logits_data[offset + v] - max_logit);
                grad_data[offset + v] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize to get softmax
            for (int64_t v = 0; v < vocab_size; ++v) {
                grad_data[offset + v] /= sum_exp;
            }

            // Cross-entropy gradient: softmax - one_hot(target)
            // grad = p - y where p is softmax, y is one-hot
            grad_data[offset + target_idx] -= 1.0f;

            // Scale by 1/total_tokens for mean reduction
            for (int64_t v = 0; v < vocab_size; ++v) {
                grad_data[offset + v] /= static_cast<float>(total_tokens);
            }

            // Compute loss
            float log_softmax = logits_data[offset + target_idx] - max_logit - std::log(sum_exp);
            total_loss -= log_softmax;
        }
    }

    loss_data[0] = total_loss / static_cast<float>(total_tokens);

    // Store gradient on the tensor via the public setter.
    const_cast<Tensor&>(logits).set_grad(grad);

    return loss_tensor;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "PromeTorch Shakespeare Language Model" << std::endl;
    std::cout << "========================================" << std::endl;

    // Load text
    std::string text;
    if (argc > 1) {
        std::ifstream file(argv[1]);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            text = buffer.str();
            std::cout << "Loaded text from: " << argv[1] << std::endl;
        } else {
            std::cerr << "Could not open file: " << argv[1] << std::endl;
            return 1;
        }
    } else {
        text = DEFAULT_TEXT;
        std::cout << "Using built-in Shakespeare sample" << std::endl;
    }

    std::cout << "Text length: " << text.size() << " characters" << std::endl;

    // Build tokenizer
    CharTokenizer tokenizer;
    tokenizer.build_vocab(text);
    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << " characters" << std::endl;

    // Tokenize text
    std::vector<int64_t> tokens = tokenizer.encode(text);
    std::cout << "Total tokens: " << tokens.size() << std::endl;

    // Model hyperparameters
    int64_t vocab_size = tokenizer.vocab_size();
    int64_t d_model = 128;      // Embedding dimension
    int64_t nhead = 4;          // Number of attention heads
    int64_t num_layers = 3;     // Number of transformer layers
    int64_t dim_feedforward = 256;  // FFN dimension
    double dropout = 0.1;
    int64_t block_size = 64;    // Context window size
    int64_t batch_size = 8;
    int64_t max_iters = 500;
    double learning_rate = 3e-4;

    std::cout << "\nModel configuration:" << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  nhead: " << nhead << std::endl;
    std::cout << "  num_layers: " << num_layers << std::endl;
    std::cout << "  dim_feedforward: " << dim_feedforward << std::endl;
    std::cout << "  block_size: " << block_size << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;

    // Create model
    auto model = std::make_shared<TransformerLM>(
        vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, block_size
    );

    std::cout << "\nModel parameters: " << count_parameters(*model) << std::endl;

    // Create dataset
    TextDataset dataset(tokens, block_size);
    std::cout << "Training examples: " << dataset.size() << std::endl;

    // Create optimizer
    Adam optimizer(model->parameters(), learning_rate);

    // Training loop
    std::cout << "\nStarting training..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int64_t iter = 0; iter < max_iters; ++iter) {
        model->train();

        // Get batch
        auto [input, target] = dataset.get_batch(batch_size, gen);

        // Forward pass
        Tensor logits = model->forward(input);

        // Compute loss
        float loss = cross_entropy_loss(logits, target, vocab_size);

        // Print progress
        if (iter % 50 == 0 || iter == max_iters - 1) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time
            ).count();

            std::cout << "Iter " << std::setw(4) << iter
                      << " | Loss: " << std::fixed << std::setprecision(4) << loss
                      << " | Time: " << duration << "s" << std::endl;
        }

        // Zero gradients
        optimizer.zero_grad();

        // Backward pass (simplified - manual gradient computation for output)
        cross_entropy_loss_with_grad(logits, target, vocab_size);

        // Update parameters
        optimizer.step();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time
    ).count();

    std::cout << "\nTraining completed in " << total_duration << " seconds" << std::endl;

    // Generate sample text
    std::cout << "\n========================================" << std::endl;
    std::cout << "Generated Text:" << std::endl;
    std::cout << "========================================" << std::endl;

    // Start with "HAMLET:"
    std::string prompt = "HAMLET:";
    std::vector<int64_t> prompt_tokens = tokenizer.encode(prompt);

    model->eval();
    std::vector<int64_t> generated = model->generate(
        prompt_tokens,
        200,   // Generate 200 characters
        0.8,   // Temperature
        true   // Sample
    );

    std::string generated_text = tokenizer.decode(generated);
    std::cout << generated_text << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Done!" << std::endl;

    return 0;
}
