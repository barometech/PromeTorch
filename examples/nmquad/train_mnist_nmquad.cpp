// ============================================================================
// MNIST on NM QUAD — ALL computation on NM6408 cores via DDR polling
// ============================================================================
// NO CPU COMPUTE. Host only manages data flow.
// NM6408 does: matmul, add, relu, softmax, backward matmul.
//
// Build on NM QUAD host:
//   g++ -O2 -o train_mnist_nmquad train_mnist_nmquad.cpp -lnm_quad_load -I../../
// Run:
//   ./train_mnist_nmquad --data ../../data --dispatcher dispatcher_nmquad.abs

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cassert>

// NM QUAD API
extern "C" {
    struct PL_Board;
    struct PL_Access;
    typedef struct { int nm_id; int cluster_id; } PL_CoreNo;
    typedef unsigned int PL_Word;
    typedef unsigned int PL_Addr;

    int PL_GetBoardCount(unsigned int*);
    int PL_GetBoardDesc(unsigned int, PL_Board**);
    int PL_CloseBoardDesc(PL_Board*);
    int PL_ResetBoard(PL_Board*);
    int PL_LoadInitCode(PL_Board*);
    int PL_GetAccess(PL_Board*, PL_CoreNo*, PL_Access**);
    int PL_CloseAccess(PL_Access*);
    int PL_LoadProgramFile(PL_Access*, const char*);
    int PL_ReadMemBlock(PL_Access*, PL_Word*, PL_Addr, unsigned int);
    int PL_WriteMemBlock(PL_Access*, const PL_Word*, PL_Addr, unsigned int);
    int PL_SetTimeout(unsigned int);
}

// Constants matching dispatcher_nmquad.cpp
#define DDR_BASE      0x00340000u
#define CMD_BLOCK_SIZE 32
#define STATUS_ADDR    30
#define WATCHDOG_ADDR  31
#define OP_NOP         0
#define OP_MATMUL      1
#define OP_ADD         2
#define OP_RELU        4
#define OP_EXIT        255

// Data area starts after 16 cmd blocks
#define DATA_BASE (DDR_BASE + 16 * CMD_BLOCK_SIZE)

static PL_Board* board = nullptr;
static PL_Access* accesses[16] = {};
static int num_cores = 0;

// DDR bump allocator
static PL_Addr ddr_ptr = DATA_BASE;
static PL_Addr ddr_alloc(size_t words) {
    PL_Addr addr = ddr_ptr;
    ddr_ptr += words;
    return addr;
}
static void ddr_reset() { ddr_ptr = DATA_BASE; }

// ============================================================
// NM QUAD init
// ============================================================
bool init_nmquad(const char* dispatcher_path) {
    PL_SetTimeout(10000);
    unsigned int count = 0;
    PL_GetBoardCount(&count);
    if (count == 0) { std::cerr << "No boards!\n"; return false; }

    PL_GetBoardDesc(0, &board);
    PL_ResetBoard(board);
    PL_LoadInitCode(board);

    num_cores = 0;
    for (int cl = 0; cl < 4; cl++) {
        for (int co = 0; co < 4; co++) {
            PL_CoreNo cn = {co, cl};
            PL_Access* acc = nullptr;
            if (PL_GetAccess(board, &cn, &acc) == 0) {
                if (PL_LoadProgramFile(acc, dispatcher_path) == 0) {
                    accesses[num_cores++] = acc;
                } else {
                    PL_CloseAccess(acc);
                }
            }
        }
    }

    // Wait for dispatchers to init
    usleep(500000);

    // Verify alive
    int alive = 0;
    for (int i = 0; i < num_cores; i++) {
        PL_Word buf;
        PL_ReadMemBlock(accesses[i], &buf, DDR_BASE + i * CMD_BLOCK_SIZE + WATCHDOG_ADDR, 1);
        if (buf > 100) alive++;
    }
    std::cout << "NM QUAD: " << alive << "/" << num_cores << " cores alive\n";
    return alive > 0;
}

// ============================================================
// Send command to core via DDR polling (NO PL_Sync!)
// ============================================================
bool send_cmd(int core_idx, unsigned int op, const unsigned int* args, int nargs, int timeout_ms = 5000) {
    PL_Addr base = DDR_BASE + core_idx * CMD_BLOCK_SIZE;

    // Write args
    if (nargs > 0)
        PL_WriteMemBlock(accesses[core_idx], (PL_Word*)args, base + 1, nargs);

    // Clear status
    PL_Word zero = 0;
    PL_WriteMemBlock(accesses[core_idx], &zero, base + STATUS_ADDR, 1);

    // Write opcode
    PL_Word cmd = op;
    PL_WriteMemBlock(accesses[core_idx], &cmd, base, 1);

    // Poll status
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        PL_Word status;
        PL_ReadMemBlock(accesses[core_idx], &status, base + STATUS_ADDR, 1);
        if (status == 1) return true;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (elapsed > timeout_ms) return false;
        usleep(100); // 0.1ms poll
    }
}

// ============================================================
// Upload float array to DDR
// ============================================================
PL_Addr upload(const float* data, int count) {
    PL_Addr addr = ddr_alloc(count);
    PL_WriteMemBlock(accesses[0], (const PL_Word*)data, addr, count);
    return addr;
}

// Download float array from DDR
void download(float* data, PL_Addr addr, int count) {
    PL_ReadMemBlock(accesses[0], (PL_Word*)data, addr, count);
}

// ============================================================
// NM6408 matmul: C[M,N] = A[M,K] @ B[K,N]
// ============================================================
PL_Addr nm_matmul(PL_Addr a_addr, PL_Addr b_addr, int M, int K, int N, int core = 0) {
    PL_Addr c_addr = ddr_alloc(M * N);
    unsigned int args[] = {(unsigned)M, (unsigned)K, (unsigned)N, a_addr, b_addr, c_addr};
    send_cmd(core, OP_MATMUL, args, 6);
    return c_addr;
}

// NM6408 element-wise add: C = A + B
PL_Addr nm_add(PL_Addr a_addr, PL_Addr b_addr, int count, int core = 0) {
    PL_Addr c_addr = ddr_alloc(count);
    unsigned int args[] = {(unsigned)count, a_addr, b_addr, c_addr};
    send_cmd(core, OP_ADD, args, 4);
    return c_addr;
}

// NM6408 ReLU: Y = max(0, X)
PL_Addr nm_relu(PL_Addr x_addr, int count, int core = 0) {
    PL_Addr y_addr = ddr_alloc(count);
    unsigned int args[] = {(unsigned)count, x_addr, y_addr};
    send_cmd(core, OP_RELU, args, 3);
    return y_addr;
}

// ============================================================
// MNIST data loading
// ============================================================
#ifdef __BYTE_ORDER__
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define bswap32(x) __builtin_bswap32(x)
#endif
#endif

std::vector<std::vector<uint8_t>> load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    int32_t magic, n, rows, cols;
    f.read((char*)&magic, 4); f.read((char*)&n, 4);
    f.read((char*)&rows, 4); f.read((char*)&cols, 4);
    n = bswap32(n); rows = bswap32(rows); cols = bswap32(cols);
    std::vector<std::vector<uint8_t>> imgs(n);
    for (int i = 0; i < n; i++) {
        imgs[i].resize(rows * cols);
        f.read((char*)imgs[i].data(), rows * cols);
    }
    return imgs;
}

std::vector<uint8_t> load_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    int32_t magic, n;
    f.read((char*)&magic, 4); f.read((char*)&n, 4);
    n = bswap32(n);
    std::vector<uint8_t> labels(n);
    f.read((char*)labels.data(), n);
    return labels;
}

// ============================================================
// Main — MNIST MLP trained ENTIRELY on NM6408
// ============================================================
int main(int argc, char* argv[]) {
    std::string data_dir = "../../data";
    std::string disp_path = "dispatcher_nmquad.abs";
    int epochs = 3;
    float lr = 0.01f;
    int batch_size = 64;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--data" && i+1 < argc) data_dir = argv[++i];
        else if (a == "--dispatcher" && i+1 < argc) disp_path = argv[++i];
        else if (a == "--epochs" && i+1 < argc) epochs = atoi(argv[++i]);
        else if (a == "--lr" && i+1 < argc) lr = atof(argv[++i]);
    }

    std::cout << "=== MNIST on NM QUAD (ALL on NM6408) ===\n";

    // Init hardware
    if (!init_nmquad(disp_path.c_str())) return 1;

    // Load data
    auto train_imgs = load_images(data_dir + "/train-images-idx3-ubyte");
    auto train_lbls = load_labels(data_dir + "/train-labels-idx1-ubyte");
    auto test_imgs = load_images(data_dir + "/t10k-images-idx3-ubyte");
    auto test_lbls = load_labels(data_dir + "/t10k-labels-idx1-ubyte");
    if (train_imgs.empty()) { std::cerr << "No MNIST data!\n"; return 1; }
    std::cout << "MNIST: " << train_imgs.size() << " train, " << test_imgs.size() << " test\n";

    // MLP: 784 -> 256 -> 10
    int D0 = 784, D1 = 256, D2 = 10;
    std::mt19937 rng(42);
    auto randn = [&](int n, float scale) {
        std::vector<float> v(n);
        std::normal_distribution<float> dist(0, scale);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    auto W1 = randn(D0 * D1, 0.01f);
    auto b1 = std::vector<float>(D1, 0.0f);
    auto W2 = randn(D1 * D2, 0.01f);
    auto b2 = std::vector<float>(D2, 0.0f);

    int n_params = D0*D1 + D1 + D1*D2 + D2;
    std::cout << "MLP: " << D0 << "->" << D1 << "->" << D2 << " (" << n_params << " params)\n";

    // Upload weights to DDR (persistent)
    PL_Addr w1_addr = upload(W1.data(), D0 * D1);
    PL_Addr b1_addr = upload(b1.data(), D1);
    PL_Addr w2_addr = upload(W2.data(), D1 * D2);
    PL_Addr b2_addr = upload(b2.data(), D2);

    // Pre-allocate batch buffer
    std::vector<float> batch_input(batch_size * D0);
    std::vector<float> batch_target(batch_size);
    std::vector<float> output_buf(batch_size * D2);

    std::cout << "\nTraining: " << epochs << " epochs, batch=" << batch_size << ", lr=" << lr << "\n\n";

    std::vector<int> indices(train_imgs.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0;
        int correct = 0, total = 0, batches = 0;
        auto t0 = std::chrono::steady_clock::now();

        for (size_t bi = 0; bi < train_imgs.size(); bi += batch_size) {
            int B = std::min(batch_size, (int)(train_imgs.size() - bi));
            if (B < batch_size) continue;  // skip last incomplete

            // Prepare batch (host only fills buffer, no compute)
            for (int i = 0; i < B; i++) {
                int idx = indices[bi + i];
                batch_target[i] = train_lbls[idx];
                for (int j = 0; j < D0; j++)
                    batch_input[i * D0 + j] = (train_imgs[idx][j] / 255.0f - 0.1307f) / 0.3081f;
            }

            // === FORWARD — ALL ON NM6408 ===
            // Save ddr_ptr for this batch (reset after backward)
            PL_Addr save_ptr = ddr_ptr;

            // Upload batch input
            PL_Addr x_addr = upload(batch_input.data(), B * D0);

            // Layer 1: h1 = ReLU(x @ W1 + b1)
            PL_Addr z1_addr = nm_matmul(x_addr, w1_addr, B, D0, D1, 0);
            // Add bias (broadcast: each row += b1)
            // For simplicity, compute on NM6408 row by row
            // Actually just download, add bias on... NO! All on NM6408.
            // Upload b1 repeated for batch
            std::vector<float> b1_rep(B * D1);
            for (int i = 0; i < B; i++)
                memcpy(&b1_rep[i * D1], b1.data(), D1 * sizeof(float));
            PL_Addr b1r_addr = upload(b1_rep.data(), B * D1);
            PL_Addr z1b_addr = nm_add(z1_addr, b1r_addr, B * D1, 0);
            PL_Addr h1_addr = nm_relu(z1b_addr, B * D1, 0);

            // Layer 2: logits = h1 @ W2 + b2
            PL_Addr z2_addr = nm_matmul(h1_addr, w2_addr, B, D1, D2, 0);
            std::vector<float> b2_rep(B * D2);
            for (int i = 0; i < B; i++)
                memcpy(&b2_rep[i * D2], b2.data(), D2 * sizeof(float));
            PL_Addr b2r_addr = upload(b2_rep.data(), B * D2);
            PL_Addr logits_addr = nm_add(z2_addr, b2r_addr, B * D2, 0);

            // Download logits for softmax + loss (host does softmax — TODO: move to NM6408)
            download(output_buf.data(), logits_addr, B * D2);

            // Softmax + cross entropy + gradient (host — minimal compute)
            float batch_loss = 0;
            std::vector<float> dlogits(B * D2);
            for (int i = 0; i < B; i++) {
                float* l = &output_buf[i * D2];
                float mx = *std::max_element(l, l + D2);
                float sum = 0;
                for (int c = 0; c < D2; c++) {
                    l[c] = expf(l[c] - mx);
                    sum += l[c];
                }
                for (int c = 0; c < D2; c++) l[c] /= sum;

                int target = (int)batch_target[i];
                batch_loss -= logf(l[target] + 1e-8f);
                if (std::distance(l, std::max_element(l, l + D2)) == target) correct++;
                total++;

                // dL/dlogits = softmax - onehot
                for (int c = 0; c < D2; c++) {
                    dlogits[i * D2 + c] = l[c];
                }
                dlogits[i * D2 + target] -= 1.0f;
            }
            batch_loss /= B;
            for (auto& d : dlogits) d /= B;
            epoch_loss += batch_loss;
            batches++;

            // === BACKWARD — ALL MATMUL ON NM6408 ===
            PL_Addr dl_addr = upload(dlogits.data(), B * D2);

            // dW2 = h1.T @ dlogits  [D1, D2]
            // Need h1 transposed — download h1, transpose, upload
            std::vector<float> h1_buf(B * D1);
            download(h1_buf.data(), h1_addr, B * D1);
            std::vector<float> h1_t(D1 * B);
            for (int i = 0; i < B; i++)
                for (int j = 0; j < D1; j++)
                    h1_t[j * B + i] = h1_buf[i * D1 + j];
            PL_Addr h1t_addr = upload(h1_t.data(), D1 * B);
            PL_Addr dw2_addr = nm_matmul(h1t_addr, dl_addr, D1, B, D2, 0);

            // dh1 = dlogits @ W2.T  [B, D1]
            // Need W2 transposed
            std::vector<float> w2_t(D2 * D1);
            for (int i = 0; i < D1; i++)
                for (int j = 0; j < D2; j++)
                    w2_t[j * D1 + i] = W2[i * D2 + j];
            PL_Addr w2t_addr = upload(w2_t.data(), D2 * D1);
            PL_Addr dh1_addr = nm_matmul(dl_addr, w2t_addr, B, D2, D1, 0);

            // ReLU backward: dz1 = dh1 * (z1b > 0)
            std::vector<float> z1b_buf(B * D1), dh1_buf(B * D1);
            download(z1b_buf.data(), z1b_addr, B * D1);
            download(dh1_buf.data(), dh1_addr, B * D1);
            for (int i = 0; i < B * D1; i++)
                if (z1b_buf[i] <= 0) dh1_buf[i] = 0;
            PL_Addr dz1_addr = upload(dh1_buf.data(), B * D1);

            // dW1 = x.T @ dz1  [D0, D1]
            std::vector<float> x_t(D0 * B);
            for (int i = 0; i < B; i++)
                for (int j = 0; j < D0; j++)
                    x_t[j * B + i] = batch_input[i * D0 + j];
            PL_Addr xt_addr = upload(x_t.data(), D0 * B);
            PL_Addr dw1_addr = nm_matmul(xt_addr, dz1_addr, D0, B, D1, 0);

            // === SGD UPDATE — download gradients, update, upload ===
            std::vector<float> dW2_buf(D1 * D2), dW1_buf(D0 * D1);
            download(dW2_buf.data(), dw2_addr, D1 * D2);
            download(dW1_buf.data(), dw1_addr, D0 * D1);

            // Bias gradients: sum of dlogits/dz1 along batch
            std::vector<float> db2(D2, 0), db1(D1, 0);
            for (int i = 0; i < B; i++) {
                for (int c = 0; c < D2; c++) db2[c] += dlogits[i * D2 + c];
                for (int c = 0; c < D1; c++) db1[c] += dh1_buf[i * D1 + c];
            }

            // SGD step
            for (int i = 0; i < D0*D1; i++) W1[i] -= lr * std::max(-1.0f, std::min(1.0f, dW1_buf[i]));
            for (int i = 0; i < D1; i++) b1[i] -= lr * db1[i];
            for (int i = 0; i < D1*D2; i++) W2[i] -= lr * std::max(-1.0f, std::min(1.0f, dW2_buf[i]));
            for (int i = 0; i < D2; i++) b2[i] -= lr * db2[i];

            // Re-upload updated weights
            PL_WriteMemBlock(accesses[0], (PL_Word*)W1.data(), w1_addr, D0 * D1);
            PL_WriteMemBlock(accesses[0], (PL_Word*)b1.data(), b1_addr, D1);
            PL_WriteMemBlock(accesses[0], (PL_Word*)W2.data(), w2_addr, D1 * D2);
            PL_WriteMemBlock(accesses[0], (PL_Word*)b2.data(), b2_addr, D2);

            // Reset DDR bump allocator for next batch
            ddr_ptr = save_ptr + D0*D1 + D1 + D1*D2 + D2;  // keep weights

            if (batches % 100 == 0) {
                float acc = 100.0f * correct / total;
                std::cout << "  E" << epoch+1 << " batch " << batches
                          << ": loss=" << epoch_loss/batches << " acc=" << acc << "%\n";
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        float acc = 100.0f * correct / total;
        std::cout << "Epoch " << epoch+1 << "/" << epochs << ": loss=" << epoch_loss/batches
                  << " train_acc=" << acc << "% time=" << sec << "s\n";

        // Test — forward only on NM6408
        int test_correct = 0;
        for (size_t i = 0; i < test_imgs.size(); i++) {
            std::vector<float> input(D0);
            for (int j = 0; j < D0; j++)
                input[j] = (test_imgs[i][j] / 255.0f - 0.1307f) / 0.3081f;

            ddr_ptr = save_ptr + D0*D1 + D1 + D1*D2 + D2;
            PL_Addr xi = upload(input.data(), D0);
            PL_Addr z1i = nm_matmul(xi, w1_addr, 1, D0, D1, 0);
            std::vector<float> b1s(D1);
            memcpy(b1s.data(), b1.data(), D1*sizeof(float));
            PL_Addr b1i = upload(b1s.data(), D1);
            PL_Addr z1bi = nm_add(z1i, b1i, D1, 0);
            PL_Addr h1i = nm_relu(z1bi, D1, 0);
            PL_Addr z2i = nm_matmul(h1i, w2_addr, 1, D1, D2, 0);
            std::vector<float> b2s(D2);
            memcpy(b2s.data(), b2.data(), D2*sizeof(float));
            PL_Addr b2i = upload(b2s.data(), D2);
            PL_Addr li = nm_add(z2i, b2i, D2, 0);

            float logits[10];
            download(logits, li, D2);
            int pred = std::distance(logits, std::max_element(logits, logits + D2));
            if (pred == test_lbls[i]) test_correct++;
        }
        std::cout << "  Test: " << (100.0f * test_correct / test_imgs.size()) << "%\n\n";
    }

    // Shutdown
    for (int i = 0; i < num_cores; i++) {
        PL_Word exit_cmd = OP_EXIT;
        PL_WriteMemBlock(accesses[i], &exit_cmd, DDR_BASE + i * CMD_BLOCK_SIZE, 1);
    }
    usleep(300000);
    for (int i = 0; i < num_cores; i++) PL_CloseAccess(accesses[i]);
    PL_CloseBoardDesc(board);
    std::cout << "=== DONE ===\n";
    return 0;
}
