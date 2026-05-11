/* ============================================================================
 * host_one_core — single-core training to avoid shared-memory race observed
 * when 4 cores in the same cluster all write to the same static arrays in EMI.
 *
 * Loads TinyStories into one core (chip 0, core 0), trains, reads weights,
 * generates text on host. ~26 min wall.
 * ============================================================================ */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <cmath>

#include "nm_quad_load.h"
#include "io_host.h"

#define DATA_SIZE   (4 * 1024 * 1024)
#define VOCAB       128
#define T           32
#define D           32
#define FF          64
#define N_WEIGHTS   25920   /* v4 with L=2 layers */

static PL_Addr ADDR_TRAINING_DATA = 0x0208d66aUL;
static PL_Addr ADDR_CORE_LOSS_OUT = 0x0208d668UL;
static PL_Addr ADDR_SAVED_WEIGHTS = 0x02028268UL;

static const char *NMC_PART  = "./nmc_part.abs";
static const char *DATA_PATH = "/home/<user>/nanogpt/data/tinystories_valid.txt";

struct Model {
    float Wtok[VOCAB][D], Wpos[T][D];
    float Wqkv[D][3 * D], Wout[D][D];
    float Wfc1[D][FF], Wfc2[FF][D];
    float Wunemb[D][VOCAB];
    float g1[D], b1[D], g2[D], b2[D], g3[D], b3[D];
};
static void load_from_flat(Model &m, const float *w) {
    int off = 0;
    std::memcpy(&m.Wtok[0][0],   w + off, sizeof(m.Wtok));   off += VOCAB * D;
    std::memcpy(&m.Wpos[0][0],   w + off, sizeof(m.Wpos));   off += T * D;
    std::memcpy(&m.Wqkv[0][0],   w + off, sizeof(m.Wqkv));   off += D * 3 * D;
    std::memcpy(&m.Wout[0][0],   w + off, sizeof(m.Wout));   off += D * D;
    std::memcpy(&m.Wfc1[0][0],   w + off, sizeof(m.Wfc1));   off += D * FF;
    std::memcpy(&m.Wfc2[0][0],   w + off, sizeof(m.Wfc2));   off += FF * D;
    std::memcpy(&m.Wunemb[0][0], w + off, sizeof(m.Wunemb)); off += D * VOCAB;
    std::memcpy(m.g1, w + off, sizeof(m.g1)); off += D;
    std::memcpy(m.b1, w + off, sizeof(m.b1)); off += D;
    std::memcpy(m.g2, w + off, sizeof(m.g2)); off += D;
    std::memcpy(m.b2, w + off, sizeof(m.b2)); off += D;
    std::memcpy(m.g3, w + off, sizeof(m.g3)); off += D;
    std::memcpy(m.b3, w + off, sizeof(m.b3)); off += D;
    (void)off;
}
static void layernorm(float *out, const float *in, const float *g, const float *b, int n) {
    float mu = 0, var = 0;
    for (int i = 0; i < n; ++i) mu += in[i]; mu /= n;
    for (int i = 0; i < n; ++i) { float d = in[i] - mu; var += d * d; } var /= n;
    float inv = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < n; ++i) out[i] = (in[i] - mu) * inv * g[i] + b[i];
}
static int predict_next(const Model &m, const int *tokens, int len) {
    float x[T][D], ln1[T][D], Q[T][D], K[T][D], V[T][D];
    float scores[T][T], attn[T][D], proj[T][D], res1[T][D], ln2[T][D];
    float fc1[T][FF], ffn[T][D], res2[T][D], ln3[T][D], logits[VOCAB];
    int n = len < T ? len : T;
    for (int t = 0; t < n; ++t)
        for (int d = 0; d < D; ++d)
            x[t][d] = m.Wtok[tokens[t]][d] + m.Wpos[t][d];
    for (int t = 0; t < n; ++t) layernorm(ln1[t], x[t], m.g1, m.b1, D);
    for (int t = 0; t < n; ++t)
        for (int d = 0; d < D; ++d) {
            float q = 0, kk = 0, v = 0;
            for (int h = 0; h < D; ++h) {
                q  += ln1[t][h] * m.Wqkv[h][d];
                kk += ln1[t][h] * m.Wqkv[h][d + D];
                v  += ln1[t][h] * m.Wqkv[h][d + 2 * D];
            }
            Q[t][d] = q; K[t][d] = kk; V[t][d] = v;
        }
    float scale = 1.0f / std::sqrt((float)D);
    for (int t = 0; t < n; ++t) {
        float maxs = -1e30f;
        for (int k = 0; k <= t; ++k) {
            float s = 0; for (int d = 0; d < D; ++d) s += Q[t][d] * K[k][d];
            s *= scale; scores[t][k] = s; if (s > maxs) maxs = s;
        }
        float sum = 0;
        for (int k = 0; k <= t; ++k) { scores[t][k] = std::exp(scores[t][k] - maxs); sum += scores[t][k]; }
        float inv = 1.0f / sum;
        for (int k = 0; k <= t; ++k) scores[t][k] *= inv;
        for (int d = 0; d < D; ++d) {
            float a = 0; for (int k = 0; k <= t; ++k) a += scores[t][k] * V[k][d];
            attn[t][d] = a;
        }
    }
    for (int t = 0; t < n; ++t)
        for (int d = 0; d < D; ++d) {
            float s = 0; for (int h = 0; h < D; ++h) s += attn[t][h] * m.Wout[h][d];
            proj[t][d] = s;
        }
    for (int t = 0; t < n; ++t) for (int d = 0; d < D; ++d) res1[t][d] = x[t][d] + proj[t][d];
    for (int t = 0; t < n; ++t) layernorm(ln2[t], res1[t], m.g2, m.b2, D);
    for (int t = 0; t < n; ++t) {
        for (int h = 0; h < FF; ++h) {
            float s = 0; for (int d = 0; d < D; ++d) s += ln2[t][d] * m.Wfc1[d][h];
            fc1[t][h] = (s > 0) ? s : 0;
        }
        for (int d = 0; d < D; ++d) {
            float s = 0; for (int h = 0; h < FF; ++h) s += fc1[t][h] * m.Wfc2[h][d];
            ffn[t][d] = s;
        }
    }
    for (int t = 0; t < n; ++t) for (int d = 0; d < D; ++d) res2[t][d] = res1[t][d] + ffn[t][d];
    layernorm(ln3[n - 1], res2[n - 1], m.g3, m.b3, D);
    for (int d = 0; d < VOCAB; ++d) {
        float s = 0; for (int h = 0; h < D; ++h) s += ln3[n - 1][h] * m.Wunemb[h][d];
        logits[d] = s;
    }
    int best = 0; float bv = logits[0];
    for (int d = 1; d < VOCAB; ++d) if (logits[d] > bv) { bv = logits[d]; best = d; }
    return best;
}

int main(int argc, char *argv[]) {
    const char *prompt = (argc > 1) ? argv[1] : "Once upon a time ";
    int gen_n = (argc > 2) ? atoi(argv[2]) : 200;

    /* Read TinyStories */
    std::ifstream fin(DATA_PATH, std::ios::binary);
    if (!fin) { std::cerr << "ERROR: cannot open " << DATA_PATH << "\n"; return 1; }
    fin.seekg(0, std::ios::end); size_t fsize = fin.tellg(); fin.seekg(0);
    size_t take = (fsize < DATA_SIZE) ? fsize : DATA_SIZE;
    std::vector<char> bytes(DATA_SIZE, ' ');
    fin.read(bytes.data(), take);
    if (take < DATA_SIZE) for (size_t i = take; i < DATA_SIZE; ++i) bytes[i] = bytes[i % take];
    std::vector<PL_Word> word_buf(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) word_buf[i] = (PL_Word)((unsigned char)bytes[i]);

    /* Open board */
    PL_Board *board = nullptr;
    unsigned int n_boards = 0;
    if (PL_GetBoardCount(&n_boards) != PL_OK || n_boards < 1) return 1;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 2;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    /* Single core: chip 0 core 0 */
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *access = nullptr;
    if (PL_GetAccess(board, &core, &access) != PL_OK) return 3;
    if (PL_LoadProgramFile(access, NMC_PART) != PL_OK) return 4;
    if (PL_WriteMemBlock(access, word_buf.data(), ADDR_TRAINING_DATA, DATA_SIZE) != PL_OK) return 5;

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *service = IO_ServiceStart(NMC_PART, access, nullptr, nullptr, nullptr);

    while (true) {
        PL_Word st = 0;
        if (PL_GetStatus(access, &st) != PL_OK) break;
        if (st == PROGRAM_PROGRESS) continue;
        if (st == PROGRAM_FINISHED) {
            PL_Word lw = 0; PL_ReadMemBlock(access, &lw, ADDR_CORE_LOSS_OUT, 1);
            float l; std::memcpy(&l, &lw, 4);
            std::cout << "[host] FINAL loss=" << l << "\n";
            break;
        }
        std::cerr << "[host] ERROR status=" << st << "\n";
        return 6;
    }

    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "[host] Training done in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    /* Read weights from core 0 slot 0 — start of saved_weights_pool */
    PL_Addr addr_slot = ADDR_SAVED_WEIGHTS;  /* slot 0 */
    std::vector<PL_Word> wbuf(N_WEIGHTS);
    if (PL_ReadMemBlock(access, wbuf.data(), addr_slot, N_WEIGHTS) != PL_OK) {
        std::cerr << "ERROR: ReadMemBlock failed\n"; return 7;
    }
    std::vector<float> wflat(N_WEIGHTS);
    for (int i = 0; i < N_WEIGHTS; ++i) std::memcpy(&wflat[i], &wbuf[i], 4);
    {
        std::ofstream fout("./trained_weights.bin", std::ios::binary);
        fout.write(reinterpret_cast<const char*>(wflat.data()), N_WEIGHTS * 4);
    }
    std::cerr << "[host] Saved trained_weights.bin\n";

    if (service) IO_ServiceStop(&service, nullptr);
    PL_CloseAccess(access);
    PL_CloseBoardDesc(board);

    /* Generate */
    Model m; load_from_flat(m, wflat.data());
    std::vector<int> ctx;
    for (size_t i = 0; i < std::strlen(prompt); ++i) ctx.push_back(((int)prompt[i]) & 0x7F);
    std::cout << "\n=== Generation (greedy, single core 0:0) ===\n" << prompt;
    std::cout.flush();
    for (int g = 0; g < gen_n; ++g) {
        int len = (int)ctx.size();
        int start = (len > T) ? (len - T) : 0;
        int n = len - start;
        int local_tokens[T] = {0};
        for (int i = 0; i < n; ++i) local_tokens[i] = ctx[start + i];
        int nxt = predict_next(m, local_tokens, n);
        ctx.push_back(nxt);
        std::cout << (char)nxt; std::cout.flush();
    }
    std::cout << "\n";
    return 0;
}
