/* ============================================================================
 * host_16core_fixed — all 16 NMC4 cores trained in parallel without shared
 * memory race.  Each core's binary is compiled with NMC_INDEX = core_id_in_cluster
 * (0..3), so all per-core data sits in a disjoint 64 MB subregion of EMI.
 *
 * Different clusters have separate physical EMI banks → no cross-cluster race.
 * Within one cluster, 4 cores use NMC_INDEX 0..3 → no within-cluster race.
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

#define N_CLUSTERS  4
#define N_CORES     4
#define DATA_SIZE   (1 * 1024 * 1024)
#define VOCAB       128
#define T           32
#define D           32
#define FF          64
#define N_WEIGHTS   17600

/* Per-binary EMI base (NMC word addresses). nmc<N>_part.abs has its data
 * region rooted at (0x02000000 + N*0x01000000) per the linker template. */
static inline PL_Addr ADDR_TRAINING_DATA(int nmc_idx)  { return 0x0006bdcaUL + (PL_Addr)(0x02000000UL + nmc_idx * 0x01000000UL); }
static inline PL_Addr ADDR_CORE_LOSS_OUT(int nmc_idx)  { return 0x0006bdc8UL + (PL_Addr)(0x02000000UL + nmc_idx * 0x01000000UL); }
static inline PL_Addr ADDR_SAVED_WEIGHTS(int nmc_idx)  { return 0x000271c8UL + (PL_Addr)(0x02000000UL + nmc_idx * 0x01000000UL); }

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
        for (int d = 0; d < D; ++d) x[t][d] = m.Wtok[tokens[t]][d] + m.Wpos[t][d];
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
    int gen_n = (argc > 2) ? atoi(argv[2]) : 300;

    /* Load TinyStories */
    std::ifstream fin(DATA_PATH, std::ios::binary);
    if (!fin) { std::cerr << "ERROR: cannot open " << DATA_PATH << "\n"; return 1; }
    fin.seekg(0, std::ios::end); size_t fsize = fin.tellg(); fin.seekg(0);
    size_t take = (fsize < DATA_SIZE) ? fsize : DATA_SIZE;
    std::vector<char> bytes(DATA_SIZE, ' ');
    fin.read(bytes.data(), take);
    if (take < DATA_SIZE) for (size_t i = take; i < DATA_SIZE; ++i) bytes[i] = bytes[i % take];
    std::vector<PL_Word> word_buf(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) word_buf[i] = (PL_Word)((unsigned char)bytes[i]);

    PL_Board *board = nullptr;
    unsigned int n_boards = 0;
    if (PL_GetBoardCount(&n_boards) != PL_OK || n_boards < 1) return 1;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 2;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    PL_Access  *access[N_CLUSTERS][N_CORES] = {};
    IO_Service *service[N_CLUSTERS][N_CORES] = {};
    bool        done[N_CLUSTERS][N_CORES] = {};
    float       loss[N_CLUSTERS][N_CORES] = {};

    auto t0 = std::chrono::steady_clock::now();
    int loaded = 0;
    for (int c = 0; c < N_CLUSTERS; ++c) {
        for (int n = 0; n < N_CORES; ++n) {
            PL_CoreNo core; core.cluster_id = c; core.nm_id = n;
            if (PL_GetAccess(board, &core, &access[c][n]) != PL_OK) continue;
            char binname[64];
            std::snprintf(binname, sizeof(binname), "./nmc%d_part.abs", n);  /* per-core binary */
            if (PL_LoadProgramFile(access[c][n], binname) != PL_OK) {
                std::cerr << "WARN: load " << binname << " into chip=" << c << " core=" << n << " failed\n";
                PL_CloseAccess(access[c][n]); access[c][n] = nullptr; continue;
            }
            if (PL_WriteMemBlock(access[c][n], word_buf.data(),
                                 ADDR_TRAINING_DATA(n), (DWORD)DATA_SIZE) != PL_OK) {
                std::cerr << "WARN: write data chip=" << c << " core=" << n << " failed\n";
                PL_CloseAccess(access[c][n]); access[c][n] = nullptr; continue;
            }
            service[c][n] = IO_ServiceStart(binname, access[c][n], nullptr, nullptr, nullptr);
            ++loaded;
        }
    }
    std::cerr << "[host] Loaded " << loaded << "/16 cores; training...\n";

    int done_count = 0, wait_limit = 100000000;
    while (done_count < loaded && wait_limit-- > 0)
        for (int c = 0; c < N_CLUSTERS; ++c)
            for (int n = 0; n < N_CORES; ++n) {
                if (done[c][n] || access[c][n] == nullptr) continue;
                PL_Word status = 0;
                if (PL_GetStatus(access[c][n], &status) != PL_OK) continue;
                if (status == PROGRAM_PROGRESS) continue;
                if (status == PROGRAM_FINISHED) {
                    PL_Word lw = 0;
                    PL_ReadMemBlock(access[c][n], &lw, ADDR_CORE_LOSS_OUT(n), 1);
                    float l; std::memcpy(&l, &lw, 4);
                    loss[c][n] = l;
                    std::cout << "[chip=" << c << " core=" << n << "] loss=" << l << std::endl;
                    done[c][n] = true; ++done_count;
                } else { done[c][n] = true; ++done_count; }
            }

    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "[host] Training done in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    /* Find best healthy core (loss in [0.5, 4.0) — exclude overfit and untrained) */
    int best_c = -1, best_n = -1; float best_loss = 1e30f;
    for (int c = 0; c < N_CLUSTERS; ++c)
        for (int n = 0; n < N_CORES; ++n)
            if (done[c][n] && loss[c][n] > 0.5f && loss[c][n] < 4.0f && loss[c][n] < best_loss) {
                best_loss = loss[c][n]; best_c = c; best_n = n;
            }
    if (best_c < 0) {
        std::cerr << "WARN: no healthy core; falling back to any non-NaN\n";
        for (int c = 0; c < N_CLUSTERS; ++c)
            for (int n = 0; n < N_CORES; ++n)
                if (done[c][n] && loss[c][n] > 0 && loss[c][n] < best_loss) {
                    best_loss = loss[c][n]; best_c = c; best_n = n;
                }
    }
    std::cerr << "[host] Best: chip=" << best_c << " core=" << best_n << " loss=" << best_loss << "\n";

    /* In NMC_INDEX'd binary, this core's slot is at start of saved_weights_pool. */
    int slot = best_c * 4 + best_n;
    PL_Addr addr_slot = ADDR_SAVED_WEIGHTS(best_n) + (PL_Addr)slot * N_WEIGHTS;
    std::vector<PL_Word> wbuf(N_WEIGHTS);
    if (PL_ReadMemBlock(access[best_c][best_n], wbuf.data(), addr_slot, N_WEIGHTS) != PL_OK) {
        std::cerr << "ERROR: ReadMemBlock failed\n"; return 4;
    }
    std::vector<float> wflat(N_WEIGHTS);
    for (int i = 0; i < N_WEIGHTS; ++i) std::memcpy(&wflat[i], &wbuf[i], 4);
    {
        std::ofstream fout("./trained_weights.bin", std::ios::binary);
        fout.write(reinterpret_cast<const char*>(wflat.data()), N_WEIGHTS * 4);
    }
    std::cerr << "[host] Saved trained_weights.bin\n";

    for (int c = 0; c < N_CLUSTERS; ++c)
        for (int n = 0; n < N_CORES; ++n) {
            if (service[c][n]) IO_ServiceStop(&service[c][n], nullptr);
            if (access[c][n])  PL_CloseAccess(access[c][n]);
        }
    PL_CloseBoardDesc(board);

    Model m; load_from_flat(m, wflat.data());
    std::vector<int> ctx;
    for (size_t i = 0; i < std::strlen(prompt); ++i) ctx.push_back(((int)prompt[i]) & 0x7F);
    std::cout << "\n=== Generation (greedy, chip=" << best_c << " core=" << best_n
              << " loss=" << best_loss << ") ===\n" << prompt;
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
