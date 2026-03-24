// ============================================================================
// train_gpt_4chip.cpp — 4-chip × 16-core ROW-PARALLEL GPT training on NM QUAD
// ============================================================================
// Architecture:
//   4 NM6408 chips × 16 cores = 64 independent fused forward/backward
//   Each core processes B_total/total_cores rows of the batch
//   Shared weights in DDR (read-only during forward)
//   Non-overlapping outputs per core
//
// Key insight: NO coordinator/worker pattern. Each core runs the FULL
// transformer for its own batch rows. Weight updates are lr-scaled so
// all cores' updates sum to the correct gradient step.
//
// Supports three modes:
//   --model small:  D=128, H=4, FF=256, L=2, T=32  (~200K params)
//   --model large:  D=768, H=12, FF=1536, L=12, T=64 (~85M params, fits 5GB/chip)
//   --model 250m:   D=768, H=12, FF=3072, L=36, T=64 (~255M params, ~2GB/chip with B=1)
//
// 250m DDR budget (per chip, B_per_core=1, 16 cores):
//   Weights: ~972 MB (shared), Per-core: ~66.6 MB × 16 = ~1066 MB
//   Total: ~2038 MB / 5120 MB (39.8%) — fits comfortably
//   HD=64 matches dispatcher_v3 stack arrays Q_h[64*64] exactly
//
// Build: g++ -O2 -o train_gpt_4chip train_gpt_4chip.cpp -ldl -lnm_quad_load
// Run:   ./train_gpt_4chip --data tiny_shakespeare.txt --model small --boards 1 --clusters 4 --cores 4

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unistd.h>

extern "C" {
    struct PL_Board; struct PL_Access;
    typedef struct { int nm_id; int cluster_id; } PL_CoreNo;
    typedef unsigned int PL_Word, PL_Addr;
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

#define DDR      0x00340000u
#define CBS      32           // cmd block size (words)
#define SA       30           // status word offset
#define DATA0    (DDR + 16*CBS)

#define OP_FUSED_FORWARD_ROWPAR  32
#define OP_FUSED_BACKWARD_ROWPAR 33
#define OP_FUSED_BACKWARD_GRADONLY 34
#define OP_EXIT  255

// ============================================================
// Hardware state
// ============================================================
struct CoreHandle {
    PL_Access* access;
    int chip_id;
    int core_id;   // 0..15 within chip
};

static PL_Board* board;
static std::vector<CoreHandle> all_cores;
static int num_chips = 0;

// Per-core DDR addresses
struct CoreAddrs {
    PL_Addr tokens;
    PL_Addr logits;
    PL_Addr h_out;        // h_cache + hn_cache + ff1r_cache
    PL_Addr scratch;      // forward scratch
    PL_Addr dlogits;      // backward input
    PL_Addr bk_scratch;   // backward scratch
    PL_Addr grad_buf;     // per-core gradient buffer for GRADONLY mode
};

// Per-chip shared addresses
struct ChipAddrs {
    PL_Addr wte, wpe, layers, lm_head;
    PL_Addr data_end;     // bump allocator end
};

static std::vector<CoreAddrs> core_addrs;
static ChipAddrs chip_addrs[4];

// I/O helpers — use the core's own access handle
void core_wr(int ci, PL_Addr a, const void* d, int n) {
    PL_WriteMemBlock(all_cores[ci].access, (const PL_Word*)d, a, n);
}
void core_rd(int ci, PL_Addr a, void* d, int n) {
    PL_ReadMemBlock(all_cores[ci].access, (PL_Word*)d, a, n);
}

bool core_wait(int ci, int max_polls = 10000000) {
    PL_Addr base = DDR + all_cores[ci].core_id * CBS;
    for (int i = 0; i < max_polls; i++) {
        PL_Word s;
        PL_ReadMemBlock(all_cores[ci].access, &s, base + SA, 1);
        if (s == 1) return true;
        if (s == 2) {
            printf("ERROR: core %d (chip %d)\n", all_cores[ci].core_id, all_cores[ci].chip_id);
            return false;
        }
        usleep(50);
    }
    printf("TIMEOUT: core %d (chip %d)\n", all_cores[ci].core_id, all_cores[ci].chip_id);
    return false;
}

// Write args to core's cmd block, then trigger opcode
void core_dispatch(int ci, unsigned op, const unsigned* args, int nargs) {
    PL_Addr base = DDR + all_cores[ci].core_id * CBS;
    if (nargs > 0)
        PL_WriteMemBlock(all_cores[ci].access, (PL_Word*)args, base + 1, nargs);
    PL_Word z = 0;
    PL_WriteMemBlock(all_cores[ci].access, &z, base + SA, 1);
    PL_Word c = op;
    PL_WriteMemBlock(all_cores[ci].access, &c, base, 1);
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    std::string data_path = "tiny_shakespeare.txt";
    std::string disp_path = "dispatcher_nmquad_v3.abs";
    std::string model_name = "small";
    int epochs = 10, steps = 200, B_total = 64;
    float lr = 0.001f;
    int max_boards = 4, max_clusters = 4, max_cores_per_cluster = 4;
    bool parallel_backward = false;  // --parallel-backward: use GRADONLY opcode
    int wave_size = 4;  // max cores per board that can run backward simultaneously

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--data") data_path = argv[++i];
        else if (a == "--dispatcher") disp_path = argv[++i];
        else if (a == "--model") model_name = argv[++i];
        else if (a == "--epochs") epochs = atoi(argv[++i]);
        else if (a == "--steps") steps = atoi(argv[++i]);
        else if (a == "--batch") B_total = atoi(argv[++i]);
        else if (a == "--lr") lr = atof(argv[++i]);
        else if (a == "--boards") max_boards = atoi(argv[++i]);
        else if (a == "--clusters") max_clusters = atoi(argv[++i]);
        else if (a == "--cores") max_cores_per_cluster = atoi(argv[++i]);
        else if (a == "--parallel-backward") parallel_backward = true;
        else if (a == "--wave-size") wave_size = atoi(argv[++i]);
    }

    // Model configs
    int D, H, FF, L, T;
    if (model_name == "250m") {
        // ~255M params: 36 layers, GPT-2 small dimensions, 4x wider FFN
        // HD = 768/12 = 64, fits dispatcher_v3 Q_h[64*64] exactly
        // DDR ~2.0 GB/chip with B_per_core=1, 16 cores (39.8% of 5GB)
        D = 768; H = 12; FF = 3072; L = 36; T = 64;
    } else if (model_name == "large") {
        D = 768; H = 12; FF = 1536; L = 12; T = 64;
    } else {
        D = 128; H = 4; FF = 256; L = 2; T = 32;
    }
    int HD = D / H;

    // ======================== Load data ========================
    std::ifstream f(data_path);
    if (!f.is_open()) { printf("Cannot open %s\n", data_path.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(f)), {});
    std::vector<char> chars;
    for (char c : text)
        if (std::find(chars.begin(), chars.end(), c) == chars.end())
            chars.push_back(c);
    std::sort(chars.begin(), chars.end());
    int V = chars.size();
    int stoi_map[256] = {};
    char itos_map[256] = {};
    for (int i = 0; i < V; i++) {
        stoi_map[(unsigned char)chars[i]] = i;
        itos_map[i] = chars[i];
    }
    std::vector<int> data_tokens(text.size());
    for (size_t i = 0; i < text.size(); i++)
        data_tokens[i] = stoi_map[(unsigned char)text[i]];
    printf("Data: %zu chars, V=%d\n", data_tokens.size(), V);

    // ======================== Init hardware ========================
    // NM QUAD: 4 boards (1 per NM6408 chip), each with 4 clusters × 4 cores = 16
    // Total: 4 boards × 16 cores = 64 cores
    PL_SetTimeout(60000);
    unsigned cnt;
    if (PL_GetBoardCount(&cnt) != 0 || cnt == 0) {
        printf("No NM QUAD boards found\n"); return 1;
    }
    printf("Boards detected: %u\n", cnt);

    // Each board = 1 NM6408 chip with its own DDR
    PL_Board* boards[4] = {};
    if (max_boards < 1) max_boards = 1;
    if (max_boards > 4) max_boards = 4;
    int num_boards = (int)cnt < max_boards ? (int)cnt : max_boards;

    for (int b = 0; b < num_boards; b++) {
        if (PL_GetBoardDesc(b, &boards[b]) != 0) { num_boards = b; break; }
        PL_ResetBoard(boards[b]);
        PL_LoadInitCode(boards[b]);

        // Each board has up to 4 clusters × 4 cores
        int cls_limit = max_clusters < 4 ? max_clusters : 4;
        int co_limit = max_cores_per_cluster < 4 ? max_cores_per_cluster : 4;
        for (int cl = 0; cl < cls_limit; cl++) {
            for (int co = 0; co < co_limit; co++) {
                PL_CoreNo cn = {co, cl};
                PL_Access* a;
                if (PL_GetAccess(boards[b], &cn, &a) == 0) {
                    if (PL_LoadProgramFile(a, disp_path.c_str()) == 0) {
                        CoreHandle ch;
                        ch.access = a;
                        ch.chip_id = b;  // board = chip
                        ch.core_id = (cl << 2) + co;  // matches dispatcher
                        all_cores.push_back(ch);
                    } else {
                        PL_CloseAccess(a);
                    }
                }
            }
        }
        printf("  Board %d: %d cores\n", b, (int)all_cores.size() - b * 16);
    }
    num_chips = num_boards;
    board = boards[0];  // keep for cleanup

    usleep(1000000);  // 1s for dispatchers to init

    int NC = (int)all_cores.size();
    printf("NM QUAD: %d boards (chips), %d total cores\n", num_boards, NC);
    if (NC == 0) { printf("No cores!\n"); return 1; }

    // ======================== Adjust batch ========================
    int B_per_core = B_total / NC;
    if (B_per_core < 1) B_per_core = 1;
    B_total = B_per_core * NC;
    int BT = B_per_core * T;

    // ======================== Init weights ========================
    std::mt19937 rng(42);
    auto rf = [&](int n) {
        std::vector<float> v(n);
        std::normal_distribution<float> d(0, 0.02f);
        for (auto& x : v) x = d(rng);
        return v;
    };

    auto wte = rf(V * D);
    auto wpe = rf(T * D);
    auto lm_w = rf(D * V);

    int lsz = 4*D*D + 2*D*FF + D;
    std::vector<float> packed(L * lsz);
    for (int i = 0; i < L; i++) {
        auto q=rf(D*D), k=rf(D*D), v=rf(D*D), o=rf(D*D), w1=rf(D*FF), w2=rf(FF*D);
        std::vector<float> g(D, 1.0f);
        float* dst = packed.data() + i * lsz;
        memcpy(dst, q.data(), D*D*4); dst += D*D;
        memcpy(dst, k.data(), D*D*4); dst += D*D;
        memcpy(dst, v.data(), D*D*4); dst += D*D;
        memcpy(dst, o.data(), D*D*4); dst += D*D;
        memcpy(dst, w1.data(), D*FF*4); dst += D*FF;
        memcpy(dst, w2.data(), FF*D*4); dst += FF*D;
        memcpy(dst, g.data(), D*4);
    }

    int np = V*D + T*D + D*V + L*lsz;
    float weights_mb = np * 4.0f / (1024*1024);
    printf("\n%s model: %.1fM params (%.1f MB), L=%d D=%d H=%d FF=%d T=%d\n",
           model_name.c_str(), np/1e6, weights_mb, L, D, H, FF, T);
    printf("Batch: %d total, %d/core, %d cores\n", B_total, B_per_core, NC);

    // ======================== Upload weights + allocate per-core areas ========================
    // Each board (chip) has its OWN 5GB DDR.
    // Weights replicated on each chip. Per-core areas on their chip's DDR.
    std::vector<std::vector<int>> chip_core_ids(num_chips);
    for (int ci = 0; ci < NC; ci++)
        chip_core_ids[all_cores[ci].chip_id].push_back(ci);

    // Per-core cache and scratch sizes (must match dispatcher layout EXACTLY)
    int BH = B_per_core * H;
    int cache_words = (L+1)*BT*D + L*BT*D + L*BT*FF;
    int fwd_scratch_words = BT*D*5 + BH*HD*T + BH*T*T
                          + BT*D + BT*D + BT*FF + BT*D + BH*T*HD*2;
    // Backward scratch must include full attention backward buffers:
    // dW needs max(D*V, D*D, D*FF, FF*D) — used for ALL gradient matrices
    // temp1 needs max(BT*FF, D*BT, V*D, D*FF, FF*D) — used for transpose buffers
    // Layout: dW + dx(BT*D) + temp1 + temp2(BT*FF) + temp3(BT*D)
    // + Q_full+K_full+V_full+d_O+d_Q+d_K+d_V (7*BT*D) + Wt(D*D) + dx_add(BT*D)
    int dW_size = D*V;
    if (D*D > dW_size) dW_size = D*D;
    if (D*FF > dW_size) dW_size = D*FF;
    if (FF*D > dW_size) dW_size = FF*D;
    int temp1_size = BT*FF;
    if (D*BT > temp1_size) temp1_size = D*BT;
    if (V*D > temp1_size) temp1_size = V*D;
    if (D*FF > temp1_size) temp1_size = D*FF;
    if (FF*D > temp1_size) temp1_size = FF*D;
    if (D*D > temp1_size) temp1_size = D*D;
    // Total: dW_size + dx(BT*D) + temp1_size + temp2(BT*FF)
    //      + temp3(BT*D) + Q/K/V/dO/dQ/dK/dV(7*BT*D) + Wt(D*D) + dx_add(BT*D)
    //      = dW_size + temp1_size + BT*FF + 10*BT*D + D*D
    int bwd_scratch_words = dW_size + temp1_size + BT*FF + 10*BT*D + D*D;
    printf("Scratch: dW=%d (was D*V=%d), temp1=%d (was BT*FF=%d), bwd_total=%d words\n",
           dW_size, D*V, temp1_size, BT*FF, bwd_scratch_words);

    // Gradient buffer size per core (for GRADONLY mode):
    // grad_lm_head[D*V] + grad_layers[L*lsz] + grad_wte[V*D]
    int grad_buf_words = D*V + L*lsz + V*D;

    core_addrs.resize(NC);

    for (int ch = 0; ch < num_chips; ch++) {
        PL_Addr dp = DATA0;
        int first_ci = chip_core_ids[ch][0];

        // Upload weights to this chip's DDR
        chip_addrs[ch].wte = dp;
        core_wr(first_ci, dp, wte.data(), V*D); dp += V*D;
        chip_addrs[ch].wpe = dp;
        core_wr(first_ci, dp, wpe.data(), T*D); dp += T*D;
        chip_addrs[ch].layers = dp;
        core_wr(first_ci, dp, packed.data(), L*lsz); dp += L*lsz;
        chip_addrs[ch].lm_head = dp;
        core_wr(first_ci, dp, lm_w.data(), D*V); dp += D*V;

        // Allocate per-core areas on this chip's DDR
        for (int ci : chip_core_ids[ch]) {
            core_addrs[ci].tokens     = dp; dp += B_per_core * T;
            core_addrs[ci].logits     = dp; dp += BT * V;
            core_addrs[ci].h_out      = dp; dp += cache_words;
            core_addrs[ci].scratch    = dp; dp += fwd_scratch_words;
            core_addrs[ci].dlogits    = dp; dp += BT * V;
            core_addrs[ci].bk_scratch = dp; dp += bwd_scratch_words;
            if (parallel_backward) {
                core_addrs[ci].grad_buf = dp; dp += grad_buf_words;
            } else {
                core_addrs[ci].grad_buf = 0;
            }
        }

        chip_addrs[ch].data_end = dp;
        float ddr_mb = (dp - DATA0) * 4.0f / (1024*1024);
        printf("  Chip %d DDR: %.1f MB / 5120 MB (%d cores)\n",
               ch, ddr_mb, (int)chip_core_ids[ch].size());
    }

    {
        float total_mb = 0;
        for (int ch = 0; ch < num_chips; ch++)
            total_mb += (chip_addrs[ch].data_end - DATA0) * 4.0f / (1024*1024);
        printf("Total DDR: %.1f MB across %d chips\n", total_mb, num_chips);
    }

    if (parallel_backward) {
        float grad_mb = grad_buf_words * 4.0f / (1024*1024);
        int cores_per_board = 0;
        for (int ch = 0; ch < num_chips; ch++)
            if ((int)chip_core_ids[ch].size() > cores_per_board)
                cores_per_board = (int)chip_core_ids[ch].size();
        int num_waves = (cores_per_board + wave_size - 1) / wave_size;
        printf("PARALLEL BACKWARD: grad_buf=%.1f MB/core, %d cores = %.1f MB total\n",
               grad_mb, NC, grad_mb * NC);
        printf("  WAVE STRATEGY: %d cores/board, wave_size=%d, %d waves per step\n",
               cores_per_board, wave_size, num_waves);
        printf("  Each wave: %d backward/board × %d boards = %d parallel backward ops (safe: ≤4/DDR)\n",
               wave_size, num_chips, wave_size * num_chips);
    }

    printf("\nTraining: %d epochs x %d steps, lr=%.4f\n", epochs, steps, lr);
    printf("ROW-PARALLEL: %d cores × fused forward+backward (%s)\n", NC,
           parallel_backward ? "BATCHED WAVE grad-only" : "SEQUENTIAL in-place SGD");
    printf("ALL compute on NM6408, ZERO CPU fallback\n\n");

    // ======================== Training loop ========================
    for (int ep = 0; ep < epochs; ep++) {
        float epoch_loss = 0;
        int correct = 0, total_tok = 0;
        auto t0 = std::chrono::steady_clock::now();

        for (int step = 0; step < steps; step++) {
            // 1. Generate batch tokens, split across cores
            std::vector<std::vector<unsigned int>> tok_per_core(NC);
            std::vector<std::vector<int>> tgt_per_core(NC);

            for (int ci = 0; ci < NC; ci++) {
                tok_per_core[ci].resize(BT);
                tgt_per_core[ci].resize(BT);
                for (int b = 0; b < B_per_core; b++) {
                    int idx = rng() % (data_tokens.size() - T - 1);
                    for (int t = 0; t < T; t++) {
                        tok_per_core[ci][b*T+t] = data_tokens[idx+t];
                        tgt_per_core[ci][b*T+t] = data_tokens[idx+t+1];
                    }
                }
            }

            // 2. Upload tokens to each core
            for (int ci = 0; ci < NC; ci++)
                core_wr(ci, core_addrs[ci].tokens,
                        tok_per_core[ci].data(), BT);

            // 3. DISPATCH FUSED FORWARD to ALL cores simultaneously
            for (int ci = 0; ci < NC; ci++) {
                int ch = all_cores[ci].chip_id;
                unsigned args[15] = {
                    (unsigned)B_per_core,
                    (unsigned)T, (unsigned)D, (unsigned)H, (unsigned)FF,
                    (unsigned)V, (unsigned)L,
                    core_addrs[ci].tokens,
                    chip_addrs[ch].wte,
                    chip_addrs[ch].wpe,
                    chip_addrs[ch].layers,
                    chip_addrs[ch].lm_head,
                    core_addrs[ci].logits,
                    core_addrs[ci].h_out,
                    core_addrs[ci].scratch
                };
                core_dispatch(ci, OP_FUSED_FORWARD_ROWPAR, args, 15);
            }

            // 4. WAIT ALL cores
            bool ok = true;
            for (int ci = 0; ci < NC; ci++)
                if (!core_wait(ci)) { ok = false; break; }
            if (!ok) { printf("FORWARD FAILED step %d\n", step); break; }

            // 5. Download logits, compute loss + dlogits on host
            float batch_loss = 0;
            for (int ci = 0; ci < NC; ci++) {
                std::vector<float> logits(BT * V);
                core_rd(ci, core_addrs[ci].logits, logits.data(), BT * V);

                std::vector<float> dl(BT * V);
                for (int bt = 0; bt < BT; bt++) {
                    float* l = &logits[bt * V];
                    float mx = *std::max_element(l, l + V);
                    float sm = 0;
                    for (int c = 0; c < V; c++) { l[c] = expf(l[c] - mx); sm += l[c]; }
                    for (int c = 0; c < V; c++) { l[c] /= sm; dl[bt*V+c] = l[c]; }
                    int tgt = tgt_per_core[ci][bt];
                    batch_loss -= logf(l[tgt] + 1e-8f);
                    dl[bt*V + tgt] -= 1.0f;
                    if (std::distance(l, std::max_element(l, l+V)) == tgt) correct++;
                    total_tok++;
                }
                for (auto& d : dl) d /= (float)B_total;  // normalize by total batch, not per-core

                // Upload dlogits
                core_wr(ci, core_addrs[ci].dlogits, dl.data(), BT * V);
            }
            epoch_loss += batch_loss / (B_total * T);

            // 6. DISPATCH FUSED BACKWARD
            unsigned lr_bits;
            memcpy(&lr_bits, &lr, 4);

            if (parallel_backward) {
                // ============================================================
                // BATCHED BACKWARD (GRADONLY): cores run in WAVES to avoid DDR hang.
                // Max `wave_size` cores per board run backward simultaneously.
                // 16 cores/board at once = HANGS. 4 cores/board at once = SAFE.
                //
                // Wave strategy (wave_size=4, 16 cores/board):
                //   Wave 0: cores 0-3 on each board (4/board × 4 boards = 16 parallel)
                //   Wave 1: cores 4-7 on each board
                //   Wave 2: cores 8-11 on each board
                //   Wave 3: cores 12-15 on each board
                //
                // After all waves: host downloads grad_buf from ALL cores, sums, SGD.
                // ============================================================

                // Determine number of waves needed
                int max_cores_on_board = 0;
                for (int ch = 0; ch < num_chips; ch++)
                    if ((int)chip_core_ids[ch].size() > max_cores_on_board)
                        max_cores_on_board = (int)chip_core_ids[ch].size();
                int num_waves = (max_cores_on_board + wave_size - 1) / wave_size;

                // 6a. Dispatch backward in waves
                for (int wave = 0; wave < num_waves; wave++) {
                    int start = wave * wave_size;

                    // Dispatch this wave's cores on ALL boards
                    for (int ch = 0; ch < num_chips; ch++) {
                        int end = start + wave_size;
                        if (end > (int)chip_core_ids[ch].size())
                            end = (int)chip_core_ids[ch].size();
                        for (int idx = start; idx < end; idx++) {
                            int ci = chip_core_ids[ch][idx];
                            unsigned bk_args[17] = {
                                (unsigned)B_per_core,
                                (unsigned)T, (unsigned)D, (unsigned)H, (unsigned)FF,
                                (unsigned)V, (unsigned)L,
                                core_addrs[ci].dlogits,
                                core_addrs[ci].tokens,
                                chip_addrs[ch].wte,           // read-only
                                chip_addrs[ch].layers,         // read-only
                                chip_addrs[ch].lm_head,        // read-only
                                core_addrs[ci].h_out,
                                (unsigned)(core_addrs[ci].h_out + (L+1)*BT*D),
                                (unsigned)(core_addrs[ci].h_out + (L+1)*BT*D + L*BT*D),
                                core_addrs[ci].bk_scratch,
                                core_addrs[ci].grad_buf        // per-core private grad buffer
                            };
                            core_dispatch(ci, OP_FUSED_BACKWARD_GRADONLY, bk_args, 17);
                        }
                    }

                    // Wait this wave's cores on ALL boards
                    for (int ch = 0; ch < num_chips; ch++) {
                        int end = start + wave_size;
                        if (end > (int)chip_core_ids[ch].size())
                            end = (int)chip_core_ids[ch].size();
                        for (int idx = start; idx < end; idx++) {
                            int ci = chip_core_ids[ch][idx];
                            if (!core_wait(ci))
                                printf("BACKWARD TIMEOUT wave %d core %d (chip %d)\n",
                                       wave, all_cores[ci].core_id, ch);
                        }
                    }
                }

                // 6b. Per-chip: download grad buffers, sum, apply SGD, upload weights
                for (int ch = 0; ch < num_chips; ch++) {
                    int cores_on_chip = (int)chip_core_ids[ch].size();
                    int first_ci = chip_core_ids[ch][0];

                    // Download first core's gradient as accumulator
                    std::vector<float> sum_grad(grad_buf_words);
                    core_rd(first_ci, core_addrs[first_ci].grad_buf,
                            sum_grad.data(), grad_buf_words);

                    // Sum gradients from remaining cores
                    for (int k = 1; k < cores_on_chip; k++) {
                        int ci = chip_core_ids[ch][k];
                        std::vector<float> core_grad(grad_buf_words);
                        core_rd(ci, core_addrs[ci].grad_buf,
                                core_grad.data(), grad_buf_words);
                        for (int i = 0; i < grad_buf_words; i++)
                            sum_grad[i] += core_grad[i];
                    }

                    // grad_out layout: [grad_lm_head(D*V), grad_layers(L*lsz), grad_wte(V*D)]
                    float* g_lm = sum_grad.data();
                    float* g_lay = sum_grad.data() + D*V;
                    float* g_wte = sum_grad.data() + D*V + L*lsz;

                    // Download current weights from this chip
                    core_rd(first_ci, chip_addrs[ch].lm_head, lm_w.data(), D*V);
                    core_rd(first_ci, chip_addrs[ch].layers, packed.data(), L*lsz);
                    core_rd(first_ci, chip_addrs[ch].wte, wte.data(), V*D);

                    // Apply SGD: W -= lr * sum_grad
                    for (int i = 0; i < D*V; i++)   lm_w[i]   -= lr * g_lm[i];
                    for (int i = 0; i < L*lsz; i++) packed[i]  -= lr * g_lay[i];
                    for (int i = 0; i < V*D; i++)   wte[i]     -= lr * g_wte[i];

                    // Upload updated weights back to chip
                    core_wr(first_ci, chip_addrs[ch].lm_head, lm_w.data(), D*V);
                    core_wr(first_ci, chip_addrs[ch].layers, packed.data(), L*lsz);
                    core_wr(first_ci, chip_addrs[ch].wte, wte.data(), V*D);
                }

            } else {
                // ============================================================
                // SEQUENTIAL BACKWARD (original): one core at a time per chip
                // Each core does in-place SGD on shared weights.
                // ============================================================

                int max_cores_per_chip = 0;
                for (int ch = 0; ch < num_chips; ch++)
                    if ((int)chip_core_ids[ch].size() > max_cores_per_chip)
                        max_cores_per_chip = (int)chip_core_ids[ch].size();

                for (int idx = 0; idx < max_cores_per_chip; idx++) {
                    std::vector<int> dispatched;
                    for (int ch = 0; ch < num_chips; ch++) {
                        if (idx >= (int)chip_core_ids[ch].size()) continue;
                        int ci = chip_core_ids[ch][idx];
                        int cores_on_chip = (int)chip_core_ids[ch].size();
                        unsigned bk_args[18] = {
                            (unsigned)B_per_core,
                            (unsigned)T, (unsigned)D, (unsigned)H, (unsigned)FF,
                            (unsigned)V, (unsigned)L,
                            core_addrs[ci].dlogits,
                            core_addrs[ci].tokens,
                            chip_addrs[ch].wte,
                            chip_addrs[ch].layers,
                            chip_addrs[ch].lm_head,
                            core_addrs[ci].h_out,
                            (unsigned)(core_addrs[ci].h_out + (L+1)*BT*D),
                            (unsigned)(core_addrs[ci].h_out + (L+1)*BT*D + L*BT*D),
                            lr_bits,
                            core_addrs[ci].bk_scratch,
                            (unsigned)cores_on_chip
                        };
                        core_dispatch(ci, OP_FUSED_BACKWARD_ROWPAR, bk_args, 18);
                        dispatched.push_back(ci);
                    }
                    for (int ci : dispatched)
                        if (!core_wait(ci)) printf("BACKWARD TIMEOUT core %d\n", ci);
                }
            }

            // 7. Cross-chip weight sync (average weights across chips)
            if (num_chips > 1 && (step + 1) % 5 == 0) {
                // Download from chip 0
                int ref_ci = chip_core_ids[0][0];
                core_rd(ref_ci, chip_addrs[0].layers, packed.data(), L*lsz);
                core_rd(ref_ci, chip_addrs[0].lm_head, lm_w.data(), D*V);
                core_rd(ref_ci, chip_addrs[0].wte, wte.data(), V*D);

                for (int ch = 1; ch < num_chips; ch++) {
                    int ci2 = chip_core_ids[ch][0];
                    std::vector<float> tl(L*lsz), tm(D*V), tw(V*D);
                    core_rd(ci2, chip_addrs[ch].layers, tl.data(), L*lsz);
                    core_rd(ci2, chip_addrs[ch].lm_head, tm.data(), D*V);
                    core_rd(ci2, chip_addrs[ch].wte, tw.data(), V*D);
                    for (int i = 0; i < L*lsz; i++) packed[i] += tl[i];
                    for (int i = 0; i < D*V; i++) lm_w[i] += tm[i];
                    for (int i = 0; i < V*D; i++) wte[i] += tw[i];
                }
                float inv = 1.0f / num_chips;
                for (auto& x : packed) x *= inv;
                for (auto& x : lm_w) x *= inv;
                for (auto& x : wte) x *= inv;

                for (int ch = 0; ch < num_chips; ch++) {
                    int ci2 = chip_core_ids[ch][0];
                    core_wr(ci2, chip_addrs[ch].layers, packed.data(), L*lsz);
                    core_wr(ci2, chip_addrs[ch].lm_head, lm_w.data(), D*V);
                    core_wr(ci2, chip_addrs[ch].wte, wte.data(), V*D);
                }
            }

            if ((step+1) % 10 == 0) {
                auto now = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now - t0).count();
                double tok_s = (double)(step+1) * B_total * T / sec;
                printf("  E%d step %3d/%d: loss=%.4f acc=%.1f%% %.0f tok/s\n",
                       ep+1, step+1, steps, epoch_loss/(step+1),
                       100.0f*correct/total_tok, tok_s);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        printf("Epoch %d: loss=%.4f acc=%.1f%% %.1fs %.0f tok/s\n\n",
               ep+1, epoch_loss/steps, 100.0f*correct/total_tok, sec,
               (double)steps*B_total*T/sec);

        // === Sample generation (greedy, single core) — disabled for multi-core ===
        if (false && (ep == epochs - 1 || (ep+1) % 5 == 0)) {
            printf("--- Sample ---\n\"");
            std::vector<unsigned int> gen(T, 0);
            gen[0] = rng() % V;

            for (int pos = 1; pos < T; pos++) {
                core_wr(0, core_addrs[0].tokens, gen.data(), T);
                unsigned args[15] = {
                    1, (unsigned)T, (unsigned)D, (unsigned)H, (unsigned)FF,
                    (unsigned)V, (unsigned)L,
                    core_addrs[0].tokens,
                    chip_addrs[0].wte, chip_addrs[0].wpe,
                    chip_addrs[0].layers, chip_addrs[0].lm_head,
                    core_addrs[0].logits, core_addrs[0].h_out,
                    core_addrs[0].scratch
                };
                core_dispatch(0, OP_FUSED_FORWARD_ROWPAR, args, 15);
                core_wait(0);

                std::vector<float> logits(T * V);
                core_rd(0, core_addrs[0].logits, logits.data(), T * V);
                float* lp = &logits[(pos-1) * V];
                int best = 0;
                for (int c = 1; c < V; c++)
                    if (lp[c] > lp[best]) best = c;
                gen[pos] = best;
            }
            for (int t = 0; t < T; t++) printf("%c", itos_map[gen[t]]);
            printf("\"\n\n");
        }
    }

    // ======================== Cleanup ========================
    for (int ci = 0; ci < NC; ci++) {
        PL_Word ex = OP_EXIT;
        PL_WriteMemBlock(all_cores[ci].access, &ex,
                        DDR + all_cores[ci].core_id * CBS, 1);
    }
    usleep(500000);
    for (int ci = 0; ci < NC; ci++)
        PL_CloseAccess(all_cores[ci].access);
    for (int b = 0; b < num_boards; b++)
        PL_CloseBoardDesc(boards[b]);
    printf("=== DONE ===\n");
    return 0;
}
