/* host_stream_layers.cpp — streaming multi-layer driver
 * Loads NMC kernel ONCE, then drives 36 layers via stream flags.
 * Eliminates 36× PL_LoadProgramFile overhead.
 */
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <functional>
#include "nm_quad_load.h"
#include "io_host.h"

static const char *NMC_PART  = "./nmc_part.abs";
static const char *GGUF_PATH = "/home/<user>/gguf/qwen3-4b-q4km.gguf";

#define K_DIM 2560
#define HEAD_DIM 128
#define N_HEADS_SUB 2
#define ATTN_OUT_K (N_HEADS_SUB * HEAD_DIM)
#define M_OUT 2560
#define M_FFN 1024
#define BLOCKS_PER_ROW (K_DIM / 256)
#define Q4K_ROW_BYTES (BLOCKS_PER_ROW * 144)
#define Q6K_ROW_BYTES (BLOCKS_PER_ROW * 210)
#define ATTN_OUT_BLOCKS (ATTN_OUT_K / 256)
#define ATTN_OUT_ROW_BYTES (ATTN_OUT_BLOCKS * 144)
#define FFN_DOWN_BLOCKS (M_FFN / 256)
#define FFN_DOWN_ROW_BYTES (FFN_DOWN_BLOCKS * 210)
#define M_HEADS (N_HEADS_SUB * HEAD_DIM)

struct GTI { std::string name; uint32_t type; uint64_t off; };
static std::vector<GTI> gguf_tensors;
static uint64_t gguf_base = 0;

static bool gguf_parse() {
    std::ifstream f(GGUF_PATH, std::ios::binary);
    char m[4]; f.read(m, 4);
    if (std::memcmp(m, "GGUF", 4) != 0) return false;
    uint32_t ver; f.read((char*)&ver, 4);
    uint64_t nt, nk; f.read((char*)&nt, 8); f.read((char*)&nk, 8);
    auto rdstr = [&](std::string &s) {
        uint64_t n; f.read((char*)&n, 8); s.resize(n); f.read(s.data(), n);
    };
    std::function<bool(uint32_t)> skip = [&](uint32_t t) -> bool {
        switch (t) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
            case 2: case 3: f.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
            case 8: { std::string s; rdstr(s); break; }
            case 9: {
                uint32_t at; uint64_t an;
                f.read((char*)&at, 4); f.read((char*)&an, 8);
                for (uint64_t i = 0; i < an; ++i) if (!skip(at)) return false;
                break;
            }
            default: return false;
        }
        return f.good();
    };
    for (uint64_t i = 0; i < nk; ++i) {
        std::string k; rdstr(k);
        uint32_t t; f.read((char*)&t, 4);
        if (!skip(t)) return false;
    }
    gguf_tensors.resize(nt);
    for (uint64_t i = 0; i < nt; ++i) {
        rdstr(gguf_tensors[i].name);
        uint32_t nd; f.read((char*)&nd, 4);
        std::vector<uint64_t> dims(nd);
        for (uint32_t d = 0; d < nd; ++d) f.read((char*)&dims[d], 8);
        f.read((char*)&gguf_tensors[i].type, 4);
        f.read((char*)&gguf_tensors[i].off, 8);
    }
    uint64_t pos = f.tellg();
    gguf_base = (pos + 31) & ~31ULL;
    return true;
}

static uint64_t gguf_off(const std::string &n) {
    for (auto &t : gguf_tensors) if (t.name == n) return gguf_base + t.off;
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 15) {
        std::cerr << "Usage: " << argv[0]
                  << " X Gan Gqn Gkn Wq Wk Wv Wo Gfn Wg Wu Wd Pos Xf POS\n";
        return 1;
    }
    int ai = 1;
    PL_Addr A_X    = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Gan  = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Gqn  = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Gkn  = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Wq   = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Wk   = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Wv   = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Wo   = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Gfn  = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Wg   = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Wu   = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Wd   = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Pos  = std::strtoul(argv[ai++], nullptr, 16);
    PL_Addr A_Xf   = std::strtoul(argv[ai++], nullptr, 16);
    int pos_int    = std::atoi(argv[ai++]);

    PL_Addr A_STREAM_GO   = 0x02000004;
    PL_Addr A_STREAM_DONE = 0x02000002;
    PL_Addr A_STREAM_EXIT = 0x02000000;

    if (!gguf_parse()) { std::cerr << "gguf parse fail\n"; return 2; }
    std::ifstream raw(GGUF_PATH, std::ios::binary);

    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    PL_Word zero = 0;
    PL_WriteMemBlock(acc, &zero, A_STREAM_GO, 1);
    PL_WriteMemBlock(acc, &zero, A_STREAM_DONE, 1);
    PL_WriteMemBlock(acc, &zero, A_STREAM_EXIT, 1);

    PL_Word pw = (PL_Word)pos_int;
    PL_WriteMemBlock(acc, &pw, A_Pos, 1);

    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);

    auto load_fp32 = [&](uint64_t off, int n) {
        raw.seekg(off);
        std::vector<float> v(n);
        raw.read((char*)v.data(), n * 4);
        return v;
    };
    auto load_bytes_strided = [&](uint64_t off, int rows, int row_stride_real, int row_take) {
        std::vector<uint8_t> v(rows * row_take);
        for (int r = 0; r < rows; ++r) {
            raw.seekg(off + (uint64_t)r * row_stride_real);
            raw.read((char*)(v.data() + r * row_take), row_take);
        }
        return v;
    };
    auto load_bytes = [&](uint64_t off, size_t n) {
        raw.seekg(off);
        std::vector<uint8_t> v(n);
        raw.read((char*)v.data(), n);
        return v;
    };
    auto upload_bytes = [&](const uint8_t *buf, int n, PL_Addr addr) {
        std::vector<PL_Word> w(n);
        for (int i = 0; i < n; ++i) w[i] = (PL_Word)buf[i];
        PL_WriteMemBlock(acc, w.data(), addr, n);
    };
    auto upload_fp = [&](const float *buf, int n, PL_Addr addr) {
        std::vector<PL_Word> w(n);
        for (int i = 0; i < n; ++i) std::memcpy(&w[i], &buf[i], 4);
        PL_WriteMemBlock(acc, w.data(), addr, n);
    };

    std::vector<float> x(K_DIM);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < K_DIM; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    auto t0 = std::chrono::steady_clock::now();
    int total_layers = 36;
    int real_attn_row = (4096 / 256) * 144;
    int real_fd_row   = (9728 / 256) * 210;

    for (int layer = 0; layer < total_layers; ++layer) {
        char buf[64];
        std::snprintf(buf, 64, "blk.%d.", layer);
        std::string p = buf;

        auto Gan = load_fp32(gguf_off(p + "attn_norm.weight"), K_DIM);
        auto Gqn = load_fp32(gguf_off(p + "attn_q_norm.weight"), HEAD_DIM);
        auto Gkn = load_fp32(gguf_off(p + "attn_k_norm.weight"), HEAD_DIM);
        auto Gfn = load_fp32(gguf_off(p + "ffn_norm.weight"), K_DIM);
        auto Wq  = load_bytes(gguf_off(p + "attn_q.weight"), M_HEADS * Q4K_ROW_BYTES);
        auto Wk  = load_bytes(gguf_off(p + "attn_k.weight"), M_HEADS * Q4K_ROW_BYTES);
        auto Wv  = load_bytes(gguf_off(p + "attn_v.weight"), M_HEADS * Q6K_ROW_BYTES);
        auto Wgate = load_bytes(gguf_off(p + "ffn_gate.weight"), M_FFN * Q4K_ROW_BYTES);
        auto Wup   = load_bytes(gguf_off(p + "ffn_up.weight"), M_FFN * Q4K_ROW_BYTES);
        auto Wo  = load_bytes_strided(gguf_off(p + "attn_output.weight"), M_OUT, real_attn_row, ATTN_OUT_ROW_BYTES);
        auto Wd  = load_bytes_strided(gguf_off(p + "ffn_down.weight"), M_OUT, real_fd_row, FFN_DOWN_ROW_BYTES);

        upload_fp(Gan.data(), K_DIM, A_Gan);
        upload_fp(Gqn.data(), HEAD_DIM, A_Gqn);
        upload_fp(Gkn.data(), HEAD_DIM, A_Gkn);
        upload_fp(Gfn.data(), K_DIM, A_Gfn);
        upload_bytes(Wq.data(), Wq.size(), A_Wq);
        upload_bytes(Wk.data(), Wk.size(), A_Wk);
        upload_bytes(Wv.data(), Wv.size(), A_Wv);
        upload_bytes(Wo.data(), Wo.size(), A_Wo);
        upload_bytes(Wgate.data(), Wgate.size(), A_Wg);
        upload_bytes(Wup.data(), Wup.size(), A_Wu);
        upload_bytes(Wd.data(), Wd.size(), A_Wd);
        upload_bytes(Wv.data(), Wv.size(), A_Wv);  /* re-upload last */
        upload_fp(x.data(), K_DIM, A_X);

        PL_Word go_val = layer + 1;
        PL_WriteMemBlock(acc, &go_val, A_STREAM_GO, 1);

        for (;;) {
            PL_Word done_val;
            PL_ReadMemBlock(acc, &done_val, A_STREAM_DONE, 1);
            if ((int)done_val >= layer + 1) break;
        }

        std::vector<PL_Word> xfw(M_OUT);
        PL_ReadMemBlock(acc, xfw.data(), A_Xf, M_OUT);
        for (int i = 0; i < M_OUT; ++i) std::memcpy(&x[i], &xfw[i], 4);

        if (layer % 6 == 0) std::cerr << "[stream] layer=" << layer << " done\n";
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[stream] 36 layers wall=" << ms << " ms = " << (ms / 1000.0) << " sec\n";

    PL_Word one = 1;
    PL_WriteMemBlock(acc, &one, A_STREAM_EXIT, 1);

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return 0;
}
