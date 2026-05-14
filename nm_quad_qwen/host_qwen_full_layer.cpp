/* host_qwen_full_layer.cpp — verify COMPLETE Qwen3-4B layer (attn + FFN, 18 ops) */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cmath>
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
#define EPS 1.0e-6f
#define ROPE_BASE 1000000.0f

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
               mant &= 0x3FF; exp++;
               bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 0x1F) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
static float q4k_dot_h(const uint8_t *blk, const float *x) {
    uint16_t db, dmb; std::memcpy(&db, blk + 0, 2); std::memcpy(&dmb, blk + 2, 2);
    float d = fp16_to_fp32(db), dmin = fp16_to_fp32(dmb);
    const uint8_t *sc = blk + 4; const uint8_t *qs = blk + 16;
    float acc = 0; int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is, sc, &s, &m); float d1 = d * (float)s; float m1 = dmin * (float)m;
        get_scale_min_k4(is + 1, sc, &s, &m); float d2 = d * (float)s; float m2 = dmin * (float)m;
        for (int l = 0; l < 32; ++l) {
            acc += (d1 * (float)(qs[l] & 0xF) - m1) * x[j + l];
            acc += (d2 * (float)(qs[l] >> 4) - m2) * x[j + l + 32];
        }
        qs += 32; is += 2;
    }
    return acc;
}
static float q6k_dot_h(const uint8_t *blk, const float *x) {
    const uint8_t *ql = blk; const uint8_t *qh = blk + 128;
    const int8_t *sc = (const int8_t*)(blk + 192);
    uint16_t db; std::memcpy(&db, blk + 208, 2); float d = fp16_to_fp32(db);
    float acc = 0;
    for (int i = 0; i < 256; ++i) {
        int is = i / 16;
        int q_lo = (ql[(i%64) + 64*(i/128)] >> (4*((i/64)&1))) & 0xF;
        int q_hi = (qh[(i%32) + 32*(i/128)] >> (2*((i/32)&3))) & 0x3;
        acc += d * (float)sc[is] * (float)((q_lo | (q_hi << 4)) - 32) * x[i];
    }
    return acc;
}

struct GGUFReader {
    std::ifstream f;
    uint64_t off_an=0, off_qn=0, off_kn=0, off_q=0, off_k=0, off_v=0, off_o=0;
    uint64_t off_fn=0, off_g=0, off_u=0, off_d=0;
    bool open_(const char *p) { f.open(p, std::ios::binary); return f.is_open(); }
    template<typename T> bool rd(T &v) { f.read((char*)&v, sizeof(T)); return f.good(); }
    bool rd_str(std::string &s) { uint64_t n; if (!rd(n)) return false; s.resize(n); f.read(s.data(), n); return f.good(); }
    bool skip_value(uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
            case 2: case 3: f.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
            case 8: { std::string s; if (!rd_str(s)) return false; break; }
            case 9: { uint32_t at; uint64_t an; if(!rd(at)||!rd(an)) return false;
                      for (uint64_t i = 0; i < an; ++i) if (!skip_value(at)) return false; break; }
            default: return false;
        }
        return f.good();
    }
    bool parse() {
        char m[4]; rd(m); if (memcmp(m, "GGUF", 4) != 0) return false;
        uint32_t ver; rd(ver);
        uint64_t nt, nk; rd(nt); rd(nk);
        for (uint64_t i = 0; i < nk; ++i) { std::string k; rd_str(k); uint32_t t; rd(t); if (!skip_value(t)) return false; }
        struct TI { std::string name; uint32_t nd; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
        std::vector<TI> tt(nt);
        for (uint64_t i = 0; i < nt; ++i) {
            rd_str(tt[i].name); rd(tt[i].nd);
            tt[i].dims.resize(tt[i].nd);
            for (uint32_t d = 0; d < tt[i].nd; ++d) rd(tt[i].dims[d]);
            rd(tt[i].type); rd(tt[i].off);
        }
        uint64_t pos = f.tellg();
        uint64_t aln = 32;
        uint64_t base = (pos + aln - 1) & ~(aln - 1);
        for (auto &t : tt) {
            if (t.name == "blk.0.attn_norm.weight") off_an = base + t.off;
            else if (t.name == "blk.0.attn_q_norm.weight") off_qn = base + t.off;
            else if (t.name == "blk.0.attn_k_norm.weight") off_kn = base + t.off;
            else if (t.name == "blk.0.attn_q.weight" && t.type == 12) off_q = base + t.off;
            else if (t.name == "blk.0.attn_k.weight" && t.type == 12) off_k = base + t.off;
            else if (t.name == "blk.0.attn_v.weight" && t.type == 14) off_v = base + t.off;
            else if (t.name == "blk.0.attn_output.weight" && t.type == 12) off_o = base + t.off;
            else if (t.name == "blk.0.ffn_norm.weight") off_fn = base + t.off;
            else if (t.name == "blk.0.ffn_gate.weight" && t.type == 12) off_g = base + t.off;
            else if (t.name == "blk.0.ffn_up.weight" && t.type == 12) off_u = base + t.off;
            else if (t.name == "blk.0.ffn_down.weight" && t.type == 14) off_d = base + t.off;
        }
        return off_an && off_qn && off_kn && off_q && off_k && off_v && off_o
            && off_fn && off_g && off_u && off_d;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 15) { std::cerr << "Usage: " << argv[0] << " X Gan Gqn Gkn Wq Wk Wv Wo Gfn Wg Wu Wd Pos Xfinal POS\n"; return 1; }
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

    GGUFReader g;
    if (!g.open_(GGUF_PATH) || !g.parse()) { std::cerr << "gguf fail\n"; return 2; }
    std::ifstream raw(GGUF_PATH, std::ios::binary);

    auto load_fp32 = [&](uint64_t off, int n){
        raw.seekg(off);
        std::vector<float> b(n); raw.read((char*)b.data(), n * 4); return b;
    };
    auto load_bytes = [&](uint64_t off, int n){
        raw.seekg(off);
        std::vector<uint8_t> b(n); raw.read((char*)b.data(), n); return b;
    };
    std::vector<float> attn_norm = load_fp32(g.off_an, K_DIM);
    std::vector<float> q_norm    = load_fp32(g.off_qn, HEAD_DIM);
    std::vector<float> k_norm    = load_fp32(g.off_kn, HEAD_DIM);
    std::vector<float> ffn_norm  = load_fp32(g.off_fn, K_DIM);

    int M_HEADS = N_HEADS_SUB * HEAD_DIM;
    std::vector<uint8_t> Wq = load_bytes(g.off_q, M_HEADS * Q4K_ROW_BYTES);
    std::vector<uint8_t> Wk = load_bytes(g.off_k, M_HEADS * Q4K_ROW_BYTES);
    std::vector<uint8_t> Wv = load_bytes(g.off_v, M_HEADS * Q6K_ROW_BYTES);
    std::vector<uint8_t> Wgate = load_bytes(g.off_g, M_FFN * Q4K_ROW_BYTES);
    std::vector<uint8_t> Wup   = load_bytes(g.off_u, M_FFN * Q4K_ROW_BYTES);

    /* attn_output: real K=4096 (16 blocks). Subset takes 1 block per row. */
    int ao_real_row = (4096 / 256) * 144;
    std::vector<uint8_t> Wo(M_OUT * ATTN_OUT_ROW_BYTES);
    std::vector<uint8_t> tmp(ao_real_row);
    for (int r = 0; r < M_OUT; ++r) {
        raw.seekg(g.off_o + (uint64_t)r * ao_real_row);
        raw.read((char*)tmp.data(), ao_real_row);
        std::memcpy(Wo.data() + r * ATTN_OUT_ROW_BYTES, tmp.data(), ATTN_OUT_ROW_BYTES);
    }
    /* ffn_down: real K=9728 (38 blocks Q6_K). Subset 1 block per row. */
    int fd_real_row = (9728 / 256) * 210;
    std::vector<uint8_t> Wd(M_OUT * FFN_DOWN_ROW_BYTES);
    std::vector<uint8_t> tmp2(fd_real_row);
    for (int r = 0; r < M_OUT; ++r) {
        raw.seekg(g.off_d + (uint64_t)r * fd_real_row);
        raw.read((char*)tmp2.data(), fd_real_row);
        std::memcpy(Wd.data() + r * FFN_DOWN_ROW_BYTES, tmp2.data(), FFN_DOWN_ROW_BYTES);
    }

    std::vector<float> x(K_DIM);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < K_DIM; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    /* HOST REFERENCE — full pipeline */
    float sum = 0;
    for (int i = 0; i < K_DIM; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum / (float)K_DIM + EPS);
    std::vector<float> y(K_DIM);
    for (int i = 0; i < K_DIM; ++i) y[i] = x[i] * inv_rms * attn_norm[i];

    std::vector<float> q(M_HEADS), k(M_HEADS), v(M_HEADS);
    for (int r = 0; r < M_HEADS; ++r) {
        float aq=0, ak=0, av=0;
        for (int b = 0; b < BLOCKS_PER_ROW; ++b) {
            aq += q4k_dot_h(Wq.data() + r * Q4K_ROW_BYTES + b * 144, y.data() + b * 256);
            ak += q4k_dot_h(Wk.data() + r * Q4K_ROW_BYTES + b * 144, y.data() + b * 256);
            av += q6k_dot_h(Wv.data() + r * Q6K_ROW_BYTES + b * 210, y.data() + b * 256);
        }
        q[r]=aq; k[r]=ak; v[r]=av;
    }
    for (int h = 0; h < N_HEADS_SUB; ++h) {
        float qs=0, ks=0;
        for (int i = 0; i < HEAD_DIM; ++i) { qs += q[h*HEAD_DIM+i]*q[h*HEAD_DIM+i]; ks += k[h*HEAD_DIM+i]*k[h*HEAD_DIM+i]; }
        float qi = 1.0f/std::sqrt(qs/(float)HEAD_DIM + EPS);
        float ki = 1.0f/std::sqrt(ks/(float)HEAD_DIM + EPS);
        for (int i = 0; i < HEAD_DIM; ++i) {
            q[h*HEAD_DIM+i] = q[h*HEAD_DIM+i] * qi * q_norm[i];
            k[h*HEAD_DIM+i] = k[h*HEAD_DIM+i] * ki * k_norm[i];
        }
        for (int i = 0; i < HEAD_DIM; i += 2) {
            float theta = 1.0f / std::pow(ROPE_BASE, (float)i / (float)HEAD_DIM);
            float angle = (float)pos_int * theta;
            float c = std::cos(angle), s = std::sin(angle);
            float q0 = q[h*HEAD_DIM+i], q1 = q[h*HEAD_DIM+i+1];
            float k0 = k[h*HEAD_DIM+i], k1 = k[h*HEAD_DIM+i+1];
            q[h*HEAD_DIM+i] = q0*c - q1*s; q[h*HEAD_DIM+i+1] = q0*s + q1*c;
            k[h*HEAD_DIM+i] = k0*c - k1*s; k[h*HEAD_DIM+i+1] = k0*s + k1*c;
        }
    }
    std::vector<float> attn_concat(M_HEADS);
    for (int i = 0; i < M_HEADS; ++i) attn_concat[i] = v[i];

    std::vector<float> attn_out(M_OUT), x_post(M_OUT);
    for (int r = 0; r < M_OUT; ++r)
        attn_out[r] = q4k_dot_h(Wo.data() + r * ATTN_OUT_ROW_BYTES, attn_concat.data());
    for (int i = 0; i < M_OUT; ++i) x_post[i] = x[i] + attn_out[i];

    /* FFN */
    sum = 0;
    for (int i = 0; i < K_DIM; ++i) sum += x_post[i] * x_post[i];
    float inv_rms2 = 1.0f / std::sqrt(sum / (float)K_DIM + EPS);
    std::vector<float> y2(K_DIM);
    for (int i = 0; i < K_DIM; ++i) y2[i] = x_post[i] * inv_rms2 * ffn_norm[i];

    std::vector<float> g_(M_FFN), u_(M_FFN), mul(M_FFN);
    for (int r = 0; r < M_FFN; ++r) {
        float ag=0, au=0;
        for (int b = 0; b < BLOCKS_PER_ROW; ++b) {
            ag += q4k_dot_h(Wgate.data() + r * Q4K_ROW_BYTES + b * 144, y2.data() + b * 256);
            au += q4k_dot_h(Wup.data()   + r * Q4K_ROW_BYTES + b * 144, y2.data() + b * 256);
        }
        g_[r] = ag; u_[r] = au;
        float silu = ag / (1.0f + std::exp(-ag));
        mul[r] = silu * au;
    }
    std::vector<float> ffn_out(M_OUT), x_final_ref(M_OUT);
    for (int r = 0; r < M_OUT; ++r)
        ffn_out[r] = q6k_dot_h(Wd.data() + r * FFN_DOWN_ROW_BYTES, mul.data());
    for (int i = 0; i < M_OUT; ++i) x_final_ref[i] = x_post[i] + ffn_out[i];

    std::cerr << "[host] x_final[0..3]=" << x_final_ref[0] << " " << x_final_ref[1] << " "
              << x_final_ref[2] << " " << x_final_ref[3] << "\n";

    /* NMC */
    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    auto upload = [&](const uint8_t *buf, int n, PL_Addr addr){
        std::vector<PL_Word> w(n);
        for (int i = 0; i < n; ++i) w[i] = (PL_Word)buf[i];
        PL_WriteMemBlock(acc, w.data(), addr, n);
    };
    upload(Wq.data(), Wq.size(), A_Wq);
    upload(Wk.data(), Wk.size(), A_Wk);
    upload(Wv.data(), Wv.size(), A_Wv);
    upload(Wo.data(), Wo.size(), A_Wo);
    upload(Wgate.data(), Wgate.size(), A_Wg);
    upload(Wup.data(), Wup.size(), A_Wu);
    upload(Wd.data(), Wd.size(), A_Wd);
    auto upload_f = [&](const float *buf, int n, PL_Addr addr){
        std::vector<PL_Word> w(n);
        for (int i = 0; i < n; ++i) std::memcpy(&w[i], &buf[i], 4);
        PL_WriteMemBlock(acc, w.data(), addr, n);
    };
    upload_f(attn_norm.data(), K_DIM,    A_Gan);
    upload_f(q_norm.data(),    HEAD_DIM, A_Gqn);
    upload_f(k_norm.data(),    HEAD_DIM, A_Gkn);
    upload_f(ffn_norm.data(),  K_DIM,    A_Gfn);
    PL_Word pw = (PL_Word)pos_int;
    PL_WriteMemBlock(acc, &pw, A_Pos, 1);
    upload_f(x.data(), K_DIM, A_X);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> ow(M_OUT);
    PL_ReadMemBlock(acc, ow.data(), A_Xf, M_OUT);
    std::vector<float> xf_n(M_OUT);
    for (int i = 0; i < M_OUT; ++i) std::memcpy(&xf_n[i], &ow[i], 4);

    float maxd = 0; int worst = -1;
    for (int i = 0; i < M_OUT; ++i) {
        float d = std::fabs(xf_n[i] - x_final_ref[i]);
        if (d > maxd) { maxd = d; worst = i; }
    }
    std::cout << "[verify] FULL LAYER x_final max_diff=" << maxd << " (row " << worst << ")\n";
    std::cout << "[perf] wall: " << ms << " ms\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-3f ? 0 : 100;
}
