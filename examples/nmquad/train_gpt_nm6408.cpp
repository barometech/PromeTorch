// ============================================================================
// NanoGPT — FUSED forward on NM6408. ONE DDR call per forward pass.
// ============================================================================
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
    int PL_GetBoardCount(unsigned int*); int PL_GetBoardDesc(unsigned int, PL_Board**);
    int PL_CloseBoardDesc(PL_Board*); int PL_ResetBoard(PL_Board*);
    int PL_LoadInitCode(PL_Board*);
    int PL_GetAccess(PL_Board*, PL_CoreNo*, PL_Access**);
    int PL_CloseAccess(PL_Access*);
    int PL_LoadProgramFile(PL_Access*, const char*);
    int PL_ReadMemBlock(PL_Access*, PL_Word*, PL_Addr, unsigned int);
    int PL_WriteMemBlock(PL_Access*, const PL_Word*, PL_Addr, unsigned int);
    int PL_SetTimeout(unsigned int);
}

#define DDR 0x00340000u
#define CBS 32
#define SA 30
#define DATA0 (DDR + 16*CBS)

static PL_Access* ac[16];
static PL_Board* bd;
static int nc = 0;
static PL_Addr dp;

PL_Addr da(int n) { PL_Addr a = dp; dp += n; return a; }
void wr(PL_Addr a, const void* d, int n) { PL_WriteMemBlock(ac[0], (const PL_Word*)d, a, n); }
void rd(PL_Addr a, void* d, int n) { PL_ReadMemBlock(ac[0], (PL_Word*)d, a, n); }
PL_Addr up(const void* d, int n) { PL_Addr a = da(n); wr(a, d, n); return a; }

bool cmd(int core, unsigned op, const unsigned* args, int nargs) {
    PL_Addr base = DDR + core * CBS;
    if (nargs > 0) PL_WriteMemBlock(ac[core], (PL_Word*)args, base+1, nargs);
    PL_Word z = 0; PL_WriteMemBlock(ac[core], &z, base+SA, 1);
    PL_Word c = op; PL_WriteMemBlock(ac[core], &c, base, 1);
    int max_polls = (op == 30) ? 5000000 : 500000; // 250s for fused, 25s for others
    for (int i = 0; i < max_polls; i++) {
        PL_Word s; PL_ReadMemBlock(ac[core], &s, base+SA, 1);
        if (s == 1) return true;
        usleep(50);
    }
    printf("TIMEOUT core %d op %d\n", core, op);
    return false;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    std::string data_path = "tiny_shakespeare.txt", disp = "dispatcher_nmquad.abs";
    int epochs=10, T=32, B=16, D=128, H=4, L=2, steps=200;
    float lr = 0.01f;

    for (int i=1;i<argc;i++) {
        std::string a=argv[i];
        if (a=="--data") data_path=argv[++i];
        else if (a=="--dispatcher") disp=argv[++i];
        else if (a=="--epochs") epochs=atoi(argv[++i]);
        else if (a=="--ctx") T=atoi(argv[++i]);
        else if (a=="--batch") B=atoi(argv[++i]);
        else if (a=="--dim") D=atoi(argv[++i]);
        else if (a=="--layers") L=atoi(argv[++i]);
        else if (a=="--lr") lr=atof(argv[++i]);
        else if (a=="--steps") steps=atoi(argv[++i]);
    }
    int FF=D*2, HD=D/H;

    // Load text
    std::ifstream f(data_path);
    std::string text((std::istreambuf_iterator<char>(f)), {});
    std::vector<char> chars;
    for (char c : text) if (std::find(chars.begin(),chars.end(),c)==chars.end()) chars.push_back(c);
    std::sort(chars.begin(), chars.end());
    int V = chars.size();
    int stoi[256]={}; char itos[256]={};
    for (int i=0;i<V;i++) { stoi[(unsigned char)chars[i]]=i; itos[i]=chars[i]; }
    std::vector<int> data(text.size());
    for (size_t i=0;i<text.size();i++) data[i]=stoi[(unsigned char)text[i]];
    printf("Data: %zu chars, V=%d\n", data.size(), V);

    // Init
    PL_SetTimeout(30000);
    unsigned cnt; PL_GetBoardCount(&cnt);
    PL_GetBoardDesc(0,&bd); PL_ResetBoard(bd); PL_LoadInitCode(bd);
    for (int cl=0;cl<4;cl++) for (int co=0;co<4;co++) {
        PL_CoreNo cn={co,cl}; PL_Access* a;
        if (PL_GetAccess(bd,&cn,&a)==0)
            if (PL_LoadProgramFile(a,disp.c_str())==0) ac[nc++]=a;
            else PL_CloseAccess(a);
    }
    usleep(500000);
    printf("NM QUAD: %d cores, FUSED forward kernels\n", nc);

    // Weights
    std::mt19937 rng(42);
    auto rf=[&](int n){std::vector<float>v(n);std::normal_distribution<float>d(0,0.02f);for(auto&x:v)x=d(rng);return v;};

    auto wte=rf(V*D), wpe=rf(T*D), lm_w=rf(D*V);
    // Pack layer weights: [Wq, Wk, Wv, Wo, W1, W2, g] per layer
    int lsz = 4*D*D + 2*D*FF + D;
    std::vector<float> packed_layers(L * lsz);
    for (int i=0;i<L;i++) {
        auto Wq=rf(D*D),Wk=rf(D*D),Wv=rf(D*D),Wo=rf(D*D),W1=rf(D*FF),W2=rf(FF*D);
        std::vector<float> g(D, 1.0f);
        float* dst = packed_layers.data() + i*lsz;
        memcpy(dst, Wq.data(), D*D*4); dst += D*D;
        memcpy(dst, Wk.data(), D*D*4); dst += D*D;
        memcpy(dst, Wv.data(), D*D*4); dst += D*D;
        memcpy(dst, Wo.data(), D*D*4); dst += D*D;
        memcpy(dst, W1.data(), D*FF*4); dst += D*FF;
        memcpy(dst, W2.data(), FF*D*4); dst += FF*D;
        memcpy(dst, g.data(), D*4);
    }

    int np = V*D + T*D + D*V + L*lsz;
    printf("NanoGPT: %dK params, L=%d D=%d H=%d T=%d B=%d\n", np/1000, L, D, H, T, B);

    // Upload to DDR
    dp = DATA0;
    PL_Addr wte_a = up(wte.data(), V*D);
    PL_Addr wpe_a = up(wpe.data(), T*D);
    PL_Addr layers_a = up(packed_layers.data(), L*lsz);
    PL_Addr lm_a = up(lm_w.data(), D*V);

    int BT = B*T;
    // Pre-allocate output areas
    PL_Addr logits_a = da(BT*V);
    // h_out extended: h_cache[L+1][BT*D] + hn_cache[L][BT*D] + ff1r_cache[L][BT*FF]
    PL_Addr h_out_a = da((L+1)*BT*D + L*BT*D + L*BT*FF);
    PL_Addr tokens_a = da(B*T);
    PL_Addr wend = dp;

    printf("DDR: weights=%dKB scratch starts at 0x%X\n", (int)((wend-DATA0)*4/1024), 0x00350000);
    printf("\nTraining: %d epochs x %d steps, lr=%.3f\n", epochs, steps, lr);
    printf("FUSED: entire forward = 1 NM6408 call (ZERO DDR round-trips)\n\n");


    for (int ep=0; ep<epochs; ep++) {
        float eloss=0; int correct=0,total=0;
        auto t0 = std::chrono::steady_clock::now();

        for (int step=0; step<steps; step++) {
            // Host: token indices only
            std::vector<unsigned int> tokens(BT);
            std::vector<int> tgt(BT);
            for (int b=0;b<B;b++) {
                int idx=rng()%(data.size()-T-1);
                for (int t=0;t<T;t++) { tokens[b*T+t]=data[idx+t]; tgt[b*T+t]=data[idx+t+1]; }
            }

            // Upload tokens
            wr(tokens_a, tokens.data(), BT);

            // === ONE FUSED FORWARD CALL — entire transformer on NM6408 ===
            // Scratch area after all pre-allocated buffers
            PL_Addr scratch_a = wend + BT*V + BT*D + BT + 0x100; // after logits, h_out, tokens
            unsigned args[] = {
                (unsigned)B, (unsigned)T, (unsigned)D, (unsigned)H, (unsigned)FF,
                (unsigned)V, (unsigned)L,
                tokens_a, wte_a, wpe_a, layers_a, lm_a,
                logits_a, h_out_a, scratch_a
            };
            if (!cmd(0, 30, args, 15)) { // OP_FUSED_FORWARD = 30
                printf("FUSED FORWARD FAILED step %d\n", step);
                break;
            }

            // Read logits (only output needed on host for loss)
            std::vector<float> logits(BT*V);
            rd(logits_a, logits.data(), BT*V);

            // Loss + dlogits (element-wise, no matmul — only on host)
            std::vector<float> dl(BT*V);
            float bloss=0;
            for (int bt=0;bt<BT;bt++) {
                float*l=&logits[bt*V]; float mx=*std::max_element(l,l+V),sm=0;
                for (int c=0;c<V;c++){l[c]=expf(l[c]-mx);sm+=l[c];}
                for (int c=0;c<V;c++){l[c]/=sm;dl[bt*V+c]=l[c];}
                bloss-=logf(l[tgt[bt]]+1e-8f);
                dl[bt*V+tgt[bt]]-=1.0f;
                if (std::distance(l,std::max_element(l,l+V))==tgt[bt]) correct++;
                total++;
            }
            for (auto&d:dl) d/=B;  // normalize by batch only, not sequence length
            eloss+=bloss/BT;

            // === FUSED BACKWARD — all weight updates on NM6408 ===
            dp = wend + (L+1)*BT*D + L*BT*D + L*BT*FF + BT*V + BT + 0x200;
            PL_Addr dl_a = up(dl.data(), BT*V);
            PL_Addr bk_scratch = da(D*V + BT*D + BT*FF + BT*FF + BT*D); // scratch for backward

            unsigned lr_bits; memcpy(&lr_bits, &lr, 4);

            unsigned bk_args[] = {
                (unsigned)B, (unsigned)T, (unsigned)D, (unsigned)H, (unsigned)FF,
                (unsigned)V, (unsigned)L,
                dl_a, tokens_a, wte_a, layers_a, lm_a,
                h_out_a,                    // h_cache
                (unsigned)(h_out_a + (L+1)*BT*D),  // hn_cache
                (unsigned)(h_out_a + (L+1)*BT*D + L*BT*D), // ff1r_cache
                lr_bits,
                bk_scratch
            };
            if (!cmd(0, 31, bk_args, 17)) { // OP_FUSED_BACKWARD
                printf("BACKWARD FAILED step %d\n", step);
            }

            if ((step+1)%10==0) {
                auto now=std::chrono::steady_clock::now();
                double sec=std::chrono::duration<double>(now-t0).count();
                printf("  E%d step %d/%d: loss=%.4f acc=%.1f%% %.0f tok/s\n",
                       ep+1,step+1,steps,eloss/(step+1),100.0f*correct/total,(step+1)*BT/sec);
            }
        }
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        printf("Epoch %d: loss=%.4f acc=%.1f%% %.1fs %.0f tok/s\n\n",
               ep+1,eloss/steps,100.0f*correct/total,sec,steps*BT/sec);
    }

    for (int i=0;i<nc;i++){PL_Word ex=255;PL_WriteMemBlock(ac[i],&ex,DDR+i*CBS,1);}
    usleep(300000);
    for (int i=0;i<nc;i++) PL_CloseAccess(ac[i]);
    PL_CloseBoardDesc(bd);
    printf("=== DONE ===\n");
}
