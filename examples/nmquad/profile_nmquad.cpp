// Full NM QUAD profiling: board init, DDR transfer, forward, backward, latency
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <unistd.h>
extern "C" {
    struct PL_Board; struct PL_Access;
    typedef struct { int nm_id; int cluster_id; } PL_CoreNo;
    typedef unsigned int PL_Word, PL_Addr;
    int PL_GetBoardCount(unsigned int*);
    int PL_GetBoardDesc(unsigned int, PL_Board**);
    int PL_ResetBoard(PL_Board*);
    int PL_LoadInitCode(PL_Board*);
    int PL_GetAccess(PL_Board*, PL_CoreNo*, PL_Access**);
    int PL_CloseAccess(PL_Access*);
    int PL_LoadProgramFile(PL_Access*, const char*);
    int PL_ReadMemBlock(PL_Access*, PL_Word*, PL_Addr, unsigned int);
    int PL_WriteMemBlock(PL_Access*, const PL_Word*, PL_Addr, unsigned int);
    int PL_SetTimeout(unsigned int);
    int PL_CloseBoardDesc(PL_Board*);
}
#define DDR 0x00340000u
#define CBS 32
#define SA 30
#define DATA0 (DDR + 16*CBS)

struct Core { PL_Access* ac; int board; int core_id; };
static std::vector<Core> cores;
static PL_Board* bds[4];
static int nboards = 0;

auto tnow() { return std::chrono::steady_clock::now(); }
double tms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
    setbuf(stdout, NULL);
    printf("=== NM QUAD FULL PROFILING ===\n\n");

    auto t0 = tnow();
    PL_SetTimeout(60000);
    unsigned cnt; PL_GetBoardCount(&cnt);
    auto t1 = tnow();
    printf("PL_GetBoardCount: %.1f ms (found %u boards)\n", tms(t0,t1), cnt);

    for (unsigned b = 0; b < cnt && nboards < 4; b++) {
        auto tb0 = tnow();
        PL_GetBoardDesc(b, &bds[b]);
        PL_ResetBoard(bds[b]);
        PL_LoadInitCode(bds[b]);
        auto tb1 = tnow();
        printf("Board %d init: %.1f ms\n", b, tms(tb0,tb1));

        for (int cl=0;cl<4;cl++) for (int co=0;co<4;co++) {
            PL_CoreNo cn={co,cl}; PL_Access* a;
            if (PL_GetAccess(bds[b],&cn,&a)==0) {
                auto tl0 = tnow();
                int r = PL_LoadProgramFile(a, "dispatcher_nmquad_v3_simd.abs");
                auto tl1 = tnow();
                if (r==0) {
                    Core c; c.ac=a; c.board=b; c.core_id=(cl<<2)+co;
                    cores.push_back(c);
                    if ((int)cores.size()<=4) printf("  Load core B%d C%d: %.1f ms\n", b, c.core_id, tms(tl0,tl1));
                } else PL_CloseAccess(a);
            }
        }
        nboards++;
    }
    usleep(500000);
    int NC = (int)cores.size();
    printf("Total: %d cores across %d boards\n\n", NC, nboards);

    int D=128, H=4, FF=256, V=65, L=2, T=32, Bpc=1;
    int BT=Bpc*T, BH=Bpc*H, HD=D/H;
    int lsz = 4*D*D + 2*D*FF + D;

    std::mt19937 rng(42);
    auto rf=[&](int n){std::vector<float>v(n);std::normal_distribution<float>d(0,0.02f);for(auto&x:v)x=d(rng);return v;};
    auto wte=rf(V*D), wpe=rf(T*D), lm=rf(D*V);
    std::vector<float> packed(L*lsz);
    for(int i=0;i<L;i++){
        auto q=rf(D*D),k=rf(D*D),v=rf(D*D),o=rf(D*D),w1=rf(D*FF),w2=rf(FF*D);
        std::vector<float>g(D,1.0f);
        float*dst=packed.data()+i*lsz;
        memcpy(dst,q.data(),D*D*4);dst+=D*D; memcpy(dst,k.data(),D*D*4);dst+=D*D;
        memcpy(dst,v.data(),D*D*4);dst+=D*D; memcpy(dst,o.data(),D*D*4);dst+=D*D;
        memcpy(dst,w1.data(),D*FF*4);dst+=D*FF; memcpy(dst,w2.data(),FF*D*4);dst+=FF*D;
        memcpy(dst,g.data(),D*4);
    }
    int cache_w=(L+1)*BT*D+L*BT*D+L*BT*FF;
    int fwd_s=BT*D*5+BH*HD*T+BH*T*T+BT*D+BT*D+BT*FF+BT*D+BH*T*HD*2;
    int bwd_s=D*V+BT*D+BT*FF+BT*FF+BT*D;

    printf("--- DDR TRANSFER ---\n");
    std::vector<std::vector<int>> bc(nboards);
    for(int ci=0;ci<NC;ci++) bc[cores[ci].board].push_back(ci);

    struct BA { PL_Addr wte,wpe,layers,lm; };
    BA ba[4];
    struct CA { PL_Addr tok,log,hout,scr,dlog,bscr; };
    std::vector<CA> ca(NC);

    for(int b=0;b<nboards;b++){
        int fci=bc[b][0]; PL_Addr dp=DATA0;
        auto tw0=tnow();
        ba[b].wte=dp; PL_WriteMemBlock(cores[fci].ac,(PL_Word*)wte.data(),dp,V*D); dp+=V*D;
        ba[b].wpe=dp; PL_WriteMemBlock(cores[fci].ac,(PL_Word*)wpe.data(),dp,T*D); dp+=T*D;
        ba[b].layers=dp; PL_WriteMemBlock(cores[fci].ac,(PL_Word*)packed.data(),dp,L*lsz); dp+=L*lsz;
        ba[b].lm=dp; PL_WriteMemBlock(cores[fci].ac,(PL_Word*)lm.data(),dp,D*V); dp+=D*V;
        auto tw1=tnow();
        int wb=(V*D+T*D+L*lsz+D*V)*4;
        printf("Board %d weights: %.1f ms (%.1f KB, %.1f MB/s)\n", b, tms(tw0,tw1), wb/1024.0, wb/tms(tw0,tw1)/1000.0);
        for(int ci:bc[b]){
            ca[ci].tok=dp; dp+=Bpc*T; ca[ci].log=dp; dp+=BT*V;
            ca[ci].hout=dp; dp+=cache_w; ca[ci].scr=dp; dp+=fwd_s;
            ca[ci].dlog=dp; dp+=BT*V; ca[ci].bscr=dp; dp+=bwd_s;
        }
    }

    std::vector<unsigned> tokens(BT);
    for(int i=0;i<BT;i++) tokens[i]=rng()%V;
    auto ttu0=tnow();
    for(int ci=0;ci<NC;ci++) PL_WriteMemBlock(cores[ci].ac,(PL_Word*)tokens.data(),ca[ci].tok,BT);
    auto ttu1=tnow();
    printf("Token upload %d cores: %.1f ms\n\n", NC, tms(ttu0,ttu1));

    printf("--- FORWARD ---\n");
    auto do_fwd = [&](int ci) {
        int b=cores[ci].board;
        unsigned args[]={(unsigned)Bpc,(unsigned)T,(unsigned)D,(unsigned)H,(unsigned)FF,
            (unsigned)V,(unsigned)L,ca[ci].tok,ba[b].wte,ba[b].wpe,ba[b].layers,ba[b].lm,
            ca[ci].log,ca[ci].hout,ca[ci].scr};
        PL_Addr base=DDR+cores[ci].core_id*CBS;
        PL_WriteMemBlock(cores[ci].ac,(PL_Word*)args,base+1,15);
        PL_Word z=0; PL_WriteMemBlock(cores[ci].ac,&z,base+SA,1);
        PL_Word op=32; PL_WriteMemBlock(cores[ci].ac,&op,base,1);
    };
    auto do_wait = [&](int ci) -> bool {
        PL_Addr base=DDR+cores[ci].core_id*CBS;
        for(int i=0;i<10000000;i++){
            PL_Word s; PL_ReadMemBlock(cores[ci].ac,&s,base+SA,1);
            if(s==1) return true; if(s==2) return false; usleep(50);
        }
        return false;
    };

    // 1 core
    do_fwd(0); auto tf0=tnow(); do_wait(0); auto tf1=tnow();
    // re-measure (first was cold)
    do_fwd(0); auto tf2=tnow(); do_wait(0); auto tf3=tnow();
    printf("1 core fwd: %.1f ms (%.1f tok/s)\n", tms(tf2,tf3), BT*1000.0/tms(tf2,tf3));

    // 4 cores
    if(NC>=4){
        for(int ci=0;ci<4;ci++) do_fwd(ci);
        auto t2=tnow(); for(int ci=0;ci<4;ci++) do_wait(ci); auto t3=tnow();
        printf("4 cores fwd: %.1f ms (%.1f tok/s)\n", tms(t2,t3), 4*BT*1000.0/tms(t2,t3));
    }
    // 16 cores
    if(NC>=16){
        for(int ci=0;ci<16;ci++) do_fwd(ci);
        auto t2=tnow(); for(int ci=0;ci<16;ci++) do_wait(ci); auto t3=tnow();
        printf("16 cores fwd: %.1f ms (%.1f tok/s)\n", tms(t2,t3), 16*BT*1000.0/tms(t2,t3));
    }
    // ALL cores
    if(NC>16){
        for(int ci=0;ci<NC;ci++) do_fwd(ci);
        auto t2=tnow(); for(int ci=0;ci<NC;ci++) do_wait(ci); auto t3=tnow();
        printf("%d cores fwd: %.1f ms (%.1f tok/s)\n", NC, tms(t2,t3), NC*BT*1000.0/tms(t2,t3));
    }

    // Logits download
    printf("\n--- DOWNLOAD ---\n");
    {
        std::vector<float> logits(BT*V);
        auto td0=tnow();
        for(int ci=0;ci<NC;ci++) PL_ReadMemBlock(cores[ci].ac,(PL_Word*)logits.data(),ca[ci].log,BT*V);
        auto td1=tnow();
        printf("Logits download %d cores: %.1f ms (%.1f KB, %.1f MB/s)\n",
            NC, tms(td0,td1), NC*BT*V*4/1024.0, NC*BT*V*4.0/tms(td0,td1)/1000.0);
    }

    // Backward 1 core
    printf("\n--- BACKWARD ---\n");
    {
        std::vector<float> dl(BT*V, 0.01f);
        PL_WriteMemBlock(cores[0].ac,(PL_Word*)dl.data(),ca[0].dlog,BT*V);
        float lr=0.01f; unsigned lr_bits; memcpy(&lr_bits,&lr,4);
        unsigned bk[]={(unsigned)Bpc,(unsigned)T,(unsigned)D,(unsigned)H,(unsigned)FF,
            (unsigned)V,(unsigned)L,ca[0].dlog,ca[0].tok,ba[0].wte,ba[0].layers,ba[0].lm,
            ca[0].hout,(unsigned)(ca[0].hout+(L+1)*BT*D),(unsigned)(ca[0].hout+(L+1)*BT*D+L*BT*D),
            lr_bits,ca[0].bscr,1};
        PL_Addr base=DDR+cores[0].core_id*CBS;
        PL_WriteMemBlock(cores[0].ac,(PL_Word*)bk,base+1,18);
        PL_Word z=0; PL_WriteMemBlock(cores[0].ac,&z,base+SA,1);
        PL_Word op=33; PL_WriteMemBlock(cores[0].ac,&op,base,1);
        auto tb0=tnow(); do_wait(0); auto tb1=tnow();
        printf("1 core backward: %.1f ms\n", tms(tb0,tb1));
    }

    // Raw DDR latency
    printf("\n--- RAW DDR LATENCY ---\n");
    {
        PL_Word w=42;
        auto tp0=tnow();
        for(int i=0;i<1000;i++) PL_WriteMemBlock(cores[0].ac,&w,DATA0,1);
        auto tp1=tnow();
        printf("Write 1w x1000: %.1f ms (%.3f ms/call)\n", tms(tp0,tp1), tms(tp0,tp1)/1000.0);
        auto tr0=tnow();
        for(int i=0;i<1000;i++) PL_ReadMemBlock(cores[0].ac,&w,DATA0,1);
        auto tr1=tnow();
        printf("Read 1w x1000: %.1f ms (%.3f ms/call)\n", tms(tr0,tr1), tms(tr0,tr1)/1000.0);
        std::vector<float> buf(65536);
        auto tb0=tnow(); PL_WriteMemBlock(cores[0].ac,(PL_Word*)buf.data(),DATA0,65536); auto tb1=tnow();
        printf("Write 256KB: %.1f ms (%.1f MB/s)\n", tms(tb0,tb1), 256.0/tms(tb0,tb1)*1000.0);
        auto tb2=tnow(); PL_ReadMemBlock(cores[0].ac,(PL_Word*)buf.data(),DATA0,65536); auto tb3=tnow();
        printf("Read 256KB: %.1f ms (%.1f MB/s)\n", tms(tb2,tb3), 256.0/tms(tb2,tb3)*1000.0);
    }

    for(int ci=0;ci<NC;ci++){
        PL_Word ex=255; PL_WriteMemBlock(cores[ci].ac,&ex,DDR+cores[ci].core_id*CBS,1);
    }
    usleep(300000);
    for(int ci=0;ci<NC;ci++) PL_CloseAccess(cores[ci].ac);
    for(int b=0;b<nboards;b++) PL_CloseBoardDesc(bds[b]);
    printf("\n=== PROFILING DONE ===\n");
}
