// ============================================================================
// dispatcher_nmquad_v2.cpp — 16-core parallel fused transformer
// ============================================================================
// Core 0: coordinator — runs fused forward/backward, dispatches matmul to workers
// Cores 1-15: workers — poll DDR for matmul tasks from core 0
//
// Inter-core protocol via DDR:
//   Worker cmd block: DDR_BASE + core_index * 32
//   Worker reads mem[0] for opcode, executes, sets mem[STATUS]=1, mem[0]=NOP
//   Core 0 writes tasks to worker cmd blocks, polls their status
//
// This gives 16x parallel matmul INSIDE fused forward — zero host round-trips.

#include "nm6408load_nmc.h"

extern "C" {
    void nmppmMul_mm_32f(float* A, int nHeight1, int nStride1,
                         float* B, int nWidth1, int nStride2,
                         float* C, int nWidth2, int nStrideDst, int bPlusDst);
}

#define DDR_BASE      0x00340000
#define CMD_BLOCK_SIZE 32
#define STATUS_ADDR    30
#define WATCHDOG_ADDR  31

#define OP_NOP           0
#define OP_MATMUL        1
#define OP_ADD           2
#define OP_RELU          4
#define OP_MATMUL_PARTIAL 22
#define OP_FUSED_FORWARD 30
#define OP_FUSED_BACKWARD 31
#define OP_EXIT          255

// Shared DDR area for inter-core communication
// Each worker's cmd block is at DDR_BASE + core_index * CMD_BLOCK_SIZE

static volatile unsigned int* my_mem;
static int my_core_index;

// ============================================================
// WORKER CODE (cores 1-15): poll cmd block, execute matmul
// ============================================================
static void worker_loop() {
    unsigned int watchdog = 0;
    while (1) {
        watchdog++;
        my_mem[WATCHDOG_ADDR] = watchdog;

        asm volatile("" ::: "memory");
        unsigned int op = my_mem[0];
        if (op == OP_NOP) continue;
        if (op == OP_EXIT) { my_mem[STATUS_ADDR] = 1; asm volatile("" ::: "memory"); my_mem[0] = OP_NOP; break; }

        my_mem[STATUS_ADDR] = 0;
        asm volatile("" ::: "memory");

        if (op == OP_MATMUL_PARTIAL || op == OP_MATMUL) {
            unsigned int M = my_mem[1], K = my_mem[2], N = my_mem[3];
            float* A = (float*)my_mem[4];
            float* B = (float*)my_mem[5];
            float* C = (float*)my_mem[6];
            nmppmMul_mm_32f(A, M, K, B, K, N, C, N, N, 0);
        } else if (op == OP_ADD) {
            unsigned int count = my_mem[1];
            float* a = (float*)my_mem[2];
            float* b = (float*)my_mem[3];
            float* out = (float*)my_mem[4];
            for (unsigned int i = 0; i < count; i++) out[i] = a[i] + b[i];
        } else if (op == OP_RELU) {
            unsigned int count = my_mem[1];
            float* x = (float*)my_mem[2];
            float* y = (float*)my_mem[3];
            for (unsigned int i = 0; i < count; i++) y[i] = x[i] > 0 ? x[i] : 0;
        }

        asm volatile("" ::: "memory");
        my_mem[STATUS_ADDR] = 1;
        asm volatile("" ::: "memory");
        my_mem[0] = OP_NOP;
        asm volatile("" ::: "memory");
    }
}

// ============================================================
// COORDINATOR CODE (core 0): dispatch matmul to workers
// ============================================================

// Dispatch matmul row-split across N workers
static void parallel_mm(float* A, float* B, float* C, int M, int K, int N, int n_workers) {
    if (n_workers <= 1 || M < 2) {
        nmppmMul_mm_32f(A, M, K, B, K, N, C, N, N, 0);
        return;
    }
    int nw = (n_workers < M) ? n_workers : M;
    int rpc = (M + nw - 1) / nw;

    // Dispatch to workers (cores 1..nw)
    for (int w = 0; w < nw; w++) {
        int rs = w * rpc, re = rs + rpc;
        if (re > M) re = M;
        if (rs >= M) break;
        int m_slice = re - rs;

        volatile unsigned int* wmem = (volatile unsigned int*)(DDR_BASE + (w + 1) * CMD_BLOCK_SIZE);
        wmem[1] = (unsigned int)m_slice;
        wmem[2] = (unsigned int)K;
        wmem[3] = (unsigned int)N;
        wmem[4] = (unsigned int)(A + rs * K);
        wmem[5] = (unsigned int)B;
        wmem[6] = (unsigned int)(C + rs * N);
        wmem[STATUS_ADDR] = 0;
        asm volatile("" ::: "memory");  // memory barrier before opcode write
        wmem[0] = OP_MATMUL_PARTIAL;   // GO!
        asm volatile("" ::: "memory");  // ensure write is visible
    }

    // Wait all workers
    for (int w = 0; w < nw; w++) {
        volatile unsigned int* wmem = (volatile unsigned int*)(DDR_BASE + (w + 1) * CMD_BLOCK_SIZE);
        while (1) {
            asm volatile("" ::: "memory");  // barrier before read
            if (wmem[STATUS_ADDR] == 1) break;
        }
    }
}

static int NUM_WORKERS = 15;  // cores 1-15

static void fused_add(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}
static void fused_relu(float* x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}
static void fused_rmsnorm(float* x, float* g, float* out, int batch, int D) {
    for (int b = 0; b < batch; b++) {
        float ss = 0;
        for (int d = 0; d < D; d++) ss += x[b*D+d] * x[b*D+d];
        float rms = ss / D + 1e-5f;
        float inv = 1.0f;
        for (int i = 0; i < 5; i++) inv = inv * (1.5f - 0.5f * rms * inv * inv);
        for (int d = 0; d < D; d++) out[b*D+d] = x[b*D+d] * inv * g[d];
    }
}

// ============================================================
// FUSED FORWARD — coordinator dispatches matmul to 15 workers
// ============================================================
static void coord_fused_forward() {
    int B = (int)my_mem[1], T = (int)my_mem[2], D = (int)my_mem[3];
    int H = (int)my_mem[4], FF = (int)my_mem[5], V = (int)my_mem[6], L = (int)my_mem[7];
    unsigned int* tokens = (unsigned int*)my_mem[8];
    float* wte = (float*)my_mem[9];
    float* wpe = (float*)my_mem[10];
    float* layers_base = (float*)my_mem[11];
    float* lm_head = (float*)my_mem[12];
    float* logits_out = (float*)my_mem[13];
    float* h_out = (float*)my_mem[14];
    float* scratch = (float*)my_mem[15];

    int HD = D / H, BT = B * T, BH = B * H;
    int layer_size = 4*D*D + 2*D*FF + D;

    // Scratch layout
    float* h = scratch;
    float* hn = h + BT*D;
    float* Q = hn + BT*D;
    float* K_buf = Q + BT*D;
    float* V_buf = K_buf + BT*D;
    float* Kt = V_buf + BT*D;
    float* scores = Kt + BH*HD*T;
    float* attn_out = scores + BH*T*T;
    float* proj = attn_out + BT*D;
    float* ff1 = proj + BT*D;
    float* ff2 = ff1 + BT*FF;
    float* Q_tmp = ff2 + BT*D;
    float* V_bh = Q_tmp + BH*T*HD;

    // Cache areas
    float* h_cache = h_out;
    float* hn_cache = h_cache + (L+1)*BT*D;
    float* ff1r_cache = hn_cache + L*BT*D;

    // Embedding
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int tok = tokens[b*T+t];
            for (int d = 0; d < D; d++)
                h[b*T*D+t*D+d] = wte[tok*D+d] + wpe[t*D+d];
        }

    for (int i = 0; i < BT*D; i++) h_cache[i] = h[i];

    for (int li = 0; li < L; li++) {
        float* lw = layers_base + li * layer_size;
        float* Wq=lw, *Wk=lw+D*D, *Wv=lw+2*D*D, *Wo=lw+3*D*D;
        float* W1=lw+4*D*D, *W2=lw+4*D*D+D*FF;
        float* g = lw+4*D*D+2*D*FF;

        fused_rmsnorm(h, g, hn, BT, D);
        for (int i = 0; i < BT*D; i++) hn_cache[li*BT*D+i] = hn[i];

        // QKV — PARALLEL across 15 workers!
        parallel_mm(hn, Wq, Q, BT, D, D, NUM_WORKERS);
        parallel_mm(hn, Wk, K_buf, BT, D, D, NUM_WORKERS);
        parallel_mm(hn, Wv, V_buf, BT, D, D, NUM_WORKERS);

        // Reshape + transpose K
        for (int b = 0; b < B; b++)
            for (int hh = 0; hh < H; hh++)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < HD; d++) {
                        int src = b*T*D + t*D + hh*HD + d;
                        int bh = b*H + hh;
                        Q_tmp[bh*T*HD + t*HD + d] = Q[src];
                        Kt[bh*HD*T + d*T + t] = K_buf[src];
                        V_bh[bh*T*HD + t*HD + d] = V_buf[src];
                    }

        // scores = Q @ Kt per head — use workers for large BH
        for (int bh = 0; bh < BH; bh++)
            nmppmMul_mm_32f(Q_tmp+bh*T*HD, T, HD, Kt+bh*HD*T, HD, T, scores+bh*T*T, T, T, 0);

        // Causal softmax
        float scale = 1.0f;
        { float hdf=(float)HD; float g2=1.0f;
          for (int i=0;i<5;i++) g2=g2*(1.5f-0.5f*hdf*g2*g2); scale=g2; }

        for (int bh=0; bh<BH; bh++)
            for (int i=0; i<T; i++) {
                float* row = scores+bh*T*T+i*T;
                float mx=-1e9f;
                for (int j=0;j<=i;j++) { row[j]*=scale; if(row[j]>mx)mx=row[j]; }
                float sm=0;
                for (int j=0;j<=i;j++) {
                    float v=row[j]-mx;
                    float e=1.0f+v+v*v*0.5f+v*v*v*0.1666667f+v*v*v*v*0.0416667f;
                    if(e<0)e=0; row[j]=e; sm+=e;
                }
                float inv=(sm>0)?1.0f/sm:0;
                for (int j=0;j<=i;j++) row[j]*=inv;
                for (int j=i+1;j<T;j++) row[j]=0;
            }

        // attn @ V per head
        for (int bh=0; bh<BH; bh++)
            nmppmMul_mm_32f(scores+bh*T*T, T, T, V_bh+bh*T*HD, T, HD, attn_out+bh*T*HD, HD, HD, 0);

        // Unreshape
        for (int b=0;b<B;b++)
            for (int hh=0;hh<H;hh++)
                for (int t=0;t<T;t++)
                    for (int d=0;d<HD;d++)
                        proj[b*T*D+t*D+hh*HD+d] = attn_out[(b*H+hh)*T*HD+t*HD+d];

        // proj @ Wo — PARALLEL
        parallel_mm(proj, Wo, hn, BT, D, D, NUM_WORKERS);
        fused_add(h, hn, BT*D);

        // FFN — PARALLEL
        fused_rmsnorm(h, g, hn, BT, D);
        parallel_mm(hn, W1, ff1, BT, D, FF, NUM_WORKERS);
        fused_relu(ff1, BT*FF);
        for (int i=0;i<BT*FF;i++) ff1r_cache[li*BT*FF+i] = ff1[i];
        parallel_mm(ff1, W2, ff2, BT, FF, D, NUM_WORKERS);
        fused_add(h, ff2, BT*D);

        for (int i=0;i<BT*D;i++) h_cache[(li+1)*BT*D+i] = h[i];
    }

    for (int i=0;i<BT*D;i++) h_out[i] = h[i];

    // LM head — PARALLEL
    parallel_mm(h, lm_head, logits_out, BT, D, V, NUM_WORKERS);
}

// ============================================================
// FUSED BACKWARD — same pattern, parallel matmul
// ============================================================
static void fused_sgd(float* W, float* dW, int n, float lr) {
    for (int i=0;i<n;i++) {
        float g=dW[i]; if(g>1)g=1; if(g<-1)g=-1; W[i]-=lr*g;
    }
}

static void coord_fused_backward() {
    int B=(int)my_mem[1],T=(int)my_mem[2],D=(int)my_mem[3];
    int H=(int)my_mem[4],FF=(int)my_mem[5],V=(int)my_mem[6],L=(int)my_mem[7];
    float* dlogits=(float*)my_mem[8];
    unsigned int* tokens=(unsigned int*)my_mem[9];
    float* wte=(float*)my_mem[10];
    float* layers_base=(float*)my_mem[11];
    float* lm_head=(float*)my_mem[12];
    float* h_cache=(float*)my_mem[13];
    float* hn_cache=(float*)my_mem[14];
    float* ff1r_cache=(float*)my_mem[15];
    float lr; {unsigned int b=my_mem[16]; float*f=(float*)&b; lr=*f;}
    float* scratch=(float*)my_mem[17];

    int HD=D/H, BT=B*T, BH=B*H;
    int lsz=4*D*D+2*D*FF+D;

    float* dW=scratch, *dx=dW+D*V, *temp1=dx+BT*D, *temp2=temp1+BT*FF;

    // dW_lm = h.T @ dlogits
    float* hf = h_cache + L*BT*D;
    for (int i=0;i<BT;i++) for (int d=0;d<D;d++) temp1[d*BT+i]=hf[i*D+d];
    parallel_mm(temp1, dlogits, dW, D, BT, V, NUM_WORKERS);
    fused_sgd(lm_head, dW, D*V, lr);

    // dx = dlogits @ lm_head.T
    for (int i=0;i<D;i++) for (int j=0;j<V;j++) temp1[j*D+i]=lm_head[i*V+j];
    parallel_mm(dlogits, temp1, dx, BT, V, D, NUM_WORKERS);

    for (int li=L-1; li>=0; li--) {
        float* lw=layers_base+li*lsz;
        float*Wq=lw,*Wk=lw+D*D,*Wv=lw+2*D*D,*Wo=lw+3*D*D;
        float*W1=lw+4*D*D,*W2=lw+4*D*D+D*FF;
        float* hn=hn_cache+li*BT*D;
        float* ff1r=ff1r_cache+li*BT*FF;

        // FFN backward
        for (int i=0;i<BT;i++) for (int j=0;j<FF;j++) temp2[j*BT+i]=ff1r[i*FF+j];
        parallel_mm(temp2, dx, dW, FF, BT, D, NUM_WORKERS);
        fused_sgd(W2, dW, FF*D, lr);

        for (int i=0;i<FF;i++) for (int j=0;j<D;j++) temp1[j*FF+i]=W2[i*D+j];
        float* dff = temp2 + FF*BT; // reuse
        parallel_mm(dx, temp1, dff, BT, D, FF, NUM_WORKERS);
        for (int i=0;i<BT*FF;i++) if(ff1r[i]<=0) dff[i]=0;

        for (int i=0;i<BT;i++) for (int d=0;d<D;d++) temp1[d*BT+i]=hn[i*D+d];
        parallel_mm(temp1, dff, dW, D, BT, FF, NUM_WORKERS);
        fused_sgd(W1, dW, D*FF, lr);

        for (int i=0;i<D;i++) for (int j=0;j<FF;j++) temp1[j*D+i]=W1[i*FF+j];
        float* dx_add = dff; // reuse
        parallel_mm(dff, temp1, dx_add, BT, FF, D, NUM_WORKERS);
        fused_add(dx, dx_add, BT*D);

        // Attention backward (Wo + QKV simplified)
        for (int i=0;i<BT;i++) for (int d=0;d<D;d++) temp1[d*BT+i]=hn[i*D+d];
        parallel_mm(temp1, dx, dW, D, BT, D, NUM_WORKERS);
        fused_sgd(Wo, dW, D*D, lr);

        for (int i=0;i<D;i++) for (int j=0;j<D;j++) temp1[j*D+i]=Wo[i*D+j];
        float* dx_wo = dx_add;
        parallel_mm(dx, temp1, dx_wo, BT, D, D, NUM_WORKERS);
        fused_add(dx, dx_wo, BT*D);

        // QKV update
        for (int i=0;i<BT;i++) for (int d=0;d<D;d++) temp1[d*BT+i]=hn[i*D+d];
        parallel_mm(temp1, dx_wo, dW, D, BT, D, NUM_WORKERS);
        fused_sgd(Wq, dW, D*D, lr);
        parallel_mm(temp1, dx, dW, D, BT, D, NUM_WORKERS);
        fused_sgd(Wk, dW, D*D, lr);
        fused_sgd(Wv, dW, D*D, lr);
    }

    for (int bt=0;bt<BT;bt++) {
        int tok=tokens[bt];
        for (int d=0;d<D;d++) wte[tok*D+d] -= lr*dx[bt*D+d];
    }
}

// ============================================================
// MAIN
// ============================================================
int main() {
    int core_id = ncl_getCoreID();
    int cluster_id = ncl_getClusterID();
    my_core_index = (cluster_id << 2) + core_id;
    my_mem = (volatile unsigned int*)(DDR_BASE + my_core_index * CMD_BLOCK_SIZE);

    my_mem[STATUS_ADDR] = 1;
    my_mem[WATCHDOG_ADDR] = 0;
    my_mem[0] = OP_NOP;

    if (my_core_index != 0) {
        // WORKER: just poll and execute matmul
        worker_loop();
        return 0;
    }

    // COORDINATOR (core 0): handle fused forward/backward
    unsigned int watchdog = 0;
    while (1) {
        watchdog++;
        my_mem[WATCHDOG_ADDR] = watchdog;

        unsigned int op = my_mem[0];
        if (op == OP_NOP) continue;
        if (op == OP_EXIT) {
            // Also exit all workers
            for (int w = 1; w < 16; w++) {
                volatile unsigned int* wmem = (volatile unsigned int*)(DDR_BASE + w * CMD_BLOCK_SIZE);
                wmem[0] = OP_EXIT;
            }
            my_mem[STATUS_ADDR] = 1;
            my_mem[0] = OP_NOP;
            break;
        }

        my_mem[STATUS_ADDR] = 0;

        switch (op) {
            case OP_MATMUL: {
                unsigned int M=my_mem[1],K=my_mem[2],N=my_mem[3];
                float*A=(float*)my_mem[4],*B=(float*)my_mem[5],*C=(float*)my_mem[6];
                parallel_mm(A, B, C, M, K, N, NUM_WORKERS);
                break;
            }
            case OP_FUSED_FORWARD: coord_fused_forward(); break;
            case OP_FUSED_BACKWARD: coord_fused_backward(); break;
            default: break;
        }

        my_mem[STATUS_ADDR] = 1;
        my_mem[0] = OP_NOP;
    }
    return 0;
}
