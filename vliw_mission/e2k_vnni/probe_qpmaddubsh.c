// E2K 8C2 (ISA v5) INT8 VNNI probe via qpmaddubsh
// Build: lcc -O3 -march=elbrus-v5 probe_qpmaddubsh.c -o probe_qpmaddubsh
//
// Result on lemur-1 (Elbrus 8CВ 1.5GHz, 1 core):
//   FP32 scalar dot:   0.74 GOPS
//   INT8 scalar dot:   8.94 GOPS  (12x)
//   INT8 qpmaddubsh:  36.33 GOPS  (49x)
//
// qpmaddubsh is the only INT8-oriented packed op available on v5.
// qpidotsbwss / qpbfdots / qpfmad require v7+ (12C/16C).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef long long v2di __attribute__((vector_size(16)));

static float dot_fp32(const float* a, const float* b, int K){
    float s=0.f;
    for(int i=0;i<K;i++) s += a[i]*b[i];
    return s;
}

static int dot_i8_scalar(const signed char* a, const signed char* b, int K){
    int s=0;
    for(int i=0;i<K;i++) s += (int)a[i]*(int)b[i];
    return s;
}

// qpmaddubsh: 16×(uint8 × int8) → 8×int16 with pairwise-add
//   r[j] = sat16(a[2j]*b[2j] + a[2j+1]*b[2j+1])
// Horizontal reduction to int32 via qpmaddh with {1,1,...} constant.
static int dot_i8_vnni(const signed char* a, const signed char* b, int K){
    v2di acc = {0,0};
    const v2di* pa = (const v2di*)a;
    const v2di* pb = (const v2di*)b;
    int nv = K/16;
    for(int v=0; v<nv; v++){
        v2di r = __builtin_e2k_qpmaddubsh(pa[v], pb[v]);
        static const v2di ones16 = {0x0001000100010001LL,0x0001000100010001LL};
        v2di r32 = __builtin_e2k_qpmaddh(r, ones16);
        acc = __builtin_e2k_qpaddw(acc, r32);
    }
    int s = ((int*)&acc)[0] + ((int*)&acc)[1] + ((int*)&acc)[2] + ((int*)&acc)[3];
    return s;
}

int main(){
    const int K = 2560;
    const long N = 2000000;
    float* a_f = (float*)aligned_alloc(64, K*sizeof(float));
    float* b_f = (float*)aligned_alloc(64, K*sizeof(float));
    signed char* a_s = (signed char*)aligned_alloc(64, K);
    signed char* b_s = (signed char*)aligned_alloc(64, K);
    for(int i=0;i<K;i++){a_f[i]=(i%7)*0.1f;b_f[i]=(i%11)*0.1f;a_s[i]=(signed char)(i%13);b_s[i]=(signed char)(i%17);}
    struct timespec t0,t1;
    volatile float sf=0.f; volatile int si=0;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(long n=0;n<N;n++) sf += dot_fp32(a_f,b_f,K);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double fp_s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double fp_gops=2.0*K*N/fp_s/1e9;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(long n=0;n<N;n++) si += dot_i8_scalar(a_s,b_s,K);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double is_s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double is_gops=2.0*K*N/is_s/1e9;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(long n=0;n<N;n++) si += dot_i8_vnni(a_s,b_s,K);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double iv_s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double iv_gops=2.0*K*N/iv_s/1e9;
    printf("K=%d N=%ld\n",K,N);
    printf("FP32 scalar:      %.3f s, %6.2f GOPS\n",fp_s,fp_gops);
    printf("INT8 scalar:      %.3f s, %6.2f GOPS\n",is_s,is_gops);
    printf("INT8 qpmaddubsh:  %.3f s, %6.2f GOPS  (%.1fx vs FP32)\n",iv_s,iv_gops,iv_gops/fp_gops);
    (void)sf; (void)si;
    return 0;
}
