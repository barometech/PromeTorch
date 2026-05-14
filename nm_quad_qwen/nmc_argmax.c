/* nmc_argmax.c — argmax over logits на NMC4 */
#include <stdio.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define VOCAB 151936
float logits[VOCAB];
int token_id;

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    { volatile int w; for (w = 0; w < 5000000  /* fast DMA settle */; ++w) ; }

    float best_val = logits[0];
    int best_idx = 0;
    int i;
    for (i = 1; i < VOCAB; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    token_id = best_idx;
    printf("[argmax] token_id=%d max_logit=%f\n", best_idx, best_val);
    return 0;
}
