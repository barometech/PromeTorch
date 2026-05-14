/* Test NMC4 inline asm — minimal */
#include <stdio.h>
#include "nm6408load_nmc.h"

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    asm volatile ("vnul;");
    asm volatile ("fpu 0 .float vreg0 = vreg0 + vreg0;");
    printf("[asm] survived\n");
    return 0;
}
