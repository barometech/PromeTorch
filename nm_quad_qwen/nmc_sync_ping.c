/* Minimal sync test: kernel pings host 5 times via ncl_hostSync */
#include <stdio.h>
#include "nm6408load_nmc.h"

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int i;
    for (i = 0; i < 5; ++i) {
        int got = ncl_hostSync(i);
        if (got < 0) break;
    }
    return 0;
}
