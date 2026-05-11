#include <stdio.h>
#include "nm6408load_nmc.h"
int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    printf("NMC%d:%d noop\n", ncl_getClusterID(), ncl_getCoreID());
    return 0;
}
