// Standalone self-test driver for torch::autograd::forward_ad.
// Build: g++ -std=c++17 -O2 -I<repo_root> torch/autograd/forward_ad_selftest.cpp \
//        -L<build>/ -laten_cpu -lc10 -ltorch_autograd -o forward_ad_selftest
#define PT_FORWARD_AD_SELFTEST 1
#include "torch/autograd/forward_ad.h"

int main() {
    return torch::autograd::forward_ad::forward_ad_selftest();
}
