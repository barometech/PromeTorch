// Probe qpmaddubsh argument semantics on E2K 8C2
//
// Goal: determine whether __builtin_e2k_qpmaddubsh(X, Y) treats X as
// unsigned bytes and Y as signed bytes, or the reverse. Expected result
// for a known-good input lets us decide.

#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef long long v2di __attribute__((vector_size(16)));

int main(void) {
    // Set up two 16-byte vectors where we know the answer:
    //   A bytes: [2, 3, 5, 7, 0, 0, ..., 0]
    //   B bytes: [10, 4, 6, 8, 0, 0, ..., 0]
    // Lane 0 expected = 2*10 + 3*4  = 32 (pair)
    // Lane 1 expected = 5*6  + 7*8  = 86
    uint8_t A[16] = {2, 3, 5, 7, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    int8_t  B[16] = {10, 4, 6, 8, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    v2di va, vb;
    memcpy(&va, A, 16);
    memcpy(&vb, B, 16);

    v2di r_ab = __builtin_e2k_qpmaddubsh(va, vb);
    v2di r_ba = __builtin_e2k_qpmaddubsh(vb, va);
    int16_t* p_ab = (int16_t*)&r_ab;
    int16_t* p_ba = (int16_t*)&r_ba;
    printf("qpmaddubsh(A, B) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           p_ab[0], p_ab[1], p_ab[2], p_ab[3], p_ab[4], p_ab[5], p_ab[6], p_ab[7]);
    printf("qpmaddubsh(B, A) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           p_ba[0], p_ba[1], p_ba[2], p_ba[3], p_ba[4], p_ba[5], p_ba[6], p_ba[7]);
    printf("Expected lanes 0,1 = 32, 86\n");

    // Test with large signed values: A=[200, 200], B=[-50, -50]
    //   unsigned*signed = 200*-50 + 200*-50 = -20000 (fits int16)
    //   signed*unsigned: if B is treated as unsigned (206), then 206*200 + 206*200 = 82400 (saturates)
    uint8_t A2[16] = {200, 200, 0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    int8_t  B2[16] = {-50, -50, 0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    v2di va2, vb2;
    memcpy(&va2, A2, 16);
    memcpy(&vb2, B2, 16);
    v2di r2_ab = __builtin_e2k_qpmaddubsh(va2, vb2);
    v2di r2_ba = __builtin_e2k_qpmaddubsh(vb2, va2);
    int16_t* p2_ab = (int16_t*)&r2_ab;
    int16_t* p2_ba = (int16_t*)&r2_ba;
    printf("big: qpmaddubsh(A=200u, B=-50s) lane0 = %d  (expect -20000 if A=u,B=s)\n", p2_ab[0]);
    printf("big: qpmaddubsh(B=-50s, A=200u) lane0 = %d  (expect -20000 if B=s reinterpreted wrong)\n", p2_ba[0]);
    return 0;
}
