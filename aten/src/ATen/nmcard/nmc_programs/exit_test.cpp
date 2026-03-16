// Minimal test - just exits immediately
// Proves card can run and complete a program

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

int main() {
    // Write markers to show we ran
    mem[0] = 0xCAFE;  // marker 1
    mem[1] = 0xBEEF;  // marker 2
    mem[2] = 42;      // answer
    mem[3] = 1;       // status = done

    return 0;  // EXIT IMMEDIATELY
}
