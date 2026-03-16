// Instant test - just write status=1 and exit
#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

int main() {
    mem[0] = 0xDEAD;   // signature
    mem[1] = 0xBEEF;   // signature
    mem[7] = 1;        // status = done
    return 0;
}
