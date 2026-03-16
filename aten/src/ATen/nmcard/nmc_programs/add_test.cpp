// Add only test - no multiply

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

int main() {
    mem[5] = 0;
    mem[6] = 0;

    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[0] == 255) { mem[5] = 1; break; }

        mem[6]++;

        if (mem[0] == 1) {
            mem[5] = 0;

            unsigned int a = mem[1];
            unsigned int b = mem[2];

            // Only addition - no multiply
            mem[3] = a + b;
            mem[4] = a - b;

            mem[5] = 1;
            mem[0] = 0;
        }
    }
    return 0;
}
