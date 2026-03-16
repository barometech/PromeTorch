// Minimal multiply test

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// [0] = cmd (1 = test)
// [1] = a (input)
// [2] = b (input)
// [3] = result a*b
// [4] = result a+b
// [5] = status
// [6] = debug counter

int main() {
    mem[5] = 0;
    mem[6] = 0;

    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[0] == 255) { mem[5] = 1; break; }

        mem[6]++;

        if (mem[0] == 1) {
            mem[5] = 0;  // busy

            unsigned int a = mem[1];
            unsigned int b = mem[2];

            // Simple add - should work
            mem[4] = a + b;

            // Multiply - might hang
            mem[3] = a * b;

            mem[5] = 1;  // done
            mem[0] = 0;
        }
    }
    return 0;
}
