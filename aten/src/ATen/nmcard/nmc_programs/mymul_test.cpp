// Test my own multiply function - no library calls

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// [0] = cmd (1 = test)
// [1] = a
// [2] = b
// [3] = a*b result
// [4] = a+b result
// [5] = status
// [6] = debug counter

// Своя функция умножения - без библиотеки!
unsigned int my_mul(unsigned int a, unsigned int b) {
    unsigned int result = 0;
    while (b != 0) {
        if (b & 1) {
            result = result + a;
        }
        a = a << 1;
        b = b >> 1;
    }
    return result;
}

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

            mem[4] = a + b;           // addition
            mem[3] = my_mul(a, b);    // my multiply

            mem[5] = 1;  // done
            mem[0] = 0;
        }
    }
    return 0;
}
