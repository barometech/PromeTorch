// Minimal test - no multiplication, no function calls
// Just write to DDR and return

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

extern "C" int __main() {
    // Write signature
    mem[0] = 0xDEADBEEF;
    mem[1] = 0x12345678;
    mem[2] = 1;  // status = done

    // Infinite loop (wait for host to read)
    // OP_EXIT = 255 - безопасный выход!
    while(1) {
        if (mem[0] == 255) { mem[2] = 1; break; }
    }

    return 0;
}
