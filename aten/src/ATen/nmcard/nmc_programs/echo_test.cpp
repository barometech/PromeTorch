// Echo test - просто копирует входные данные на выход
// Диагностика: проверяем что NMC читает и пишет память правильно

#define DDR_BASE 0x00340000

// Memory layout:
// [0] = command (1 = echo, 0 = idle)
// [1] = source address
// [2] = dest address
// [3] = count (words)
// [4] = status (0 = busy, 1 = done)
// [5] = debug counter
// [6] = last seen cmd value

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

#define CMD       0
#define SRC_ADDR  1
#define DST_ADDR  2
#define COUNT     3
#define STATUS    4
#define DBG_CNT   5
#define LAST_CMD  6

int main() {
    // Initialize
    mem[STATUS] = 0;
    mem[DBG_CNT] = 0;
    mem[LAST_CMD] = 0xFFFF;  // marker

    // Main loop
    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[CMD] == 255) { mem[STATUS] = 1; break; }

        // Increment debug counter
        mem[DBG_CNT]++;

        // Save last seen cmd value
        unsigned int cmd = mem[CMD];
        mem[LAST_CMD] = cmd;

        // Check for echo command
        if (cmd == 1) {
            mem[STATUS] = 0;  // busy

            // Get parameters
            volatile unsigned int* src = (volatile unsigned int*)mem[SRC_ADDR];
            volatile unsigned int* dst = (volatile unsigned int*)mem[DST_ADDR];
            unsigned int count = mem[COUNT];

            // Copy data
            for (unsigned int i = 0; i < count; i++) {
                dst[i] = src[i];
            }

            // Done
            mem[STATUS] = 1;
            mem[CMD] = 0;  // ready for next
        }
    }

    return 0;
}
