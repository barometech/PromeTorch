// Simple test - writes a marker to memory
// NMC4 использует СЛОВАРНУЮ адресацию! Каждый адрес = 32-bit слово.

// DDR base в словарных адресах
#define DDR_BASE 0x00340000

// Указатели
volatile unsigned int* marker  = (volatile unsigned int*)(DDR_BASE);      // слово 0
volatile unsigned int* counter = (volatile unsigned int*)(DDR_BASE + 1);  // слово 1

int main() {
    // Write marker to show program started
    *marker = 0xDEADBEEF;

    // Initialize counter
    *counter = 0;

    // Infinite loop incrementing counter
    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (*marker == 255) { *marker = 1; break; }

        (*counter)++;

        // Small delay
        for (volatile int i = 0; i < 100000; i++) {}
    }

    return 0;
}
