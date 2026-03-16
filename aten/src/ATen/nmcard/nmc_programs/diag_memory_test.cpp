// Memory diagnostic kernel for NM Card
// Tests DDR read/write patterns to detect memory issues
// Run in QEMU first, then on card with timeout

#include "mymath.h"

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory test patterns
#define PATTERN_ZEROS     0x00000000
#define PATTERN_ONES      0xFFFFFFFF
#define PATTERN_CHECKERBOARD_A 0xAAAAAAAA
#define PATTERN_CHECKERBOARD_B 0x55555555
#define PATTERN_WALKING_1 0x00000001

// Status codes
#define STATUS_RUNNING    0
#define STATUS_OK         1
#define STATUS_FAIL       2

// Memory layout:
// mem[0] = command (1 = run test)
// mem[1] = test_type (0=all, 1=write-read, 2=pattern, 3=stress)
// mem[2] = start_offset
// mem[3] = size (in words)
// mem[4] = status
// mem[5] = error_offset (if failed)
// mem[6] = expected_value (if failed)
// mem[7] = actual_value (if failed)
// mem[8..] = test area

int test_write_read(unsigned int start, unsigned int size) {
    // Simple write-read test
    for (unsigned int i = 0; i < size; i++) {
        unsigned int offset = start + i;
        unsigned int value = offset ^ 0xDEADBEEF;

        mem[offset] = value;

        // Read back immediately
        unsigned int read_val = mem[offset];
        if (read_val != value) {
            mem[5] = offset;
            mem[6] = value;
            mem[7] = read_val;
            return STATUS_FAIL;
        }
    }
    return STATUS_OK;
}

int test_pattern(unsigned int start, unsigned int size, unsigned int pattern) {
    // Write pattern to all locations
    for (unsigned int i = 0; i < size; i++) {
        mem[start + i] = pattern;
    }

    // Verify pattern
    for (unsigned int i = 0; i < size; i++) {
        unsigned int read_val = mem[start + i];
        if (read_val != pattern) {
            mem[5] = start + i;
            mem[6] = pattern;
            mem[7] = read_val;
            return STATUS_FAIL;
        }
    }
    return STATUS_OK;
}

int test_walking_ones(unsigned int start, unsigned int size) {
    // Walking ones test - good for detecting stuck bits
    unsigned int pattern = 1;
    for (int bit = 0; bit < 32; bit++) {
        for (unsigned int i = 0; i < size; i++) {
            mem[start + i] = pattern;
        }
        for (unsigned int i = 0; i < size; i++) {
            if (mem[start + i] != pattern) {
                mem[5] = start + i;
                mem[6] = pattern;
                mem[7] = mem[start + i];
                return STATUS_FAIL;
            }
        }
        pattern = pattern << 1;
        if (pattern == 0) pattern = 1;
    }
    return STATUS_OK;
}

int test_address_uniqueness(unsigned int start, unsigned int size) {
    // Each location stores its own address - detects address line issues
    for (unsigned int i = 0; i < size; i++) {
        mem[start + i] = start + i;
    }
    for (unsigned int i = 0; i < size; i++) {
        if (mem[start + i] != start + i) {
            mem[5] = start + i;
            mem[6] = start + i;
            mem[7] = mem[start + i];
            return STATUS_FAIL;
        }
    }
    return STATUS_OK;
}

int main() {
    // Signal we're running
    mem[4] = STATUS_RUNNING;

    // Wait for command
    while (mem[0] == 0) {
        // Idle loop - in real hardware, host writes mem[0]=1 to start
        // For QEMU test, we'll run immediately
        break;  // Remove this break for real card test
    }

    unsigned int test_type = mem[1];
    unsigned int start = mem[2];
    unsigned int size = mem[3];

    // Sanity check
    if (size == 0) size = 64;  // Default 64 words
    if (start < 16) start = 16;  // Don't overwrite control area
    if (size > 1024) size = 1024;  // Limit for safety

    int result = STATUS_OK;

    if (test_type == 0 || test_type == 1) {
        result = test_write_read(start, size);
        if (result != STATUS_OK) {
            mem[4] = result;
            return 1;
        }
    }

    if (test_type == 0 || test_type == 2) {
        // Test multiple patterns
        result = test_pattern(start, size, PATTERN_ZEROS);
        if (result != STATUS_OK) { mem[4] = result; return 2; }

        result = test_pattern(start, size, PATTERN_ONES);
        if (result != STATUS_OK) { mem[4] = result; return 3; }

        result = test_pattern(start, size, PATTERN_CHECKERBOARD_A);
        if (result != STATUS_OK) { mem[4] = result; return 4; }

        result = test_pattern(start, size, PATTERN_CHECKERBOARD_B);
        if (result != STATUS_OK) { mem[4] = result; return 5; }
    }

    if (test_type == 0 || test_type == 3) {
        result = test_walking_ones(start, size);
        if (result != STATUS_OK) { mem[4] = result; return 6; }

        result = test_address_uniqueness(start, size);
        if (result != STATUS_OK) { mem[4] = result; return 7; }
    }

    mem[4] = STATUS_OK;
    return 0;
}
