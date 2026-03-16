"""
Operation codes for NMC4 dispatcher
Must match dispatcher.cpp
"""

# Operation codes
OP_NOP = 0
OP_MATMUL = 1
OP_RMSNORM = 2
OP_SOFTMAX = 3
OP_SILU = 4
OP_ROPE = 5
OP_ATTENTION = 6
OP_ELEM_ADD = 10
OP_ELEM_MUL = 11
OP_ELEM_SUB = 12
OP_GATE_MUL = 13
OP_MUL_SCALAR = 14
OP_GELU = 15
OP_LAYERNORM = 16

# Pre-loaded weight operations
OP_MATMUL_DDR = 20
OP_RMSNORM_DDR = 21

# Multi-core operations
OP_MATMUL_PARTIAL = 22  # matmul computing only a column range

# Exit command for graceful shutdown
OP_EXIT = 255

# Memory layout
DDR_BASE = 0x00340000
CMD_BLOCK_SIZE = 32     # words per core's command block

# Single-core layout (1 core, cmd block = 32 words = DDR_BASE+0..31)
DATA_START = DDR_BASE + 64       # After command block (for activations)
WEIGHT_START = DDR_BASE + 65536  # 256KB offset, pre-loaded weights region
TMP_BUF = DDR_BASE + 32768      # Temp buffer for A row conversion (128KB)

# Multi-core layout (16 cores, cmd blocks = DDR_BASE+0..511)
MC_DATA_START = DDR_BASE + 512   # After 16 command blocks (16*32=512)
MC_MAX_CORES = 16

STATUS_ADDR = 30
WATCHDOG_ADDR = 31
