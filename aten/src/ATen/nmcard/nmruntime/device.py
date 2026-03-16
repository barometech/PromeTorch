"""
Device class for NM Card Mini
Handles board initialization, memory management, and operation dispatch
"""

import ctypes
import os
import struct
import time
import numpy as np
from typing import Optional, List, Tuple

from .ops import (
    DDR_BASE, DATA_START, WEIGHT_START, TMP_BUF, STATUS_ADDR, WATCHDOG_ADDR,
    CMD_BLOCK_SIZE, MC_DATA_START, MC_MAX_CORES,
    OP_NOP, OP_MATMUL, OP_RMSNORM, OP_SOFTMAX, OP_SILU,
    OP_ROPE, OP_ELEM_ADD, OP_ELEM_MUL, OP_GATE_MUL,
    OP_MATMUL_DDR, OP_RMSNORM_DDR, OP_MATMUL_PARTIAL, OP_EXIT
)

# Add DLL paths
for path in [r"C:\Program Files\Module\NMDL\bin",
             r"C:\Program Files\Module\NM_Card\libload\bin"]:
    if os.path.exists(path):
        os.environ["PATH"] = path + ";" + os.environ.get("PATH", "")
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(path)

# Load DLL
_nm = ctypes.CDLL(r"C:\Program Files\Module\NM_Card\libload\bin\nm_card_load.dll")
_PL_Word = ctypes.c_uint32

class _PL_CoreNo(ctypes.Structure):
    _fields_ = [("nm_id", ctypes.c_int), ("cluster_id", ctypes.c_int)]

# Setup function signatures
_nm.PL_GetBoardCount.argtypes = [ctypes.POINTER(ctypes.c_uint)]
_nm.PL_GetBoardCount.restype = ctypes.c_int
_nm.PL_GetBoardDesc.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
_nm.PL_GetBoardDesc.restype = ctypes.c_int
_nm.PL_ResetBoard.argtypes = [ctypes.c_void_p]
_nm.PL_ResetBoard.restype = ctypes.c_int
_nm.PL_LoadInitCode.argtypes = [ctypes.c_void_p]
_nm.PL_LoadInitCode.restype = ctypes.c_int
_nm.PL_GetAccess.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PL_CoreNo), ctypes.POINTER(ctypes.c_void_p)]
_nm.PL_GetAccess.restype = ctypes.c_int
_nm.PL_LoadProgramFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_nm.PL_LoadProgramFile.restype = ctypes.c_int
_nm.PL_WriteMemBlock.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PL_Word), ctypes.c_uint32, ctypes.c_uint32]
_nm.PL_WriteMemBlock.restype = ctypes.c_int
_nm.PL_ReadMemBlock.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PL_Word), ctypes.c_uint32, ctypes.c_uint32]
_nm.PL_ReadMemBlock.restype = ctypes.c_int
_nm.PL_CloseAccess.argtypes = [ctypes.c_void_p]
_nm.PL_CloseBoardDesc.argtypes = [ctypes.c_void_p]


def float_to_word(f: float) -> int:
    return struct.unpack('I', struct.pack('f', f))[0]

def word_to_float(w: int) -> float:
    return struct.unpack('f', struct.pack('I', w))[0]


class Device:
    """
    High-level interface to NM Card Mini

    Usage:
        device = Device(0)  # Open board 0
        device.load_dispatcher()  # Load unified dispatcher
        result = device.matmul(A, B)  # Run operations
        device.close()
    """

    def __init__(self, board_index: int = 0):
        self.board_index = board_index
        self.board = None
        self.access = None
        self.dispatcher_loaded = False
        self._next_addr = DATA_START

        self._open()

    def _open(self):
        """Initialize board connection"""
        count = ctypes.c_uint()
        if _nm.PL_GetBoardCount(ctypes.byref(count)) != 0 or count.value == 0:
            raise RuntimeError("No NM Card found")

        self.board = ctypes.c_void_p()
        if _nm.PL_GetBoardDesc(self.board_index, ctypes.byref(self.board)) != 0:
            raise RuntimeError(f"Cannot open board {self.board_index}. Is it in use?")

        if _nm.PL_ResetBoard(self.board) != 0:
            print("Warning: ResetBoard failed")

        if _nm.PL_LoadInitCode(self.board) != 0:
            print("Warning: LoadInitCode failed")

        core = _PL_CoreNo(nm_id=0, cluster_id=0)
        self.access = ctypes.c_void_p()
        if _nm.PL_GetAccess(self.board, ctypes.byref(core), ctypes.byref(self.access)) != 0:
            raise RuntimeError("Cannot get access to NMC core")

    def close(self):
        """Close board connection"""
        if self.access:
            _nm.PL_CloseAccess(self.access)
            self.access = None
        if self.board:
            _nm.PL_CloseBoardDesc(self.board)
            self.board = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def load_dispatcher(self, path: Optional[str] = None):
        """Load the unified dispatcher program"""
        if path is None:
            # Find dispatcher.abs relative to this file
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(module_dir, "nmc_programs", "dispatcher.abs")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dispatcher not found: {path}")

        if _nm.PL_LoadProgramFile(self.access, path.encode()) != 0:
            raise RuntimeError(f"Failed to load: {path}")

        time.sleep(0.3)  # Wait for program to initialize
        self.dispatcher_loaded = True
        self._next_addr = DATA_START
        self._weight_addr = WEIGHT_START
        self._preloaded = {}  # name -> (addr, shape)

    def write_array(self, data: np.ndarray) -> int:
        """
        Write numpy array to device memory.
        Returns the address where data was written.
        Uses numpy ctypes interface for fast transfer.
        """
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        flat = np.ascontiguousarray(data.flatten())
        count = flat.size
        addr = self._next_addr
        ptr = flat.ctypes.data_as(ctypes.POINTER(_PL_Word))
        if _nm.PL_WriteMemBlock(self.access, ptr, addr, count) != 0:
            raise RuntimeError(f"Write failed at 0x{addr:08X}")
        self._next_addr = addr + count
        return addr

    def read_array(self, addr: int, shape: Tuple[int, ...]) -> np.ndarray:
        """Read array from device memory"""
        count = int(np.prod(shape))
        result = np.empty(count, dtype=np.float32)
        ptr = result.ctypes.data_as(ctypes.POINTER(_PL_Word))
        if _nm.PL_ReadMemBlock(self.access, ptr, addr, count) != 0:
            raise RuntimeError(f"Read failed at 0x{addr:08X}")
        return result.reshape(shape)

    def _write_words(self, addr: int, words: List[int]):
        """Write raw words to memory"""
        arr = (_PL_Word * len(words))(*words)
        if _nm.PL_WriteMemBlock(self.access, arr, addr, len(words)) != 0:
            raise RuntimeError(f"Write failed at 0x{addr:08X}")

    def _read_words(self, addr: int, count: int) -> List[int]:
        """Read raw words from memory"""
        arr = (_PL_Word * count)()
        if _nm.PL_ReadMemBlock(self.access, arr, addr, count) != 0:
            raise RuntimeError(f"Read failed at 0x{addr:08X}")
        return list(arr)

    def _execute(self, op: int, args: List[int], timeout: float = 120.0) -> bool:
        """
        Execute operation and wait for completion.
        args should NOT include op code (it's added automatically).

        Protocol fix: write args first (positions 1-31), then write opcode (position 0).
        Wait for card to set mem[0]=NOP before sending next command.
        """
        # Step 1: Wait for card to be ready (mem[0] == NOP)
        start = time.time()
        while time.time() - start < 2.0:
            if self._read_words(DDR_BASE, 1)[0] == OP_NOP:
                break
            time.sleep(0.001)

        # Step 2: Write args (positions 1-31), including STATUS=0
        cmd_args = list(args)
        while len(cmd_args) < 31:
            cmd_args.append(0)
        cmd_args[STATUS_ADDR - 1] = 0  # STATUS = busy
        self._write_words(DDR_BASE + 1, cmd_args)

        # Step 3: Write opcode (position 0) - triggers card
        self._write_words(DDR_BASE, [op])

        # Step 4: Wait for completion
        start = time.time()
        while time.time() - start < timeout:
            status = self._read_words(DDR_BASE + STATUS_ADDR, 1)[0]
            if status == 1:
                # Wait for card to reset mem[0] to NOP before returning
                for _ in range(100):
                    if self._read_words(DDR_BASE, 1)[0] == OP_NOP:
                        break
                    time.sleep(0.001)
                return True
            elif status == 2:
                raise RuntimeError("Operation failed (error status)")
            time.sleep(0.01)

        raise TimeoutError(f"Operation {op} timed out after {timeout}s")

    # ============================================================
    # High-level operations
    # ============================================================

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication: C = A @ B
        A: shape (M, K)
        B: shape (K, N)
        Returns: C shape (M, N)
        """
        assert A.ndim == 2 and B.ndim == 2
        assert A.shape[1] == B.shape[0]

        M, K = A.shape
        N = B.shape[1]

        addr_A = self.write_array(A)
        addr_B = self.write_array(B)
        addr_C = self._next_addr
        self._next_addr += M * N

        self._execute(OP_MATMUL, [M, K, N, addr_A, addr_B, addr_C])
        return self.read_array(addr_C, (M, N))

    def rmsnorm(self, x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """
        RMS Normalization: y = x * gamma / sqrt(mean(x^2) + eps)
        x: shape (batch, hidden) or (hidden,)
        gamma: shape (hidden,)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch, hidden = x.shape

        addr_in = self.write_array(x)
        addr_out = self._next_addr
        self._next_addr += batch * hidden
        addr_gamma = self.write_array(gamma)

        self._execute(OP_RMSNORM, [batch, hidden, addr_in, addr_out, addr_gamma])
        return self.read_array(addr_out, (batch, hidden)).squeeze()

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax over last dimension
        x: shape (batch, dim) or (dim,)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch, dim = x.shape

        addr_in = self.write_array(x)
        addr_out = self._next_addr
        self._next_addr += batch * dim

        self._execute(OP_SOFTMAX, [batch, dim, addr_in, addr_out])
        return self.read_array(addr_out, (batch, dim)).squeeze()

    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation: y = x * sigmoid(x)"""
        flat = x.flatten()
        count = len(flat)

        addr_in = self.write_array(flat)
        addr_out = self._next_addr
        self._next_addr += count

        self._execute(OP_SILU, [count, addr_in, addr_out])
        return self.read_array(addr_out, x.shape)

    def rope(self, x: np.ndarray, freqs: np.ndarray, pos_offset: int = 0) -> np.ndarray:
        """
        Rotary Position Embedding
        x: shape (seq_len, head_dim)
        freqs: shape (head_dim // 2,) - precomputed inverse frequencies
        """
        seq_len, head_dim = x.shape

        addr_in = self.write_array(x)
        addr_out = self._next_addr
        self._next_addr += seq_len * head_dim
        addr_freqs = self.write_array(freqs)

        self._execute(OP_ROPE, [seq_len, head_dim, pos_offset, addr_in, addr_out, addr_freqs])
        return self.read_array(addr_out, (seq_len, head_dim))

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Elementwise add"""
        assert a.shape == b.shape
        flat_a = a.flatten()
        flat_b = b.flatten()
        count = len(flat_a)

        addr_a = self.write_array(flat_a)
        addr_b = self.write_array(flat_b)
        addr_out = self._next_addr
        self._next_addr += count

        self._execute(OP_ELEM_ADD, [count, addr_a, addr_b, addr_out])
        return self.read_array(addr_out, a.shape)

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Elementwise multiply"""
        assert a.shape == b.shape
        flat_a = a.flatten()
        flat_b = b.flatten()
        count = len(flat_a)

        addr_a = self.write_array(flat_a)
        addr_b = self.write_array(flat_b)
        addr_out = self._next_addr
        self._next_addr += count

        self._execute(OP_ELEM_MUL, [count, addr_a, addr_b, addr_out])
        return self.read_array(addr_out, a.shape)

    def gate_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Gate multiplication: out = a * silu(b) - for Llama FFN"""
        assert a.shape == b.shape
        flat_a = a.flatten()
        flat_b = b.flatten()
        count = len(flat_a)

        addr_a = self.write_array(flat_a)
        addr_b = self.write_array(flat_b)
        addr_out = self._next_addr
        self._next_addr += count

        self._execute(OP_GATE_MUL, [count, addr_a, addr_b, addr_out])
        return self.read_array(addr_out, a.shape)

    # ============================================================
    # Pre-loaded weight operations
    # ============================================================

    def preload_weight(self, name: str, data: np.ndarray) -> int:
        """
        Pre-load a weight matrix to DDR as Q16.16 fixed-point.
        Returns the DDR address where weight was stored.

        data: float32 numpy array (will be converted to Q16.16)
        """
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Convert float32 -> Q16.16 int32
        q16_data = np.clip(data.flatten() * 65536.0, -2147483648, 2147483647).astype(np.int32)

        addr = self._weight_addr
        words = q16_data.view(np.uint32).tolist()
        self._write_words(addr, words)
        self._weight_addr += len(words)
        self._preloaded[name] = (addr, data.shape)
        return addr

    def get_preloaded_addr(self, name: str) -> int:
        """Get DDR address of a pre-loaded weight"""
        return self._preloaded[name][0]

    def matmul_ddr(self, A: np.ndarray, weight_addr: int, N: int) -> np.ndarray:
        """
        Matrix multiplication with pre-loaded Q16.16 weight in DDR.
        C = A @ B where B is pre-loaded at weight_addr, shape (K, N).
        A: float32 (M, K)
        Returns: float32 (M, N)
        """
        assert A.ndim == 2
        M, K = A.shape

        self.reset_memory()
        addr_A = self.write_array(A)
        addr_C = self._next_addr
        self._next_addr += M * N

        self._execute(OP_MATMUL_DDR, [
            M, K, N, addr_A, weight_addr, addr_C, TMP_BUF
        ])
        return self.read_array(addr_C, (M, N))

    def rmsnorm_ddr(self, x: np.ndarray, gamma_addr: int) -> np.ndarray:
        """
        RMSNorm with pre-loaded Q16.16 gamma in DDR.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch, hidden = x.shape

        self.reset_memory()
        addr_in = self.write_array(x)
        addr_out = self._next_addr
        self._next_addr += batch * hidden

        self._execute(OP_RMSNORM_DDR, [batch, hidden, addr_in, addr_out, gamma_addr])
        return self.read_array(addr_out, (batch, hidden)).squeeze()

    def preload_stats(self) -> dict:
        """Return stats about pre-loaded weights"""
        total_words = self._weight_addr - WEIGHT_START
        return {
            'count': len(self._preloaded),
            'total_words': total_words,
            'total_mb': total_words * 4 / 1024 / 1024,
            'names': list(self._preloaded.keys()),
        }

    def reset_memory(self):
        """Reset activation memory allocator (does NOT clear pre-loaded weights)"""
        self._next_addr = DATA_START

    def shutdown(self):
        """
        Send EXIT command to dispatcher for graceful shutdown.
        This allows the kernel to exit cleanly instead of being killed.
        Call this before close() to avoid leaving the card in a bad state.
        """
        if not self.dispatcher_loaded:
            return

        try:
            # Send EXIT command
            cmd = [OP_EXIT] + [0] * 31
            self._write_words(DDR_BASE, cmd)

            # Wait for acknowledgment (short timeout)
            for _ in range(10):
                status = self._read_words(DDR_BASE + STATUS_ADDR, 1)[0]
                if status == 1:
                    break
                time.sleep(0.1)

            self.dispatcher_loaded = False
        except Exception:
            pass  # Ignore errors during shutdown


class MultiCoreDevice:
    """
    Multi-core NM Card Mini interface.
    Loads dispatcher_mc.abs on multiple NMC4 cores and distributes matmul work.

    Usage:
        dev = MultiCoreDevice(num_cores=16)
        dev.load_dispatcher()
        C = dev.matmul(A, B)  # automatically parallelized across cores
        dev.shutdown()
        dev.close()
    """

    def __init__(self, board_index: int = 0, num_cores: int = 16):
        self.board_index = board_index
        self.num_cores = min(num_cores, MC_MAX_CORES)
        self.board = None
        self.accesses = []  # PL_Access per core
        self.core_list = []  # (nm_id, cluster_id) per core
        self.dispatcher_loaded = False
        self._next_addr = MC_DATA_START

        # Generate core list: iterate clusters first, then cores within
        for cl in range(4):
            for nm in range(4):
                if len(self.core_list) < self.num_cores:
                    self.core_list.append((nm, cl))

        self._open()

    def _open(self):
        """Initialize board and get access to all cores"""
        count = ctypes.c_uint()
        if _nm.PL_GetBoardCount(ctypes.byref(count)) != 0 or count.value == 0:
            raise RuntimeError("No NM Card found")

        self.board = ctypes.c_void_p()
        if _nm.PL_GetBoardDesc(self.board_index, ctypes.byref(self.board)) != 0:
            raise RuntimeError(f"Cannot open board {self.board_index}")

        if _nm.PL_ResetBoard(self.board) != 0:
            print("Warning: ResetBoard failed")

        if _nm.PL_LoadInitCode(self.board) != 0:
            print("Warning: LoadInitCode failed")

        # Get access to each core
        for nm_id, cluster_id in self.core_list:
            core = _PL_CoreNo(nm_id=nm_id, cluster_id=cluster_id)
            access = ctypes.c_void_p()
            r = _nm.PL_GetAccess(self.board, ctypes.byref(core), ctypes.byref(access))
            if r != 0:
                print(f"Warning: PL_GetAccess failed for nm={nm_id}, cl={cluster_id}: {r}")
                continue
            self.accesses.append(access)

        self.num_cores = len(self.accesses)
        print(f"Opened {self.num_cores} cores")

    def _write_words(self, addr: int, words: List[int]):
        """Write raw words via first access handle (DDR is shared, with retries)"""
        arr = (_PL_Word * len(words))(*words)
        for attempt in range(5):
            if _nm.PL_WriteMemBlock(self.accesses[0], arr, addr, len(words)) == 0:
                return
            time.sleep(0.01 * (attempt + 1))
        raise RuntimeError(f"Write failed at 0x{addr:08X} after 5 retries")

    def _read_words(self, addr: int, count: int) -> List[int]:
        """Read raw words via first access handle (with retries for PCIe contention)"""
        arr = (_PL_Word * count)()
        for attempt in range(5):
            if _nm.PL_ReadMemBlock(self.accesses[0], arr, addr, count) == 0:
                return list(arr)
            time.sleep(0.01 * (attempt + 1))
        raise RuntimeError(f"Read failed at 0x{addr:08X} after 5 retries")

    def write_array(self, data: np.ndarray) -> int:
        """Write numpy array to shared DDR (with retries)"""
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        flat = np.ascontiguousarray(data.flatten())
        count = flat.size
        addr = self._next_addr
        ptr = flat.ctypes.data_as(ctypes.POINTER(_PL_Word))
        for attempt in range(5):
            if _nm.PL_WriteMemBlock(self.accesses[0], ptr, addr, count) == 0:
                self._next_addr = addr + count
                return addr
            time.sleep(0.01 * (attempt + 1))
        raise RuntimeError(f"Write failed at 0x{addr:08X} after 5 retries")

    def read_array(self, addr: int, shape: Tuple[int, ...]) -> np.ndarray:
        """Read array from shared DDR (with retries)"""
        count = int(np.prod(shape))
        result = np.empty(count, dtype=np.float32)
        ptr = result.ctypes.data_as(ctypes.POINTER(_PL_Word))
        for attempt in range(5):
            if _nm.PL_ReadMemBlock(self.accesses[0], ptr, addr, count) == 0:
                return result.reshape(shape)
            time.sleep(0.01 * (attempt + 1))
        raise RuntimeError(f"Read failed at 0x{addr:08X} after 5 retries")

    def load_dispatcher(self, path: Optional[str] = None):
        """Load multi-core dispatcher on all cores"""
        if path is None:
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(module_dir, "nmc_programs", "dispatcher_mc.abs")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dispatcher not found: {path}")

        path_bytes = path.encode()
        for i, access in enumerate(self.accesses):
            r = _nm.PL_LoadProgramFile(access, path_bytes)
            if r != 0:
                nm_id, cl_id = self.core_list[i]
                print(f"Warning: Failed to load on core nm={nm_id}, cl={cl_id}: {r}")

        time.sleep(0.5)  # Wait for all cores to initialize
        self.dispatcher_loaded = True
        self._next_addr = MC_DATA_START

        # Verify cores are ready (each should set STATUS=1)
        ready = 0
        for i in range(self.num_cores):
            nm_id, cl_id = self.core_list[i]
            core_index = cl_id * 4 + nm_id
            cmd_base = DDR_BASE + core_index * CMD_BLOCK_SIZE
            status = self._read_words(cmd_base + STATUS_ADDR, 1)[0]
            if status == 1:
                ready += 1
        print(f"Dispatcher loaded: {ready}/{self.num_cores} cores ready")

    def _core_cmd_base(self, core_idx: int) -> int:
        """Get DDR address of a core's command block"""
        nm_id, cl_id = self.core_list[core_idx]
        core_index = cl_id * 4 + nm_id
        return DDR_BASE + core_index * CMD_BLOCK_SIZE

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Parallel matrix multiplication: C = A @ B
        Splits output columns across all cores.
        """
        assert A.ndim == 2 and B.ndim == 2
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        N = B.shape[1]

        self.reset_memory()

        # Write shared data
        addr_A = self.write_array(A)
        addr_B = self.write_array(B)
        addr_C = self._next_addr
        self._next_addr += M * N

        # Split N across cores
        chunk = N // self.num_cores
        remainder = N % self.num_cores

        # Write commands to all cores simultaneously
        for i in range(self.num_cores):
            col_start = i * chunk + min(i, remainder)
            col_end = (i + 1) * chunk + min(i + 1, remainder)
            if col_start >= col_end:
                # No work for this core, skip
                continue

            cmd_base = self._core_cmd_base(i)

            # Write args first (positions 1-29), then STATUS=0, then opcode
            args = [0] * 31
            args[0] = M           # mem[1]
            args[1] = K           # mem[2]
            args[2] = N           # mem[3]
            args[3] = addr_A      # mem[4]
            args[4] = addr_B      # mem[5]
            args[5] = addr_C      # mem[6]
            args[6] = col_start   # mem[7]
            args[7] = col_end     # mem[8]
            args[STATUS_ADDR - 1] = 0  # STATUS = busy
            self._write_words(cmd_base + 1, args)

            # Write opcode to trigger
            self._write_words(cmd_base, [OP_MATMUL_PARTIAL])

        # Wait for all cores to finish
        start = time.time()
        timeout = 120.0
        done = [False] * self.num_cores

        while time.time() - start < timeout:
            all_done = True
            for i in range(self.num_cores):
                if done[i]:
                    continue
                cmd_base = self._core_cmd_base(i)
                status = self._read_words(cmd_base + STATUS_ADDR, 1)[0]
                if status == 1:
                    done[i] = True
                elif status == 2:
                    raise RuntimeError(f"Core {i} reported error")
                else:
                    all_done = False
            if all_done:
                break
            time.sleep(0.01)

        if not all(done):
            not_done = [i for i, d in enumerate(done) if not d]
            raise TimeoutError(f"Cores {not_done} timed out after {timeout}s")

        return self.read_array(addr_C, (M, N))

    def reset_memory(self):
        """Reset shared data allocator"""
        self._next_addr = MC_DATA_START

    def shutdown(self):
        """Send OP_EXIT to all cores"""
        if not self.dispatcher_loaded:
            return

        for i in range(self.num_cores):
            cmd_base = self._core_cmd_base(i)
            cmd = [OP_EXIT] + [0] * 31
            self._write_words(cmd_base, cmd)

        time.sleep(0.3)
        self.dispatcher_loaded = False
        print("All cores shut down")

    def close(self):
        """Close all access handles and board"""
        if self.dispatcher_loaded:
            try:
                self.shutdown()
            except Exception:
                pass

        for access in self.accesses:
            _nm.PL_CloseAccess(access)
        self.accesses.clear()

        if self.board:
            _nm.PL_CloseBoardDesc(self.board)
            self.board = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def get_board_count() -> int:
    """Get number of available NM Cards"""
    count = ctypes.c_uint()
    _nm.PL_GetBoardCount(ctypes.byref(count))
    return count.value
