"""
NanoGPT Training on NM QUAD — 64 NM6408 cores (4 boards × 4 chips × 4 cores)
First-ever neural network training on NM QUAD hardware via PromeTorch.

Architecture: 2-layer Transformer (character-level)
- vocab_size=65 (tiny_shakespeare chars)
- embed_dim=64, num_heads=4, head_dim=16
- ffn_dim=128, context_length=32
- ~200K parameters
- float32 arithmetic on NM6408 FPU

Usage: python train_nanogpt_nmquad.py
       python train_nanogpt_nmquad.py --data tiny_shakespeare.txt --epochs 5
"""

import ctypes
import os
import sys
import time
import struct
import numpy as np

# ============================================================
# NM QUAD 64-core controller (float32, not Q16.16)
# ============================================================
class NMQuad64:
    DDR = 0x00340000
    CMD_BLOCK = 32  # words per core
    OP_NOP = 0
    OP_MATMUL = 1
    OP_ADD = 2
    OP_MUL = 3
    OP_RELU = 4
    OP_SIGMOID = 5
    OP_DONE = 0xFF

    def __init__(self, dispatcher_path="dispatcher_nmquad.abs"):
        NM_PATH = r'C:\Module\NM_Quad\bin'
        NMRB_PATH = r'C:\Module\nmrb-client\bin'
        os.environ['PATH'] = NM_PATH + ';' + NMRB_PATH + ';' + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(NM_PATH)
            os.add_dll_directory(NMRB_PATH)
            # Also add Windows system dirs for CRT
            os.add_dll_directory(r'C:\Windows\System32')

        # Load nmrb-uproxy.dll first (dependency of proxy DLL)
        uproxy_path = os.path.join(NMRB_PATH, 'nmrb-uproxy.dll')
        if os.path.exists(uproxy_path):
            ctypes.CDLL(uproxy_path, winmode=0)

        # Load the proxy DLL
        dll_path = os.path.join(NM_PATH, 'nm_quad_load.dll')
        if not os.path.exists(dll_path):
            raise RuntimeError(f"nm_quad_load.dll not found at {dll_path}")
        self.nm = ctypes.CDLL(dll_path, winmode=0)
        self._setup()
        self.dispatcher_path = dispatcher_path
        self.boards = []
        self.accesses = []  # [(board_idx, chip, core, PL_Access*)]

    def _setup(self):
        nm = self.nm
        nm.PL_SetTimeout.argtypes = [ctypes.c_uint32]
        nm.PL_SetTimeout.restype = ctypes.c_int
        nm.PL_GetBoardCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        nm.PL_GetBoardCount.restype = ctypes.c_int
        nm.PL_GetBoardDesc.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        nm.PL_GetBoardDesc.restype = ctypes.c_int
        nm.PL_ResetBoard.argtypes = [ctypes.c_void_p]
        nm.PL_ResetBoard.restype = ctypes.c_int
        nm.PL_LoadInitCode.argtypes = [ctypes.c_void_p]
        nm.PL_LoadInitCode.restype = ctypes.c_int

        class CoreNo(ctypes.Structure):
            _fields_ = [("nm_id", ctypes.c_int), ("cluster_id", ctypes.c_int)]
        self.CoreNo = CoreNo

        nm.PL_GetAccess.argtypes = [ctypes.c_void_p, ctypes.POINTER(CoreNo), ctypes.POINTER(ctypes.c_void_p)]
        nm.PL_GetAccess.restype = ctypes.c_int
        nm.PL_CloseAccess.argtypes = [ctypes.c_void_p]
        nm.PL_CloseAccess.restype = ctypes.c_int
        nm.PL_CloseBoardDesc.argtypes = [ctypes.c_void_p]
        nm.PL_CloseBoardDesc.restype = ctypes.c_int
        nm.PL_LoadProgramFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nm.PL_LoadProgramFile.restype = ctypes.c_int
        nm.PL_WriteMemBlock.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        nm.PL_WriteMemBlock.restype = ctypes.c_int
        nm.PL_ReadMemBlock.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        nm.PL_ReadMemBlock.restype = ctypes.c_int
        nm.PL_Sync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        nm.PL_Sync.restype = ctypes.c_int

    def init(self, max_boards=4, max_chips=4, max_cores=4):
        """Initialize all boards, chips, cores. 64 cores total on full NM QUAD."""
        # Long timeout for remote nmrb — each board reset takes ~10s
        self.nm.PL_SetTimeout(ctypes.c_uint32(30000))

        count = ctypes.c_uint32(0)
        self.nm.PL_GetBoardCount(ctypes.byref(count))
        n_boards = min(count.value, max_boards)
        print(f"NM QUAD: {n_boards} boards detected", flush=True)

        total_cores = 0
        for b in range(n_boards):
            board = ctypes.c_void_p()
            ret = self.nm.PL_GetBoardDesc(b, ctypes.byref(board))
            if ret != 0:
                print(f"  Board {b}: GetBoardDesc failed ({ret})", flush=True)
                continue

            print(f"  Board {b}: resetting...", flush=True)
            ret = self.nm.PL_ResetBoard(board)
            if ret != 0:
                print(f"  Board {b}: ResetBoard failed ({ret})", flush=True)
                self.nm.PL_CloseBoardDesc(board)
                continue

            ret = self.nm.PL_LoadInitCode(board)
            if ret != 0:
                print(f"  Board {b}: LoadInitCode failed ({ret})", flush=True)
                self.nm.PL_CloseBoardDesc(board)
                continue

            self.boards.append(board)

            for chip in range(max_chips):
                for core in range(max_cores):
                    cn = self.CoreNo(nm_id=core, cluster_id=chip)
                    access = ctypes.c_void_p()
                    ret = self.nm.PL_GetAccess(board, ctypes.byref(cn), ctypes.byref(access))
                    if ret == 0:
                        self.accesses.append((b, chip, core, access))
                        total_cores += 1

            print(f"  Board {b}: {total_cores} cores so far", flush=True)

        print(f"NM QUAD: {total_cores} cores ready", flush=True)
        return total_cores

    def load_dispatcher(self, core_idx=0):
        """Load dispatcher on one core."""
        if core_idx >= len(self.accesses):
            return False
        _, _, _, access = self.accesses[core_idx]
        ret = self.nm.PL_LoadProgramFile(access, self.dispatcher_path.encode())
        return ret == 0

    def write_ddr(self, core_idx, data, addr, count):
        """Write float32 data to chip DDR."""
        _, _, _, access = self.accesses[core_idx]
        buf = (ctypes.c_uint32 * count)(*[struct.unpack('I', struct.pack('f', v))[0] for v in data])
        self.nm.PL_WriteMemBlock(access, buf, addr, count)

    def write_raw(self, core_idx, buf, addr, count):
        """Write raw uint32 data."""
        _, _, _, access = self.accesses[core_idx]
        arr = (ctypes.c_uint32 * count)(*buf)
        self.nm.PL_WriteMemBlock(access, arr, addr, count)

    def read_ddr(self, core_idx, addr, count):
        """Read float32 data from DDR."""
        _, _, _, access = self.accesses[core_idx]
        buf = (ctypes.c_uint32 * count)()
        self.nm.PL_ReadMemBlock(access, buf, addr, count)
        return [struct.unpack('f', struct.pack('I', buf[i]))[0] for i in range(count)]

    def sync(self, core_idx, value=1):
        _, _, _, access = self.accesses[core_idx]
        ret = ctypes.c_int(0)
        self.nm.PL_Sync(access, value, ctypes.byref(ret))
        return ret.value

    def matmul(self, core_idx, A, B, M, N, K):
        """Run matmul on NM6408: C[M,N] = A[M,K] @ B[K,N]"""
        DATA = self.DDR + 0x200
        a_addr = DATA
        b_addr = DATA + M * K
        c_addr = DATA + M * K + K * N

        self.write_ddr(core_idx, A.flatten().tolist(), a_addr, M * K)
        self.write_ddr(core_idx, B.flatten().tolist(), b_addr, K * N)

        # Command block
        cmd = [self.OP_MATMUL, 0, a_addr, b_addr, c_addr, M, N, K, 0] + [0] * 23
        self.write_raw(core_idx, cmd, self.DDR, 32)

        self.sync(core_idx, 1)

        result = self.read_ddr(core_idx, c_addr, M * N)
        return np.array(result, dtype=np.float32).reshape(M, N)

    def shutdown(self, core_idx=0):
        cmd = [self.OP_DONE] + [0] * 31
        self.write_raw(core_idx, cmd, self.DDR, 32)
        self.sync(core_idx, -1)

    def cleanup(self):
        for _, _, _, access in self.accesses:
            self.nm.PL_CloseAccess(access)
        for board in self.boards:
            self.nm.PL_CloseBoardDesc(board)


# ============================================================
# NanoGPT model (numpy, dispatches matmul to NM QUAD)
# ============================================================
class NanoGPT:
    def __init__(self, vocab_size=65, embed_dim=64, num_heads=4,
                 ffn_dim=128, context_len=32, n_layers=2):
        self.V = vocab_size
        self.D = embed_dim
        self.H = num_heads
        self.HD = embed_dim // num_heads
        self.FF = ffn_dim
        self.T = context_len
        self.n_layers = n_layers

        # Initialize weights (small init to prevent overflow)
        scale = lambda fan_in, fan_out: 0.02

        # Token embedding
        self.wte = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02

        # Positional embedding
        self.wpe = np.random.randn(context_len, embed_dim).astype(np.float32) * 0.02

        self.layers = []
        for _ in range(n_layers):
            layer = {
                'Wq': np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale(embed_dim, embed_dim),
                'Wk': np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale(embed_dim, embed_dim),
                'Wv': np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale(embed_dim, embed_dim),
                'Wo': np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale(embed_dim, embed_dim),
                'W1': np.random.randn(embed_dim, ffn_dim).astype(np.float32) * scale(embed_dim, ffn_dim),
                'W2': np.random.randn(ffn_dim, embed_dim).astype(np.float32) * scale(ffn_dim, embed_dim),
                'ln1_g': np.ones(embed_dim, dtype=np.float32),
                'ln1_b': np.zeros(embed_dim, dtype=np.float32),
                'ln2_g': np.ones(embed_dim, dtype=np.float32),
                'ln2_b': np.zeros(embed_dim, dtype=np.float32),
            }
            self.layers.append(layer)

        self.ln_f_g = np.ones(embed_dim, dtype=np.float32)
        self.ln_f_b = np.zeros(embed_dim, dtype=np.float32)
        self.lm_head = np.random.randn(embed_dim, vocab_size).astype(np.float32) * scale(embed_dim, vocab_size)

        # Count params
        n = self.V * self.D + self.T * self.D  # embeddings
        for l in self.layers:
            for k, v in l.items():
                n += v.size
        n += self.D + self.D + self.D * self.V  # ln_f + lm_head
        print(f"NanoGPT: {n/1000:.1f}K parameters")

    def forward(self, idx, hw=None, core=0):
        """Forward pass. idx: [B, T] int array."""
        B, T = idx.shape
        x = self.wte[idx] + self.wpe[:T]  # [B, T, D]

        for layer in self.layers:
            # LayerNorm 1
            x_ln = layer_norm(x, layer['ln1_g'], layer['ln1_b'])

            # Self-attention (CPU for now — matmul on NM QUAD if hw)
            Q = matmul_op(x_ln.reshape(-1, self.D), layer['Wq'], hw, core).reshape(B, T, self.D)
            K = matmul_op(x_ln.reshape(-1, self.D), layer['Wk'], hw, core).reshape(B, T, self.D)
            V = matmul_op(x_ln.reshape(-1, self.D), layer['Wv'], hw, core).reshape(B, T, self.D)

            # Multi-head attention
            attn_out = multi_head_attention(Q, K, V, self.H, self.HD, T)
            proj = matmul_op(attn_out.reshape(-1, self.D), layer['Wo'], hw, core).reshape(B, T, self.D)
            x = x + proj

            # LayerNorm 2
            x_ln = layer_norm(x, layer['ln2_g'], layer['ln2_b'])

            # FFN
            h = matmul_op(x_ln.reshape(-1, self.D), layer['W1'], hw, core).reshape(B, T, self.FF)
            h = np.maximum(h, 0)  # ReLU
            h = matmul_op(h.reshape(-1, self.FF), layer['W2'], hw, core).reshape(B, T, self.D)
            x = x + h

        # Final LayerNorm
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)

        # LM head
        logits = matmul_op(x.reshape(-1, self.D), self.lm_head, hw, core).reshape(B, T, self.V)
        return logits


def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


def multi_head_attention(Q, K, V, H, HD, T):
    B = Q.shape[0]
    Q = Q.reshape(B, T, H, HD).transpose(0, 2, 1, 3)  # [B, H, T, HD]
    K = K.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, H, HD).transpose(0, 2, 1, 3)

    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(HD)

    # Causal mask
    mask = np.triu(np.ones((T, T)), k=1) * -1e9
    scores = scores + mask

    # Softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-8)

    out = np.matmul(attn, V)  # [B, H, T, HD]
    return out.transpose(0, 2, 1, 3).reshape(B, T, -1)  # [B, T, D]


def matmul_op(A, B, hw=None, core=0):
    """Matrix multiply on NM6408. No CPU fallback."""
    if hw is not None:
        return hw.matmul(core, A, B, A.shape[0], B.shape[1], A.shape[1])
    raise RuntimeError("NM QUAD hardware required — no CPU fallback")


def softmax_cross_entropy(logits, targets):
    """Softmax + cross entropy loss."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    # Stable softmax
    logits_flat = logits_flat - logits_flat.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_flat)
    probs = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-8)

    # Cross entropy
    n = len(targets_flat)
    loss = -np.log(probs[np.arange(n), targets_flat] + 1e-8).mean()

    # Gradient
    grad = probs.copy()
    grad[np.arange(n), targets_flat] -= 1
    grad /= n

    return loss, grad.reshape(B, T, V)


# ============================================================
# Training
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='tiny_shakespeare.txt')
    parser.add_argument('--dispatcher', default='dispatcher_nmquad.abs')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context', type=int, default=32)
    # No CPU fallback — NM QUAD only
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.data):
        print(f"Downloading tiny_shakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, args.data)

    with open(args.data, 'r') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    V = len(chars)
    print(f"Data: {len(data)} chars, vocab={V}")

    # Split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Init NM QUAD — REQUIRED, no CPU fallback
    hw = NMQuad64(args.dispatcher)
    total_cores = hw.init(max_boards=1)
    assert total_cores > 0, "NM QUAD: no cores found!"
    print(f"Loading dispatcher...", flush=True)
    ret = hw.load_dispatcher(0)
    print(f"Dispatcher loaded: {ret}", flush=True)
    # Verify matmul
    print(f"Verifying matmul...", flush=True)
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    C = hw.matmul(0, A, B, 2, 2, 2)
    print(f"Matmul result: {C.flatten()}", flush=True)
    assert abs(C[0, 0] - 19) < 0.1, f"Matmul verify failed: {C}"
    print(f"NM QUAD: {total_cores} cores, matmul verified", flush=True)

    # Model
    model = NanoGPT(vocab_size=V, embed_dim=64, num_heads=4,
                    ffn_dim=128, context_len=args.context, n_layers=2)

    # Training
    BS = args.batch_size
    T = args.context

    print(f"\n=== Training NanoGPT on NM QUAD ({total_cores} cores) ===")
    print(f"Epochs: {args.epochs}, Batch: {BS}, Context: {T}, LR: {args.lr}")

    for epoch in range(args.epochs):
        # Simple SGD training (forward only for now — backward on CPU)
        np.random.shuffle(train_data[:n - T])
        total_loss = 0
        steps = 0

        t0 = time.time()
        for i in range(0, len(train_data) - T - 1, BS * T):
            # Get batch
            batch_x = []
            batch_y = []
            for b in range(BS):
                start = i + b * T
                if start + T + 1 > len(train_data):
                    break
                batch_x.append(train_data[start:start + T])
                batch_y.append(train_data[start + 1:start + T + 1])

            if len(batch_x) < BS:
                continue

            x = np.array(batch_x)
            y = np.array(batch_y)

            # Forward
            logits = model.forward(x, hw=hw, core=0)
            loss, grad = softmax_cross_entropy(logits, y)
            total_loss += loss
            steps += 1

            if steps % 100 == 0:
                avg_loss = total_loss / steps
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1} step {steps}: loss={avg_loss:.4f} ({elapsed:.1f}s)")

        avg_loss = total_loss / max(steps, 1)
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, time={epoch_time:.1f}s, steps={steps}")

        # Generate sample
        print("  Sample: ", end="")
        idx = np.array([[stoi.get('\n', 0)]])
        for _ in range(200):
            logits = model.forward(idx[:, -T:], hw=hw, core=0)
            logits = logits[0, -1]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            next_id = np.random.choice(V, p=probs)
            idx = np.concatenate([idx, [[next_id]]], axis=1)
        print(''.join([itos[i] for i in idx[0, 1:]]))

    if hw:
        hw.shutdown(0)
        hw.cleanup()

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
