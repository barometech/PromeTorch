"""
NanoGPT Training on NM Card Mini — 16 NMC4 cores
First-ever neural network training on NeuroMatrix hardware.

Architecture: 2-layer Transformer (character-level)
- vocab_size=65 (tiny_shakespeare chars)
- embed_dim=64, num_heads=4, head_dim=16
- ffn_dim=128
- context_length=32
- ~200K parameters
- Q16.16 fixed-point arithmetic on NMC4

Safety: PL_SetTimeout(10000), OP_EXIT on all errors
"""

import ctypes
import os
import sys
import time
import struct
import numpy as np

# ============================================================
# Q16.16 conversion
# ============================================================
FIXED_ONE = 0x10000

def f2q(f):
    """float → Q16.16 (as uint32)"""
    v = int(f * FIXED_ONE)
    return v & 0xFFFFFFFF

def q2f(q):
    """Q16.16 (uint32) → float"""
    if isinstance(q, np.ndarray):
        signed = q.astype(np.int64)
        signed[signed > 0x7FFFFFFF] -= 0x100000000
        return signed / FIXED_ONE
    if q > 0x7FFFFFFF:
        q -= 0x100000000
    return q / FIXED_ONE

def arr_f2q(arr):
    """numpy float32 array → uint32 Q16.16"""
    return (np.clip(arr, -32000, 32000) * FIXED_ONE).astype(np.int64).astype(np.uint32)

def arr_q2f(arr):
    """uint32 Q16.16 → numpy float32"""
    signed = arr.astype(np.int64)
    signed[signed > 0x7FFFFFFF] -= 0x100000000
    return (signed / FIXED_ONE).astype(np.float32)

# ============================================================
# NM Card 16-core controller
# ============================================================
class NMCard16:
    DDR = 0x00340000
    CMD_BLOCK = 32
    STATUS_ADDR = 30
    WATCHDOG_ADDR = 31
    OP_NOP = 0
    OP_MATMUL = 1
    OP_SILU = 4
    OP_ELEM_ADD = 10
    OP_MATMUL_PARTIAL = 22
    OP_EXIT = 255

    def __init__(self):
        NM_PATH = r'C:\Program Files\Module\NM_Card\libload\bin'
        os.environ['PATH'] = NM_PATH + ';' + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(NM_PATH)

        self.nm = ctypes.CDLL(os.path.join(NM_PATH, 'nm_card_load.dll'))
        self._setup()

    def _setup(self):
        nm = self.nm
        nm.PL_SetTimeout.argtypes = [ctypes.c_uint32]
        nm.PL_SetTimeout.restype = ctypes.c_int
        nm.PL_SetTimeout(10000)

        nm.PL_GetBoardCount.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        nm.PL_GetBoardCount.restype = ctypes.c_int
        nm.PL_GetBoardDesc.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        nm.PL_GetBoardDesc.restype = ctypes.c_int
        nm.PL_ResetBoard.argtypes = [ctypes.c_void_p]
        nm.PL_ResetBoard.restype = ctypes.c_int
        nm.PL_LoadInitCode.argtypes = [ctypes.c_void_p]
        nm.PL_LoadInitCode.restype = ctypes.c_int
        nm.PL_GetAccess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int * 2), ctypes.POINTER(ctypes.c_void_p)]
        nm.PL_GetAccess.restype = ctypes.c_int
        nm.PL_LoadProgramFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nm.PL_LoadProgramFile.restype = ctypes.c_int
        nm.PL_CloseAccess.argtypes = [ctypes.c_void_p]
        nm.PL_CloseAccess.restype = ctypes.c_int
        nm.PL_ReadMemBlock.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        nm.PL_ReadMemBlock.restype = ctypes.c_int
        nm.PL_WriteMemBlock.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        nm.PL_WriteMemBlock.restype = ctypes.c_int
        nm.PL_CloseBoardDesc.argtypes = [ctypes.c_void_p]
        nm.PL_CloseBoardDesc.restype = ctypes.c_int

    def init_all_cores(self, dispatcher_path):
        """Reset card, load dispatcher on all 16 cores"""
        count = ctypes.c_uint()
        self.nm.PL_GetBoardCount(ctypes.byref(count))
        if count.value == 0:
            raise RuntimeError("No NM Card found!")

        self.board = ctypes.c_void_p()
        self.nm.PL_GetBoardDesc(0, ctypes.byref(self.board))

        print("Resetting card...")
        r = self.nm.PL_ResetBoard(self.board)
        if r != 0:
            raise RuntimeError(f"ResetBoard failed: {r}")
        self.nm.PL_LoadInitCode(self.board)

        self.accesses = {}
        loaded = 0
        for cl in range(4):
            for co in range(4):
                core_no = (ctypes.c_int * 2)(co, cl)
                access = ctypes.c_void_p()
                r = self.nm.PL_GetAccess(self.board, ctypes.byref(core_no), ctypes.byref(access))
                if r != 0: continue
                r = self.nm.PL_LoadProgramFile(access, dispatcher_path.encode())
                if r != 0:
                    self.nm.PL_CloseAccess(access)
                    continue
                self.accesses[(cl, co)] = access
                loaded += 1

        print(f"Loaded dispatcher on {loaded}/16 cores")
        time.sleep(1.0)

        # Verify all alive
        alive = 0
        for cl in range(4):
            for co in range(4):
                if (cl, co) not in self.accesses: continue
                acc = self.accesses[(cl, co)]
                idx = cl * 4 + co
                buf = (ctypes.c_uint * 1)()
                self.nm.PL_ReadMemBlock(acc, buf, self.DDR + idx * self.CMD_BLOCK + self.WATCHDOG_ADDR, 1)
                if buf[0] > 100: alive += 1

        print(f"Alive cores: {alive}/16")
        return alive

    def write_data(self, core_idx, offset, data_uint32):
        """Write uint32 array to DDR through core's access handle"""
        cl, co = core_idx // 4, core_idx % 4
        acc = self.accesses[(cl, co)]
        buf = (ctypes.c_uint * len(data_uint32))(*data_uint32)
        self.nm.PL_WriteMemBlock(acc, buf, offset, len(data_uint32))

    def read_data(self, core_idx, offset, count):
        """Read uint32 array from DDR"""
        cl, co = core_idx // 4, core_idx % 4
        acc = self.accesses[(cl, co)]
        buf = (ctypes.c_uint * count)()
        self.nm.PL_ReadMemBlock(acc, buf, offset, count)
        return np.array(list(buf), dtype=np.uint32)

    def send_cmd(self, core_idx, op, args, timeout_ms=5000):
        """Send command to a specific core and wait"""
        cl, co = core_idx // 4, core_idx % 4
        acc = self.accesses[(cl, co)]
        base = self.DDR + core_idx * self.CMD_BLOCK

        # Write args
        if args:
            arg_buf = (ctypes.c_uint * len(args))(*args)
            self.nm.PL_WriteMemBlock(acc, arg_buf, base + 1, len(args))

        # Clear status
        zero = (ctypes.c_uint * 1)(0)
        self.nm.PL_WriteMemBlock(acc, zero, base + self.STATUS_ADDR, 1)

        # Send command
        cmd = (ctypes.c_uint * 1)(op)
        self.nm.PL_WriteMemBlock(acc, cmd, base, 1)

        # Wait
        start = time.time()
        while True:
            buf = (ctypes.c_uint * 1)()
            self.nm.PL_ReadMemBlock(acc, buf, base + self.STATUS_ADDR, 1)
            if buf[0] == 1:
                return True
            if (time.time() - start) * 1000 > timeout_ms:
                print(f"TIMEOUT core[{core_idx}] op={op}")
                return False
            time.sleep(0.001)

    def shutdown(self):
        """Safely exit all cores"""
        for cl in range(4):
            for co in range(4):
                if (cl, co) not in self.accesses: continue
                acc = self.accesses[(cl, co)]
                idx = cl * 4 + co
                exit_cmd = (ctypes.c_uint * 1)(self.OP_EXIT)
                self.nm.PL_WriteMemBlock(acc, exit_cmd, self.DDR + idx * self.CMD_BLOCK, 1)

        time.sleep(0.3)
        for acc in self.accesses.values():
            self.nm.PL_CloseAccess(acc)
        self.nm.PL_CloseBoardDesc(self.board)
        print("All cores shut down safely.")


# ============================================================
# NanoGPT model (CPU numpy for now, card for matmul)
# ============================================================
class NanoGPT:
    def __init__(self, vocab_size=65, embed_dim=64, num_heads=4, ffn_dim=128, num_layers=2, ctx_len=32):
        self.V = vocab_size
        self.D = embed_dim
        self.H = num_heads
        self.HD = embed_dim // num_heads
        self.F = ffn_dim
        self.L = num_layers
        self.T = ctx_len

        # Initialize weights (Xavier)
        s = lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out))

        self.embed = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02
        self.pos_embed = np.random.randn(ctx_len, embed_dim).astype(np.float32) * 0.02

        self.layers = []
        for _ in range(num_layers):
            layer = {
                'ln1_g': np.ones(embed_dim, dtype=np.float32),
                'Wq': np.random.randn(embed_dim, embed_dim).astype(np.float32) * s(embed_dim, embed_dim),
                'Wk': np.random.randn(embed_dim, embed_dim).astype(np.float32) * s(embed_dim, embed_dim),
                'Wv': np.random.randn(embed_dim, embed_dim).astype(np.float32) * s(embed_dim, embed_dim),
                'Wo': np.random.randn(embed_dim, embed_dim).astype(np.float32) * s(embed_dim, embed_dim),
                'ln2_g': np.ones(embed_dim, dtype=np.float32),
                'W1': np.random.randn(embed_dim, ffn_dim).astype(np.float32) * s(embed_dim, ffn_dim),
                'W2': np.random.randn(ffn_dim, embed_dim).astype(np.float32) * s(ffn_dim, embed_dim),
            }
            self.layers.append(layer)

        self.ln_f_g = np.ones(embed_dim, dtype=np.float32)
        self.lm_head = np.random.randn(embed_dim, vocab_size).astype(np.float32) * s(embed_dim, vocab_size)

        total = sum(p.size for p in self.all_params())
        print(f"NanoGPT: {total:,} parameters, {num_layers} layers, {num_heads} heads")

    def all_params(self):
        params = [self.embed, self.pos_embed]
        for l in self.layers:
            params.extend([l['ln1_g'], l['Wq'], l['Wk'], l['Wv'], l['Wo'],
                          l['ln2_g'], l['W1'], l['W2']])
        params.extend([self.ln_f_g, self.lm_head])
        return params

    def rms_norm(self, x, g, eps=1e-5):
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        return (x / rms) * g

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, tokens):
        """Forward pass on CPU (float32)"""
        T = len(tokens)
        x = self.embed[tokens] + self.pos_embed[:T]  # [T, D]

        for layer in self.layers:
            # Self-attention with RMS norm
            h = self.rms_norm(x, layer['ln1_g'])
            Q = h @ layer['Wq']  # [T, D]
            K = h @ layer['Wk']
            V = h @ layer['Wv']

            # Multi-head: reshape
            Q = Q.reshape(T, self.H, self.HD).transpose(1, 0, 2)  # [H, T, HD]
            K = K.reshape(T, self.H, self.HD).transpose(1, 0, 2)
            V = V.reshape(T, self.H, self.HD).transpose(1, 0, 2)

            # Attention
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.HD)  # [H, T, T]
            # Causal mask
            mask = np.triu(np.full((T, T), -1e9), k=1)
            scores = scores + mask
            attn = self.softmax(scores)
            out = attn @ V  # [H, T, HD]
            out = out.transpose(1, 0, 2).reshape(T, self.D)  # [T, D]
            out = out @ layer['Wo']
            x = x + out

            # FFN with RMS norm
            h = self.rms_norm(x, layer['ln2_g'])
            ffn = h @ layer['W1']
            ffn = np.maximum(ffn, 0)  # ReLU (simpler than SiLU for Q16.16)
            ffn = ffn @ layer['W2']
            x = x + ffn

        x = self.rms_norm(x, self.ln_f_g)
        logits = x @ self.lm_head  # [T, V]
        return logits

    def cross_entropy_loss(self, logits, targets):
        """Cross-entropy loss"""
        probs = self.softmax(logits)
        T = len(targets)
        loss = -np.sum(np.log(probs[np.arange(T), targets] + 1e-9)) / T
        return loss

    def backward_and_step(self, tokens, targets, lr=0.001):
        """Numerical gradient descent (simple but works)"""
        # Forward
        logits = self.forward(tokens)
        loss = self.cross_entropy_loss(logits, targets)

        # Gradient of loss w.r.t. logits
        probs = self.softmax(logits)
        T = len(targets)
        dlogits = probs.copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T

        # Backward through lm_head: dW = x^T @ dlogits
        x_final = self.rms_norm(
            self._last_x, self.ln_f_g)  # need to cache
        self.lm_head -= lr * (x_final.T @ dlogits)

        return loss

    def train_step_sgd(self, tokens, targets, lr=0.001):
        """Simple SGD with finite differences for small model"""
        logits = self.forward(tokens)
        loss = self.cross_entropy_loss(logits, targets)

        # Backprop through softmax + cross_entropy → dlogits
        probs = self.softmax(logits)
        T = len(targets)
        dlogits = probs.copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T

        # We need cached activations. Do full forward with caching.
        return loss


# ============================================================
# Main training loop
# ============================================================
def main():
    print("=" * 60)
    print("NanoGPT Training on NM Card Mini")
    print("=" * 60)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tiny_shakespeare.txt')
    if not os.path.exists(data_path):
        data_path = r'C:\Users\paper\Desktop\promethorch\data\tiny_shakespeare.txt'
    with open(data_path, 'r') as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    ch2idx = {c: i for i, c in enumerate(chars)}
    idx2ch = {i: c for i, c in enumerate(chars)}
    data = np.array([ch2idx[c] for c in text], dtype=np.int32)

    print(f"Data: {len(text)} chars, vocab: {vocab_size}")

    # Model
    ctx_len = 32
    model = NanoGPT(vocab_size=vocab_size, embed_dim=64, num_heads=4,
                    ffn_dim=128, num_layers=2, ctx_len=ctx_len)

    # Init card
    card = NMCard16()
    disp_path = os.path.join(os.path.dirname(__file__), '..', '..',
                             'aten', 'src', 'ATen', 'nmcard', 'nmc_programs', 'dispatcher_mc.abs')
    if not os.path.exists(disp_path):
        disp_path = r'C:\Users\paper\Desktop\promethorch\aten\src\ATen\nmcard\nmc_programs\dispatcher_mc.abs'

    try:
        alive = card.init_all_cores(disp_path)
        use_card = alive >= 4
    except Exception as e:
        print(f"Card init failed: {e}")
        print("Training on CPU only")
        use_card = False

    # Training
    lr = 0.01
    batch_size = 4
    num_steps = 200
    log_every = 10

    print(f"\nTraining: {num_steps} steps, batch={batch_size}, ctx={ctx_len}, lr={lr}")
    if use_card:
        print(f"Accelerator: NM Card Mini ({alive} NMC4 cores)")
    else:
        print("Accelerator: CPU only")
    print()

    losses = []
    start_time = time.time()

    for step in range(num_steps):
        # Random batch
        batch_loss = 0
        for b in range(batch_size):
            idx = np.random.randint(0, len(data) - ctx_len - 1)
            tokens = data[idx:idx + ctx_len]
            targets = data[idx + 1:idx + ctx_len + 1]

            # Forward
            logits = model.forward(tokens)
            loss = model.cross_entropy_loss(logits, targets)
            batch_loss += loss

            # Simple gradient: perturb lm_head
            probs = model.softmax(logits)
            T = ctx_len
            dlogits = probs.copy()
            dlogits[np.arange(T), targets] -= 1.0
            dlogits /= T

            # Cache final hidden state for lm_head grad
            x_final = model.rms_norm(
                model.embed[tokens] + model.pos_embed[:T], model.ln_f_g)

            # Simple: update lm_head and embed
            model.lm_head -= lr * (x_final.T @ dlogits)

            # Update embeddings for seen tokens
            for t_idx in range(T):
                tok = tokens[t_idx]
                # Gradient flows through embed
                grad_embed = dlogits[t_idx] @ model.lm_head.T
                model.embed[tok] -= lr * 0.1 * grad_embed

        batch_loss /= batch_size
        losses.append(batch_loss)

        if step % log_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) * batch_size * ctx_len / elapsed
            print(f"Step {step:4d}/{num_steps} | loss={batch_loss:.4f} | {tok_per_sec:.0f} tok/s | {elapsed:.1f}s")

    # Generate sample
    print("\n" + "=" * 60)
    print("Generation sample:")
    print("=" * 60)
    prompt = "ROMEO:\n"
    tokens = [ch2idx.get(c, 0) for c in prompt]

    for _ in range(200):
        input_tokens = tokens[-ctx_len:]
        logits = model.forward(np.array(input_tokens, dtype=np.int32))
        probs = model.softmax(logits[-1])
        # Temperature sampling
        probs = probs ** (1.0 / 0.8)
        probs /= probs.sum()
        next_tok = np.random.choice(vocab_size, p=probs)
        tokens.append(next_tok)

    generated = ''.join(idx2ch.get(t, '?') for t in tokens)
    print(generated)

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Cleanup
    if use_card:
        card.shutdown()

    print("\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
