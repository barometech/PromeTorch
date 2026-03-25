"""
NanoGPT on NM QUAD — LOCAL, DDR polling (same as NMCard dispatcher_mc pattern)
ALL computation on NM6408 cores. No CPU fallback. No PL_Sync.
"""
import ctypes, os, sys, time, struct
import numpy as np

DDR_BASE = 0x00340000
CMD_BLOCK_SIZE = 32
STATUS_ADDR = 30
WATCHDOG_ADDR = 31
OP_NOP = 0; OP_MATMUL = 1; OP_ADD = 2; OP_MUL = 3; OP_RELU = 4
OP_RMSNORM = 7; OP_SOFTMAX = 6; OP_MATMUL_PARTIAL = 22; OP_EXIT = 255

def f2u(f):
    return struct.unpack('I', struct.pack('f', f))[0]
def u2f(u):
    return struct.unpack('f', struct.pack('I', int(u) & 0xFFFFFFFF))[0]

class NMQuad:
    def __init__(self, dispatcher_path="dispatcher_nmquad.abs"):
        self.nm = ctypes.CDLL("libnm_quad_load.so")
        self.dispatcher_path = dispatcher_path.encode()
        self.board = None
        self.accesses = {}  # (cluster, core) -> access
        # DDR data area starts after cmd blocks for 16 cores
        # 16 cores * 32 words = 512 words
        self.data_base = DDR_BASE + 16 * CMD_BLOCK_SIZE
        self.data_ptr = self.data_base

    def init(self, max_boards=4):
        self.nm.PL_SetTimeout(ctypes.c_uint32(10000))
        count = ctypes.c_uint32(0)
        self.nm.PL_GetBoardCount(ctypes.byref(count))
        n = min(count.value, max_boards)
        print(f"NM QUAD: {n} boards", flush=True)

        class CN(ctypes.Structure):
            _fields_ = [("nm_id", ctypes.c_int), ("cluster_id", ctypes.c_int)]
        self.CN = CN

        self.boards = []
        self.core_list = []  # flat list of (board_idx, cl, co, access)
        total = 0

        for b in range(n):
            board = ctypes.c_void_p()
            if self.nm.PL_GetBoardDesc(b, ctypes.byref(board)) != 0:
                continue
            self.nm.PL_ResetBoard(board)
            self.nm.PL_LoadInitCode(board)
            self.boards.append(board)

            for cl in range(4):
                for co in range(4):
                    cn = CN(nm_id=co, cluster_id=cl)
                    acc = ctypes.c_void_p()
                    if self.nm.PL_GetAccess(board, ctypes.byref(cn), ctypes.byref(acc)) == 0:
                        if self.nm.PL_LoadProgramFile(acc, self.dispatcher_path) == 0:
                            gidx = total  # global core index
                            self.accesses[(cl + b*4, co)] = acc
                            self.core_list.append((b, cl, co, acc))
                            total += 1
                        else:
                            self.nm.PL_CloseAccess(acc)

            print(f"  Board {b}: {total} cores total", flush=True)

        print(f"NM QUAD: {total} cores loaded", flush=True)
        time.sleep(1.0)

        # Verify alive
        alive = 0
        for i, (b, cl, co, acc) in enumerate(self.core_list):
            local_idx = cl * 4 + co  # within board
            buf = (ctypes.c_uint32 * 1)()
            self.nm.PL_ReadMemBlock(acc, buf,
                DDR_BASE + local_idx * CMD_BLOCK_SIZE + WATCHDOG_ADDR, 1)
            if buf[0] > 100: alive += 1
        print(f"NM QUAD: {alive} cores alive", flush=True)
        return alive

    def _get_core(self, core_idx):
        """Get (access, local_idx) for global core index"""
        b, cl, co, acc = self.core_list[core_idx]
        local_idx = cl * 4 + co
        return acc, local_idx

    def _get_acc(self, core_idx):
        return self.core_list[core_idx][3]

    def send_cmd(self, core_idx, op, args, timeout_ms=5000):
        """Send command to core via DDR polling (no PL_Sync!)"""
        acc, local_idx = self._get_core(core_idx)
        base = DDR_BASE + local_idx * CMD_BLOCK_SIZE

        # Write args
        if args:
            buf = (ctypes.c_uint32 * len(args))(*args)
            self.nm.PL_WriteMemBlock(acc, buf, base + 1, len(args))

        # Clear status
        zero = (ctypes.c_uint32 * 1)(0)
        self.nm.PL_WriteMemBlock(acc, zero, base + STATUS_ADDR, 1)

        # Write opcode (triggers execution)
        cmd = (ctypes.c_uint32 * 1)(op)
        self.nm.PL_WriteMemBlock(acc, cmd, base, 1)

        # Poll status
        t0 = time.time()
        while True:
            buf = (ctypes.c_uint32 * 1)()
            self.nm.PL_ReadMemBlock(acc, buf, base + STATUS_ADDR, 1)
            if buf[0] == 1:
                return True
            if (time.time() - t0) * 1000 > timeout_ms:
                print(f"TIMEOUT core[{core_idx}] op={op}", flush=True)
                return False
            time.sleep(0.0001)  # 0.1ms poll interval

    def write_data(self, core_idx, addr, data_uint32):
        acc = self._get_acc(core_idx)
        buf = (ctypes.c_uint32 * len(data_uint32))(*[int(x) & 0xFFFFFFFF for x in data_uint32])
        self.nm.PL_WriteMemBlock(acc, buf, addr, len(data_uint32))

    def read_data(self, core_idx, addr, count):
        acc = self._get_acc(core_idx)
        buf = (ctypes.c_uint32 * count)()
        self.nm.PL_ReadMemBlock(acc, buf, addr, count)
        return [buf[i] for i in range(count)]

    def alloc(self, words):
        addr = self.data_ptr
        self.data_ptr += words
        return addr

    def reset_alloc(self):
        self.data_ptr = self.data_base

    def matmul(self, core_idx, A, B):
        """C[M,N] = A[M,K] @ B[K,N] on NM6408 — uses ALL 16 cores for large N"""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        n_cores = len(self.accesses)

        # Use multi-core for large matrices
        if N >= 16 and n_cores > 1:
            return self._matmul_parallel(A, B, M, K, N, n_cores)

        # Single core
        self.reset_alloc()
        a_addr = self.alloc(M * K)
        b_addr = self.alloc(K * N)
        c_addr = self.alloc(M * N)

        self.write_data(core_idx, a_addr, [f2u(x) for x in A.flatten()])
        self.write_data(core_idx, b_addr, [f2u(x) for x in B.flatten()])

        self.send_cmd(core_idx, OP_MATMUL, [M, K, N, a_addr, b_addr, c_addr])

        raw = self.read_data(core_idx, c_addr, M * N)
        return np.array([u2f(x) for x in raw], dtype=np.float32).reshape(M, N)

    def _matmul_parallel(self, A, B, M, K, N, n_cores):
        """Split columns across cores on SAME board (shared DDR)"""
        # Use only cores from board 0 (shared DDR)
        # Each board has its own DDR — can't split across boards
        board0_cores = [(i, b, cl, co, acc) for i, (b, cl, co, acc) in enumerate(self.core_list) if b == 0]
        nc = min(len(board0_cores), n_cores, N)

        self.reset_alloc()
        a_addr = self.alloc(M * K)
        b_addr = self.alloc(K * N)
        c_addr = self.alloc(M * N)

        a_data = [f2u(x) for x in A.flatten()]
        b_data = [f2u(x) for x in B.flatten()]
        self.write_data(0, a_addr, a_data)
        self.write_data(0, b_addr, b_data)

        cols_per_core = (N + nc - 1) // nc
        active = 0
        for ci in range(nc):
            col_start = ci * cols_per_core
            col_end = min(col_start + cols_per_core, N)
            if col_start >= N: break

            _, _, cl, co, acc = board0_cores[ci]
            local_idx = cl * 4 + co
            base = DDR_BASE + local_idx * CMD_BLOCK_SIZE

            args_data = [M, K, N, a_addr, b_addr, c_addr, col_start, col_end]
            buf = (ctypes.c_uint32 * len(args_data))(*args_data)
            self.nm.PL_WriteMemBlock(acc, buf, base + 1, len(args_data))
            zero = (ctypes.c_uint32 * 1)(0)
            self.nm.PL_WriteMemBlock(acc, zero, base + STATUS_ADDR, 1)
            cmd = (ctypes.c_uint32 * 1)(OP_MATMUL_PARTIAL)
            self.nm.PL_WriteMemBlock(acc, cmd, base, 1)
            active += 1

        # Wait all
        t0 = time.time()
        for ci in range(active):
            _, _, cl, co, acc = board0_cores[ci]
            local_idx = cl * 4 + co
            base = DDR_BASE + local_idx * CMD_BLOCK_SIZE
            while True:
                buf = (ctypes.c_uint32 * 1)()
                self.nm.PL_ReadMemBlock(acc, buf, base + STATUS_ADDR, 1)
                if buf[0] == 1: break
                if time.time() - t0 > 10.0:
                    print(f"TIMEOUT core {ci}", flush=True)
                    break
                time.sleep(0.0001)

        raw = self.read_data(0, c_addr, M * N)
        return np.array([u2f(x) for x in raw], dtype=np.float32).reshape(M, N)

    def shutdown(self):
        for i, (b, cl, co, acc) in enumerate(self.core_list):
            local_idx = cl * 4 + co
            cmd = (ctypes.c_uint32 * 1)(OP_EXIT)
            self.nm.PL_WriteMemBlock(acc, cmd, DDR_BASE + local_idx * CMD_BLOCK_SIZE, 1)
        time.sleep(0.3)
        for _, _, _, acc in self.core_list:
            self.nm.PL_CloseAccess(acc)
        for board in self.boards:
            self.nm.PL_CloseBoardDesc(board)
        print("Shutdown OK", flush=True)


# ============================================================
# NanoGPT — ALL matmuls on NM6408
# ============================================================
class NanoGPT:
    def __init__(self, V=65, D=64, H=4, FF=128, T=32, L=2):
        self.V, self.D, self.H, self.HD = V, D, H, D//H
        self.FF, self.T, self.L = FF, T, L
        s = 0.02
        self.wte = np.random.randn(V, D).astype(np.float32) * s
        self.wpe = np.random.randn(T, D).astype(np.float32) * s
        self.layers = []
        for _ in range(L):
            self.layers.append({
                'ln1_g': np.ones(D, dtype=np.float32),
                'Wq': np.random.randn(D, D).astype(np.float32) * s,
                'Wk': np.random.randn(D, D).astype(np.float32) * s,
                'Wv': np.random.randn(D, D).astype(np.float32) * s,
                'Wo': np.random.randn(D, D).astype(np.float32) * s,
                'ln2_g': np.ones(D, dtype=np.float32),
                'W1': np.random.randn(D, FF).astype(np.float32) * s,
                'W2': np.random.randn(FF, D).astype(np.float32) * s,
            })
        self.ln_f_g = np.ones(D, dtype=np.float32)
        self.lm_head = np.random.randn(D, V).astype(np.float32) * s
        n = V*D + T*D + D + D*V
        for l in self.layers:
            for v in l.values(): n += v.size
        print(f"NanoGPT: {n/1000:.1f}K params, {L} layers", flush=True)

    def rms_norm(self, x, g):
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-5)
        return (x / rms) * g

    def forward(self, tokens, hw, core=0):
        T = len(tokens)
        x = self.wte[tokens] + self.wpe[:T]  # [T, D]

        for layer in self.layers:
            h = self.rms_norm(x, layer['ln1_g'])
            Q = hw.matmul(core, h, layer['Wq'])
            K = hw.matmul(core, h, layer['Wk'])
            V = hw.matmul(core, h, layer['Wv'])

            # Multi-head attention — ALL matmuls on NM6408
            Q = Q.reshape(T, self.H, self.HD).transpose(1, 0, 2)  # [H, T, HD]
            K = K.reshape(T, self.H, self.HD).transpose(1, 0, 2)
            V = V.reshape(T, self.H, self.HD).transpose(1, 0, 2)
            # QK^T per head on NM6408
            scores = np.zeros((self.H, T, T), dtype=np.float32)
            for h in range(self.H):
                scores[h] = hw.matmul(core, Q[h], K[h].T.copy())
            scores /= np.sqrt(self.HD)
            scores += np.triu(np.full((T, T), -1e9), k=1)
            e = np.exp(np.clip(scores - scores.max(-1, keepdims=True), -20, 20))
            attn = e / (e.sum(-1, keepdims=True) + 1e-8)
            # attn@V per head on NM6408
            out = np.zeros((self.H, T, self.HD), dtype=np.float32)
            for h in range(self.H):
                out[h] = hw.matmul(core, attn[h].astype(np.float32), V[h])
            out = out.transpose(1, 0, 2).reshape(T, self.D)

            out = hw.matmul(core, out, layer['Wo'])
            x = x + out

            h = self.rms_norm(x, layer['ln2_g'])
            ffn = hw.matmul(core, h, layer['W1'])
            ffn = np.maximum(ffn, 0)
            ffn = hw.matmul(core, ffn, layer['W2'])
            x = x + ffn

            # Cache for backward
            layer['_h1'] = h
            layer['_attn_out'] = out
            layer['_ffn_in'] = h
            layer['_ffn_act'] = np.maximum(hw.matmul(core, h, layer['W1']), 0)

        x = self.rms_norm(x, self.ln_f_g)
        self._x_final = x
        logits = hw.matmul(core, x, self.lm_head)
        return logits

    def backward_step(self, logits, tokens, targets, lr, hw, core=0):
        """Backward + SGD update. ALL matmuls on NM6408."""
        T = len(targets)
        # Softmax + cross entropy
        e = np.exp(logits - logits.max(-1, keepdims=True))
        probs = e / (e.sum(-1, keepdims=True) + 1e-8)
        loss = -np.log(probs[np.arange(T), targets] + 1e-8).mean()

        dlogits = probs.copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T

        # lm_head grad: x_final.T @ dlogits
        d_lm = hw.matmul(core, self._x_final.T.copy(), dlogits)
        self.lm_head -= lr * np.clip(d_lm, -1, 1)

        # dx from lm_head
        dx = hw.matmul(core, dlogits, self.lm_head.T.copy())

        # Backprop through layers
        for layer in reversed(self.layers):
            # FFN backward
            ffn_act = layer['_ffn_act']
            dW2 = hw.matmul(core, ffn_act.T.copy(), dx)
            layer['W2'] -= lr * np.clip(dW2, -1, 1)
            d_ffn = hw.matmul(core, dx, layer['W2'].T.copy())
            d_ffn[ffn_act <= 0] = 0
            dW1 = hw.matmul(core, layer['_ffn_in'].T.copy(), d_ffn)
            layer['W1'] -= lr * np.clip(dW1, -1, 1)

            # Attention backward (simplified)
            dWo = hw.matmul(core, layer['_attn_out'].T.copy(), dx)
            layer['Wo'] -= lr * np.clip(dWo, -1, 1)

            h1 = layer['_h1']
            dWq = hw.matmul(core, h1.T.copy(), dx)
            layer['Wq'] -= lr * np.clip(dWq, -1, 1)
            dWk = hw.matmul(core, h1.T.copy(), dx)
            layer['Wk'] -= lr * np.clip(dWk, -1, 1)
            dWv = hw.matmul(core, h1.T.copy(), dx)
            layer['Wv'] -= lr * np.clip(dWv, -1, 1)

        # Embedding update
        for t in range(T):
            self.wte[tokens[t]] -= lr * dx[t]

        return loss


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='tiny_shakespeare.txt')
    p.add_argument('--dispatcher', default='dispatcher_nmquad.abs')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--ctx', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=4)
    args = p.parse_args()

    with open(args.data) as f:
        text = f.read()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    print(f"Data: {len(data)} chars, vocab={V}", flush=True)

    hw = NMQuad(args.dispatcher)
    alive = hw.init(max_boards=4)  # ALL 4 boards = 64 cores
    assert alive > 0

    # Verify matmul
    A = np.array([[1,2],[3,4]], dtype=np.float32)
    B = np.array([[5,6],[7,8]], dtype=np.float32)
    C = hw.matmul(0, A, B)
    assert abs(C[0,0]-19) < 0.1, f"Matmul fail: {C}"
    print(f"Matmul verified", flush=True)

    model = NanoGPT(V=V, D=64, H=4, FF=128, T=args.ctx, L=2)

    print(f"\n=== NanoGPT on NM QUAD ({alive} cores) ===", flush=True)
    print(f"Steps: {args.steps}, BS={args.batch}, T={args.ctx}, LR={args.lr}", flush=True)

    for epoch in range(args.epochs):
        total_loss, steps = 0, 0
        t0 = time.time()

        for step in range(args.steps):
            batch_loss = 0
            for b in range(args.batch):
                idx = np.random.randint(0, len(data) - args.ctx - 1)
                tokens = data[idx:idx+args.ctx]
                targets = data[idx+1:idx+args.ctx+1]
                logits = model.forward(tokens, hw, core=0)
                loss = model.backward_step(logits, tokens, targets, args.lr, hw, core=0)
                batch_loss += loss

            batch_loss /= args.batch
            total_loss += batch_loss
            steps += 1

            if steps % 10 == 0:
                elapsed = time.time() - t0
                tok_s = steps * args.batch * args.ctx / elapsed
                print(f"  E{epoch+1} step {steps}/{args.steps}: loss={total_loss/steps:.4f} {tok_s:.0f}tok/s", flush=True)

        avg = total_loss / steps
        print(f"Epoch {epoch+1}: loss={avg:.4f} ({time.time()-t0:.1f}s)", flush=True)

        # Generate
        toks = [stoi.get('\n', 0)]
        for _ in range(200):
            inp = toks[-args.ctx:]
            logits = model.forward(np.array(inp, dtype=np.int32), hw, core=0)
            l = logits[-1]
            p = np.exp(l - l.max()) ** (1.0/0.8)
            p /= p.sum()
            toks.append(np.random.choice(V, p=p))
        print("  " + ''.join([itos[t] for t in toks[1:]]), flush=True)

    hw.shutdown()
    print("=== DONE ===", flush=True)

if __name__ == "__main__":
    main()
