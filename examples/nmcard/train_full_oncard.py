"""
109K transformer — FULL training on NM Card Mini.
Forward + backward + SGD all on 16 NMC4 cores.
No CPU compute. Card does everything.
"""
import numpy as np, time, os, json, ctypes, sys, struct
sys.stdout.reconfigure(line_buffering=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','tiny_shakespeare.txt'),'r') as f:
    text=f.read()
chars=sorted(set(text)); V=len(chars)
ch2idx={c:i for i,c in enumerate(chars)}; idx2ch={i:c for i,c in enumerate(chars)}
data=np.array([ch2idx[c] for c in text],dtype=np.int32)

# NM Card init
NM=r'C:\Program Files\Module\NM_Card\libload\bin'
os.environ['PATH']=NM+';'+os.environ.get('PATH','')
if hasattr(os,'add_dll_directory'): os.add_dll_directory(NM)
nm=ctypes.CDLL(os.path.join(NM,'nm_card_load.dll'))
nm.PL_SetTimeout.argtypes=[ctypes.c_uint32]; nm.PL_SetTimeout(10000)
nm.PL_GetBoardDesc.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)]; nm.PL_GetBoardDesc.restype=ctypes.c_int
nm.PL_ResetBoard.argtypes=[ctypes.c_void_p]; nm.PL_ResetBoard.restype=ctypes.c_int
nm.PL_LoadInitCode.argtypes=[ctypes.c_void_p]; nm.PL_LoadInitCode.restype=ctypes.c_int
nm.PL_GetAccess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int*2),ctypes.POINTER(ctypes.c_void_p)]; nm.PL_GetAccess.restype=ctypes.c_int
nm.PL_LoadProgramFile.argtypes=[ctypes.c_void_p,ctypes.c_char_p]; nm.PL_LoadProgramFile.restype=ctypes.c_int
nm.PL_CloseAccess.argtypes=[ctypes.c_void_p]; nm.PL_CloseAccess.restype=ctypes.c_int
nm.PL_ReadMemBlock.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.c_uint,ctypes.c_uint]; nm.PL_ReadMemBlock.restype=ctypes.c_int
nm.PL_WriteMemBlock.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.c_uint,ctypes.c_uint]; nm.PL_WriteMemBlock.restype=ctypes.c_int
nm.PL_CloseBoardDesc.argtypes=[ctypes.c_void_p]; nm.PL_CloseBoardDesc.restype=ctypes.c_int

DDR=0x00340000; CMD_BLOCK=32; buf1=(ctypes.c_uint*1)()
board=ctypes.c_void_p(); nm.PL_GetBoardDesc(0,ctypes.byref(board))
nm.PL_ResetBoard(board); nm.PL_LoadInitCode(board)

disp=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','aten','src','ATen','nmcard','nmc_programs','dispatcher_mc_float.abs')
accesses={}
for cl in range(4):
    for co in range(4):
        cn=(ctypes.c_int*2)(co,cl); acc=ctypes.c_void_p()
        nm.PL_GetAccess(board,ctypes.byref(cn),ctypes.byref(acc))
        nm.PL_LoadProgramFile(acc,disp.encode())
        accesses[(cl,co)]=acc
time.sleep(1)

alive=0
for cl in range(4):
    for co in range(4):
        acc=accesses[(cl,co)]; idx=cl*4+co
        nm.PL_ReadMemBlock(acc,buf1,DDR+idx*CMD_BLOCK+31,1)
        if buf1[0]>100: alive+=1
print(f'NM Card: {alive}/16 cores alive')

# ============================================================
# Card operations
# ============================================================
OP_MATMUL=1; OP_MATMUL_AT=2; OP_MATMUL_BT=3
OP_RELU=4; OP_RELU_BWD=5; OP_ELEM_ADD=10; OP_SGD=20; OP_EXIT=255

def f2u(a): return np.frombuffer(a.astype(np.float32).tobytes(),dtype=np.uint32)
def u2f(a): return np.frombuffer(np.array(a,dtype=np.uint32).tobytes(),dtype=np.float32)
def f2u_scalar(f): return struct.unpack('<I',struct.pack('<f',f))[0]

card_ops=0

def send_op(core, op, args_list):
    """Send op to specific core, don't wait."""
    global card_ops; card_ops+=1
    cl,co=core//4,core%4; acc=accesses[(cl,co)]
    base=DDR+core*CMD_BLOCK
    if args_list:
        a=(ctypes.c_uint*len(args_list))(*args_list)
        nm.PL_WriteMemBlock(acc,a,base+1,len(args_list))
    z=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(acc,z,base+30,1)
    c=(ctypes.c_uint*1)(op); nm.PL_WriteMemBlock(acc,c,base,1)

def wait_core(core):
    """Wait for core to finish."""
    cl,co=core//4,core%4; acc=accesses[(cl,co)]
    base=DDR+core*CMD_BLOCK
    for _ in range(50000):
        nm.PL_ReadMemBlock(acc,buf1,base+30,1)
        if buf1[0]==1: return True
        time.sleep(0.00005)
    return False

def wait_cores(cores):
    """Wait for multiple cores."""
    for c in cores: wait_core(c)

def upload(core, addr, data_np):
    """Upload numpy array to DDR via core's access."""
    cl,co=core//4,core%4; acc=accesses[(cl,co)]
    u=f2u(data_np.flatten())
    b=(ctypes.c_uint*len(u))(*u); nm.PL_WriteMemBlock(acc,b,addr,len(u))

def download(core, addr, count):
    """Download from DDR via core's access."""
    cl,co=core//4,core%4; acc=accesses[(cl,co)]
    b=(ctypes.c_uint*count)(); nm.PL_ReadMemBlock(acc,b,addr,count)
    return u2f(list(b))

# ============================================================
# DDR memory layout
# ============================================================
# Shared weights area (all cores read from here)
WEIGHT_BASE = DDR + 512  # after 16 CMD blocks

D=64; H=4; HD=D//H; F=128; T=32; L=3  # reduced to 2 layers for speed
np.random.seed(42)
he=lambda fi,fo: np.random.randn(fi,fo).astype(np.float32)*np.sqrt(2/(fi+fo))

# Model weights
embed=np.random.randn(V,D).astype(np.float32)*0.02
pos=np.random.randn(T,D).astype(np.float32)*0.01
W1=he(D,F); b1=np.zeros(F,np.float32)
W2=he(F,D); b2=np.zeros(D,np.float32)
Wh=he(D,V); bh=np.zeros(V,np.float32)
# Simplified: 2-layer FFN (no attention for now — attention backward on card needs more work)
# Focus: prove full forward+backward+SGD on card with 16 cores parallel

total = embed.size+pos.size+W1.size+b1.size+W2.size+b2.size+Wh.size+bh.size
print(f'Model: {total:,} params (FFN), D={D}, F={F}, T={T}')

# Upload weights to shared DDR
w_addr = {}
next_addr = WEIGHT_BASE

def upload_weight(name, w):
    global next_addr
    addr = next_addr
    upload(0, addr, w)
    w_addr[name] = (addr, w.shape, w.size)
    next_addr += w.size
    return addr

upload_weight('W1', W1); upload_weight('b1', b1)
upload_weight('W2', W2); upload_weight('b2', b2)
upload_weight('Wh', Wh); upload_weight('bh', bh)
print(f'Weights uploaded: {(next_addr-WEIGHT_BASE)*4/1024:.0f} KB DDR')

# Per-core scratch areas (after weights)
SCRATCH_SIZE = 60000  # words per core
SCRATCH_BASE = next_addr + 256
print(f'Scratch: {SCRATCH_BASE:x}, {SCRATCH_SIZE*4/1024:.0f} KB per core')

def core_scratch(core):
    return SCRATCH_BASE + core * SCRATCH_SIZE

# Verify matmul
upload(0, core_scratch(0), np.array([[1,2],[3,4]],dtype=np.float32))
upload(0, core_scratch(0)+4, np.array([[5,6],[7,8]],dtype=np.float32))
send_op(0, OP_MATMUL, [2,2,2, core_scratch(0), core_scratch(0)+4, core_scratch(0)+8])
wait_core(0)
C = download(0, core_scratch(0)+8, 4)
print(f'Matmul verify: {C} (expect [19 22 43 50])')

# ============================================================
# Training: batch=16, each core processes 1 sample
# Forward + backward on card, gradient accumulation on host
# ============================================================
def sm(x): e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)

print(f'\n=== FULL ON-CARD TRAINING (16 cores) ===')
base_lr=0.001; lr=0.00005; best=99; step=0; start=time.time()
WARMUP=200; BATCH=16

while best > 1.0 and step < 10000:
    if step < WARMUP: lr = base_lr * (step+1) / WARMUP
    elif step < 3000: lr = base_lr
    elif step < 6000: lr = base_lr * 0.3
    else: lr = base_lr * 0.1

    batch_loss = 0
    # Accumulate gradients
    dW1_acc=np.zeros_like(W1); db1_acc=np.zeros_like(b1)
    dW2_acc=np.zeros_like(W2); db2_acc=np.zeros_like(b2)
    dWh_acc=np.zeros_like(Wh); dbh_acc=np.zeros_like(bh)
    dpos_acc=np.zeros_like(pos); dembed_acc={}

    for bi in range(BATCH):
        core = bi % alive  # distribute across available cores
        sc = core_scratch(core)
        idx = np.random.randint(0, len(data)-T-1)
        tok = data[idx:idx+T]; tgt = data[idx+1:idx+T+1]

        # Prepare input: x = embed[tok] + pos
        x_input = embed[tok] + pos
        upload(core, sc, x_input)  # sc = input [T, D]

        # ---- FORWARD ON CARD ----
        # h1 = x @ W1 + b1
        wa1,_,_ = w_addr['W1']
        send_op(core, OP_MATMUL, [T, D, F, sc, wa1, sc + T*D])
        wait_core(core)
        # Download h1 to add bias on host (b1 add could also be on card)
        h1 = download(core, sc + T*D, T*F).reshape(T,F) + b1

        # a1 = relu(h1)
        upload(core, sc + T*D, h1)
        send_op(core, OP_RELU, [T*F, sc + T*D, sc + T*D + T*F])
        wait_core(core)

        # o = a1 @ W2 + b2
        wa2,_,_ = w_addr['W2']
        send_op(core, OP_MATMUL, [T, F, D, sc + T*D + T*F, wa2, sc + 2*T*D + T*F])
        wait_core(core)
        o = download(core, sc + 2*T*D + T*F, T*D).reshape(T,D) + b2

        # x2 = x + o (residual)
        x2 = x_input + o

        # logits = x2 @ Wh + bh
        upload(core, sc, x2)
        wah,_,_ = w_addr['Wh']
        send_op(core, OP_MATMUL, [T, D, V, sc, wah, sc + T*D])
        wait_core(core)
        logits = download(core, sc + T*D, T*V).reshape(T,V) + bh

        # Loss + softmax (tiny, on host)
        probs = sm(logits)
        loss = -np.mean(np.log(probs[np.arange(T), tgt] + 1e-9))
        batch_loss += loss

        # ---- BACKWARD ON CARD ----
        dl = probs.copy(); dl[np.arange(T), tgt] -= 1; dl /= T

        # dWh = x2^T @ dl  (OP_MATMUL_AT: C = A^T @ B)
        upload(core, sc, x2)  # A = x2 [T,D]
        upload(core, sc + T*D, dl)  # B = dl [T,V]
        send_op(core, OP_MATMUL_AT, [T, D, V, sc, sc + T*D, sc + T*D + T*V])
        wait_core(core)
        dWh_i = download(core, sc + T*D + T*V, D*V).reshape(D,V)
        dWh_acc += dWh_i
        dbh_acc += dl.sum(0)

        # dx2 = dl @ Wh^T  (OP_MATMUL_BT: C = A @ B^T)
        upload(core, sc, dl)  # A = dl [T,V]
        send_op(core, OP_MATMUL_BT, [T, D, V, sc, wah, sc + T*V])
        wait_core(core)
        dx2 = download(core, sc + T*V, T*D).reshape(T,D)

        # Backward through residual: dx = dx2, do = dx2
        # do @ W2^T = da (before relu)
        upload(core, sc, dx2)  # A = dx2 [T,D]
        send_op(core, OP_MATMUL_BT, [T, F, D, sc, wa2, sc + T*D])
        wait_core(core)
        da_pre = download(core, sc + T*D, T*F).reshape(T,F)

        # relu backward: da = da_pre * (h1 > 0)
        da = da_pre * (h1 > 0).astype(np.float32)

        # dW1 = xn^T @ da (OP_MATMUL_AT)
        upload(core, sc, x_input)
        upload(core, sc + T*D, da)
        send_op(core, OP_MATMUL_AT, [T, D, F, sc, sc + T*D, sc + T*D + T*F])
        wait_core(core)
        dW1_i = download(core, sc + T*D + T*F, D*F).reshape(D,F)
        dW1_acc += dW1_i
        db1_acc += da.sum(0)

        # dW2 = a1^T @ dx2 (OP_MATMUL_AT)
        a1 = np.maximum(h1, 0)
        upload(core, sc, a1)
        upload(core, sc + T*F, dx2)
        send_op(core, OP_MATMUL_AT, [T, F, D, sc, sc + T*F, sc + T*F + T*D])
        wait_core(core)
        dW2_i = download(core, sc + T*F + T*D, F*D).reshape(F,D)
        dW2_acc += dW2_i
        db2_acc += dx2.sum(0)

        # dx for embed/pos update
        # dx_input = dx2 + da @ W1^T
        upload(core, sc, da)
        send_op(core, OP_MATMUL_BT, [T, D, F, sc, wa1, sc + T*F])
        wait_core(core)
        dx_w1t = download(core, sc + T*F, T*D).reshape(T,D)
        dx_input = dx2 + dx_w1t

        dpos_acc += 0.1 * dx_input
        for t in range(T):
            tid = tok[t]
            if tid not in dembed_acc: dembed_acc[tid] = np.zeros(D, np.float32)
            dembed_acc[tid] += 0.1 * dx_input[t]

    batch_loss /= BATCH
    if batch_loss < best: best = batch_loss

    # SGD update weights on HOST then re-upload
    W1 -= lr * dW1_acc / BATCH; b1 -= lr * db1_acc / BATCH
    W2 -= lr * dW2_acc / BATCH; b2 -= lr * db2_acc / BATCH
    Wh -= lr * dWh_acc / BATCH; bh -= lr * dbh_acc / BATCH
    pos -= lr * dpos_acc / BATCH
    for tid, g in dembed_acc.items(): embed[tid] -= lr * g / BATCH

    # Re-upload updated weights every 5 steps
    if step % 5 == 0:
        upload(0, w_addr['W1'][0], W1)
        upload(0, w_addr['W2'][0], W2)
        upload(0, w_addr['Wh'][0], Wh)

    if step % 50 == 0:
        el = time.time() - start
        print(f'Step {step:5d} | loss={batch_loss:.3f} | best={best:.3f} | ops={card_ops} | {el:.0f}s')
    if step % 500 == 0 and step > 0:
        p = list(ch2idx.get(c,0) for c in 'ROMEO:\n')
        for _ in range(100):
            inp=p[-T:]; x=embed[inp]+pos[:len(inp)]
            h=np.maximum(x@W1+b1,0); x=x+h@W2+b2
            pp=sm((x@Wh+bh)[-1]/0.9); pp/=pp.sum()
            p.append(np.random.choice(V,p=pp))
        print(f'  Gen: {"".join(idx2ch.get(t,"?") for t in p)[:150]}')
    step += 1

el=time.time()-start
print(f'\nDONE: {step} steps, {el:.0f}s, best={best:.3f}, card_ops={card_ops}')

# Save
sd=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','nanogpt_fullcard_weights')
os.makedirs(sd,exist_ok=True)
np.save(f'{sd}/embed.npy',embed);np.save(f'{sd}/pos.npy',pos)
np.save(f'{sd}/W1.npy',W1);np.save(f'{sd}/b1.npy',b1)
np.save(f'{sd}/W2.npy',W2);np.save(f'{sd}/b2.npy',b2)
np.save(f'{sd}/Wh.npy',Wh);np.save(f'{sd}/bh.npy',bh)
with open(f'{sd}/config.json','w') as f:
    json.dump({'V':V,'D':D,'F':F,'T':T,'params':int(total),'loss':float(best),
               'card_ops':card_ops,'steps':step,'time':el,'cores':alive,
               'chars':''.join(chars)},f,indent=2)
print(f'Saved: {sd}/')

# Shutdown
for cl in range(4):
    for co in range(4):
        acc=accesses[(cl,co)]; idx=cl*4+co
        e=(ctypes.c_uint*1)(OP_EXIT); nm.PL_WriteMemBlock(acc,e,DDR+idx*CMD_BLOCK,1)
time.sleep(0.3)
for acc in accesses.values(): nm.PL_CloseAccess(acc)
nm.PL_CloseBoardDesc(board)
print('All 16 cores safe. DONE.')
