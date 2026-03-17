"""
109K transformer — TRUE PARALLEL 16-core training on NM Card Mini.
Each forward/backward op fires on ALL 16 cores simultaneously.
Pattern: upload_all → send_all → wait_all → download_all
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
nm.PL_SetTimeout.argtypes=[ctypes.c_uint32]; nm.PL_SetTimeout(15000)
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
NCORES=16
accesses={}
for cl in range(4):
    for co in range(4):
        cn=(ctypes.c_int*2)(co,cl); acc=ctypes.c_void_p()
        nm.PL_GetAccess(board,ctypes.byref(cn),ctypes.byref(acc))
        idx=cl*4+co
        ci=(ctypes.c_uint*1)(idx); nm.PL_WriteMemBlock(acc,ci,DDR+29,1)
        nm.PL_LoadProgramFile(acc,disp.encode())
        accesses[idx]=acc
time.sleep(1.5)

alive=0
for i in range(NCORES):
    nm.PL_ReadMemBlock(accesses[i],buf1,DDR+i*CMD_BLOCK+31,1)
    if buf1[0]>100: alive+=1
print(f'NM Card: {alive}/16 cores alive')
if alive<16:
    print('WARNING: not all cores alive!')

# ============================================================
# Helpers
# ============================================================
OP_MATMUL=1; OP_MATMUL_AT=2; OP_MATMUL_BT=3
OP_RELU=4; OP_RELU_BWD=5; OP_ELEM_ADD=10; OP_SGD=20; OP_EXIT=255

def f2u(a): return np.frombuffer(a.astype(np.float32).tobytes(),dtype=np.uint32)
def u2f(a): return np.frombuffer(np.array(a,dtype=np.uint32).tobytes(),dtype=np.float32)
def f2u_scalar(f): return struct.unpack('<I',struct.pack('<f',f))[0]

card_ops=0

def upload(core, addr, data_np):
    u=f2u(data_np.flatten())
    b=(ctypes.c_uint*len(u))(*u); nm.PL_WriteMemBlock(accesses[core],b,addr,len(u))

def download(core, addr, count):
    b=(ctypes.c_uint*count)(); nm.PL_ReadMemBlock(accesses[core],b,addr,count)
    return u2f(list(b))

def send_op(core, op, args_list):
    """Send op: args first, status clear, opcode LAST (triggers dispatcher)."""
    global card_ops; card_ops+=1
    acc=accesses[core]; base=DDR+core*CMD_BLOCK
    # 1. Write args (words 1..N) — must arrive before opcode
    if args_list:
        a=(ctypes.c_uint*len(args_list))(*args_list)
        nm.PL_WriteMemBlock(acc,a,base+1,len(args_list))
    # 2. Clear status (word 30)
    z=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(acc,z,base+30,1)
    # 3. Write opcode LAST (word 0) — this triggers the dispatcher
    c=(ctypes.c_uint*1)(op); nm.PL_WriteMemBlock(acc,c,base,1)

def wait_core(core, timeout_ms=8000):
    acc=accesses[core]; base=DDR+core*CMD_BLOCK
    t0=time.time()
    # First wait for core to acknowledge (status goes to 0 = busy)
    while (time.time()-t0)*1000 < 500:
        nm.PL_ReadMemBlock(acc,buf1,base+30,1)
        if buf1[0]==0: break  # core started processing
        nm.PL_ReadMemBlock(acc,buf1,base,1)
        if buf1[0]==0: break  # cmd already consumed (NOP), check status
        time.sleep(0.0001)
    # Now wait for completion (status goes to 1 = done)
    while (time.time()-t0)*1000 < timeout_ms:
        nm.PL_ReadMemBlock(acc,buf1,base+30,1)
        if buf1[0]==1: return True
        time.sleep(0.0002)
    nm.PL_ReadMemBlock(acc,buf1,base,1); cmd=buf1[0]
    nm.PL_ReadMemBlock(acc,buf1,base+30,1); st=buf1[0]
    nm.PL_ReadMemBlock(acc,buf1,base+31,1); wd=buf1[0]
    print(f'TIMEOUT core[{core}] cmd={cmd} st={st} wd={wd}',flush=True)
    return False

def wait_all(cores):
    """Wait for all cores to complete. Returns False if any timed out."""
    ok=True
    for c in cores:
        if not wait_core(c): ok=False
    return ok

# ============================================================
# DDR layout: shared weights + per-core scratch
# ============================================================
WEIGHT_BASE = DDR + 512

D=64; H=4; HD=D//H; F=128; T=32; L=3
np.random.seed(42)
he=lambda fi,fo: np.random.randn(fi,fo).astype(np.float32)*np.sqrt(2/(fi+fo))

embed=np.random.randn(V,D).astype(np.float32)*0.02
pos=np.random.randn(T,D).astype(np.float32)*0.01
layers=[]
for _ in range(L):
    layers.append({'g1':np.ones(D,np.float32),
        'Wq':he(D,D),'Wk':he(D,D),'Wv':he(D,D),'Wo':he(D,D),
        'g2':np.ones(D,np.float32),
        'W1':he(D,F),'b1':np.zeros(F,np.float32),
        'W2':he(F,D),'b2':np.zeros(D,np.float32)})
gf=np.ones(D,np.float32); Wh=he(D,V); bh=np.zeros(V,np.float32)
total=embed.size+pos.size+gf.size+Wh.size+bh.size+sum(sum(v.size for v in l.values()) for l in layers)
print(f'Model: {total:,} params, D={D}, H={H}, F={F}, T={T}, L={L}')

# Upload shared weights to DDR
w_addr={}
next_addr=WEIGHT_BASE
def upload_weight(name, w):
    global next_addr
    addr=next_addr; upload(0, addr, w)
    w_addr[name]=(addr,w.shape,w.size)
    next_addr+=w.size; return addr

for li in range(L):
    l=layers[li]
    upload_weight(f'Wq{li}',l['Wq']); upload_weight(f'Wk{li}',l['Wk'])
    upload_weight(f'Wv{li}',l['Wv']); upload_weight(f'Wo{li}',l['Wo'])
    upload_weight(f'W1_{li}',l['W1']); upload_weight(f'W2_{li}',l['W2'])
upload_weight('Wh',Wh)
print(f'Weights uploaded: {(next_addr-WEIGHT_BASE)*4/1024:.1f} KB DDR')

# Per-core scratch areas
SCRATCH_SIZE = 60000  # words per core
SCRATCH_BASE = next_addr + 256
for i in range(NCORES):
    print(f'  Core {i:2d} scratch: 0x{SCRATCH_BASE+i*SCRATCH_SIZE:08x}')
def sc(core): return SCRATCH_BASE + core * SCRATCH_SIZE

# Verify matmul on core 0
upload(0, sc(0), np.array([[1,2],[3,4]],dtype=np.float32))
upload(0, sc(0)+4, np.array([[5,6],[7,8]],dtype=np.float32))
send_op(0, OP_MATMUL, [2,2,2, sc(0), sc(0)+4, sc(0)+8])
wait_core(0)
C=download(0, sc(0)+8, 4)
print(f'Matmul verify core 0: {C} (expect [19 22 43 50])')

# Verify PARALLEL matmul on ALL 16 cores
print('Testing parallel matmul on all 16 cores...')
for i in range(NCORES):
    upload(i, sc(i), np.array([[1+i,2],[3,4]],dtype=np.float32))
    upload(i, sc(i)+4, np.array([[5,6],[7,8]],dtype=np.float32))
# Send to ALL cores first
for i in range(NCORES):
    send_op(i, OP_MATMUL, [2,2,2, sc(i), sc(i)+4, sc(i)+8])
# Wait for ALL cores
if wait_all(range(NCORES)):
    for i in range(NCORES):
        r=download(i, sc(i)+8, 4)
        expected=[5*(1+i)+14, 6*(1+i)+16, 43, 50]
        ok='OK' if abs(r[0]-expected[0])<0.01 else 'FAIL'
        if i<4 or i>=14: print(f'  Core {i:2d}: {r} {ok}')
    print(f'  ... all 16 cores tested')
else:
    print('PARALLEL TEST FAILED — some cores timed out')

# ============================================================
# RMSNorm, softmax helpers (host-side, these are tiny)
# ============================================================
def sm(x): e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def rn(x,g): r=np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6); return x/r*g, r
def rn_bwd(dy,x,g,r):
    D_=x.shape[-1]; xn=x/r; dy_g=dy*g
    return (dy_g-xn*np.sum(dy_g*xn,axis=-1,keepdims=True)/D_)/r

# ============================================================
# PARALLEL TRAINING
# Each step: 16 samples in parallel across 16 cores
# For each op: upload_all → send_all → wait_all → download_all
# ============================================================
print(f'\n=== TRUE PARALLEL 16-CORE TRAINING ===')
base_lr=0.0006; lr=0.00001; best=99; step=0; start=time.time(); WARMUP=300; BATCH=NCORES

# Pre-allocate per-core data arrays
core_data = [None]*NCORES  # will store per-sample state for backward

while best > 1.0 and step < 15000:
    if step < WARMUP: lr = base_lr * (step+1) / WARMUP
    elif step < 5000: lr = base_lr
    elif step < 10000: lr = base_lr * 0.3
    else: lr = base_lr * 0.1

    # Prepare batch: 16 random samples
    toks=[None]*NCORES; tgts=[None]*NCORES; x_inputs=[None]*NCORES
    for i in range(NCORES):
        idx=np.random.randint(0,len(data)-T-1)
        toks[i]=data[idx:idx+T]; tgts[i]=data[idx+1:idx+T+1]
        x_inputs[i]=embed[toks[i]]+pos

    # ============ FORWARD ============
    # For each layer: rmsnorm(host) → QKV matmul(card) → attention(host) → proj(card) → residual
    #                 rmsnorm(host) → FFN matmul(card) → relu(host) → FFN matmul(card) → residual
    #
    # Card ops per layer: 3 matmuls (QKV) + 1 matmul (Wo) + 1 matmul (W1) + 1 matmul (W2) = 6
    # Card ops total: 6*L + 1 (Wh) = 19 matmuls per forward, x16 cores = 304 card ops

    x = list(x_inputs)  # current activation per core
    caches = [[] for _ in range(NCORES)]

    for li in range(L):
        l=layers[li]
        # RMSNorm on host (cheap, [T,D] per sample)
        xns=[None]*NCORES; r1s=[None]*NCORES
        for i in range(NCORES):
            xns[i],r1s[i]=rn(x[i],l['g1'])

        # === QKV matmul: xn @ Wq, xn @ Wk, xn @ Wv ===
        # Upload xn to each core, send matmul with shared Wq
        wq_a=w_addr[f'Wq{li}'][0]; wk_a=w_addr[f'Wk{li}'][0]; wv_a=w_addr[f'Wv{li}'][0]

        # Q = xn @ Wq  (all 16 cores)
        for i in range(NCORES):
            upload(i, sc(i), xns[i])  # [T,D] at scratch start
        for i in range(NCORES):
            send_op(i, OP_MATMUL, [T, D, D, sc(i), wq_a, sc(i)+T*D])
        wait_all(range(NCORES))

        # K = xn @ Wk  (xn already at sc(i))
        for i in range(NCORES):
            send_op(i, OP_MATMUL, [T, D, D, sc(i), wk_a, sc(i)+T*D+T*D])
        wait_all(range(NCORES))

        # V = xn @ Wv
        for i in range(NCORES):
            send_op(i, OP_MATMUL, [T, D, D, sc(i), wv_a, sc(i)+2*T*D+T*D])
        wait_all(range(NCORES))

        # Download QKV
        Qs=[None]*NCORES; Ks=[None]*NCORES; Vs=[None]*NCORES
        for i in range(NCORES):
            Qs[i]=download(i, sc(i)+T*D, T*D).reshape(T,D)
            Ks[i]=download(i, sc(i)+2*T*D, T*D).reshape(T,D)
            Vs[i]=download(i, sc(i)+3*T*D, T*D).reshape(T,D)

        # Attention on host (causal, multi-head)
        aos=[None]*NCORES; ats=[None]*NCORES
        for i in range(NCORES):
            Q=Qs[i].reshape(T,H,HD).transpose(1,0,2)
            K_=Ks[i].reshape(T,H,HD).transpose(1,0,2)
            V_=Vs[i].reshape(T,H,HD).transpose(1,0,2)
            sc_=np.matmul(Q,K_.transpose(0,2,1))/np.sqrt(HD)+np.triu(np.full((T,T),-1e9),k=1)
            ats[i]=sm(sc_); aos[i]=np.matmul(ats[i],V_).transpose(1,0,2).reshape(T,D)
            Qs[i]=Q; Ks[i]=K_; Vs[i]=V_  # keep reshaped for backward

        # Projection: ao @ Wo
        wo_a=w_addr[f'Wo{li}'][0]
        for i in range(NCORES):
            upload(i, sc(i), aos[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL, [T, D, D, sc(i), wo_a, sc(i)+T*D])
        wait_all(range(NCORES))

        prs=[None]*NCORES; xrs=[None]*NCORES
        for i in range(NCORES):
            prs[i]=download(i, sc(i)+T*D, T*D).reshape(T,D)
            xrs[i]=x[i]+prs[i]  # residual

        # RMSNorm 2
        xn2s=[None]*NCORES; r2s=[None]*NCORES
        for i in range(NCORES):
            xn2s[i],r2s[i]=rn(xrs[i],l['g2'])

        # FFN: h = xn2 @ W1 + b1
        w1_a=w_addr[f'W1_{li}'][0]; w2_a=w_addr[f'W2_{li}'][0]
        for i in range(NCORES):
            upload(i, sc(i), xn2s[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL, [T, D, F, sc(i), w1_a, sc(i)+T*D])
        wait_all(range(NCORES))

        hs=[None]*NCORES; als=[None]*NCORES
        for i in range(NCORES):
            hs[i]=download(i, sc(i)+T*D, T*F).reshape(T,F)+l['b1']
            als[i]=np.maximum(hs[i],0)  # relu

        # o = relu(h) @ W2 + b2
        for i in range(NCORES):
            upload(i, sc(i), als[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL, [T, F, D, sc(i), w2_a, sc(i)+T*F])
        wait_all(range(NCORES))

        for i in range(NCORES):
            o=download(i, sc(i)+T*F, T*D).reshape(T,D)+l['b2']
            caches[i].append((x[i],xns[i],r1s[i],Qs[i],Ks[i],Vs[i],ats[i],aos[i],
                             xrs[i],xn2s[i],r2s[i],hs[i],als[i]))
            x[i]=xrs[i]+o  # residual

    # Final RMSNorm + logits
    xfs=[None]*NCORES; rfs=[None]*NCORES
    for i in range(NCORES):
        xfs[i],rfs[i]=rn(x[i],gf)

    # logits = xf @ Wh + bh  (card)
    wh_a=w_addr['Wh'][0]
    for i in range(NCORES):
        upload(i, sc(i), xfs[i])
    for i in range(NCORES):
        send_op(i, OP_MATMUL, [T, D, V, sc(i), wh_a, sc(i)+T*D])
    wait_all(range(NCORES))

    # Loss
    batch_loss=0
    dls=[None]*NCORES
    for i in range(NCORES):
        logits=download(i, sc(i)+T*D, T*V).reshape(T,V)+bh
        probs=sm(logits)
        loss=-np.mean(np.log(probs[np.arange(T),tgts[i]]+1e-9))
        batch_loss+=loss
        dl=probs.copy(); dl[np.arange(T),tgts[i]]-=1; dl/=T
        dls[i]=dl

    # ============ BACKWARD ============
    # dWh = xf^T @ dl  (MATMUL_AT on card)
    for i in range(NCORES):
        upload(i, sc(i), xfs[i])
        upload(i, sc(i)+T*D, dls[i])
    for i in range(NCORES):
        send_op(i, OP_MATMUL_AT, [T, D, V, sc(i), sc(i)+T*D, sc(i)+T*D+T*V])
    wait_all(range(NCORES))

    acc_dWh=np.zeros_like(Wh); acc_dbh=np.zeros_like(bh)
    for i in range(NCORES):
        acc_dWh+=download(i, sc(i)+T*D+T*V, D*V).reshape(D,V)
        acc_dbh+=dls[i].sum(0)

    # dx = dl @ Wh^T  (MATMUL_BT on card)
    for i in range(NCORES):
        upload(i, sc(i), dls[i])
    for i in range(NCORES):
        send_op(i, OP_MATMUL_BT, [T, D, V, sc(i), wh_a, sc(i)+T*V])
    wait_all(range(NCORES))

    dxs=[None]*NCORES
    for i in range(NCORES):
        dx_raw=download(i, sc(i)+T*V, T*D).reshape(T,D)
        dxs[i]=rn_bwd(dx_raw, x[i], gf, rfs[i])

    # Layer backward (reverse order)
    layer_grads=[{k:np.zeros_like(v) for k,v in l.items() if k not in ('g1','g2')} for l in layers]

    for li in range(L-1,-1,-1):
        l=layers[li]
        wq_a=w_addr[f'Wq{li}'][0]; wk_a=w_addr[f'Wk{li}'][0]
        wv_a=w_addr[f'Wv{li}'][0]; wo_a=w_addr[f'Wo{li}'][0]
        w1_a=w_addr[f'W1_{li}'][0]; w2_a=w_addr[f'W2_{li}'][0]

        # Unpack caches
        x_ins=[None]*NCORES; xns_b=[None]*NCORES; r1s_b=[None]*NCORES
        Qs_b=[None]*NCORES; Ks_b=[None]*NCORES; Vs_b=[None]*NCORES
        ats_b=[None]*NCORES; aos_b=[None]*NCORES; xrs_b=[None]*NCORES
        xn2s_b=[None]*NCORES; r2s_b=[None]*NCORES; hs_b=[None]*NCORES; als_b=[None]*NCORES

        for i in range(NCORES):
            (x_ins[i],xns_b[i],r1s_b[i],Qs_b[i],Ks_b[i],Vs_b[i],ats_b[i],
             aos_b[i],xrs_b[i],xn2s_b[i],r2s_b[i],hs_b[i],als_b[i])=caches[i][li]

        # --- FFN backward ---
        # dW2 = relu(h)^T @ dx  (MATMUL_AT)
        for i in range(NCORES):
            upload(i, sc(i), als_b[i])          # a [T,F]
            upload(i, sc(i)+T*F, dxs[i])        # dx [T,D]
        for i in range(NCORES):
            send_op(i, OP_MATMUL_AT, [T, F, D, sc(i), sc(i)+T*F, sc(i)+T*F+T*D])
        wait_all(range(NCORES))

        for i in range(NCORES):
            layer_grads[li]['W2']+=download(i, sc(i)+T*F+T*D, F*D).reshape(F,D)
            layer_grads[li]['b2']+=dxs[i].sum(0)

        # da = dx @ W2^T * (h>0)  (MATMUL_BT)
        for i in range(NCORES):
            upload(i, sc(i), dxs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_BT, [T, F, D, sc(i), w2_a, sc(i)+T*D])
        wait_all(range(NCORES))

        das=[None]*NCORES
        for i in range(NCORES):
            da_pre=download(i, sc(i)+T*D, T*F).reshape(T,F)
            das[i]=da_pre*(hs_b[i]>0).astype(np.float32)

        # dW1 = xn2^T @ da  (MATMUL_AT)
        for i in range(NCORES):
            upload(i, sc(i), xn2s_b[i])
            upload(i, sc(i)+T*D, das[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_AT, [T, D, F, sc(i), sc(i)+T*D, sc(i)+T*D+T*F])
        wait_all(range(NCORES))

        for i in range(NCORES):
            layer_grads[li]['W1']+=download(i, sc(i)+T*D+T*F, D*F).reshape(D,F)
            layer_grads[li]['b1']+=das[i].sum(0)

        # dxn2 = da @ W1^T  (MATMUL_BT)
        for i in range(NCORES):
            upload(i, sc(i), das[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_BT, [T, D, F, sc(i), w1_a, sc(i)+T*F])
        wait_all(range(NCORES))

        dxn2s=[None]*NCORES
        for i in range(NCORES):
            dxn2s[i]=download(i, sc(i)+T*F, T*D).reshape(T,D)

        # dx through RMSNorm2 + residual
        dx_rs=[None]*NCORES
        for i in range(NCORES):
            dx_rs[i]=dxs[i]+rn_bwd(dxn2s[i],xrs_b[i],l['g2'],r2s_b[i])

        # --- Attention backward (host) ---
        # dWo = ao^T @ dx_r  (MATMUL_AT on card)
        for i in range(NCORES):
            upload(i, sc(i), aos_b[i])
            upload(i, sc(i)+T*D, dx_rs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_AT, [T, D, D, sc(i), sc(i)+T*D, sc(i)+2*T*D])
        wait_all(range(NCORES))

        for i in range(NCORES):
            layer_grads[li]['Wo']+=download(i, sc(i)+2*T*D, D*D).reshape(D,D)

        # dao = dx_r @ Wo^T  (MATMUL_BT)
        for i in range(NCORES):
            upload(i, sc(i), dx_rs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_BT, [T, D, D, sc(i), wo_a, sc(i)+T*D])
        wait_all(range(NCORES))

        # Attention backward (host — softmax derivative is cheap)
        dQs=[None]*NCORES; dKs=[None]*NCORES; dVs=[None]*NCORES
        for i in range(NCORES):
            dao=download(i, sc(i)+T*D, T*D).reshape(T,D)
            dao_h=dao.reshape(T,H,HD).transpose(1,0,2)
            dV_=np.matmul(ats_b[i].transpose(0,2,1),dao_h)
            dat=np.matmul(dao_h,Vs_b[i].transpose(0,2,1))
            ds=ats_b[i]*(dat-np.sum(dat*ats_b[i],-1,keepdims=True))/np.sqrt(HD)
            dQ=np.matmul(ds,Ks_b[i]).transpose(1,0,2).reshape(T,D)
            dK=np.matmul(ds.transpose(0,2,1),Qs_b[i]).transpose(1,0,2).reshape(T,D)
            dV_=dV_.transpose(1,0,2).reshape(T,D)
            dQs[i]=dQ; dKs[i]=dK; dVs[i]=dV_

        # dWq = xn^T @ dQ, dWk = xn^T @ dK, dWv = xn^T @ dV  (MATMUL_AT on card)
        # Do all 3 sequentially (same xn, different grads)
        # dWq
        for i in range(NCORES):
            upload(i, sc(i), xns_b[i])
            upload(i, sc(i)+T*D, dQs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_AT, [T, D, D, sc(i), sc(i)+T*D, sc(i)+2*T*D])
        wait_all(range(NCORES))
        for i in range(NCORES):
            layer_grads[li]['Wq']+=download(i, sc(i)+2*T*D, D*D).reshape(D,D)

        # dWk
        for i in range(NCORES):
            upload(i, sc(i)+T*D, dKs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_AT, [T, D, D, sc(i), sc(i)+T*D, sc(i)+2*T*D])
        wait_all(range(NCORES))
        for i in range(NCORES):
            layer_grads[li]['Wk']+=download(i, sc(i)+2*T*D, D*D).reshape(D,D)

        # dWv
        for i in range(NCORES):
            upload(i, sc(i)+T*D, dVs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_AT, [T, D, D, sc(i), sc(i)+T*D, sc(i)+2*T*D])
        wait_all(range(NCORES))
        for i in range(NCORES):
            layer_grads[li]['Wv']+=download(i, sc(i)+2*T*D, D*D).reshape(D,D)

        # dx through QKV projection + RMSNorm1
        # dx_qkv = dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T (3 MATMUL_BT on card)
        # dQ @ Wq^T
        for i in range(NCORES):
            upload(i, sc(i), dQs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_BT, [T, D, D, sc(i), wq_a, sc(i)+T*D])
        wait_all(range(NCORES))
        dx_accs=[download(i, sc(i)+T*D, T*D).reshape(T,D) for i in range(NCORES)]

        # + dK @ Wk^T
        for i in range(NCORES):
            upload(i, sc(i), dKs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_BT, [T, D, D, sc(i), wk_a, sc(i)+T*D])
        wait_all(range(NCORES))
        for i in range(NCORES):
            dx_accs[i]+=download(i, sc(i)+T*D, T*D).reshape(T,D)

        # + dV @ Wv^T
        for i in range(NCORES):
            upload(i, sc(i), dVs[i])
        for i in range(NCORES):
            send_op(i, OP_MATMUL_BT, [T, D, D, sc(i), wv_a, sc(i)+T*D])
        wait_all(range(NCORES))
        for i in range(NCORES):
            dx_accs[i]+=download(i, sc(i)+T*D, T*D).reshape(T,D)
            dxs[i]=dx_rs[i]+rn_bwd(dx_accs[i],x_ins[i],l['g1'],r1s_b[i])

    # Accumulate embed/pos grads
    acc_dpos=np.zeros_like(pos); acc_dembed={}
    for i in range(NCORES):
        acc_dpos+=0.1*dxs[i]
        for t in range(T):
            tid=toks[i][t]
            if tid not in acc_dembed: acc_dembed[tid]=np.zeros(D,np.float32)
            acc_dembed[tid]+=0.1*dxs[i][t]

    batch_loss/=BATCH
    if batch_loss<best: best=batch_loss

    # SGD update
    Wh-=lr*acc_dWh/BATCH; bh-=lr*acc_dbh/BATCH; pos-=lr*acc_dpos/BATCH
    for tid,g in acc_dembed.items(): embed[tid]-=lr*g/BATCH
    for li,l in enumerate(layers):
        for k in layer_grads[li]: l[k]-=lr*layer_grads[li][k]/BATCH

    # Re-upload weights every step (they changed)
    for li in range(L):
        l=layers[li]
        upload(0, w_addr[f'Wq{li}'][0], l['Wq'])
        upload(0, w_addr[f'Wk{li}'][0], l['Wk'])
        upload(0, w_addr[f'Wv{li}'][0], l['Wv'])
        upload(0, w_addr[f'Wo{li}'][0], l['Wo'])
        upload(0, w_addr[f'W1_{li}'][0], l['W1'])
        upload(0, w_addr[f'W2_{li}'][0], l['W2'])
    upload(0, w_addr['Wh'][0], Wh)

    if step%10==0:
        el=time.time()-start
        ops_per_step=card_ops/(step+1)
        print(f'Step {step:5d} | loss={batch_loss:.3f} | best={best:.3f} | ops={card_ops} ({ops_per_step:.0f}/step) | {el:.0f}s | lr={lr:.6f}')
    if step%200==0 and step>0:
        p=list(ch2idx.get(c,0) for c in 'ROMEO:\n')
        for _ in range(100):
            inp=p[-T:]; xg=embed[inp]+pos[:len(inp)]
            for l in layers:
                xn=xg/np.sqrt(np.mean(xg**2,-1,keepdims=True)+1e-6)*l['g1']
                Qg=(xn@l['Wq']).reshape(-1,H,HD).transpose(1,0,2)
                Kg=(xn@l['Wk']).reshape(-1,H,HD).transpose(1,0,2)
                Vg=(xn@l['Wv']).reshape(-1,H,HD).transpose(1,0,2)
                sc_=np.matmul(Qg,Kg.transpose(0,2,1))/np.sqrt(HD)+np.triu(np.full((Qg.shape[1],Qg.shape[1]),-1e9),k=1)
                xg=xg+(sm(sc_)@Vg).transpose(1,0,2).reshape(-1,D)@l['Wo']
                xn2=xg/np.sqrt(np.mean(xg**2,-1,keepdims=True)+1e-6)*l['g2']
                xg=xg+np.maximum(xn2@l['W1']+l['b1'],0)@l['W2']+l['b2']
            xf=xg/np.sqrt(np.mean(xg**2,-1,keepdims=True)+1e-6)*gf
            pp=sm((xf@Wh+bh)[-1]/0.9);pp/=pp.sum()
            p.append(np.random.choice(V,p=pp))
        print(f'  Gen: {"".join(idx2ch.get(t,"?") for t in p)[:150]}')
    step+=1

el=time.time()-start
print(f'\nDONE: {step} steps, {el:.0f}s, best={best:.3f}, card_ops={card_ops}')

# Save
sd=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','nanogpt_parallel16_weights')
os.makedirs(sd,exist_ok=True)
np.save(f'{sd}/embed.npy',embed);np.save(f'{sd}/pos.npy',pos)
np.save(f'{sd}/gf.npy',gf);np.save(f'{sd}/Wh.npy',Wh);np.save(f'{sd}/bh.npy',bh)
for i,l in enumerate(layers):
    for k,v in l.items(): np.save(f'{sd}/l{i}_{k}.npy',v)
with open(f'{sd}/config.json','w') as f:
    json.dump({'V':V,'D':D,'H':H,'F':F,'T':T,'L':L,'HD':HD,'params':int(total),'loss':float(best),
               'card_ops':card_ops,'steps':step,'time':el,'cores':alive,'chars':''.join(chars)},f,indent=2)
print(f'Saved: {sd}/')

# Shutdown
for i in range(NCORES):
    e=(ctypes.c_uint*1)(OP_EXIT); nm.PL_WriteMemBlock(accesses[i],e,DDR+i*CMD_BLOCK,1)
time.sleep(0.3)
for acc in accesses.values(): nm.PL_CloseAccess(acc)
nm.PL_CloseBoardDesc(board)
print('All 16 cores safe. DONE.')
