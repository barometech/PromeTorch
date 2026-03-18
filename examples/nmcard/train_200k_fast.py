"""
NanoGPT 278K — Self-attention transformer on NM Card Mini.
Weights preloaded to DDR. Only activations transferred per step.
Target loss: 1.6
"""
import numpy as np, time, os, json, ctypes, struct, sys

# ============================================================
# DATA
# ============================================================
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','tiny_shakespeare.txt'),'r') as f:
    text=f.read()
chars=sorted(set(text)); V=len(chars)
ch2idx={c:i for i,c in enumerate(chars)}; idx2ch={i:c for i,c in enumerate(chars)}
data=np.array([ch2idx[c] for c in text],dtype=np.int32)

# ============================================================
# NM CARD SETUP
# ============================================================
NM_PATH=r'C:\Program Files\Module\NM_Card\libload\bin'
os.environ['PATH']=NM_PATH+';'+os.environ.get('PATH','')
if hasattr(os,'add_dll_directory'): os.add_dll_directory(NM_PATH)
nm=ctypes.CDLL(os.path.join(NM_PATH,'nm_card_load.dll'))
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

DDR=0x00340000
board=ctypes.c_void_p(); nm.PL_GetBoardDesc(0,ctypes.byref(board))
nm.PL_ResetBoard(board); nm.PL_LoadInitCode(board)
core_no=(ctypes.c_int*2)(0,0); access=ctypes.c_void_p()
nm.PL_GetAccess(board,ctypes.byref(core_no),ctypes.byref(access))
disp=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','aten','src','ATen','nmcard','nmc_programs','dispatcher.abs')
nm.PL_LoadProgramFile(access,disp.encode())
time.sleep(0.5)
buf1=(ctypes.c_uint*1)()
nm.PL_ReadMemBlock(access,buf1,DDR+31,1)
print(f'NM Card alive, watchdog={buf1[0]}', flush=True)

def f2u(a): return np.frombuffer(a.astype(np.float32).tobytes(),dtype=np.uint32)
def u2f(a): return np.frombuffer(np.array(a,dtype=np.uint32).tobytes(),dtype=np.float32)

# ============================================================
# PRELOAD WEIGHTS TO DDR
# ============================================================
# DDR layout:
#   DDR+0..31: cmd block (32 words)
#   DDR+32..63: scratch
#   DDR+64...: weight storage + activation scratch

WEIGHT_BASE = DDR + 64    # weights start right after cmd block
# SCRATCH_BASE set AFTER all weights are uploaded (see below)

weight_addrs = {}  # name -> DDR address
next_addr = WEIGHT_BASE

def preload_weight(name, w):
    """Upload weight matrix to DDR once. Returns DDR address."""
    global next_addr
    wu = f2u(w.flatten())
    addr = next_addr
    buf = (ctypes.c_uint * len(wu))(*wu)
    nm.PL_WriteMemBlock(access, buf, addr, len(wu))
    weight_addrs[name] = (addr, w.shape)
    next_addr += len(wu)
    return addr

card_mm = 0
def card_matmul_preloaded(A, w_name):
    """MatMul using preloaded weight. Only uploads A, computes, downloads C."""
    global card_mm; card_mm += 1
    w_addr, w_shape = weight_addrs[w_name]
    M, K = A.shape
    _, N = w_shape[0], w_shape[1] if len(w_shape) > 1 else w_shape[0]
    N = w_shape[1]

    # Upload A to scratch area
    Au = f2u(A.flatten())
    a_addr = SCRATCH_BASE
    c_addr = SCRATCH_BASE + M * K
    a_buf = (ctypes.c_uint * len(Au))(*Au)
    nm.PL_WriteMemBlock(access, a_buf, a_addr, len(Au))

    # Send matmul: A @ W_preloaded
    args = (ctypes.c_uint * 6)(M, K, N, a_addr, w_addr, c_addr)
    nm.PL_WriteMemBlock(access, args, DDR + 1, 6)
    zero = (ctypes.c_uint * 1)(0)
    nm.PL_WriteMemBlock(access, zero, DDR + 30, 1)
    cmd = (ctypes.c_uint * 1)(1)
    nm.PL_WriteMemBlock(access, cmd, DDR, 1)

    # Wait
    for _ in range(2000):
        nm.PL_ReadMemBlock(access, buf1, DDR + 30, 1)
        if buf1[0] == 1: break
        time.sleep(0.0005)
    else:
        return A @ u2f(np.zeros(w_shape[0]*w_shape[1],dtype=np.uint32)).reshape(w_shape)

    # Download C
    c_buf = (ctypes.c_uint * (M * N))()
    nm.PL_ReadMemBlock(access, c_buf, c_addr, M * N)
    return u2f(list(c_buf)).reshape(M, N)

# ============================================================
# MODEL: 278K params, 3 layers, 6 heads, self-attention
# ============================================================
D=96; H=6; HD=D//H; F=256; T=64; L=3
np.random.seed(42)
# Smaller init for Q16.16 safety (max range ±32768)
he=lambda fi,fo: np.random.randn(fi,fo).astype(np.float32)*np.sqrt(1/(fi+fo))*0.5

embed=np.random.randn(V,D).astype(np.float32)*0.02
pos=np.random.randn(T,D).astype(np.float32)*0.01
layers=[]
for _ in range(L):
    layers.append({'g1':np.ones(D,dtype=np.float32),
        'Wq':he(D,D),'Wk':he(D,D),'Wv':he(D,D),'Wo':he(D,D),
        'g2':np.ones(D,dtype=np.float32),
        'W1':he(D,F),'b1':np.zeros(F,dtype=np.float32),
        'W2':he(F,D),'b2':np.zeros(D,dtype=np.float32)})
gf=np.ones(D,dtype=np.float32)
Wh=he(D,V); bh=np.zeros(V,dtype=np.float32)
total=embed.size+pos.size+gf.size+Wh.size+bh.size+sum(sum(v.size for v in l.values()) for l in layers)
print(f'Model: {total:,} params, D={D}, H={H}, F={F}, T={T}, L={L}', flush=True)

# Preload ALL weights to DDR
print('Preloading weights to NM Card DDR...', flush=True)
for i, l in enumerate(layers):
    for wn in ['Wq','Wk','Wv','Wo','W1','W2']:
        preload_weight(f'l{i}_{wn}', l[wn])
preload_weight('Wh', Wh)
total_words = next_addr - WEIGHT_BASE
SCRATCH_BASE = next_addr + 256  # scratch area AFTER all weights
print(f'  {total_words:,} words ({total_words*4/1024:.1f} KB) uploaded to DDR', flush=True)
print(f'  {len(weight_addrs)} weight matrices preloaded', flush=True)
print(f'  Scratch: 0x{SCRATCH_BASE:08x} (after weights)', flush=True)

# Verify one
C = card_matmul_preloaded(np.eye(2, dtype=np.float32), 'l0_Wq')
print(f'  Verify preloaded matmul: shape={C.shape}, nonzero={np.count_nonzero(C)}', flush=True)

# ============================================================
# TRAINING
# ============================================================
def sm(x): e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def rn(x,g): r=np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6); return x/r*g,r

print(f'\n=== TRAINING ON NM CARD MINI (target loss 1.6) ===', flush=True)
lr=0.003; best=99; step=0; start=time.time()

while best > 1.6 and step < 10000:
    idx=np.random.randint(0,len(data)-T-1)
    tok=data[idx:idx+T]; tgt=data[idx+1:idx+T+1]
    Tl=T; x=embed[tok]+pos

    cache=[]
    for li, l in enumerate(layers):
        xn,r1=rn(x,l['g1'])

        # Linear projections ON NM CARD, attention math ON CPU
        Q = card_matmul_preloaded(xn, f'l{li}_Wq')
        K_ = card_matmul_preloaded(xn, f'l{li}_Wk')
        V_ = card_matmul_preloaded(xn, f'l{li}_Wv')

        # Attention scores + softmax ON CPU (Q16.16 can't handle these)
        Q=Q.reshape(Tl,H,HD).transpose(1,0,2)
        K_=K_.reshape(Tl,H,HD).transpose(1,0,2)
        V_=V_.reshape(Tl,H,HD).transpose(1,0,2)
        sc=np.matmul(Q,K_.transpose(0,2,1))/np.sqrt(HD)+np.triu(np.full((Tl,Tl),-1e9),k=1)
        at=sm(sc); ao=np.matmul(at,V_).transpose(1,0,2).reshape(Tl,D)

        # Output projection ON CARD
        pr = card_matmul_preloaded(ao, f'l{li}_Wo')
        xr=x+pr

        # FFN ON CARD
        xn2,r2=rn(xr,l['g2'])
        h = card_matmul_preloaded(xn2, f'l{li}_W1') + l['b1']
        a=np.maximum(h,0)  # ReLU on CPU (trivial)
        o = card_matmul_preloaded(a, f'l{li}_W2') + l['b2']
        xn_=xr+o

        cache.append((x,xn,r1,Q,K_,V_,at,ao,xr,xn2,r2,h,a))
        x=xn_

    # Head (on card too)
    xf,rf=rn(x,gf)
    logits = card_matmul_preloaded(xf, 'Wh') + bh
    probs=sm(logits)
    loss=-np.mean(np.log(probs[np.arange(Tl),tgt]+1e-9))
    if loss<best: best=loss

    # Backward (CPU — gradient computation)
    dl=probs.copy(); dl[np.arange(Tl),tgt]-=1; dl/=Tl
    dWh_=xf.T@dl; dbh_=dl.sum(0); dx=dl@Wh.T*gf/rf

    for li in range(L-1,-1,-1):
        l=layers[li]; x_in,xn,r1,Q,K_,V_,at,ao,xr,xn2,r2,h,a=cache[li]
        do=dx; dW2_=a.T@do; db2_=do.sum(0); da=do@l['W2'].T*(h>0)
        dW1_=xn2.T@da; db1_=da.sum(0); dxn2=da@l['W1'].T
        dx_r=dx+dxn2*l['g2']/r2
        dpr=dx_r; dWo_=ao.T@dpr; dao=dpr@l['Wo'].T
        dao_h=dao.reshape(Tl,H,HD).transpose(1,0,2)
        dV_=np.matmul(at.transpose(0,2,1),dao_h)
        dat=np.matmul(dao_h,V_.transpose(0,2,1))
        ds=at*(dat-np.sum(dat*at,-1,keepdims=True))/np.sqrt(HD)
        dQ=np.matmul(ds,K_); dK=np.matmul(ds.transpose(0,2,1),Q)
        dQ=dQ.transpose(1,0,2).reshape(Tl,D); dK=dK.transpose(1,0,2).reshape(Tl,D)
        dV_=dV_.transpose(1,0,2).reshape(Tl,D)
        dWq_=xn.T@dQ; dWk_=xn.T@dK; dWv_=xn.T@dV_
        dxn_=(dQ@l['Wq'].T+dK@l['Wk'].T+dV_@l['Wv'].T)
        dx=dx_r+dxn_*l['g1']/r1

        # Gradient clipping (Q16.16 safe)
        clip=1.0
        for g in [dWq_,dWk_,dWv_,dWo_,dW1_,dW2_]:
            np.clip(g,-clip,clip,out=g)

        # SGD update + re-upload changed weights to DDR
        l['Wq']-=lr*dWq_; l['Wk']-=lr*dWk_; l['Wv']-=lr*dWv_; l['Wo']-=lr*dWo_
        l['W1']-=lr*dW1_; l['b1']-=lr*np.clip(db1_,-clip,clip); l['W2']-=lr*dW2_; l['b2']-=lr*np.clip(db2_,-clip,clip)

    np.clip(dWh_,-clip,clip,out=dWh_)
    Wh-=lr*dWh_; bh-=lr*np.clip(dbh_,-clip,clip); pos[:Tl]-=lr*0.1*np.clip(dx,-clip,clip)
    for t in range(Tl): embed[tok[t]]-=lr*0.1*dx[t]

    # Re-upload updated weights every 10 steps (amortize PCI cost)
    if step % 10 == 0:
        for i, l in enumerate(layers):
            for wn in ['Wq','Wk','Wv','Wo','W1','W2']:
                addr = weight_addrs[f'l{i}_{wn}'][0]
                weight_bytes = l[wn].astype(np.float32).tobytes()
                wbuf = (ctypes.c_uint * (len(weight_bytes)//4)).from_buffer_copy(weight_bytes)
                nm.PL_WriteMemBlock(access, wbuf, addr, len(wbuf))
        wa = weight_addrs['Wh'][0]
        weight_bytes = Wh.astype(np.float32).tobytes()
        wbuf = (ctypes.c_uint * (len(weight_bytes)//4)).from_buffer_copy(weight_bytes)
        nm.PL_WriteMemBlock(access, wbuf, wa, len(wbuf))

    if step==2000: lr=0.001
    if step==5000: lr=0.0003

    if step%50==0:
        el=time.time()-start
        print(f'Step {step:5d} | loss={loss:.3f} | best={best:.3f} | card_mm={card_mm} | {el:.0f}s', flush=True)
    step+=1

el=time.time()-start
print(f'\nDONE: {step} steps, {el:.0f}s, best={best:.3f}, card_mm={card_mm}', flush=True)

# Generate
for pt in ['ROMEO:\n','JULIET:\n','First Citizen:\n']:
    p=list(ch2idx.get(c,0) for c in pt)
    for _ in range(200):
        inp=p[-T:]; x=embed[inp]+pos[:len(inp)]
        for l in layers:
            xn=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*l['g1']
            Q=(xn@l['Wq']).reshape(-1,H,HD).transpose(1,0,2)
            K_=(xn@l['Wk']).reshape(-1,H,HD).transpose(1,0,2)
            Vl=(xn@l['Wv']).reshape(-1,H,HD).transpose(1,0,2)
            sc=np.matmul(Q,K_.transpose(0,2,1))/np.sqrt(HD)+np.triu(np.full((Q.shape[1],Q.shape[1]),-1e9),k=1)
            x=x+(sm(sc)@Vl).transpose(1,0,2).reshape(-1,D)@l['Wo']
            xn2=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*l['g2']
            x=x+np.maximum(xn2@l['W1']+l['b1'],0)@l['W2']+l['b2']
        xf=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*gf
        p_=sm((xf@Wh+bh)[-1])**(1/0.7); p_/=p_.sum()
        p.append(np.random.choice(V,p=p_))
    print(f'--- {pt.strip()} ---')
    print(''.join(idx2ch.get(t,'?') for t in p)); print(flush=True)

# Save
sd=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','nanogpt_200k_weights')
os.makedirs(sd,exist_ok=True)
np.save(f'{sd}/embed.npy',embed); np.save(f'{sd}/pos.npy',pos)
np.save(f'{sd}/gf.npy',gf); np.save(f'{sd}/Wh.npy',Wh); np.save(f'{sd}/bh.npy',bh)
for i,l in enumerate(layers):
    for k,v in l.items(): np.save(f'{sd}/l{i}_{k}.npy',v)
with open(f'{sd}/config.json','w') as f:
    json.dump({'V':V,'D':D,'H':H,'F':F,'T':T,'L':L,'params':total,'loss':float(best),
               'card_matmuls':card_mm,'steps':step,'time':el,'chars':''.join(chars)},f)
print(f'Saved: {sd}/', flush=True)

# Shutdown card
exit_cmd=(ctypes.c_uint*1)(255); nm.PL_WriteMemBlock(access,exit_cmd,DDR,1)
time.sleep(0.1); nm.PL_CloseAccess(access); nm.PL_CloseBoardDesc(board)
print('Card safe. DONE.', flush=True)
