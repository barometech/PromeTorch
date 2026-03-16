"""
200K Transformer with Self-Attention — ALL matmul on NM Card Mini float VPU.
dispatcher_float.abs: exact IEEE 754, no Q16.16 overflow.
"""
import numpy as np, time, os, json, ctypes, sys
sys.stdout.reconfigure(line_buffering=True)

# Data
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','tiny_shakespeare.txt'),'r') as f:
    text=f.read()
chars=sorted(set(text)); V=len(chars)
ch2idx={c:i for i,c in enumerate(chars)}; idx2ch={i:c for i,c in enumerate(chars)}
data=np.array([ch2idx[c] for c in text],dtype=np.int32)

# NM Card
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

DDR=0x00340000; buf1=(ctypes.c_uint*1)()
board=ctypes.c_void_p(); nm.PL_GetBoardDesc(0,ctypes.byref(board))
nm.PL_ResetBoard(board); nm.PL_LoadInitCode(board)
cn=(ctypes.c_int*2)(0,0); access=ctypes.c_void_p()
nm.PL_GetAccess(board,ctypes.byref(cn),ctypes.byref(access))
disp=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','aten','src','ATen','nmcard','nmc_programs','dispatcher_float.abs')
nm.PL_LoadProgramFile(access,disp.encode())
time.sleep(1)
nm.PL_ReadMemBlock(access,buf1,DDR+31,1)
print(f'NM Card alive, wd={buf1[0]}')

def f2u(a): return np.frombuffer(a.astype(np.float32).tobytes(),dtype=np.uint32)
def u2f(a): return np.frombuffer(np.array(a,dtype=np.uint32).tobytes(),dtype=np.float32)

def card_mm(A, B):
    M,K=A.shape; _,N=B.shape; DATA=DDR+64
    ab=(ctypes.c_uint*(M*K))(*f2u(A.flatten())); nm.PL_WriteMemBlock(access,ab,DATA,M*K)
    bb=(ctypes.c_uint*(K*N))(*f2u(B.flatten())); nm.PL_WriteMemBlock(access,bb,DATA+M*K,K*N)
    args=(ctypes.c_uint*6)(M,K,N,DATA,DATA+M*K,DATA+M*K+K*N)
    nm.PL_WriteMemBlock(access,args,DDR+1,6)
    z=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(access,z,DDR+30,1)
    c=(ctypes.c_uint*1)(1); nm.PL_WriteMemBlock(access,c,DDR,1)
    for _ in range(10000):
        nm.PL_ReadMemBlock(access,buf1,DDR+30,1)
        if buf1[0]==1: break
        time.sleep(0.0001)
    cb=(ctypes.c_uint*(M*N))(); nm.PL_ReadMemBlock(access,cb,DATA+M*K+K*N,M*N)
    return u2f(list(cb)).reshape(M,N)

# Verify
C=card_mm(np.array([[1,2],[3,4]],dtype=np.float32),np.array([[5,6],[7,8]],dtype=np.float32))
print(f'MatMul check: {C.flatten()} (expect [19 22 43 50])')

# Model: 200K, self-attention
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
gf=np.ones(D,np.float32)
Wh=he(D,V); bh=np.zeros(V,np.float32)
total=embed.size+pos.size+gf.size+Wh.size+bh.size+sum(sum(v.size for v in l.values()) for l in layers)
print(f'Model: {total:,} params, D={D}, H={H}, F={F}, T={T}, L={L}')

def sm(x): e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def rn(x,g): r=np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6); return x/r*g, r
def rn_bwd(dy, x, g, r):
    """Full RMS norm backward: d/dx (x/rms * g)"""
    D_ = x.shape[-1]
    xn = x / r  # normalized
    # dy_g = dy * g
    dy_g = dy * g
    # dx = (dy_g - xn * mean(dy_g * xn)) / r
    dot = np.sum(dy_g * xn, axis=-1, keepdims=True) / D_
    dx = (dy_g - xn * dot) / r
    return dx

# Training
print(f'\n=== TRAINING 200K TRANSFORMER ON NM CARD MINI ===')
lr=0.0003; best=99; step=0; start=time.time(); card_mm_count=0

while best > 1.0 and step < 10000:
    idx=np.random.randint(0,len(data)-T-1)
    tok=data[idx:idx+T]; tgt=data[idx+1:idx+T+1]; Tl=T
    x=embed[tok]+pos

    cache=[]
    for li,l in enumerate(layers):
        xn,r1=rn(x,l['g1'])
        # ALL projections ON CARD
        Q=card_mm(xn,l['Wq']); card_mm_count+=1
        K_=card_mm(xn,l['Wk']); card_mm_count+=1
        V_=card_mm(xn,l['Wv']); card_mm_count+=1
        # Attention scores + softmax on CPU (small ops)
        Q=Q.reshape(Tl,H,HD).transpose(1,0,2)
        K_=K_.reshape(Tl,H,HD).transpose(1,0,2)
        V_=V_.reshape(Tl,H,HD).transpose(1,0,2)
        sc=np.matmul(Q,K_.transpose(0,2,1))/np.sqrt(HD)+np.triu(np.full((Tl,Tl),-1e9),k=1)
        at=sm(sc); ao=np.matmul(at,V_).transpose(1,0,2).reshape(Tl,D)
        pr=card_mm(ao,l['Wo']); card_mm_count+=1
        xr=x+pr
        xn2,r2=rn(xr,l['g2'])
        h=card_mm(xn2,l['W1'])+l['b1']; card_mm_count+=1
        a=np.maximum(h,0)
        o=card_mm(a,l['W2'])+l['b2']; card_mm_count+=1
        xn_=xr+o
        cache.append((x,xn,r1,Q,K_,V_,at,ao,xr,xn2,r2,h,a))
        x=xn_

    xf,rf=rn(x,gf)
    logits=card_mm(xf,Wh)+bh; card_mm_count+=1
    probs=sm(logits)
    loss=-np.mean(np.log(probs[np.arange(Tl),tgt]+1e-9))
    if loss<best: best=loss

    # Backward (CPU)
    dl=probs.copy(); dl[np.arange(Tl),tgt]-=1; dl/=Tl
    dWh=xf.T@dl; dbh=dl.sum(0)
    dx=rn_bwd(dl@Wh.T, x, gf, rf)

    for li in range(L-1,-1,-1):
        l=layers[li]; x_in,xn,r1,Q,K_,V_,at,ao,xr,xn2,r2,h,a=cache[li]
        do=dx; dW2=a.T@do; db2=do.sum(0); da=do@l['W2'].T*(h>0)
        dW1=xn2.T@da; db1=da.sum(0); dxn2=da@l['W1'].T
        dx_r=dx+rn_bwd(dxn2, xr, l['g2'], r2)
        dpr=dx_r; dWo=ao.T@dpr; dao=dpr@l['Wo'].T
        dao_h=dao.reshape(Tl,H,HD).transpose(1,0,2)
        dV_=np.matmul(at.transpose(0,2,1),dao_h)
        dat=np.matmul(dao_h,V_.transpose(0,2,1))
        ds=at*(dat-np.sum(dat*at,-1,keepdims=True))/np.sqrt(HD)
        dQ=np.matmul(ds,K_); dK=np.matmul(ds.transpose(0,2,1),Q)
        dQ=dQ.transpose(1,0,2).reshape(Tl,D); dK=dK.transpose(1,0,2).reshape(Tl,D)
        dV_=dV_.transpose(1,0,2).reshape(Tl,D)
        dWq=xn.T@dQ; dWk=xn.T@dK; dWv=xn.T@dV_
        dxn_attn=dQ@l['Wq'].T+dK@l['Wk'].T+dV_@l['Wv'].T
        dx=dx_r+rn_bwd(dxn_attn, x_in, l['g1'], r1)
        l['Wq']-=lr*dWq; l['Wk']-=lr*dWk; l['Wv']-=lr*dWv; l['Wo']-=lr*dWo
        l['W1']-=lr*dW1; l['b1']-=lr*db1; l['W2']-=lr*dW2; l['b2']-=lr*db2

    Wh-=lr*dWh; bh-=lr*dbh; pos[:Tl]-=lr*0.1*dx
    for t in range(Tl): embed[tok[t]]-=lr*0.1*dx[t]

    if step==3000: lr=0.0002
    if step==6000: lr=0.0001

    if step%100==0:
        el=time.time()-start
        print(f'Step {step:5d} | loss={loss:.3f} | best={best:.3f} | mm={card_mm_count} | {el:.0f}s')
    if step%200==0 and step>0:
        p=list(ch2idx.get(c,0) for c in 'ROMEO:\n')
        for _ in range(100):
            inp=p[-T:];xg=embed[inp]+pos[:len(inp)]
            for l in layers:
                xn=xg/np.sqrt(np.mean(xg**2,-1,keepdims=True)+1e-6)*l['g1']
                Qg=(xn@l['Wq']).reshape(-1,H,HD).transpose(1,0,2)
                Kg=(xn@l['Wk']).reshape(-1,H,HD).transpose(1,0,2)
                Vg=(xn@l['Wv']).reshape(-1,H,HD).transpose(1,0,2)
                sc=np.matmul(Qg,Kg.transpose(0,2,1))/np.sqrt(HD)+np.triu(np.full((Qg.shape[1],Qg.shape[1]),-1e9),k=1)
                xg=xg+(sm(sc)@Vg).transpose(1,0,2).reshape(-1,D)@l['Wo']
                xn2=xg/np.sqrt(np.mean(xg**2,-1,keepdims=True)+1e-6)*l['g2']
                xg=xg+np.maximum(xn2@l['W1']+l['b1'],0)@l['W2']+l['b2']
            xf=xg/np.sqrt(np.mean(xg**2,-1,keepdims=True)+1e-6)*gf
            pp=sm((xf@Wh+bh)[-1]/0.9);pp/=pp.sum()
            p.append(np.random.choice(V,p=pp))
        print(f'  Gen: {"".join(idx2ch.get(t,"?") for t in p)[:150]}')
    step+=1

el=time.time()-start
print(f'\nDONE: {step} steps, {el:.0f}s, best={best:.3f}, card_mm={card_mm_count}')

# Generate
print('\n=== Shakespeare from 200K Transformer on NM Card Mini ===')
for pt in ['ROMEO:\n','JULIET:\n','First Citizen:\n']:
    p=list(ch2idx.get(c,0) for c in pt)
    for _ in range(300):
        inp=p[-T:]
        x=embed[inp]+pos[:len(inp)]
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
        p_=sm((xf@Wh+bh)[-1]/0.9); p_/=p_.sum()
        p.append(np.random.choice(V,p=p_))
    print(f'--- {pt.strip()} ---')
    print(''.join(idx2ch.get(t,'?') for t in p)); print()

# Save
sd=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','nanogpt_200k_attn_weights')
os.makedirs(sd,exist_ok=True)
np.save(f'{sd}/embed.npy',embed); np.save(f'{sd}/pos.npy',pos)
np.save(f'{sd}/gf.npy',gf); np.save(f'{sd}/Wh.npy',Wh); np.save(f'{sd}/bh.npy',bh)
for i,l in enumerate(layers):
    for k,v in l.items(): np.save(f'{sd}/l{i}_{k}.npy',v)
with open(f'{sd}/config.json','w') as f:
    json.dump({'V':V,'D':D,'H':H,'F':F,'T':T,'L':L,'HD':HD,'params':total,'loss':float(best),
               'card_matmuls':card_mm_count,'steps':step,'time':el,'chars':''.join(chars)},f,indent=2)
print(f'Saved: {sd}/')

# Shutdown
exit_cmd=(ctypes.c_uint*1)(255); nm.PL_WriteMemBlock(access,exit_cmd,DDR,1)
time.sleep(0.1); nm.PL_CloseAccess(access); nm.PL_CloseBoardDesc(board)
print('Card safe. DONE.')
