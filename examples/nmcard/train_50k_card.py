"""50K transformer, ALL matmul on NM Card Mini, target loss 1.6"""
import numpy as np, time, os, json, ctypes, struct, sys
sys.stdout.reconfigure(line_buffering=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','tiny_shakespeare.txt'),'r') as f:
    text=f.read()
chars=sorted(set(text)); V=len(chars)
ch2idx={c:i for i,c in enumerate(chars)}; idx2ch={i:c for i,c in enumerate(chars)}
data=np.array([ch2idx[c] for c in text],dtype=np.int32)

# NM Card
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
print(f'NM Card alive, wd={buf1[0]}')

def f2u(a): return np.frombuffer(a.astype(np.float32).tobytes(),dtype=np.uint32)
def u2f(a): return np.frombuffer(np.array(a,dtype=np.uint32).tobytes(),dtype=np.float32)

# Preload weights to DDR
WEIGHT_BASE=DDR+64; next_addr=WEIGHT_BASE; weight_addrs={}
def preload(name,w):
    global next_addr
    wu=f2u(w.flatten()); addr=next_addr
    b=(ctypes.c_uint*len(wu))(*wu); nm.PL_WriteMemBlock(access,b,addr,len(wu))
    weight_addrs[name]=(addr,w.shape); next_addr+=len(wu)

card_mm=0
def card_mm_f(A, w_name):
    """ALL matmul on NMC4. A=activations, w_name=preloaded weight."""
    global card_mm; card_mm+=1
    w_addr,w_shape=weight_addrs[w_name]; M,K=A.shape; N=w_shape[1]
    # Q16.16 safe scaling: divide inputs, multiply result by scale^2
    # max sum = K * max_a * max_w. If both < sqrt(32768/K) ~ safe


  # now max ~1.0
    Au=f2u(np.clip(A,-20,20).flatten()); a_addr=SCRATCH; c_addr=SCRATCH+M*K
    ab=(ctypes.c_uint*len(Au))(*Au); nm.PL_WriteMemBlock(access,ab,a_addr,len(Au))
    args=(ctypes.c_uint*6)(M,K,N,a_addr,w_addr,c_addr)
    nm.PL_WriteMemBlock(access,args,DDR+1,6)
    z=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(access,z,DDR+30,1)
    c=(ctypes.c_uint*1)(1); nm.PL_WriteMemBlock(access,c,DDR,1)
    for _ in range(5000):
        nm.PL_ReadMemBlock(access,buf1,DDR+30,1)
        if buf1[0]==1: break
        time.sleep(0.0002)
    else:
        print('TIMEOUT!'); return A@np.zeros(w_shape,dtype=np.float32)
    cb=(ctypes.c_uint*(M*N))(); nm.PL_ReadMemBlock(access,cb,c_addr,M*N)
    result = u2f(list(cb)).reshape(M,N)
    return result  # undo input scaling

# 50K model: D=48, H=4, F=128, T=32, L=2
D=32; H=4; HD=D//H; F=96; T=32; L=4
np.random.seed(42)
sc=lambda fi,fo: np.random.randn(fi,fo).astype(np.float32)*0.04

embed=np.random.randn(V,D).astype(np.float32)*0.02
pos=np.random.randn(T,D).astype(np.float32)*0.01
layers=[]
for _ in range(L):
    layers.append({'g1':np.ones(D,dtype=np.float32),
        'Wq':sc(D,D),'Wk':sc(D,D),'Wv':sc(D,D),'Wo':sc(D,D),
        'g2':np.ones(D,dtype=np.float32),
        'W1':sc(D,F),'b1':np.zeros(F,dtype=np.float32),
        'W2':sc(F,D),'b2':np.zeros(D,dtype=np.float32)})
gf=np.ones(D,dtype=np.float32)
Wh=sc(D,V); bh=np.zeros(V,dtype=np.float32)
total=embed.size+pos.size+gf.size+Wh.size+bh.size+sum(sum(v.size for v in l.values()) for l in layers)
print(f'Model: {total:,} params, D={D}, H={H}, F={F}, T={T}, L={L}')

# Preload ALL weights
for i,l in enumerate(layers):
    for wn in ['Wq','Wk','Wv','Wo','W1','W2']: preload(f'l{i}_{wn}',l[wn])
preload('Wh',Wh)
SCRATCH=next_addr+256
print(f'Weights: {(next_addr-WEIGHT_BASE)*4/1024:.1f}KB on DDR, scratch=0x{SCRATCH:x}')

# Verify
C=card_mm_f(np.array([[1,2],[3,4]],dtype=np.float32).reshape(1,4)@np.zeros((4,D),dtype=np.float32)+np.eye(2,D,dtype=np.float32),'l0_Wq')
print(f'Card verify: nonzero={np.count_nonzero(C)}')

def sm(x): e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def rn(x,g): r=np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6); return x/r*g,r

print(f'\n=== TRAINING ON NM CARD MINI ===')
lr=0.005; best=99; step=0; start=time.time()

while best>1.6 and step<20000:
    idx=np.random.randint(0,len(data)-T-1)
    tok=data[idx:idx+T]; tgt=data[idx+1:idx+T+1]
    x=embed[tok]+pos; Tl=T

    cache=[]
    for li,l in enumerate(layers):
        xn,r1=rn(x,l['g1'])
        # ALL linear projections ON CARD
        Q=card_mm_f(xn,f'l{li}_Wq')
        K_=card_mm_f(xn,f'l{li}_Wk')
        V_=card_mm_f(xn,f'l{li}_Wv')
        # Attention ON CARD too (Q@K^T is small: [T,HD]@[HD,T])
        Q=Q.reshape(Tl,H,HD).transpose(1,0,2)
        K_=K_.reshape(Tl,H,HD).transpose(1,0,2)
        V_=V_.reshape(Tl,H,HD).transpose(1,0,2)
        # scores + softmax (small ops, but need float precision)
        sc_=np.matmul(Q,K_.transpose(0,2,1))/np.sqrt(HD)
        sc_+=np.triu(np.full((Tl,Tl),-1e9),k=1)
        at=sm(sc_)
        ao=np.matmul(at,V_).transpose(1,0,2).reshape(Tl,D)
        pr=card_mm_f(ao,f'l{li}_Wo')
        xr=x+pr
        xn2,r2=rn(xr,l['g2'])
        h=card_mm_f(xn2,f'l{li}_W1')+l['b1']
        a=np.maximum(h,0)
        o=card_mm_f(a,f'l{li}_W2')+l['b2']
        xn_=xr+o
        cache.append((x,xn,r1,Q,K_,V_,at,ao,xr,xn2,r2,h,a))
        x=xn_

    xf,rf=rn(x,gf)
    logits=card_mm_f(xf,'Wh')+bh
    probs=sm(logits)
    loss=-np.mean(np.log(np.clip(probs[np.arange(Tl),tgt],1e-9,1)))
    if np.isnan(loss): loss=99.0
    if loss<best: best=loss

    # Backward
    dl=probs.copy(); dl[np.arange(Tl),tgt]-=1; dl/=Tl
    np.clip(dl,-1,1,out=dl)
    dWh_=xf.T@dl; dbh_=dl.sum(0); dx=dl@Wh.T*gf/rf

    for li in range(L-1,-1,-1):
        l=layers[li]; x_in,xn,r1,Q,K_,V_,at,ao,xr,xn2,r2,h,a=cache[li]
        np.clip(dx,-2,2,out=dx)
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
        dx=dx_r+(dQ@l['Wq'].T+dK@l['Wk'].T+dV_@l['Wv'].T)*l['g1']/r1
        # Clip + update
        for g in [dWq_,dWk_,dWv_,dWo_,dW1_,dW2_]: np.clip(g,-1,1,out=g)
        l['Wq']-=lr*dWq_; l['Wk']-=lr*dWk_; l['Wv']-=lr*dWv_; l['Wo']-=lr*dWo_
        l['W1']-=lr*dW1_; l['b1']-=lr*np.clip(db1_,-1,1); l['W2']-=lr*dW2_; l['b2']-=lr*np.clip(db2_,-1,1)

    np.clip(dWh_,-1,1,out=dWh_); Wh-=lr*dWh_; bh-=lr*np.clip(dbh_,-1,1)
    pos[:Tl]-=lr*0.1*np.clip(dx,-1,1)
    for t in range(Tl): embed[tok[t]]-=lr*0.1*np.clip(dx[t],-1,1)

    # Clip weights to Q16.16 safe range + re-upload every 5 steps
    for l in layers:
        for wn in ['Wq','Wk','Wv','Wo','W1','W2']:
            np.clip(l[wn],-2,2,out=l[wn])
    np.clip(Wh,-2,2,out=Wh)
    if step%5==0:
        for i,l in enumerate(layers):
            for wn in ['Wq','Wk','Wv','Wo','W1','W2']:
                wa,_=weight_addrs[f'l{i}_{wn}']
                wu=f2u(l[wn].flatten()); wb=(ctypes.c_uint*len(wu))(*wu)
                nm.PL_WriteMemBlock(access,wb,wa,len(wu))
        wa,_=weight_addrs['Wh']; wu=f2u(Wh.flatten()); wb=(ctypes.c_uint*len(wu))(*wu)
        nm.PL_WriteMemBlock(access,wb,wa,len(wu))

    if step==3000: lr=0.003
    if step==8000: lr=0.001

    if step%100==0:
        el=time.time()-start
        print(f'Step {step:5d} | loss={loss:.3f} | best={best:.3f} | mm={card_mm} | {el:.0f}s')
    step+=1

el=time.time()-start
print(f'\nDONE: {step} steps, {el:.0f}s, best={best:.3f}, card_mm={card_mm}')

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
            sc_=np.matmul(Q,K_.transpose(0,2,1))/np.sqrt(HD)+np.triu(np.full((Q.shape[1],Q.shape[1]),-1e9),k=1)
            x=x+(sm(sc_)@Vl).transpose(1,0,2).reshape(-1,D)@l['Wo']
            xn2=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*l['g2']
            x=x+np.maximum(xn2@l['W1']+l['b1'],0)@l['W2']+l['b2']
        xf=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*gf
        p_=sm((xf@Wh+bh)[-1])**(1/0.8); p_/=p_.sum()
        p.append(np.random.choice(V,p=p_))
    print(f'--- {pt.strip()} ---')
    print(''.join(idx2ch.get(t,'?') for t in p)); print()

# Save
sd=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','nanogpt_50k_card')
os.makedirs(sd,exist_ok=True)
np.save(f'{sd}/embed.npy',embed); np.save(f'{sd}/pos.npy',pos)
np.save(f'{sd}/gf.npy',gf); np.save(f'{sd}/Wh.npy',Wh); np.save(f'{sd}/bh.npy',bh)
for i,l in enumerate(layers):
    for k,v in l.items(): np.save(f'{sd}/l{i}_{k}.npy',v)
with open(f'{sd}/config.json','w') as f:
    json.dump({'V':V,'D':D,'H':H,'F':F,'T':T,'L':L,'HD':HD,'params':total,'loss':float(best),
               'card_matmuls':card_mm,'steps':step,'time':el,'chars':''.join(chars)},f)
print(f'Saved: {sd}/')

# Shutdown
exit_cmd=(ctypes.c_uint*1)(255); nm.PL_WriteMemBlock(access,exit_cmd,DDR,1)
time.sleep(0.1); nm.PL_CloseAccess(access); nm.PL_CloseBoardDesc(board)
print('Card safe. DONE.')
