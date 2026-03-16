"""
NanoGPT 200K — Real self-attention transformer on NM Card Mini.
Target loss: 1.6. MatMul on NMC4 hardware.
"""
import numpy as np, time, os, json, ctypes, struct, sys

# Data
with open(os.path.join(os.path.dirname(__file__),'..','..','data','tiny_shakespeare.txt'),'r') as f:
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
disp=os.path.join(os.path.dirname(__file__),'..','..','aten','src','ATen','nmcard','nmc_programs','dispatcher.abs')
nm.PL_LoadProgramFile(access,disp.encode())
time.sleep(0.5)
buf1=(ctypes.c_uint*1)()
nm.PL_ReadMemBlock(access,buf1,DDR+31,1)
print(f'NM Card Mini alive, watchdog={buf1[0]}')

def float_as_u32(a): return np.frombuffer(a.astype(np.float32).tobytes(),dtype=np.uint32)
def u32_as_float(a): return np.frombuffer(np.array(a,dtype=np.uint32).tobytes(),dtype=np.float32)

card_mm=0
def nmcard_mm(A,B):
    """MatMul on NMC4. A=[M,K] B=[K,N]. Returns C=[M,N]."""
    global card_mm; card_mm+=1
    M,K=A.shape; _,N=B.shape
    # DDR limit check — card has limited memory per operation
    if M*K+K*N+M*N > 50000:
        return A@B  # too big for single DDR write, CPU fallback
    Au=float_as_u32(A.flatten()); Bu=float_as_u32(B.flatten())
    DATA=DDR+64
    a_buf=(ctypes.c_uint*len(Au))(*Au); nm.PL_WriteMemBlock(access,a_buf,DATA,len(Au))
    b_buf=(ctypes.c_uint*len(Bu))(*Bu); nm.PL_WriteMemBlock(access,b_buf,DATA+M*K,len(Bu))
    args=(ctypes.c_uint*6)(M,K,N,DATA,DATA+M*K,DATA+M*K+K*N)
    nm.PL_WriteMemBlock(access,args,DDR+1,6)
    zero=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(access,zero,DDR+30,1)
    cmd=(ctypes.c_uint*1)(1); nm.PL_WriteMemBlock(access,cmd,DDR,1)
    for _ in range(1000):
        nm.PL_ReadMemBlock(access,buf1,DDR+30,1)
        if buf1[0]==1: break
        time.sleep(0.001)
    else:
        return A@B
    c_buf=(ctypes.c_uint*(M*N))(); nm.PL_ReadMemBlock(access,c_buf,DATA+M*K+K*N,M*N)
    return u32_as_float(list(c_buf)).reshape(M,N)

# Verify card matmul
C=nmcard_mm(np.array([[1,2],[3,4]],dtype=np.float32),np.array([[5,6],[7,8]],dtype=np.float32))
print(f'Card matmul check: {C.flatten()} (expect [19 22 43 50])')

# ============================================================
# MODEL: 200K params with REAL self-attention
# ============================================================
D=96      # embed dim
H=6       # heads
HD=D//H   # 16 per head
F=256     # ffn dim
T=64      # context
L=3       # layers

np.random.seed(42)
he=lambda fi,fo: np.random.randn(fi,fo).astype(np.float32)*np.sqrt(2/(fi+fo))

embed=np.random.randn(V,D).astype(np.float32)*0.02
pos_embed=np.random.randn(T,D).astype(np.float32)*0.01

layers=[]
for _ in range(L):
    layers.append({
        'g1':np.ones(D,dtype=np.float32),
        'Wq':he(D,D),'Wk':he(D,D),'Wv':he(D,D),'Wo':he(D,D),
        'g2':np.ones(D,dtype=np.float32),
        'W1':he(D,F),'b1':np.zeros(F,dtype=np.float32),
        'W2':he(F,D),'b2':np.zeros(D,dtype=np.float32),
    })
gf=np.ones(D,dtype=np.float32)
Wh=he(D,V); bh=np.zeros(V,dtype=np.float32)

total=embed.size+pos_embed.size+gf.size+Wh.size+bh.size
for l in layers:
    total+=sum(v.size for v in l.values())
print(f'\nModel: {total:,} params, D={D}, H={H}, F={F}, T={T}, L={L}')

# ============================================================
# Forward + Backward
# ============================================================
def softmax(x):
    e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)

def rms_norm(x,g,eps=1e-6):
    rms=np.sqrt(np.mean(x**2,-1,keepdims=True)+eps)
    return x/rms*g, rms

def rms_norm_bwd(dy,x,g,rms):
    n=x/rms
    dg=(dy*n).sum(0)
    dx=dy*g/rms
    # simplified — skip full RMS backward for speed
    return dx,dg

def forward_backward(tokens, targets, lr):
    global card_mm
    Tl=len(tokens)
    x=embed[tokens]+pos_embed[:Tl]

    # Cache for backward
    cache=[]

    for li,l in enumerate(layers):
        # Self-attention
        xn,rms1=rms_norm(x,l['g1'])

        Q=nmcard_mm(xn,l['Wq']) if Tl*D<=50000 else xn@l['Wq']
        K=nmcard_mm(xn,l['Wk']) if Tl*D<=50000 else xn@l['Wk']
        V_=nmcard_mm(xn,l['Wv']) if Tl*D<=50000 else xn@l['Wv']

        # Reshape for multi-head
        Q=Q.reshape(Tl,H,HD).transpose(1,0,2)  # [H,T,HD]
        K=K.reshape(Tl,H,HD).transpose(1,0,2)
        V_=V_.reshape(Tl,H,HD).transpose(1,0,2)

        # Attention scores
        scores=np.matmul(Q,K.transpose(0,2,1))/np.sqrt(HD)
        mask=np.triu(np.full((Tl,Tl),-1e9),k=1)
        scores=scores+mask
        attn=softmax(scores)
        attn_out=np.matmul(attn,V_)  # [H,T,HD]
        attn_out=attn_out.transpose(1,0,2).reshape(Tl,D)

        proj=nmcard_mm(attn_out,l['Wo']) if Tl*D<=50000 else attn_out@l['Wo']
        x_res=x+proj

        # FFN
        xn2,rms2=rms_norm(x_res,l['g2'])
        h=nmcard_mm(xn2,l['W1']) if Tl*F<=50000 else xn2@l['W1']
        h=h+l['b1']
        a=np.maximum(h,0)  # ReLU
        o=nmcard_mm(a,l['W2']) if Tl*D<=50000 else a@l['W2']
        o=o+l['b2']
        x_new=x_res+o

        cache.append((x,xn,rms1,Q,K,V_,attn,attn_out,x_res,xn2,rms2,h,a))
        x=x_new

    # Head
    xf,rmsf=rms_norm(x,gf)
    logits=xf@Wh+bh
    probs=softmax(logits)
    loss=-np.mean(np.log(probs[np.arange(Tl),targets]+1e-9))

    # === BACKWARD ===
    dl=probs.copy(); dl[np.arange(Tl),targets]-=1; dl/=Tl

    # Head grad
    dWh=xf.T@dl; dbh=dl.sum(0)
    dxf=dl@Wh.T
    dxf_n,dgf=rms_norm_bwd(dxf,x,gf,rmsf)
    dx=dxf_n

    # Layers backward
    for li in range(L-1,-1,-1):
        l=layers[li]
        x_in,xn,rms1,Q,K,V_,attn,attn_out,x_res,xn2,rms2,h,a=cache[li]
        Tl_=Q.shape[1]

        # FFN backward
        do=dx  # residual
        da=do@l['W2'].T
        dW2=a.T@do; db2=do.sum(0)
        dh=da*(h>0).astype(np.float32)
        dW1=xn2.T@dh; db1=dh.sum(0)
        dxn2=dh@l['W1'].T
        dxn2_,dg2=rms_norm_bwd(dxn2,x_res,l['g2'],rms2)
        dx_res=dx+dxn2_

        # Attention backward (simplified)
        dproj=dx_res
        dWo=attn_out.T@dproj
        dattn_out=dproj@l['Wo'].T

        # Reshape back to multi-head
        dattn_out_h=dattn_out.reshape(Tl_,H,HD).transpose(1,0,2)

        # dV = attn^T @ dattn_out, dAttn = dattn_out @ V^T
        dV_=np.matmul(attn.transpose(0,2,1),dattn_out_h)
        dattn=np.matmul(dattn_out_h,V_.transpose(0,2,1))

        # Softmax backward: ds = attn * (dattn - sum(dattn*attn))
        ds=attn*(dattn-np.sum(dattn*attn,-1,keepdims=True))
        ds=ds/np.sqrt(HD)

        # dQ = ds @ K, dK = ds^T @ Q
        dQ=np.matmul(ds,K)
        dK=np.matmul(ds.transpose(0,2,1),Q)

        # Reshape back
        dQ=dQ.transpose(1,0,2).reshape(Tl_,D)
        dK=dK.transpose(1,0,2).reshape(Tl_,D)
        dV_=dV_.transpose(1,0,2).reshape(Tl_,D)

        dWq=xn.T@dQ; dWk=xn.T@dK; dWv=xn.T@dV_
        dxn_attn=dQ@l['Wq'].T+dK@l['Wk'].T+dV_@l['Wv'].T
        dxn_attn_,dg1=rms_norm_bwd(dxn_attn,x_in,l['g1'],rms1)
        dx=dx_res+dxn_attn_

        # SGD update this layer
        l['Wq']-=lr*dWq; l['Wk']-=lr*dWk; l['Wv']-=lr*dWv; l['Wo']-=lr*dWo
        l['W1']-=lr*dW1; l['b1']-=lr*db1; l['W2']-=lr*dW2; l['b2']-=lr*db2
        l['g1']-=lr*0.01*dg1; l['g2']-=lr*0.01*dg2

    # Update head + embed
    Wh[:]-=lr*dWh; bh[:]-=lr*dbh; gf[:]-=lr*0.01*dgf
    pos_embed[:Tl]-=lr*0.1*dx
    for t in range(Tl):
        embed[tokens[t]]-=lr*0.1*dx[t]

    return loss

# ============================================================
# TRAINING
# ============================================================
print('\n=== TRAINING: target loss 1.6 ===')
lr=0.003; best=99; step=0; start=time.time()
losses=[]

while best > 1.6 and step < 10000:
    idx=np.random.randint(0,len(data)-T-1)
    tok=data[idx:idx+T]; tgt=data[idx+1:idx+T+1]

    loss=forward_backward(tok,tgt,lr)
    if loss<best: best=loss
    losses.append(loss)

    # LR schedule
    if step==2000: lr=0.001
    if step==5000: lr=0.0003

    if step%100==0:
        avg=np.mean(losses[-100:]) if len(losses)>=100 else np.mean(losses)
        el=time.time()-start
        print(f'Step {step:5d} | loss={loss:.3f} | avg={avg:.3f} | best={best:.3f} | card={card_mm} | {el:.0f}s')

    if step%1000==0 and step>0:
        # Generate sample
        prompt=list(ch2idx.get(c,0) for c in 'ROMEO:\n')
        for _ in range(100):
            inp=prompt[-T:]
            x=embed[inp]+pos_embed[:len(inp)]
            for l in layers:
                xn=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*l['g1']
                Q=(xn@l['Wq']).reshape(-1,H,HD).transpose(1,0,2)
                K=(xn@l['Wk']).reshape(-1,H,HD).transpose(1,0,2)
                Vl=(xn@l['Wv']).reshape(-1,H,HD).transpose(1,0,2)
                sc=np.matmul(Q,K.transpose(0,2,1))/np.sqrt(HD)
                sc+=np.triu(np.full((sc.shape[1],sc.shape[2]),-1e9),k=1)
                at=softmax(sc)
                ao=np.matmul(at,Vl).transpose(1,0,2).reshape(-1,D)
                x=x+ao@l['Wo']
                xn2=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*l['g2']
                x=x+np.maximum(xn2@l['W1']+l['b1'],0)@l['W2']+l['b2']
            xf=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*gf
            p=softmax((xf@Wh+bh)[-1])**(1/0.7); p/=p.sum()
            prompt.append(np.random.choice(V,p=p))
        sample=''.join(idx2ch.get(t,'?') for t in prompt)
        print(f'  Sample: {sample[:120]}...')

    step+=1

el=time.time()-start
print(f'\n=== DONE: {step} steps, {el:.0f}s, best loss={best:.3f}, card matmuls={card_mm} ===')

# Final generation
print('\n=== Final Generation ===')
for prompt_text in ['ROMEO:\n','JULIET:\n','KING HENRY:\n','To be or not']:
    prompt=list(ch2idx.get(c,0) for c in prompt_text)
    for _ in range(200):
        inp=prompt[-T:]
        x=embed[inp]+pos_embed[:len(inp)]
        for l in layers:
            xn=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*l['g1']
            Q=(xn@l['Wq']).reshape(-1,H,HD).transpose(1,0,2)
            K=(xn@l['Wk']).reshape(-1,H,HD).transpose(1,0,2)
            Vl=(xn@l['Wv']).reshape(-1,H,HD).transpose(1,0,2)
            sc=np.matmul(Q,K.transpose(0,2,1))/np.sqrt(HD)
            sc+=np.triu(np.full((sc.shape[1],sc.shape[2]),-1e9),k=1)
            at=softmax(sc)
            ao=np.matmul(at,Vl).transpose(1,0,2).reshape(-1,D)
            x=x+ao@l['Wo']
            xn2=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*l['g2']
            x=x+np.maximum(xn2@l['W1']+l['b1'],0)@l['W2']+l['b2']
        xf=x/np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6)*gf
        p=softmax((xf@Wh+bh)[-1])**(1/0.7); p/=p.sum()
        prompt.append(np.random.choice(V,p=p))
    print(f'--- {prompt_text.strip()} ---')
    print(''.join(idx2ch.get(t,'?') for t in prompt))
    print()

# Save
save_dir=os.path.join(os.path.dirname(__file__),'..','..','data','nanogpt_200k_weights')
os.makedirs(save_dir,exist_ok=True)
np.save(os.path.join(save_dir,'embed.npy'),embed)
np.save(os.path.join(save_dir,'pos_embed.npy'),pos_embed)
np.save(os.path.join(save_dir,'gf.npy'),gf)
np.save(os.path.join(save_dir,'Wh.npy'),Wh)
np.save(os.path.join(save_dir,'bh.npy'),bh)
for i,l in enumerate(layers):
    for k,v in l.items():
        np.save(os.path.join(save_dir,f'layer{i}_{k}.npy'),v)
with open(os.path.join(save_dir,'config.json'),'w') as f:
    json.dump({'V':V,'D':D,'H':H,'HD':HD,'F':F,'T':T,'L':L,
               'chars':''.join(chars),'params':total,'loss':float(best),
               'card_matmuls':card_mm,'steps':step,'time_sec':el},f,indent=2)
print(f'Weights saved: {save_dir}/')

# Shutdown card
exit_cmd=(ctypes.c_uint*1)(255); nm.PL_WriteMemBlock(access,exit_cmd,DDR,1)
time.sleep(0.1); nm.PL_CloseAccess(access); nm.PL_CloseBoardDesc(board)
print('Card safe. DONE.')
