"""Train 2-layer LM on NM Card Mini — MatMul on NMC4 hardware, save weights."""
import numpy as np, time, os, json, ctypes

# Data
with open(os.path.join(os.path.dirname(__file__),'..','..','data','tiny_shakespeare.txt'),'r') as f: text=f.read()
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

DDR=0x00340000; FIXED=0x10000
def arr_f2q(a): return (np.clip(a,-30000,30000)*FIXED).astype(np.int64).astype(np.uint32)
def arr_q2f(a):
    s=a.astype(np.int64); s[s>0x7FFFFFFF]-=0x100000000
    return (s/FIXED).astype(np.float32)

print('=== Init NM Card Mini ===')
board=ctypes.c_void_p(); nm.PL_GetBoardDesc(0,ctypes.byref(board))
nm.PL_ResetBoard(board); nm.PL_LoadInitCode(board)
core_no=(ctypes.c_int*2)(0,0); access=ctypes.c_void_p()
nm.PL_GetAccess(board,ctypes.byref(core_no),ctypes.byref(access))
# CRITICAL: clear DDR before loading dispatcher to prevent stale cmd execution
zeros=(ctypes.c_uint*1024)(*([0]*1024))
nm.PL_WriteMemBlock(access,zeros,DDR,1024)
disp=os.path.join(os.path.dirname(__file__),'..','..','aten','src','ATen','nmcard','nmc_programs','dispatcher.abs')
nm.PL_LoadProgramFile(access,disp.encode())
time.sleep(2)
buf1=(ctypes.c_uint*1)()
nm.PL_ReadMemBlock(access,buf1,DDR+31,1)
print(f'Core 0 watchdog: {buf1[0]} (alive)')

card_mm=0
def float_as_uint32(arr):
    """IEEE 754 float32 → uint32 bit-cast (dispatcher expects this!)"""
    return np.frombuffer(arr.astype(np.float32).tobytes(), dtype=np.uint32)
def uint32_as_float(arr):
    """uint32 → IEEE 754 float32 bit-cast"""
    return np.frombuffer(np.array(arr, dtype=np.uint32).tobytes(), dtype=np.float32)

def nmcard_matmul(A, B):
    global card_mm; card_mm+=1
    M,K=A.shape; _,N=B.shape
    Au=float_as_uint32(A.flatten()); Bu=float_as_uint32(B.flatten())
    DATA=DDR+64; a_off=0; b_off=M*K; c_off=M*K+K*N
    a_buf=(ctypes.c_uint*len(Au))(*Au); nm.PL_WriteMemBlock(access,a_buf,DATA+a_off,len(Au))
    b_buf=(ctypes.c_uint*len(Bu))(*Bu); nm.PL_WriteMemBlock(access,b_buf,DATA+b_off,len(Bu))
    args=(ctypes.c_uint*6)(M,K,N,DATA+a_off,DATA+b_off,DATA+c_off)
    nm.PL_WriteMemBlock(access,args,DDR+1,6)
    zero=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(access,zero,DDR+30,1)
    cmd=(ctypes.c_uint*1)(1); nm.PL_WriteMemBlock(access,cmd,DDR,1)
    for _ in range(500):
        nm.PL_ReadMemBlock(access,buf1,DDR+30,1)
        if buf1[0]==1:
            # CRITICAL: reset cmd to NOP so dispatcher doesn't re-execute
            nop=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(access,nop,DDR,1)
            break
        time.sleep(0.002)
    else:
        nop=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(access,nop,DDR,1)
        print('TIMEOUT! Fallback CPU'); return A@B
    c_buf=(ctypes.c_uint*(M*N))(); nm.PL_ReadMemBlock(access,c_buf,DATA+c_off,M*N)
    return uint32_as_float(list(c_buf)).reshape(M,N)

# Verify
print('\n=== Card MatMul Test ===')
A=np.array([[1,2],[3,4]],dtype=np.float32); B=np.array([[5,6],[7,8]],dtype=np.float32)
C=nmcard_matmul(A,B); print(f'Card: {C.flatten()} Expected: [19 22 43 50]')

# Model
D,T,F=32,16,64; np.random.seed(42)
embed=np.random.randn(V,D).astype(np.float32)*0.05
pos=np.random.randn(T,D).astype(np.float32)*0.02
W1=np.random.randn(D,F).astype(np.float32)*np.sqrt(2/D); b1=np.zeros(F,dtype=np.float32)
W2=np.random.randn(F,D).astype(np.float32)*np.sqrt(2/F); b2=np.zeros(D,dtype=np.float32)
W3=np.random.randn(D,F).astype(np.float32)*np.sqrt(2/D); b3=np.zeros(F,dtype=np.float32)
W4=np.random.randn(F,D).astype(np.float32)*np.sqrt(2/F); b4=np.zeros(D,dtype=np.float32)
Wh=np.random.randn(D,V).astype(np.float32)*np.sqrt(2/D); bh=np.zeros(V,dtype=np.float32)
g1=np.ones(D,dtype=np.float32); g2=np.ones(D,dtype=np.float32)
total=sum(p.size for p in [embed,pos,W1,b1,W2,b2,W3,b3,W4,b4,Wh,bh,g1,g2])
print(f'\nModel: {total:,} params, D={D}, F={F}, T={T}')

def softmax(x):
    e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def rms_norm(x,g,eps=1e-5): return x/np.sqrt(np.mean(x**2,-1,keepdims=True)+eps)*g
def relu(x): return np.maximum(x,0)
def drelu(x): return (x>0).astype(np.float32)

# Training
print('\n=== TRAINING (MatMul on NMC4) ===')
lr=0.05; steps=6000; start=time.time(); best=99

for step in range(steps):
    idx=np.random.randint(0,len(data)-T-1)
    tok=data[idx:idx+T]; tgt=data[idx+1:idx+T+1]
    x0=embed[tok]+pos
    n1=rms_norm(x0,g1)
    h1=nmcard_matmul(n1,W1)+b1  # CARD!
    a1=relu(h1); o1=a1@W2+b2; x1=x0+o1
    n2=rms_norm(x1,g2)
    h2=nmcard_matmul(n2,W3)+b3  # CARD!
    a2=relu(h2); o2=a2@W4+b4; x2=x1+o2
    logits=x2@Wh+bh
    probs=softmax(logits)
    loss=-np.mean(np.log(probs[np.arange(T),tgt]+1e-9))
    if loss<best: best=loss

    dl=probs.copy(); dl[np.arange(T),tgt]-=1; dl/=T
    dWh=x2.T@dl; dbh=dl.sum(0); dx2=dl@Wh.T
    da2=(dx2@W4.T)*drelu(h2); dW4=a2.T@dx2; db4=dx2.sum(0)
    dW3=n2.T@da2; db3=da2.sum(0)
    da1=(dx2@W2.T)*drelu(h1); dW2=a1.T@dx2; db2=dx2.sum(0)
    dW1=n1.T@da1; db1=da1.sum(0)
    dx0=dx2

    Wh-=lr*dWh; bh-=lr*dbh; W4-=lr*dW4; b4-=lr*db4
    W3-=lr*dW3; b3-=lr*db3; W2-=lr*dW2; b2-=lr*db2
    W1-=lr*dW1; b1-=lr*db1; pos-=lr*dx0
    for t in range(T): embed[tok[t]]-=lr*dx0[t]

    if step%50==0 or step==steps-1:
        el=time.time()-start
        print(f'Step {step:4d} | loss={loss:.3f} | best={best:.3f} | card_mm={card_mm} | {el:.1f}s')

# Generate
print('\n=== Shakespeare (trained on NM Card Mini) ===')
prompt=list(ch2idx.get(c,0) for c in 'ROMEO:\n')
for _ in range(300):
    inp=prompt[-T:]
    x0=embed[inp]+pos[:len(inp)]
    n1=rms_norm(x0,g1); x1=x0+relu(n1@W1+b1)@W2+b2
    n2=rms_norm(x1,g2); x2=x1+relu(n2@W3+b3)@W4+b4
    p=softmax((x2@Wh+bh)[-1])**(1/0.8); p/=p.sum()
    prompt.append(np.random.choice(V,p=p))
print(''.join(idx2ch.get(t,'?') for t in prompt))

# Save
save_dir=os.path.join(os.path.dirname(__file__),'..','..','data','nanogpt_nmcard_weights')
os.makedirs(save_dir,exist_ok=True)
for name,w in {'embed':embed,'pos':pos,'W1':W1,'b1':b1,'W2':W2,'b2':b2,
    'W3':W3,'b3':b3,'W4':W4,'b4':b4,'Wh':Wh,'bh':bh,'g1':g1,'g2':g2}.items():
    np.save(os.path.join(save_dir,f'{name}.npy'),w)
with open(os.path.join(save_dir,'config.json'),'w') as f:
    json.dump({'V':V,'D':D,'F':F,'T':T,'layers':2,'chars':''.join(chars),
               'params':total,'loss':float(best),'card_matmuls':card_mm},f,indent=2)
print(f'\nWeights saved: {save_dir}/')
print(f'Card matmuls: {card_mm}, Loss: {best:.3f}, Params: {total:,}')

# Shutdown
exit_cmd=(ctypes.c_uint*1)(255); nm.PL_WriteMemBlock(access,exit_cmd,DDR,1)
time.sleep(0.1); nm.PL_CloseAccess(access); nm.PL_CloseBoardDesc(board)
print('Card safely shut down. === DONE ===')
