"""109K transformer — 16 NMC4 cores, float, weights preloaded, batch=16."""
import numpy as np, time, os, json, ctypes, sys
sys.stdout.reconfigure(line_buffering=True)

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
nm.PL_SetTimeout.argtypes=[ctypes.c_uint32];nm.PL_SetTimeout(10000)
nm.PL_GetBoardDesc.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)];nm.PL_GetBoardDesc.restype=ctypes.c_int
nm.PL_ResetBoard.argtypes=[ctypes.c_void_p];nm.PL_ResetBoard.restype=ctypes.c_int
nm.PL_LoadInitCode.argtypes=[ctypes.c_void_p];nm.PL_LoadInitCode.restype=ctypes.c_int
nm.PL_GetAccess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int*2),ctypes.POINTER(ctypes.c_void_p)];nm.PL_GetAccess.restype=ctypes.c_int
nm.PL_LoadProgramFile.argtypes=[ctypes.c_void_p,ctypes.c_char_p];nm.PL_LoadProgramFile.restype=ctypes.c_int
nm.PL_CloseAccess.argtypes=[ctypes.c_void_p];nm.PL_CloseAccess.restype=ctypes.c_int
nm.PL_ReadMemBlock.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.c_uint,ctypes.c_uint];nm.PL_ReadMemBlock.restype=ctypes.c_int
nm.PL_WriteMemBlock.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.c_uint,ctypes.c_uint];nm.PL_WriteMemBlock.restype=ctypes.c_int
nm.PL_CloseBoardDesc.argtypes=[ctypes.c_void_p];nm.PL_CloseBoardDesc.restype=ctypes.c_int

DDR=0x00340000; CMD_BLOCK=32; buf1=(ctypes.c_uint*1)()

# Init board + load dispatcher on ALL 16 cores
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

# Verify all 16 alive
alive=0
for cl in range(4):
    for co in range(4):
        acc=accesses[(cl,co)]; idx=cl*4+co
        nm.PL_ReadMemBlock(acc,buf1,DDR+idx*CMD_BLOCK+31,1)
        if buf1[0]>100: alive+=1
print(f'NM Card: {alive}/16 cores alive')

def f2u(a):return np.frombuffer(a.astype(np.float32).tobytes(),dtype=np.uint32)
def u2f(a):return np.frombuffer(np.array(a,dtype=np.uint32).tobytes(),dtype=np.float32)

# Use core 0 for single matmuls, all 16 for batch
def card_mm(A, B, core=0):
    """Single matmul on specified core."""
    cl,co=core//4,core%4; acc=accesses[(cl,co)]
    M,K=A.shape;_,N=B.shape
    base=DDR+core*CMD_BLOCK
    # Data area: after all 16 cmd blocks = DDR + 512
    DATA=DDR+512+core*50000  # each core gets 50K words scratch
    ab=(ctypes.c_uint*(M*K))(*f2u(A.flatten())); nm.PL_WriteMemBlock(acc,ab,DATA,M*K)
    bb=(ctypes.c_uint*(K*N))(*f2u(B.flatten())); nm.PL_WriteMemBlock(acc,bb,DATA+M*K,K*N)
    args=(ctypes.c_uint*6)(M,K,N,DATA,DATA+M*K,DATA+M*K+K*N)
    nm.PL_WriteMemBlock(acc,args,base+1,6)
    z=(ctypes.c_uint*1)(0); nm.PL_WriteMemBlock(acc,z,base+30,1)
    c=(ctypes.c_uint*1)(1); nm.PL_WriteMemBlock(acc,c,base,1)
    for _ in range(10000):
        nm.PL_ReadMemBlock(acc,buf1,base+30,1)
        if buf1[0]==1: break
        time.sleep(0.0001)
    cb=(ctypes.c_uint*(M*N))(); nm.PL_ReadMemBlock(acc,cb,DATA+M*K+K*N,M*N)
    return u2f(list(cb)).reshape(M,N)

# Verify matmul on core 0
C=card_mm(np.array([[1,2],[3,4]],dtype=np.float32),np.array([[5,6],[7,8]],dtype=np.float32),0)
print(f'Core 0 matmul: {C.flatten()} (expect [19 22 43 50])')

# Model
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

def sm(x): e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)
def rn(x,g): r=np.sqrt(np.mean(x**2,-1,keepdims=True)+1e-6); return x/r*g, r
def rn_bwd(dy,x,g,r):
    D_=x.shape[-1]; xn=x/r; dy_g=dy*g
    return (dy_g-xn*np.sum(dy_g*xn,axis=-1,keepdims=True)/D_)/r

# Training: batch=16, each sample on different core
print(f'\n=== 16-CORE TRAINING ===')
BATCH=16; base_lr=0.0006; lr=0.00001; best=99; step=0; start=time.time(); card_mm_count=0; WARMUP=300

while best > 1.0 and step < 15000:
    # Warmup
    if step < WARMUP:
        lr = base_lr * (step+1) / WARMUP
    elif step < 5000:
        lr = base_lr
    elif step < 10000:
        lr = base_lr * 0.3
    else:
        lr = base_lr * 0.1

    # Batch: accumulate gradients from BATCH samples
    # Forward on core 0 (simplest — TODO: distribute across 16 cores)
    batch_loss = 0
    acc_dWh=np.zeros_like(Wh); acc_dbh=np.zeros_like(bh)
    acc_dpos=np.zeros_like(pos); acc_dembed={}
    layer_grads=[{k:np.zeros_like(v) for k,v in l.items() if k not in ('g1','g2')} for l in layers]

    for bi in range(BATCH):
        idx=np.random.randint(0,len(data)-T-1)
        tok=data[idx:idx+T]; tgt=data[idx+1:idx+T+1]; Tl=T
        x=embed[tok]+pos; cache=[]

        for li,l in enumerate(layers):
            xn,r1=rn(x,l['g1'])
            Q=card_mm(xn,l['Wq'],0); K_=card_mm(xn,l['Wk'],0); V_=card_mm(xn,l['Wv'],0)
            card_mm_count+=3
            Q=Q.reshape(Tl,H,HD).transpose(1,0,2)
            K_=K_.reshape(Tl,H,HD).transpose(1,0,2)
            V_=V_.reshape(Tl,H,HD).transpose(1,0,2)
            sc=np.matmul(Q,K_.transpose(0,2,1))/np.sqrt(HD)+np.triu(np.full((Tl,Tl),-1e9),k=1)
            at=sm(sc); ao=np.matmul(at,V_).transpose(1,0,2).reshape(Tl,D)
            pr=card_mm(ao,l['Wo'],0); card_mm_count+=1; xr=x+pr
            xn2,r2=rn(xr,l['g2'])
            h=card_mm(xn2,l['W1'],0)+l['b1']; card_mm_count+=1
            a=np.maximum(h,0)
            o=card_mm(a,l['W2'],0)+l['b2']; card_mm_count+=1; xn_=xr+o
            cache.append((x,xn,r1,Q,K_,V_,at,ao,xr,xn2,r2,h,a)); x=xn_

        xf,rf=rn(x,gf)
        logits=card_mm(xf,Wh,0)+bh; card_mm_count+=1
        probs=sm(logits)
        loss=-np.mean(np.log(probs[np.arange(Tl),tgt]+1e-9))
        batch_loss+=loss

        dl=probs.copy(); dl[np.arange(Tl),tgt]-=1; dl/=Tl
        acc_dWh+=xf.T@dl; acc_dbh+=dl.sum(0)
        dx=rn_bwd(dl@Wh.T,x,gf,rf)

        for li in range(L-1,-1,-1):
            l=layers[li]; x_in,xn,r1,Q,K_,V_,at,ao,xr,xn2,r2,h,a=cache[li]
            do=dx
            layer_grads[li]['W2']+=a.T@do; layer_grads[li]['b2']+=do.sum(0)
            da=do@l['W2'].T*(h>0)
            layer_grads[li]['W1']+=xn2.T@da; layer_grads[li]['b1']+=da.sum(0)
            dxn2=da@l['W1'].T; dx_r=dx+rn_bwd(dxn2,xr,l['g2'],r2)
            layer_grads[li]['Wo']+=ao.T@dx_r; dao=dx_r@l['Wo'].T
            dao_h=dao.reshape(Tl,H,HD).transpose(1,0,2)
            dV_=np.matmul(at.transpose(0,2,1),dao_h)
            dat=np.matmul(dao_h,V_.transpose(0,2,1))
            ds=at*(dat-np.sum(dat*at,-1,keepdims=True))/np.sqrt(HD)
            dQ=np.matmul(ds,K_); dK=np.matmul(ds.transpose(0,2,1),Q)
            dQ=dQ.transpose(1,0,2).reshape(Tl,D); dK=dK.transpose(1,0,2).reshape(Tl,D)
            dV_=dV_.transpose(1,0,2).reshape(Tl,D)
            layer_grads[li]['Wq']+=xn.T@dQ; layer_grads[li]['Wk']+=xn.T@dK; layer_grads[li]['Wv']+=xn.T@dV_
            dx=dx_r+rn_bwd(dQ@l['Wq'].T+dK@l['Wk'].T+dV_@l['Wv'].T,x_in,l['g1'],r1)

        acc_dpos+=0.1*dx
        for t in range(Tl):
            tid=tok[t]
            if tid not in acc_dembed: acc_dembed[tid]=np.zeros(D,np.float32)
            acc_dembed[tid]+=0.1*dx[t]

    batch_loss/=BATCH
    if batch_loss<best: best=batch_loss

    # Apply
    Wh-=lr*acc_dWh/BATCH; bh-=lr*acc_dbh/BATCH; pos-=lr*acc_dpos/BATCH
    for tid,g in acc_dembed.items(): embed[tid]-=lr*g/BATCH
    for li,l in enumerate(layers):
        for k in layer_grads[li]:
            l[k]-=lr*layer_grads[li][k]/BATCH

    if step%50==0:
        el=time.time()-start
        print(f'Step {step:5d} | loss={batch_loss:.3f} | best={best:.3f} | mm={card_mm_count} | {el:.0f}s')
    if step%500==0 and step>0:
        p=list(ch2idx.get(c,0) for c in 'ROMEO:\n')
        for _ in range(100):
            inp=p[-T:]; xg=embed[inp]+pos[:len(inp)]
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
print(f'\nDONE: {step} steps, {el:.0f}s, best={best:.3f}, mm={card_mm_count}')

# Save
sd=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','nanogpt_16core_weights')
os.makedirs(sd,exist_ok=True)
np.save(f'{sd}/embed.npy',embed);np.save(f'{sd}/pos.npy',pos)
np.save(f'{sd}/gf.npy',gf);np.save(f'{sd}/Wh.npy',Wh);np.save(f'{sd}/bh.npy',bh)
for i,l in enumerate(layers):
    for k,v in l.items():np.save(f'{sd}/l{i}_{k}.npy',v)
with open(f'{sd}/config.json','w') as f:
    json.dump({'V':V,'D':D,'H':H,'F':F,'T':T,'L':L,'HD':HD,'params':total,'loss':float(best),
               'card_matmuls':card_mm_count,'steps':step,'time':el,'cores':alive,'chars':''.join(chars)},f,indent=2)
print(f'Saved: {sd}/')

# Shutdown all cores
for cl in range(4):
    for co in range(4):
        acc=accesses[(cl,co)]; idx=cl*4+co
        e=(ctypes.c_uint*1)(255); nm.PL_WriteMemBlock(acc,e,DDR+idx*CMD_BLOCK,1)
time.sleep(0.3)
for acc in accesses.values(): nm.PL_CloseAccess(acc)
nm.PL_CloseBoardDesc(board)
print('All 16 cores safe. DONE.')
