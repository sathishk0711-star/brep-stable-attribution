"""Attribution baselines on the pretrained AAGNet predictor (64 held-out MFInstSeg parts, fixed targets)."""
import sys,json,time,numpy as np
sys.path[:0]=['/home/claude/shim','/home/claude/pkg/ACD_Compact_Implementation_MFInstSeg_224/validation']
import torch,shared as S
from scipy.stats import spearmanr
torch.set_num_threads(4)
m=S.AAGNetSegmentor(**S.CFG);m.load_state_dict(torch.load(S.ROOT/'AAGNet/weights/weight_on_MFInstseg.pth',map_location='cpu'));m.eval();S.m=m
for p in m.parameters():p.requires_grad_(False)
tg=json.load(open(S.ROOT/'results/fixed_test_targets.json'))
test=[S.load(n) for n in S.sel['selected']['test']]
train=[S.load(n) for n in S.sel['selected']['train']]
keys=S.keys; values=S.values; put=S.put
def F(g,c):return m(g)[0][:,c].mean()
def grads(g,vs,c):
    vs=[v.detach().requires_grad_() for v in vs];put(g,vs);return torch.autograd.grad(F(g,c),vs)
def reduce(g,at):
    f=at[0].flatten(1).sum(1)+at[1].flatten(1).sum(1);s,t=g.edges();e=at[2].sum(-1)
    return f.index_add(0,s,e*.5).index_add(0,t,e*.5)
pool=[torch.cat([values(g)[j] for _,g,*_ in train],0) for j in range(3)]
def zeros_like(x):return [torch.zeros_like(v) for v in x]
def ig(g,x,b,c,steps):
    acc=[torch.zeros_like(v) for v in x]
    for j in range(steps+1):
        w=.5 if j in(0,steps) else 1.;d=grads(g,[q+(p-q)*j/steps for p,q in zip(x,b)],c);acc=[a+w*k for a,k in zip(acc,d)]
    return reduce(g,[(p-q)*a/steps for p,q,a in zip(x,b,acc)])
def gxi(g,x,b,c):return reduce(g,[(p-q)*k for p,q,k in zip(x,b,grads(g,x,c))])
def smoothgrad(g,x,b,c,n=32,sig=.15,seed=0):
    r=torch.Generator().manual_seed(seed);acc=zeros_like(x)
    for _ in range(n):
        d=grads(g,[v+sig*torch.randn(v.shape,generator=r) for v in x],c);acc=[a+k/n for a,k in zip(acc,d)]
    return reduce(g,[(p-q)*a for p,q,a in zip(x,b,acc)])
def expgrad(g,x,b,c,n=32,seed=0):
    r=torch.Generator().manual_seed(seed);acc=zeros_like(x)
    for _ in range(n):
        al=torch.rand((),generator=r).item()
        ref=[pool[j][torch.randint(0,len(pool[j]),(x[j].shape[0],),generator=r)] for j in range(3)]
        d=grads(g,[q+al*(p-q) for p,q in zip(x,ref)],c);acc=[a+(p-q)*k/n for a,p,q,k in zip(acc,x,ref,d)]
    return reduce(g,acc)
def gnnexp(g,x,b,c,steps=100,seed=0):
    torch.manual_seed(seed);N=x[0].shape[0];s,t=g.edges();w=torch.nn.Parameter(torch.randn(N)*.1+1.0);opt=torch.optim.Adam([w],lr=.1)
    for _ in range(steps):
        mk=torch.sigmoid(w);me=(mk[s]*mk[t])[:,None];xs=[x[0]*mk[:,None],x[1]*mk[:,None,None,None],x[2]*me];put(g,xs)
        ent=-(mk*torch.log(mk+1e-8)+(1-mk)*torch.log(1-mk+1e-8)).mean()
        loss=-F(g,c)+.005*mk.sum()+1.0*ent;opt.zero_grad();loss.backward();opt.step()
    put(g,x);return torch.sigmoid(w).detach()
def occlusion(g,x,b,c):
    N=x[0].shape[0];s,t=g.edges();out=torch.zeros(N)
    with torch.no_grad():
        put(g,x);f0=float(F(g,c))
        for i in range(N):
            v=[q.clone() for q in x];v[0][i]=b[0][i];v[1][i]=b[1][i];inc=(s==i)|(t==i);v[2][inc]=b[2][inc];put(g,v);out[i]=f0-float(F(g,c))
    put(g,x);return out
def rand(g,x,b,c,seed=0):return torch.rand(x[0].shape[0],generator=torch.Generator().manual_seed(seed))
fr=np.linspace(0,1,11)
def curves(g,x,b,order):
    s,t=g.edges();N=len(order);dl=[];ins=[]
    with torch.no_grad():
        for f in fr:
            ix=order[:round(N*f)];sel=torch.zeros(N,dtype=torch.bool);sel[ix]=True;inc=sel[s]|sel[t]
            vd=[q.clone() for q in x];vi=[q.clone() for q in b]
            for j in(0,1):vd[j][sel]=b[j][sel];vi[j][sel]=x[j][sel]
            vd[2][inc]=b[2][inc];vi[2][inc]=x[2][inc]
            put(g,vd);dl.append(float(F(g,c)));put(g,vi);ins.append(float(F(g,c)))
    put(g,x);return dl,ins
METH={'IG16':lambda g,x,b,c:ig(g,x,b,c,16),'IG64':lambda g,x,b,c:ig(g,x,b,c,64),'GradxInput':gxi,'SmoothGrad':smoothgrad,'ExpectedGradients':expgrad,'GNNExplainer':gnnexp,'Occlusion':occlusion,'Random':rand}
rows=[];T0=time.time()
for n,g,y,inst,bot in test:
    c=int(tg[n]);x=values(g);b=zeros_like(x)
    with torch.no_grad():put(g,b);fb=float(F(g,c));put(g,x);fx=float(F(g,c))
    tgt=(y==c);res={}
    for k,f in METH.items():
        t0=time.time();a=f(g,x,b,c);el=time.time()-t0;a=a.detach()
        dl,ins=curves(g,x,b,torch.argsort(a,descending=True))
        res[k]=a
        rows.append(dict(id=n,method=k,target=c,faces=len(y),seconds=el,deletion_auc=float(np.trapezoid(dl,fr)),insertion_auc=float(np.trapezoid(ins,fr)),
            completeness_abs=abs(float(a.sum())-(fx-fb)) if k in('IG16','IG64','GradxInput','SmoothGrad','ExpectedGradients') else None,
            mass=float(a.abs()[tgt].sum()/(a.abs().sum()+1e-12)),target_frac=float(tgt.float().mean()),fx=fx,fb=fb))
    ref=res['IG64'].numpy()
    for r in rows[-len(METH):]:
        a=res[r['method']].numpy();r['cos_ig64']=float(a@ref/(np.linalg.norm(a)*np.linalg.norm(ref)+1e-12));r['spearman_ig64']=float(spearmanr(a,ref).statistic) if a.std()>0 else None
    print(n,len(y),round(time.time()-T0),flush=True)
    json.dump(rows,open('/home/claude/work/baselines/rows.json','w'))
print('DONE')
