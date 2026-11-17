"""Stage 2: amortized attribution student (no path gradients at inference) distilled from IG64; evaluated on the 64 held-out parts."""
import sys,json,time,pickle,numpy as np
sys.path[:0]=['/home/claude/shim','/home/claude/pkg/ACD_Compact_Implementation_MFInstSeg_224/validation']
import torch,shared as S
torch.set_num_threads(4)
m=S.AAGNetSegmentor(**S.CFG);m.load_state_dict(torch.load(S.ROOT/'AAGNet/weights/weight_on_MFInstseg.pth',map_location='cpu'));m.eval();S.m=m
for p in m.parameters():p.requires_grad_(False)
keys=S.keys;values=S.values;put=S.put
def F(g,c):return m(g)[0][:,c].mean()
def feats(g,x,c):
    """Inference-time inputs: face/edge descriptors, neighbourhood means, class code and the predictor's own face logits. No gradients."""
    N=x[0].shape[0];s,t=g.edges()
    with torch.no_grad():
        put(g,x);lg=m(g)[0][:,c]
    gm=x[1].flatten(2).mean(2);gs=x[1].flatten(2).std(2)
    f=torch.cat([x[0],gm,gs],1)
    nb=torch.zeros_like(f).index_add(0,t,f[s]);deg=torch.zeros(N).index_add(0,t,torch.ones(len(s))).clamp(min=1)[:,None];nb=nb/deg
    ef=torch.zeros(N,x[2].shape[1]).index_add(0,t,x[2])/deg
    oh=torch.zeros(N,25);oh[:,c]=1
    return torch.cat([f,nb,ef,oh,lg[:,None],lg.mean().expand(N)[:,None],torch.full((N,1),float(np.log(N)))],1)
T=pickle.load(open('teacher.pkl','rb'))
cache={}
def load(n):
    if n not in cache: cache[n]=S.load(n)
    return cache[n]
def build(rows):
    X=[];Y=[];sc=[]
    for r in rows:
        n,g,y,_,_=load(r['id']);x=values(g);f=feats(g,x,r['c']);s=abs(r['fx']-r['fb'])+1.0
        X.append(f);Y.append(torch.tensor(r['a']/s,dtype=torch.float32))
    return torch.cat(X),torch.cat(Y)
tr=[r for r in T if r['split']=='train'];va=[r for r in T if r['split']=='val']
Xtr,Ytr=build(tr);Xva,Yva=build(va)
mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
print('train faces',len(Xtr),'val faces',len(Xva),flush=True)
tg=json.load(open(S.ROOT/'results/fixed_test_targets.json'))
test=[S.load(n) for n in S.sel['selected']['test']]
fr=np.linspace(0,1,11)
def curves(g,x,b,order):
    s,t=g.edges();N=len(order);dl=[];ins=[]
    with torch.no_grad():
        for f in fr:
            ix=order[:round(N*f)];sel=torch.zeros(N,dtype=torch.bool);sel[ix]=True;inc=sel[s]|sel[t]
            vd=[q.clone() for q in x];vi=[q.clone() for q in b]
            for j in(0,1):vd[j][sel]=b[j][sel];vi[j][sel]=x[j][sel]
            vd[2][inc]=b[2][inc];vi[2][inc]=x[2][inc]
            put(g,vd);dl.append(float(F(g,c_)));put(g,vi);ins.append(float(F(g,c_)))
    put(g,x);return dl,ins
def grads(g,vs,c):
    vs=[v.detach().requires_grad_() for v in vs];put(g,vs);return torch.autograd.grad(F(g,c),vs)
def reduce(g,at):
    f=at[0].flatten(1).sum(1)+at[1].flatten(1).sum(1);s,t=g.edges();e=at[2].sum(-1)
    return f.index_add(0,s,e*.5).index_add(0,t,e*.5)
def ig(g,x,b,c,steps):
    acc=[torch.zeros_like(v) for v in x]
    for j in range(steps+1):
        w=.5 if j in(0,steps) else 1.;d=grads(g,[q+(p-q)*j/steps for p,q in zip(x,b)],c);acc=[a+w*k for a,k in zip(acc,d)]
    return reduce(g,[(p-q)*a/steps for p,q,a in zip(x,b,acc)])
from scipy.stats import spearmanr
REF={}
for n,g,y,inst,bot in test:
    x=values(g);REF[n]=ig(g,x,[torch.zeros_like(v) for v in x],int(tg[n]),64).detach().numpy();put(g,x)
print('ref ok',flush=True)
rows=[]
for seed in [11,23,37,53,71]:
    torch.manual_seed(seed);np.random.seed(seed)
    net=torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1],128),torch.nn.GELU(),torch.nn.Linear(128,128),torch.nn.GELU(),torch.nn.Linear(128,1))
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);best=(1e9,None,0)
    Xt=((Xtr-mu)/sd).clamp(-5,5);Xv=((Xva-mu)/sd).clamp(-5,5)
    for ep in range(600):
        net.train();perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),256):
            ix=perm[i:i+256];loss=((net(Xt[ix])[:,0]-Ytr[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():v=float(((net(Xv)[:,0]-Yva)**2).mean())
        if v<best[0]:best=(v,{k:w.clone() for k,w in net.state_dict().items()},ep)
        elif ep-best[2]>40:break
    net.load_state_dict(best[1]);net.eval();print('seed',seed,'val mse',best[0],'epoch',best[2],flush=True)
    for n,g,y,inst,bot in test:
        c_=int(tg[n]);x=values(g);b=[torch.zeros_like(v) for v in x]
        t0=time.time()
        with torch.no_grad():
            f=((feats(g,x,c_)-mu)/sd).clamp(-5,5);put(g,x);fx=float(F(g,c_));put(g,b);fb=float(F(g,c_));put(g,x)
            s=abs(fx-fb)+1.0;raw=net(f)[:,0]*s
        t_raw=time.time()-t0
        proj=raw+(fx-fb-raw.sum())/len(raw)
        t_proj=time.time()-t0
        ref=REF[n]
        tgt=(y==c_)
        for nm,a,el in(('Student',raw,t_raw),('Student+projection',proj,t_proj)):
            dl,ins=curves(g,x,b,torch.argsort(a,descending=True))
            rows.append(dict(seed=seed,id=n,method=nm,seconds=el,deletion_auc=float(np.trapezoid(dl,fr)),insertion_auc=float(np.trapezoid(ins,fr)),
              completeness_abs=abs(float(a.sum())-(fx-fb)),cos_ig64=float(a.numpy()@ref/(np.linalg.norm(a.numpy())*np.linalg.norm(ref)+1e-12)),spearman_ig64=float(spearmanr(a.numpy(),ref).statistic),mae_ig64=float(np.abs(a.numpy()-ref).mean()),mass=float(a.abs()[tgt].sum()/(a.abs().sum()+1e-12))))
    json.dump(rows,open('student_rows5c.json','w'))
print('DONE')
