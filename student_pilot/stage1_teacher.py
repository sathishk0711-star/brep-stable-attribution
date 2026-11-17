"""Stage 1: IG64 teacher attributions + endpoint predictions for training/validation parts (AAGNet predictor, zero reference)."""
import sys,json,time,numpy as np,pickle
sys.path[:0]=['/home/claude/shim','/home/claude/pkg/ACD_Compact_Implementation_MFInstSeg_224/validation']
import torch,shared as S
torch.set_num_threads(4)
m=S.AAGNetSegmentor(**S.CFG);m.load_state_dict(torch.load(S.ROOT/'AAGNet/weights/weight_on_MFInstseg.pth',map_location='cpu'));m.eval();S.m=m
for p in m.parameters():p.requires_grad_(False)
exec(open('run_baselines.py').read().split("pool=[")[0].split("keys=S.keys")[1].join(["keys=S.keys",""]) if False else "")
keys=S.keys; values=S.values; put=S.put
def F(g,c):return m(g)[0][:,c].mean()
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
out=[];T0=time.time()
for split in('train','val'):
    for n in S.sel['selected'][split]:
        n,g,y,inst,bot=S.load(n);x=values(g);b=[torch.zeros_like(v) for v in x]
        with torch.no_grad():
            put(g,x);lg=m(g)[0];pred=lg.argmax(-1)
            cnt=torch.bincount(pred,minlength=25);cnt[24]=0
            cls=[int(c) for c in torch.argsort(cnt,descending=True)[:3] if cnt[c]>0]
        for c in cls:
            a=ig(g,x,b,c,64).detach()
            with torch.no_grad():
                put(g,b);fb=float(F(g,c));put(g,x);fx=float(F(g,c))
            out.append(dict(split=split,id=n,c=c,a=a.numpy(),fx=fx,fb=fb))
        print(split,n,len(out),round(time.time()-T0),flush=True)
        pickle.dump(out,open('teacher.pkl','wb'))
print('DONE')
