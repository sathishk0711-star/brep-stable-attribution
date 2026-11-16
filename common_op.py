import sys,json,glob,numpy as np,torch
sys.path[:0]=['/home/claude/shim','/home/claude/pkg/ACD_Compact_Implementation_MFInstSeg_224/proposed','/home/claude/pkg/ACD_Compact_Implementation_MFInstSeg_224/AAGNet','/home/claude/pkg/ACD_Compact_Implementation_MFInstSeg_224/validation']
import shared
from framework import Proposed,inputs,put
from models.inst_segmentors import AAGNetSegmentor
torch.set_num_threads(2)
D={k:[shared.load(n) for n in ns] for k,ns in shared.sel['selected'].items()}
test={n:(g,y) for n,g,y,_,_ in D['test']}
out={}
for seed in [11,23,53]:
    st=torch.load(f'/home/claude/runs_stable/aagnet_full_seed_{seed}/best.pt',map_location='cpu',weights_only=False)
    base=AAGNetSegmentor(**shared.CFG);net=Proposed(base,D['train'][0][1].ndata['grid'].shape[1:]);net.load_state_dict(st['model']);net.eval()
    rows=json.load(open(f'/home/claude/runs_stable/aagnet_full_seed_{seed}/evaluation/parts.json'))
    res=[]
    for r in rows:
        g,y=test[r['id']];c=r['target_class'];x=[v.detach().clone() for v in inputs(g)];b=[torch.zeros_like(v) for v in x];a=torch.tensor(r['attribution']);order=a.argsort(descending=True);s,d=g.edges();dl=[];ins=[]
        with torch.no_grad():
            for f in np.linspace(0,1,11):
                m=torch.zeros(len(a),dtype=torch.bool);m[order[:round(f*len(a))]]=True;em=m[s]|m[d]
                vd=[v.clone() for v in x];vi=[v.clone() for v in b]
                for j,mm in enumerate([m,m,em]):vd[j][mm]=b[j][mm];vi[j][mm]=x[j][mm]
                put(g,vd);dl.append(float(base(g)[0][:,c].mean()));put(g,vi);ins.append(float(base(g)[0][:,c].mean()))
            put(g,x)
        res.append(dict(id=r['id'],method=r['method'],deletion_auc=float(np.trapezoid(dl,dx=.1)),insertion_auc=float(np.trapezoid(ins,dx=.1))))
    out[seed]=res;print('seed',seed,len(res),flush=True)
json.dump(out,open('/home/claude/runs_stable/common_operator.json','w'))
