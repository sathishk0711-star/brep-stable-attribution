import json,glob,numpy as np,os
M=['zero_ig','mean_ig','learned_ig','proposed']
seeds=[];data={}
for f in sorted(glob.glob('/home/claude/runs/aagnet_full_seed_*/evaluation/parts.json')):
    s=int(f.split('seed_')[1].split('/')[0]);rows=json.load(open(f))
    ids=sorted({r['id'] for r in rows if sum(1 for q in rows if q['id']==r['id'])==4})
    if len(ids)<64: print('seed',s,'incomplete',len(ids));continue
    d={(r['id'],r['method']):r for r in rows};data[s]=(ids,d);seeds.append(s)
print('seeds',seeds)
rng=np.random.default_rng(20261104)
def mat(f,m):  # seeds x parts
    return np.array([[f(data[s][1][(i,m)]) for i in data[s][0]] for s in seeds],float)
sc=lambda r:r['insertion_auc']-r['deletion_auc']
def boot(v,B=5000):
    S,N=v.shape;out=[]
    for _ in range(B):
        si=rng.integers(0,S,S);pi=rng.integers(0,N,N);out.append(v[np.ix_(si,pi)].mean())
    return float(v.mean()),float(np.quantile(out,.025)),float(np.quantile(out,.975))
res={'seeds':seeds,'methods':{}}
for m in M:
    o={'deletion':float(mat(lambda r:r['deletion_auc'],m).mean()),'insertion':float(mat(lambda r:r['insertion_auc'],m).mean()),'score':boot(mat(sc,m)),
       'completeness':float(mat(lambda r:r['completeness_abs'],m).mean()),'mass':float(mat(lambda r:r['semantic_mass_alignment'],m).mean()),'seconds':float(mat(lambda r:r['seconds'],m).mean())}
    for ref in ['zero_ig','mean_ig','learned_ig']:
        if m!=ref: o['vs_'+ref]=boot(mat(sc,m)-mat(sc,ref))
    res['methods'][m]=o
st=mat(lambda r:np.mean(r['reference_normalized_l1']) if r['reference_normalized_l1'] else np.nan,'proposed');res['proposed_noise_l1']=float(np.nanmean(st))
iou=mat(lambda r:r['foreground_conditional_instance_iou'] if r['foreground_conditional_instance_iou'] is not None else np.nan,'proposed');res['proposed_instance_iou']=float(np.nanmean(iou))
json.dump(res,open('/home/claude/runs/full_summary.json','w'),indent=1)
for m,o in res['methods'].items():print(m,{k:(np.round(v,3).tolist() if not isinstance(v,float) else round(v,3)) for k,v in o.items()})
print('noise_l1',res['proposed_noise_l1'],'iou',res['proposed_instance_iou'])
