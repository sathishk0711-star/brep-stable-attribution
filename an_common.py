import json,numpy as np
d=json.load(open('runs_stable/common_operator.json'));M=['zero_ig','mean_ig','learned_ig','proposed'];seeds=sorted(d)
ids=sorted({r['id'] for r in d[seeds[0]]})
def mat(m,key):
    return np.array([[ {(r['id'],r['method']):r for r in d[s]}[(i,m)][key] for i in ids] for s in seeds])
sc=lambda m:mat(m,'insertion_auc')-mat(m,'deletion_auc')
rng=np.random.default_rng(20261104)
def boot(v):
    S,N=v.shape;o=[v[np.ix_(rng.integers(0,S,S),rng.integers(0,N,N))].mean() for _ in range(5000)];return [float(v.mean()),float(np.quantile(o,.025)),float(np.quantile(o,.975))]
res={}
for m in M:
    res[m]={'deletion':float(mat(m,'deletion_auc').mean()),'insertion':float(mat(m,'insertion_auc').mean()),'score':boot(sc(m))}
    for ref in ['zero_ig','mean_ig','learned_ig']:
        if m!=ref: res[m]['vs_'+ref]=boot(sc(m)-sc(ref))
json.dump(res,open('runs_stable/common_summary.json','w'),indent=1)
for m,o in res.items():print(m,{k:np.round(v,2).tolist() for k,v in o.items()})
