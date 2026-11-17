import json,numpy as np,sys
R=json.load(open(sys.argv[1])); B=json.load(open('rows.json'))
ids=sorted({r['id'] for r in R}); seeds=sorted({r['seed'] for r in R})
def arr(m,k):
    M=np.full((len(seeds),len(ids)),np.nan)
    for r in R:
        if r['method']==m: M[seeds.index(r['seed']),ids.index(r['id'])]=r[k]
    return M
met={k:(arr('Student',k),arr('Student+projection',k)) for k in ['completeness_abs','mae_ig64','cos_ig64','mass','seconds']}
rng=np.random.default_rng(20261103)
def boot(D,n=5000):
    S,P=D.shape; o=[]
    for _ in range(n): o.append(D[np.ix_(rng.integers(0,S,S),rng.integers(0,P,P))].mean())
    return np.quantile(o,[.025,.975]).tolist()
res={}
for k,(a,b) in met.items():
    if k=='seconds': continue
    d=b-a; res[k]=dict(raw=float(a.mean()),proj=float(b.mean()),diff=float(d.mean()),ci=boot(d),raw_ci=boot(a),proj_ci=boot(b)); print(k,{x:(round(v,6) if not isinstance(v,list) else [round(q,6) for q in v]) for x,v in res[k].items()})
pp=(met['cos_ig64'][1]-met['cos_ig64'][0]).mean(0); pm=(met['mass'][1]-met['mass'][0]).mean(0); pe=(met['mae_ig64'][1]-met['mae_ig64'][0]).mean(0)
ig=[r for r in B if r['method']=='IG64']
sec=dict(raw=float(met['seconds'][0].mean()*1000),proj=float(met['seconds'][1].mean()*1000),ig64=float(np.mean([r['seconds'] for r in ig])*1000))
print('cos pos',int((pp>0).sum()),'mae neg',int((pe<0).sum()),'mass pos',int((pm>0).sum()),'ms',sec,'IG64 mass',np.mean([r['mass'] for r in ig]),'IG comp',np.mean([r['completeness_abs'] for r in ig]))
print('max raw completeness',met['completeness_abs'][0].max(), 'median', np.median(met['completeness_abs'][0]))
json.dump(dict(res=res,pp=pp.tolist(),pm=pm.tolist(),pe=pe.tolist(),ids=ids,sec=sec,ig_mass=float(np.mean([r['mass'] for r in ig])),ig_comp=float(np.mean([r['completeness_abs'] for r in ig])),pos_cos=int((pp>0).sum()),pos_mass=int((pm>0).sum()),neg_mae=int((pe<0).sum())),open('student5c_summary.json','w'))
