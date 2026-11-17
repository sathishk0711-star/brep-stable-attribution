import json,numpy as np
from scipy.stats import spearmanr
S=[11,23,37,53,71];R={k:[] for k in ['rho','cos','norm_ratio','sum_ratio','neg_prop','neg_ig','top5','sign_agree','n']}
for s in S:
    rows=json.load(open(f'parts_stable_seed_{s}.json'));d={(r['id'],r['method']):r for r in rows}
    for i in sorted({r['id'] for r in rows}):
        a=np.array(d[(i,'proposed')]['attribution']);t=np.array(d[(i,'zero_ig')]['attribution'])
        if len(a)<6 or a.std()==0 or t.std()==0: continue
        R['rho'].append(spearmanr(a,t)[0]);R['cos'].append(a@t/np.linalg.norm(a)/np.linalg.norm(t))
        R['norm_ratio'].append(np.linalg.norm(a)/np.linalg.norm(t));R['sum_ratio'].append(a.sum()/t.sum() if abs(t.sum())>1e-6 else np.nan)
        R['neg_prop'].append((a<0).mean());R['neg_ig'].append((t<0).mean())
        k=5;R['top5'].append(len(set(np.argsort(-a)[:k])&set(np.argsort(-t)[:k]))/k)
        R['sign_agree'].append((np.sign(a)==np.sign(t)).mean());R['n'].append(len(a))
out={k:[float(np.nanmean(v)),float(np.nanmedian(v)),float(np.nanpercentile(v,10)),float(np.nanpercentile(v,90))] for k,v in R.items()}
out['n_pairs']=len(R['rho']);json.dump(out,open('diagnostics.json','w'),indent=1)
for k,v in out.items():print(k,np.round(v,3) if isinstance(v,list) else v)
