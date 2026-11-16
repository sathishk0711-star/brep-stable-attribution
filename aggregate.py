import json,numpy as np,collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open('/home/claude/work/baselines/rows.json'))
M=['IG64','IG16','GradxInput','SmoothGrad','ExpectedGradients','GNNExplainer','Occlusion','Random']
ids=sorted({r['id'] for r in R}); n=len(ids)
def arr(k,f):
    d={(r['id'],r['method']):r for r in R};return np.array([ (f(d[(i,k)]) if d[(i,k)] is not None else np.nan) for i in ids],float)
rng=np.random.default_rng(20261103); B=rng.integers(0,n,(5000,n))
def ci(v):
    b=v[B].mean(1);return float(v.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))
out={'parts':n,'methods':{}}
score=lambda r:r['insertion_auc']-r['deletion_auc']
for k in M:
    dl=arr(k,lambda r:r['deletion_auc']);ins=arr(k,lambda r:r['insertion_auc']);sc=ins-dl
    o={'deletion':ci(dl),'insertion':ci(ins),'ins_minus_del':ci(sc),'mass':ci(arr(k,lambda r:r['mass'])),'seconds':float(arr(k,lambda r:r['seconds']).mean()),
       'cos_ig64':float(np.nanmean(arr(k,lambda r:r['cos_ig64']))),'spearman_ig64':float(np.nanmean(arr(k,lambda r:r['spearman_ig64'] if r['spearman_ig64'] is not None else np.nan)))}
    c=arr(k,lambda r:r['completeness_abs'] if r['completeness_abs'] is not None else np.nan)
    o['completeness']=None if np.isnan(c).all() else float(c.mean())
    ig=arr('IG64',score);o['paired_vs_IG64_score']=ci(sc-ig)
    o['paired_vs_IG64_deletion']=ci(dl-arr('IG64',lambda r:r['deletion_auc']));o['paired_vs_IG64_insertion']=ci(ins-arr('IG64',lambda r:r['insertion_auc']))
    rn=arr('Random',score);o['paired_vs_Random_score']=ci(sc-rn)
    out['methods'][k]=o
json.dump(out,open('/home/claude/work/baselines/summary.json','w'),indent=1)
for k in M:
    o=out['methods'][k];print(f"{k:18s} del {o['deletion'][0]:7.3f} ins {o['insertion'][0]:7.3f} score {o['ins_minus_del'][0]:6.3f} vsIG {o['paired_vs_IG64_score'][0]:6.3f} [{o['paired_vs_IG64_score'][1]:.3f},{o['paired_vs_IG64_score'][2]:.3f}] vsRand {o['paired_vs_Random_score'][0]:6.3f} [{o['paired_vs_Random_score'][1]:.3f},{o['paired_vs_Random_score'][2]:.3f}] mass {o['mass'][0]:.3f} cos {o['cos_ig64']:.3f} t {o['seconds']:.3f}")
# figure
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9})
fig,ax=plt.subplots(1,2,figsize=(6.0,2.7),dpi=300)
names={'IG64':'IG (64 steps)','IG16':'IG (16 steps)','GradxInput':'Gradient×input','SmoothGrad':'SmoothGrad','ExpectedGradients':'Expected gradients','GNNExplainer':'GNNExplainer','Occlusion':'Occlusion','Random':'Random'}
for a,key,t in [(ax[0],'paired_vs_Random_score','(a) Gain over random ranking'),(ax[1],'paired_vs_IG64_score','(b) Difference from IG-64')]:
    ms=[k for k in M if not(key=='paired_vs_IG64_score' and k=='IG64') and not(key=='paired_vs_Random_score' and k=='Random')]
    for i,k in enumerate(ms):
        m,lo,hi=out['methods'][k][key];a.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#1f3a5f',capsize=2,ms=4)
    a.axvline(0,color='#888',lw=.8,ls='--');a.set_yticks(range(len(ms)));a.set_yticklabels([names[k] for k in ms] if a is ax[0] else [names[k] for k in ms]);a.invert_yaxis();a.set_title(t,fontsize=8.5);a.set_xlabel('Insertion − deletion AUC (logits)',fontsize=7.5)
plt.tight_layout();plt.savefig('/home/claude/work/baselines/fig_baselines.png');print('fig ok')
