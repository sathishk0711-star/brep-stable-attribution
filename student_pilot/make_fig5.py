import json,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=json.load(open('/home/claude/work/baselines/student5c_summary.json'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,axs=plt.subplots(1,2,figsize=(6.6,2.9),gridspec_kw={'wspace':.12})
rng=np.random.default_rng(3)
for ax,vals,key,title in [(axs[0],np.array(S['pp']),'cos_ig64','Change in cosine agreement with IG64'),(axs[1],np.array(S['pm']),'mass','Change in target-class attribution mass')]:
    r=S['res'][key]; m=vals.mean(); lo,hi=r['ci']
    y=rng.uniform(-.28,.28,len(vals))
    ax.scatter(vals,y,s=14,color='#1f77b4',alpha=.75,edgecolor='#2b6ea3',lw=.4,zorder=3)
    ax.boxplot(vals,vert=False,positions=[0],widths=.62,showfliers=False,manage_ticks=False,medianprops=dict(color='#ff7f0e',lw=1.8),boxprops=dict(lw=1),whiskerprops=dict(lw=1),capprops=dict(lw=1),zorder=2)
    ax.scatter([m],[0],marker='D',s=38,color='#1f77b4',zorder=4)
    ax.errorbar(m,-.78,xerr=[[m-lo],[hi-m]],fmt='none',ecolor='#1f77b4',elinewidth=2,capsize=4,capthick=1.6)
    ax.axvline(0,color='#1f77b4',ls='--',lw=.9,zorder=1)
    ax.text(0.5,.93,f'Mean {m:+.4f}',transform=ax.transAxes,ha='center',fontsize=8.5,bbox=dict(fc='white',ec='none',pad=1.2))
    ax.text(0.5,.04,f'95% CI [{lo:+.4f}, {hi:+.4f}]',transform=ax.transAxes,ha='center',fontsize=8.5,bbox=dict(fc='white',ec='none',pad=1.2))
    ax.set_ylim(-1.05,.95); ax.set_yticks([]); ax.spines[['left','top','right']].set_visible(False)
    ax.grid(axis='x',color='#e6e6e6',lw=.6,zorder=0); ax.set_axisbelow(True)
    ax.set_title(title,fontsize=9.5); ax.set_xlabel('Calibrated − raw (per part, seed-averaged)',fontsize=8.8)
    ax.tick_params(axis='x',labelsize=8.5)
fig.subplots_adjust(left=.03,right=.99,bottom=.2,top=.9)
fig.savefig('fig5_new.png',dpi=600)
print([ (min(S['pp']),max(S['pp'])), (min(S['pm']),max(S['pm']))])
