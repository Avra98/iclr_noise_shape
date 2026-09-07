from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
R=Path(__file__).resolve().parents[1];a=R/'experiments/directional_ladders/anchor';t=dict(np.load(a/'trace.npz'));f=dict(np.load(a/'fresh_probes.npz'));alignment=json.loads((a/'eigenvector_alignment.json').read_text());loss={v['step']:v['loss'] for v in json.loads((a/'full_loss.json').read_text())}
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(3,2,figsize=(10,7.0));ax=axes[0,0];ax.plot(t['step'],t['batch_loss'],color='#b4cce0',lw=.5);ax.plot(list(loss),list(loss.values()),color='#185e9c',lw=1.6);ax.set(yscale='log',title='(a) Full-subset and minibatch loss',xlabel='Additional updates',ylabel='Squared loss')
ax=axes[0,1];ax.plot([v['step'] for v in alignment],[v['alignment'] for v in alignment],'o-',color='#b65d22',ms=3);ax.set(ylim=(0,1.04),title='(b) Leading-direction alignment',xlabel='Additional updates',ylabel=r'$|q_1(t)^\top q_*|$')
x=t['coordinates'][:,0]*1000
ax=axes[1,0];ax.hist(x[2048:2560],bins=30,density=True,color='#3988af');ax.set(title='(c) Fixed-axis occupancy: 512 updates',xlabel=r'$10^3q_*^\top(\theta_t-\theta_*)$',ylabel='Density')
ax=axes[1,1];ax.plot(t['step'][2048:2112],x[2048:2112],'.-',ms=2,lw=.65,color='#3988af');ax.set(title='(d) Fixed-axis detail: 64 updates',xlabel='Additional updates',ylabel='Coordinate ×1000')
ss=[s for s in sorted(set(f['probe_step'])) if s%512==0];colors=['#b65d22','#185e9c','#359a8b','#8064a2']
for ax,gg,title in zip(axes[2],[f['fixed_gain'][:,0],f['scalar_gain']],['(e) Fixed-axis gain ladder','(f) Batch-gradient gain ladder']):
 mm=[]
 for s in ss:
  z=np.log(np.abs(gg[f['probe_step']==s]));mm.append([np.exp(np.log(len(z))-logsumexp(-z)),np.exp(z.mean()),np.exp(logsumexp(z)-np.log(len(z))),np.exp(.5*(logsumexp(2*z)-np.log(len(z))))])
 for k,name in enumerate(['H','G','A1','A2']):ax.plot(ss,np.array(mm)[:,k],['-','--','-.',':'][k],color=colors[k],label=name,lw=1.3)
 ax.axhline(1,color='.3',ls='--',lw=.8);ax.set(title=title,xlabel='Additional updates',ylabel='Conditional gain');ax.legend(ncol=4,fontsize=7)
for ax in axes.flat:ax.grid(alpha=.13)
fig.tight_layout(pad=.6)
for ext in ['pdf','png']:fig.savefig(R/'figures'/('empirical_iid_mlp_dynamics.'+ext),bbox_inches='tight',dpi=180)
