from pathlib import Path
import json, shutil, pickle, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import logsumexp
ROOT=Path('/scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906')
REPO=Path('/scratch/users/ghoshavr/eoss/papers/iclr_noise_shape')
OUT=REPO/'img/current_results'; OUT.mkdir(parents=True,exist_ok=True)
OLD=Path('/scratch/users/ghoshavr/eoss/results/CNN_DIRECTIONAL_WINDOWS_20260905')
plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
manifest=[]; audit={}
def save(fig,name,caption):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 manifest.append(dict(name=name,caption=caption))
def moments(a):
 a=np.abs(np.asarray(a,dtype=float));assert np.isfinite(a).all()
 with np.errstate(divide='ignore'):
  z=np.log(a);lam=float(z.mean());h=0. if (a==0).any() else float(np.exp(-logsumexp(-z)+np.log(len(a))))
 return dict(H=h,G=float(np.exp(lam)),A1=float(a.mean()),A2=float(np.sqrt(np.mean(a*a))),Lambda=lam,n=len(a),min_abs=float(a.min()),zeros=int((a==0).sum()))
def reference(ax,y=1):ax.axhline(y,color='.25',ls='--',lw=.9);ax.grid(alpha=.15)
b=OLD/'branches/mlp_iid_control';t=dict(np.load(b/'trace.npz'));f=dict(np.load(b/'fresh_probes.npz'));meta=json.loads((b/'meta.json').read_text());eta=meta['eta'];time=t['step'];x=t['coordinates'][:,0]
alignpath=ROOT/'branches/mlp_iid_control/eigenvector_alignment.json'
a=json.loads(alignpath.read_text()); replay=dict(np.load(ROOT/'branches/mlp_iid_control/trace.npz'))
assert len(replay['step'])==4096
# Alignment is only attached to the archived trajectory after numerical replay verification.
err={k:float(np.max(np.abs(replay[k]-t[k]))) for k in ['coordinates','batch_loss','scalar_s','moving_log_gain']}
assert err['coordinates']<1e-7 and err['batch_loss']<1e-7,err
audit['replay_max_absolute_errors']=err;audit['alignment']=a
full=json.loads((b/'full_loss.json').read_text());full={v['step']:v['loss'] for v in full}
import sys
sys.path.insert(0,'/scratch/users/ghoshavr/eoss/results/APPENDIX_DIRECTIONAL_LADDERS_20260906/source')
import render as appendix_renderer
main_record=appendix_renderer.render(appendix_renderer.load_case(b,True),'mlp_silu_B4096_anchor_revised')
for ext in ['pdf','png']:
    shutil.copy2(appendix_renderer.OUT/('mlp_silu_B4096_anchor_revised.'+ext),OUT/('fig01_mlp_loss_alignment_bimodality.'+ext))
manifest.append(dict(name='fig01_mlp_loss_alignment_bimodality',caption=main_record['caption']+' The original anchor uses128 fresh draws at zero and24 at subsequent even states512k.'))
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
labels=['Batch gradient $q_B$','Fixed leading $q_*$','Fixed checkpoint gradient','Fixed random direction'];cols=['#185e9c','#b65d22','#359a8b','#888888']
steps=sorted(set(f['probe_step']));selected=[s for s in steps if s%512==0]
for label,col,gain in zip(labels,cols,[f['scalar_gain'],f['fixed_gain'][:,0],f['fixed_gain'][:,1],f['fixed_gain'][:,2]]):
 mm=[moments(gain[f['probe_step']==s]) for s in selected]
 for ax,key,title in zip(axes.flat,['H','G','A1','A2'],[r'$H=(\langle |a|^{-1}\rangle)^{-1}$',r'$G=\exp\langle\log|a|\rangle$',r'$A_1=\langle|a|\rangle$',r'$A_2=\sqrt{\langle|a|^2\rangle}$']):
  ax.plot(selected,[v[key] for v in mm],'o-',label=label,color=col,ms=4);ax.set(title=title,ylabel='Conditional scalar gain');reference(ax)
 audit[label+' fresh_even']=[dict(step=int(s),**m) for s,m in zip(selected,mm)]
for ax in axes[-1]:ax.set_xlabel('Additional SGD updates (states at 512k)')
axes[0,0].legend(fontsize=8);fig.suptitle('The direction is part of the measurement · identical four gain summaries',fontsize=14);fig.tight_layout(rect=[0,0,1,.95])
save(fig,'fig02_direction_resolved_ladder','Fresh conditional minibatch probes at each displayed fixed state, using the same draws across directions. There are 128 draws at update zero and 24 at other displayed states. Only states at multiples of 512 are displayed here; the adjacent phase appears in Figure 3. The batch-gradient direction changes with the sampled batch. All other directions are checkpoint-fixed. Geometric gain G=exp(Lambda) shares the unit reference with H, A1 and A2. Finite H does not establish a finite population inverse moment.')
fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
for parity in [0,1]:
 ss=[s for s in steps if s%512==parity and s<4096];mm=[]
 for s in ss:
  mask=(f['probe_step']==s)&(f['draw_index']<24);mm.append(moments(f['fixed_gain'][mask,0]))
 for ax,key in zip(axes.flat,['H','G','A1','A2']):
  ax.plot(np.array(ss)-parity,[m[key] for m in mm],'o-',label=f'State 512k+{parity}');reference(ax);ax.set(title=key,ylabel='Fixed-axis conditional gain')
axes[0,0].legend();[ax.set_xlabel('Adjacent-pair starting update') for ax in axes[-1]];fig.suptitle('The two oscillation phases have different gain laws');fig.tight_layout(rect=[0,0,1,.94])
save(fig,'fig03_paired_phase_ladder','Fixed leading-axis scalar gains: 24 identical fresh minibatches reused across the two states of each adjacent pair. All four measures are computed separately by phase. These are conditional one-step compression statistics, not the two-step product exponent.')
# Cross-architecture gain laws: fixed final checkpoint, matched parent batch; no figure per seed.
runs=pickle.loads(Path('/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/figure_review_20260905/figure_inputs.pkl').read_bytes())
configs=[('cnn_silu',.08,'CNN (SiLU)'),('mlp_silu',.02,'MLP (SiLU)'),('wrn_no_bn',.002,'WideResNet'),('resnet32',.05,'ResNet-32')]
fig,axes=plt.subplots(2,4,figsize=(13,6));pa=[]
for j,(arch,eta_a,title) in enumerate(configs):
 for bsize,c in zip([32,128,512,4096],['#6a4994','#277da8','#359a8b','#ce942c']):
  gs=[]
  for r in runs:
   if r['arch']==arch and r['eta']==eta_a and r['B']==bsize:
    candidates=[g for g in r['gains'] if g['step']==20000 and g['probe_B']==bsize]
    if candidates:gs.append(candidates[-1])
  if not gs:continue
  zz=[np.log(np.abs(g['a'])) for g in gs];assert all(np.isfinite(z).all() for z in zz)
  lo=min(z.min() for z in zz);hi=max(z.max() for z in zz);xx=np.linspace(lo,hi,250);cdf=np.mean([[np.mean(z<=v) for v in xx] for z in zz],axis=0)
  pp=np.linspace(-1,2,91);pressure=np.mean([[0. if p==0 else logsumexp(p*z)-np.log(len(z)) for p in pp] for z in zz],axis=0)
  axes[0,j].plot(xx,cdf,label=f'B={bsize}',color=c);axes[1,j].plot(pp,pressure,color=c)
  pa.append(dict(arch=arch,eta=eta_a,batch=bsize,step=20000,runs=len(gs),sources=[g['path'] for g in gs]))
 axes[0,j].axvline(0,color='.4',ls='--',lw=.8);axes[0,j].set(title=f'{title}\nη={eta_a:g}',xlabel=r'$\log|a_B|$',ylabel='CDF' if j==0 else '',ylim=(0,1));axes[0,j].legend(fontsize=7)
 axes[1,j].axhline(0,color='.4',ls='--',lw=.8);axes[1,j].axvline(0,color='.6',ls=':',lw=.8);axes[1,j].set(xlabel='Moment order p',ylabel=r'$\psi(p)=\log\langle|a_B|^p\rangle$' if j==0 else '')
fig.suptitle('Gain laws across architectures · CIFAR-10 · batch-gradient direction',fontsize=14);fig.tight_layout(rect=[0,0,1,.93]);audit['cross_architecture_selection']=pa
save(fig,'fig04_cross_architecture_gain_laws','Batch-gradient scalar gains at update 20000 and matched training/probe batch size. Curves average within-run CDFs or log moments equally across available runs; this is not the log moment of pooled gains. No separate seed panels. Architecture-specific step sizes are labeled; this is not a controlled comparison changing architecture alone. Only measured final-checkpoint cases are shown; the full grids retain earlier/pre-escape observations. Positive and negative moments summarize gain distributions, not parameter-coordinate histograms.')
# Include all existing architecture/dataset figures, with their source captions.
base=Path('/tmp/ghoshavr_part1_concise');sources=[]
for fn in ['vision_manifest.json','grids_vision/manifest.json','gpt/manifest.json','grids_gpt/manifest.json']:
 content=json.loads((base/fn).read_text());print(fn,type(content).__name__)
 if isinstance(content,dict):content=content.get('figures',content.get('records',[]))
 for item in content:
  name=item.get('figure_id','');pdf=Path(item.get('pdf',''))
  if not name or not pdf.is_file():continue
  # Only architecture-level plots; exclude older summary montages.
  if name.startswith(('H','C','P')):continue
  for ext in ['pdf','png']:
   src=pdf.with_suffix('.'+ext)
   if src.is_file():shutil.copy2(src,OUT/(name+'.'+ext))
  sources.append(dict(name=name,caption=item.get('caption',''),title=item.get('title',name),source=str(pdf)))
manifest+=sources
(ROOT/'NUMERIC_AUDIT.json').write_text(json.dumps(audit,indent=2));(ROOT/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('Saved',len(manifest),'figure pairs')
