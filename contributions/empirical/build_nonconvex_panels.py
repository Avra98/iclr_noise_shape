#!/usr/bin/env python3
"""Existing nonconvex controls: immutable arrays and completed-run summaries only."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.stats import t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;FIG=HERE.parents[1]/'figures'
ROOT=Path('/scratch/users/ghoshavr/eoss/results/nonconvex_campaign_20260905')
audit={'inputs':{},'scalar':{},'matched_variance':[],'constructed_networks':[],'ordinary_controls':{}}
def register(path):
    path=Path(path);audit['inputs'][str(path)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
def rows(path):
    path=ROOT/path;register(path);return list(csv.DictReader(open(path)))
def read(path):
    path=ROOT/path;register(path);return json.loads(path.read_text())
def agg(a):
    a=np.asarray(a,float);return dict(mean=float(a.mean()),configuration_means=a.tolist(),nominal_95_t_halfwidth=float(tdist.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))))
def save(fig,stem):
    fig.savefig(FIG/(stem+'.pdf'),bbox_inches='tight',pad_inches=.025)
    fig.savefig(FIG/(stem+'.png'),dpi=240,bbox_inches='tight',pad_inches=.025);plt.close(fig)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
  'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.7,'pdf.fonttype':42,
  'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
for p in ['RESEARCH_VERDICT.md','EVIDENCE_LEDGER.md','toys/FINAL_TOY_REPORT.md','toys/MATCHED_NOISE_ANALYSIS.md',
          'bounded_networks/FINAL_REPORT.md','bounded_networks/METHODS.md','networks/FINAL_NEURAL_REPORT.md']:
    register(ROOT/p)

# Smooth scalar: select FP64 confirmation trajectories by scientific configuration.
extension=rows('toys/extension_summary.csv')
rr=[r for r in extension if r['source']=='chaos_confirmation' and r['family']=='saturating' and
    float(r['eta'])==8 and int(r['batch'])==1 and float(r['noise'])==.001 and r['dtype']=='float64']
assert len(rr)==3
occupancy=[];means=[]
for r in rr:
    p=ROOT/'toys/chaos_confirmation'/(r['id']+'.npz');register(p);z=np.load(p)
    lam=float(np.mean(z['tangent_sum']/z['tangent_count']));assert abs(lam-float(r['moving_tangent']))<2e-12
    means.append(lam)
    p=ROOT/'toys/chaos_confirmation'/(r['id']+'_diagnostics.npz');register(p);z=np.load(p)
    assert z['dense_observable'].shape==(20000,16);occupancy.append(z['dense_observable'].astype(float).ravel())
    assert np.all(np.isfinite(occupancy[-1]))
audit['scalar']={'setting':{'eta':8.,'B':1,'beta':.92,'mult':.05,'residual_sigma':.001},
    'completed_configs':3,'chains_per_config':16,'updates_per_chain':1000000,'growth':agg(means),
    'density':'All saved consecutive 20000-update followups, all16chains, combined across3completed FP64 confirmations; 960000 correlated occupancy samples.',
    'source_ids':[r['id'] for r in rr], 'analytic_scope':'Recurrence and all polynomial moments follow from uniform outer contraction; displayed positive smooth tangent growth is numerical, not an analytic asymptotic-exponent proof.',
    'precision_exception':'The separately saved full-FP32 tangent run with an exact zero gain is not folded into FP64 estimates. Its bit-identical-state high-precision tangent replay is described in the original audit.'}
matched=rows('toys/matched_noise_summary.csv')
for sigma in [.001,.01,.03]:
    for B,m in [(1,.05),(4,.05),(16,.05),(1,0.)]:
        rs=[r for r in matched if float(r['effective_noise'])==sigma and int(r['batch'])==B and float(r['mult'])==m]
        assert len(rs)==3 and all(int(r['escaped'])==0 for r in rs)
        for r in rs:
            assert abs(float(r['Var_xi'])-sigma**2)<1e-14
            assert abs(float(r['Var_h'])-m*m/B)<1e-14
        audit['matched_variance'].append({'sigma_eff':sigma,'batch':B,'mult':m,'ids':[r['id'] for r in rs],**agg([float(r['moving_Lambda']) for r in rs])})
fig,ax=plt.subplots(1,2,figsize=(5.5,2.6),layout='constrained')
x=np.concatenate(occupancy);ax[0].hist(x,bins=np.linspace(-3.85,3.85,121),density=True,color='#377b86',alpha=.85)
ax[0].set(title='(a) Smooth recurrent scalar, '+r'$\eta=8$',xlabel='State x',ylabel='Occupancy density')
ax[0].text(.03,.95,r'Moving growth $+0.18959$',transform=ax[0].transAxes,va='top',fontsize=7)
ax[0].text(.03,.86,'Numerical estimate; recurrence proved',transform=ax[0].transAxes,va='top',fontsize=6.4)
for sigma,color in zip([.001,.01,.03],['#286981','#d68e36','#78629e']):
    vals=[next(r['mean'] for r in audit['matched_variance'] if r['sigma_eff']==sigma and r['batch']==B and r['mult']==m) for B,m in [(1,.05),(4,.05),(16,.05),(1,0.)]]
    ax[1].plot(range(4),vals,'o-',color=color,markersize=3,label=rf'$\sigma_{{\rm eff}}={sigma:g}$',lw=1)
ax[1].axhline(0,color='black',ls='--',lw=.65)
ax[1].set(title='(b) Matched additive variance',xticks=range(4),xticklabels=['B=1','B=4','B=16','Pure\nadditive'],ylabel='Moving log growth / update')
ax[1].legend(loc='lower left',fontsize=6.5)
for a in ax:a.grid(alpha=.15,lw=.5)
save(fig,'empirical_nonconvex_scalar_controls')

# Constructed network confirmation means, verified against per-chain raw summaries.
bounded=rows('bounded_networks/summary.csv');groups=read('bounded_networks/groups.json')
for g in groups:
    rs=[r for r in bounded if r['stage']=='confirmation' and r['activation']==g['activation'] and int(r['depth'])==g['depth'] and
        float(r['eta'])==g['eta'] and (r['freeze_output_bias']=='True')==g['freeze_output_bias']]
    assert len(rs)==3
    means=[]
    for r in rs:
        z=read('bounded_networks/confirmation/'+r['id']+'.json');assert z['all_finite']
        m=float(np.mean(z['moving_by_chain']));assert abs(m-float(r['moving']))<2e-12
        means.append(m)
    assert abs(np.mean(means)-g['mean'])<2e-12
    audit['constructed_networks'].append({'activation':g['activation'],'affine_layers':g['depth'],'eta':g['eta'],
      'freeze_output_bias':g['freeze_output_bias'],'source_ids':[r['id'] for r in rs],**agg(means)})

# Main ordinary CSV already includes long continuations: do not count them twice.
ordinary=rows('networks/summary.csv');common=rows('networks/common_minimum/summary.csv')
cohorts=[('Initial grid',[r for r in ordinary if r['cell'].startswith('cell_')]),
         ('Long followup',[r for r in ordinary if r['cell'].startswith('long_')]),('Common minimum',common)]
assert [(len(rs),sum(r['escaped']=='True' for r in rs)) for _,rs in cohorts]==[(216,61),(117,13),(162,61)]
counts=read('networks/execution_counts.json')
assert counts['total']['cells']==495 and counts['total']['cutoff_escapes']==135
audit['ordinary_controls']['cohorts']=[{'label':name,'executions':len(rs),'cutoff_escapes':sum(r['escaped']=='True' for r in rs)} for name,rs in cohorts]
audit['ordinary_controls']['counting']='Initial216+long117+common-minimum162=495 distinct completed production cells;135 cutoff escapes. The main ordinary summary already contains117long rows. No proposed next-study runs counted.'
cnn=read('real_networks/FINAL_CNN_200K_COMPARISON.json')
paired=[r for r in cnn['rows'] if r['config']['parent_seed']==88881]
assert len(paired)==2 and paired[0]['parent_sha256']==paired[1]['parent_sha256']
audit['ordinary_controls']['paired_cnn']=paired
fig,ax=plt.subplots(2,2,figsize=(5.5,3.85),layout='constrained')
for col,act in enumerate(['tanh','sigmoid']):
    a=ax[0,col];xx=np.arange(2);w=.23
    for j,(eta,freeze,color,label) in enumerate([(.5,False,'#89929c',r'$\eta=.5$'),(10.,False,'#377b86',r'$\eta=10$'),(10.,True,'#d4943e',r'$\eta=10$, bias frozen')]):
        yy=[next(g['mean'] for g in audit['constructed_networks'] if g['activation']==act and g['affine_layers']==depth and g['eta']==eta and g['freeze_output_bias']==freeze) for depth in [2,4]]
        a.bar(xx+(j-1)*w,yy,width=w,color=color,label=label)
    a.axhline(0,color='black',lw=.65);a.set(title=f'({chr(97+col)}) Constructed {act} networks',xticks=xx,xticklabels=['2 layers','4 layers'],ylabel='Moving growth / update',ylim=(-.065,1.08))
    if col==0:a.legend(fontsize=5.9,loc='upper left')
    a.grid(axis='y',alpha=.15)
a=ax[1,0];cc=audit['ordinary_controls']['cohorts'];rate=[c['cutoff_escapes']/c['executions'] for c in cc]
a.bar(range(3),rate,color='#b86056',width=.62)
for j,c in enumerate(cc):a.text(j,rate[j]+.017,f"{c['cutoff_escapes']}/{c['executions']}",ha='center',fontsize=7)
a.set(title='(c) Ordinary-network cutoff escapes',xticks=range(3),xticklabels=['Initial\ngrid','Long\nfollowup','Common\nminimum'],ylabel='Observed fraction',ylim=(0,.49));a.grid(axis='y',alpha=.15)
a=ax[1,1]
for r in paired:
    path=Path(r['trajectory_path']);register(path);z=np.load(path);t=z['diagnostic_steps'];norm=z['param_norm'].astype(float)
    early=(t>=100000)&(t<150000);late=(t>=150000)&(t<200000)
    shift=float(norm[late].mean()-norm[early].mean());assert abs(shift-r['param_norm']['mean_shift'])<2e-12
    keep=t>=100000;v=norm[keep]-norm[t==100000][0]
    noisy=r['config']['label_noise']>0
    a.plot(t[keep]/1000,v,color='#b86056' if noisy else '#377b86',lw=1.3,label='Target noise .01' if noisy else 'No target noise')
a.set(title='(d) Ordinary CNN: continuing drift',xlabel='Additional updates (thousands)',ylabel=r'$\|\theta_t\|-\|\theta_{100k}\|$')
a.legend(fontsize=6.5,loc='upper left');a.grid(alpha=.15)
save(fig,'empirical_nonconvex_network_controls')
(HERE/'nonconvex_provenance.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
print('Saved two nonconvex control panels with verified summaries and immutable-source hashes.')
