#!/usr/bin/env python3
"""Publication panels from immutable corrected measurements; no training."""
from pathlib import Path
import sys, json, hashlib, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
FIG = PAPER / 'figures'
BASE = Path('/scratch/users/ghoshavr/eoss/results')
RUN = BASE / 'CNN_DIRECTIONAL_WINDOWS_20260905'
GRID = BASE / 'ALL_EMPIRICAL_RESULTS_20260905/PART_01_CONCISE/support/grids_vision/numeric_grid_audit.json'
SNAP = BASE / 'marc_moment_campaign_20260905/figure_review_20260905/figure_inputs.pkl'
sys.path.insert(0, str(HERE))
from local_window_bimodality import density, shape, controls, moments, clean

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
    'axes.titlesize':9, 'axes.labelsize':8, 'legend.fontsize':7,
    'xtick.labelsize':7, 'ytick.labelsize':7, 'axes.spines.top':False,
    'axes.spines.right':False, 'pdf.fonttype':42, 'savefig.facecolor':'white',
    'axes.unicode_minus':True})
FIG.mkdir(exist_ok=True)
audit = {'inputs': {}, 'grid': {}, 'iid_mlp': {}, 'cnn_confirmation': {}}
def record(path):
    path=Path(path); audit['inputs'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
def save(fig, name):
    fig.savefig(FIG/(name+'.pdf'), bbox_inches='tight', pad_inches=.025)
    fig.savefig(FIG/(name+'.png'), dpi=240, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
def panel(ax,title):
    ax.set_title(title,loc='left',pad=5)
def tidy(ax):
    ax.grid(alpha=.17,lw=.5)
def fmt(x,center):
    # Three significant digits preserve small log gains; no rounded side reversal.
    s=f'{x:.2g}' if center==0 else f'{x:.3g}'
    if s.startswith('-0.'): s='-'+s[2:]
    elif s.startswith('0.'): s=s[1:]
    if float(s)==center and x!=center: s+='+' if x>center else '−'
    return s

# Recompute each displayed statistic from saved raw gains, then average within cells.
record(GRID); record(SNAP); record(HERE/'local_window_bimodality.py')
grid=json.loads(GRID.read_text()); runs=pickle.loads(SNAP.read_bytes())
selected=[next(g for g in grid['figures'] if g['arch']==arch) for arch in ['cnn_silu','mlp_silu']]
for g in selected:
    for c in g['cells']:
        rr=[r for r in runs if r['arch']==g['arch'] and r['eta']==c['eta'] and r['B']==c['B']]
        vals=[]
        for r in rr:
            pool=next(p for p in r['pools'] if p['probe_B']==c['B'])
            a=np.abs(np.asarray(pool['a'],float)); assert len(a)>0 and np.all(a>0)
            vals.append({'Lambda':float(np.log(a).mean()),'H':float(1/(1/a).mean())})
        for key in ['Lambda','H']:
            val=float(np.mean([v[key] for v in vals]))
            assert abs(val-c['metrics'][key]['mean'])<2e-12
    audit['grid'][g['arch']]=g
audit['grid']['aggregation']='Pool final three available checkpoints within each run; compute Lambda and H on each run pool; equally average those statistics across three runs. This is not H of a combined gain pool. Horizons vary.'
audit['grid']['raw_gain_recomputation']='All 72 displayed means agree with raw saved gains to <2e-12.'

fig, axes=plt.subplots(2,2,figsize=(5.5,3.05),layout='constrained')
for i,g in enumerate(selected):
    for j,(metric,title,norm) in enumerate([
        ('Lambda',r'$\widehat\Lambda_g$',TwoSlopeNorm(vmin=-1.05,vcenter=0,vmax=.20)),
        ('H',r'$\widehat H_g$',TwoSlopeNorm(vmin=0,vcenter=1,vmax=1.20))]):
        ax=axes[i,j]; etas=g['learning_rates']; bs=g['batch_sizes']; z=np.zeros((len(etas),len(bs)))
        for c in g['cells']: z[etas.index(c['eta']),bs.index(c['B'])]=c['metrics'][metric]['mean']
        im=ax.imshow(z,cmap='coolwarm',norm=norm,aspect='auto')
        for (r,c),v in np.ndenumerate(z):
            ax.text(c,r,fmt(v,0 if metric=='Lambda' else 1),ha='center',va='center',fontsize=6.6,
                    color='white' if norm(v)<.13 or norm(v)>.9 else '#172027')
        ax.set(xticks=range(len(bs)),xticklabels=bs,yticks=range(len(etas)),yticklabels=[f'{e:g}' for e in etas],ylabel=r'Step size $\eta$')
        if i==1: ax.set_xlabel('Batch size')
        panel(ax,f'({chr(97+2*i+j)}) '+('CNN SiLU' if i==0 else 'MLP SiLU')+'   '+title)
        cb=fig.colorbar(im,ax=ax,shrink=.90,pad=.025,aspect=20)
        cb.set_ticks([-1,-.5,0,.2] if metric=='Lambda' else [0,.5,1,1.2])
save(fig,'empirical_stability_heatmaps')

# Main neural panel: a fixed choice of window, not an optimized best-looking interval.
branch=RUN/'branches/mlp_iid_control'
for f in ['trace.npz','fresh_probes.npz','meta.json','full_loss.json']: record(branch/f)
trace=dict(np.load(branch/'trace.npz')); probes=dict(np.load(branch/'fresh_probes.npz'))
meta=json.loads((branch/'meta.json').read_text()); assert meta['status']=='completed' and meta['completed_updates']==4096
summary_path=RUN/'directional_analysis/mlp_iid_control/summary.json';record(summary_path)
summ=json.loads(summary_path.read_text())
time=trace['step']; x=trace['coordinates'][:,0]; lo,hi=2048,2560
assert np.array_equal(time,np.arange(4096))
shape_result=shape(x[lo:hi],factors=(.7,1.,1.4,2.));control_result=controls(x[lo:hi],shape_result)
moving=trace['moving_log_gain'];late=moving[2048:].mean(axis=0)
assert np.allclose(late,summ['moving_late_half'],atol=2e-12,rtol=0)
late_fixed=moments(1-meta['eta']*trace['fixed_rayleigh'][2048:,0])
assert abs(late_fixed['Lambda']-summ['realized_late_fixed_axes']['fixed_top_ritz']['Lambda'])<2e-12
phase=[]
for pair in range(8):
    n=min(np.sum(probes['probe_step']==512*pair),np.sum(probes['probe_step']==512*pair+1))
    assert n==24 or pair==0
    for adjacent in [0,1]:
        mask=(probes['probe_step']==512*pair+adjacent)&(probes['draw_index']<n)
        mm=moments(probes['fixed_gain'][mask,0]);phase.append({'pair':pair,'step':512*pair+adjacent,'adjacent':adjacent,**mm})
audit['iid_mlp']={'metadata':meta,'display_window':[lo,hi],'selection':'First 512 updates of late half; descriptive display selected after runs, not an optimized shape window or prospective hypothesis test.',
    'shape':shape_result,'controls':control_result,'late_propagated_growth':late.tolist(),
    'late_fixed_axis_time_mixture':late_fixed,'phase_metrics':phase,'loss_initial':summ['full_loss_initial'],
    'loss_final':summ['full_loss_final'],'initial_batch_scalar':summ['initial_batch_scalar'],
    'full_summary_source':str(summary_path)}
fig,axes=plt.subplots(2,2,figsize=(5.5,4.25),layout='constrained')
ax=axes[0,0];ax.plot(time[lo:lo+64],1000*x[lo:lo+64],'.-',color='#216679',markersize=2,lw=.65)
panel(ax,'(a) Fixed-axis trace (64 updates)')
ax.set(xlabel='Additional SGD updates',ylabel=r'$10^3 q_\star^\top(\theta_t-\theta_\star)$',xticks=[2048,2080,2112]);tidy(ax)
ax=axes[0,1];xx=1000*x[lo:hi]
ax.hist(xx,bins=30,density=True,color='#cfd8dd',edgecolor='none')
for factor,col in zip([.7,1,1.4,2],['#293a64','#208b8d','#dc9533','#b25444']):
    gx,gy,_=density(xx,factor);ax.plot(gx,gy,label=rf'$\times{factor:g}$',color=col,lw=1)
panel(ax,'(b) 512-update occupancy')
ax.set(xlabel=r'$10^3 q_\star^\top(\theta_t-\theta_\star)$',ylabel='Density')
ax.legend(title='Bandwidth',ncol=2,fontsize=6,title_fontsize=6,loc='upper center',framealpha=.85);tidy(ax)
ax=axes[1,0]
for j,(name,color,ls) in enumerate(zip(['Leading-axis start','Gradient start','Random start'],['#216679','#d89237','#776192'],['-','--',':'])):
    ax.plot(time+1,np.cumsum(moving[:,j]),label=name,color=color,ls=ls,lw=1.05)
ax.axhline(0,color='#39424b',lw=.65,ls='--');ax.axvspan(lo,hi,color='#216679',alpha=.09)
panel(ax,'(c) Propagated tangent growth')
ax.set(xlabel='Additional SGD updates',ylabel='Cumulative log norm gain')
ax.text(.03,.96,r'Late growth $+0.00642$/update',transform=ax.transAxes,va='top',fontsize=7)
ax.legend(fontsize=6,loc='lower right');tidy(ax)
ax=axes[1,1]
for adjacent,col,marker,label in [(0,'#216679','o',r'State $t$'),(1,'#d89237','s',r'State $t+1$')]:
    rr=[p for p in phase if p['adjacent']==adjacent]
    ax.plot([r['step']-adjacent for r in rr],[r['H'] for r in rr],marker+'-',color=col,markersize=3,lw=1,label=label)
ax.axhline(1,color='#39424b',lw=.8,ls='--')
panel(ax,'(d) Paired-state harmonic gain')
ax.set(xlabel=r'Pair starting update $t$',ylabel=r'Fixed-axis $\widehat H_{q_\star}$',ylim=(.05,1.83))
ax.legend(fontsize=6,ncol=2,loc='lower left');tidy(ax)
save(fig,'empirical_iid_mlp_dynamics')

# Compact negative control for the appendix, retaining the unsuccessful confirmation.
confirm_path=RUN/'analysis/confirmation.json';record(confirm_path)
confirmation=json.loads(confirm_path.read_text()); audit['cnn_confirmation']=confirmation
fig,axes=plt.subplots(2,2,figsize=(6.6,3.65),layout='constrained')
for j,(case,label) in enumerate([('cnn_discovery','Discovery'),('cnn_confirmation','Confirmation')]):
    record(RUN/'branches'/case/'trace.npz')
    t=dict(np.load(RUN/'branches'/case/'trace.npz'));xx=t['coordinates'][672:736,0]*1000
    axes[0,j].plot(np.arange(672,736),xx,'.-',lw=.6,markersize=2,color='#216679')
    panel(axes[0,j],f'({chr(97+j)}) {label}: same fixed axis/window')
    axes[0,j].set(xlabel='Additional SGD updates',ylabel=r'$10^3 q_\star^\top(\theta_t-\theta_\star)$')
    axes[1,j].hist(xx,bins=14,density=True,color='#cfd8dd',edgecolor='none')
    for factor,col in zip([.7,1,1.4],['#293a64','#208b8d','#dc9533']):
        gx,gy,_=density(xx,factor);axes[1,j].plot(gx,gy,label=rf'$\times{factor:g}$',color=col,lw=1)
    panel(axes[1,j],f'({chr(99+j)}) '+('Passes all three bandwidths' if j==0 else 'Does not pass largest bandwidth'))
    axes[1,j].set(xlabel='Scaled fixed-axis coordinate',ylabel='Density');axes[1,j].legend(fontsize=6)
for ax in axes.flat:tidy(ax)
save(fig,'empirical_cnn_confirmation')

(HERE/'numerical_audit.json').write_text(json.dumps(clean(audit),indent=2,allow_nan=False))
print('Saved 3 empirical PDF/PNG panels, raw-gain verification and complete numerical audit.')
