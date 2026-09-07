from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'img/appendix_directional_ladders'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
LABELS={'mlp_silu':'SiLU MLP','cnn_silu':'SiLU CNN','resnet32':'ResNet-32','wrn_no_bn':'WideResNet-10-2'}

def means(a):
    a=np.abs(np.asarray(a,dtype=float))
    if not len(a) or not np.isfinite(a).all():
        return {k:float('nan') for k in ['H','G','A1','A2']}
    with np.errstate(divide='ignore'):
        z=np.log(a)
        h=0. if np.any(a==0) else float(np.exp(np.log(len(a))-logsumexp(-z)))
    return dict(H=h,G=float(np.exp(z.mean())),A1=float(a.mean()),A2=float(np.sqrt(np.mean(a*a))))

def load_case(path,anchor=False):
    meta=json.loads((path/'meta.json').read_text());t=dict(np.load(path/'trace.npz'))
    if anchor:
        f=dict(np.load(path/'fresh_probes.npz'))
        d=dict(time=t['step'],coordinate=t['coordinates'][:,0],loss_time=t['step'],loss=t['batch_loss'],
               probe_time=f['probe_step'],fixed=f['fixed_gain'][:,0],batch_gain=f['scalar_gain'],
               batch=meta['batch'],eta=meta['eta'],arch=meta['architecture'],parent_step=meta['parent_step'],status=meta['status'])
        alignment=ROOT/'anchor/eigenvector_alignment.json'
    else:
        f=dict(np.load(path/'same_axis_gains.npz'))
        d=dict(time=t['step'],coordinate=t['projection'][:,0],loss_time=t['loss_step'],loss=t['loss'],
               probe_time=f['probe_step'],fixed=f['a'][:,0],batch_gain=f['batch_gradient_a'],
               batch=meta['batch_size'],eta=meta['learning_rate'],arch=meta['architecture'],parent_step=meta['checkpoint_step'],status=meta['status'])
        alignment=path/'eigenvector_alignment.json'
    d['alignment']=json.loads(alignment.read_text())
    frame_info=meta['fixed_frame_eigenpair'] if anchor else meta['eigenpair']
    d['frame_residual']=frame_info['relative_residual']
    d['frame_converged']=frame_info['residual_converged']
    d['full_loss']=json.loads((path/'full_loss.json').read_text())
    d['source']=str(path)
    return d

def render(d,name):
    fig=plt.figure(figsize=(12,12.5))
    gs=GridSpec(6,4,figure=fig,height_ratios=[1.05,.72,1,.65,1,1],hspace=.7,wspace=.48)
    fig.subplots_adjust(top=.94,bottom=.05)
    ax=fig.add_subplot(gs[0,:]);ax.plot(d['loss_time'],d['loss'],color='#b3cfe5',lw=.5,label='Training minibatch')
    fl={v['step']:v['loss'] for v in d['full_loss']}
    ax.plot(list(fl),list(fl.values()),color='#185e9c',lw=1.7,label='Full training subset')
    ax.set(yscale='log',xlabel='Additional SGD updates',ylabel='Squared loss',title='(a) Training loss');ax.legend(fontsize=8,ncol=2)
    windows=[(512,1024),(1024,1536),(2048,2560),(3072,3584)]
    colors=['#277da8','#359a8b','#b65d22','#8064a2']
    for (lo,hi),color in zip(windows,colors):ax.axvspan(lo,hi,color=color,alpha=.07)
    ax=fig.add_subplot(gs[1,:]);aa=d['alignment'];ax.plot([v['step'] for v in aa],[v['alignment'] for v in aa],'o-',ms=4,color='#b65d22')
    ax.set(xlabel='Additional SGD updates',ylabel=r'$|q_1(t)^\top q_*|$',ylim=(0,1.04),title='(b) Full-data leading Ritz direction versus the frozen frame');ax.grid(alpha=.15)
    if not d['frame_converged']:
        ax.text(.99,.05,f"Frozen frame residual {d['frame_residual']:.2g} exceeds tolerance; approximate axis",ha='right',transform=ax.transAxes,fontsize=8,color='#a23b22')
    for i,((lo,hi),color) in enumerate(zip(windows,colors)):
        ax=fig.add_subplot(gs[2,i]);mask=(d['time']>=lo)&(d['time']<hi)
        if mask.sum()==hi-lo:ax.hist(1000*d['coordinate'][mask],bins=30,density=True,color=color,alpha=.85)
        else:ax.text(.5,.5,'Window not reached',ha='center',va='center',transform=ax.transAxes)
        ax.set(title=f'(c{i+1}) Updates {lo}–{hi-1}',xlabel=r'$10^3 q_*^\top(\theta_t-\theta_*)$',ylabel='Occupancy density' if i==0 else '')
    ax=fig.add_subplot(gs[3,:]);mask=(d['time']>=2048)&(d['time']<2112)
    if mask.sum()==64:ax.plot(d['time'][mask],1000*d['coordinate'][mask],'.-',ms=3,lw=.8,color='#277da8')
    else:ax.text(.5,.5,'Declared interval not reached',ha='center',transform=ax.transAxes)
    ax.set(title='(d) Fixed-coordinate detail: updates 2048–2111',xlabel='Additional SGD updates',ylabel='Coordinate ×1000')
    records=[]
    for i,(key,title) in enumerate([('H',r'Harmonic $H=1/\langle |a|^{-1}\rangle$'),('G',r'Geometric $G=\exp\langle\log|a|\rangle$'),('A1',r'Mean absolute $A_1=\langle|a|\rangle$'),('A2',r'RMS $A_2=\sqrt{\langle|a|^2\rangle}$')]):
        ax=fig.add_subplot(gs[4+i//2,2*(i%2):2*(i%2)+2])
        ss=[int(s) for s in sorted(set(d['probe_time'])) if s%256==0]
        for g,label,color in [('fixed','Fixed leading direction','#b65d22'),('batch_gain','Batch-gradient direction','#185e9c')]:
            mm=[means(d[g][d['probe_time']==s]) for s in ss]
            ax.plot(ss,[v[key] for v in mm],'o-',ms=3,color=color,label=label)
            if i==0:records.extend([dict(direction=g,step=s,draws=int(np.sum(d['probe_time']==s)),**v) for s,v in zip(ss,mm)])
        ax.axhline(1,ls='--',lw=.8,color='.3');ax.grid(alpha=.15)
        ax.set(title=f'(e{i+1}) {title}',xlabel='Additional SGD updates',ylabel='Conditional gain')
        if i==0:ax.legend(fontsize=8)
    fig.suptitle(f"CIFAR-10 · {LABELS.get(d['arch'],d['arch'])} · B={d['batch']}, η={d['eta']:g}\nCheckpoint {d['parent_step']} · IID minibatch SGD · status: {d['status']}",fontsize=13,y=.995)
    for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    return dict(name=name,arch=d['arch'],batch=d['batch'],eta=d['eta'],parent_step=d['parent_step'],status=d['status'],source=d['source'],gains=records,windows=windows,frame_converged=d['frame_converged'],frame_residual=d['frame_residual'],
                caption='The center and plotting direction are fixed at the parent checkpoint. All runs use IID subsets, original parent learning rate, and the full 8192-example training subset. Curvature matrices in the alignment panel use that same subset. Histograms use four declared 512-update intervals and 30 bins without selecting intervals by modality. The short trace uses updates2048–2111. Gain ladders use fresh minibatches at each plotted state: fixed-axis compression versus batch-gradient compression, each summarized by H,G,A1,A2. No propagated-tangent panel and no static ladder panels are included. Sparse even-update conditional probes do not average over oscillation phases; they do not establish stationarity or a full-network Lyapunov exponent. Alignment is measured, not assumed, and Ritz extremality is not certified.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--anchor-only',action='store_true');args=parser.parse_args()
    records=[]
    anchor=ROOT/'anchor'
    records.append(render(load_case(anchor,True),'mlp_silu_B4096_anchor_revised'))
    if not args.anchor_only:
        for p in sorted((ROOT/'runs').glob('*/meta.json')):
            if not (p.parent/'same_axis_gains.npz').exists():continue
            meta=json.loads(p.read_text())
            if meta['status']=='running':continue
            records.append(render(load_case(p.parent),p.parent.name))
    (ROOT/('ANCHOR_MANIFEST.json' if args.anchor_only else 'FIGURE_MANIFEST.json')).write_text(json.dumps(records,indent=2))
    print('Rendered',len(records),'figure pairs')
