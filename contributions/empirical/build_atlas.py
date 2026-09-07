#!/usr/bin/env python3
"""Seven-page architecture atlas derived from the accepted Part 1 grids."""
from pathlib import Path
import json,hashlib,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE=Path(__file__).resolve().parent; PAPER=HERE.parents[1]; FIG=PAPER/'figures'
PART=Path('/scratch/users/ghoshavr/eoss/results/ALL_EMPIRICAL_RESULTS_20260905/PART_01_CONCISE')
MASTER=PART/'SOURCE_MANIFEST.json'
VISION=PART/'support/grids_vision/numeric_grid_audit.json'
GPT=PART/'support/grids_gpt/numeric_grid_audit.json'
master=json.loads(MASTER.read_text()); vv=json.loads(VISION.read_text()); gg=json.loads(GPT.read_text())
byid={f['figure_id']:f for f in master['figures']}
vision={g['arch']:g for g in vv['figures']}
gpt={g['figure_id']:g for g in gg['figure_records']}
snapnew=Path(vv['source_snapshot']);snapold=Path(vv['old_snapshot'])
new=pickle.loads(snapnew.read_bytes());old=pickle.loads(snapold.read_bytes())
proof={'inputs':{},'pages':[],'convention':'Replot only Lambda and H from accepted six-panel grids. Every eta/batch cell is retained. Gray N/A denotes missing gain estimates; E denotes recorded cutoff escape, including pre-escape estimates. Each run pools final three available checkpoint samples, then run-level statistics are averaged. All available means independently recomputed from raw saved gains.',
       'glyphs':'A trailing + or minus preserves the reference side when rounded; N/A is never a stable result.',
       'colors':'Signed log(1+abs(value-reference)/0.05), centered at references0 and1; panel-specific negative and positive ranges. Cell text and colorbar ticks retain original units.'}
def register(p):
    p=Path(p);proof['inputs'][str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
for p in [MASTER,VISION,GPT,snapnew,snapold]:register(p)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
  'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,
  'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def tx(v,ref):
    d=np.asarray(v)-ref
    return np.sign(d)*np.log1p(np.abs(d)/.05)
def label(v,ref,gpt=False):
    s=f'{v:.3g}' if gpt or ref==1 else f'{v:.2g}'
    if not gpt and ref==0 and 0<abs(v)<.01:
        s=f'{v:.0e}'.replace('e-0','e-').replace('e+0','e')
    if s.startswith('-0.'):s='-'+s[2:]
    elif s.startswith('0.'):s=s[1:]
    if len(s)>6 and not gpt:s=f'{v:.1e}'.replace('e-0','e-').replace('e+0','e')
    if float(s)==ref and v!=ref:s+='+' if v>ref else '−'
    return s
def source_and_verify(g,isgpt):
    ident=g['figure_id'];src=byid[ident];vector=Path(src['published_vector_source']);register(vector)
    if isgpt:
        arch=g['metric'];etas=g['grid_axes']['learning_rate'];bs=g['grid_axes']['batch'];source=new
        cells=[{'eta':c['eta'],'B':c['batch'],'values':{m:c['displayed_means'][m] for m in ['Lambda','H']},'recorded_escapes':sum(r['status']=='diverged' for r in c['seeds']),'missing':False,'gain_samples':c['total_gain_samples']} for c in g['cells']]
    else:
        arch=g['arch'];etas=g['learning_rates'];bs=g['batch_sizes'];source=new if arch in ['wrn_no_bn','resnet32'] else old
        cells=[{'eta':c['eta'],'B':c['B'],'values':{m:c['metrics'][m]['mean'] for m in ['Lambda','H']},'recorded_escapes':c['recorded_escapes'],'missing':c['metrics']['Lambda']['mean'] is None,'pool_windows':c['pool_windows']} for c in g['cells']]
    verified=0
    for c in cells:
        if c['missing']:continue
        rr=[r for r in source if r['arch']==arch and r['eta']==c['eta'] and r['B']==c['B']]
        vals=[]
        for r in rr:
            pool=[p for p in r['pools'] if p['probe_B']==c['B']]
            assert len(pool)==1
            a=np.abs(np.asarray(pool[0]['a'],float)); assert np.all(a>0)
            vals.append({'Lambda':np.log(a).mean(),'H':1/(1/a).mean()})
        assert vals
        for m in ['Lambda','H']:
            value=float(np.mean([v[m] for v in vals]))
            assert abs(value-c['values'][m]) < max(2e-12,abs(value)*2e-12),(arch,c,m,value,c['values'][m])
            verified+=1
    return {'source_figure_id':ident,'architecture':src['architecture'],'dataset':src['dataset'],
            'source_vector':str(vector),'accepted_part1_page':src['page'],'epoch':'new20260905' if source is new else 'archived_marc_arch_sweep',
            'axes':{'eta':etas,'batch':bs},'cells':cells,'raw_recomputed_statistics':verified,
            'missing_cells':sum(c['missing'] for c in cells),'recorded_escapes':sum(c['recorded_escapes'] for c in cells),
            'optimizer_caveat':'Adam parents; raw metrics are SGD counterfactuals; frozen metrics omit momentum and preconditioner evolution.' if isgpt else 'Plain SGD; gradient-direction scalar compression, not propagated network growth.'}

groups=[
('01_residual_new',[vision['wrn_no_bn'],vision['resnet32']],['WideResNet-10-2','ResNet-32 (no BN)'],False),
('02_archived_conv',[vision['cnn'],vision['resnet20']],['ReLU CNN','ResNet-20 (no BN)'],False),
('03_shallow_mlp',[vision['mlp_d1'],vision['mlp_d2']],['ReLU MLP: 1 × 256','ReLU MLP: 2 × 512'],False),
('04_deep_mlp',[vision['mlp_d3'],vision['mlp_d4']],['ReLU MLP: 3 × 256','ReLU MLP: 4 × 512'],False),
('05_tanh_vit',[vision['mlp_tanh'],vision['vit']],['Tanh MLP: 2 × 512','ViT: 2 blocks × 64'],False),
('06_gpt_raw',[gpt['gpt_grid01_raw_hessian'],gpt['gpt_grid02_raw_ggn']],['GPT: raw Hessian','GPT: raw GGN'],True),
('07_gpt_frozen',[gpt['gpt_grid03_frozen_adam_hessian'],gpt['gpt_grid04_frozen_adam_ggn']],['GPT: frozen Adam Hessian','GPT: frozen Adam GGN'],True)]
for suffix,gs,titles,isgpt in groups:
    records=[source_and_verify(g,isgpt) for g in gs]
    fig,axes=plt.subplots(2,2,figsize=(5.5,3.6),layout='constrained')
    for i,(rec,title) in enumerate(zip(records,titles)):
        etas=rec['axes']['eta'];bs=rec['axes']['batch']
        for j,(metric,ref) in enumerate([('Lambda',0.),('H',1.)]):
            ax=axes[i,j];z=np.full((len(etas),len(bs)),np.nan)
            for c in rec['cells']:
                if c['values'][metric] is not None:z[etas.index(c['eta']),bs.index(c['B'])]=c['values'][metric]
            finite=z[np.isfinite(z)];tz=tx(z,ref);ft=tx(finite,ref)
            norm=TwoSlopeNorm(vmin=min(-.05,ft.min()),vcenter=0,vmax=max(.05,ft.max()))
            cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#dce0e3')
            image=ax.imshow(np.ma.masked_invalid(tz),cmap=cmap,norm=norm,aspect='auto')
            for c in rec['cells']:
                r=etas.index(c['eta']);k=bs.index(c['B']);v=c['values'][metric]
                text='N/A' if v is None else label(v,ref,isgpt)
                if c['recorded_escapes']:text+='\nE'
                col='#27343c' if v is None or .14<=norm(tx(v,ref))<=.88 else 'white'
                ax.text(k,r,text,ha='center',va='center',fontsize=6.4 if len(bs)==6 else 7,color=col,linespacing=.9)
            ax.set(xticks=range(len(bs)),xticklabels=[str(b) for b in bs],yticks=range(len(etas)),
                yticklabels=[f'{e:.1e}'.replace('e-0','e-') if isgpt else f'{e:g}' for e in etas],
                xlabel='Batch (sequences)' if isgpt else 'Batch size',ylabel=r'Step size $\eta$')
            ax.set_title(f'({chr(97+2*i+j)}) '+title+('  '+r'$\widehat\Lambda$' if j==0 else '  '+r'$\widehat H$'),loc='left',fontsize=8 if isgpt else 8.5,pad=5)
            # The transform is only a color map. Tick labels retain true metric units.
            ticks=sorted(set([float(finite.min()),ref,float(finite.max())]))
            cb=fig.colorbar(image,ax=ax,pad=.025,shrink=.9,aspect=23)
            cb.set_ticks([float(tx(t,ref)) for t in ticks]);cb.set_ticklabels([label(t,ref,True) for t in ticks]);cb.ax.tick_params(labelsize=6)
    stem='empirical_atlas_'+suffix
    fig.savefig(FIG/(stem+'.pdf'),bbox_inches='tight',pad_inches=.025)
    fig.savefig(FIG/(stem+'.png'),dpi=240,bbox_inches='tight',pad_inches=.025);plt.close(fig)
    proof['pages'].append({'file':stem,'views':records})
proof['total_views']=sum(len(p['views']) for p in proof['pages'])
proof['total_missing_cells']=sum(r['missing_cells'] for p in proof['pages'] for r in p['views'])
proof['total_recorded_vision_escapes']=sum(r['recorded_escapes'] for p in proof['pages'] for r in p['views'] if r['dataset'].startswith('CIFAR'))
proof['raw_recomputed_statistics']=sum(r['raw_recomputed_statistics'] for p in proof['pages'] for r in p['views'])
(HERE/'atlas_provenance.json').write_text(json.dumps(proof,indent=2,allow_nan=False))
print(f"Saved{len(groups)} atlas pages, {proof['total_views']} views, {proof['total_missing_cells']} missing cells retained; {proof['raw_recomputed_statistics']} statistics recomputed.")
