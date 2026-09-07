from pathlib import Path
import json,sys,subprocess,html,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=Path(__file__).resolve().parents[3]
(ROOT/'logs').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_window_bimodality import shape,controls,clean
import render
records=[];stats=[]
anchor=ROOT/'anchor'
records.append(render.render(render.load_case(anchor,True),'mlp_silu_B4096_anchor_revised'))
for p in sorted((ROOT/'runs').glob('*/meta.json')):
 m=json.loads(p.read_text())
 if m['status']!='completed':continue
 d=render.load_case(p.parent);records.append(render.render(d,p.parent.name));t=np.load(p.parent/'trace.npz');x=t['projection'][:,0];ww=[]
 for lo in [512,1024,2048,3072]:
  xx=x[lo:lo+512];sh=shape(xx);ct=controls(xx,sh);ww.append(dict(start=lo,shape=sh,controls=ct))
 scans={}
 for width in [64,256,512,1024]:
  results=[]
  for lo in range(0,4096-width+1,width//2):
   xx=x[lo:lo+width];sh=shape(xx);ct=controls(xx,sh)
   results.append(dict(start=lo,passes=sh['passes'],**ct))
  scans[str(width)]=results
 fl=d['full_loss'];stats.append(dict(id=p.parent.name,arch=d['arch'],batch=d['batch'],parent_step=d['parent_step'],loss_start=fl[0]['loss'],loss_end=fl[-1]['loss'],windows=ww,scans=scans))
 print(p.parent.name,[w['shape']['passes'] for w in ww],flush=True)
(ROOT/'SHAPE_AUDIT.json').write_text(json.dumps(clean(stats),indent=2));(ROOT/'FIGURE_MANIFEST.json').write_text(json.dumps(records,indent=2))
caption=r'''Loss is shown for both training minibatches and the full 8192-example training subset. The plotting direction $q_*$ and center $\theta_*$ stay fixed at the parent checkpoint; $\theta_*$ is not a verified minimizer. Alignment with separately recomputed leading Hessian Ritz vectors is measured on the full subset, not assumed. Four declared 512-update windows and one 64-update detail are displayed without selection by modality. The lower four panels show fresh conditional scalar gains, comparing $a_{*,B}=1-\eta q_*^\top H_Bq_*$ and $a_{g,B}=1-\eta g_B^\top H_Bg_B/(g_B^\top g_B)$. Their summaries are $H=1/\langle|a|^{-1}\rangle$, $G=\exp\langle\log|a|\rangle$, $A_1=\langle|a|\rangle$, and $A_2=\sqrt{\langle|a|^2\rangle}$, all with reference one. Except for the original anchor, each state has 64 fresh probes every256 updates; the anchor uses 128 probes initially and24 every512 updates. Only even-update states appear here; no phase-averaged or stationary claim follows. No inverse-gain floor is applied.'''
body=''
for r in records:
 title=f"{render.LABELS.get(r['arch'],r['arch'])}, batch {r['batch']}, learning rate {r['eta']:g}, parent checkpoint {r['parent_step']}"
 body+=r'\begin{figure}[p]\centering'+'\n'+r'\includegraphics[width=\linewidth,height=0.77\textheight,keepaspectratio]{img/appendix_directional_ladders/'+r['name']+'.pdf}\n'+r'\caption{\textbf{'+title+'.} '+caption+'}\n'+r'\end{figure}\clearpage'+'\n'
(REPO/'appendix_directional_ladders_body.tex').write_text(body)
head=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=12mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,hyperref}
\begin{document}
\section*{Directional occupancy and gain ladders across networks and batches}
'''
head+=f"This snapshot contains the original MLP anchor and {len(stats)} completed continuations from the declared four-architecture, four-batch sweep. Incomplete jobs are excluded from numerical conclusions and remain recorded in the campaign status. Different parent checkpoints are explicitly labeled. Each continuation uses IID subsets and4096 updates. Plots are organized by architecture and batch, with no separate seed panels.\n"
head+=r'\input{appendix_directional_ladders_body}\end{document}'
(REPO/'appendix_directional_ladders.tex').write_text(head)
for _ in range(2):
 with (ROOT/'logs/appendix_latex.log').open('w') as log:
  subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','appendix_directional_ladders.tex'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
with zipfile.ZipFile(REPO/'APPENDIX_DIRECTIONAL_LADDERS_OVERLEAF.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['appendix_directional_ladders.tex','appendix_directional_ladders_body.tex']:z.write(REPO/name,name)
 for r in records:
  p=REPO/'img/appendix_directional_ladders'/(r['name']+'.pdf');z.write(p,p.relative_to(REPO))
 z.writestr('latexmkrc',"@default_files = ('appendix_directional_ladders.tex');\n")
page='<html><head><meta charset="utf-8"></head><body style="max-width:1100px;margin:auto;font-family:Arial"><h1>Directional histograms and gain ladders</h1><p><a href="appendix_directional_ladders.pdf">Complete current PDF</a></p>'
for r in records:page+=f'<h2>{html.escape(r["name"])}</h2><a href="img/appendix_directional_ladders/{r["name"]}.pdf">Vector PDF</a><img loading="lazy" style="width:100%" src="img/appendix_directional_ladders/{r["name"]}.png">'
(REPO/'appendix_directional_ladders.html').write_text(page+'</body></html>')
print('Published',len(records),'figure pairs',flush=True)

# Keep the compact GitHub experiment package current after scheduled publication.
exporter=ROOT/"source/export_github.py"
if exporter.exists():subprocess.run([sys.executable,str(exporter)],check=True)
