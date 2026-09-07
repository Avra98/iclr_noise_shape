from pathlib import Path
import json,hashlib,zipfile,shutil
import sys
sys.path.insert(0,'/tmp/eoss_pdf_tools')
import pymupdf as fitz
from PIL import Image
ROOT=Path('/scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906');R=Path('/scratch/users/ghoshavr/eoss/papers/iclr_noise_shape');F=R/'img/current_results'
m=json.loads((ROOT/'FIGURE_MANIFEST.json').read_text());a=json.loads((ROOT/'NUMERIC_AUDIT.json').read_text());assert len(m)==73
for rec in m:
 p=F/(rec['name']+'.pdf');d=fitz.open(p);assert len(d)==1
 with Image.open(F/(rec['name']+'.png')) as im: im.verify()
for v in a.values():
 vals=v if isinstance(v,list) else [v]
 for q in vals:
  if isinstance(q,dict) and all(k in q for k in ['H','G','A1','A2']):
   assert q['H']<=q['G']+1e-12 and q['G']<=q['A1']+1e-12 and q['A1']<=q['A2']+1e-12
assert all(v==0 for v in a['replay_max_absolute_errors'].values())
assert all(z['residual_converged'] for z in a['alignment'])
pdf=R/'current_experiments.pdf';d=fitz.open(pdf);assert len(d)>=74
assert not any(s in (R/'current_experiments.log').read_text() for s in ['Overfull','Undefined control sequence','Missing $','undefined references'])
checks=dict(figures=len(m),pdf_pages=len(d),pdf_bytes=pdf.stat().st_size,replay_exact=True,eigenvector_probes=len(a['alignment']),max_relative_ritz_residual=max(z['relative_residual'] for z in a['alignment']),gain_order_verified=True,png_decode_verified=True,latex_compiled=True)
(ROOT/'VALIDATION.json').write_text(json.dumps(checks,indent=2));shutil.copy2(ROOT/'VALIDATION.json',F/'VALIDATION.json')
# Compact, portable Overleaf source archive; PNGs remain separately available.
zipout=R/'CURRENT_EXPERIMENTS_OVERLEAF.zip'
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED) as z:
 for name in ['current_experiments.tex','current_results_methods.tex','current_results_body.tex','current_results_appendix.tex','CURRENT_FIGURES_README.md']:
  z.write(R/name,name)
 for p in sorted(F.glob('*')):
  if p.suffix in ['.pdf','.json']:z.write(p,p.relative_to(R))
 z.writestr('latexmkrc',"@default_files = ('current_experiments.tex');\n")
preview=ROOT/'review';preview.mkdir(exist_ok=True)
for i in [2,4,5,7,11,75]:
 if i<len(d):d[i].get_pixmap(matrix=fitz.Matrix(1,1)).save(preview/f'page_{i+1:02d}.png')
# An entry point next to the new experiment outputs.
(ROOT/'README.md').write_text(f'# Current Overleaf figures\n\nReport: {pdf}\n\nGallery: {R}/current_figures.html\n\nOverleaf source: {zipout}\n\n73 PDF/PNG figure pairs, 12 vision architectures and GPT. New MLP alignment probes match the archived trajectory exactly. See VALIDATION.json and NUMERIC_AUDIT.json.\n')
print(json.dumps(checks,indent=2));print('Source ZIP bytes',zipout.stat().st_size)
