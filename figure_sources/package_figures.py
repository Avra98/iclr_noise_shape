from pathlib import Path
import json,shutil,hashlib,html,zipfile
ROOT=Path('/scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906')
REPO=Path('/scratch/users/ghoshavr/eoss/papers/iclr_noise_shape')
m=json.loads((ROOT/'FIGURE_MANIFEST.json').read_text());a=json.loads((ROOT/'NUMERIC_AUDIT.json').read_text())
def esc(s):
 for x,y in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#'),('^',r'\textasciicircum{}'),('~',r'\textasciitilde{}')]:s=s.replace(x,y)
 return s.replace('η',r'$\eta$').replace('×',r'$\times$').replace('–','--').replace('−','-').replace('Λ',r'$\Lambda$').replace('λ',r'$\lambda$').replace('θ',r'$\theta$').replace('≤',r'$\leq$').replace('≥',r'$\geq$').replace('→',r'$\to$').replace('√',r'$\sqrt{\ }$').replace('ψ',r'$\psi$')
methods=r'''\subsection{Directions, gains, and finite-window occupancy}
We distinguish the direction used for measurement from the statistic applied to its gain law. For plain SGD let $H_B(\theta)=\nabla^2 L_B(\theta)$. A unit direction $q$ defines the scalar compression
\begin{equation}
 a_B(q;\theta)=q^\top(I-\eta H_B(\theta))q=1-\eta q^\top H_B(\theta)q.
\end{equation}
The batch-gradient measurement uses $q_B=g_B/\|g_B\|$, where $g_B=\nabla L_B(\theta)$; its sampled curvature is $s_B=g_B^\top H_Bg_B/(g_B^\top g_B)$. This is the batch-gradient curvature statistic used by the vision heatmaps. The gain law is formed before taking moments: $\mathbb E|1-\eta s_B|^p$ is not $|1-\eta\mathbb E s_B|^p$. An unweighted average of these Rayleigh quotients also differs in general from a gradient-norm-weighted batch-sharpness ratio.

For the occupancy measurement, $q_*$ is a checkpoint-fixed leading full-data Hessian Ritz vector and $\theta_*$ is the checkpoint parameter vector, not an estimated minimizer. Every histogram uses $z_t=q_*^\top(\theta_t-\theta_*)$. We separately recompute a leading Ritz vector $q_1(t)$ and report $|q_1(t)^\top q_*|$. The plotting frame remains fixed even when the evolving eigenvector changes. Fixed checkpoint-gradient and random directions provide additional scalar comparisons.

For any one declared gain pool, we report four generalized means:
\begin{equation}
 \mathsf H=\bigl\langle |a|^{-1}\bigr\rangle^{-1},\qquad
 \mathsf G=\exp\langle\log|a|\rangle,\qquad
 \mathsf A_1=\langle|a|\rangle,\qquad
 \mathsf A_2=\sqrt{\langle|a|^2\rangle}.
\end{equation}
Their reference is one; $\Lambda=\log\mathsf G$ has reference zero. On the same pool $\mathsf H\leq\mathsf G\leq\mathsf A_1\leq\mathsf A_2$. Here $\mathsf H$ is the reciprocal of the inverse moment, so harmonic expansion means $\mathsf H>1$, equivalently $\langle|a|^{-1}\rangle<1$. This convention must not be confused with denoting the inverse moment itself by $H$. Inverse gains are not clipped; finite samples do not certify existence of a population negative moment.

A scalar compression is also distinct from a norm gain $\|(I-\eta H_B)q\|$. To assess transported perturbations we propagate $v_{t+1}=(I-\eta H_{B_t}(\theta_t))v_t$, normalizing only its length between updates. The plotted cumulative sum is $\sum_{t<T}\log(\|v_{t+1}\|/\|v_t\|)$ before normalization. This preserves directional coupling; resetting $v_t$ to $q_*$ would measure a different object.

\paragraph{Current observation.}
The CIFAR-10 SiLU MLP continuation uses IID subsets of size 4096, learning rate 0.04, and 4096 additional updates from checkpoint 10001. Loss on its 8192-example training subset decreases from 0.02485 to 0.00935. Fixed-coordinate windows show alternating two-lobe occupancy while the late transported log-growth rate is approximately $+0.00642$ per update. The late fixed-axis scalar log average can nevertheless be negative. New eigenvector probes show variable alignment, including sharp decreases; they do not support a permanently fixed leading eigendirection. The paired probes also show phase-dependent harmonic gains. These observations establish finite-window coexistence, not an invariant neural distribution or a universal harmonic-to-bimodal transition. The previously selected CNN fixed-axis candidate fails confirmation under the same shape screen.

\paragraph{Across architectures and optimizers.}
The architecture-level figures combine runs without separate seed panels. Heatmap cells average per-run statistics from the last three available checkpoints; different cells can use different occupied states and times. The gain-law comparison explicitly labels its architecture-specific learning rates. For GPT 168M on the FineWeb-Edu subset, distinguish raw Hessian/GGN probes from frozen-Adam probes. With $D=\operatorname{diag}(1/(\sqrt{\hat v}+\epsilon))$ fixed, the latter use
\begin{equation}
 s_{D,B}=\frac{(Dg_B)^\top H_B(Dg_B)}{g_B^\top Dg_B},\qquad a_{D,B}=1-\eta s_{D,B},
\end{equation}
with $H_B$ replaced by GGN when specified. This is a scalar compression in symmetrically preconditioned coordinates; it is not the full Adam state-space Jacobian. All GPT parents process 16.384 million tokens.
'''
(REPO/'current_results_methods.tex').write_text(methods)
# Ready-to-include main figure block, separate from the existing unfinished manuscript.
mainnames=['fig01_mlp_loss_alignment_bimodality','fig02_direction_resolved_ladder','fig04_cross_architecture_gain_laws','cnn_silu_grid_eta_batch','mlp_silu_grid_eta_batch','gpt_grid03_frozen_adam_hessian']
def figure(r,report=False):
 name=r['name']; size=r'width=\linewidth,height=0.74\textheight,keepaspectratio' if report else r'width=\linewidth,height=0.78\textheight,keepaspectratio'
 return '\n'+r'\begin{figure}[p]\centering'+'\n'+r'\includegraphics['+size+']{img/current_results/'+name+'.pdf}\n'+r'\caption{'+(r'\textbf{'+esc(r['title'])+r'.} ' if 'title' in r else '')+esc(r['caption'])+'}\n'+r'\label{fig:current-'+name.replace('_','-')+'}\n'+r'\end{figure}\clearpage'+'\n'
(REPO/'current_results_body.tex').write_text('\\input{current_results_methods}\n'+''.join(figure(r) for r in m if r['name'] in mainnames))
(REPO/'current_results_appendix.tex').write_text(''.join(figure(r) for r in m if r['name'] not in mainnames))
head=r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=16mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,hyperref}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\title{Current neural-network experiments: directions, gain ladders, and occupancy}
\author{Figure review for the Overleaf manuscript}
\date{6 September 2026}
\begin{document}\maketitle
\noindent The main examples precede the architecture/dataset atlas. Vector PDFs and PNGs are supplied for every figure. The atlas reuses the audited experiment data and plots, not the legacy draft's empirical claims. No separate seed figures or result tables are included.
\input{current_results_body}
\section*{Architecture and dataset atlas}
\input{current_results_appendix}
\end{document}
'''
(REPO/'current_experiments.tex').write_text(head)
shutil.copy2(ROOT/'NUMERIC_AUDIT.json',REPO/'img/current_results/NUMERIC_AUDIT.json');shutil.copy2(ROOT/'FIGURE_MANIFEST.json',REPO/'img/current_results/FIGURE_MANIFEST.json')
readme='''# Current Overleaf figures

Open `current_experiments.pdf` for the complete review, or compile `current_experiments.tex` in Overleaf.

For the manuscript, insert `\\input{current_results_body}` after the Empirical Observations heading. Insert `\\input{current_results_appendix}` in the appendix if the complete atlas is wanted. The existing main_2.tex is unchanged: its abstract and older theory claims are not rewritten by this figure package.

- Main figure: img/current_results/fig01_mlp_loss_alignment_bimodality.pdf (and .png).
- Direction-resolved four means: fig02_direction_resolved_ladder.
- Paired phases: fig03_paired_phase_ladder.
- Cross-architecture gain laws: fig04_cross_architecture_gain_laws.
- Every vision architecture: curvature, four means, gain law, loss/gain dynamics, and batch-rate grid.
- GPT: token dynamics, gain ladder/law, diagnostic controls, and four batch-rate grids.

All means distinguish batch-gradient vs frozen directions. H denotes reciprocal inverse moment, not the inverse moment itself. Raw/preconditioned GPT diagnostics are explicitly separated. Histograms are finite-window occupancy, not proved stationary densities. The new replay adds sparse full-subset leading-Ritz alignment probes; it does not assume alignment. Numerical replay checks and residuals are in NUMERIC_AUDIT.json.

Rebuild scripts and GPU submission are under /scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906/source.
'''
(REPO/'CURRENT_FIGURES_README.md').write_text(readme)
page='<html><head><meta charset="utf-8"><title>Current Overleaf figures</title></head><body style="max-width:1100px;margin:auto;font-family:Arial"><h1>Current neural-network figure package</h1><p><a href="current_experiments.pdf">Complete PDF</a></p>'
for r in m:page+=f'<h2>{html.escape(r.get("title",r["name"]))}</h2><a href="img/current_results/{r["name"]}.pdf">Vector PDF</a><p>{html.escape(r["caption"])}</p><img style="width:100%" loading="lazy" src="img/current_results/{r["name"]}.png">'
(REPO/'current_figures.html').write_text(page+'</body></html>')
print('Packaged',len(m),'figures')
