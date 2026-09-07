# SGD stability, nonlinear return, and neural occupancy

**Compile `main.tex` in Overleaf.** The working ICLR manuscript has nine main-text pages; `main.pdf` includes references, proofs and the current empirical appendix (50 pages at the recovery snapshot). It is a working draft, not a submitted or accepted paper.

## Read and edit

- `main.tex`, `sections/`, `contributions/`, `references.bib`: active manuscript and proofs.
- `appendix_directional_ladders.pdf` / `.tex`: current architecture-by-batch figures with loss, frozen-coordinate histograms, measured direction alignment and four conditional gain curves.
- `img/appendix_directional_ladders/`: revised figures as vector PDF and PNG.
- `current_experiments.pdf` / `.tex`: broader 73-figure review with architecture/dataset heatmaps and gain laws.
- `old_versions/`: the two former Overleaf main files and the additional post-harmonic and regression theory drafts, preserved from the incoming GitHub commit.
- `manuscript_20260906/`: unchanged reference copy of the earlier audited manuscript.

## Experiments

`experiments/directional_ladders/` contains plotting/probing source, configuration, compact raw arrays for completed runs, and the numerical shape audit. Its README distinguishes portable figure reproduction from the external model/data/checkpoint requirements of new training. Weights and datasets are not committed.

The expanded campaign uses four network families and batches 32, 128, 512, and 4096. At this commit's snapshot, 11/16 continuations are complete; five retries remain active or queued. Figures preserve null results and label parent checkpoint differences. The current appendix is not represented as a completed 16-case sweep.

## Main interpretation

A gain requires both a direction and a quantity. We distinguish checkpoint-fixed compression, batch-gradient compression and continuously transported norm growth. Harmonic gain is the reciprocal inverse moment. Finite-window two-lobe occupancy is not proof of a stationary neural law or a universal harmonic-instability threshold for bimodality.

`RECOVERED_WORK.md` and `recovery_audit/` explain what survives locally, what checkpoint copies are missing, and what has an exact verified source elsewhere. The original MLP anchor is recoverable by exact SHA256 match.

## Build

Run `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. Figure reproduction additionally uses NumPy, SciPy and Matplotlib. `latexmkrc` selects `main.tex` by default. Pull GitHub changes into the linked Overleaf project to make the repository update visible there.
