# Current Overleaf figures

Open `current_experiments.pdf` for the complete review, or compile `current_experiments.tex` in Overleaf.

For the manuscript, insert `\input{current_results_body}` after the Empirical Observations heading. Insert `\input{current_results_appendix}` in the appendix if the complete atlas is wanted. The existing main_2.tex is unchanged: its abstract and older theory claims are not rewritten by this figure package.

- Main figure: img/current_results/fig01_mlp_loss_alignment_bimodality.pdf (and .png).
- Direction-resolved four means: fig02_direction_resolved_ladder.
- Paired phases: fig03_paired_phase_ladder.
- Cross-architecture gain laws: fig04_cross_architecture_gain_laws.
- Every vision architecture: curvature, four means, gain law, loss/gain dynamics, and batch-rate grid.
- GPT: token dynamics, gain ladder/law, diagnostic controls, and four batch-rate grids.

All means distinguish batch-gradient vs frozen directions. H denotes reciprocal inverse moment, not the inverse moment itself. Raw/preconditioned GPT diagnostics are explicitly separated. Histograms are finite-window occupancy, not proved stationary densities. The new replay adds sparse full-subset leading-Ritz alignment probes; it does not assume alignment. Numerical replay checks and residuals are in NUMERIC_AUDIT.json.

Rebuild scripts and GPU submission are under /scratch/users/ghoshavr/eoss/results/OVERLEAF_FIGURES_20260906/source.
