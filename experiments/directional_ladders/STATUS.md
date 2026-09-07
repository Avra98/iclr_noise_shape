# Directional appendix campaign

11 of 16 declared continuations completed in the first attempt. Five incomplete attempts are preserved in interrupted_attempts/ and restarted with periodic saved-state resumption under Slurm array3515484. Final report rebuild is scheduled as dependent job3515490. The source now supports resuming automatically requeued jobs from their saved model and sampler state.

Current completed-case report: /scratch/users/ghoshavr/eoss/papers/iclr_noise_shape/appendix_directional_ladders.pdf

Current PNG/PDF figures: /scratch/users/ghoshavr/eoss/papers/iclr_noise_shape/img/appendix_directional_ladders/

The revised anchor replaces transported-growth and static-mean panels with four conditional gain curves over training. The same layout is used across completed networks and batches.

Preliminary findings: among the first eleven completed cases, MLP SiLU B4096 is the strongest fixed-axis two-lobe case. From checkpoint20000, it passes the declared512-step histogram screen in the first two displayed windows but not the latter two; the width-sensitivity scan records additional overlapping passing windows. CNN B128 andB512 have a few passing64-step windows but none at widths256 or longer. Other completed cases have no passing long fixed-axis window. These are exploratory finite-window shape screens, not stationarity or causal harmonic-threshold proofs.

SHAPE_AUDIT.json records the bandwidth checks, temporal controls, and all scanned windows. FIGURE_MANIFEST.json records source paths and gain estimates.
