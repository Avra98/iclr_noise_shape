# Recovered Overleaf and experimental work

Audit time: 2026-09-07T01:18:19.982493+00:00

## What remains in the requested folder

`eoss/results/NEURAL_NETWORK_RESULTS_20260905/` still contains the seven numbered collections. Before this audit it had 12,973 files (1.840 GB): original networks, recomputed older results, new network sweeps, optimizer controls, nonconvex network experiments, additional diagnostics, and fixed-frame bimodality. Its original SOURCE_MANIFEST lists2,409 files: every one still exists with its recorded byte size, and all2,409 recorded source paths also exist.

Comparison with `NEURAL_NETWORK_RESULTS_20260905.zip` found zero missing archived files. Content CRCs differ only for README.md and index.html, which were updated after that archive; archived figure content is intact. This archive covers the original gallery, not the later checkpoint-snapshot additions.

## What is missing locally

The whole `07_FIXED_FRAME_BIMODALITY/checkpoint_snapshot/` directory is absent. The checkpoint inventory identifies333 missing protected checkpoint copies:

- 112 have an existing original source with an exact SHA256 match to the saved frame's checkpoint fingerprint.
- 213 have an existing source file but no saved frame fingerprint available for this check; exact identity is unverified.
- 8 also have their recorded original checkpoint path missing. Some have newer or alternative weights in the source folder, but those are not automatically the same training state.

The original MLP anchor (checkpoint10001, batch4096) is among the112 exact matches. Its source still exists at `results/marc_moment_campaign_20260905/vision/mlp_silu/eta0.04/seed88882/B4096/latest_resume.pt`, with SHA256 `38913367ad977256c41e061dec5852154ffe8fa083146cc084dd767414f65366`. Its traces, fixed frame, gain probes, and figures are also present. Replotting uses the saved arrays and does not require restoring model weights.

No checkpoint copies were restored into the deleted directory. The recovery map records how an exact-matched source can be used without duplicating storage. Presence/absence establishes the current local state, not who deleted a file or whether it was uploaded. Google Drive verification failed because rclone remote `grive:` currently has an empty token. Thus cloud-copy existence is not verified.

The exact missing source paths are:

- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/vision/resnet32/eta0.05/seed88881/B2048/latest_resume.pt`
- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/vision/resnet32/eta0.05/seed88881/B32/latest_resume.pt`
- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/vision/resnet32/eta0.05/seed88881/B8/latest_resume.pt`
- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/vision/resnet32/eta0.1/seed88883/B4096/latest_resume.pt`
- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/vision/wrn_no_bn/eta0.002/seed88881/B2048/latest_resume.pt`
- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/vision/wrn_no_bn/eta0.002/seed88881/B32/latest_resume.pt`
- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/vision/wrn_no_bn/eta0.002/seed88881/B4096/latest_resume.pt`
- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/vision/wrn_no_bn/eta0.002/seed88881/B8/latest_resume.pt`

## Where the recent work actually lives

- `eoss/papers/iclr_noise_shape/`: the existing Git checkout for `Avra98/iclr_noise_shape`. It now contains the root working manuscript, Overleaf figure modules, PNG/PDF assets, and compact experiment export.
- `eoss/papers/iclr_stochastic_stability_20260906/`: the earlier audited nine-page main manuscript, full37-page proof/experiment version, source ZIP and claim audit. An unchanged source/reference copy is also kept in the Git checkout's `manuscript_20260906/`.
- `eoss/results/CNN_DIRECTIONAL_WINDOWS_20260905/`: original four4096-update continuations, iid MLP control, full tangent traces, fresh paired-state gains, and short-window figures.
- `eoss/results/OVERLEAF_FIGURES_20260906/`: exact-replay eigenvector alignment audit and the73-figure assembly.
- `eoss/results/APPENDIX_DIRECTIONAL_LADDERS_20260906/`: four architectures crossed with batches32,128,512,4096, with fixed-coordinate histograms, eigenvector alignment and gain ladders. At the recovery snapshot11 cases completed; five retries were running or queued. Earlier interrupted attempts are retained. The final report builder is scheduled after those jobs.
- `handoff/eoss/`: theory proofs, older handoffs, manuscript handoff, and this recovery note.

The revised main appendix figure is `img/appendix_directional_ladders/mlp_silu_B4096_anchor_revised.png` inside the Git checkout. Its lower panels are time-varying H,G,A1,A2, separately showing fixed-axis and batch-gradient gains. It omits the discarded transported-growth/static-ladder panels. The current multi-network appendix is `appendix_directional_ladders.pdf`; its LaTeX entry is `appendix_directional_ladders.tex`.

## The additional Overleaf files recovered from GitHub

Fetching origin recovered commit `7da61b8` (Updates from Overleaf,6 September2026). It moved the former root drafts and added two further theory manuscripts:

- Old `main.tex` is now `old_versions/last_version_avrajit.tex`.
- Old `main_2.tex` is now `old_versions/last_version_pier_draft_intro.tex`.
- `old_versions/post_harmonic_self_stabilization.tex` is present.
- `old_versions/stochastic_self_stabilization_regression_paper.tex` is present.

These files are retained as distinct drafts. The current main document assembles the previously audited ICLR manuscript and the new empirical appendix. It does not silently treat every proposition in the other drafts as a verified theorem.

Missing conversation messages cannot be reconstructed as exact chat history from these artifacts. This document recovers the files, chronology and experimental state that are verifiable on disk and in Git.

## Reproducibility and scientific interpretation

The figure ladder uses H=1/mean(1/abs(a)), G=exp(mean(log(abs(a)))), A1=mean(abs(a)), and A2=sqrt(mean(abs(a)^2)). The batch-gradient compression and checkpoint-fixed compression are separate. Histograms measure a fixed coordinate, not a distribution of gains. The MLP4096 example is strongest; some CNN shape candidates appear only in64-update windows. No universal harmonic-to-bimodal implication or stationary neural law has been established. Some leading Ritz directions rotate, and the CNN512 frozen frame has an explicitly flagged residual above tolerance.

## Audit files

`MANIFEST_COMPARISON.csv`, `CHECKPOINT_REFERENCE_COMPARISON.csv`, `CHECKPOINT_RECOVERY_MAP.csv`, `MISSING_ORIGINAL_ALTERNATIVES.json`, `ZIP_CONTENT_DIFFERENCES.json`, and `LOCAL_COUNTS.json` are in `eoss/results/RECOVERY_AUDIT_20260906/`. Counts for running experiment folders are a point-in-time inventory.
