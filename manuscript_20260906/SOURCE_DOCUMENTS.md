# Source documents and experimental provenance

Canonical root: `/scratch/users/ghoshavr/eoss/papers/iclr_stochastic_stability_20260906`.
Original reports and experiments are retained in their original locations; this manuscript is a new, compact presentation.

## Theory inputs

- `/accounts/projects/binyu/ghoshavr/handoff/eoss/claude_drafts_bimodality/edge_ladder_report.pdf` — motivating scalar moment ladder and harmonic-shape proposal; not used as an unquestioned theorem source.
- `/accounts/projects/binyu/ghoshavr/handoff/eoss/MATRIX_RENEWAL_PROOF_20260905.tex` and `.pdf` — matrix renewal starting point.
- `/accounts/projects/binyu/ghoshavr/handoff/eoss/HIGH_DIMENSIONAL_REGRESSION_PROOF_20260905.tex` and `.pdf` — exact matrix recurrence/projection counterexample.
- `/accounts/projects/binyu/ghoshavr/handoff/eoss/POST_LYAPUNOV_PROOF_AUDIT_20260905.md` — prior proof gaps and distinctions.
- `/accounts/projects/binyu/ghoshavr/handoff/eoss/sgd_equilibrium_20260905/paper/` — exact scalar/nonconvex/piecewise proofs and prior bibliography.
- `/accounts/projects/binyu/ghoshavr/handoff/eoss/sgd_equilibrium_20260905/theory/NOVELTY_STRESS_TEST.md` and `CHAOS_NOVELTY_COMPARISON.md` — comparison against existing random-recursion and chaos literature.

## Current empirical inputs

- `/scratch/users/ghoshavr/eoss/results/CNN_DIRECTIONAL_WINDOWS_20260905/DIRECTIONAL_STABILITY_AND_BIMODALITY.pdf` and its raw branches/analysis/manifests — all four fixed-frame continuations.
- `/scratch/users/ghoshavr/eoss/results/ALL_EMPIRICAL_RESULTS_20260905/PART_01_CONCISE/SOURCE_MANIFEST.json` — accepted architecture/dataset collection, including archived and new sweeps.
- `/scratch/users/ghoshavr/eoss/results/marc_moment_campaign_20260905/figure_review_20260905/figure_inputs.pkl` — saved new-campaign gain pools.
- `/scratch/users/ghoshavr/eoss/results/nonconvex_campaign_20260905/RESEARCH_VERDICT.md`, `EVIDENCE_LEDGER.md`, and completed execution summaries — scalar, constructed-network, and ordinary-network controls.

Exact per-figure source paths, inputs and aggregation checks are in `contributions/empirical/numerical_audit.json`, `atlas_provenance.json`, and `nonconvex_provenance.json`. The analytic figure has its own parameter audit in `figures/theory_inverse_return.json`.

## External primary references

The bibliography records the primary Kesten, Goldie, Markov-renewal, nonlinear random-system, and optimization papers cited in the text. In particular, the user-supplied arXiv papers 2412.20553, 2607.21716, and 2206.04172 are cited in their appropriate roles. Classical results are attributed; source titles and publication metadata were checked against primary author/journal/arXiv pages.

## Format

Official ICLR 2026 style files were obtained from the official Master-Template repository. The verified 2026 author guide specifies at most nine main-text pages, excluding references and appendices. A current 2027 author guide was not verified. The working header was changed in the document, without altering the official style's margins, font sizes, or line spacing, so the PDF does not falsely state submission or acceptance.
