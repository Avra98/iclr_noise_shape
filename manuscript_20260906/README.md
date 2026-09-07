# Beyond Local Lyapunov Stability
## Inverse Moments and Nonlinear Return in SGD

Start with [the full manuscript](ICLR_STOCHASTIC_STABILITY_FULL.pdf), or [the nine-page main text](MANUSCRIPT_MAIN_9_PAGES.pdf).

- Main text: **9 pages** in the official ICLR layout.
- Full manuscript: **37 pages**, including references, full proofs and extra experiments; **0.75 MB**.
- Editable source: [main.tex](main.tex), [references.bib](references.bib), and `sections/` / `contributions/`.
- Figures: **13 vector PDFs with PNG counterparts** in `figures/`; the main text uses three.
- Audit: [CLAIM_AUDIT.md](CLAIM_AUDIT.md), [SOURCE_DOCUMENTS.md](SOURCE_DOCUMENTS.md), [VALIDATION.json](VALIDATION.json).

## Reading order

1. Deterministic and scalar stochastic quadratics; the positive, logarithmic and negative moment ordering.
2. Scalar returned-process renewal and the harmonic central-shape boundary.
3. Matrix renewal with evolving directions and explicit return assumptions.
4. Smooth nonconvex SGD return, exact equilibrium balance, and a piecewise control with positive stationary tangent growth.
5. Neural scalar-versus-transported diagnostics, CNN/MLP stability heatmaps, and the MLP fixed-frame occupancy experiment.
6. Related work and scope. Full proofs and counterexamples then precede the empirical appendix and architecture atlas.

The manuscript attributes the classical probability results. Its strongest neural conclusion concerns finite-time occupancy and directional growth, with no claim of a proved stationary neural bimodal law. General nonlinear renewal transfer remains conditional; see the claim audit.

## Main figures

- [Figure 1: exact inverse-moment / central-density example](figures/theory_inverse_return.png).
- [Figure 2: CNN and MLP step-size / batch-size heatmaps](figures/empirical_stability_heatmaps.png).
- [Figure 3: MLP fixed-frame dynamics and harmonic phase](figures/empirical_iid_mlp_dynamics.png).

The appendix retains ten other vision architectures, the four GPT metric views, smooth nonconvex noise controls, constructed neural controls, and ordinary-network counterevidence. Missing cells and cutoff escapes are visible. There are no seed-specific figure pages or result tables.

## Build and reproduce

The supplied sources and figures compile without access to the original result directories:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

To rebuild the named deliverables and validation manifest as well:

```sh
./build.sh
```

The finishing script additionally uses PyMuPDF and pikepdf. It asserts that the main text ends on page nine and references start on page ten, checks PDF syntax, renders every page, verifies all PNG counterparts, and creates the source archive.

Regenerating the analytic figure uses `python build_theory_figure.py`. Empirical plotting scripts are in `contributions/empirical/`; those require the original result inputs at the provenance paths. The manuscript source archive supplies every figure needed for offline LaTeX compilation, but is not an archive of all training checkpoints.

## Format and authorship

This is an anonymous working manuscript. It uses the verified official ICLR 2026 template with the working header substituted in the document; margins, type sizes and spacing are unchanged. No submission or acceptance is implied. Conference-year details and authorship remain editable before any actual submission.
