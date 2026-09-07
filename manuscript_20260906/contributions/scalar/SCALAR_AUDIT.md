# Scalar theorem audit for the ICLR manuscript

This module was independently checked against the supplied proof audits and re-derived. The main-text candidate is `scalar_main.tex`; detailed proofs are `scalar_proofs.tex`. The main text can be shortened substantially by retaining only the diagnostic ordering, inverse central-mass theorem, and smooth SGD recurrence theorem. All labels use `sc:`.

## Recommended sequence and attribution

1. **Deterministic quadratic product.** One multiplier gives one coincident boundary. This is elementary background, not a contribution.
2. **Scalar iid quadratic product and affine Kesten–Goldie baseline.** Use the absolute gain M=|1−ηh_B|; m(p)=E M^p, λ=E log M, H=1/m(−1). Never take a general real power of a signed coefficient without its absolute value. The product identity and Jensen ordering are proved completely. Affine upper-tail renewal is attributed to Kesten/Goldie rather than claimed as a new theorem.
3. **Central mass after positive local log drift, with nonlinear return.** An exact reflected-log model gives an invariant law and the inverse root m(−α)=1. This is a classical Cramér/Lindley renewal construction recast as a central-amplitude law; do not claim first discovery of an extension of Kesten theory. Complete likelihood/overshoot proof is supplied. A bounded continuous compactly supported log-gain density is an explicit sufficient hypothesis for the density version, proved using ladder-height renewal rather than differentiation of a CDF.
4. **A solvable calibration.** M=exp(V−U), U~Exp(b), V~Exp(c), b>c>0 yields α=b−c and exact density plus a real outer-boundary atom. This verifies density regularity without an asymptotic theorem. It is a saturation model, not a quadratic SGD update.
5. **Exact smooth SGD return.** Sample regression plus a saturated nonlinear regularizer has contracting outer slopes and expansive central slopes. The general theorem proves unique attracting stationary law with every polynomial moment, for 0<s<μ, β>8μ, 0<η<2/(μ+s), positive Gaussian residual variance. No clipping is applied. The proof handles dependence between curvature and innovation variance by conditioning on h.
6. **Matrix and empirical sections.** Scalar density statements do not automatically transfer to fixed projections in a high-dimensional model. The matrix audit owns the angular/pressure extension.

## Main exact smooth example

β=.92, μ=.08, s=.05, η=2.015, B=1, h=.95 or1.05 equiprobably, and residual e~N(0,σ²), σ>0.

- Exact loss: .5(√h x−e)²−.46[x²−log(1+x²)].
- Population loss: .04x²+.46log(1+x²)+constant, unique minimum zero, min Hessian −.035.
- Central signed Jacobians: −.91425 and−1.11575.
- λ0 =0.009937801287298001.
- H0 =1.004999445812808.
- Central inverse root α0=2.0174266710217497.
- Outer slopes: .93955,.73805.
- Positive λ0, H0>1, and the existence of α0 are exact sign conclusions; the decimal root is numerical. The adjacent root bracket can be checked by the included validation script.
- **Not proved:** the exact noisy SGD density has central power α0−1, its trajectory exponent is positive, or its law has two modes. Fixed additive Gaussian noise gives a positive density at zero, so literal vanishing there is false.

The same loss at η=8 gives central gains 6.6,7.4 and outer slopes .76,−.04. This is a clean far-post-central-harmonic recurrence control, but there is no finite central inverse root because all central gains exceed one. At this larger step, occupied Jacobian zeros force the stationary occupied inverse moment at order1 to diverge; the corresponding occupied harmonic gain is zero even though H0>1.

## Invalid stronger statements that must not enter the manuscript

- **Positive global affine Lyapunov exponent plus nondegenerate Gaussian forcing cannot have a stationary law.** An explicit weighted-noise escape proof is included. Point masses in degenerate/noiseless models are exceptions, so nondegeneracy matters.
- **H>1 does not universally imply bimodality, even in scalar regenerative models.** The counterexample in the proof appendix has λ>0, H=5/4, α=2 but a centrally peaked exact invariant density because reinjection places mass near zero. This directly contradicts the overly broad Theorem6 claim in `edge_ladder_report.pdf`.
- **Density depletion is not exactly two modes.** Evenness, continuity, zero at the origin, decay at infinity, and positive mass give at least two off-center global maximizers. A unique maximum on each half-line or another global shape condition is needed for exactly two. The signed saturated model has outer atoms and should not be described as a smooth two-mode density.
- **A raw cubic explicit update with Gaussian forcing is not globally confined.** The supplied POST_LYAPUNOV audit proves escape. Its local cubic restoring term explains local amplitude reduction, not a stationary distribution on the entire real line.
- **Positive-power drift has the wrong sign in the expanding linear core.** If λ>0 then E M^p≥exp(pλ)>1 for p>0. Inverse powers contract away from zero; that is not return from infinity.
- **A finite inverse-moment sample does not establish finiteness of the population inverse moment.** Gains with a nonzero density at zero give E|A|^(−1)=∞, even if the offending samples are rare.
- **Moment-matched diffusion is not an exact finite-step theorem.** The confined cubic diffusion has an exact two-mode transition and can be included as an additional analytic control. It does not certify the neural trajectories or the exact discrete Gaussian-gain harmonic moment.

## Source provenance

Local sources read:

- `handoff/eoss/POST_LYAPUNOV_PROOF_AUDIT_20260905.md`.
- `handoff/eoss/HIGH_DIMENSIONAL_REGRESSION_PROOF_20260905.tex`.
- `handoff/eoss/sgd_equilibrium_20260905/paper/theory.tex`.
- `handoff/eoss/sgd_equilibrium_20260905/paper/fractional_drift_extension.tex`.
- `handoff/eoss/sgd_equilibrium_20260905/paper/scalar_supercritical.tex`.
- `handoff/eoss/sgd_equilibrium_20260905/paper/piecewise_exact.tex`.
- `handoff/eoss/sgd_equilibrium_20260905/paper/diffusion_theory.tex`.
- Extracted supplied PDF `theory/pdf_text/stochastic_self_stabilization_regression_paper.txt`, Theorem8 and Corollary9.

Primary sources checked online:

- Kesten (1973), DOI https://doi.org/10.1007/BF02392040. Publisher metadata/repository search found the primary manuscript; browser fetch of Tsinghua PDF failed, so this module does not claim a newly checked full Kesten proof.
- Goldie (1991), DOI https://doi.org/10.1214/aoap/1177005985. Publisher page is served through a viewer. Classical theorem attribution is limited to the standard baseline already used in the supplied drafts.
- Blanchet and Glynn (2006), author page https://web.stanford.edu/~glynn/papers/2006/BlanchetG06.html; confirms the Cramér–Lundberg random-walk/overshoot framework.
- Hairer and Mattingly (2008), https://arxiv.org/abs/0810.2777; checked the Harris drift/minorization statement.

`scalar_references.bib` contains these entries and the established nonlinear affine-tail paper Alsmeyer–Brofferio–Buraczewski (2023). The latter should be cited if the earlier η19 even-moment/infinite-variance construction is discussed as prior-theorem-compatible, rather than a novel general tail criterion.
