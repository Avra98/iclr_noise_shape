# Independent scalar and equilibrium cross-review

Date: 2026-09-06. Reviewed `contributions/scalar/scalar_main.tex`, `scalar_proofs.tex`, `sections/scalar.tex`, `sections/equilibrium.tex`, `sections/piecewise_appendix.tex`, and the introduction/prior-work framing. No source theorem files were edited in this review.

## Verdict

No substantive mathematical error found in the newly added concrete ladder-density proof, the smooth nonconvex SGD recurrence construction, or the occupied-positive-exponent piecewise SGD construction. All three address their principal possible gaps explicitly. The claims do not depend on the older incorrect PDF experiments.

Two small clarifications are recommended:

1. The final scalar diffusion paragraph should say that its **nontrivial zero-additive-noise invariant density on the punctured line exists when a > nu^2/2**. Otherwise alpha=2a/nu^2-1 is nonpositive and the proposed center density is not integrable. The positive-additive-noise density and its exact mode-count statement are correct as written.
2. Main equilibrium prose should call the positive tangent statement separation of **infinitesimal common-noise perturbations**. Bounded physical trajectories do not maintain exponential separation forever for a fixed finite initial displacement. The theorem itself correctly concerns the derivative product.

## Concrete ladder-density proof

The proof's bounded continuous compact-support density condition suffices:

- Negative mean bounded log increments imply summable probabilities of occupying a fixed interval near zero at large times (Hoeffding gives exponential decay).
- The first positive ladder-height density is therefore bounded and continuous in its support interior, with one-sided endpoint limits. Its compact support makes it directly Riemann integrable.
- The defective ladder mass p is strictly below one: p=1 would yield infinitely many strict upper records, contradicting the negative-drift strong law.
- Exponential tilting at the first ladder epoch makes the ladder-height measure a proper probability. It has bounded positive heights, positive finite mean, and an absolutely continuous nonarithmetic law.
- Applying the ordinary positive-increment key renewal theorem to that density gives the renewal-density limit directly. Undoing the tilt yields the supremum density and the constant C=(1-p)/(alpha mu_hat).
- The cap atom (1-p) is explicitly retained. The proof does not incorrectly differentiate an asymptotic CDF.

The compact-entry regenerative extension also has sufficient domination: original entry and exit overshoot lie in compact intervals, shifted renewal limits are uniform there, and fixed-width interval bounds permit geometric summation toward minus infinity. The density argument uses a continuous compactly supported convolution test function, which is valid.

The scalar moment ordering and reciprocal-root shape classification are correct under the displayed finite-moment assumptions. The affine nonexistence proof with positive outer logarithmic drift correctly handles bounded nonlinear remainder and conditionally nondegenerate Gaussian forcing; its stationary escape contradiction does not assume that every path escapes.

## Smooth nonconvex SGD recurrence

All structural checks pass:

- Every sampled loss is coercive because h-beta >= mu-s>0.
- The population derivative is x[mu+beta/(1+x^2)], so zero is the unique minimizer.
- The minimum population second derivative is mu-beta/8<0, attained at x^2=3.
- The exact gradient map has outer slopes 1-eta(mu±s), both strictly below one in absolute value under the stated step-size interval, plus a remainder bounded by eta beta/2.
- Polynomial Foster inequalities and Gaussian minorization give recurrence, uniqueness and convergence. Moments are established by uniform transient bounds and Portmanteau, rather than by integrating an unverified invariant moment.
- Differentiating a mixture of translated Gaussian densities is valid because each derivative is uniformly bounded over translations and the variance range is bounded above and away from zero.
- Symmetry and uniqueness imply the density is even. No unproved mode-count or occupied-tangent claim is attached.

Independent numerical recomputation of the stated concrete constants:

```
gamma0 = 0.009937801287297904
H0     = 1.004999445812808
alpha0 = 2.0174266710216475
```

The eta=8 inverse-Jacobian divergence argument is valid once a simple sampled derivative zero is identified; the displayed rational derivative indeed has such zeros. The strictly positive invariant density then makes the reciprocal moment diverge locally, whereas the logarithmic singularity is integrable.

## Positive occupied exponent: exact piecewise SGD

The construction is mathematically sound and more precise than a numerical moving-exponent assertion:

- Psi and its first derivative match at -1, 0, and 1. The sampled losses are globally C1 with Lipschitz gradients and coercive quadratic tails; they are not incorrectly called globally C2.
- Population curvature is -0.2 on (-1,0) and its unique minimizer is 3/11.
- At eta=1, every initial state maps to [-1,1] in one update. The minimum overlap with I=[-.5,.5] is .26, and all second-step intervals from I contain J=[.27,.35]. The global two-step Doeblin constant is correct.
- The invariant density is bounded by 1/.76. A stationary path avoids every derivative kink almost surely.
- On the occupied interval, derivative magnitude is exactly 2a. The iid amplitude strong law therefore proves the stationary exponent directly.
- The eta∈[.98,1] interval checks pass: invariant interval, center bounds [-.6276,.62], noise half-width >=.3724, overlap >=.2448, second-step common interval J, and both derivative branches bounded below by 1.1168.
- The outside-state distance contraction and repeated independent xi>=0 entry event establish finite-time entrance almost surely for eta<1. Convergence is correctly not asserted uniform over the whole unbounded line.
- The eta=1 outside-start tangent annihilation is explicitly disclosed. A positive exponent applies under the stationary law or to a tangent initialized after entry, not to a zero derivative product transported from outside the interval.

Independent numerical checks:

```
lambda_pi   = 0.18176569236760937
epsilon     = 0.036011080332409975
epsilon_*   = 0.03390581717451523
log(1.1168) = 0.11046745302597098
```

The stationary balance identity is also valid. In the stated smooth example, gradients have linear growth and the invariant law has all moments, so every term is integrable. More generally, for an exact stationary SGD step with finite second moments, g=(X-X+)/eta itself has finite second moment, making the expansion legitimate when batches are fresh and the conditional mean gradient is the population gradient.

## Novelty framing and citations

The introduction and related-work text correctly present the contributions as a disciplined scalar-to-matrix formulation, exact compatibility constructions, and empirical diagnostic distinctions. They do not claim the first confined chaotic optimizer or a new general heavy-tail theorem. The reflected-walk, Kesten--Goldie, Harris, and random-expanding-map foundations are acknowledged.

The piecewise map's background citation is accurate: Kalle and Maggioni's *Invariant densities for random systems of the interval*, arXiv:1805.11430, explicitly covers random piecewise linear expanding systems and random tent maps; journal metadata is ETDS 42 (2022), 141–179, DOI 10.1017/etds.2020.127. It is appropriately cited as background, since the manuscript's explicit minorization and exponent computation supply its own proof. Source checked: https://arxiv.org/abs/1805.11430 . The standard Harris source is https://arxiv.org/abs/0810.2777 .
