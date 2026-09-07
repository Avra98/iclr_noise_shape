# First-nine-page reviewer/AC reading

Date: 2026-09-06. Read the assembled 34-page main.pdf, focusing on its first nine manuscript pages, plus the current main and section sources. Review concerns clarity and supported contribution framing; no paper sources edited.

## Overall assessment

The sequence is coherent: deterministic scalar baseline, stochastic moment ordering, exact scalar returned central law, matrix radius-direction renewal, genuinely nonconvex SGD recurrence, positive occupied tangent control, measurement definitions, and neural observations. The paper clearly distinguishes three logically separate issues: local growth, return, and coordinate shape. This is understandable to an optimization reviewer, although a few probability terms should be glossed before the main theorems.

The introduction and related-work text do not claim an unsupported general extension of affine Kesten stationarity into the positive-global-exponent regime. Classical foundations are acknowledged, and the empirical limitations are consistent with the reported measurements. The explicit smooth construction is not claimed to satisfy the zero-radius density law at fixed additive noise. The matrix projection counterexample is retained. These are substantive strengths of the current reasoning.

## Actionable corrections

1. **Page 2, paragraph after Proposition 1:** contraction of a negative perturbation moment is described as 'hence depletion near zero'. This is too quick before return and density assumptions. Replace it with decay of an inverse-distance moment or a local repulsion diagnostic. The stationary central-density implication belongs to Theorem 2's hypotheses.

2. **Page 5, proof idea after Theorem 4:** 'the local derivative at the origin is expansive' overstates a random derivative law that includes the contracting absolute gain .91425. Say that the local random product has positive mean log gain at the displayed step size.

3. **Figure 1 caption:** 'Appendix 7' is a reference-type error; sc:solvable labels a proposition. Use 'Proposition 7 in Appendix A' or the relevant appendix subsection reference.

4. **Definition timing:** invariant and attracting laws are first explicitly defined in Section 4 after being used by Theorems 2 and 3. Move the short definition sentence earlier. This makes the theorem sequence self-contained for the intended reader without adding net length.

5. **Technical terms needing a short main-text gloss:** coercive means the loss tends to infinity as the parameter magnitude grows; nonarithmetic rules out a log-radius lattice; k(q) is the long-run moment-growth factor including direction changes. The exact technical assumptions can remain in the appendix. 'Tilted' can be introduced as reweighting paths to make inward excursions typical. The theorem's Assumption 1 reference should explicitly point to Appendix B.

## Contribution clarity

The third introduction bullet would be stronger if it named the concrete empirical payoff: in the highlighted MLP, a negative late fixed-axis scalar log average coexists with positive transported growth and two-lobed occupancy of that fixed coordinate. This finding directly motivates retaining direction dynamics and using a fixed frame for shape. Merely naming the three measurement types makes the originality harder to identify.

Figure 3 uses a fixed checkpoint theta_star as its coordinate reference, and its two lobes need not straddle zero. A short caption phrase saying this would prevent a reader from identifying that coordinate origin with the ideal scalar renewal center. Current statements already avoid claiming that the neural histogram is the theorem's stationary law; preserve that limitation.

## No further mathematical blockers found

The stationary radial drift formula newly added to Section 4 is correct:

`d(x)=([(1-eta*mu-eta*beta/(1+x^2))^2+eta^2*s^2-1]x^2)+eta^2*sigma^2*(beta+mu)`.

For the displayed instance, it is outward near the center and inward sufficiently far away. Its zero is correctly called a local balance radius, not a point-mass equilibrium. The main positive occupied-exponent theorem is proved by the appendix's explicit Doeblin calculation and retains the correct distinction between invariant-law attraction and derivative-product growth.
