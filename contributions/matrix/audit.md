# Matrix renewal audit and repaired proof module

Date: 2026-09-06. Scope: independent mathematical audit; no new numerical experiment.

## Recommended contribution claim

**Nonlinear return can produce a stationary law despite positive local linear-product growth. In an exactly specified returning matrix-excursion model, a negative pressure root controls the stationary small-ball exponent. Its application to a general nonlinear SGD chain requires an additional, explicit occupation-transfer estimate.**

Do not say the affine Kesten theorem itself supplies a stationary causal affine recursion when its global top Lyapunov exponent is positive. The familiar heavy-tail mechanism uses a contractive affine system and a positive pressure root for its outer tail. Here local expansion, independently established global return, and a negative pressure root concern central occupancy. These are different hypotheses and different limits.

The post-Lyapunov adjective should be **post-local-Lyapunov** unless the actual tangent exponent along an invariant nonlinear trajectory has separately been proved positive. The high-dimensional construction proves positive linear-product growth at the population minimizer, not the occupied nonlinear tangent exponent.

## Files ready for integration

- `matrix_main.tex`: readable main-text discussion and compact theorem, with reference to exact appendix assumptions.
- `matrix_appendix.tex`: explicit spectral, Harris, and nonarithmetic assumptions; a new domination lemma repairing the older proof's vague uniform-renewal condition; full change-of-measure, killed occupation, and regeneration proof; exact conditional nonlinear transfer proposition.
- `highdim_example.tex`: complete explicit SGD recurrence proof and rigorous fixed-projection counterexample.
- `references.bib`: four primary mathematical references.
- `check.tex`: standalone compilation driver; the paper root can input the three modules directly using its own theorem environments.

All existing source drafts and handoffs remain unchanged. Labels are namespaced descriptively except conventional theorem environments. `matrix_main.tex` requires the usual `assumption` environment in the parent preamble as well as theorem/proposition/lemma.

## Stronger proof than the existing handoff

The previous `MATRIX_RENEWAL_PROOF_20260905.md` correctly distinguished the exact regenerative theorem from its nonlinear transfer. It nevertheless bundled uniform potential domination into the renewal assumptions and left the cited renewal theorem's almost-every-start condition to be handled later.

The new module resolves those two points under explicit sufficient conditions:

1. At the negative root, assume a continuous eigenfunction h bounded above and away from zero. For some t in (0,alpha), assume k(-alpha+t)<1 on the continuous-function supremum-norm space.
2. The exact tilt gives E_hat exp(t(Z_n-Z_0)) = P_{-alpha+t}^n h(s)/h(s), uniformly bounded by C rho^n for rho<1 using the spectral-radius formula.
3. From an initial state within a finite log interval, exponential Markov inequality bounds expected total interval occupancy by a geometric series. From any other state, stop at the first visit and use the same bound. This proves uniform interval occupation, independent of the starting angle and radius.
4. Apply Alsmeyer's precise Markov renewal theorem to -Z under a positive Harris recurrent, nonarithmetic tilted angular chain of finite positive log drift. Its theorem is initially almost-every-start. Assuming every one-step angular law is absolutely continuous with respect to the tilted stationary angular law extends the result to every start by a first-step decomposition and the proved uniform bound.
5. Untilt and subtract post-exit occupation. The angular eigenfunction cancels the tilted invariant measure, leaving the left eigenmeasure chi. The constant is

   D_alpha = E_zeta[e^(-alpha Z_0) h(S_0) P_hat_{Z_0,S_0}(tau=infinity)].

   It is positive under the explicit tilted-survival condition. Sufficiently deep reinjection gives survival from the same geometric bound.
6. The derived uniform bound means only an alpha-order inverse entry moment is needed here, rather than alpha+epsilon plus an unexplained exit domination assumption. Upper exit has Z_tau>=0 and bounded h, so its domination is automatic.
7. Dividing expected occupation by mean cycle length L gives

   pi(R<=r,S in D)/(r/r0)^alpha -> D_alpha chi(D)/(alpha m L).

This is an application/reduction to an established Markov renewal theorem, not a new proof of that theorem. No originality claim has been verified for this formulation or domination argument.

An explicit realizability example now appears in the appendix: A=M(I-2vv^T), with v uniform on the sphere and M an independent mixture of uniform [.49,.51] and [1.99,2.01] laws with probabilities 1/4 and 3/4. The matrices are symmetric and noncommuting. The angular reflection kernel has exact density ||s-s'||^(2-d) relative to uniform surface probability, hence a Doeblin lower bound 2^(2-d). Norm gains are iid M in this special example, so every spectral, renewal, entry, exit, and positive-survival condition is verified. In dimension two the population Hessian is positive definite. The independent restart is still explicitly a reference return mechanism, not claimed to be literal SGD.

## Nonlinear transfer: what is proved and what remains conditional

For actual regenerative nonlinear SGD, let G count all local visits in a complete return cycle and nu count **every local entry**. Let Q be its actual local killed kernel and L0 the exact linear killed kernel. Then

G = nu + GQ = J + GL0, with J = nu + G(Q-L0).

This identity is exact, and the source J is signed. The new transfer proposition proves that the matrix exponent transfers if:

- the genuine chain has finite-mean complete regeneration cycles;
- the killed linear chain exits for G-almost every start;
- J has a finite weighted total variation integral against exp(-alpha z);
- its renewal coefficient D_Q = integral H dJ is strictly positive.

The old global recurrence audit does not prove these conditions for ordinary neural networks. A Taylor remainder alone cannot control total variation of atomic kernels. Fixed nondegenerate additive noise is also of leading order at sufficiently small radius; the literal zero-radius multiplicative law then needs replacement by a controlled intermediate small-noise annulus.

The proposition is not logically circular, but its J contains the unknown occupation G. The appendix provides a concrete route to noncircular verification: prove an independent inverse-radius Foster inequality QV_p<=rho_p V_p for some p<alpha, and a weighted kernel discrepancy bounded by C V_p. Summing true local excursions gives G(V_p)<=nu(V_p)/(1-rho_p), which proves the source's weighted integrability without assuming the desired alpha law. Positivity of D_Q is still a separate requirement; a nonnegative reinjection measure alone does not prevent cancellation by the nonlinear correction.

## Matrix harmonicity and density geometry

For a general noncommuting product, `E ||A||^q`, a Rayleigh quotient moment, and the spectral pressure k(q) are different objects. The conditional gradient-Hessian scalar gain used in the experiments is a useful directional diagnostic; it is not equal to k(q) in general.

For a strictly convex log pressure with roots -alpha and 0:

- k(-1)<1 iff alpha>1: radial density vanishes at the center **if** a renewal-density theorem applies.
- k(-d)<1 iff alpha>d: the ambient d-dimensional density vanishes in regularly occupied directions **if** both density and angular regularity apply.
- A fixed projection integrates over directions and radii. Neither inequality establishes projected bimodality.

Even scalar alpha>1 gives a central dip only after passing validly from the small-ball law to a density. Two off-center peaks further require sign balance/symmetry and enough regularity/return mass. CDF asymptotics must not be differentiated without an extra theorem or regularity assumption.

## Fully proved matrix recurrence witness and counterexample

`HIGH_DIMENSIONAL_REGRESSION_PROOF_20260905.md` withstands the key checks. The new module gives a full proof with all distributions specified:

- v uniform on the sphere, h=1/2 with probability 1/4 and h=3 with probability 3/4, independent Gaussian residual e, eta=1.
- Local A=I-h vv^T has an actual top matrix-product exponent at least log(2)/(2d)>0, proved from isotropic iid norm increments and a finite basis comparison.
- Isotropy gives k(q)=E||As||^q exactly in this special construction. A determinant/spherical integral gives k(-d)=7/8; convexity gives k(-1)<=1-1/(8d). The unique negative root therefore satisfies alpha>d.
- The exact saturated update equals a contracting outer linear map plus a uniformly bounded remainder and fresh rank-one Gaussian forcing. The averaged outer square contraction is theta=1-43/(64d), and the nonlinear remainder norm is at most sqrt(3 rho/8).
- A direct one-step full-dimensional density calculation after averaging the random axis proves irreducibility, compact-set minorization and aperiodicity. Quadratic drift proves unique attracting stationarity and finite second moment.
- Rotational equivariance plus uniqueness forces an isotropic stationary law. For d>=3, every conditional fixed-coordinate density given radius is centrally unimodal; its radius mixture remains so.

This is a rigorous counterexample to unconditional fixed-projection bimodality inferred from local negative pressure. It should stay in the paper, at least as a short main-text remark plus appendix proof, because it makes the neural fixed-direction experiments scientifically necessary.

**Classification correction:** this particular loss is nonquadratic but convex along its observed coordinate. Its scalar second derivative is 1/2+(1-t)/(2(1+t)^2)>=7/16. Do not label it the nonconvex example. The separate bounded-feedback/nonconvex scalar construction belongs elsewhere.

## Primary references checked

- Alsmeyer (1997), *The Markov Renewal Theorem and Related Results*, Theorem 1. Author PDF: https://www.uni-muenster.de/Stochastik/alsmeyer/MRTheorem2.pdf . Journal metadata: https://math-mprf.org/journal/articles/id778/ . The theorem is Harris plus nonarithmetic and initially almost-every-start; the new proof explicitly resolves that last issue under an extra absolute-continuity assumption.
- Guivarc'h and Le Page (2016), *Spectral gap properties for linear random walks and Pareto's asymptotics for affine stochastic recursions*: https://arxiv.org/abs/1204.6004 . DOI 10.1214/15-AIHP668. The abstract and paper concern positive-order homogeneous operators and outer affine tails. They do not by themselves verify the negative-order conditions used here.
- Kesten (1973), *Random difference equations and renewal theory for products of random matrices*, Acta Mathematica 131, 207–248, DOI 10.1007/BF02392040. This is established background, not a claimed contribution.
- Hairer and Mattingly, *Yet another look at Harris' ergodic theorem for Markov chains*: https://arxiv.org/abs/0810.2777 . Standard drift/minorization result used in the explicit recurrence proof.

## Manuscript positioning

Avoid claiming priority for all stationarity beyond local instability or for a general extension of Kesten theory. The current valid package is an organized relation among local moment diagnostics, a conditional central renewal law with an explicit direction process, exact nonlinear recurrence examples, and carefully delimited network observations. The proof audit in `handoff/eoss/sgd_equilibrium_20260905/theory/NOVELTY_STRESS_TEST.md` already states that concrete scalar heavy-tail rows are consequences of stronger existing asymptotically linear IFS theory.
