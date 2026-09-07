"""Independent exact arithmetic and numerical certificates for scalar text."""
from fractions import Fraction as F
from pathlib import Path
import json
import math

OUT = Path(__file__).resolve().parent
eta, beta, mu, s = F(403, 200), F(23, 25), F(2, 25), F(1, 20)
hs = [beta + mu - s, beta + mu + s]
central = [abs(1 - eta * h) for h in hs]
outer = [1 - eta * (h - beta) for h in hs]
prod = central[0] * central[1]
H = 2 / sum(1 / a for a in central)
assert prod > 1 and H > 1 and max(map(abs, outer)) < 1
assert min(central) < 1 < max(central)

def phi(q):
    return sum(float(a) ** (-q) for a in central) / 2

coarse_bracket = [2.01742, 2.01744]
lo, hi = coarse_bracket
assert phi(lo) < 1 < phi(hi)
for _ in range(60):
    mid = (lo + hi) / 2
    if phi(mid) < 1:
        lo = mid
    else:
        hi = mid

# Exact polynomial counterexample without a symbolic-algebra dependency.
# A polynomial is a coefficient list (constant, linear, quadratic).
f = [F(12, 11), F(0), -F(3, 11)]
gleft = [F(24, 41), F(0), F(21, 41)]
gright = [F(72, 41), F(0), -F(27, 41)]

def integral(coeffs, a, b):
    return sum(v * (b ** (i+1) - a ** (i+1)) / (i+1)
               for i, v in enumerate(coeffs))

assert integral(f, F(0), F(1)) == 1
assert integral(gleft, F(0), F(1,2)) + integral(gright, F(1,2), F(1)) == 1
reset = F(4,5) * integral(f, F(1,2), F(1))
assert reset == F(41,110)
left_residual = [v - F(2,5)*v*2**i - F(2,5)*v*F(1,2)**i
                 for i, v in enumerate(f)]
right_residual = [v - F(2,5)*v*F(1,2)**i for i, v in enumerate(f)]
assert left_residual == [reset*v for v in gleft]
assert right_residual == [reset*v for v in gright]

# Exact return model sample choices used to cross alpha=1.
return_examples = []
for b, c in [(F(3,2), F(1)), (F(2), F(1)), (F(3), F(1))]:
    alpha, rho = b-c, c/b
    neg_moment = b*c/((b-1)*(c+1))
    harmonic = 1/neg_moment
    assert b*c/((b-alpha)*(c+alpha)) == 1
    assert harmonic - 1 == (alpha-1)/(b*c)
    assert (1-rho)+rho == 1
    return_examples.append(dict(b=float(b), c=float(c), alpha=float(alpha),
                                H=float(harmonic), boundary_atom=float(1-rho)))

result = {
    'status': 'passed',
    'smooth_example': {
        'eta': float(eta), 'beta': float(beta), 'mu': float(mu), 's': float(s),
        'central_absolute_gains': [float(a) for a in central],
        'outer_slopes': [float(a) for a in outer],
        'central_product_exact': str(prod), 'central_H_exact': str(H),
        'lambda0': math.log(float(prod))/2, 'H0': float(H),
        'alpha0_numeric': (lo+hi)/2, 'alpha0_numerical_sign_bracket': coarse_bracket,
        'min_population_hessian_exact': str(mu-beta/8),
    },
    'counterexample': {'reset_probability_exact': str(reset),
                      'normalizations_exact': True, 'invariance_polynomials_exact': True},
    'exact_return_examples': return_examples,
    'scope': 'Certificates check constants and algebra; analytic proofs are in scalar_proofs.tex.'
}
(OUT / 'SCALAR_VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
