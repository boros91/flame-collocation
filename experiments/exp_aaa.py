"""A posteriori location of the complex singularities from the COMPUTED collocation solutions
with the AAA algorithm (Nakatsukasa, Sete, Trefethen 2018; scipy.interpolate.AAA).
For each case the pinned collocation solution is evaluated (barycentric interpolant) at 3000
equispaced points of a window in the master variable s (or in x for the unmapped grid), AAA is
run with rtol = 1e-13, and the poles in the upper half-plane closest to the real axis are
reported (poles with |residue| < 1e-8, which arise from rounding noise, are discarded), in the variable s, together with the exact branch point s* of Proposition 2.2 /
Remark 2.3.  For comparison the same is done with the exact solution sampled at the same points.
Writes results/aaa_poles.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import warnings
import numpy as np, pandas as pd
from scipy.interpolate import AAA
from collocation_q import solve as solve_q, setup as setup_q, params, initial_guess
from collocation import cheb, bary_row, bary_eval
from flame_q import master, Theta

warnings.filterwarnings('ignore')
NS = 3000


def lowest_poles(s, y, k=3):
    r = AAA(s, y, rtol=1e-13)
    p, res = r.poles(), r.residues()
    p = p[(p.imag > 0) & (np.abs(res) > 1e-8)]     # upper half-plane; drop spurious poles (|residue| ~ 1e-13)
    p = p[np.argsort(p.imag)]
    return p[:k], len(r.poles())


def unmapped_q2(delta, N, tol=1e-12, rtol=1e-10, maxit=80):
    """Unmapped Chebyshev collocation, pinned form (as exp_q2_unmapped.py); returns nodal values."""
    q = 2
    mu, x0 = params(delta, q)
    D, xi = cheb(N)
    ystar = 0.5; s_pin = Theta(ystar, q); x_pin = x0 + mu * s_pin
    u = np.log(initial_guess((xi - x0) / mu, s_pin, q))
    F = lambda u: np.exp(u) - np.exp(2 * u); dF = lambda u: np.exp(u) - 2 * np.exp(2 * u)
    prow = bary_row(xi, x_pin)
    res = lambda u: np.concatenate([(mu * D @ u - F(u))[:-1], [prow @ u - np.log(ystar)]])
    R = res(u)
    for it in range(maxit):
        J = mu * D - np.diag(dF(u)); J[-1, :] = prow
        du = np.linalg.solve(J, -R)
        if np.max(np.abs(du)) < tol:
            ut = u + du; Rt = res(ut)
            if np.max(np.abs(Rt)) < rtol:
                u = ut; break
        lam, r0 = 1.0, np.linalg.norm(R)
        while True:
            un = u + lam * du; Rn = res(un)
            if np.linalg.norm(Rn) < (1 - 1e-4 * lam) * r0 or lam < 1e-3:
                break
            lam /= 2
        u, R = un, Rn
    return xi, np.exp(u), x0, mu


rows = []
def add(case, q, delta, N, grid, s, y_comp, y_exact, s_star):
    pc, nc = lowest_poles(s, y_comp); pe, ne = lowest_poles(s, y_exact)
    rows.append(dict(case=case, q=q, delta=delta, N=N, grid=grid, window=f'[{s.min():.3g},{s.max():.3g}]',
                     s_star_re=s_star.real, s_star_im=s_star.imag,
                     pole1_re=pc[0].real, pole1_im=pc[0].imag, pole2_im=pc[1].imag, pole3_im=pc[2].imag,
                     pole1_im_over_s_star_im=pc[0].imag / s_star.imag, npoles=nc,
                     exact_pole1_re=pe[0].real, exact_pole1_im=pe[0].imag))
    print(rows[-1], flush=True)


# (a) q = 2, delta = 1e-2, UNMAPPED Chebyshev grid: nothing about the singularities is used
xi, y, x0, mu = unmapped_q2(1e-2, 1024)
x = np.linspace(-1, 1, NS)
yc = bary_eval(xi, y, x)
s = (x - x0) / mu
add('unmapped', 2, 1e-2, 1024, 'Chebyshev, no map', s, yc, master(s, 2), 1j * np.pi)

# (b) sinh-mapped pinned solutions, window s in [-30, 30]
s = np.linspace(-30, 30, NS)
for q, d, N in [(2, 1e-6, 256), (2, 1e-10, 384), (3, 1e-3, 256), (3, 1e-6, 416)]:
    sol = solve_q(d, N, q, pin=True); S = setup_q(d, N, q)
    yc = bary_eval(sol['xi'], sol['y'], (np.arcsinh(s / np.pi) - S['M']) / S['L'])
    add('sinh map', q, d, N, 'sinh map', s, yc, master(s, q), 1j * np.pi)

pd.DataFrame(rows).to_csv(ROOT + '/results/aaa_poles.csv', index=False)
