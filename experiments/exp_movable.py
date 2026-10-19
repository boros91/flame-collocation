"""Movable singularities: dependence of the complex singularities on the initial value delta,
in the original time variable t (y' = y^q(1-y), y(0) = delta).
Exact (Proposition 2.1/2.2):  t* = s* - Theta_q(delta) = -Theta_q(delta) +- (2k+1) pi i.
Two independent numerical estimates of the nearest singularity:
  (a) Pade: [m/m] Pade approximants of the Taylor series of y_delta at t = 0, computed from the
      ODE alone (IVP data only, 80 digits); the pole nearest to t = 0 in the upper half-plane.
  (b) AAA: the computed pinned collocation solution, sampled at 3000 equispaced points of
      t in [t_h - 30, t_h + 30] intersected with [0, 2/mu], where t_h is the point at which the computed solution equals 1/2;
      AAA (rtol 1e-13, poles with |residue| < 1e-8 discarded) in tau = t - t_h; lowest pole.
Writes results/movable_singularities.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import warnings
import numpy as np, pandas as pd, mpmath as mp
from scipy.interpolate import AAA
from scipy.optimize import brentq
from collocation_q import solve, setup
from collocation import bary_eval

warnings.filterwarnings('ignore')


def taylor(delta, q, N):
    """Taylor coefficients of y_delta at t = 0 from y' = y^q - y^{q+1} (Cauchy products)."""
    a = [mp.mpf(delta)]
    for n in range(N):
        p = [mp.mpf(1)] + [mp.mpf(0)] * n
        for _ in range(q):
            p = [mp.fsum(p[k] * a[j - k] for k in range(j + 1)) for j in range(n + 1)]
        a.append((p[n] - mp.fsum(p[k] * a[n - k] for k in range(n + 1))) / (n + 1))
    return a


def pade_pole(delta, q, m):
    mp.mp.dps = 60
    a = taylor(mp.mpf(delta), q, 2 * m)
    S = 1 / ((q - 1) * mp.mpf(delta)**(q - 1))          # time scale of the slow phase (scaling only)
    P, Q = mp.pade([a[n] * S**n for n in range(2 * m + 1)], m, m)
    roots = [z * S for z in mp.polyroots(Q[::-1], maxsteps=400, extraprec=300)]
    up = [z for z in roots if mp.im(z) > 0]
    return complex(min(up, key=abs))


def aaa_pole(delta, q, N):
    sol = solve(delta, N, q, pin=True); S = setup(delta, N, q)
    mu, x0, eps, L, M = S['mu'], S['x0'], S['eps'], S['L'], S['M']
    yfun = lambda xi: bary_eval(sol['xi'], sol['y'], np.atleast_1d(xi))[0]
    xi_h = brentq(lambda xi: yfun(xi) - 0.5, -1, 1, xtol=1e-15)
    xmx0_h = eps * np.sinh(L * xi_h + M)                  # x_h - x0 without cancellation
    t_h = (x0 + xmx0_h + 1) / mu
    tau = np.linspace(-min(30, t_h), min(30, 2 / mu - t_h), 3000)     # stay inside [0, 2/mu]
    xi = (np.arcsinh((xmx0_h + mu * tau) / eps) - M) / L
    r = AAA(tau, bary_eval(sol['xi'], sol['y'], xi), rtol=1e-13)
    p, res = r.poles(), r.residues()
    p = p[(p.imag > 0) & (np.abs(res) > 1e-8)]
    return t_h + p[np.argmin(p.imag)]


Theta = lambda d, q: float(mp.log(mp.mpf(d) / (1 - mp.mpf(d))) - sum(1 / (j * mp.mpf(d)**j) for j in range(1, q)))
rows = []
for q, d, N in [(2, 1e-1, 64), (2, 1e-2, 64), (2, 1e-3, 128), (2, 1e-4, 160), (2, 1e-6, 224),
                (3, 1e-1, 64), (3, 3e-2, 128), (3, 1e-2, 128), (3, 1e-3, 192)]:
    mu = (q - 1) * d**(q - 1)
    r = dict(q=q, delta=d, mu=mu, exact_re=-Theta(d, q), exact_im=np.pi, dist_x_plane=np.pi * mu)
    pa = aaa_pole(d, q, N); r.update(N=N, aaa_re=pa.real, aaa_im=pa.imag)
    if (q, d) in [(2, 1e-1), (2, 1e-2), (2, 1e-3), (3, 1e-1), (3, 3e-2)]:
        pp = pade_pole(d, q, 40); r.update(pade_re=pp.real, pade_im=pp.imag)
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(ROOT + '/results/movable_singularities.csv', index=False)
