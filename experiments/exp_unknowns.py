"""Choice of the unknown (q = 2): sinh-mapped collocation for the same problem written in
three variables, each with the initial condition (IVP form) and with an interior condition
(pinned form), on the grid N = N_min of Table 1:
   y        : mu y' = y^2 (1 - y),                y(-1) = delta  | pin y = 2/3 (maximiser of f)
   v = ln y : mu v' = e^v - e^{2v},               v(-1) = ln delta | pin y = 1/2 (paper)
   w = 1/y  : mu w' = -(1 - 1/w),                 w(-1) = 1/delta  | pin y = 1/2
Same map, initial guess (generic profile), damped Newton and error measure as collocation_q.py.
Newton stops when the full correction is below 1e-12 * max(1, max|unknown|) and, after it, the max residual
is below 1e-10 * max(1, max|unknown|); failure after 80 iterations.
Writes results/unknowns_q2.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd, warnings
from collocation_q import setup, initial_guess, error
from collocation import bary_row
from flame_q import Theta

warnings.filterwarnings('ignore')
q = 2
FORMS = {
    'y':    dict(to=lambda y: y,         back=lambda u: u,         F=lambda u: u**2 - u**3,
                 dF=lambda u: 2 * u - 3 * u**2, ystar=2 / 3),
    'ln y': dict(to=lambda y: np.log(y), back=lambda u: np.exp(u), F=lambda u: np.exp(u) - np.exp(2 * u),
                 dF=lambda u: np.exp(u) - 2 * np.exp(2 * u), ystar=1 / 2),
    '1/y':  dict(to=lambda y: 1 / y,     back=lambda u: 1 / u,     F=lambda u: -(1 - 1 / u),
                 dF=lambda u: -1 / u**2, ystar=1 / 2),
}


def solve(delta, N, var, pin, tol=1e-12, rtol=1e-10, maxit=80):
    S = setup(delta, N, q); xi, Dx, mu = S['xi'], S['Dx'], S['mu']
    fm = FORMS[var]
    y0 = initial_guess(S['svar'](xi), Theta(0.5, q), q); y0[-1] = delta
    u = fm['to'](y0)
    if pin:
        xi_pin = (np.arcsinh(Theta(fm['ystar'], q) / np.pi) - S['M']) / S['L']
        prow = bary_row(xi, xi_pin); pval = fm['to'](fm['ystar'])

    def res(u):
        R = mu * Dx @ u - fm['F'](u)
        R[-1] = (prow @ u - pval) if pin else (u[-1] - fm['to'](delta))
        return R
    R = res(u); conv = False; nfb = 0
    for it in range(1, maxit + 1):
        J = mu * Dx - np.diag(fm['dF'](u))
        if pin:
            J[-1, :] = prow
        else:
            J[-1, :] = 0; J[-1, -1] = 1
        du = np.linalg.solve(J, -R)
        sc = max(1.0, np.max(np.abs(u)))                 # tolerances relative to max|u| (matters for 1/y)
        if np.max(np.abs(du)) < tol * sc:                # full Newton correction small: take it, check residual
            ut = u + du; Rt = res(ut)
            if np.max(np.abs(Rt)) < rtol * sc:
                u, R, lam, conv = ut, Rt, 1.0, True
                break
        lam, r0 = 1.0, np.linalg.norm(R)
        while True:
            un = u + lam * du; Rn = res(un)
            if np.linalg.norm(Rn) < (1 - 1e-4 * lam) * r0:
                break
            if lam < 1e-3:
                nfb += 1; break                            # no sufficient decrease: take the smallest step
            lam /= 2
        u, R = un, Rn
    sol = dict(xi=xi, y=fm['back'](u), svar=S['svar'], q=q)
    return dict(error=error(sol), its=it, conv=int(conv), cond=np.linalg.cond(J),
                umax=float(np.max(np.abs(u))), res_inf=float(np.max(np.abs(R))), lam_last=lam, n_fallback=nfb)


if __name__ == '__main__':
  ref = pd.read_csv(ROOT + '/results/colloc_q2.csv')
  rows = []
  for d in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
      g = ref[np.isclose(ref.delta, d, rtol=1e-9, atol=0)]
      N = int(g[g.pin <= 1e-10].N.min()); interp = float(g[g.N == N].interp.iloc[0])
      for var in FORMS:
          for pin in (False, True):
              r = solve(d, N, var, pin)
              r.update(delta=d, N=N, var=var, form='pinned' if pin else 'IVP', interp=interp)
              rows.append(r); print(r, flush=True)
  pd.DataFrame(rows)[['delta', 'N', 'var', 'form', 'error', 'interp', 'its', 'conv', 'cond', 'umax']].to_csv(
      ROOT + '/results/unknowns_q2.csv', index=False)
