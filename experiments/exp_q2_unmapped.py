"""Unmapped Chebyshev collocation (pinned form, q = 2) with exactly the initial guess,
pin and Newton iteration of collocation_q.py, for comparison with the sinh map.
Writes results/colloc_q2_unmapped.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, csv
from flame_q import Theta, master
from collocation import cheb, bary_row, bary_eval
from collocation_q import params, initial_guess

q = 2


def solve_unmapped(delta, N, tol=1e-12, rtol=1e-10, maxit=80):
    mu, x0 = params(delta, q)
    D, xi = cheb(N)                      # x = xi, nodes ordered from +1 to -1
    Dx = D
    svar = lambda x: (x - x0) / mu
    ystar = (q - 1) / q
    s_pin = Theta(ystar, q)
    x_pin = x0 + mu * s_pin
    y = initial_guess(svar(xi), s_pin, q); y[-1] = delta
    u = np.log(y)
    F = lambda u: np.exp((q - 1) * u) - np.exp(q * u)
    dF = lambda u: (q - 1) * np.exp((q - 1) * u) - q * np.exp(q * u)
    prow = bary_row(xi, x_pin); pval = np.log(ystar)

    def res(u):
        R = mu * Dx @ u - F(u); R[-1] = prow @ u - pval
        return R
    R = res(u); nfb = 0; ok = False; lam = 1.0
    for it in range(maxit):
        J = mu * Dx - np.diag(dF(u)); J[-1, :] = prow
        du = np.linalg.solve(J, -R)
        if np.max(np.abs(du)) < tol:                       # full Newton correction small: take it
            ut = u + du; Rt = res(ut)
            if np.max(np.abs(Rt)) < rtol:                  # ... and check the residual
                u, R, lam, ok = ut, Rt, 1.0, True
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
    M = 4000
    xe = np.cos(np.pi * (np.arange(M) + 0.5) / M)
    err = np.max(np.abs(bary_eval(xi, np.exp(u), xe) - master(svar(xe), q)))
    return dict(err=err, its=it + 1, conv=int(ok),
                res_inf=float(np.max(np.abs(R))), lam_last=lam, n_fallback=nfb)


if __name__ == '__main__':
  rows = []
  for d in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
      for N in [32, 64, 128, 256, 512, 768, 1024, 1536]:
          r = solve_unmapped(d, N); r.update(delta=d, N=N); rows.append(r)
          print(r, flush=True)
  with open(ROOT + '/results/colloc_q2_unmapped.csv', 'w', newline='') as fh:
      w = csv.DictWriter(fh, fieldnames=['delta', 'N', 'err', 'its', 'conv'], extrasaction='ignore'); w.writeheader(); w.writerows(rows)
