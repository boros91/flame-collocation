"""256-bit check of Section 5.4: the IVP form for q = 3, delta = 1e-2, started from the
converged pinned solution instead of the generic initial guess.
Writes results/highprec_warm_q3.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import time, pandas as pd, mpmath as mp
import exp_highprec as H

q, d = 3, 1e-2
rows = []
for N in [64, 128]:
    ie = H.interp_error(d, N, q)
    t0 = time.time()
    sp = H.solve(d, N, q, True, maxit=40)
    v0 = [mp.log(y) for y in sp['y']]
    si = H.solve(d, N, q, False, maxit=40, v0=v0)
    for form, s in (('pinned', sp), ('IVP from pinned', si)):
        e = H.error(s, q)
        rows.append(dict(q=q, delta=d, N=N, form=form, error=float(e), interp=float(ie),
                         ratio=float(e / ie), its=s['its'], conv=s['conv'],
                         secs=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT + '/results/highprec_warm_q3.csv', index=False)
