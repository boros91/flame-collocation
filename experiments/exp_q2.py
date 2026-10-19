"""q = 2 with the general-q code (collocation_q.py), i.e. with exactly the initial guess,
pin and stopping rule described in the paper (Section 4).  Also records
  - the 2-norm condition numbers of the final Newton matrices (pinned and IVP form),
  - the intermediate decades delta = 1e-1, 1e-3, 1e-5 and the full N-grid
    32, 64, ..., 512 in steps of 32 (used for the Chebfun comparison table).
Writes results/colloc_q2.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, csv
from collocation_q import solve, error, interp_error

q = 2
Ns = list(range(32, 513, 32))
deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
rows = []
for d in deltas:
    for N in Ns:
        sp = solve(d, N, q, pin=True)
        si = solve(d, N, q, pin=False)
        rows.append(dict(delta=d, N=N, L=sp['L'], interp=interp_error(d, N, q),
                         pin=error(sp), pin_its=sp['its'], pin_conv=int(sp['converged']),
                         y0_relerr=abs(sp['y'][-1] / d - 1), pin_cond=sp['cond'],
                         ivp=error(si), ivp_its=si['its'], ivp_conv=int(si['converged']),
                         ivp_cond=si['cond']))
        print({k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
with open(ROOT + '/results/colloc_q2.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
