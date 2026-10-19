"""Sect. 3: interpolation error of the exact q = 2 solution when the sinh map is centred at the
midpoint x0 = 0 instead of at the exact front position (delta = 1e-4, 1e-6, 1e-8).
Writes results/map_centre.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import csv
from collocation import error, singularity, setup, master

rows = []
for d in [1e-4, 1e-6, 1e-8]:
    for N in [64, 128, 256, 384, 512]:
        x0, eps = singularity(d)
        xi, x, Dx, tau = setup(d, N, 'sinh', 0.0, eps)
        e_int = error(dict(xi=xi, y=master(tau(xi)), tau=tau))
        rows.append(dict(delta=d, N=N, interp_center0=e_int, offset_in_widths=x0 / eps))
        print('center0', rows[-1], flush=True)
with open(ROOT + '/results/map_centre.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
