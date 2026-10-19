"""Check of the 256-bit reference values used in exp_highprec.py: the master profile Y_q at the 600 error-evaluation
points is recomputed at 512 bits (same Newton iteration in the logit variable, tolerance scaled to the precision)
and compared with the 256-bit values. Writes results/hp_reference_check.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd, mpmath as mp
from flame_q import master as master_double


def master(s, q, prec):
    with mp.workprec(prec):
        s = mp.mpf(s)
        y0 = float(master_double(np.array([float(s)]), q)[0]); y0 = min(max(y0, 1e-300), 1 - 1e-16)
        l = mp.log(mp.mpf(y0)) - mp.log1p(-mp.mpf(y0))
        for _ in range(200):
            y = 1 / (1 + mp.exp(-l))
            th = l
            for j in range(1, q): th -= 1 / (j * y**j)
            dl = (th - s) * y**(q - 1); l -= dl
            if abs(dl) < mp.mpf(2)**(-prec + 8) * (1 + abs(l)): break
        return 1 / (1 + mp.exp(-l))


def params(delta, q, prec):
    with mp.workprec(prec):
        d = mp.mpf(delta); mu = (q - 1) * d**(q - 1)
        th = mp.log(d) - mp.log1p(-d)
        for j in range(1, q): th -= 1 / (j * d**j)
        x0 = -mu * th - 1; eps = mp.pi * mu
        sm = mp.asinh((-1 - x0) / eps); sp = mp.asinh((1 - x0) / eps)
        return (sp - sm) / 2, (sp + sm) / 2


rows = []
M = 600
for q, deltas in [(2, ['1e-2', '1e-4', '1e-6']), (3, ['1e-1', '1e-2', '1e-3'])]:
    for d in deltas:
        diffs = []
        for k in range(M):
            vals = []
            for prec in (256, 512):
                with mp.workprec(prec):
                    L, Mm = params(d, q, prec)
                    t = mp.cos(mp.pi * (k + mp.mpf(1) / 2) / M)
                    s = mp.pi * mp.sinh(L * t + Mm)
                    vals.append(master(s, q, prec))
            with mp.workprec(512):
                diffs.append(abs(vals[0] - vals[1]))
        rows.append(dict(q=q, delta=float(d), max_diff=float(max(diffs)))); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT + '/results/hp_reference_check.csv', index=False)
