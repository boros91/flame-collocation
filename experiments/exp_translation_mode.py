"""Sect. 4: the translation mode in the discrete problem.
phi = nodal values of v'(x) = F_q(v)/mu is (up to discretization error) annihilated by the collocation rows of the
Newton matrix J, so J phi ~ c e_N and the last column of J^{-1} is ~ phi/c, with c the value of phi seen by the
condition row: c = v'(-1) for the IVP form (max ~ Q(y*)/delta^(q-1)) and c = v'(x*) = max phi for the pinned form
(max ~ 1). J is evaluated at the converged pinned solution, at N = N_min of Tables 1 and 3.
Writes results/translation_mode.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd
from collocation_q import setup, solve
from collocation import bary_row
from flame_q import Theta
R = ROOT + '/results/'
c2 = pd.read_csv(R + 'colloc_q2.csv'); c3 = pd.read_csv(R + 'colloc_q3.csv')


def nmin(df, d):
    h = df[np.isclose(df.delta, d, rtol=1e-9, atol=0)]
    return int(h[h.pin <= 1e-10].N.min())


rows = []
for q, df, deltas in [(2, c2, [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]), (3, c3, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])]:
    for d in deltas:
        N = nmin(df, d)
        S = setup(d, N, q); sol = solve(d, N, q, True); u = np.log(sol['y']); mu = S['mu']
        J = mu * S['Dx'] - np.diag((q - 1) * np.exp((q - 1) * u) - q * np.exp(q * u))
        Jp = J.copy(); sp = Theta((q - 1) / q, q)
        Jp[-1, :] = bary_row(S['xi'], (np.arcsinh(sp / np.pi) - S['M']) / S['L'])
        Ji = J.copy(); Ji[-1, :] = 0; Ji[-1, -1] = 1
        eN = np.zeros(N + 1); eN[-1] = 1
        Qs = ((q - 1) / q)**(q - 1) / q
        rows.append(dict(q=q, delta=d, N=N, col_ivp=np.abs(np.linalg.solve(Ji, eN)).max(),
                         pred_ivp=Qs / (d**(q - 1) * (1 - d)), col_pin=np.abs(np.linalg.solve(Jp, eN)).max(),
                         cond_ivp=np.linalg.cond(Ji)))
r = pd.DataFrame(rows); r['ratio'] = r.col_ivp / r.pred_ivp
r.to_csv(R + 'translation_mode.csv', index=False); print(r.to_string())
