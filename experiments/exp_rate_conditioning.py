"""Additional checks quoted in Sects. 3-5.
(1) exact Bernstein-ellipse exponent log(rho*) of the mapped singularity z* = (-M + i pi/2)/L and refit of the
    observed rates (high precision, results/highprec_colloc_q*.csv; Arrhenius, results/arrhenius_runs.csv);
(2) 2-norm condition numbers of the final Newton matrices, raw and after scaling each row to unit inf-norm;
(3) error of the two reconstructions I_N(y_j) and exp(I_N v_j);
(4) off-grid ODE defect max |mu p' - F(p)| of p = I_N v at 4000 points (a check that needs no exact solution).
Writes results/rate_exact.csv, results/conditioning_scaled.csv, results/reconstruction_defect.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd
from collocation_q import setup, solve
from collocation import bary_row, bary_eval, cheb
from flame_q import Theta, master
from arrhenius_model import Arr
R = ROOT + '/results/'


def log_rho(L, M):
    z = (-M + 0.5j * np.pi) / L
    w = np.sqrt(z * z - 1 + 0j)
    return np.log(max(abs(z + w), abs(z - w)))


# (1) rates
rows = []
for q in (2, 3):
    hp = pd.read_csv(R + f'highprec_colloc_q{q}.csv')
    for d, g in hp[hp.form == 'pinned'].groupby('delta'):
        g = g[g.error > 1e-70].sort_values('N')
        slope = -np.polyfit(g.N, np.log(g.error), 1)[0]
        S = setup(d, 64, q)
        lr = log_rho(S['L'], S['M'])
        rows.append(dict(set='256-bit', q=q, param=d, L=S['L'], M=S['M'], thin=np.pi / (2 * S['L']), log_rho=lr,
                         observed=slope, ratio_thin=slope / (np.pi / (2 * S['L'])), ratio_exact=slope / lr))
ar = pd.read_csv(R + 'arrhenius_runs.csv')
for beta, g in ar[ar.form == 'pin'].groupby('beta'):
    P = Arr(beta, 0.8); sst = P.singularity()
    xc, eps = P.mu * sst.real, P.mu * sst.imag
    al, be = np.arcsinh((-1 - xc) / eps), np.arcsinh((1 - xc) / eps)
    L, M = (be - al) / 2, (be + al) / 2
    g = g[(g.err > 1e-12) & (g.err < 1e-2)]
    slope = -np.polyfit(g.N, np.log(g.err), 1)[0]
    lr = log_rho(L, M)
    rows.append(dict(set='Arrhenius', q=0, param=beta, L=L, M=M, thin=np.pi / (2 * L), log_rho=lr, observed=slope,
                     ratio_thin=slope / (np.pi / (2 * L)), ratio_exact=slope / lr))
rt = pd.DataFrame(rows); rt.to_csv(R + 'rate_exact.csv', index=False); print(rt.to_string())


def jac(sol, S, q, pin):
    u = np.log(sol['y']); mu = S['mu']
    J = mu * S['Dx'] - np.diag((q - 1) * np.exp((q - 1) * u) - q * np.exp(q * u))
    if pin:
        sp = Theta((q - 1) / q, q)
        J[-1, :] = bary_row(S['xi'], (np.arcsinh(sp / np.pi) - S['M']) / S['L'])
    else:
        J[-1, :] = 0; J[-1, -1] = 1
    return J


def nmin(df, d):
    h = df[np.isclose(df.delta, d, rtol=1e-9, atol=0)]
    return int(h[h.pin <= 1e-10].N.min())


# (2) conditioning
rows = []
for q, deltas in [(2, [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]), (3, [1e-2, 1e-4, 1e-6])]:
    for d in deltas:
        for N in (256, 320, 384, 448, 512):
            S = setup(d, N, q)
            for pin in (True, False):
                sol = solve(d, N, q, pin)
                J = jac(sol, S, q, pin)
                Js = J / np.abs(J).sum(1)[:, None]
                rows.append(dict(q=q, delta=d, N=N, form='pinned' if pin else 'IVP', conv=sol['converged'],
                                 cond_raw=np.linalg.cond(J), cond_rowscaled=np.linalg.cond(Js)))
cs = pd.DataFrame(rows); cs.to_csv(R + 'conditioning_scaled.csv', index=False)
print(cs.groupby(['q', 'form', 'delta'])[['cond_raw', 'cond_rowscaled']].agg(['min', 'max']).to_string())

# (3)+(4) reconstruction and off-grid defect
c2 = pd.read_csv(R + 'colloc_q2.csv'); c3 = pd.read_csv(R + 'colloc_q3.csv')
xe = np.cos(np.pi * (np.arange(4000) + 0.5) / 4000)
rows = []
for q, df, deltas in [(2, c2, [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]), (3, c3, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])]:
    for d in deltas:
        for N in sorted({nmin(df, d), 384}):
            S = setup(d, N, q); mu = S['mu']
            gpe = S['eps'] * S['L'] * np.cosh(S['L'] * xe + S['M'])
            ex = master(S['svar'](xe), q)
            D, _ = cheb(N)
            for pin in (True, False):
                sol = solve(d, N, q, pin); u = np.log(sol['y'])
                p = bary_eval(S['xi'], u, xe); dp = bary_eval(S['xi'], D @ u, xe) / gpe
                defect = np.max(np.abs(mu * dp - (np.exp((q - 1) * p) - np.exp(q * p))))
                rows.append(dict(q=q, delta=d, N=N, Nmin=(N == nmin(df, d)), form='pinned' if pin else 'IVP',
                                 conv=sol['converged'], err_Iy=np.max(np.abs(bary_eval(S['xi'], sol['y'], xe) - ex)),
                                 err_expIv=np.max(np.abs(np.exp(p) - ex)), defect=defect))
rd = pd.DataFrame(rows); rd.to_csv(R + 'reconstruction_defect.csv', index=False); print(rd.to_string())
