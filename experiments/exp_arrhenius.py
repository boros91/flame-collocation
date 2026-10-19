"""Sect. 5.6: pinned vs IVP collocation for the Arrhenius rate, alpha = 0.8, beta = 2..6.
Writes results/arrhenius_runs.csv, results/arrhenius_params.csv, results/arrhenius_checks.txt."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd, mpmath as mp
from scipy.interpolate import AAA
from arrhenius_model import Arr, solve

ALPHA = 0.8
BETAS = [2, 3, 4, 5, 6]
NS = list(range(32, 513, 32))
rows, prows, chk = [], [], []

# check 1: alpha = 0 (Frank-Kamenetskii exponential), s* = Ei(beta u_f) + i pi exactly
for beta in [4, 6]:
    P0 = Arr(beta, 0.0); s0 = P0.singularity()
    ex = complex(mp.ei(beta * P0.uf) + 1j * mp.pi)
    chk.append(f'alpha=0 beta={beta}: s* quad = {s0:.15g}  Ei formula = {ex:.15g}  |diff| = {abs(s0-ex):.2e}')

for beta in BETAS:
    P = Arr(beta, ALPHA); sst = P.singularity()
    imf = np.pi * (1 - np.exp(-beta / ALPHA))
    # check 2: reference Y against 40-digit quadrature
    mp.mp.dps = 40
    b, a = mp.mpf(beta), mp.mpf(ALPHA); yfm = mp.mpf(P.yf)
    finv = lambda z: mp.exp(b * (1 - z) / (1 - a * (1 - z))) / (1 - z)
    dmax = 0.0
    for S in [-0.9 * P.T, -0.5 * P.T, -0.1 * P.T, -30, -3, -0.5, 0.5, 2, 10, 20]:
        y = P.Y(np.array([S]))[0]
        if y < 1:
            dmax = max(dmax, abs(float(mp.mpf(P.f(y)) * (mp.quad(finv, [yfm, mp.mpf(y)]) - S))))
    # check 3: AAA on Y sampled on the real s-axis
    Sg = np.linspace(-60, 60, 3000); r = AAA(Sg, P.Y(Sg), rtol=1e-13)
    pol, res = r.poles(), r.residues()
    m = (np.abs(res) > 1e-8) & (np.abs(pol.real) < 60) & (pol.imag > 0)
    pa = pol[m][np.argmin(pol[m].imag)]
    for N in NS:
        for pin in (True, False):
            rr = solve(P, N, sst, pin); rr.update(beta=beta)
            rows.append(rr)
            print(beta, N, rr['form'], f"{rr['err']:.2e} {rr['interp']:.2e} its={rr['its']} conv={rr['conv']}", flush=True)
    df = pd.DataFrame([x for x in rows if x['beta'] == beta and x['form'] == 'pin'])
    sel = df[(df.err > 1e-12) & (df.err < 1e-2)]
    slope = -np.polyfit(sel.N, np.log(sel.err), 1)[0]
    L = df.L.iloc[0]
    prows.append(dict(beta=beta, alpha=ALPHA, f0=P.f(0.0), T=P.T, mu=P.mu, yf=P.yf, fyf=P.f(P.yf),
                      amp=P.f(P.yf) / P.f(0.0), sre=sst.real, sim=sst.imag, im_formula=imf,
                      im_diff=abs(sst.imag - imf), aaa_re=pa.real, aaa_im=pa.imag, L=L,
                      rate_ratio=slope / (np.pi / (2 * L)), ref_check=dmax))
    print(prows[-1], flush=True)

pd.DataFrame(rows).to_csv(ROOT + '/results/arrhenius_runs.csv', index=False)
pd.DataFrame(prows).to_csv(ROOT + '/results/arrhenius_params.csv', index=False)
open(ROOT + '/results/arrhenius_checks.txt', 'w').write('\n'.join(chk) + '\n')
print('\n'.join(chk))
