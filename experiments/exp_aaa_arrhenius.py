"""AAA location of the singularity from the COMPUTED pinned solution of the Arrhenius problem
(alpha = 0.8, beta = 6, N = 384), as in exp_aaa.py: 3000 equispaced points of s in [-30, 30]
(s = Theta(y) - Theta(y_f)), rtol = 1e-13, poles with |residue| < 1e-8 discarded.
Writes results/aaa_poles_arrhenius.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import warnings
import numpy as np, pandas as pd
from scipy.interpolate import AAA
from arrhenius_model import Arr, solve
from collocation import bary_eval
warnings.filterwarnings('ignore')


def lowest_poles(s, y, k=3):
    r = AAA(s, y, rtol=1e-13)
    p, res = r.poles(), r.residues()
    p = p[(p.imag > 0) & (np.abs(res) > 1e-8)]
    p = p[np.argsort(p.imag)]
    return p[:k], len(r.poles())


beta, alpha, N = 6, 0.8, 384
P = Arr(beta, alpha); sst = P.singularity()
r = solve(P, N, sst, True, return_sol=True)
xi, y, L, M = r['sol']
s = np.linspace(-30, 30, 3000)
yc = bary_eval(xi, y, (np.arcsinh((s - sst.real) / sst.imag) - M) / L)
pc, nc = lowest_poles(s, yc); pe, ne = lowest_poles(s, P.Y(s))
row = dict(case='Arrhenius', beta=beta, alpha=alpha, N=N, err=r['err'], s_star_re=sst.real, s_star_im=sst.imag,
           pole1_re=pc[0].real, pole1_im=pc[0].imag, pole2_im=pc[1].imag, pole3_im=pc[2].imag,
           pole1_im_over_s_star_im=pc[0].imag / sst.imag, npoles=nc,
           exact_pole1_re=pe[0].real, exact_pole1_im=pe[0].imag)
print(row)
pd.DataFrame([row]).to_csv(ROOT + '/results/aaa_poles_arrhenius.csv', index=False)
