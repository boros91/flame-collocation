"""Figure 6 of the paper (Sect. 5.6): Arrhenius rate, alpha = 0.8, beta = 2..6.
(a) solution profiles y(x); (b) maximal error vs N, pinned and IVP forms. Uses results/arrhenius_runs.csv."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arrhenius_model import Arr
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a4fd1']
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
df = pd.read_csv(ROOT + '/results/arrhenius_runs.csv')
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.9))
S = np.linspace(-60, 15, 1500)
def tex(m):
    e = int(np.floor(np.log10(m))); c = m / 10**e
    return fr'{c:.1f}\cdot10^{{{e}}}'
for i, beta in enumerate(sorted(df.beta.unique())):
    P = Arr(beta, 0.8)
    a.plot(S, P.Y(S), color=C[i], label=fr'$\beta={beta}$, $\mu={tex(P.mu)}$')
    g = df[df.beta == beta]
    p = g[g.form == 'pin'].sort_values('N'); v = g[g.form == 'IVP'].sort_values('N')
    b.semilogy(p.N, p.err, 'o-', color=C[i], ms=3)
    b.semilogy(v.N, v.err, 'x--', color=C[i], ms=3.5, lw=0.7)
    L = p.L.iloc[0]; pp = p[(p.err > 1e-13) & p.err.notna()]
    nn = np.linspace(pp.N.iloc[0], pp.N.iloc[-1], 20)
    b.semilogy(nn, pp.err.iloc[0] * np.exp(-np.pi * (nn - nn[0]) / (2 * L)), ':', color='#777777', lw=0.8)
a.set_xlabel('$s=x/\\mu$'); a.set_ylabel('$y$'); a.set_title('(a)', fontsize=9); a.legend(fontsize=6.5, loc='upper left')
b.set_xlabel('$N$'); b.set_ylabel('maximal error'); b.set_ylim(1e-16, 10); b.set_title('(b)', fontsize=9)
fig.tight_layout(); fig.savefig(ROOT + '/figs/fig6_arrhenius.pdf')
