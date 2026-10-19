"""Figures 2-4 of the paper (all q = 2 data from the general-q code collocation_q.py).
Reads results/colloc_q2.csv, colloc_q2_unmapped.csv, colloc_q3.csv.
Writes figs/fig2_grid_cost.pdf, fig3_collocation.pdf, fig4_q3.pdf."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collocation_q import solve
from flame_q import master

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#555555']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
R = ROOT + '/results/'; FIG = ROOT + '/figs/'
c2 = pd.read_csv(R + 'colloc_q2.csv'); cc = pd.read_csv(R + 'colloc_q2_unmapped.csv')
c3 = pd.read_csv(R + 'colloc_q3.csv')

# ---------------------------------------------------------------- Fig B1: grid and cost
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
d, N = 1e-6, 128
sol = solve(d, N, 2, pin=True)
s_nodes = sol['svar'](sol['xi'])
ss = np.linspace(-40, 25, 1500)
ax[0].plot(ss, master(ss, 2), color=C[0], lw=1.2, label=r'$Y_2(s)$')
w = (s_nodes > -40) & (s_nodes < 25)
ax[0].plot(s_nodes[w], sol['y'][w], 'o', color=C[1], ms=3, mfc='none', label=f'nodes, $N={N}$')
ax[0].set_xlabel(r'$s=(x-x_0)/\mu$'); ax[0].set_ylabel('$y$')
ax[0].set_title(r'(a) $q=2$, $\delta=10^{-6}$: nodes near the front', fontsize=9)
ax[0].legend(fontsize=7, loc='upper left')
pts = []
for q, df in [(2, c2[c2.delta.isin([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])]), (3, c3)]:
    for dd, g in df.groupby('delta'):
        mu = dd if q == 2 else 2 * dd**2
        pts.append((q, mu, g[g.pin <= 1e-10].N.min()))
pts = pd.DataFrame(pts, columns=['q', 'mu', 'N'])
for i, q in enumerate([2, 3]):
    s = pts[pts.q == q].sort_values('mu')
    ax[1].semilogx(s.mu, s.N, MK[i], color=C[i], ms=5, mfc='none', label=f'$q={q}$: smallest $N$ with error $\\leq10^{{-10}}$')
mm = np.logspace(-12.5, -0.8, 50)
ax[1].semilogx(mm, 2 / np.pi * np.log(2 / (np.pi * mm)) * np.log(1e10), '-', color='#555555', lw=0.9,
               label=r'$\frac{2}{\pi}\ln\frac{2}{\pi\mu}\,\ln 10^{10}$')
ax[1].invert_xaxis(); ax[1].set_xlabel(r'layer width parameter $\mu$'); ax[1].set_ylabel('$N$')
ax[1].set_title('(b) nodes needed for $10^{-10}$', fontsize=9)
ax[1].legend(fontsize=6.5, loc='upper left')
fig.tight_layout(); fig.savefig(FIG + 'fig2_grid_cost.pdf'); plt.close(fig)
print(pts)

# ---------------------------------------------------------------- Fig B2: q = 2 convergence
fig, ax = plt.subplots(1, 2, figsize=(7, 3.3))
for i, d in enumerate([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]):
    s = c2[c2.delta == d]
    ax[0].semilogy(s.N, s.pin, MK[i] + '-', color=C[i], ms=3, lw=1, label=fr'$10^{{{int(np.log10(d))}}}$')
for i, d in enumerate([1e-6, 1e-10]):
    s = c2[c2.delta == d]
    ax[0].semilogy(s.N, s.ivp, 'x--', color=C[[2, 4][i]], ms=3.5, lw=0.8)
ax[0].set_xlabel('$N$'); ax[0].set_ylabel('max error'); ax[0].set_ylim(1e-16, 1)
ax[0].legend(title=r'$\delta$ (pinned form, solid)', fontsize=6.5, title_fontsize=7, ncol=6, loc='upper center',
             bbox_to_anchor=(0.5, -0.2), columnspacing=0.8, handletextpad=0.3)
ax[0].set_title('(a) sinh-mapped Chebyshev; dashed: IVP form', fontsize=9)
for i, d in enumerate([1e-1, 1e-2, 1e-3]):
    s = cc[cc.delta == d]
    ax[1].semilogy(s.N, s.err, MK[i] + '-', color=C[i], ms=3.5, lw=1, label=fr'plain, $\delta=10^{{{int(np.log10(d))}}}$')
s = c2[c2.delta == 1e-12]
ax[1].semilogy(s.N, s.pin, '*-', color=C[5], ms=4, lw=1, label=r'sinh map, $\delta=10^{-12}$')
ax[1].set_xlabel('$N$'); ax[1].set_xscale('log')
ax[1].legend(fontsize=6.5, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2), columnspacing=0.8)
ax[1].set_title('(b) unmapped Chebyshev vs. mapped', fontsize=9); ax[1].set_ylim(1e-16, 1)
fig.tight_layout(); fig.savefig(FIG + 'fig3_collocation.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig B3: q = 3 convergence
fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.1))
for i, dd in enumerate([1e-2, 1e-4, 1e-6]):
    s = c3[c3.delta == dd]
    ax.semilogy(s.N, s.pin, MK[i] + '-', color=C[i], ms=3, lw=1, label=fr'$\delta=10^{{{int(np.log10(dd))}}}$, pinned')
    ax.semilogy(s.N, s.ivp, 'x--', color=C[i], ms=3, lw=0.8, label=fr'$\delta=10^{{{int(np.log10(dd))}}}$, IVP form')
ax.set_xlabel('$N$'); ax.set_ylabel('max error'); ax.set_ylim(1e-16, 1)
ax.legend(fontsize=6.5, ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout(); fig.savefig(FIG + 'fig4_q3.pdf'); plt.close(fig)
