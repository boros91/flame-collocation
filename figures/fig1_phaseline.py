"""Phase line of y' = y^q (1-y), q = 2, 3: f_q and f_q/y, equilibria and pin points."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.3,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
y = np.linspace(-0.3, 1.15, 600)
for k, q in enumerate((2, 3)):
    a = ax[k]
    f = y**q * (1 - y); Q = y**(q - 1) * (1 - y)
    a.plot(y, f, color='#2a78d6', label=fr'$f_{q}(y)=y^{q}(1-y)$')
    a.plot(y, Q, '--', color='#eb6834', lw=1.0, label=fr'$f_{q}(y)/y=y^{{{q-1}}}(1-y)$' if q > 2 else r'$f_2(y)/y=y(1-y)$')
    a.axhline(0, color='k', lw=0.6)
    # equilibria: 0 (degenerate), 1 (stable)
    a.plot([0], [0], 'o', ms=6, mfc='white', mec='k', zorder=5)
    a.plot([1], [0], 'o', ms=6, color='k', zorder=5)
    # arrows on the phase line
    for x0, dx in [(0.35, 0.12), (0.7, 0.12), (1.08, -0.05)]:
        a.annotate('', xy=(x0 + dx, 0), xytext=(x0, 0), arrowprops=dict(arrowstyle='->', color='k', lw=1))
    if q % 2 == 0:
        a.annotate('', xy=(-0.08, 0), xytext=(-0.25, 0), arrowprops=dict(arrowstyle='->', color='k', lw=1))
    else:
        a.annotate('', xy=(-0.25, 0), xytext=(-0.08, 0), arrowprops=dict(arrowstyle='->', color='k', lw=1))
    ys = (q - 1) / q
    a.plot([ys], [ys**(q - 1) * (1 - ys)], 's', ms=4, color='#eb6834')
    a.annotate(fr'$y^\ast=\frac{{{q-1}}}{{{q}}}$', (ys, ys**(q - 1) * (1 - ys)), textcoords='offset points',
               xytext=(4, 4), fontsize=8, color='#eb6834')
    a.set_xlabel('$y$'); a.set_xlim(-0.3, 1.15)
    a.set_title(f'({"ab"[k]}) $q={q}$', fontsize=9)
    a.legend(fontsize=7, loc='upper left')
fig.tight_layout(); fig.savefig(ROOT + '/figs/fig1_phaseline.pdf'); plt.close(fig)
