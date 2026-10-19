"""Figure 5 of the paper: high-precision (256-bit) collocation errors vs N."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
df = pd.concat([pd.read_csv(ROOT + f'/results/highprec_colloc_q{q}.csv') for q in (2, 3)])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
floor = 2.0**-256
for k, q in enumerate((2, 3)):
    a = ax[k]
    for i, (d, g) in enumerate(df[df.q == q].groupby('delta', sort=False)):
        p = g[g.form == 'pinned'].sort_values('N'); v = g[g.form == 'IVP'].sort_values('N')
        mu = d if q == 2 else 2 * d**2
        lab = fr'$\delta=10^{{{int(round(np.log10(d)))}}}$'
        a.semilogy(p.N, p.error, 'o-', color=C[i], ms=3.5, label=lab + ', pinned')
        a.semilogy(v.N, v.error, '--', color=C[i], lw=0.9)
        vc = v[v.conv]; vn = v[~v.conv]
        a.semilogy(vc.N, vc.error, 'x', color=C[i], ms=4.5, label=lab + ', IVP')
        if len(vn):
            a.semilogy(vn.N, vn.error, 's', color=C[i], ms=4, mfc='none')
        L = p.L.iloc[0]; n0 = p.N.iloc[0]; e0 = p.error.iloc[0]
        nn = np.linspace(n0, 512, 50)
        a.semilogy(nn, e0 * np.exp(-np.pi * (nn - n0) / (2 * L)), ':', color='#777777', lw=0.8)
    a.axhline(floor, color='#999999', lw=0.6)
    a.set_xlabel('$N$'); a.set_ylim(1e-80, 1e2)
    a.set_yticks([1e0, 1e-20, 1e-40, 1e-60, 1e-80])
    a.set_title(f'$q={q}$', fontsize=9)
    a.legend(fontsize=6, loc='lower left', ncol=1)
ax[0].set_ylabel('maximal error')
fig.tight_layout(); fig.savefig(ROOT + '/figs/fig5_highprec.pdf'); plt.close(fig)

# observed vs predicted rate (pinned, points above 1e-70)
rows = []
for (q, d), g in df[df.form == 'pinned'].groupby(['q', 'delta']):
    g = g[g.error > 1e-70].sort_values('N')
    slope = -np.polyfit(g.N, np.log(g.error), 1)[0]
    rows.append(dict(q=q, delta=d, L=g.L.iloc[0], observed=slope, predicted=np.pi / (2 * g.L.iloc[0]),
                     ratio=slope / (np.pi / (2 * g.L.iloc[0])), max_err_over_interp=(g.error / g.interp).max()))
r = pd.DataFrame(rows); print(r.to_string(index=False)); r.to_csv(ROOT + '/results/highprec_rates.csv', index=False)
