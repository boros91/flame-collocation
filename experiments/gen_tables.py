"""Tables and in-text numbers of the paper, generated from the CSV files in results/.
Writes tables/table1_q2.tex ... table7_aaa.tex (the table bodies)
and results/text_numbers.txt (the numbers quoted in the text)."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd
R = ROOT + '/results/'; OUT = ROOT + '/tables/'


def sci(x, d=1):
    if not np.isfinite(x):
        return '--'
    m, e = f'{x:.{d}e}'.split('e')
    return f'${m}\\cdot10^{{{int(e)}}}$'


def colloc_rows(df, deltas):
    lines = []
    for d in deltas:
        g = df[np.isclose(df.delta, d, rtol=1e-9, atol=0)]
        L = g.L.iloc[0]
        n = int(g[g.pin <= 1e-10].N.min()); r = g[g.N == n].iloc[0]; r384 = g[g.N == 384].iloc[0]
        ivp = sci(r384.ivp) if r384.ivp_conv else sci(r384.ivp)[:-1] + '{}^{\\ast}$'
        lines.append(' & '.join([f'$10^{{{int(round(np.log10(d)))}}}$', f'{L:.2f}', f'{2*L/np.pi*np.log(1e10):.0f}',
                                 f'{n}', sci(r.pin), f'{int(r.pin_its)}', sci(r.y0_relerr), ivp,
                                 f'{int(r384.ivp_its)}']) + r' \\')
    return lines


c2 = pd.read_csv(R + 'colloc_q2.csv'); c3 = pd.read_csv(R + 'colloc_q3.csv')
open(OUT + 'table1_q2.tex', 'w').write('\n'.join(colloc_rows(c2, [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])) + '\n')
open(OUT + 'table4_q3.tex', 'w').write('\n'.join(colloc_rows(c3, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])) + '\n')

# Chebfun table: MATLAB runs + this paper (pinned, sinh map) on the full N-grid
cf = pd.read_csv(R + 'matlab/chebfun_flame.csv')
lines = []
for d in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    g = cf[np.isclose(cf.delta, d, rtol=1e-9, atol=0)].set_index('mode')
    o, ci, cp = g.loc['ode113'], g.loc['colloc_ivp'], g.loc['colloc_pin']
    ok = lambda r: isinstance(r.message, float) or r.message == ''
    cells = [f'$10^{{{int(round(np.log10(d)))}}}$']
    cells += [f'{int(o.length)}', sci(o.maxerr)] if ok(o) else ['--', 'not resolved']
    cells += [f'{int(ci.length)}', f'{int(ci.newton_its)}', sci(ci.maxerr) if ok(ci) else 'failed']
    cells += ([f'{int(cp.length)}', f'{int(cp.newton_its)}', sci(cp.maxerr)] if ok(cp)
              else ['--', f'{int(cp.newton_its)}', 'failed'])
    h = c2[np.isclose(c2.delta, d, rtol=1e-9, atol=0)]
    n = int(h[h.pin <= 1e-10].N.min()); r = h[h.N == n].iloc[0]
    cells += [f'{n + 1}', f'{int(r.pin_its)}', sci(r.pin)]
    lines.append(' & '.join(cells) + r' \\')
open(OUT + 'table5_chebfun.tex', 'w').write('\n'.join(lines) + '\n')

# numbers quoted in the text
T = []
big = c2[c2.N >= 256]
for d in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    g = big[np.isclose(big.delta, d, rtol=1e-9, atol=0)]
    T.append(f'q=2 delta={d:.0e}, N=256..512: cond pinned {g.pin_cond.min():.2e}..{g.pin_cond.max():.2e}, '
             f'cond IVP {g.ivp_cond.min():.2e}..{g.ivp_cond.max():.2e}')
nc = c2[(c2.ivp_conv == 0)]
T.append('q=2 IVP not converged (delta, N): ' + ', '.join(f'({d:.0e},{n})' for d, n in zip(nc.delta, nc.N)))
wr = c2[(c2.ivp_conv == 1) & (c2.ivp > 1e-2)]
T.append('q=2 IVP converged with error > 1e-2: ' + ', '.join(f'({d:.0e},{n}: {e:.3f})' for d, n, e in zip(wr.delta, wr.N, wr.ivp)))
p = c2[c2.delta <= 1e-4]
T.append(f'q=2 pinned its: {sorted(c2.pin_its.unique())}; max |y(-1)/delta-1| at N>=N_min, delta<=1e-4: '
         + f"{max(p[(p.delta == d) & (p.N >= p[(p.delta == d) & (p.pin <= 1e-10)].N.min())].y0_relerr.max() for d in p.delta.unique()):.2e}")
m = c2.interp > 1e-13
T.append(f'q=2 pinned/interp (interp > 1e-13): {(c2.pin / c2.interp)[m].min():.3f}..{(c2.pin / c2.interp)[m].max():.3f}')
T.append(f'q=3 pinned cond: delta=1e-1 {c3[c3.delta == 1e-1].pin_cond.min():.2e}..{c3[c3.delta == 1e-1].pin_cond.max():.2e}, '
         f'delta=1e-6 {c3[np.isclose(c3.delta, 1e-6, rtol=1e-9, atol=0)].pin_cond.min():.2e}..{c3[np.isclose(c3.delta, 1e-6, rtol=1e-9, atol=0)].pin_cond.max():.2e}; '
         f'IVP cond max {c3.ivp_cond.max():.2e}')
T.append('q=3 pinned its != 5: ' + ', '.join(f'({d:.0e},{n}:{k})' for d, n, k in zip(c3.delta, c3.N, c3.pin_its) if k != 5))
rs = pd.read_csv(R + 'rounding_sensitivity_q2.csv'); mm = rs.pin_1thread > 1e-12
T.append(f'rounding sensitivity (1 vs 2 BLAS threads, {len(rs)} runs): pinned max rel. change '
         f'{(abs(rs.pin_1thread - rs.pin_2threads) / rs.pin_1thread)[mm].max():.1e}, pinned its changed '
         f'{(rs.pin_its_1thread != rs.pin_its_2threads).sum()}; IVP its changed {(rs.ivp_its_1thread != rs.ivp_its_2threads).sum()}, '
         f'IVP convergence changed {(rs.ivp_conv_1thread != rs.ivp_conv_2threads).sum()}, IVP error changed >0.1%: '
         f'{((abs(rs.ivp_1thread - rs.ivp_2threads) / rs.ivp_1thread) > 1e-3).sum()}')
same = rs.merge(c2, on=['delta', 'N'])
T.append(f'2-thread run vs colloc_q2.csv: max rel. diff pinned {(abs(same.pin_2threads - same.pin) / same.pin).max():.1e}, '
         f'IVP its differ {(same.ivp_its_2threads != same.ivp_its).sum()}, IVP conv differ {(same.ivp_conv_2threads != same.ivp_conv).sum()}')
from collocation_q import params
for q, d in [(2, 1e-4), (2, 1e-6), (2, 1e-8), (3, 1e-1), (3, 1e-2), (3, 1e-3), (3, 1e-4), (3, 1e-6)]:
    mu, x0 = params(d, q)
    T.append(f'x0/(pi mu): q={q}, delta={d:.0e}: {x0 / (np.pi * mu):.1f}')
# Table: AAA poles of the computed solutions (exp_aaa.py)
aa = pd.read_csv(R + 'aaa_poles.csv')
names = {'unmapped': 'unmapped', 'sinh map': 'sinh map'}
T.append(f'AAA: Im(pole1)/Im(s*) {aa.pole1_im_over_s_star_im.min():.3f}..{aa.pole1_im_over_s_star_im.max():.3f}, '
         f'max |Re(pole1) - Re(s*)| {(aa.pole1_re - aa.s_star_re).abs().max():.4f}')
# movable singularities (exp_movable.py)
mv = pd.read_csv(R + 'movable_singularities.csv')
T.append(f'movable: AAA max |Re t* - exact| {(mv.aaa_re - mv.exact_re).abs().max():.4f}, Im {mv.aaa_im.min():.3f}..{mv.aaa_im.max():.3f}; '
         f'Pade [40/40] max rel Re error {((mv.pade_re - mv.exact_re).abs() / mv.exact_re).max():.1e}, Im {mv.pade_im.min():.3f}..{mv.pade_im.max():.3f}; '
         f'Re t* range {mv.exact_re.min():.4g}..{mv.exact_re.max():.4g}')
# Table: choice of the unknown (exp_unknowns.py)
va = pd.read_csv(R + 'unknowns_q2.csv')
def cell(r):
    c = sci(r.error)
    return c if r.conv else c[:-1] + '{}^{\\ast}$'
lines = []
for d in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    g = va[np.isclose(va.delta, d, rtol=1e-9, atol=0)]
    cells = [f'$10^{{{int(round(np.log10(d)))}}}$', f'{int(g.N.iloc[0])}', sci(g.interp.iloc[0])]
    for var in ['y', 'ln y', '1/y']:
        for form in ['IVP', 'pinned']:
            cells.append(cell(g[(g['var'] == var) & (g.form == form)].iloc[0]))
    lines.append(' & '.join(cells) + r' \\')
open(OUT + 'table2_unknowns.tex', 'w').write('\n'.join(lines) + '\n')
w = va[va['var'] == '1/y']
T.append('variables: 1/y its ' + ', '.join(f'{f}:{int(w[w.form == f].its.min())}-{int(w[w.form == f].its.max())}' for f in ['IVP', 'pinned'])
         + '; 1/y IVP error/(u/delta): ' + ', '.join(f'{r.delta:.0e}:{r.error / (2.0**-53 / r.delta):.2f}' for _, r in w[w.form == 'IVP'].iterrows()))

# Table: Arrhenius rate (exp_arrhenius.py), alpha = 0.8
ar = pd.read_csv(R + 'arrhenius_runs.csv'); ap = pd.read_csv(R + 'arrhenius_params.csv')
lines = []
for _, q in ap.iterrows():
    g = ar[ar.beta == q.beta]; pn = g[g.form == 'pin']; iv = g[g.form == 'IVP']
    n = int(pn[pn.err <= 1e-10].N.min()); r = pn[pn.N == n].iloc[0]; v = iv[iv.N == n].iloc[0]
    ivc = sci(v.err) if v.conv else sci(v.err)[:-1] + '{}^{\\ast}$'
    lines.append(' & '.join([f'{int(q.beta)}', sci(q.f0), sci(q.mu), f'{q.yf:.4f}',
                             f'${q.sre:.4f}+{q.sim:.4f}i$', f'{q.L:.1f}', f'{n}', sci(r.err), f'{int(r.its)}', ivc]) + r' \\')
open(OUT + 'table6_arrhenius.tex', 'w').write('\n'.join(lines) + '\n')
p64 = ar[(ar.form == 'pin') & (ar.N >= 64)]; iv = ar[ar.form == 'IVP']
T.append('arrhenius: pinned N>=64 its by beta ' + ', '.join(f'{int(b)}:{int(g.its.min())}-{int(g.its.max())}' for b, g in p64.groupby('beta'))
         + f'; all conv {bool(p64.conv.all())}; N=32 failures: ' + ', '.join(str(int(b)) for b in ar[(ar.form == "pin") & (ar.N == 32) & (~ar.conv)].beta))
T.append(f'arrhenius: pinned err/interp (N>=64) {(p64.err / p64.interp).min():.2f}..{(p64.err / p64.interp).max():.2f}; '
         f'min err by beta ' + ', '.join(f'{int(b)}:{g.err.min():.1e}' for b, g in p64.groupby('beta')))
T.append('arrhenius: IVP err range by beta ' + ', '.join(f'{int(b)}:{g.err.min():.1e}..{g.err.max():.1e} (nonconv {int((~g.conv).sum())})' for b, g in iv.groupby('beta'))
         + '; beta=2 N>=160: ' + f'{iv[(iv.beta == 2) & (iv.N >= 160)].err.min():.1e}..{iv[(iv.beta == 2) & (iv.N >= 160)].err.max():.1e}')
T.append('arrhenius: rate ratio ' + ', '.join(f'{int(r.beta)}:{r.rate_ratio:.3f}' for _, r in ap.iterrows())
         + f'; max |Im s* - pi(1-exp(-beta/alpha))| {ap.im_diff.max():.1e}; ref check max {ap.ref_check.max():.1e}'
         + '; Im s*/pi ' + ', '.join(f'{r.sim / np.pi:.5f}' for _, r in ap.iterrows())
         + '; amp f(yf)/f(0) ' + ', '.join(f'{r.amp:.1e}' for _, r in ap.iterrows()))
T.append('arrhenius checks: ' + open(R + 'arrhenius_checks.txt').read().replace('\n', ' | '))
# q-family rows of exp_aaa.py + Arrhenius row (exp_aaa_arrhenius.py)
aq = aa; ah = pd.read_csv(R + 'aaa_poles_arrhenius.csv').iloc[0]
lines = []
for _, r in aq.iterrows():
    lines.append(' & '.join([f'$q={int(r.q)}$, $\\delta=10^{{{int(round(np.log10(r.delta)))}}}$', f'{int(r.N)}', names[r.case],
                             f'${r.s_star_im:.4f}i$', f'${r.pole1_re:.4f}+{r.pole1_im:.4f}i$', f'{r.pole2_im:.3f}, {r.pole3_im:.3f}',
                             f'${r.exact_pole1_re:.4f}+{r.exact_pole1_im:.4f}i$']).replace('+-', '-') + r' \\')
lines.append(' & '.join([f'Arrhenius, $\\beta={int(ah.beta)}$', f'{int(ah.N)}', 'sinh map',
                         f'${ah.s_star_re:.4f}+{ah.s_star_im:.4f}i$', f'${ah.pole1_re:.4f}+{ah.pole1_im:.4f}i$',
                         f'{ah.pole2_im:.3f}, {ah.pole3_im:.3f}', f'${ah.exact_pole1_re:.4f}+{ah.exact_pole1_im:.4f}i$']).replace('+-', '-') + r' \\')
open(OUT + 'table7_aaa.tex', 'w').write('\n'.join(lines) + '\n')
allr = pd.concat([aq[['pole1_re', 's_star_re', 'pole1_im_over_s_star_im']], pd.DataFrame([ah])[['pole1_re', 's_star_re', 'pole1_im_over_s_star_im']]])
T.append(f'AAA: Im(pole1)/Im(s*) {allr.pole1_im_over_s_star_im.min():.4f}..{allr.pole1_im_over_s_star_im.max():.4f}, '
         f'max |Re(pole1) - Re(s*)| {(allr.pole1_re - allr.s_star_re).abs().max():.4f}')

# numbers (exp_rate_conditioning.py, exp_highprec_refcheck.py)
re_ = pd.read_csv(R + 'rate_exact.csv'); hp_ = re_[re_.set == '256-bit']; ah_ = re_[re_.set == 'Arrhenius']
T.append(f'rate: 256-bit ratio to log rho* {hp_.ratio_exact.min():.4f}..{hp_.ratio_exact.max():.4f}, to pi/2L {hp_.ratio_thin.min():.4f}..{hp_.ratio_thin.max():.4f}; '
         f'Arrhenius ratio to log rho* {ah_.ratio_exact.min():.4f}..{ah_.ratio_exact.max():.4f}; q=2 d=1e-2 log rho* {hp_[(hp_.q == 2) & np.isclose(hp_.param, 1e-2)].log_rho.iloc[0]:.4f} '
         f'pi/2L {hp_[(hp_.q == 2) & np.isclose(hp_.param, 1e-2)].thin.iloc[0]:.4f}; max rel diff for mu<=2e-4: '
         f'{(1 - hp_[hp_.param.isin([1e-4, 1e-6, 1e-2, 1e-3]) & ~((hp_.q == 2) & np.isclose(hp_.param, 1e-2))].pipe(lambda g: g.log_rho / g.thin)).max():.4f}')
cs_ = pd.read_csv(R + 'conditioning_scaled.csv')
for q in (2, 3):
    for f in ('pinned', 'IVP'):
        g = cs_[(cs_.q == q) & (cs_.form == f)]
        T.append(f'cond q={q} {f}: row-scaled ' + ', '.join(f'{d:.0e}:{h.cond_rowscaled.min():.2g}..{h.cond_rowscaled.max():.2g}' for d, h in g.groupby('delta'))
                 + f'; overall {g.cond_rowscaled.min():.3g}..{g.cond_rowscaled.max():.3g}')
rd_ = pd.read_csv(R + 'reconstruction_defect.csv'); pm = rd_[(rd_.form == 'pinned') & rd_.Nmin]
T.append(f'reconstruction err_Iy/err_expIv at Nmin (pinned): {(pm.err_Iy / pm.err_expIv).min():.1f}..{(pm.err_Iy / pm.err_expIv).max():.1f}; '
         f'pinned defect at Nmin max {pm.defect.max():.2e}')
wr = rd_[(rd_.form == 'IVP') & (rd_.err_Iy >= 2e-3)]
T.append(f'defect: IVP with err>=2e-3 defect {wr.defect.min():.2e}..{wr.defect.max():.2e}; phase-error cases: ' + '; '.join(
    f'q=2 d={r.delta:.0e} N={r.N} IVP err {r.err_Iy:.1e} defect {r.defect:.2e}' for _, r in rd_[(rd_.form == 'IVP') & rd_.Nmin & (rd_.q == 2) & rd_.delta.isin([1e-6, 1e-8])].iterrows()))
hr_ = pd.read_csv(R + 'hp_reference_check.csv')
T.append(f'HP reference 256 vs 512 bits: max diff {hr_.max_diff.max():.1e}')

# translation mode (exp_translation_mode.py)
tm_ = pd.read_csv(R + 'translation_mode.csv'); ok_ = tm_[tm_.cond_ivp < 3e15]
T.append(f'translation mode: IVP col/pred within {abs(ok_.ratio - 1).max() * 100:.1f}% for cond<3e15 (' + ', '.join(f'q={int(r.q)} d={r.delta:.0e}' for _, r in ok_.iterrows())
         + f'); pinned col {tm_.col_pin.min():.4f}..{tm_.col_pin.max():.4f}; smallest cond among failures {tm_[tm_.cond_ivp >= 3e15].cond_ivp.min():.1e}')
# ablation (mapping x condition x formulation) and robustness (exp_ablation.py)
ab = pd.read_csv(R + 'ablation.csv')
def acell(r):
    if not np.isfinite(r.err) or r.err > 1e3 or (not r.conv and r.its < 80):
        return 'diverged'
    c = sci(r.err)
    return (c if r.conv else c[:-1] + '{}^{\\ast}$') + f' ({int(r.its)})'
lines = []
for d in sorted(ab.delta.unique(), reverse=True):
    g = ab[ab.delta == d]
    cells = [f'$10^{{{int(round(np.log10(d)))}}}$', sci(g[g.mapping == 'sinh'].interp.iloc[0])]
    for m, f in [('none', 'diff'), ('sinh', 'diff'), ('sinh', 'int')]:
        for c in ['IVP', 'pin']:
            cells.append(acell(g[(g.mapping == m) & (g.form == f) & (g.cond == c)].iloc[0]))
    lines.append(' & '.join(cells) + r' \\')
open(OUT + 'table3_ablation.tex', 'w').write('\n'.join(lines) + '\n')
sm = ab[ab.mapping == 'sinh']
def piv(f, c): return sm[(sm.form == f) & (sm.cond == c)].set_index('delta')
dI, iI, dP, iP = piv('diff', 'IVP'), piv('int', 'IVP'), piv('diff', 'pin'), piv('int', 'pin')
T.append('ablation: int IVP err/interp ' + ', '.join(f'{d:.0e}:{iI.err[d] / iI.interp[d]:.1e}' for d in iI.index)
         + '; diff IVP err / int IVP err ' + ', '.join(f'{d:.0e}:{dI.err[d] / iI.err[d]:.2f}' for d in iI.index)
         + f'; pinned errors (both forms) {min(dP.err.min(), iP.err.min()):.1e}..{max(dP.err.max(), iP.err.max()):.1e}; '
         + f'max final nodal residual of converged runs {ab[ab.conv].res_nodal.max():.1e}; unmapped int vs diff max rel err diff '
         + f"{(abs(ab[(ab.mapping == 'none') & (ab.form == 'int')].err.values - ab[(ab.mapping == 'none') & (ab.form == 'diff')].err.values) / ab[(ab.mapping == 'none') & (ab.form == 'diff')].err.values).max():.1e}")
rb = pd.read_csv(R + 'robustness.csv'); rb['ratio'] = rb.err / rb.interp
for t, g in rb.groupby('test'):
    T.append(f'robustness {t}: err/interp {g.ratio.min():.2f}..{g.ratio.max():.2f}, its {int(g.its.min())}-{int(g.its.max())}, interp '
             + ', '.join(f'({r.delta:.0e},{r.value:g}):{r.interp:.1e}' for _, r in g.iterrows())
             + (f'; bound max {g.bound.max():.2f}' if t == 'pin value' else ''))
# final residuals of all runs (exp_residuals.py), row-scaled condition numbers of the ablation (exp_ablation.py)
T += ['residuals: ' + l.strip() for l in open(R + 'residuals_summary.txt')]
for (m, f, c), g in ab.groupby(['mapping', 'form', 'cond']):
    T.append(f'ablation row-scaled cond2 {m} {f} {c}: {g.cond_rowscaled.min():.1e}..{g.cond_rowscaled.max():.1e}')
T.append('timing: ' + ' | '.join(l.strip() for l in open(R + 'timing.txt')))
open(R + 'text_numbers.txt', 'w').write('\n'.join(T) + '\n')
print('\n'.join(T))
