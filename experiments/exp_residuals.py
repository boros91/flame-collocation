"""Final residuals of all double-precision Newton runs reported in the paper.
Reruns the solvers of exp_q2.py (Table 1, Fig. 3a), exp_q3.py (Table 4, Fig. 4),
exp_q2_unmapped.py (Fig. 3b), exp_unknowns.py (Table 2), exp_arrhenius.py (Table 6, Fig. 6)
and exp_ablation.py (Table 3, robustness), and records for each run the max-norm of the final
residual vector (collocation rows and condition row), the last damping factor, and the number
of iterations in which the line search found no sufficient decrease (smallest step taken).
The errors, iteration counts and convergence flags are compared with the saved CSV files.
Writes results/residuals_all.csv and results/residuals_summary.txt."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import numpy as np, pandas as pd, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')
R = ROOT + '/results/'


def g_q(q):
    from collocation_q import solve, error
    Ns = range(32, 513, 32)
    ds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12] if q == 2 else [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    out = []
    for d in ds:
        for N in Ns:
            for pin in (True, False):
                s = solve(d, N, q, pin)
                out.append(dict(group=f'q{q}', delta=d, N=N, form='pin' if pin else 'IVP', err=error(s), its=s['its'],
                                conv=bool(s['converged']), res_inf=s['res_inf'], lam_last=s['lam_last'], n_fallback=s['n_fallback']))
    return out


def g_unmapped():
    from exp_q2_unmapped import solve_unmapped
    out = []
    for d in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        for N in [32, 64, 128, 256, 512, 768, 1024, 1536]:
            s = solve_unmapped(d, N)
            out.append(dict(group='unmapped', delta=d, N=N, form='pin', err=s['err'], its=s['its'], conv=bool(s['conv']),
                            res_inf=s['res_inf'], lam_last=s['lam_last'], n_fallback=s['n_fallback']))
    return out


def g_variables():
    from exp_unknowns import solve, FORMS
    ref = pd.read_csv(R + 'unknowns_q2.csv')
    out = []
    for _, r in ref[['delta', 'N']].drop_duplicates().iterrows():
        for var in FORMS:
            for pin in (False, True):
                s = solve(r.delta, int(r.N), var, pin)
                out.append(dict(group='variables ' + var, delta=r.delta, N=int(r.N), form='pin' if pin else 'IVP', err=s['error'],
                                its=s['its'], conv=bool(s['conv']), res_inf=s['res_inf'], lam_last=s['lam_last'], n_fallback=s['n_fallback'],
                                umax=s['umax']))
    return out


def g_arrhenius():
    from arrhenius_model import Arr, solve
    out = []
    for beta in [2, 3, 4, 5, 6]:
        P = Arr(beta, 0.8); sst = P.singularity()
        for N in range(32, 513, 32):
            for pin in (True, False):
                s = solve(P, N, sst, pin)
                out.append(dict(group='arrhenius', delta=beta, N=N, form=s['form'], err=s['err'], its=s['its'], conv=bool(s['conv']),
                                res_inf=s['res_inf'], lam_last=s['lam_last'], n_fallback=s['n_fallback']))
    return out


def g_review2():
    from exp_ablation import solve
    out = []
    ab = pd.read_csv(R + 'ablation.csv')
    for _, r in ab.iterrows():
        s = solve(r.delta, 512, 2, r.cond == 'pin', r.mapping, r.form)
        out.append(dict(group=f'ablation {r.mapping} {r.form}', delta=r.delta, N=512, form=r.cond, err=s['err'], its=s['its'],
                        conv=bool(s['conv']), res_inf=s['res_inf'], lam_last=s['lam_last'], n_fallback=s['n_fallback'],
                        res_nodal=s['res_nodal']))
    rb = pd.read_csv(R + 'robustness.csv')
    for _, r in rb.iterrows():
        kw = {'pin value': dict(y_pin=r.value), 'centre shift / (pi mu)': dict(c_shift=r.value), 'width factor': dict(eps_fac=r.value)}[r.test]
        s = solve(r.delta, int(r.N), 2, True, **kw)
        out.append(dict(group='robustness ' + r.test, value=r.value, delta=r.delta, N=int(r.N), form='pin', err=s['err'], its=s['its'],
                        conv=bool(s['conv']), res_inf=s['res_inf'], lam_last=s['lam_last'], n_fallback=s['n_fallback'],
                        res_nodal=s['res_nodal']))
    return out


def saved():
    """Errors, iterations and flags of the saved CSV files, in the order of the runs above."""
    S = []
    for f, g in [('colloc_q2.csv', 'q2'), ('colloc_q3.csv', 'q3')]:
        c = pd.read_csv(R + f)
        for _, r in c.iterrows():
            S.append((g, r.delta, r.N, 'pin', r.pin, r.pin_its, bool(r.pin_conv)))
            S.append((g, r.delta, r.N, 'IVP', r.ivp, r.ivp_its, bool(r.ivp_conv)))
    c = pd.read_csv(R + 'colloc_q2_unmapped.csv')
    S += [('unmapped', r.delta, r.N, 'pin', r.err, r.its, bool(r.conv)) for _, r in c.iterrows()]
    c = pd.read_csv(R + 'unknowns_q2.csv')
    S += [('variables ' + r['var'], r.delta, r.N, 'pin' if r.form == 'pinned' else 'IVP', r.error, r.its, bool(r.conv)) for _, r in c.iterrows()]
    c = pd.read_csv(R + 'arrhenius_runs.csv')
    S += [('arrhenius', r.beta, r.N, r.form, r.err, r.its, bool(r.conv)) for _, r in c.iterrows()]
    c = pd.read_csv(R + 'ablation.csv')
    S += [(f'ablation {r.mapping} {r.form}', r.delta, 512, r.cond, r.err, r.its, bool(r.conv)) for _, r in c.iterrows()]
    c = pd.read_csv(R + 'robustness.csv')
    S = [x + (0.0,) for x in S]
    S += [('robustness ' + r.test, r.delta, r.N, 'pin', r.err, r.its, bool(r.conv), r.value) for _, r in c.iterrows()]
    S = pd.DataFrame(S, columns=['group', 'delta', 'N', 'form', 'err_saved', 'its_saved', 'conv_saved', 'value'])
    S['N'] = S.N.astype(int)
    return S


if __name__ == '__main__':
    jobs = [(g_q, 2), (g_q, 3), (g_unmapped,), (g_variables,), (g_arrhenius,), (g_review2,)]
    import sys
    if '--summary' in sys.argv:                            # only recompute the summary from residuals_all.csv
        df = pd.read_csv(R + 'residuals_all.csv').drop(columns=['err_saved', 'its_saved', 'conv_saved', 'res'])
        df['value'] = df['value'].fillna(0.0)
    else:
        with ProcessPoolExecutor(2) as ex:
            futs = [ex.submit(*j) for j in jobs]
            rows = sum((f.result() for f in futs), [])
        df = pd.DataFrame(rows)
    df['value'] = df['value'].fillna(0.0)
    sv = saved()
    key = ['group', 'delta', 'N', 'form', 'value']
    df = df.merge(sv, on=key, how='left', validate='one_to_one')
    # residual in the scaling of the differentiation form, R_j = mu (D_x u)_j - F(u_j): for the integral form the
    # collocation rows are those of the differentiation form applied to v; for the unknown 1/y relative to max|u|
    df['res'] = df.res_inf
    m = df.res_nodal.notna() if 'res_nodal' in df else np.zeros(len(df), bool)
    df.loc[m, 'res'] = df.loc[m, 'res_nodal']
    m = (df.group == 'variables 1/y').values
    df.loc[m, 'res'] = df.loc[m, 'res_inf'] / np.maximum(1.0, df.loc[m, 'umax'])
    df.to_csv(R + 'residuals_all.csv', index=False)
    same_conv = (df.conv == df.conv_saved).all()
    same_its = (df.its == df.its_saved).all()
    ok = df.err.notna() & df.err_saved.notna()
    rel = (abs(df.err - df.err_saved) / df.err_saved)[ok]
    T = [f'runs: {len(df)}; converged: {int(df.conv.sum())}; same convergence flags as saved: {same_conv}; same iteration counts: {same_its}; '
         f'max relative change of the error: {rel.max():.1e}']
    cv = df[df.conv]
    lastfb = cv[cv.lam_last < 2e-3]
    T.append(f'converged runs: max final residual {cv.res.max():.1e} (differentiation scaling, relative for 1/y); '
             f'last damping factor 1 in {int((cv.lam_last == 1).sum())} of {len(cv)}; '
             f'last step = smallest step (no sufficient decrease): {len(lastfb)} runs, of which pinned {int((lastfb.form == "pin").sum())}, '
             f'max residual of these {lastfb.res.max():.1e}')
    T.append('  by group (last step smallest): ' + ', '.join(f'{g} {f}: {len(h)}' for (g, f), h in lastfb.groupby(['group', 'form'])))
    for g, h in cv.groupby('group'):
        T.append(f'  {g}: {len(h)} converged, max residual {h.res.max():.1e}, min last lambda {h.lam_last.min():g}, fallbacks {int(h.n_fallback.sum())}')
    pinr = cv[(cv.form == 'pin') & ~cv.group.str.startswith('unmapped') & ~cv.group.str.contains('none')]
    T.append(f'pinned runs on the sinh grid (all groups): {len(pinr)} converged, last damping factor >= {pinr.lam_last.min():g}, '
             f'smallest-step fallbacks {int(pinr.n_fallback.sum())}, max residual {pinr.res.max():.1e}')
    nc = df[~df.conv]
    T.append(f'not converged: {len(nc)} runs, final residual {nc.res.min():.1e}..{nc.res.max():.1e}')
    open(R + 'residuals_summary.txt', 'w').write('\n'.join(T) + '\n')
    print('\n'.join(T))
