"""Additional experiments (q = 2 unless stated), all with v = ln y as unknown:
(A) ablation: unmapped / sinh-mapped grid  x  initial condition / pin  x  differentiation (row replacement)
    / integral (Birkhoff, pseudospectral integration matrix) formulation, at N = 512;
(B) robustness: pin at a non-optimal value y_p, and sinh map with perturbed centre or width;
(C) final nodal residuals of all runs of (A), and the run time of a pinned solve.
The differentiation form reproduces collocation_q.solve exactly (same guess, damping, stopping rule).
Writes results/ablation.csv, results/robustness.csv, results/timing.txt."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import time
import numpy as np, pandas as pd
from numpy.polynomial import chebyshev as C
from collocation import cheb, bary_row, bary_eval
from collocation_q import params, initial_guess
from flame_q import Theta, master
R = ROOT + '/results/'


def grid(delta, N, q, mapping='sinh', c_shift=0.0, eps_fac=1.0):
    """Nodes in xi (Chebyshev-Lobatto, from +1 to -1), g'(xi), and the master variable s(xi)."""
    mu, x0 = params(delta, q)
    D, xi = cheb(N)
    if mapping == 'sinh':
        c = x0 + c_shift * np.pi * mu; eps = eps_fac * np.pi * mu
        al = np.arcsinh((-1 - c) / eps); be = np.arcsinh((1 - c) / eps)
        L, M = (be - al) / 2, (be + al) / 2
        gp = lambda t: eps * L * np.cosh(L * t + M)
        sv = lambda t: (c - x0) / mu + (eps / mu) * np.sinh(L * t + M)
        to_xi = lambda s: (np.arcsinh((s - (c - x0) / mu) / (eps / mu)) - M) / L
    else:
        gp = lambda t: np.ones_like(t)
        sv = lambda t: (t - x0) / mu
        to_xi = lambda s: x0 + mu * s
    return dict(mu=mu, x0=x0, D=D, xi=xi, gp=gp, sv=sv, to_xi=to_xi)


def birkhoff(xi):
    """B[i, j] = int_{-1}^{xi_i} l_j, l_j the Lagrange basis on xi_0..xi_{N-1} (degree N-1)."""
    N = len(xi) - 1
    V = C.chebvander(xi[:N], N - 1)                      # values of T_0..T_{N-1}
    Vi = np.linalg.inv(V)                                  # coefficients of the l_j
    E = C.chebvander(xi, N)                                # evaluation of degree-N polynomials
    Ik = np.zeros((N + 1, N))                              # coefficients of int_{-1} T_k
    for k in range(N):
        e = np.zeros(N); e[k] = 1
        Ik[:, k] = C.chebint(e, lbnd=-1)
    return E @ Ik @ Vi


def solve(delta, N, q, pin=True, mapping='sinh', form='diff', y_pin=None, c_shift=0.0, eps_fac=1.0,
          tol=1e-12, rtol=1e-10, maxit=80):
    G = grid(delta, N, q, mapping, c_shift, eps_fac)
    mu, xi, D, gp = G['mu'], G['xi'], G['D'], G['gp'](G['xi'])
    ystar = (q - 1) / q
    yp = ystar if y_pin is None else y_pin
    s_nodes = G['sv'](xi)
    y = initial_guess(s_nodes, Theta(ystar, q), q); y[-1] = delta
    v = np.log(y)
    F = lambda u: np.exp((q - 1) * u) - np.exp(q * u)
    dF = lambda u: (q - 1) * np.exp((q - 1) * u) - q * np.exp(q * u)
    prow = bary_row(xi, G['to_xi'](Theta(yp, q))); pval = np.log(yp)
    if form == 'diff':
        Dx = D / gp[:, None]
        z = v.copy()
        def vof(z): return z
        def res(z):
            R_ = mu * Dx @ z - F(z)
            R_[-1] = (prow @ z - pval) if pin else (z[-1] - np.log(delta))
            return R_
        def jac(z):
            J = mu * Dx - np.diag(dF(z))
            if pin: J[-1, :] = prow
            else: J[-1, :] = 0; J[-1, -1] = 1
            return J
    else:
        B = birkhoff(xi); one = np.ones(N + 1)
        z = np.concatenate([(D @ v)[:N], [v[-1]]])        # unknowns: dv/dxi at xi_0..xi_{N-1}, and v(-1)
        def vof(z): return z[N] * one + B @ z[:N]
        def res(z):
            vv = vof(z)
            R_ = np.empty(N + 1)
            R_[:N] = z[:N] - gp[:N] / mu * F(vv[:N])
            R_[N] = (prow @ vv - pval) if pin else (z[N] - np.log(delta))
            return R_
        def jac(z):
            vv = vof(z); a = gp[:N] / mu * dF(vv[:N])
            J = np.zeros((N + 1, N + 1))
            J[:N, :N] = np.eye(N) - a[:, None] * B[:N, :]
            J[:N, N] = -a
            if pin: J[N, :N] = prow @ B; J[N, N] = prow @ one
            else: J[N, N] = 1
            return J
    Dxd = D / gp[:, None]
    def res_diff(z):                                   # residual in the scaling of the differentiation form
        vv = vof(z); Rd = mu * Dxd @ vv - F(vv)
        Rd[-1] = (prow @ vv - pval) if pin else (vv[-1] - np.log(delta))
        return np.max(np.abs(Rd))
    R_ = res(z); conv = False; nfb = 0
    for it in range(maxit):
        J = jac(z)
        dz = np.linalg.solve(J, -R_)
        if np.max(np.abs(vof(z + dz) - vof(z))) < tol:     # full Newton correction of v small: take it,
            zt = z + dz                                    # and check the residual (differentiation scaling)
            if res_diff(zt) < rtol:
                z, R_, lam, conv = zt, res(zt), 1.0, True
                break
        lam, r0 = 1.0, np.linalg.norm(R_)
        while True:
            zn = z + lam * dz; Rn = res(zn)
            if np.linalg.norm(Rn) < (1 - 1e-4 * lam) * r0:
                break
            if lam < 1e-3:
                nfb += 1; break                            # no sufficient decrease: take the smallest step
            lam /= 2
        if not np.all(np.isfinite(Rn)):
            break                                          # overflow: Newton failed
        z, R_ = zn, Rn
    vv = vof(z)
    xe = np.cos(np.pi * (np.arange(4000) + 0.5) / 4000)
    err = np.max(np.abs(bary_eval(xi, np.exp(vv), xe) - master(G['sv'](xe), q)))
    interp = np.max(np.abs(bary_eval(xi, master(s_nodes, q), xe) - master(G['sv'](xe), q)))
    Rd = mu * (D / gp[:, None]) @ vv - F(vv)           # nodal residual of the collocation equations
    J = jac(z); Js = J / np.abs(J).sum(1)[:, None]
    if not (np.all(np.isfinite(Js)) and np.isfinite(err)):
        return dict(err=np.nan, interp=interp, its=it + 1, conv=False, res_nodal=np.nan, cond_rowscaled=np.nan, y0_rel=np.nan,
                    res_inf=np.nan, lam_last=lam, n_fallback=nfb)
    return dict(err=err, interp=interp, its=it + 1, conv=conv, res_nodal=np.max(np.abs(Rd[:N])),
                cond_rowscaled=np.linalg.cond(Js), y0_rel=abs(np.exp(vv[-1]) / delta - 1),
                res_inf=float(np.max(np.abs(R_))), lam_last=lam, n_fallback=nfb)


if __name__ == '__main__':
    q = 2
    # check: identical to collocation_q.solve
    from collocation_q import solve as solve_q, error as error_q
    for d, N in [(1e-6, 224), (1e-10, 352)]:
        for pin in (True, False):
            a = solve_q(d, N, q, pin); b = solve(d, N, q, pin)
            print('check', d, N, pin, error_q(a), b['err'], a['its'], b['its'])
    # (A) ablation at N = 512
    rows = []
    for d in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        for mapping in ('none', 'sinh'):
            for form in ('diff', 'int'):
                for pin in (False, True):
                    r = solve(d, 512, q, pin, mapping, form)
                    r.update(delta=d, N=512, mapping=mapping, form=form, cond='pin' if pin else 'IVP')
                    rows.append(r); print({k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(R + 'ablation.csv', index=False)
    # (B) robustness at N = N_min of Table 1
    c2 = pd.read_csv(R + 'colloc_q2.csv')
    rows = []
    Q = lambda y: y**(q - 1) * (1 - y)
    for d in [1e-6, 1e-10]:
        h = c2[np.isclose(c2.delta, d, rtol=1e-9, atol=0)]; N = int(h[h.pin <= 1e-10].N.min())
        for yp in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            r = solve(d, N, q, True, y_pin=yp); r.update(delta=d, N=N, test='pin value', value=yp,
                                                     bound=Q(0.5) / Q(yp)); rows.append(r)
        for cs in [-5, -2, 2, 5]:
            r = solve(d, N, q, True, c_shift=cs); r.update(delta=d, N=N, test='centre shift / (pi mu)', value=cs); rows.append(r)
        for ef in [0.25, 0.5, 2, 4]:
            r = solve(d, N, q, True, eps_fac=ef); r.update(delta=d, N=N, test='width factor', value=ef); rows.append(r)
        for r in rows[-15:]:
            print({k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(R + 'robustness.csv', index=False)
    # (C) timing of a pinned solve (differentiation form, sinh map)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); solve_q(1e-10, 384, q, True); ts.append(time.perf_counter() - t0)
    import platform, sys, scipy
    try:
        from threadpoolctl import threadpool_info
        blas = '; '.join(f"{b['internal_api']} {b['version']} ({b['num_threads']} threads)" for b in threadpool_info() if b['user_api'] == 'blas')
    except ImportError:
        blas = 'unknown'
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor()) \
        if platform.system() == 'Linux' else platform.processor()
    open(R + 'timing.txt', 'w').write(
        f'pinned solve q=2 delta=1e-10 N=384: median {np.median(ts):.3f} s over 5 runs\n'
        f'environment: {cpu}; {platform.platform()}; Python {sys.version.split()[0]}, NumPy {np.__version__}, '
        f'SciPy {scipy.__version__}; BLAS {blas}\n')
    print(open(R + 'timing.txt').read())
