"""High-precision (256-bit) version of the collocation experiment of the paper.
Sinh-mapped Chebyshev collocation for mu*v' = e^{(q-1)v} - e^{qv}, pinned and IVP forms.
Transcendental functions: mpmath; linear solves: FLINT (python-flint arb_mat, approximate LU).
Usage: python exp_highprec.py q   (q = 2 or 3).  Output: results/highprec_colloc_q{q}.csv"""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import sys, time
import numpy as np, pandas as pd
import mpmath as mp
from flint import arb_mat, arb, ctx
from flame_q import master as master_double

PREC = 256
mp.mp.prec = PREC
ctx.prec = PREC


def Theta(y, q):
    out = mp.log(y) - mp.log1p(-y)
    for j in range(1, q):
        out -= 1 / (j * y**j)
    return out


def theta_logit(l, q):
    y = 1 / (1 + mp.exp(-l))
    out = l
    for j in range(1, q):
        out -= 1 / (j * y**j)
    return out


def master(s, q):
    """Y_q(s) to full working precision: Newton in the logit variable from a double start."""
    y0 = float(master_double(np.array([float(s)]), q)[0])
    y0 = min(max(y0, 1e-300), 1 - 1e-16)
    l = mp.log(mp.mpf(y0)) - mp.log1p(-mp.mpf(y0))
    for _ in range(100):
        y = 1 / (1 + mp.exp(-l))
        dl = (theta_logit(l, q) - s) * y**(q - 1)
        l -= dl
        if abs(dl) < mp.mpf(2)**(-PREC + 8) * (1 + abs(l)):
            break
    return 1 / (1 + mp.exp(-l))


def cheb(N):
    x = [mp.cos(mp.pi * j / N) for j in range(N + 1)]
    c = [(2 if j in (0, N) else 1) * (-1)**j for j in range(N + 1)]
    D = [[mp.mpf(0)] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i][j] = mp.mpf(c[i]) / c[j] / (x[i] - x[j])
        D[i][i] = -mp.fsum(D[i][j] for j in range(N + 1) if j != i)
    return D, x


def bary_weights(N):
    w = [mp.mpf((-1)**j) for j in range(N + 1)]
    w[0] /= 2; w[N] /= 2
    return w


def bary_row(x, w, t):
    d = [w[j] / (t - x[j]) for j in range(len(x))]
    S = mp.fsum(d)
    return [dj / S for dj in d]


def bary_eval(x, w, f, ts):
    out = []
    for t in ts:
        num = mp.mpf(0); den = mp.mpf(0); hit = None
        for j in range(len(x)):
            dt = t - x[j]
            if dt == 0:
                hit = f[j]; break
            a = w[j] / dt; num += a * f[j]; den += a
        out.append(hit if hit is not None else num / den)
    return out


def setup(delta, q):
    delta = mp.mpf(delta)
    mu = (q - 1) * delta**(q - 1)
    x0 = -mu * Theta(delta, q) - 1
    eps = mp.pi * mu
    sm = mp.asinh((-1 - x0) / eps); sp = mp.asinh((1 - x0) / eps)
    L, M = (sp - sm) / 2, (sp + sm) / 2
    return dict(delta=delta, mu=mu, x0=x0, eps=eps, L=L, M=M)


def to_arb(A):
    n = len(A); m = len(A[0])
    return arb_mat(n, m, [arb(str(A[i][j])) if False else arb(mp.nstr(A[i][j], PREC // 3, strip_zeros=False))
                          for i in range(n) for j in range(m)])


def to_arb_fast(A):
    # exact binary conversion: mpf -> (man, exp) -> arb
    n = len(A); m = len(A[0])
    vals = []
    for i in range(n):
        for j in range(m):
            v = A[i][j]
            sign, man, exp, _ = mp.mpf(v)._mpf_
            vals.append(((-1)**sign) * arb(man) * arb(2)**exp if man != 0 else arb(0))
    return arb_mat(n, m, vals)


def from_arb_vec(X, n):
    out = []
    for i in range(n):
        mid = X[i, 0].mid()
        out.append(mp.mpf(mid.str(PREC // 3 + 5, radius=False)))
    return out


def solve(delta, N, q, pin, maxit=60, v0=None):
    S = setup(delta, q)
    D, xi = cheb(N)
    w = bary_weights(N)
    gp = [S['eps'] * S['L'] * mp.cosh(S['L'] * t + S['M']) for t in xi]
    svar = [mp.pi * mp.sinh(S['L'] * t + S['M']) for t in xi]
    ystar = mp.mpf(q - 1) / q
    s_pin = Theta(ystar, q)
    xi_pin = (mp.asinh(s_pin / mp.pi) - S['M']) / S['L']
    prow = bary_row(xi, w, xi_pin)
    # initial guess: the converged double-precision solution would do; use the generic profile
    spv = mp.mpf(1) / (q - 1)**q
    s0 = s_pin + mp.log(mp.expm1(spv))
    v = []
    for s in svar:
        z = s0 - s
        sp_ = z if z > 50 else mp.log1p(mp.exp(z))
        y = 1 / (1 + ((q - 1) * sp_)**(mp.mpf(1) / (q - 1)))
        v.append(mp.log(y))
    if v0 is not None:
        v = list(v0)
    if not pin:
        v[-1] = mp.log(S['delta'])
    mu = S['mu']
    Dx = [[D[i][j] / gp[i] for j in range(N + 1)] for i in range(N + 1)]
    tol = mp.mpf(2)**(-PREC + 20)                       # full Newton correction below 2^-236 ...
    rtol = mp.mpf(2)**(-PREC + 30)                      # ... and then the residual below 2^-226

    def resid(v):
        R = []
        for i in range(N + 1):
            R.append(mu * mp.fsum(Dx[i][j] * v[j] for j in range(N + 1)) - (mp.exp((q - 1) * v[i]) - mp.exp(q * v[i])))
        R[-1] = (mp.fsum(prow[j] * v[j] for j in range(N + 1)) - mp.log(ystar)) if pin else (v[-1] - mp.log(S['delta']))
        return R

    R = resid(v)
    conv = False
    for it in range(1, maxit + 1):
        J = [[mu * Dx[i][j] for j in range(N + 1)] for i in range(N + 1)]
        for i in range(N + 1):
            J[i][i] -= (q - 1) * mp.exp((q - 1) * v[i]) - q * mp.exp(q * v[i])
        J[-1] = prow[:] if pin else [mp.mpf(0)] * N + [mp.mpf(1)]
        Ja = to_arb_fast(J); Ra = to_arb_fast([[-r] for r in R])
        dv = from_arb_vec(Ja.solve(Ra, algorithm='approx'), N + 1)
        if max(abs(d) for d in dv) < tol:
            vt = [v[i] + dv[i] for i in range(N + 1)]
            Rt = resid(vt)
            if max(abs(r) for r in Rt) < rtol:
                v, R, conv = vt, Rt, True
                break
        r0 = mp.sqrt(mp.fsum(r * r for r in R))
        lam = mp.mpf(1)
        while True:
            vn = [v[i] + lam * dv[i] for i in range(N + 1)]
            Rn = resid(vn)
            if mp.sqrt(mp.fsum(r * r for r in Rn)) < (1 - lam / 10000) * r0 or lam < mp.mpf('1e-3'):
                break
            lam /= 2
        v, R = vn, Rn
        if VERBOSE: print(it, mp.nstr(lam, 3), mp.nstr(max(abs(d) for d in dv), 3), mp.nstr(r0, 3))
    y = [mp.exp(t) for t in v]
    return dict(xi=xi, w=w, y=y, its=it, conv=conv, S=S, y_left=y[-1])


def error(sol, q, M=600):
    S = sol['S']
    ts = [mp.cos(mp.pi * (k + mp.mpf(1) / 2) / M) for k in range(M)]
    key = (float(S['delta']), q, M)
    if key not in EXACT:
        EXACT[key] = [master(mp.pi * mp.sinh(S['L'] * t + S['M']), q) for t in ts]
    p = bary_eval(sol['xi'], sol['w'], sol['y'], ts)
    return max(abs(a - b) for a, b in zip(p, EXACT[key]))


def interp_error(delta, N, q, M=600):
    S = setup(delta, q)
    _, xi = cheb(N); w = bary_weights(N)
    y = [master(mp.pi * mp.sinh(S['L'] * t + S['M']), q) for t in xi]
    return error(dict(xi=xi, w=w, y=y, S=S), q, M)


EXACT = {}
VERBOSE = False

if __name__ == '__main__':
    q = int(sys.argv[1])
    deltas = {2: [1e-2, 1e-4, 1e-6], 3: [1e-1, 1e-2, 1e-3]}[q]
    Ns = [64, 128, 192, 256, 320, 384, 448, 512]
    rows = []
    for d in deltas:
        for N in Ns:
            ie = interp_error(d, N, q)
            for pin in (True, False):
                t0 = time.time()
                sol = solve(d, N, q, pin, maxit=40)
                e = error(sol, q)
                rows.append(dict(q=q, delta=d, N=N, form='pinned' if pin else 'IVP', error=float(e),
                                 interp=float(ie), its=sol['its'], conv=sol['conv'],
                                 y0_rel=float(abs(sol['y_left'] / mp.mpf(d) - 1)), L=float(sol['S']['L']),
                                 secs=round(time.time() - t0, 1)))
                print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(ROOT + f'/results/highprec_colloc_q{q}.csv', index=False)
