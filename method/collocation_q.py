"""Sinh-mapped Chebyshev collocation for the generalised flame family
    mu * y'(x) = y^q (1 - y),  x in [-1, 1],  y(-1) = delta,   mu = (q-1) delta^{q-1}
(x = mu*t - 1, t in [0, 2/mu]).  With s = t + Theta_q(delta) the solution is the master
profile Y_q(s); its nearest complex singularities are at s = +- i*pi, i.e. at
x0 +- i*pi*mu with x0 = -mu*Theta_q(delta) - 1.  We solve for v = ln y:
    mu v' = e^{(q-1)v} - e^{qv}.
IVP form: v(-1) = ln delta.  Pinned form: v(x*) = ln y*, y* = (q-1)/q (maximiser of
f(y)/y), x* = x0 + mu*Theta_q(y*)."""
import numpy as np
from flame_q import Theta, master
from collocation import cheb, bary_row, bary_eval


def params(delta, q):
    mu = (q - 1) * delta**(q - 1)
    x0p1 = -mu * Theta(delta, q)           # x0 + 1
    return mu, x0p1 - 1.0


def setup(delta, N, q):
    mu, x0 = params(delta, q)
    eps = np.pi * mu
    D, xi = cheb(N)
    al = np.arcsinh((-1 - x0) / eps); be = np.arcsinh((1 - x0) / eps)
    L, M = (be - al) / 2, (be + al) / 2
    gp = lambda s: eps * L * np.cosh(L * s + M)
    svar = lambda s: np.pi * np.sinh(L * s + M)     # s - 0 = (x - x0)/mu, exact
    Dx = D / gp(xi)[:, None]
    return dict(xi=xi, Dx=Dx, svar=svar, mu=mu, x0=x0, L=L, M=M, eps=eps)


def initial_guess(s, s_pin, q):
    """Generic shape 1/(1 + ((q-1)*softplus(s0 - s))^{1/(q-1)}): reproduces the algebraic
    left tail y ~ (-(q-1)s)^{-1/(q-1)} and saturates at 1; s0 is chosen so that the guess
    equals y* = (q-1)/q at the pin point."""
    spv = (1.0 / (q - 1))**(q - 1) / (q - 1)
    s0 = s_pin + np.log(np.expm1(spv))
    z = s0 - s
    sp = np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))
    return 1.0 / (1.0 + ((q - 1) * sp)**(1.0 / (q - 1)))


def solve(delta, N, q, pin=True, tol=1e-12, rtol=1e-10, maxit=80, test='full'):
    """Damped Newton (step 6 of the algorithm in the paper).  test='full' (paper): converged when the
    full correction max|du| < tol and, after taking it, max|R| < rtol; failure after maxit iterations.
    test='damped': converged when the damped update max|lam*du| < tol and max|R| < rtol."""
    S = setup(delta, N, q)
    xi, Dx, mu = S['xi'], S['Dx'], S['mu']
    ystar = (q - 1) / q
    s_pin = Theta(ystar, q)
    s_nodes = S['svar'](xi)
    y = initial_guess(s_nodes, s_pin, q)
    y[-1] = delta
    u = np.log(y)
    F = lambda u: np.exp((q - 1) * u) - np.exp(q * u)
    dF = lambda u: (q - 1) * np.exp((q - 1) * u) - q * np.exp(q * u)
    if pin:
        xi_pin = (np.arcsinh(s_pin / np.pi) - S['M']) / S['L']
        prow = bary_row(xi, xi_pin); pval = np.log(ystar)

    def res(u):
        R = mu * Dx @ u - F(u)
        R[-1] = (prow @ u - pval) if pin else (u[-1] - np.log(delta))
        return R
    R = res(u); nfb = 0
    for it in range(maxit):
        J = mu * Dx - np.diag(dF(u))
        if pin:
            J[-1, :] = prow
        else:
            J[-1, :] = 0; J[-1, -1] = 1
        du = np.linalg.solve(J, -R)
        if test == 'full' and np.max(np.abs(du)) < tol:
            ut = u + du; Rt = res(ut)
            if np.max(np.abs(Rt)) < rtol:
                u, R, lam, ok = ut, Rt, 1.0, True
                break
        lam, r0 = 1.0, np.linalg.norm(R)
        while True:
            un = u + lam * du; Rn = res(un)
            if np.linalg.norm(Rn) < (1 - 1e-4 * lam) * r0:
                break
            if lam < 1e-3:
                nfb += 1; break                            # no sufficient decrease: take the smallest step
            lam /= 2
        u, R = un, Rn
        step = np.max(np.abs(lam * du))
        ok = test == 'damped' and step < tol and np.max(np.abs(R)) < rtol    # small update and small residual
        if ok:
            break
    return dict(xi=xi, y=np.exp(u), its=it + 1, converged=bool(ok), svar=S['svar'],
                L=S['L'], q=q, cond=np.linalg.cond(J),
                res_inf=float(np.max(np.abs(R))), lam_last=lam, n_fallback=nfb)


def error(sol, M=4000):
    xi_e = np.cos(np.pi * (np.arange(M) + 0.5) / M)
    p = bary_eval(sol['xi'], sol['y'], xi_e)
    return np.max(np.abs(p - master(sol['svar'](xi_e), sol['q'])))


def interp_error(delta, N, q, M=4000):
    S = setup(delta, N, q)
    y = master(S['svar'](S['xi']), q)
    return error(dict(xi=S['xi'], y=y, svar=S['svar'], q=q), M)
