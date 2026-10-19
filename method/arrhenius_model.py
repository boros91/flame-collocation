"""Sect. 5.6: Arrhenius (Zeldovich--Frank-Kamenetskii) rate
    f(y) = (1-y) exp(-beta (1-y) / (1 - alpha (1-y))),   y(0) = 0.
No closed-form time function for alpha > 0.  Everything is computed numerically:
  * Theta(y) - Theta(y_f) by composite Gauss-Legendre quadrature of 1/f - 1/(1-y), plus -ln(1-y);
  * the singularity s* by complex contour quadrature (mpmath), ending at the essential singularity
    y_e = 1 - 1/alpha of f, approached from the left;
  * the pin y_f = argmax f (Corollary 4.2 with the unknown y).
Variables: s = Theta(y) - Theta(y_f) (pin at s = 0), T = -s(y=0), mu = 1/T, x = mu*s, so x* = 0."""
import numpy as np, mpmath as mp, sys
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from collocation import cheb, bary_row, bary_eval

class Arr:
    def __init__(s, beta, alpha, npan=4000, ng=14):
        s.b, s.a = beta, alpha
        # pin = argmax f
        r = minimize_scalar(lambda y: -s.f(y), bounds=(0.0, 1.0), method='bounded', options=dict(xatol=1e-15))
        s.yf = r.x; s.uf = 1 - s.yf
        # panels in u on [0,1], cumulative integral of g from u_f
        s.ue = np.linspace(0, 1, npan + 1); s.h = 1.0 / npan
        s.xg, s.wg = leggauss(ng)
        a, b = s.ue[:-1], s.ue[1:]
        nodes = (a + b)[:, None] / 2 + (b - a)[:, None] / 2 * s.xg
        pan = (s.g(nodes) * s.wg).sum(1) * s.h / 2            # int over each panel
        C = np.concatenate([[0], np.cumsum(pan)])               # int_0^{u_e}
        # anchor at u_f: Gc(u_e) = int_{u_e}^{u_f} g
        kf = min(int(s.uf / s.h), npan - 1)
        Gf = C[kf] + s._partial(np.array([kf]), np.array([s.uf]))[0]   # int_0^{u_f}
        s.C, s.Gf = C, Gf
        # recompute cumulative relative to u_f to avoid cancellation: sums outward from kf
        s.Crel = np.empty_like(C)
        s.Crel[kf] = C[kf] - Gf  # = -int_{u_kf}^{u_f}
        s.Crel[kf] = -s._partial_ab(np.array([s.ue[kf]]), np.array([s.uf]))[0]
        for k in range(kf + 1, npan + 1): s.Crel[k] = s.Crel[k - 1] + pan[k - 1]
        for k in range(kf - 1, -1, -1): s.Crel[k] = s.Crel[k + 1] - pan[k]
        s.s0 = s.S_of_u(np.array([1.0]))[0]        # s at y = 0 (relative to pin)
        s.T = -s.s0; s.mu = 1 / s.T

    def f(s, y):
        u = 1 - y; return u * np.exp(-s.b * u / (1 - s.a * u))
    def fp(s, y):
        u = 1 - y; E = np.exp(-s.b * u / (1 - s.a * u))
        return -E * (1 - s.b * u / (1 - s.a * u) ** 2)
    def g(s, u):   # 1/f - 1/u, smooth
        z = s.b * u / (1 - s.a * u)
        return np.where(u > 1e-8, np.expm1(z) / np.where(u > 1e-8, u, 1), s.b + 0 * u)
    def _partial_ab(s, a, b):
        nodes = (a + b)[:, None] / 2 + (b - a)[:, None] / 2 * s.xg
        return (s.g(nodes) * s.wg).sum(1) * (b - a) / 2
    def _partial(s, k, u):
        return s._partial_ab(s.ue[k], u)
    def Gint(s, u):    # int_{u}^{u_f} g = -(int_{u_f}^{u})
        k = np.minimum((u / s.h).astype(int), len(s.ue) - 2)
        return -(s.Crel[k] + s._partial(k, u))
    def S_of_u(s, u, l=None):  # s = Theta(y) - Theta(y_f), y = 1-u
        if l is None: l = np.log(u)
        return -(l - np.log(s.uf)) + s.Gint(u)
    def Y(s, S):
        S = np.asarray(S, float)
        # initial guess from panel table
        ue = s.ue[1:]; Se = s.S_of_u(ue)               # decreasing in u
        l = np.interp(-S, -Se[::-1], np.log(ue[::-1]))
        l = np.where(S > Se[0], np.log(ue[0]) - (S - Se[0]), l)  # beyond first panel: ~ -ln u
        for _ in range(60):
            l = np.minimum(l, 0.0); u = np.exp(l); F = s.S_of_u(u, l) - S
            dl = F / (-np.exp(s.b * u / (1 - s.a * u)))
            l = l - dl
            if np.max(np.abs(dl)) < 1e-15: break
        return 1 - np.exp(l)

    def singularity(s):
        mp.mp.dps = 30
        b, a = mp.mpf(s.b), mp.mpf(s.a)
        finv = lambda z: mp.exp(b * (1 - z) / (1 - a * (1 - z))) / (1 - z)
        yf = mp.mpf(s.yf)
        if s.a == 0:
            val = mp.quad(finv, [yf, yf + 2j]) + mp.quad(lambda r: finv(yf + 2j + r), [0, mp.inf])
        else:
            ye = 1 - 1 / a; H = 40
            val = mp.quad(finv, [yf, yf + 1j * H, ye - 1 + 1j * H, ye - 1, ye - mp.mpf('1e-15')])
        return complex(val)

def solve(P, N, sst, pin=True, tol=1e-12, rtol=1e-10, maxit=80, guess=None, return_sol=False):
    mu = P.mu
    D, xi = cheb(N)
    xc, eps = mu * sst.real, mu * sst.imag
    al = np.arcsinh((-1 - xc) / eps); be = np.arcsinh((1 - xc) / eps)
    L, M = (be - al) / 2, (be + al) / 2
    gp = eps * L * np.cosh(L * xi + M)
    Sn = sst.real + sst.imag * np.sinh(L * xi + M)       # s at nodes (pin at s=0)
    Dx = D / gp[:, None]
    xi_pin = (np.arcsinh((0 - sst.real) / sst.imag) - M) / L
    prow = bary_row(xi, xi_pin)
    if guess is None:
        # generic q = 2 profile of Sect. 4, equal to y_f at the pin s = 0
        s0 = np.log(np.expm1(1 / P.yf - 1)); z = s0 - Sn
        sp = np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))
        y = 1 / (1 + sp)
    else:
        y = guess(Sn)
    def res(y):
        R = mu * Dx @ y - P.f(y)
        R[-1] = (prow @ y - P.yf) if pin else y[-1]
        return R
    R = res(y); conv = False; nfb = 0; lam = 1.0
    for it in range(1, maxit + 1):
        J = mu * Dx - np.diag(P.fp(y))
        if pin: J[-1, :] = prow
        else: J[-1, :] = 0; J[-1, -1] = 1
        dy = np.linalg.solve(J, -R)
        if np.max(np.abs(dy)) < tol:          # Newton update below tol: accept it and stop if the residual is small
            yt = y + dy; Rt = res(yt)
            if np.max(np.abs(Rt)) < rtol:
                y, R = yt, Rt; conv = True; lam = 1.0; break
        lam, r0 = 1.0, np.linalg.norm(R)
        while True:
            yn = y + lam * dy; Rn = res(yn)
            if np.linalg.norm(Rn) < (1 - 1e-4 * lam) * r0: break
            if lam < 1e-3: nfb += 1; break    # no sufficient decrease: take the smallest step
            lam /= 2
        y, R = yn, Rn
    xe = np.cos(np.pi * (np.arange(4000) + 0.5) / 4000)
    Se = sst.real + sst.imag * np.sinh(L * xe + M)
    Yex = P.Y(Se)
    err = np.max(np.abs(bary_eval(xi, y, xe) - Yex))
    interp = np.max(np.abs(bary_eval(xi, P.Y(Sn), xe) - Yex))
    out = dict(N=N, form='pin' if pin else 'IVP', err=err, interp=interp, its=it, conv=conv,
               y0=y[-1], L=L, res_inf=float(np.max(np.abs(R))), lam_last=lam, n_fallback=nfb)
    if return_sol: out['sol'] = (xi, y, L, M)
    return out

