"""Chebyshev and barycentric utilities (cheb, bary_row, bary_eval), and the q = 2 sinh-map
setup used for the experiment with a map centred at the midpoint (exp_center0.py).
Rescaled flame problem  delta * y'(x) = y^2 - y^3,  x in [-1, 1],  y(-1) = delta  (x = delta*t - 1).
Exact: y(x) = Y((x - x_half)/delta), x_half = delta*c(delta) - 1.
Nearest complex singularities of y:  x = delta*ln(a) +- i*pi*delta,  a = 1/delta - 1.
"""
import numpy as np
from flame import shift, master


def cheb(N):
    """Chebyshev differentiation matrix and Gauss-Lobatto points (Trefethen),
    ordered from +1 to -1."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def x_half(delta):
    return delta * shift(delta) - 1.0


def singularity(delta):
    """Nearest complex singularities of y(x): x = delta*ln(a) +- i*pi*delta."""
    a = 1 / delta - 1
    return delta * np.log(a), np.pi * delta


def setup(delta, N, mapping, x0=None, eps=None):
    """Nodes, differentiation matrix in x, and tau(xi) = (x - x_half)/delta computed
    without cancellation (needed to evaluate the exact solution accurately)."""
    D, xi = cheb(N)
    xh = x_half(delta)
    if mapping == 'sinh':
        al = np.arcsinh((-1 - x0) / eps); be = np.arcsinh((1 - x0) / eps)
        A, B = (be - al) / 2, (be + al) / 2
        g = lambda s: x0 + eps * np.sinh(A * s + B)
        gp = lambda s: eps * A * np.cosh(A * s + B)
        tau = lambda s: (x0 - xh) / delta + (eps / delta) * np.sinh(A * s + B)
    else:
        g = lambda s: s
        gp = lambda s: np.ones_like(s)
        tau = lambda s: (s - xh) / delta
    Dx = D / gp(xi)[:, None]
    return xi, g(xi), Dx, tau


def bary_row(xi_nodes, s):
    """Row vector l with l @ vals = interpolant(s)."""
    N = len(xi_nodes) - 1
    w = (-1.0) ** np.arange(N + 1); w[0] *= 0.5; w[-1] *= 0.5
    d = s - xi_nodes
    if np.any(d == 0):
        r = (d == 0).astype(float); return r
    c = w / d
    return c / c.sum()


def bary_eval(xi_nodes, vals, xi_eval):
    """Barycentric interpolation on Chebyshev-Lobatto nodes."""
    N = len(xi_nodes) - 1
    w = (-1.0) ** np.arange(N + 1); w[0] *= 0.5; w[-1] *= 0.5
    diff = xi_eval[:, None] - xi_nodes[None, :]
    hit = diff == 0
    diff[hit] = 1.0
    C = w / diff
    out = (C @ vals) / C.sum(axis=1)
    r, c = np.nonzero(hit)
    out[r] = vals[c]
    return out


def error(sol, M=4000):
    """Max error of the interpolant on M Chebyshev points of the first kind (in xi)."""
    xi_e = np.cos(np.pi * (np.arange(M) + 0.5) / M)
    p = bary_eval(sol['xi'], sol['y'], xi_e)
    return np.max(np.abs(p - master(sol['tau'](xi_e))))
