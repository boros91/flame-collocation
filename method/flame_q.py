"""Generalised flame family  y' = f_q(y) = y^q (1 - y),  y(0) = delta,  q = 2, 3, ...
(q = 2: flame problem; q = 3: cubic autocatalysis).

Time function: Theta_q(y) = sum_{j=1}^{q-1} -1/(j y^j) + ln(y/(1-y)),
T_delta(y) = Theta_q(y) - Theta_q(delta), master profile Y_q = Theta_q^{-1},
y_delta(t) = Y_q(t + Theta_q(delta)).  Nearest complex singularities: s = +- i*pi.
"""
import numpy as np


def Theta(y, q):
    y = np.asarray(y, dtype=float)
    out = np.log(y) - np.log1p(-y)
    for j in range(1, q):
        out = out - 1.0 / (j * y**j)
    return out


def T(y, delta, q):
    return Theta(y, q) - Theta(delta, q)


def _theta_logit(l, q):
    y = 1.0 / (1.0 + np.exp(-l))
    out = l.copy()
    for j in range(1, q):
        out = out - 1.0 / (j * y**j)
    return out


def master(s, q, iters=200):
    """Y_q(s) = Theta_q^{-1}(s): vectorised bisection in the logit variable l = ln(y/(1-y))
    (robust for y -> 0 and y -> 1), followed by Newton polishing in l."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    lo = np.full(s.shape, -60.0); hi = np.full(s.shape, 60.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = _theta_logit(mid, q) < s
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
        if np.max(hi - lo) < 1e-13:
            break
    l = 0.5 * (lo + hi)
    for _ in range(3):   # dTheta/dl = 1/(y^{q-1}(1-y)) * y(1-y) = y^{1-q}
        y = 1.0 / (1.0 + np.exp(-l))
        l = l - (_theta_logit(l, q) - s) * y**(q - 1)
    return 1.0 / (1.0 + np.exp(-l))


def exact(t, delta, q):
    return master(np.asarray(t, dtype=float) + Theta(delta, q), q)
