"""Exact solution of the flame problem  y' = y^2 - y^3,  y(0) = delta  (q = 2), via the
Wright omega function:
    y_delta(t) = 1 / (1 + omega(a + ln a - t)),   a = 1/delta - 1,
so y_delta(t) = Y(t - c(delta)) with the master profile Y(t) = 1/(1+omega(1-t))
(Y(0) = 1/2) and c(delta) = a + ln a - 1.  Used by collocation.py (q = 2 utilities);
the general-q functions are in flame_q.py.
"""
import numpy as np
from scipy.special import wrightomega


def shift(delta):
    a = 1.0 / delta - 1.0
    return a + np.log(a) - 1.0


def master(t):
    """Master profile Y(t) = 1/(1+omega(1-t)), Y(0) = 1/2."""
    return 1.0 / (1.0 + np.real(wrightomega(1.0 - np.asarray(t, dtype=float))))


def exact(t, delta):
    return master(np.asarray(t, dtype=float) - shift(delta))
