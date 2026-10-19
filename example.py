"""Minimal example: front-pinned sinh-mapped collocation for the flame problem
    y' = y^2 (1 - y),  y(0) = delta,
with delta = 1e-10 and N = 352 (the smallest N with error <= 1e-10, Table 1 of the paper).
Run from the repository root:  python example.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'method'))
import numpy as np
from collocation_q import solve, error, interp_error, params
from flame_q import Theta

delta, N, q = 1e-10, 352, 2

sol = solve(delta, N, q)                       # pinned form (interior condition y = (q-1)/q)
print(f'pinned form: {sol["its"]} Newton iterations, converged = {sol["converged"]}')
print(f'  sampled maximum error {error(sol):.2e}  (interpolation error of the exact solution '
      f'{interp_error(delta, N, q):.2e})')
print(f'  recovered initial value: |y(0)/delta - 1| = {abs(sol["y"][-1] / delta - 1):.1e}')

ivp = solve(delta, N, q, pin=False)            # the same scheme with the initial condition
print(f'IVP form:    {ivp["its"]} Newton iterations, converged = {ivp["converged"]}, error {error(ivp):.2e}')

# nodes in the original time t: s = (x - x0)/mu = t + Theta_q(delta)
mu, x0 = params(delta, q)
t = sol['svar'](sol['xi']) - Theta(delta, q)
print(f'layer width mu = {mu:.1e}; the front (y = 1/2) is at t = {Theta(0.5, q) - Theta(delta, q):.6e}')

try:
    import matplotlib.pyplot as plt
    s = sol['svar'](sol['xi'])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(t, sol['y'], '.-', ms=3)
    ax[0].set_xlabel('t'); ax[0].set_ylabel('y'); ax[0].set_title(f'delta = {delta:g}, N = {N}')
    w = (s > -40) & (s < 25)
    ax[1].plot(s[w], sol['y'][w], 'o', ms=3, mfc='none')
    ax[1].set_xlabel('s = t + Theta_q(delta)'); ax[1].set_title('nodes near the front')
    fig.tight_layout(); plt.show()
except ImportError:
    pass
