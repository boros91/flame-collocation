MATLAB implementation of the front-pinned collocation method (Section 4 of the paper)

Files
  flame_pinned.m       the method: sinh-mapped Chebyshev collocation for y' = y^q (1-y),
                       y(0) = delta, in the unknown v = log y, with the interior (pin) condition
                       y = (q-1)/q, or with the initial condition ('pin', false), damped Newton
  example_flame.m      example: flame problem, delta = 1e-10, N = 352; prints iterations and
                       errors of the pinned and IVP forms and plots the solution
  verify_method.m      runs the cases of reference_python.csv (Python implementation,
                       method/collocation_q.py) and compares iterations and errors; writes
                       verify_method_matlab.csv
  cheb_lobatto.m       Chebyshev differentiation matrix and Chebyshev-Lobatto points
  bary_row.m, bary_eval.m   barycentric interpolation on Chebyshev-Lobatto points
  flame_theta.m, flame_master.m, flame_exact.m   time function Theta_q, master profile
                       Y_q = Theta_q^{-1}, exact solution (used for the error and the pin)
  reference_python.csv reference results of the Python implementation

Requirements
  MATLAB R2016b or later (implicit expansion), no toolboxes.

How to run
  >> cd matlab/method
  >> verify_method
  >> example_flame

Usage
  sol = flame_pinned(delta, N, q);                     % pinned form
  sol = flame_pinned(delta, N, q, 'pin', false);       % initial-condition (IVP) form
  Fields: sol.t, sol.y (nodes in the original time and nodal values), sol.its, sol.converged,
  sol.err (sampled maximum error), sol.interp (interpolation error of the exact solution),
  sol.y0_relerr (recovered initial value), and the map parameters.
  A suitable N: the smallest N with error <= 1e-10 is about (2L/pi)*log(1e10), where
  L = sol.L grows like log(1/mu); e.g. N = 64, 160, 224, 288, 352, 416 for
  delta = 1e-2, 1e-4, ..., 1e-12 (q = 2).
