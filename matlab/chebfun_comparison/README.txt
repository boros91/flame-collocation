MATLAB/Chebfun comparison (Section 5.4 and Table 5 of the paper)

Files
  run_chebfun_flame.m   solves y' = y^2 - y^3, y(0) = delta, t in [0, 2/delta], with Chebfun in
                        three ways (ode113; collocation with y(0) = delta; collocation with the
                        interior condition y(t*) = 2/3) for delta = 1e-1, ..., 1e-6
  It uses flame_theta.m, flame_master.m, flame_exact.m from ../method (added to the path).

Requirements
  MATLAB (the paper used R2023b Update 3) and Chebfun 5.7.0 (https://www.chebfun.org) on the path.
  No other toolboxes.

How to run
  >> cd matlab/chebfun_comparison
  >> run_chebfun_flame
  The results are written to ../../results/matlab/chebfun_flame.csv, which
  experiments/gen_tables.py reads. The collocation runs for small delta need several GB of
  memory. A rerun on a second machine reproduced all successes/failures, Newton iteration counts
  and errors; only the lengths of the failed collocation runs (at termination) differed by a few
  coefficients.
