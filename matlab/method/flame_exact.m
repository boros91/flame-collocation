function y = flame_exact(t, delta, q)
%FLAME_EXACT  Exact solution of y' = y^q (1-y), y(0) = delta (q = 2: flame problem).
%   For q = 2 this equals 1./(1 + wrightOmega(1/delta - 1 + log(1/delta - 1) - t)),
%   i.e. Moler's Lambert-W formula, but it is evaluated without overflow.
if nargin < 3, q = 2; end
y = flame_master(t + flame_theta(delta, q), q);
end
