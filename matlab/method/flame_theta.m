function th = flame_theta(y, q)
%FLAME_THETA  Time function Theta_q(y) for y' = y^q (1-y).
%   T_delta(y) = flame_theta(y,q) - flame_theta(delta,q) is the exact time the
%   solution started at delta needs to reach the level y (0 < delta <= y < 1).
%   Theta_q(y) = log(y/(1-y)) - sum_{j=1}^{q-1} 1/(j*y^j)
th = log(y) - log1p(-y);
for j = 1:q-1
    th = th - 1 ./ (j * y.^j);
end
end
