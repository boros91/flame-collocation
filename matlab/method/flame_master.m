function y = flame_master(s, q)
%FLAME_MASTER  Master profile Y_q = Theta_q^{-1} (inverse of FLAME_THETA).
%   Every solution of y' = y^q (1-y), y(0) = delta, is a time translate:
%   y_delta(t) = Y_q(t + Theta_q(delta)).
%   Vectorised bisection in the logit variable l = log(y/(1-y)), followed by
%   three Newton steps (dTheta/dl = y^(1-q)).  No toolboxes required.
sz = size(s);
s  = s(:);
lo = -60 * ones(size(s));
hi =  60 * ones(size(s));
for k = 1:200
    mid = 0.5 * (lo + hi);
    up  = theta_logit(mid, q) < s;
    lo(up)  = mid(up);
    hi(~up) = mid(~up);
    if max(hi - lo) < 1e-13
        break
    end
end
l = 0.5 * (lo + hi);
for k = 1:3
    yy = 1 ./ (1 + exp(-l));
    l  = l - (theta_logit(l, q) - s) .* yy.^(q - 1);
end
y = reshape(1 ./ (1 + exp(-l)), sz);
end

function th = theta_logit(l, q)
yy = 1 ./ (1 + exp(-l));
th = l;
for j = 1:q-1
    th = th - 1 ./ (j * yy.^j);
end
end
