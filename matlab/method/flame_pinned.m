function sol = flame_pinned(delta, N, q, varargin)
%FLAME_PINNED  Front-pinned sinh-mapped Chebyshev collocation for the ignition problems
%       y' = y^q (1-y),  y(0) = delta,  q >= 2   (q = 2: the flame problem).
%
%   sol = flame_pinned(delta, N, q)
%   sol = flame_pinned(delta, N, q, 'pin', false)      % initial-condition (IVP) form
%
%   The problem is rescaled to x = mu*t - 1 in [-1, 1], mu = (q-1)*delta^(q-1), and solved for
%   v = log(y):  mu*v' = F_q(v) = exp((q-1)v) - exp(qv).  Algorithm of Section 4 of the paper:
%     1. x0 = -mu*Theta_q(delta) - 1, pin value y* = (q-1)/q, s_pin = Theta_q(y*);
%     2. nearest singularity in the upper half-plane s_sing = pi*i: eps = pi*mu, centre x0;
%     3. sinh map x = x0 + eps*sinh(L*xi + M) of the N+1 Chebyshev-Lobatto points xi_j
%        (xi_1 = 1, ..., xi_{N+1} = -1), D_x = diag(1/g'(xi)) D;
%     4. residual R_j = mu*(D_x v)_j - F_q(v_j) for the first N rows; the last row is the
%        condition: ell'*v - log(y*) (barycentric row ell at the pin) or v(-1) - log(delta);
%     5. initial guess: the generic profile 1/(1 + ((q-1)*softplus(s0 - s))^(1/(q-1)));
%     6. damped Newton: first lambda in {1, 1/2, ..., 2^-10} with
%        ||R(v + lambda*Delta)||_2 < (1 - 1e-4*lambda)*||R(v)||_2, else lambda = 2^-10;
%        converged when max|Delta| < tol and, after the full step, max|R| < rtol
%        ('test','full', default) or when max|lambda*Delta| < tol and max|R| < rtol ('damped').
%
%   Options (name/value): 'pin' (true), 'test' ('full' | 'damped'), 'tol' (1e-12),
%   'rtol' (1e-10), 'maxit' (80), 'nerr' (4000: number of points for the error; 0 = skip).
%
%   Output fields: xi, x (nodes), s (master variable (x - x0)/mu), t (original time), y (nodal
%   values), its (Newton iterations), converged, lambda_last, n_fallback, res (final max|R|),
%   y0_relerr = |y(-1)/delta - 1|, L, M, mu, x0, and (if nerr > 0) err = sampled maximum error
%   of the barycentric interpolant of y against the exact solution, interp = the same for the
%   interpolant of the exact nodal values.
%
%   Requires flame_theta.m, flame_master.m, cheb_lobatto.m, bary_row.m, bary_eval.m.

p = inputParser;
addParameter(p, 'pin', true);
addParameter(p, 'test', 'full');
addParameter(p, 'tol', 1e-12);
addParameter(p, 'rtol', 1e-10);
addParameter(p, 'maxit', 80);
addParameter(p, 'nerr', 4000);
parse(p, varargin{:});
o = p.Results;

% 1. scaling, position of the front, pin
mu    = (q - 1) * delta^(q - 1);
x0    = -mu * flame_theta(delta, q) - 1;
ystar = (q - 1) / q;
s_pin = flame_theta(ystar, q);

% 2.-3. sinh map centred at x0 with width eps = mu*Im(s_sing) = pi*mu
ep = pi * mu;
[D, xi] = cheb_lobatto(N);
al = asinh((-1 - x0) / ep);
be = asinh(( 1 - x0) / ep);
L  = (be - al) / 2;
M  = (be + al) / 2;
gp = ep * L * cosh(L * xi + M);            % g'(xi)
x  = x0 + ep * sinh(L * xi + M);           % nodes in x
s  = pi * sinh(L * xi + M);                % s = (x - x0)/mu, without cancellation
Dx = diag(1 ./ gp) * D;

% 5. initial guess (generic profile, equal to y* at the pin; softplus without overflow)
spv = (1 / (q - 1))^(q - 1) / (q - 1);
s0  = s_pin + log(expm1(spv));
z   = s0 - s;
sp  = z;
k   = z <= 30;
sp(k) = log1p(exp(z(k)));
y   = 1 ./ (1 + ((q - 1) * sp).^(1 / (q - 1)));
y(end) = delta;
v   = log(y);

F  = @(v) exp((q - 1) * v) - exp(q * v);
dF = @(v) (q - 1) * exp((q - 1) * v) - q * exp(q * v);
if o.pin
    xi_pin = (asinh(s_pin / pi) - M) / L;
    ell    = bary_row(xi, xi_pin);         % row vector
    pval   = log(ystar);
    res    = @(v) [mu * (Dx(1:N, :) * v) - F(v(1:N)); ell * v - pval];
else
    res    = @(v) [mu * (Dx(1:N, :) * v) - F(v(1:N)); v(end) - log(delta)];
end

% 6. damped Newton
R = res(v);
ok = false; nfb = 0; lam = 1;
for it = 1:o.maxit
    J = mu * Dx - diag(dF(v));
    if o.pin
        J(end, :) = ell;
    else
        J(end, :) = 0; J(end, end) = 1;
    end
    dv = -(J \ R);
    if strcmp(o.test, 'full') && max(abs(dv)) < o.tol
        vt = v + dv; Rt = res(vt);
        if max(abs(Rt)) < o.rtol
            v = vt; R = Rt; lam = 1; ok = true;
            break
        end
    end
    lam = 1; r0 = norm(R);
    while true
        vn = v + lam * dv; Rn = res(vn);
        if norm(Rn) < (1 - 1e-4 * lam) * r0
            break
        end
        if lam < 1e-3
            nfb = nfb + 1;                 % no sufficient decrease: take the smallest step
            break
        end
        lam = lam / 2;
    end
    v = vn; R = Rn;
    if strcmp(o.test, 'damped') && max(abs(lam * dv)) < o.tol && max(abs(R)) < o.rtol
        ok = true;
        break
    end
end

sol = struct('xi', xi, 'x', x, 's', s, 't', (x + 1) / mu, 'y', exp(v), 'its', it, ...
             'converged', ok, 'lambda_last', lam, 'n_fallback', nfb, 'res', max(abs(R)), ...
             'y0_relerr', abs(exp(v(end)) / delta - 1), 'L', L, 'M', M, 'mu', mu, 'x0', x0, ...
             'q', q, 'delta', delta, 'N', N, 'pin', o.pin);

% sampled maximum error on nerr Chebyshev points of the first kind in xi
if o.nerr > 0
    xe  = cos(pi * ((0:o.nerr-1)' + 0.5) / o.nerr);
    yex = flame_master(pi * sinh(L * xe + M), q);
    sol.err    = max(abs(bary_eval(xi, sol.y, xe) - yex));
    sol.interp = max(abs(bary_eval(xi, flame_master(s, q), xe) - yex));
end
end
