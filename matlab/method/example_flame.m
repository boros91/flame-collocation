%EXAMPLE_FLAME  Front-pinned collocation for the flame problem y' = y^2 (1-y), y(0) = delta.
%   Solves for delta = 1e-10 with N = 352 (N_min of Table 1 of the paper), prints the number of
%   Newton iterations and the errors, and plots the solution against the original time t.
delta = 1e-10; N = 352; q = 2;

sol = flame_pinned(delta, N, q);
fprintf('pinned form: %d Newton iterations, converged = %d\n', sol.its, sol.converged);
fprintf('  sampled maximum error %.2e (interpolation error of the exact solution %.2e)\n', ...
        sol.err, sol.interp);
fprintf('  recovered initial value: |y(0)/delta - 1| = %.1e\n', sol.y0_relerr);

ivp = flame_pinned(delta, N, q, 'pin', false);
fprintf('IVP form:    %d Newton iterations, converged = %d, error %.2e\n', ivp.its, ivp.converged, ivp.err);

% the solution on a fine grid (barycentric interpolation in xi) against t = (x + 1)/mu
xf = linspace(-1, 1, 4001)';
xif = (asinh((xf - sol.x0) / (pi * sol.mu)) - sol.M) / sol.L;     % inverse sinh map
yf  = bary_eval(sol.xi, sol.y, xif);
figure
subplot(1, 2, 1)
plot((xf + 1) / sol.mu, yf, '-', sol.t, sol.y, '.')
xlabel('t'), ylabel('y'), title(sprintf('\\delta = %g, N = %d', delta, N))
subplot(1, 2, 2)
plot(sol.s, sol.y, '.'), xlim([-40 25])
xlabel('s = t - t_{front}'), ylabel('y'), title('nodes near the front')
