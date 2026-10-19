%RUN_CHEBFUN_FLAME  The flame problem y' = y^2 - y^3, y(0) = delta, 0 <= t <= 2/delta,
%   solved with Chebfun in three ways:
%     'ode113'      : Chebfun's default IVP solver (time stepping, result converted to a chebfun)
%     'colloc_ivp'  : Chebfun's global spectral collocation with the initial condition y(0) = delta
%                     (solvebvp, i.e. what Chebfun did by default before version 5.1)
%     'colloc_pin'  : Chebfun's global spectral collocation with the initial condition replaced by
%                     the equivalent interior condition y(t*) = 2/3, t* = T_delta(2/3)
%                     (front pinning as in the paper, but without the sinh map)
%   For every run: success, length of the chebfun (polynomial degree + 1), Newton iterations,
%   maximal error against the exact solution, recovered initial value, run time, warnings.
%
%   Requirements: Chebfun (https://www.chebfun.org, add its folder to the MATLAB path) and
%   flame_theta.m, flame_master.m, flame_exact.m from ../method (added to the path below).
%   Output: ../../results/matlab/chebfun_flame.csv.   Run time: a few minutes; the collocation runs need
%   several GB of memory.

clear; clc;
here   = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'method'));
outdir = fullfile(here, '..', '..', 'results', 'matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
assert(exist('chebop', 'file') == 2 || exist('chebop', 'class') == 8, ...
       'Chebfun not found: download it from https://www.chebfun.org and add it to the path');

q = 2;
deltas = [1e-1 1e-2 1e-3 1e-4 1e-5 1e-6];
modes  = {'ode113', 'colloc_ivp', 'colloc_pin'};
rows = {};

for d = deltas
    T  = 2 / d;
    tt = unique([linspace(0, T, 20001)'; ...                                   % whole interval
                 (flame_theta(0.5, q) - flame_theta(d, q)) + linspace(-40, 40, 4001)']);
    tt = tt(tt >= 0 & tt <= T);
    ye = flame_exact(tt, d, q);
    for im = 1:numel(modes)
        mode = modes{im};
        N = chebop(0, T);
        N.op = @(t, y) diff(y) - y.^2 + y.^3;
        pref = cheboppref;
        switch mode
            case 'ode113'
                N.lbc = d;
                pref.ivpSolver = 'ode113';
            case 'colloc_ivp'
                N.lbc = d;
            case 'colloc_pin'
                tstar = flame_theta(2/3, q) - flame_theta(d, q);
                N.bc = @(t, y) y(tstar) - 2/3;
        end
        r = struct('q', q, 'delta', d, 'mode', mode, 'success', 0, 'length', NaN, ...
                   'newton_its', NaN, 'maxerr', NaN, 'y0_rel', NaN, 'time_s', NaN, 'message', '');
        lastwarn('');
        tic;
        try
            switch mode
                case 'ode113'
                    y = solveivp(N, 0, pref);
                    info = struct('normDelta', []);
                otherwise
                    [y, info] = solvebvp(N, 0, pref);
            end
            r.time_s  = toc;
            r.success = 1;
            r.length  = length(y);
            if isfield(info, 'normDelta'), r.newton_its = numel(info.normDelta); end
            r.maxerr  = max(abs(y(tt) - ye));
            r.y0_rel  = abs(y(0) / d - 1);
        catch err
            r.time_s  = toc;
            r.message = err.message;
        end
        [w, ~] = lastwarn;
        if ~isempty(w) && isempty(r.message), r.message = w; end
        r.message = strrep(strrep(r.message, newline, ' '), ',', ';');
        rows{end + 1} = r; %#ok<SAGROW>
        fprintf('delta=%.0e %-11s ok=%d length=%6g its=%4g maxerr=%.2e y0rel=%.2e %6.1fs %s\n', ...
            d, mode, r.success, r.length, r.newton_its, r.maxerr, r.y0_rel, r.time_s, r.message);
    end
end

Tab = struct2table([rows{:}]);
writetable(Tab, fullfile(outdir, 'chebfun_flame.csv'));
fprintf('Results written to %s\n', fullfile(outdir, 'chebfun_flame.csv'));
