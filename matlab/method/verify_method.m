%VERIFY_METHOD  Runs the MATLAB implementation on the cases of reference_python.csv (computed
%   with method/collocation_q.py, test = 'full') and compares iterations and errors.
%   Pinned form: expect the same iteration counts and errors equal up to rounding (a few 1e-15
%   absolute).  IVP form: rounding-sensitive (Section 5.1 of the paper), so differences in
%   iteration counts and errors are expected and only reported.
%   Writes verify_method_matlab.csv in this folder.
ref = readtable('reference_python.csv');
n = height(ref);
its = zeros(n, 1); conv = zeros(n, 1); err = zeros(n, 1); interp = zeros(n, 1); y0 = zeros(n, 1);
tm = zeros(n, 1);
for k = 1:n
    pin = strcmp(char(ref.form(k)), 'pin');
    tic
    sol = flame_pinned(ref.delta(k), ref.N(k), ref.q(k), 'pin', pin);
    tm(k) = toc;
    its(k) = sol.its; conv(k) = sol.converged; err(k) = sol.err; interp(k) = sol.interp;
    y0(k) = sol.y0_relerr;
end
out = ref;
out.its_matlab = its; out.converged_matlab = conv; out.err_matlab = err;
out.interp_matlab = interp; out.y0_relerr_matlab = y0; out.time_s = tm;
writetable(out, 'verify_method_matlab.csv');

fprintf('%2s %8s %4s %4s | %4s %4s | %10s %10s | %10s %10s\n', 'q', 'delta', 'N', 'form', ...
        'its', 'ref', 'err', 'ref', 'interp', 'ref');
for k = 1:n
    fprintf('%2d %8.0e %4d %4s | %4d %4d | %10.3e %10.3e | %10.3e %10.3e\n', ref.q(k), ref.delta(k), ...
            ref.N(k), char(ref.form(k)), its(k), ref.its(k), err(k), ref.err(k), interp(k), ref.interp(k));
end
P = strcmp(ref.form, 'pin');
fprintf('\npinned: iteration counts equal in %d of %d cases, all converged: %d\n', ...
        sum(its(P) == ref.its(P)), sum(P), all(conv(P)));
fprintf('pinned: max |err - err_ref| = %.1e, max |interp - interp_ref| = %.1e\n', ...
        max(abs(err(P) - ref.err(P))), max(abs(interp(P) - ref.interp(P))));
fprintf('IVP: same convergence flag in %d of %d cases (rounding-sensitive, see Section 5.1)\n', ...
        sum(conv(~P) == ref.converged(~P)), sum(~P));
