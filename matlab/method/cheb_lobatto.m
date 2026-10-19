function [D, xi] = cheb_lobatto(N)
%CHEB_LOBATTO  Chebyshev differentiation matrix and the N+1 Chebyshev-Lobatto points
%   xi_j = cos(pi*j/N), j = 0..N, ordered from +1 to -1 (Trefethen, Spectral Methods in
%   MATLAB, 2000, program cheb.m).  xi is a column vector.
if N == 0
    D = 0; xi = 1; return
end
xi = cos(pi * (0:N)' / N);
c  = [2; ones(N - 1, 1); 2] .* (-1).^(0:N)';
X  = repmat(xi, 1, N + 1);
dX = X - X';
D  = (c * (1 ./ c)') ./ (dX + eye(N + 1));
D  = D - diag(sum(D, 2));
end
