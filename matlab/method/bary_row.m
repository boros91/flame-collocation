function ell = bary_row(xi, t)
%BARY_ROW  Row vector ell with ell*f = value at t of the polynomial interpolant of the values
%   f at the Chebyshev-Lobatto points xi (barycentric formula of the second kind).
N = numel(xi) - 1;
w = (-1).^(0:N)'; w([1, end]) = w([1, end]) / 2;
d = t - xi(:);
if any(d == 0)
    ell = double(d == 0)';
    return
end
c = w ./ d;
ell = (c / sum(c))';
end
