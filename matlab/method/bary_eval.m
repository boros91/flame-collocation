function p = bary_eval(xi, f, te)
%BARY_EVAL  Values at the points te of the polynomial interpolant of the values f at the
%   Chebyshev-Lobatto points xi (barycentric formula of the second kind).  Column output.
N  = numel(xi) - 1;
w  = (-1).^(0:N); w([1, end]) = w([1, end]) / 2;     % row
te = te(:);
d  = te - xi(:)';                                     % numel(te) x (N+1)
hit = (d == 0);
d(hit) = 1;
C  = w ./ d;
p  = (C * f(:)) ./ sum(C, 2);
[r, c] = find(hit);
p(r) = f(c);
end
