function Y = svdinv(X)

% Singular Value Decomposition
[U,S,V] = svd(X);
s = diag(S);

% Invert the diagonal singular value matrix
s(s>0) = 1./s(s>0);

% Reconstruct the matrix
Y = U*diag(s)*V';

end
