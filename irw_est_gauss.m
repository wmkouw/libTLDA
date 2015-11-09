% Function to do ML estimation of the transfer parameter q.
function [iw] = irw_est_gauss(X, Z, lambda)
% Function expects MxN matrices.

% Parse input
if ~exist('lambda', 'var'); lambda = 1; end

% Shape
[M,N] = size(X);

% Calculate sparse Gaussian parameters
mu0 = mean(X,2);
mu1 = mean(Z,2);
X0_ = bsxfun(@minus, X, mu0);
X1_ = bsxfun(@minus, Z, mu1);
S0 = 1./N .* (X0_*X0_' + lambda*eye(M));
S1 = 1./N .* (X1_*X1_' + lambda*eye(M));
L0 = chol(S0, 'lower');
L1 = chol(S1, 'lower');

% Calculate reweighting of X_0 based on difference in Gaussians
temp = (L1'\(L1\mu1)) - (L0'\(L0\mu0));
mu_ = L1'\(L1\temp) - L0'\(L0\temp);
X = bsxfun(@minus, X, mu_);
iw = exp(-1/2.*(sum(X.*(L1\X),1) - sum(X.*(L0\X),1)));
end