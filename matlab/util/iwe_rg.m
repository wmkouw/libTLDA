function [iw] = iwe_rG(X, Z, l2)
% Ratio of Gaussian distributions for importance weight estimation.
% ! Do not use for high-dimensional data
% Function expects MxN matrices.
%
% Reference: Shimodaira (2000), Improving predictive inference under covariate shift by weighting the log-likelihood function. JSPI.

% Shape
[N,D] = size(X);
[M,~] = size(Z);

% Calculate sparse Gaussian parameters
muX = mean(X,1);
muZ = mean(Z,1);
X_ = bsxfun(@minus, X, muX);
Z_ = bsxfun(@minus, Z, muZ);
SiX = 1./N .* (X_'*X_ + l2*eye(D));
SiZ = 1./M .* (Z_'*Z_ + l2*eye(D));

% Ratio of Gaussians
iw = mvnpdf(X, muZ, SiZ) ./ mvnpdf(X, muX, SiX);

end
