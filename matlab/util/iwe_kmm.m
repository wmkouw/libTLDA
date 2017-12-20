function [iw] = iwe_kmm(X, Z, bandwidth, Ktype)
% Kernel Mean Matching for importance weight estimation.
%
% Reference: Huang et al. (2007), Correcting Sample Selection Bias by unlabeled data. NIPS.

% Shapes
[N,~] = size(X);
[M,~] = size(Z);

% Compute Euclidean distances
K = pdist2(X, X);
k = pdist2(X, Z);

% Check for non-negativity
K(K<0) = 0;

switch Ktype
    case 'diste'
        % Compute normalizing constant
        k = N./M*sum(k,2);
    case 'rbf'
        % Radial basis function
        K = exp(-K/(2*bandwidth.^2));
        k = exp(-k/(2*bandwidth.^2));
        k = N./M*sum(k,2);
end

% Solve quadratic program
options.Display = 'final';
iw = quadprog(K,k,[ones(1,N); -ones(1,N)],[N./sqrt(N)+N, N./sqrt(N)-N],[],[], zeros(N,1), [], [], options);

end
