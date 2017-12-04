function [W,EX,EZ] = sa(X,Z,y,varargin)
% Function to train a domain adaptive classifier using Subspace Alignment
% 
% Fernando, B., Habrard, A., Sebban, M. & Tuytelaars, T. Subspace alignment
% for domain adaption. ArXiv 2014
% Input:
%    X is source set in MxN format (no augmentation)
%    Z is target set in MxN format (no augmentation)
%    y is source label vector in [1,..K]
%    l2 is regularization parameter
%    nE is dimensionality to which the subspace is reduced to
% Output:
%    W is KxM resulting classifier
%    EX is source subspace
%    EZ is target subspace (must be used to map novel target data on)
%
% Wouter Kouw
% 15-09-2014

% Dependencies
addpath(genpath('minFunc'));

% Shapes
[MQ,NQ] = size(X);
[MP,NP] = size(Z);

% Parse input
p = inputParser;
addParameter(p, 'l2', 1e-3);
addParameter(p, 'nE', min(min(MP,MQ),100));
parse(p, varargin{:});

% Optimization options
options.Method = 'lbfgs';
options.Display = 'final';

% Number of classes
K = max(y);

% Check for y in {1,..K}
y(y== 0) = 2;
y(y==-1) = 2;

% Covariance
SQ = cov(X');
SP = cov(Z');

% Remove nans (result from sparsity)
SQ(isnan(SQ)) = 0;
SP(isnan(SP)) = 0;

% Eigenvectors
[EX,~] = eigs(SQ,p.Results.nE);
[EZ,~] = eigs(SP,p.Results.nE);

% Optimal linear transformation
M = EX'*EZ;

% Projecting source data on aligned space
XA = X'*EX*M;

% Minimize loss
W = minFunc(@mLR_grad, zeros((p.Results.nE+1)*K,1), options, XA', y, p.Results.l2);

% Output MxK weight matrix
W = [reshape(W(1:end-K), [p.Results.nE K]); W(end-K+1:end)'];

end

function [L, dL] = mLR_grad(W, X, y, lambda)
% Implementation of logistic regression
% Wouter Kouw
% 29-09-2014
% This function expects an 1xN label vector y with labels [1,..,K]

% Shape
[M,N] = size(X);
K = max(y);
W0 = reshape(W(M*K+1:end), [1 K]);
W = reshape(W(1:M*K), [M K]);

% Compute p(y|x)
WX = bsxfun(@plus, W' * X, W0');
WX = exp(bsxfun(@minus, WX, max(WX, [], 1)));
WX = bsxfun(@rdivide, WX, max(sum(WX, 1), realmin));

% Negative log-likelihood of each sample
L = 0;
for i=1:N
    L = L - log(max(WX(y(i), i), realmin));
end
L = L + lambda .* sum([W(:); W0(:)] .^ 2);

% Only compute gradient if requested
if nargout > 1
    
    % Compute positive part of gradient
    pos_E = zeros(M, K);
    pos_E0 = zeros(1, K);
    for k=1:K
        pos_E(:,k) = sum(X(:,y == k), 2);
        pos_E0(k) = sum(y == k);
    end
    
    % Compute negative part of gradient
    neg_E = X * WX';
    neg_E0 = sum(WX, 2)';
    
    % Compute gradient
    dL = -[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)] + 2 .* lambda .* [W(:); W0(:)];
    
end
end
