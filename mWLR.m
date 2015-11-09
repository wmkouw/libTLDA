function [W] = mWLR(X,y,iw,varargin)
% Function to start optimize an instance reweighted loss function
% Input:
%    X is in MxN format (no augmentation)
%    y is label vector in [1,..K]
%    iw is reweighting vector (1xN)
% Output:
%    W is KxM resulting classifier
%
% Wouter Kouw
% 15-09-2014

% Dependencies
addpath(genpath('minFunc'));

% Parse input
p = inputParser;
addParameter(p, 'l2', 1e-3);
parse(p, varargin{:});

% Optimization options
options.DerivativeCheck = 'off';
options.Display = 'final';
options.Method = 'lbfgs';

% Check for y in {1,..K}
if any(y== 0); y(y== 0) = 2; end
if any(y==-1); y(y==-1) = 2; end

% Shape
[M,N] = size(X);
K = max(y);

% Minimize loss
W_star = minFunc(@mWLR_grad, zeros((M+1)*K,1), options, X, y, iw, p.Results.l2);

% Output multiclass weight vector
W = [reshape(W_star(1:end-K), [M K]); W_star(end-K+1:end)'];

end

function [L, dL] = mWLR_grad(W,X,y,iw, lambda)
% Implementation of weighted logistic regression
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
    L = L - log(max(iw(i)*WX(y(i), i), realmin));
end
L = L + lambda .* sum([W(:); W0(:)] .^ 2);

% Only compute gradient if requested
if nargout > 1
    
    % Compute positive part of gradient
    pos_E = zeros(M, K);
    pos_E0 = zeros(1, K);
    for k=1:K
        pos_E(:,k) = sum(bsxfun(@times, iw(y == k), X(:,y == k)), 2);
    end
    for k=1:K
        pos_E0(k) = sum(y == k);
    end
    
    % Compute negative part of gradient
    neg_E = bsxfun(@times, iw, X) * WX';
    neg_E0 = sum(WX, 2)';
    
    % Compute gradient
    dL = -[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)] + 2 .* lambda .* [W(:); W0(:)];
    
end
end
