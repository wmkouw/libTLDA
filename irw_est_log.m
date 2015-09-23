% Function to estimate reweighting coefficients for instance reweighting
function [iw] = irw_est_log(X, Z)
% Function expects MxN matrices.

% Dependencies
addpath(genpath('minFunc'));

% Shape
[M0,N0] = size(X);
[M1,N1] = size(Z);
Y = [ones(1,N0) 2*ones(1,N1)];

% Run logistic regressor on domains
options.DerivativeCheck = 'off';
options.method = 'lbfgs';
w = minFunc(@mLR_grad, randn((M0+1)*2,1), options, [X Z], Y, 0);
w2 = [w(1:M0); w(end-1)];

% Calculate posterior over samples of X0
iw = 1./(1+exp(-w2'*[X; ones(1,N0)]));

end

function [L, dL] = mLR_grad(W,X,y, lambda)
% Implementation of logistic regression
% Wouter Kouw
% 29-09-2014
% This function expects an 1xN label vector y with labels [1,..,K]

% Shape
[M,N] = size(X);
K = numel(unique(y)); if K==1; K=2; end
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
    end
    for k=1:K
        pos_E0(k) = sum(y == k);
    end
    
    % Compute negative part of gradient
    neg_E = X * WX';
    neg_E0 = sum(WX, 2)';
    
    % Compute gradient
    dL = -[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)] + 2 .* lambda .* [W(:); W0(:)];
    
end
end
