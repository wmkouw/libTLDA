function [W] = mLR(X,y,lambda)
% Function that performs multiclass logistic regression
% Assumes DxN data

addpath(genpath('minFunc'));

% Optimization options
options.Display = 'final';
options.Method = 'lbfgs';

% Check for bias augmentation
if ~all(X(end,:)==1); X = [X; ones(1,size(X,2))]; end

% Shape
[M,N] = size(X);

% Check vector y
if size(y,1)~=N; y = y'; end

% Check for y in {1,..K}
y(y== 0) = 2;
y(y==-1) = 2;

% Number of classes
K = max(y);

% Minimize loss
w = minFunc(@mLR_grad, zeros(M*K,1), options, X(1:end-1,:),y,lambda);

% Reshape into K-class matrix
W = [reshape(w(1:end-K), [M-1 K]); w(end-K+1:end)'];

end

function [L, dL] = mLR_grad(W,X,y, lambda)
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
