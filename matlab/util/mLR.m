function [W] = mLR(X,y,varargin)
% Multi-class logistic regression
%
% Input:
%    X is N samples by D features
%    y is N samples by 1 vector in [1,..K]
% Output:
%    W is D features by K classes
%
% Wouter Kouw
% 15-09-2014

% Check for solver
if isempty(which('minFunc')); error('Can not find minFunc'); end
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'final';

% Parse input
p = inputParser;
addParameter(p, 'l2', 1e-3);
parse(p, varargin{:});

% Shape
[N,D] = size(X);

% Check for bias augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; D = D+1; end

% Number of classes
labels = unique(y);
K = length(labels); 

% Check column vector y
if size(y,1)~=N; y = y'; end

% Check if labels are in [1,...,K]
if ~isempty(setdiff(labels,1:K)); error('Labels should be in [1,...,K]'); end

% Minimize loss
W = minFunc(@mLR_grad, rand(D*K,1), options, X,y, p.Results.l2);

% Reshape into K-class matrix
W = reshape(W, [D K]);

end

function [L, dL] = mLR_grad(W,X,y,l2)
% Logistic regression gradient
% Wouter Kouw
% 29-09-2014

% Shape
[~,D] = size(X);
K = max(max(y),numel(unique(y))); 
W = reshape(W, [D K]);

% Numerical stability
XW = X*W;
max_const = max(XW, [], 2);
XW = bsxfun(@minus, XW, max_const);

% Convenient variables
eXW = exp(XW);
eA = sum(eXW,2);

% Point-wise weighted negative log-likelihood
ll = -sum(X.*W(:,y)',2) + log(eA) + max_const;

% Add l2-regularizer
L = sum(ll,1) + l2.*sum(W(:).^2);

% Only compute gradient if requested
if nargout > 1
  
    % Gradient with respect to B_k
    dll = zeros(D,K);
    for k = 1:K
        dll(:,k) = -X'*(y==k) + X'*(eXW(:,k)./eA);
    end
    
	% Add l2-regularizer
	dL = dll(:) + 2*l2.*W(:);
    
end
end
