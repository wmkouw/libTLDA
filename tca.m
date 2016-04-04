function [W,M,pred,varargout] = tca(X,Z,yX,varargin)
% Function to perform Transfer Component Analysis.
% Ref: Pan, Tsang, Kwok, Yang (2009). Domain Adaptation via Transfer Component Analysis.
%
% Input:    X       source data (D features x N samples)
%           Z       target data (D features x M samples)
%           yX      source labels (N x 1)
% Optional:
%           yZ      target labels
%           l2      l2-regularization parameters (default: 1e-3)
%           m       Number of components to reduce to (default: 1)
%           mu      trade-off parameter transfer components (default: 1)
%           theta   radial basis function kernel width (default: 1)
%
% Output:   W       Classifier parameters
%           M       Transfer Components
%           pred    target label predictions
% Optional:
%           {1}     Error of target label predictions
%
% Copyright: Wouter M. Kouw
% Last update: 04-04-2016

addpath(genpath('~/Codes/minFunc'));

% Shapes
[~,NX] = size(X);
[~,NZ] = size(Z);

% Parse input
p = inputParser;
addParameter(p, 'yZ', []);
addParameter(p, 'm', 1);
addParameter(p, 'l2', 1e-3);
addParameter(p, 'mu', 1);
addParameter(p, 'theta', 1);
parse(p, varargin{:});

% Find components
[M,K] = tc(X, Z, 'theta', p.Results.theta, 'mu', p.Results.mu, 'm', p.Results.m);
clear Z

% Build parametric kernel map
B = K'*M;
clear K

% Train classifier
W = mLR(B(1:NX,:)', yX, p.Results.l2);

% Do classification on target set
[~,pred] = max(W'*[B(NX+1:NX+NZ,:)'; ones(1,NZ)], [], 1);

if ~isempty(p.Results.yZ);
    varargout{1} = mean(pred~=p.Results.yZ');
end

end


function [M,K] = tc(X,Z,varargin)
% At the moment, only a radial basis function kernel implemented

% Parse input
p = inputParser;
addParameter(p, 'theta', 1);
addParameter(p, 'mu', 1);
addParameter(p, 'm', 1);
parse(p, varargin{:});

% Shapes
[~,NX] = size(X);
[~,NZ] = size(Z);

% Form block kernels
K = rbf_kernel(X,Z, 'theta', p.Results.theta);

% Objective function
[M,~] = eigs((eye(NX+NZ)+p.Results.mu*K*[ones(NX)./NX.^2 -ones(NX,NZ)./(NX*NZ); ...
    -ones(NZ,NX)./(NX*NZ) ones(NZ)./NZ.^2]*K)\(K*((1-1./(NX+NZ)).*eye(NX+NZ))*K), p.Results.m);
M = real(M);

end

function K = rbf_kernel(X,Z,varargin)

p = inputParser;
addParameter(p, 'theta', 1);
parse(p, varargin{:});

Kst = exp(-pdist2(X', Z')/(2*p.Results.theta.^2));
K = [exp(-pdist2(X', X')/(2*p.Results.theta.^2)) Kst; Kst' exp(-pdist2(Z', Z')/(2*p.Results.theta.^2))];


end

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
[D,N] = size(X);

% Number of classes
K = numel(unique(y)); 

% Check column vector y
if size(y,1)~=N; y = y'; end

% Check for y in {1,..K}
y(y== 0) = 2;
y(y==-1) = 2;

% Minimize loss
w = minFunc(@mLR_grad, zeros(D*K,1), options, X(1:end-1,:),y,lambda);

% Reshape into K-class matrix
W = [reshape(w(1:end-K), [D-1 K]); w(end-K+1:end)'];

end

function [L, dL] = mLR_grad(W,X,y, lambda)
% Implementation of logistic regression
% Wouter Kouw
% 29-09-2014
% This function expects an 1xN label vector y with labels [1,..,K]

% Shape
[D,N] = size(X);
K = numel(unique(y)); 
W0 = reshape(W(D*K+1:end), [1 K]);
W = reshape(W(1:D*K), [D K]);

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
	pos_E = zeros(D, K);
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
