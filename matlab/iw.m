function [W,pred,iw] = iw(X,Z,y,varargin)
% Implementation of an importance-weighted classifier
%
% References:
% Kernel density estimators.
% Ratio of Gaussians: Shimodaira (2000), Improving predictive inference under covariate shift by weighting the log-likelihood function. JSPI.
% Logistic discrimination: Bickel et al. (2009), Discriminative learning under covariate shift. JMLR.
% Kernel mean matching: Huang et al. (2007), Correcting sample selection bias by unlabeled data. NIPS.
% Nearest-neighbour-based: Loog (2015), Nearest neighbor-based importance weighting. MLSP.
%
% Input:    X       source data (N samples x D features)
%           Z       target data (M samples x D features)
%           y       source labels (N x 1) in {1,...,K}
% Optional:
%           l2      additional l2-regularization parameters (default: 1e-3)
%           iwe     choice of importance weight estimation method (default: 'lr')
%           loss    choice of loss function (default: 'log')
%
% Output:   W       trained linear classifier
%           pred    predictions by trained classifier on target data
%           iw      estimated importance weights for source samples
%
% Copyright: Wouter M. Kouw
% Last update: 19-12-2017

% Add dependencies to path
addpath(genpath('util'));
addpath(genpath('minFunc'));

% Check for solver
if isempty(which('minFunc')); error('Can not find minFunc'); end
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'final';

% Parse optionals
p = inputParser;
addOptional(p, 'l2', 1e-3);
addOptional(p, 'iwe', 'log');
addOptional(p, 'loss', 'log');
parse(p, varargin{:});

% Data shape
[N,D] = size(X);
[M,E] = size(Z);
labels = unique(y);
K = length(labels);

% Check if dimensionalities are the same
if D~=E; error('Data dimensionalities not the same in both domains'); end

% Check if labels are in [1,...,K]
if ~isempty(setdiff(labels,1:K)); error('Labels should be in [1,...,K]'); end

% Check for bias augmentation
if ~all(X(:,end)==1) && ~all(Z(:,end)==1)
    X = [X ones(N,1)];
    Z = [Z ones(M,1)];
    D = D+1;
end

% Estimate importance weights
switch lower(p.Results.iwe)
    case {'lr', 'log'}
        iw = iwe_lr(X,Z, p.Results.l2);
    case {'rg', 'gauss'}
        iw = iwe_rg(X(:,1:end-1),Z(:,1:end-1), p.Results.l2);
    case {'kd', 'kde'}
        iw = iwe_kd(X(:,1:end-1),Z(:,1:end-1), p.Results.l2);
    case 'kmm'
        iw = iwe_kmm(X(:,1:end-1),Z(:,1:end-1), 1,'rbf');
    case 'nn'
        iw = iwe_nn(X(:,1:end-1),Z(:,1:end-1));
    otherwise
        print('Importance-weight estimator unknown')
end

switch p.Results.loss
    case {'ls','qd'}
        % Map y to one-hot encoding in {-1,+1} (= one-v-rest)
        Y = -ones(N,K);
        for n = 1:N
            Y(n,y(n)) = 1;
        end

        % Least-squares solution to importance-weighted quadratic loss
        W = (X'*diag(iw)*X + p.Results.l2*eye(D))\(X'*diag(iw)*Y);

    case {'lr','log'}
        % Minimize loss
        W = minFunc(@mwlr_grad, zeros(D*K,1), options, X, y, iw, p.Results.l2);
        W = reshape(W, [D,K]);

    otherwise
        error('Loss function not implemented');
end

% Predict target labels
[~,pred] = max(Z*W, [], 2);

end

function [L, dL] = mwlr_grad(W,X,y,iw, l2)
% Multi-class importance-weighted logistic regression gradient
% Function expects bias-augmented X and y in [1,...,K]

% Shapes
[~,D] = size(X);
labels = unique(y);
K = length(labels);
W = reshape(W, [D,K]);

% Numerical stability
XW = X*W;
max_const = max(XW, [], 2);
XW = bsxfun(@minus, XW, max_const);

% Point-wise weighted negative log-likelihood
L = -iw.*sum(X.*W(:,y)',2) + iw.*log(sum(exp(XW),2)) + iw.*max_const;

% Add l2-regularizer
L = sum(L,1) + l2 .* sum(W(:).^2);

% Only compute gradient if requested
if nargout > 1

    % Gradient with respect to B_k
    dL = zeros(D,K);
    for k = 1:K
        dL(:,k) = sum( -bsxfun(@times,iw.*(y==labels(k)),X)' + ...
            bsxfun(@times,iw.*exp(XW(:,k))./sum(exp(XW),2),X)',2);
    end

	% Add l2-regularizer
	dL = dL(:) + 2*l2.*W(:);

end

end
