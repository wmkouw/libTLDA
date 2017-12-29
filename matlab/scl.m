function [W,pred,C] = scl(X,Z,y,varargin)
% Implementation of a structural correspondence learning classifier
% Assumes Bag-of-Words encoding with count features
%
% Reference: Blitzer et al. (2006), Domain adaptation with structural
% correspondence learning. EMNLP.
%
% Input:    X       source data (N samples x D features)
%           Z       target data (M samples x D features)
%           y       source labels (N x 1) in {1,...,K}
% Optional:
%           l2      additional l2-regularization parameters (default: 1e-3)
%           m       number of pivot features (default: 20)
%           h       number of pivot components (default: 15)
%
% Output:   W       trained linear classifier
%           pred    predictions by trained classifier on target data
%           C       pivot predictor weight components
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

% Parse hyperparameters
p = inputParser;
addOptional(p, 'l2', 0);
addOptional(p, 'm',  2);
addOptional(p, 'h',  2);
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

% Find indices of m most frequent features (assumes BoW encoding)
[~,ix] = sort(sum([X(:,1:end-1); Z(:,1:end-1)],1), 'descend');
ix = ix(1:p.Results.m);

% Slice out pivot features and relabel them as present(=1)/absent(=0)
pivot = [X(:,ix); Z(:,ix)];
pivot = double(pivot>0);

% Solve m binary prediction tasks
P = zeros(D,p.Results.m);
for l = 1:p.Results.m
    disp(['Pivot feature #' num2str(l)]);
    P(:,l) = minFunc(@Huber_grad, rand(D,1), options, [X; Z], pivot(:,l), p.Results.l2);
end

% Decompose pivot predictor matrix into h components
[C,~] = eigs(cov(P'), p.Results.h);

% Augment features
Xa = [X*C, X];
Za = [Z*C, Z];

% Minimize loss
W = mlr(Xa, y, 'l2', p.Results.l2);

% Make predictions
[~,pred] = max(Za*W, [], 2);

end

function [L,dL] = Huber_grad(w,X,y, l2)
% Modified Huber loss function
%
% Reference: Ando & Zhang (2005a). A framework for learning predictive 
% structures from multiple tasks and unlabeled data. JMLR.

% Precompute
Xy = bsxfun(@times, X, y);
Xyw = Xy*w;

% Indices of discontinuity
ix = (Xyw >= -1);

% Loss
L = sum(max(0, 1 - Xyw(ix)).^2,1) ...
    + sum(-4*Xyw(~ix),1);

% Add l2-regularization
L = L + l2*sum(w.^2);

if nargout > 1
    
    % Gradient
    dL = sum(bsxfun(@times, 2*max(0,1-Xyw(ix)), (-Xy(ix,:))),1) ...
        + sum(-4*Xy(~ix,:),1);
    
    % Add l2-regularization
    dL = dL' + 2*l2*w;

end

end
