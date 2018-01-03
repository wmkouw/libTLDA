function [W,pred,V,PX,PZ] = suba(X,Z,y,varargin)
% Implementation of a Subspace Alignment classifier.
%
% Reference: Fernando, et al. (2014). Subspace alignment for domain adaption. ICCV.
%
% Input:    X        source data (N samples x D features)
%           Z        target data (M samples x D features)
%           y        source labels (N x 1) in  {1,...,K}
% Optional:
%           l2       l2-regularization parameters (default: 1e-3)
%           zscored  already z-scored or not (default: false)
%           nE       subspace dimensionality (default: 'lr')
%
% Output:   W        trained linear classifier
%           pred     predictions for given target data
%           V        alignment transformation matrix
%           PX       principal components of source data
%           PZ       principal components of target data
%
% Copyright: Wouter M. Kouw
% Last update: 19-12-2017

% Add dependencies to path
addpath(genpath('util'));

% Parse input
p = inputParser;
addParameter(p, 'l2', 1e-3);
addParameter(p, 'zscored', false);
addParameter(p, 'nE', 1);
parse(p, varargin{:});

% Data shape
[N, D] = size(X);
[M, E] = size(Z);
labels = unique(y);
K = length(labels);

% Check if dimensionalities are the same
if D~=E; error('Data dimensionalities not the same in both domains'); end

% Check if labels are in [1,...,K]
if ~isempty(setdiff(labels,1:K)); error('Labels should be in [1,...,K]'); end

% Check for z-scoring
if ~p.Results.zscored
    X = zscore(X);
    Z = zscore(Z);
end

% Principal components of each domain
PX = pca(X, 'Algorithm', 'eig', 'NumComponents', p.Results.nE);
PZ = pca(Z, 'Algorithm', 'eig', 'NumComponents', p.Results.nE);

% Aligned source components
V = PX'*PZ;

% Map source data on aligned components
XP = X*PX*V;

% Minimize loss
W = mlr(XP,y, 'l2', p.Results.l2);

% Map target data on target components
ZP = Z*PZ;

% Predict target labels
[~,pred] = max([ZP, ones(M, 1)]*W, [], 2);

end
