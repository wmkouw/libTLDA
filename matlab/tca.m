function [W,pred,C,K] = tca(X,Z,y,varargin)
% Implementation of a Transfer Component Analysis classifier
%
% Reference: Pan, Tsang, Kwok, Yang (2009). Domain Adaptation via Transfer
% Component Analysis. IJCAI
%
% Input:    X        source data (N samples x D features)
%           Z        target data (M samples x D features)
%           y        source labels (N x 1) in  {1,...,K}
% Optional:
%           l2      l2-regularization parameters (default: 1e-5)
%           nC      Number of components to reduce to (default: 1)
%           mu      trade-off parameter transfer components (default: 1)
%           bw      radial basis function kernel bandwidth (default: 1)
%
% Output:   W       Classifier parameters
%           pred    target label predictions
%           C       Transfer Components
%           K       Cross-domain kernel
%
% Copyright: Wouter M. Kouw
% Last update: 19-12-2017

% Add dependencies to path
addpath(genpath('util'));

% Parse input
p = inputParser;
addOptional(p, 'l2', 1e-5);
addOptional(p, 'nC', 1);
addOptional(p, 'mu', .1);
addOptional(p, 'bw', 1);
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

% At most, N+M-1 transfer components can be extracted
if p.Results.m > N+M-1; error('Too many components specified'); end

% Find transfer components
[C,K] = tc(X, Z, 'bw', p.Results.bw, 'mu', p.Results.mu, 'm', p.Results.nC);

% Map kernelized source data on transfer components
X = K(1:N,:)*C;

% Map kernelized target data on transfer components
Z = K(N+1:end,:)*C;

% Train classifier
W = mlr(X, y, 'l2', p.Results.l2);

% Do classification on target set
[~,pred] = max([Z ones(M,1)]*W, [], 2);

end


function [C,K] = tc(X,Z,varargin)
% Find transfer components based on distances between datasets
%
% Input:    X        source data (N samples x D features)
%           Z        target data (M samples x D features)
% Optional:
%           l2      l2-regularization parameters (default: 1e-5)
%           nC      Number of components to reduce to (default: 1)
%           mu      trade-off parameter transfer components (default: 1)
%           bw      radial basis function kernel bandwidth (default: 1)

% Parse input
p = inputParser;
addOptional(p, 'l2', 1e-5);
addOptional(p, 'bw', 1);
addOptional(p, 'mu', 1);
addOptional(p, 'nC', 1);
parse(p, varargin{:});

% Shapes
[N,~] = size(X);
[M,~] = size(Z);

% Compute radial basis distance matrices
dXX = exp( -pdist2(X,X) / (2 * p.Results.bw.^2));
dXZ = exp( -pdist2(X,Z) / (2 * p.Results.bw.^2));
dZZ = exp( -pdist2(Z,Z) / (2 * p.Results.bw.^2));

% Form block kernel
K = [dXX dXZ; dXZ' dZZ];

% Ensure positive-definiteness by regularization
K = K + p.Results.l2*eye(N + M);

% Normalization matrix
L = [ones(N)./N.^2 -ones(N,M)./(N*M); -ones(M,N)./(N*M) ones(M)./M.^2];

% Centering matrix
H = eye(N + M) - ones(N + M)./(N + M);

% Matrix Lagrangian objective function: (I + mu*K*L*K)^(-1)*K*H*K
J = (eye(N + M) + p.Results.mu*K*L*K)\(K*H*K);

% Eigenvector decomposition as solution to trace minimization
[C,~] = eigs(J, p.Results.m);

% Discard imaginary numbers (possible computation issue)
C = real(C);

end
