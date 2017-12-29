function [W,pred,G] = gfk(X,Z,y,varargin)
% Implementation of Geodesic Flow Kernel classifier
%
% Reference: Geodesic Flow Kernel for Unsupervised Domain Adaptation. Gong, et al. (2008). CVPR.
%
% Input:    X        source data (N samples x D features)
%           Z        target data (M samples x D features)
%           y        source labels (N x 1) in  {1,...,K}
% Optional:
%           d        subspace dimensionality (default: 'lr')
%           l2       l2-regularization parameters (default: 1e-3)
%           clf      classifier (default: kknn)
%
% Output:   W        trained linear classifier (empty for kknn)
%           pred     predictions for given target data
%           G        geodesic flow kernel
%
% Copyright: Wouter M. Kouw
% Last update: 19-12-2017

% Parse input
p = inputParser;
addParameter(p, 'd', 1);
addParameter(p, 'l2', 1e-3);
addParameter(p, 'clf', 'knn')
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

% Check whether subspace is sufficiently small
if p.Results.d > D/2; error('Subspace dimensionality not smaller than half of full dimensionality'); end

% Find principal components
PX = pca(X, 'Algorithm', 'eig', 'NumComponents', D);
PZ = pca(X, 'Algorithm', 'eig', 'NumComponents', p.Results.d);

% Ortogonal complement to PX
RX = null(PX');

% Find geodesic flow kernel
G = compute_kernel([PX,RX], PZ);

% Perform classification
switch p.Results.clf
    case {'lr', 'log'}
        % Multi-class logistic regression
        W = mlr(X*G, y, 'l2', p.Results.l2);
        
        % Predict target labels
        [~,pred] = max([Z, ones(M, 1)]*W, [], 2);
        
    case {'knn', 'kknn'}
        % Kernel k-nearest-neighbours
        [pred] = kknn(X,Z,y, G, 'l2', p.Results.l2);
        W = [];
end

end

function G = compute_kernel(Q,PZ)
% Input:    
%   Q   =   [PX, null(PX')], where PX are the source principal components
%   PZ  =   target principal components, D features by nE subspace dimensionality
%           (nE should be less than 0.5*D)
% Output: 
%   G   =   solution to integrating the kernel mapping along the full path from source to target domain
%           ( \int_{0}^1 \Phi(t)\Phi(t)' dt )
%
% Reference: Geodesic Flow Kernel for Unsupervised Domain Adaptation. Gong, et al. (2008). CVPR.

% Shapes
D = size(Q, 2);
d = size(PZ,2);

% Principal angles
QPZ = Q' * PZ;
[U,V,~,Gamma,~] = gsvd(QPZ(1:d,:), QPZ(d+1:end,:));
theta = real(acos(diag(Gamma))); 

% Ensure non-zero angles for computational stability
theta = max(theta, 1e-20);

% Filler zero matrices
A1 = zeros(d,D-d);
A2 = zeros(d,D-2*d);
A3 = zeros(D,D-2*d);

% Angle matrices
L1 = 0.5.*diag(1 + sin(2*theta)./ (2.*theta));
L2 = 0.5.*diag((-1 + cos(2*theta))./ (2.*theta));
L3 = 0.5.*diag(1 - sin(2*theta)./ (2.*theta));

% Constituent matrices
C1 = [U, A1; A1', -V]; 
C2 = [L1, L2, A2; L2, L3, A2; A3'];
C3 = [U, A1; A1', -V]';

% Geodesic flow kernel
G = Q * C1 * C2 * C3 * Q';

end
