function [W,M,pred,err] = tca(X,Z,yX,yZ,varargin)
% Function to train a domain adaptive classifier using Transfer Component Analysis
%
% Pan, Tsang, Kwok, Yang (2009). Domain Adaptation via Transfer Component Analysis.

addpath(genpath('minFunc'));

% Shapes
[~,NX] = size(X);
[~,NZ] = size(Z);

% Parse input
p = inputParser;
addParameter(p, 'm', 100);
addParameter(p, 'l2', 0);
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
err = mean(pred~=yZ');

end


function [M,K] = tc(X,Z,varargin)
% At the moment, only a radial basis function kernel is implemented

% Parse input
p = inputParser;
addParameter(p, 'theta', 1);
addParameter(p, 'mu', 1);
addParameter(p, 'm', 100);
parse(p, varargin{:});

% Shapes
[~,NX] = size(X);
[~,NZ] = size(Z);

% Form block kernels
K = rbf_kernel(X,Z, 'theta', p.Results.theta);
clear X Z

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


