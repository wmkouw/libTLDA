function [W,pred,iw] = rba(X,Z,y,varargin)
% Implementation of a Robust Bias-Aware classifier
%
% Reference: Liu & Ziebart (20140. Robust Classification under
%            Sample Selection Bias. NIPS.
%
% Input:    X        source data (N samples x D features)
%           Z        target data (M samples x D features)
%           y        source labels (N x 1) in  {1,...,K}
% Optional:
%           order    order of feature statistics (default: 'first')
%           l2       l2-regularization parameters (default: 1e-3)
%           gamma    decaying learning rate (default: 1)
%           tau      convergence threshold (default: 1e-5)
%           maxIter  maximum number of iterations (default: 500)
%           clip     clipping value for importance weights (default: 1000)
%
% Output:   W        trained classifier parameters
%           pred     predictions by trained classifier on target data
%           iw       importance weights
%
% Copyright: Wouter M. Kouw
% Last update: 04-01-2018

% Add dependencies to path
addpath(genpath('util'));

% Parse hyperparameters
p = inputParser;
addOptional(p, 'order', 'first');
addOptional(p, 'l2', 1e-5);
addOptional(p, 'gamma', 1);
addOptional(p, 'tau', 1e-5);
addOptional(p, 'maxIter', 1000);
addOptional(p, 'clip', 1000);
parse(p, varargin{:});

% Data shape
[N, D] = size(X);
[M, E] = size(Z);
labels = unique(y);
K = length(labels);

% Check if dimensionalities are the same
if D~=E; error('Data dimensionalities not the same in both domains'); end

% Feature function
switch p.Results.order
    case 'first'
        % First-order moment statistics plus the intercept
        fs = @(x,y) [y y.*x ones(size(x,1),1)];
        
        % Set dimensionality to feature statistics dimensionality
        D = 1 + D + 1;
        
    case 'second'
        % First-order moments, second-order mixed moments and the intercept
        fs = @(x,y) [y y.*x y.*kron2(x,x) ones(size(x,1),1)];
        
        % Set dimensionality to feature statistics dimensionality
        D = 1 + D + D.^2 + 1;
        
    otherwise
        error('Higher-order moments not implemented yet');
end

% Compute moment-matching constraint
c = mean(fs(X,y),1);

% Estimate importance weights
iw = iwe_kd(X,Z);

% Inverse weights to achieve p_S(x)/p_T(x)
iw = 1./iw;

% Clip weights if necessary
iw = min(p.Results.clip, iw);

% Preallocate arrays
psi = zeros(N,K);
pyx = zeros(N,K);

% Initialize classifier weights
W = randn(1,D)*0.01;

% Start gradient descent
for t = 1:p.Results.maxIter
    
    %%% Calculate psi function 
    for k = 1:K
        psi(:,k) = iw.*(fs(X,k*ones(N,1))*W');
    end
    
    %%% Estimate posterior p^(Y=y | x_i) (added numerical stability trick)
    for k = 1:K
        max_const = max(psi,[],2);
        pyx(:,k) = exp(psi(:,k) - max_const)./ sum(exp(psi - max_const), 2);
    end
        
    % Compute product of estimated posterior and source feature statistics
    pfs = 0;
    for k = 1:K
        pfs = pfs + pyx(:,k).* fs(X,k*ones(N,1));
    end

    % Gradient computation
    dL = c - mean(pfs,1);
    
    % Add regularization
    dL = dL + p.Results.l2.*2.*W;
    
    % Apply learning rate to gradient
    dW = dL./(t*p.Results.gamma);
    
    % Update classifier weights
    W = W + dW;
    
    % Report progress
    if rem(t,10)==1
        fprintf('Iteration %4i/%4i - Norm gradient: %3.5f\n', t, p.Results.maxIter, norm(dL));
    end
    
    % Check for convergence
    if norm(dL) <= p.Results.tau
        disp(['Broke at ' num2str(t)]);
        break
    end
    
end

% Calculate psi function for target samples
psi = zeros(M,K);
for k = 1:K
    psi(:,k) = fs(Z,k*ones(M,1))*W';
end

% Compute posteriors for target samples
post = zeros(M,K);
for k = 1:K
    max_const = max(psi, [], 2); 
    post(:,k) = exp(psi(:,k) - max_const) ./ sum(exp(psi - max_const),2);
end

% Predictions through max-posteriors
[~,pred] = max(post, [], 2);

end


function [f] = kron2(A,B)
% Row-wise Kronecker delta product to expand data matrix to second-order
% moments

% Must be expansion of data matrix
if A~=B; error('A not equal to B'); end

% Preallocate
f = zeros(size(A,1),size(B,2)*2);

% Loop over rows
for i = 1:size(A,1)
    % Row-wise Kronecker delta product
    f(i,:) = kron(A(i,:),B(i,:));
end

end

