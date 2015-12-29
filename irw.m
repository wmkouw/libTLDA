function [W,theta] = irw(XQ,XP,yQ,varargin)
% Function to run an instance reweighted domain adaptive classifier
% Input:    XQ      source data (M features x NQ samples)
%           XP      target data (M features x NP samples)
%           yQ      source labels (NQ x 1)
% Optional:
%           l2      Additional l2-regularization parameters (default: 1e-3)
%           iw      Choice of importance weight estimation method (default: 'log')
%           loss    Choice of loss function (default: 'log')
%
% Output:   W       Trained linear classifier
%           theta   Found importance weights for source samples
%
% Wouter M. Kouw
% Last update: 22-12-2015

% Parse optionals
p = inputParser;
addOptional(p, 'l2', 1e-3);
addOptional(p, 'iw', 'log');
addOptional(p, 'loss', 'log');
parse(p, varargin{:});

% Add dependencies to path
addpath(genpath('util'));
addpath(genpath('minFunc'));

% Check for solver
if isempty(which('minFunc')); error('Can not find minFunc'); end
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'final';

% Data shape
[M,NQ] = size(XQ);
K = max(yQ);

% Check if labels are in [1,...,K]
if length(unique(yQ))==2 && all(unique(yQ)==[-1 1] | unique(yQ)==[0 1]);
    yQ(yQ==-1 | yQ==0) = 2;
elseif all(unique(yQ)~=[1:K]); 
    error(['Labels should be a NQx1 vector in [1,...,K]']); 
end

% Estimate importance weights
switch p.Results.iw
    case 'Gauss'
        theta = iw_Gauss(XQ,XP,p.Results.l2);
    case 'log'
        theta = iw_log(XQ,XP,p.Results.l2);
    case 'kmm'
        theta = iw_kmm(XQ,XP, 1,'rbf');
end

switch p.Results.loss
    case 'qd'
        % Map NQx1 vector in [1,...,K] to KxNQ binary matrix
        if min(size(yQ))==1;
            Y = zeros(K,NQ);
            for i = 1:NQ; Y(yQ(i),i) = 1; end
        else
            Y = yQ;
        end
        
        % Least-squares solution to instance reweighted quadratic loss
        bX = [bsxfun(@times, theta, XQ); ones(1,NQ)];
        W = (Y*bX'/ (bX*bX'+p.Results.l2*eye(M+1)))';
        
    case 'log'
        % Minimize loss
        W_star = minFunc(@mWLR_grad, zeros((M+1)*K,1), options, XQ, yQ, theta, p.Results.l2);
        
        % Output multiclass weight vector
        W = [reshape(W_star(1:end-K), [M K]); W_star(end-K+1:end)'];
        
    otherwise
        error('Reweighted loss function not implemented');
end

end


function [iw] = iw_Gauss(X, Z, l2)
% Uses two Gaussian distributions to estimate importance weights
% ! Do not use for high-dimensional data
% Function expects MxN matrices.

% Shape
[M,N] = size(X);

% Calculate sparse Gaussian parameters
mu0 = mean(X,2);
mu1 = mean(Z,2);
X0_ = bsxfun(@minus, X, mu0);
X1_ = bsxfun(@minus, Z, mu1);
S0 = 1./N .* (X0_*X0_' + l2*eye(M));
S1 = 1./N .* (X1_*X1_' + l2*eye(M));
L0 = chol(S0, 'lower');
L1 = chol(S1, 'lower');

% Calculate reweighting of X_0 based on difference in Gaussians
temp = (L1'\(L1\mu1)) - (L0'\(L0\mu0));
mu_ = L1'\(L1\temp) - L0'\(L0\temp);
X = bsxfun(@minus, X, mu_);
iw = exp(-1/2.*(sum(X.*(L1\X),1) - sum(X.*(L0\X),1)));
end

function [iw] = iw_kmm(X, Z, theta, kernel)
% Use Kernel Mean Matching to estimate weights for importance weighting.
%
% Jiayuan Huang, Alex Smola, Arthur Gretton, Karsten Borgwardt & Bernhard
% Schoelkopf. Correcting Sample Selection Bias by unlabeled data.

% Shapes
[~,NX] = size(X);
[~,NZ] = size(Z);

switch kernel
    case 'rbf'
        
        % Calculate Euclidean distances
        K = pdist2(X', X');
        k = pdist2(X', Z');
        
        % Cleanup
        I = find(K<0); K(I) = zeros(size(I));
        J = find(K<0); K(J) = zeros(size(J));
        
        % Radial basis function
        K = exp(-K/(2*theta.^2));
        k = exp(-k/(2*theta.^2));
        k = NX./NZ*sum(k,2);
        
    case 'diste'
        % Calculate Euclidean distances
        K = pdist2(X', X');
        k = pdist2(X', Z');
        if theta ~= 2
            K = sqrt(K).^theta;
            k = sqrt(k).^theta;
        end
        k = NX./NZ*sum(k,2);
end

% % Approximate if memory shortage
% a = whos('K');
% if a.bytes > 2e9;
%     K(K<.2) = 0;
%     K = sparse(K);
% end

% Solve quadratic program
options.Display = 'final';
iw = quadprog(K,k,[ones(1,NX); -ones(1,NX)],[NX./sqrt(NX)+NX NX./sqrt(NX)-NX],[],[],zeros(NX,1),ones(NX,1), [], options)';

end

function [iw] = iw_log(X, Z, l2)
% Function to estimate importance weights using a logistic regressor
% Wouter Kouw
% 22-12-2015
% X,Z = MxN matrices

% Shape
[M0,N0] = size(X);
[~,N1] = size(Z);
Y = [zeros(NX,1); ones(NZ,1)];

% Run logistic regressor on domains
options.method = 'lbfgs';
options.Display = 'final';

% Minimize loss
w = minFunc(@LR_grad, randn(M+1,1), options, [X Z]', Y, lambda);

% Calculate p(y=1|x)
iw = exp(w'*[X; ones(1,NX)])./(1+exp(w'*[X; ones(1,NX)]));

end

function [L, dL] = LR_grad(w,D,y, ld)
% Implementation of logistic regression
% Wouter Kouw
% 22-12-2015
% This function expects y in [0,1] and no augmentation for X

% Shape
[N,~] = size(D);
D = [D ones(N,1)];

% Numerical stability trick
Dw = D*w;
ma = max(0,max(Dw, [], 2));

% Logistic loss
L = -1./N*sum(y.*(Dw) - log(exp(-ma)+exp(Dw-ma)) - ma,1) + ld*sum(w.^2);

% Only compute gradient if requested
if nargout > 1
    
    % Gradient with respect to w
    dL = -1./N*(D'*y - D'*(exp(Dw-ma)./(exp(-ma)+exp(Dw-ma)))) + 2*ld*w;
    
end
end

function [L, dL] = mWLR_grad(W,X,y,iw, lambda)
% Implementation of instance reweighted logistic regression
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
    L = L - iw(i)*log(max(WX(y(i), i),realmin));
end
L = L./N + lambda .* sum([W(:); W0(:)] .^ 2);

% Only compute gradient if requested
if nargout > 1
    
    % Compute positive part of gradient
    pos_E = zeros(M, K);
    pos_E0 = zeros(1, K);
    for k=1:K
        pos_E(:,k) = sum(bsxfun(@times, iw(y == k), X(:,y == k)), 2);
        pos_E0(k) = sum(iw(y == k));
    end
    
    % Compute negative part of gradient
    neg_E = bsxfun(@times, iw, X) * WX';
    neg_E0 = sum(bsxfun(@times, iw, WX), 2)';
    
    % Compute gradient
    dL = -1./N*[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)] + 2 .* lambda .* [W(:); W0(:)];
    
end
end
