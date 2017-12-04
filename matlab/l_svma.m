function [theta,varargout] = l_svma(X,yX,Z,varargin)
% Function that performs \lambda-shift Support Vector Machine Adaptation
%
% Code written per reference:
% Robust Domain Adaptation, Mansour & Schain (2014), Annals Math & AI

% Parse hyperparameters
p = inputParser;
addOptional(p, 'yZ', []);
addOptional(p, 'C', 1);
addOptional(p, 'lambda', 1e-3);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'clip', 100);
parse(p, varargin{:});

% Shapes
[N,D] = size(X);
[M,~] = size(Z);
uy = unique(yX);
K = numel(uy);

% Set options
options.Display = 'iter';
options.MaxIter = p.Results.maxIter;
options.DerivativeCheck = 'off';
options.TolX = p.Results.xTol;

% Warning
disp(['Pessimistic version only for now'])

% Probability of region k sampled from Q
pX = zeros(N,1);
for k = 1:K
    mu = mean(X(yX==uy(k),:));
    S = cov(X(yX==uy(k),:)) + p.Results.lambda*eye(D);
    pX(yX==uy(k)) = max(realmin,mvnpdf(X(yX==uy(k),:),mu,S));
end

% Target-Source density ratio
pZpX = iw_gauss(X(:,1:end-1)', Z(:,1:end-1)', 'order','ZX', 'lambda', p.Results.lambda, 'clip', p.Results.clip);

% Probability of region k sampled from P
T = zeros(K,1);
for k = 1:K
    T(k,:) = sum(pZpX(yX==uy(k)).*pX(yX==uy(k))');
end

% Sum of alpha's in each class
A = zeros(K,N);
for k = 1:K
    A(k,:) = (yX==uy(k));
end

% Gather components for quadratic program
H = bsxfun(@times, yX, X)*bsxfun(@times, yX, X)';
H = (H+H')./2;
f = -ones(N,1);
ineqA = A;
ineqb = p.Results.C*T;
eqA = yX';
eqb = 0;

% Run quadratic program 
alpha = quadprog(H,f,ineqA,ineqb,eqA,eqb,zeros(N,1),[],[],options);

% Relate dual solution to primal
theta = (alpha.*yX)'*X;

% Compute bias
bias = (max(X(yX==-1,:)*theta') + min(X(yX==1,:)*theta'))./2;
theta = [theta -bias];

% % Evaluate with target labels
if ~isempty(p.Results.yZ);
    
    % Augment target data
    Z = [Z ones(M,1)];
    
    % Error on target set
    [~,varargout{2}] = max(Z*[theta' -theta'], [], 2);
    varargout{1} = mean(varargout{2}~=p.Results.yZ);    
    
end


end
