function [W,pred,theta] = flda(X,Z,y,varargin)
% Implementation of a feature-level domain adaptive classifier.
%
% Reference: Kouw, Krijthe, Loog & Van der Maaten (2016). Feature-level
%            domain adaptation. JMLR.
%
% Input:    X      source data (N samples x D features)
%           Z      target data (M samples x D features)
%           y      source labels (N x 1) in {1,...,K}
% Optional:
%           l2      l2-regularization parameters (default: 1e-3)
%           td      Transfer distribution (default: 'blankout')
%           loss    Choice of loss function (default: 'log')
%[
% Output:   W       trained classifier parameters
%           pred    predictions by trained classifier on target data
%           theta   transfer distribution parameters
%
% Copyright: Wouter M. Kouw
% Last update: 04-01-2018

% Add dependencies to path
addpath(genpath('util'));

% Check for solver
if isempty(which('minFunc')); error('Can not find minFunc'); end
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'final';

% Parse optionals
p = inputParser;
addOptional(p, 'l2', 1e-3);
addOptional(p, 'td', 'blankout');
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

% Map to one-hot encoding
Y = -ones(N,K);
for n = 1:N
    Y(n,y(n)) = 1;
end

% Check for bias augmentation
if ~all(X(:,end)==1) && ~all(Z(:,end)==1)
    X = [X ones(N,1)];
    Z = [Z ones(M,1)];
    D = D + 1;
end

% Estimate parameters of transfer distribution
theta = mle_td(X,Z, p.Results.td);

switch p.Results.td
    case 'dropout'
        % First moment of transfer distribution
        E = bsxfun(@times, (1-theta), X);

        % Second moment of transfer distribution
        V = zeros(D,D,N);
        for i = 1:N
            V(:,:,i) = diag(theta.*(1-theta)).*(X(i,:)'*X(i,:));
        end
    case 'blankout'
        % First moment of transfer distribution
        E = X;

        % Second moment of transfer distribution
        V = zeros(D,D,N);
        for i = 1:N
            V(:,:,i) = diag(theta.*(1-theta)).*(X(i,:)'*X(i,:));
        end
    otherwise
        error('Transfer distribution not implemented');
end

switch p.Results.loss

    case {'qd', 'quadratic', 'squared'}

        % Least squares solution
        W = (E'*E + sum(V,3) + p.Results.l2*eye(D))\(E'*Y);

    case {'lr', 'log', 'logistic'}

        % Set up a one-vs-rest 
        W = zeros(D,K);
        for k = 1:K

            % Minimize loss function for classifier parameters
            W(:,k) = minFunc(@flda_log_grad, randn(D,1), options, X, Y(:,k), E, V, p.Results.l2);

        end

    otherwise
        error('Loss function not implemented yet');
end

% Predict target labels
[~,pred] = max(Z*W, [], 2);

end


function [theta] = mle_td(X,Z,dist)
% Maximum likelihood estimation of transfer model parameters

switch dist
    case {'blankout','dropout'}

        % Rate parameters
        eta  = mean(X>0,1);
        zeta = mean(Z>0,1);

        % Ratio of rate parameters
        theta = max(0, 1 - zeta ./ eta);

    otherwise
        error('Transfer model not implemented');
end

end

function [R, dR] = flda_log_grad(w,X,y,E,V,l2)
% Input:    W       classifier weights (D features x 1 classes)
%           X       source data (N samples x D features)
%           y       source labels (N x 1) in {-1,+1}
%           E       expected value of the transfer distribution
%           V       variance of the transfer distribution
%           l2      l2-regularization parameters
%
% Output:   R       value of risk function
%           dR      gradient of risk w.r.t. w
%
% Wouter Kouw
% Last update: 04-01-2018

% Data shape
[N,~] = size(X);

% Check for y in {-1,+1}
if ~isempty(setdiff(unique(y), [-1,+1]))
    y(y~=1) = -1;
end

% Precompute terms
Xw = X*w;
Ew = E*w;
alpha = exp( Xw) + exp(-Xw);
beta  = exp( Xw) - exp(-Xw);
gamma = exp( Xw).*X + exp(-Xw).*X;
delta = exp( Xw).*X - exp(-Xw).*X;

% Log-partition function
A = log(alpha);

% First-order partial derivative of log-partition w.r.t. XW
dA = beta./ alpha;

% Second-order partial derivative of log-partition w.r.t. XW
d2A = 1 - beta.^2./alpha.^2;

% Compute pointwise loss (negative log-likelihood)
L = zeros(N,1);
for i = 1:N
    L(i) = -y(i).*Ew(i) + A(i) + dA(i).*(Ew(i) - Xw(i)) + 1./2*d2A(i)*w'*V(:,:,i)*w;
end

% Risk is average loss
R = sum(L,1);

% Add regularization
R = R + l2*sum(sum(w.^2, 2), 1);

if nargout > 1

    dR = 0;
    for i = 1:N

        t1 = -y(i)*E(i,:)';

        t2 = beta(i)./alpha(i)*X(i,:)';

        t3 = (gamma(i,:)./alpha(i) - beta(i)*delta(i,:)./alpha(i).^2)' *(Ew(i) - Xw(i));

        t4 = beta(i)./alpha(i).*(E(i,:) - X(i,:))';

        t5 = 1./2.*(1 - beta(i).^2./alpha(i).^2)*(V(:,:,i) + V(:,:,i)')*w;

        t6 = -(beta(i).*gamma(i,:)./alpha(i).^2 - beta(i).^2.*delta(i,:)./alpha(i).^3)'*(w'*V(:,:,i)*w);

        dR = dR + t1 + t2 + t3 + t4 + t5 + t6;

    end

    % Add regularization
    dR = dR + l2*2.*w;
end

end
