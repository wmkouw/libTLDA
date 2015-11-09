function [W,theta] = fir_log_dropout(XQ,XP,yQ,lambda)
% Function to run the optimization procedure of the feature importance
% regularized domain adaptation classifier.

addpath(genpath('minFunc'));

% Optimization options
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'final';

% Shape
[MQ,~] = size(XQ);

% Number of classes
K = numel(unique(yQ));

if K==2
    
    % Estimate blankout transfer parameters
    theta = est_transfer_params_drop(XQ,XP);
    
    % Analytical solution to theta=1 => w=0
    ix = find(theta~=1);
    
    % Check for y in {-1,+1}
    if ~isempty(setdiff(unique(yQ), [-1,1]));
        yQ(yQ~=1) = -1;
    end
    
    % Minimize loss
    if ~isrow(yQ); yQ = yQ'; end
    w = minFunc(@flda_log_dropout_grad, zeros(length(ix),1), options, XQ(ix,:),yQ,theta(ix),lambda);
    
    % Bookkeeping
    W = zeros(MQ,1);
    W(ix) = w;
    W = [W -W];
    
else
    
    W = zeros(MQ,K);
    for k = 1:K
        
        % Labels
        yk = (yQ==k);
        
        % 50up-50down resampling
        ix = randsample(find(yk==0), floor(.5*sum(1-yk)));
        Xk = [XQ(:,ix) repmat(XQ(:,yk), [1 floor((K-1)/2)])];
        yk = [double(yk(ix))'; ones(floor((K-1)./2)*sum(yk),1)];
        yk(yk==0) = -1;
        
        % Estimate blankout transfer parameters
        theta = est_transfer_params_drop(Xk,XP);
        
        % Analytical solution to theta=1 => w=0
        ix = find(theta~=1);
        
        % Minimize loss
        if ~isrow(yk); yk = yk'; end
        w = minFunc(@flda_log_dropout_grad, zeros(length(ix),1), options, Xk(ix,:),yk,theta(ix),lambda);
        
        % Bookkeeping
        W(ix,k) = w;
    end
end

end

function [L, dL] = flda_log_dropout_grad(W,X,Y,theta,l2)
% Implementation of the quadratic approximation of the expected log-loss
% This function expects an 1xN label vector Y with labels -1 and +1.

% Precompute
wx = W' * X;
m = 1*max(-wx,wx);
Ap = exp(wx -m) + exp(-wx -m);
An = exp(wx -m) - exp(-wx -m);
dAwx = An./Ap;
d2Awx = 2*exp(-2*m)./Ap.^2;
qX1 = bsxfun(@times, 1-theta, X);
qX2 = bsxfun(@times, -theta, X);
qX3 = bsxfun(@times, theta.*(1-theta),X.^2);
qX4 = bsxfun(@times, 2-theta,X);

% Negative log-likelihood (-log p(y|x))
L = sum(-Y.* (W'*qX1) + log(Ap) +m,2);

% First order expansion term
T1 = sum(dAwx.*(W'*qX2),2);

% Second order expansion term
Q2 = bsxfun(@times,d2Awx,qX2)*qX4' + diag(sum(bsxfun(@times,qX3,d2Awx),2));
T2 = W'*Q2*W;

% Compute loss
L = L + T1 + T2;

% Additional l2-regularization
L = L +  l2 *(sum(W(:).^2));

% Only compute gradient if requested
if nargout > 1
    
    % Compute partial derivative of negative log-likelihood
    dL = qX1*-Y' + X*dAwx';
    
    % Compute partial derivative of first-order term
    dT1 = X*((1-dAwx.^2).*(W'*qX2))' + qX2*dAwx';
    
    % Compute partial derivative of second-order term
    wQw = (W'*qX2).*(W'*qX4) + W'.^2*qX3;
    dT2 = (Q2+Q2')*W + X*(-4*exp(-2*m).*An./Ap.^3.*wQw)';
    
    % Gradient
    dL = dL + dT1 + dT2;
    
    % Additional l2-regularization
    dL = dL + 2.*l2.*W(:);
end
end

function [theta] = est_transfer_params_drop(XQ,XP)
% Function to estimate the parameters of a dropout transfer distribution

theta = max(0,1-mean(XP>0,2)./mean(XQ>0,2));

end
