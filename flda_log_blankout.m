function [W,theta] = flda_log_blankout(XQ,XP,yQ,lambda)
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
    theta = est_transfer_params_blank(XQ,XP);
    
    % Analytical solution to theta=1 => w=0
    ix = find(theta~=1);
    
    % Check for y in {-1,+1}
    if ~isempty(setdiff(unique(yQ), [-1,1]));
        yQ(yQ~=1) = -1;
    end
    
    % Minimize loss
    if ~isrow(yQ); yQ = yQ'; end
    w = minFunc(@flda_log_blankout_grad, zeros(length(ix),1), options, XQ(ix,:),yQ,theta(ix),lambda);
    
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
        theta{k} = est_transfer_params_blank(Xk,XP);
        
        % Analytical solution to theta=1 => w=0
        ix = find(theta{k}~=1);
        
        % Minimize loss
        if ~isrow(yk); yk = yk'; end
        w = minFunc(@flda_log_blankout_grad, zeros(length(ix),1), options, Xk(ix,:),yk,theta{k}(ix),lambda);
        
        % Bookkeeping
        W(ix,k) = w;
    end
end

end

% Implementation of the quadratic approximation of the expected log-loss
function [L, dL] = flda_log_blankout_grad(W,X,Y,theta,l2)
% This function expects an 1xN label vector Y with labels -1 and +1.

    % Compute negative log-likelihood
    wx = W' * X;
    m = max(wx, -wx);
    Awx = exp( wx-m) + exp(-wx-m);
    ll = -Y .* wx + log(Awx) + m;
    
    % Numerical stability issues
    theta = min(theta,0.9999);
    
    % Compute corrupted log-partition function
    sgm = 1./(1 + exp(-2*wx));
    Vx = bsxfun(@times, 1 ./ (1 - theta) - 1, X .^ 2);
    Vy = (W .^ 2)' * Vx;
    Vwx = 4 * sgm .* (1 - sgm);
    R = .5*Vy.*Vwx;
    
    % Expected cost
    L = sum(ll+R, 2) + l2 .*sum(W.^2);

    % Only compute gradient if requested
    if nargout > 1
        
        % Compute likelihood gradient
        dll = -Y*X' + ((exp( wx-m) - exp(-wx-m))./ Awx) * X';
        
        % Compute regularizer gradient
        dR = Vwx.*((1-sgm) - sgm).*Vy*X' + W'.* (Vwx * Vx'); 
        
        % Gradient
        dL = dll' + dR' + 2.*l2.*W;
        
    end
end

function [theta] = est_transfer_params_blank(XQ,XP)
% Function to estimate the parameters of a dropout transfer distribution

theta = max(0,1-mean(XP>0,2)./mean(XQ>0,2));

end
