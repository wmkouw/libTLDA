function [F,theta,Pp] = scl(XQ,XP,yQ,varargin)
% Function to calculate structural correspondence learning
% J. Blitzer,R. McDonald & F. Pereira (2006). Domain adaptation with
% Structural Correspondence Learning. EMNLP

addpath(genpath('minFunc'));

% Shape
[MQ,~] = size(XQ);
[MP,~] = size(XP);

% Parse hyperparameters
p = inputParser;
addOptional(p, 'l2', 0);
addOptional(p, 'm', 20);
addOptional(p, 'h', 15);
parse(p, varargin{:});

% Check for y in {1,..K}
if any(yQ==0) || any(yQ==-1); error('y not in [1,..K]'); end
K = max(yQ);

% Optimization options
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'final';

% Choose m pivot features;
[~,ix1] = sort(sum(XQ,2)+sum(XP,2), 'descend');
pivot = [XQ(ix1(1:p.Results.m),:) XP(ix1(1:p.Results.m),:)];
pivot(pivot>0) = 1;

% Solve m binary prediction tasks
Pp = zeros(MQ,p.Results.m);
for l = 1:p.Results.m
    disp(['Pivot feature #' num2str(l)]);
    Pp(:,l) = minFunc(@Huber_grad, randn(MQ,1), options, [XQ XP], pivot(p.Results.m,:), p.Results.l2);
end
clear pivot XP

% Decompose pivot predictors
[theta,~] = eigs(cov(Pp'), p.Results.h);
theta = theta'; 

% Minimize loss
f = minFunc(@mLR_grad, randn((MQ+p.Results.h+1)*K,1), options, [XQ; theta*XQ], yQ(:)', p.Results.l2);

% Output MxK weight matrix
F = [reshape(f(1:end-K), [MQ+p.Results.h K]); f(end-K+1:end)'];

end

function [L, dL] = mLR_grad(W,X,y, lambda)
% Implementation of logistic regression
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
    L = L - log(max(WX(y(i), i), realmin));
end
L = L + lambda .* sum([W(:); W0(:)] .^ 2);

% Only compute gradient if requested
if nargout > 1
    
    % Compute positive part of gradient
	pos_E = zeros(M, K);
    pos_E0 = zeros(1, K);
    for k=1:K
        pos_E(:,k) = sum(X(:,y == k), 2);            
    end
    for k=1:K
        pos_E0(k) = sum(y == k);
    end
    
    % Compute negative part of gradient    
    neg_E = X * WX';
    neg_E0 = sum(WX, 2)';
        
	% Compute gradient
	dL = -[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)] + 2 .* lambda .* [W(:); W0(:)];
    
end
end

function [L,dL] = Huber_grad(w,X,y,la)
% Modified Huber loss function 
% R. Ando & T. Zhang (2005a). A framework for learning predictive
% structures from multiple tasks and unlabeled data. JMLR.

% Precompute
Xy = bsxfun(@times, X, y);
wXy = w'*Xy;

% Indices of discontinuity
ix = (wXy>=-1);

% Loss
L = sum(max(0,1-wXy(ix)).^2,2) + sum(-4*wXy(~ix),2);
dL = sum(bsxfun(@times, 2*max(0,1-wXy(ix)), (-Xy(:,ix))),2) + sum(-4*Xy(:,~ix),2);
    
% Add l2-regularization
L = L + la*sum(w.^2);
dL = dL + 2*la*w;

end
