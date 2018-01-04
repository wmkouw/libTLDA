function [iw] = iwe_lr(X, Z, varargin)
% Logistic discrimination for importance weight estimation.
%
% Reference: Bickel et al. (2009), Discriminative learning under covariate shift. JMLR.

% Parse optionals
p = inputParser;
addOptional(p, 'l2', 0);
parse(p, varargin{:});

% Shape
[N,~] = size(X);
[M,~] = size(Z);
y = [ones(N,1); 2*ones(M,1)];

% Check for bias augmentation
if ~all(X(:,end)==1) && ~all(Z(:,end)==1)
    X = [X ones(N,1)];
    Z = [Z ones(M,1)];
end

% Fit logistic regressor
W = mlr([X;Z], y, 'l2', p.Results.l2);

% Calculate p(y=1|x)
iw = exp(X*W(:,2))./sum(exp(X*W),2);

end
