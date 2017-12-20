function [iw] = iwe_lr(X, Z, l2)
% Logistic discrimination for importance weight estimation.
%
% Reference: Bickel et al. (2009), Discriminative learning under covariate shift. JMLR.

% Shape
[N,~] = size(X);
[M,~] = size(Z);
y = [ones(N,1); 2*ones(M,1)];

% Fit logistic regressor
W = mLR([X;Z], y, 'l2', l2);

% Calculate p(y=1|x)
iw = exp(X*W(:,2))./sum(exp(X*W),2);

end
