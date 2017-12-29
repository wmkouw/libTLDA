function [iw] = iwe_kd(X, Z, l2)
% Ratio of distributions as estimated by kernel density estimators
%
% Reference:   

% Target distribution
pZ = ksdensity(Z, X);

% Source distribution
pX = ksdensity(X,X);

iw =  pZ./ (l2 + pX);

end
